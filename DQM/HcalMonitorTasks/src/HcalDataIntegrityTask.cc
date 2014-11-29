#include "DQM/HcalMonitorTasks/interface/HcalDataIntegrityTask.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include <iostream>

HcalDataIntegrityTask::HcalDataIntegrityTask(const edm::ParameterSet& ps) :HcalBaseDQMonitor(ps)
{

  Online_                = ps.getUntrackedParameter<bool>("online",false);
  mergeRuns_             = ps.getUntrackedParameter<bool>("mergeRuns",false);
  enableCleanup_         = ps.getUntrackedParameter<bool>("enableCleanup",false);
  debug_                 = ps.getUntrackedParameter<int>("debug",0);
  prefixME_              = ps.getUntrackedParameter<std::string>("subSystemFolder","Hcal"); 
  if (prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
    prefixME_.append("/");
  subdir_                = ps.getUntrackedParameter<std::string>("TaskFolder","HcalDataIntegrityTask");
  if (subdir_.size()>0 && subdir_.substr(subdir_.size()-1,subdir_.size())!="/")
    subdir_.append("/");
  subdir_=prefixME_+subdir_;
  AllowedCalibTypes_     = ps.getUntrackedParameter<std::vector<int> > ("AllowedCalibTypes");
  skipOutOfOrderLS_      = ps.getUntrackedParameter<bool>("skipOutOfOrderLS",false);
  NLumiBlocks_           = ps.getUntrackedParameter<int>("NLumiBlocks",4000);
  makeDiagnostics_       = ps.getUntrackedParameter<bool>("makeDiagnostics",false);

  // Specific Data Integrity Task parameters
  inputLabelRawData_     = ps.getUntrackedParameter<edm::InputTag>("RawDataLabel",edm::InputTag("source"));
  inputLabelReport_      = ps.getUntrackedParameter<edm::InputTag>("UnpackerReportLabel",edm::InputTag("hcalDigis"));


  // register for data access
  tok_raw_ = consumes<FEDRawDataCollection>(inputLabelRawData_);
  tok_report_ = consumes<HcalUnpackerReport>(inputLabelReport_);

} // HcalDataIntegrityTask::HcalDataIntegrityTask()

HcalDataIntegrityTask::~HcalDataIntegrityTask() {}

void HcalDataIntegrityTask::reset()
{
  if (debug_>0)  std::cout <<"HcalDataIntegrityTask::reset()"<<std::endl;
  HcalBaseDQMonitor::reset();
  fedEntries_->Reset();
  fedFatal_->Reset();
  fedNonFatal_->Reset();

}


void HcalDataIntegrityTask::bookHistograms(DQMStore::IBooker &ib, const edm::Run& run, const edm::EventSetup& c)
{

  if (debug_>0) std::cout <<"HcalDataIntegrityTask::bookHistograms():  task =  '"<<subdir_<<"'"<<std::endl;

  HcalBaseDQMonitor::bookHistograms(ib,run, c);
  if (mergeRuns_ && tevt_>0) return;

  if (debug_>1)  std::cout<<"\t<HcalDataIntegrityTask::getting eMap..."<<std::endl;
  edm::ESHandle<HcalDbService> pSetup;
  c.get<HcalDbRecord>().get( pSetup );
  readoutMap_=pSetup->getHcalMapping();

  if (tevt_==0) // create histograms, if they haven't been created already
    this->setup(ib);
  // Clear histograms at the start of each run if not merging runs
  if (mergeRuns_==false)
  this->reset();

} // bookHistograms(const edm::Run& run, const edm::EventSetup& c)


void HcalDataIntegrityTask::setup(DQMStore::IBooker &ib)
{
  // Setup Creates all necessary histograms
  HcalBaseDQMonitor::setup(ib);
  
  //Initialize phatmap to a vector of vectors of uint64_t 0
  const static size_t iphirange = IPHIMAX - IPHIMIN;
  const static size_t ietarange = IETAMAX - IETAMIN;
  
  std::vector<uint64_t> phatv (iphirange + 1, 0);
  
  if (debug_>0) std::cout <<"<HcalDataIntegrityTask::setup>  Clearing old vectors"<<std::endl;
  // Clear any old vectors
  phatmap.clear();
  HBmap.clear();
  HEmap.clear();
  HFmap.clear();
  HOmap.clear();
  problemhere.clear();
  problemHB.clear();
  problemHE.clear();
  problemHF.clear();
  problemHO.clear();

  // ... nothing goes at ieta=0, so an extra bin goes there.
  phatmap = std::vector< std::vector < uint64_t> > ( ietarange + 1, phatv);
  HBmap   = std::vector< std::vector < uint64_t> > ( ietarange + 1, phatv);
  HEmap   = std::vector< std::vector < uint64_t> > ( ietarange + 1, phatv);
  HFmap   = std::vector< std::vector < uint64_t> > ( ietarange + 1, phatv);
  HOmap   = std::vector< std::vector < uint64_t> > ( ietarange + 1, phatv);
  std::vector<bool> probvect (iphirange + 1, 0);
  // ... nothing goes at ieta=0, so an extra bin goes there.
  problemhere = std::vector< std::vector <bool> > ( ietarange + 1, probvect);
  problemHB   = std::vector< std::vector <bool> > ( ietarange + 1, probvect);
  problemHE   = std::vector< std::vector <bool> > ( ietarange + 1, probvect);
  problemHF   = std::vector< std::vector <bool> > ( ietarange + 1, probvect);
  problemHO   = std::vector< std::vector <bool> > ( ietarange + 1, probvect);



  if(debug_>1) 
    std::cout << "About to pushback fedUnpackList_" << std::endl;

  // Use this in CMSSW_3_0_X and above:
  firstFED_ = FEDNumbering::MINHCALFEDID;
  for (int i=FEDNumbering::MINHCALFEDID; 
       i<=FEDNumbering::MAXHCALFEDID;
       ++i)
    {
      if(debug_>1) std::cout << "[DFMon:]Pushback for fedUnpackList_: " << i <<std::endl;
      fedUnpackList_.push_back(i);
    }

      if (debug_>1)
	std::cout <<"\t<HcalDataIntegrityTask> Setting folder to "<<subdir_<<std::endl;

      ib.setCurrentFolder(subdir_);
      
      fedEntries_ = ib.book1D("FEDEntries","# entries per HCAL FED",32,700,732);
      fedFatal_ = ib.book1D("FEDFatal","# fatal errors HCAL FED",32,700,732);
      fedNonFatal_ = ib.book1D("FEDNonFatal","# non-fatal errors HCAL FED",32,700,732);

  this->reset(); // clear all histograms at start
  return;
}

void HcalDataIntegrityTask::analyze(edm::Event const&e, edm::EventSetup const&s)
{
  if (!IsAllowedCalibType()) return;
  if (LumiInOrder(e.luminosityBlock())==false) return;
  
  // Now get the collections we need
  
  edm::Handle<FEDRawDataCollection> rawraw;

  if (!(e.getByToken(tok_raw_,rawraw)))
    {
      if (debug_>0) edm::LogWarning("HcalDataIntegrityTask")<<" raw data with label "<<inputLabelRawData_<<" not available";
      return;
    }
  
  edm::Handle<HcalUnpackerReport> report;
  if (!(e.getByToken(tok_report_,report)))
    {
      if (debug_>0) edm::LogWarning("HcalDataIntegrityTask")<<" UnpackerReport with label "<<inputLabelReport_<<" \not available";
      return;
    }
  
  // Collections were found; increment counters
  HcalBaseDQMonitor::analyze(e,s);

  processEvent(*rawraw, *report, *readoutMap_);
}



void HcalDataIntegrityTask::processEvent(const FEDRawDataCollection& rawraw, 
					 const HcalUnpackerReport& report, 
					 const HcalElectronicsMap& emap){
  
  // Loop over all FEDs reporting the event, unpacking if good.
  for (std::vector<int>::const_iterator i=fedUnpackList_.begin();i!=fedUnpackList_.end(); i++) 
    {
      const FEDRawData& fed = rawraw.FEDData(*i);
      if (fed.size()<12) continue; // Was 16. How do such tiny events even get here?
      unpack(fed,emap);
    }

  return;
} //void HcalDataIntegrityTask::processEvent()


// Process one FED's worth (one DCC's worth) of the event data.
void HcalDataIntegrityTask::unpack(const FEDRawData& raw, 
				   const HcalElectronicsMap& emap){
  // get the DCC header
  const HcalDCCHeader* dccHeader=(const HcalDCCHeader*)(raw.data());
  if(!dccHeader) return;

  // get the DCC trailer 
  unsigned char* trailer_ptr = (unsigned char*) (raw.data()+raw.size()-sizeof(uint64_t));
  FEDTrailer trailer = FEDTrailer(trailer_ptr);

  int dccid=dccHeader->getSourceId();

  ////////// Histogram problems with the Common Data Format compliance;////////////
  bool CDFProbThisDCC = false; 
  /* 1 */ //There should always be a second CDF header word indicated.
  if (!dccHeader->thereIsASecondCDFHeaderWord()) 
    {
      CDFProbThisDCC = true; 
    }

  /* 2 */ //Make sure a reference CDF Version value has been recorded for this dccid
  CDFvers_it = CDFversionNumber_list.find(dccid);
  if (CDFvers_it  == CDFversionNumber_list.end()) 
    {
      CDFversionNumber_list.insert(std::pair<int,short>
				   (dccid,dccHeader->getCDFversionNumber() ) );
      CDFvers_it = CDFversionNumber_list.find(dccid);
    } // then check against it.

  if (dccHeader->getCDFversionNumber()!= CDFvers_it->second) 
    {
      CDFProbThisDCC = true; 
    }
  
  /* 3 */ //Make sure a reference CDF EventType value has been recorded for this dccid
  CDFEvT_it = CDFEventType_list.find(dccid);
  if (CDFEvT_it  == CDFEventType_list.end()) 
    {
      CDFEventType_list.insert(std::pair<int,short>
			       (dccid,dccHeader->getCDFEventType() ) );
      CDFEvT_it = CDFEventType_list.find(dccid);
    } // then check against it.
  
  if (dccHeader->getCDFEventType()!= CDFEvT_it->second) 
    {
      // On probation until safe against Orbit Gap Calibration Triggers...
      // CDFProbThisDCC = true; 
    }

  /* 4 */ //There should always be a '5' in CDF Header word 0, bits [63:60]
  if (dccHeader->BOEshouldBe5Always()!=5) 
    {
      CDFProbThisDCC = true; 
    }

  /* 5 */ //There should never be a third CDF Header word indicated.
  if (dccHeader->thereIsAThirdCDFHeaderWord()) 
    {
      CDFProbThisDCC = true; 
    }

  /* 6 */ //Make sure a reference value of Reserved Bits has been recorded for this dccid

  CDFReservedBits_it = CDFReservedBits_list.find(dccid);
  if (CDFReservedBits_it  == CDFReservedBits_list.end()) {
    CDFReservedBits_list.insert(std::pair<int,short>
				(dccid,dccHeader->getSlink64ReservedBits() ) );
    CDFReservedBits_it = CDFReservedBits_list.find(dccid);
  } // then check against it.
  
  if ((int) dccHeader->getSlink64ReservedBits()!= CDFReservedBits_it->second) 
    {
    // On probation until safe against Orbit Gap Calibration Triggers...
    //       CDFProbThisDCC = true; 
    }

  /* 7 */ //There should always be 0x0 in CDF Header word 1, bits [63:60]
  if (dccHeader->BOEshouldBeZeroAlways() !=0) 
    {
      CDFProbThisDCC = true; 
    }
  
  /* 8 */ //There should only be one trailer
  if (trailer.moreTrailers()) 
    {
      CDFProbThisDCC = true; 
    }
  //  if trailer.

  /* 9 */ //CDF Trailer [55:30] should be the # 64-bit words in the EvFragment
  if ((uint64_t) raw.size() != ( (uint64_t) trailer.lenght()*sizeof(uint64_t)) )  //The function name is a typo! Awesome.
    {
      CDFProbThisDCC = true; 
    }
  /*10 */ //There is a rudimentary sanity check built into the FEDTrailer class
  if (!trailer.check()) 
    {
      CDFProbThisDCC = true; 
    }

  if (CDFProbThisDCC)
    fedFatal_->Fill(dccid);
  fedEntries_->Fill(dccid);

  return;
} // void HcalDataIntegrityTask::unpack()

DEFINE_FWK_MODULE(HcalDataIntegrityTask);

