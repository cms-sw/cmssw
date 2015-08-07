#include "DQM/HcalMonitorTasks/interface/HcalNZSMonitor.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include <math.h>

HcalNZSMonitor::HcalNZSMonitor(const edm::ParameterSet& ps) :HcalBaseDQMonitor(ps)
{
  Online_                = ps.getUntrackedParameter<bool>("online",false);
  mergeRuns_             = ps.getUntrackedParameter<bool>("mergeRuns",false);
  enableCleanup_         = ps.getUntrackedParameter<bool>("enableCleanup",false);
  debug_                 = ps.getUntrackedParameter<int>("debug",0);
  prefixME_              = ps.getUntrackedParameter<std::string>("subSystemFolder","Hcal/");
  if (prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
    prefixME_.append("/");
  subdir_                = ps.getUntrackedParameter<std::string>("TaskFolder","NZSMonitor_Hcal"); 
  if (subdir_.size()>0 && subdir_.substr(subdir_.size()-1,subdir_.size())!="/")
    subdir_.append("/");
  subdir_=prefixME_+subdir_;
  AllowedCalibTypes_     = ps.getUntrackedParameter<std::vector<int> > ("AllowedCalibTypes");
  skipOutOfOrderLS_      = ps.getUntrackedParameter<bool>("skipOutOfOrderLS",false);
  NLumiBlocks_           = ps.getUntrackedParameter<int>("NLumiBlocks",4000);
  makeDiagnostics_       = ps.getUntrackedParameter<bool>("makeDiagnostics",false);

  triggers_=ps.getUntrackedParameter<std::vector<std::string> >("nzsHLTnames"); //["HLT_HcalPhiSym","HLT_HcalNZS_8E29]
  period_=ps.getUntrackedParameter<int>("NZSeventPeriod",4096); //4096
  rawdataLabel_          = ps.getUntrackedParameter<edm::InputTag>("RawDataLabel");
  hltresultsLabel_       = ps.getUntrackedParameter<edm::InputTag>("HLTResultsLabel");

  // register for data access
  tok_raw_ = consumes<FEDRawDataCollection>(rawdataLabel_);
  tok_res_ = consumes<edm::TriggerResults>(hltresultsLabel_);

} 

HcalNZSMonitor::~HcalNZSMonitor() {}

void HcalNZSMonitor::reset()
{
  meFEDsizeVsLumi_->Reset();
  meFEDsizesNZS_->Reset();
  meUTCAFEDsizesNZS_->Reset();
  meL1evtNumber_->Reset();
  meIsUS_->Reset();
  meBXtriggered_->Reset();
  meTrigFrac_->Reset();
  meFullCMSdataSize_->Reset();
} // void HcalNZSMonitor::reset()


void HcalNZSMonitor::bookHistograms(DQMStore::IBooker &ib, const edm::Run& run, const edm::EventSetup& c)
{
  if (debug_>1) std::cout <<"HcalNZSMonitor::bookHistograms"<<std::endl;
  HcalBaseDQMonitor::bookHistograms(ib,run,c);

  if (tevt_==0) this->setup(ib); // set up histograms if they have not been created before
  if (mergeRuns_==false)
    this->reset();

  return;

} // void HcalNZSMonitor::bookHistograms(...)


void HcalNZSMonitor::setup(DQMStore::IBooker &ib)
{
  HcalBaseDQMonitor::setup(ib);
  
  if(debug_>1) std::cout << "<HcalNZSMonitor::setup> About to pushback fedUnpackList_" << std::endl;

  selFEDs_.clear();
  for (int i=FEDNumbering::MINHCALFEDID; 
		  i<=FEDNumbering::MAXHCALuTCAFEDID; i++)
    {
		if (i>FEDNumbering::MAXHCALFEDID && i<FEDNumbering::MINHCALuTCAFEDID)
			continue;

      selFEDs_.push_back(i);
    }

  nAcc.clear();
  for (unsigned int i=0; i<triggers_.size(); i++) nAcc.push_back(0);

  nAndAcc=0;
  nAcc_Total=0;
  
  if (debug_>1) std::cout <<"<HcalNZSMonitor::setup>  Creating histograms"<<std::endl;
      ib.setCurrentFolder(subdir_);
      
      meFEDsizesNZS_=ib.bookProfile("FED sizes","FED sizes",32,699.5,731.5,100,-1000.0,12000.0,"");
      meFEDsizesNZS_->setAxisTitle("FED number",1);
      meFEDsizesNZS_->setAxisTitle("average size (KB)",2);
      meFEDsizesNZS_->getTProfile()->SetMarkerStyle(22);

	  meUTCAFEDsizesNZS_ = ib.bookProfile("uTCA FED sizes",
			  "uTCA FED sizes", 5, 1117.5, 1122.5,100, -1000., 12000., "");
      meUTCAFEDsizesNZS_->setAxisTitle("FED number",1);
      meUTCAFEDsizesNZS_->setAxisTitle("average size (KB)",2);
      meUTCAFEDsizesNZS_->getTProfile()->SetMarkerStyle(22);
      
      meFEDsizeVsLumi_=ib.bookProfile("FED_size_Vs_lumi_block_number",
					 "FED size Vs lumi block number;lumiblock number;average HCAL FED size (kB)",
					 NLumiBlocks_,0.5,NLumiBlocks_+0.5,100,0,10000,"");
      meFEDsizeVsLumi_->getTProfile()->SetMarkerStyle(22);

      meL1evtNumber_=ib.book1D("Is_L1_event_number_multiple_of_NZS_period",
				  "Is L1 event number multiple of NZS period",2,0,2);
      meL1evtNumber_->setBinLabel(1, "NO", 1);
      meL1evtNumber_->setBinLabel(2, "YES", 1);

      meIsUS_=ib.book1D("IsUnsuppressed_bit","IsUnsuppressed bit",2,0,2);
      meIsUS_->setBinLabel(1,"NO",1);
      meIsUS_->setBinLabel(2,"YES",1);

      meBXtriggered_=ib.book1D("Triggered_BX_number","Triggered BX number",3850,0,3850);
      meBXtriggered_->setAxisTitle("BX number",1);

      meTrigFrac_=ib.book1D("HLT_accept_fractions","HLT accept fractions",triggers_.size()+1,0,triggers_.size()+1);
      for (unsigned int k=0; k<triggers_.size(); k++) meTrigFrac_->setBinLabel(k+1,triggers_[k].c_str(),1);
      meTrigFrac_->setBinLabel(triggers_.size()+1,"AND",1);

      meFullCMSdataSize_=ib.bookProfile("full_CMS_datasize",
					   "full CMS data size;lumiblock number;average FEDRawDataCollection size (kB)",
					   NLumiBlocks_,0.5,NLumiBlocks_+0.5,100,0,10000,"");
      meFullCMSdataSize_->getTProfile()->SetMarkerStyle(22);

  return;
} // void HcalNZSMonitor::setup()


void HcalNZSMonitor::analyze(edm::Event const&e, edm::EventSetup const&s)
{
  HcalBaseDQMonitor::analyze(e,s);
  if (!IsAllowedCalibType()) return;
  if (LumiInOrder(e.luminosityBlock())==false) return;
  
  edm::Handle<FEDRawDataCollection> rawraw;

  if (!(e.getByToken(tok_raw_,rawraw)))
    {
      edm::LogWarning("HcalNZSMonitor")<<" raw data with label "<<rawdataLabel_<<" not available";
      return;
    }

  edm::Handle<edm::TriggerResults> hltRes;
  if (!(e.getByToken(tok_res_,hltRes)))
    {
      if (debug_>0) edm::LogWarning("HcalNZSMonitor")<<" Could not get HLT results with tag "<<hltresultsLabel_<<std::endl;
      return;
    }

  const edm::TriggerNames & triggerNames = e.triggerNames(*hltRes);
  // Collections were found; increment counters
//  HcalBaseDQMonitor::analyze(e,s);

  processEvent(*rawraw, *hltRes, e.bunchCrossing(), triggerNames);

} // void HcalNZSMonitor::analyze(...)


void HcalNZSMonitor::processEvent(const FEDRawDataCollection& rawraw, 
				  const edm::TriggerResults& trigRes, 
				  int bxNum, 
				  const edm::TriggerNames& triggerNames)
{

  const unsigned int nTrig(triggerNames.size());
 
  std::vector<bool> trigAcc;
  for (unsigned int i=0; i<triggers_.size(); i++) trigAcc.push_back(false);
  
   for (unsigned int k=0; k<nTrig; k++)
     {
       for (unsigned int i=0; i<triggers_.size(); i++)
	 {
	   if (triggerNames.triggerName(k) == triggers_[i] && trigRes.accept(k)) trigAcc[i]=true;
	 }
     }
   bool andAcc=true;
   bool orAcc=false;
   for (unsigned int p=0; p<triggers_.size(); p++)
     {
       if (!trigAcc[p]) andAcc=false;
       if (trigAcc[p]) 
	 {
	   orAcc=true;
	   nAcc[p]++;
	 }
     }
   
   if (orAcc)
     {
       nAcc_Total++;
       for (unsigned int l=0; l<triggers_.size(); l++)
	 {
	   meTrigFrac_->setBinContent(l+1,nAcc[l]/(float)nAcc_Total);
	 }
     }

   if (andAcc) 
     {
       nAndAcc++;
       meTrigFrac_->setBinContent(triggers_.size()+1,nAndAcc/(float)nAcc_Total);
     }
   
  bool processevent=false;
  if (orAcc) processevent=true;

  if (!processevent) return;

  meBXtriggered_->Fill(bxNum+0.001,1);

  //calculate full HCAL data size:
  size_t hcalSize=0;
  bool hcalIsZS = false;
  for (unsigned int k=0; k<selFEDs_.size(); k++)
    {
      const FEDRawData & fedData = rawraw.FEDData(selFEDs_[k]);

	  //	not to include empty FEDs
	  if (fedData.size()<12)
		  continue;

      hcalSize+=fedData.size();
	  if (selFEDs_[k]<FEDNumbering::MINHCALuTCAFEDID)
	      meFEDsizesNZS_->Fill(selFEDs_[k]+0.001,fedData.size()/1024);
	  else
	      meUTCAFEDsizesNZS_->Fill(selFEDs_[k]+0.001,fedData.size()/1024);

      const HcalDCCHeader* dccHeader=(const HcalDCCHeader*)(fedData.data());
      if (dccHeader==0) continue;  // protection against bad data -- saw this happen in file /store/streamer/Data/A/000/131/540/Data.00131540.0200.A.storageManager.00.0000.dat; not yet sure why -- Jeff, 22 March 2010; this was due to empty (masked?) HO FEDs 724 and 727 -- Grigory, 25/03/2010 

      HcalHTRData htr;
      int nspigot =0; 
      for (int spigot=0; spigot<HcalDCCHeader::SPIGOT_COUNT; spigot++) 
	{    
	  nspigot++;
	  if (!dccHeader->getSpigotPresent(spigot)) continue;
	  // Load the given decoder with the pointer and length from this spigot.
	  dccHeader->getSpigotData(spigot,htr, fedData.size()); 
	  if(k != 20 && nspigot !=14 ) 
	    {      
	      if ( !htr.isUnsuppressed() ) hcalIsZS = true; 
	    }
	}
    }
  
  if (hcalIsZS) meIsUS_->Fill(0,1);
  else meIsUS_->Fill(1,1);

  meFEDsizeVsLumi_->Fill(currentLS+0.001, hcalSize/1024);
  
  size_t fullSize=0;
  for (int j=0; j<=FEDNumbering::MAXFEDID; ++j )
    {
      const FEDRawData & fedData = rawraw.FEDData(j);
      fullSize+=fedData.size();
    }

  meFullCMSdataSize_->Fill(currentLS+0.001,fullSize/1024);
  
  // get Trigger FED-Id
  const FEDRawData& fedData = rawraw.FEDData(FEDNumbering::MINTriggerGTPFEDID) ;
  FEDHeader header(fedData.data()) ;
  
  /// Level-1 event number generated by the TTC system
  if (header.lvl1ID()%period_==0) meL1evtNumber_->Fill(1,1);
  else meL1evtNumber_->Fill(0,1);
  return;

} //void HcalNZSMonitor::processEvent(...)

DEFINE_FWK_MODULE(HcalNZSMonitor);
