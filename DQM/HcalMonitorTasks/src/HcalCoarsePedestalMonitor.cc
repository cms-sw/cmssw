#include "DQM/HcalMonitorTasks/interface/HcalCoarsePedestalMonitor.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#include <cmath>

#include "FWCore/Common/interface/TriggerNames.h" 
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "Geometry/HcalTowerAlgo/src/HcalHardcodeGeometryData.h" // for eta bounds
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

// constructor
HcalCoarsePedestalMonitor::HcalCoarsePedestalMonitor(const edm::ParameterSet& ps) : HcalBaseDQMonitor(ps)
{
  Online_                = ps.getUntrackedParameter<bool>("online",false);
  mergeRuns_             = ps.getUntrackedParameter<bool>("mergeRuns",false);
  enableCleanup_         = ps.getUntrackedParameter<bool>("enableCleanup",false);
  debug_                 = ps.getUntrackedParameter<int>("debug",0);
  prefixME_              = ps.getUntrackedParameter<std::string>("subSystemFolder","Hcal/");
  if (prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
    prefixME_.append("/");
  subdir_                = ps.getUntrackedParameter<std::string>("TaskFolder","CoarsePedestalMonitor_Hcal"); 
  if (subdir_.size()>0 && subdir_.substr(subdir_.size()-1,subdir_.size())!="/")
    subdir_.append("/");
  subdir_=prefixME_+subdir_;
  AllowedCalibTypes_     = ps.getUntrackedParameter<std::vector<int> > ("AllowedCalibTypes");
  skipOutOfOrderLS_      = ps.getUntrackedParameter<bool>("skipOutOfOrderLS",true);
  NLumiBlocks_           = ps.getUntrackedParameter<int>("NLumiBlocks",4000);
  makeDiagnostics_       = ps.getUntrackedParameter<bool>("makeDiagnostics",false);
  digiLabel_             = ps.getUntrackedParameter<edm::InputTag>("digiLabel");
  ADCDiffThresh_         = ps.getUntrackedParameter<double>("ADCDiffThresh",1.);
  minEvents_             = ps.getUntrackedParameter<int>("minEvents",100); // minimum number of events needed before histograms are filled
  excludeHORing2_       = ps.getUntrackedParameter<bool>("excludeHORing2",false);

  tok_hbhe_ = consumes<HBHEDigiCollection>(digiLabel_);
  tok_ho_ = consumes<HODigiCollection>(digiLabel_);
  tok_hf_ = consumes<HFDigiCollection>(digiLabel_);
  tok_report_ = consumes<HcalUnpackerReport>(digiLabel_);

}


// destructor
HcalCoarsePedestalMonitor::~HcalCoarsePedestalMonitor() {}


void HcalCoarsePedestalMonitor::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  // Anything to do here?
}

void HcalCoarsePedestalMonitor::endJob()
{
  if (debug_>0) std::cout <<"HcalCoarsePedestalMonitor::endJob()"<<std::endl;
  if (enableCleanup_) cleanup(); // when do we force cleanup?
}


void HcalCoarsePedestalMonitor::setup(DQMStore::IBooker &ib)
{
  // Call base class setup
  HcalBaseDQMonitor::setup(ib);

  /******* Set up all histograms  ********/
  if (debug_>1)
    std::cout <<"<HcalCoarsePedestalMonitor::setup>  Setting up histograms"<<std::endl;

  std::ostringstream name;
  ib.setCurrentFolder(subdir_ +"CoarsePedestalSumPlots");
  SetupEtaPhiHists(ib,CoarsePedestalsSumByDepth,"Coarse Pedestal Summed Map","");
  SetupEtaPhiHists(ib,CoarsePedestalsOccByDepth,"Coarse Pedestal Occupancy Map","");
  for (unsigned int i=0;i<CoarsePedestalsSumByDepth.depth.size();++i)
    (CoarsePedestalsSumByDepth.depth[i]->getTH2F())->SetOption("colz");
  for (unsigned int i=0;i<CoarsePedestalsOccByDepth.depth.size();++i)
    (CoarsePedestalsOccByDepth.depth[i]->getTH2F())->SetOption("colz");


  ib.setCurrentFolder(subdir_+"CoarsePedestal_parameters");
  MonitorElement* ADCDiffThresh = ib.bookFloat("ADCdiff_Problem_Threshold");
  ADCDiffThresh->Fill(ADCDiffThresh_);
  MonitorElement* minevents = ib.bookInt("minEventsNeededForPedestalCalculation");
  minevents->Fill(minEvents_);
  MonitorElement* excludeHORing2 = ib.bookInt("excludeHORing2");
  if (excludeHORing2_==true)
    excludeHORing2->Fill(1);
  else
    excludeHORing2->Fill(0);

  this->reset();
  return;
} // void HcalCoarsePedestalMonitor::setup()

void HcalCoarsePedestalMonitor::bookHistograms(DQMStore::IBooker &ib, const edm::Run& run, const edm::EventSetup& c)
{
  HcalBaseDQMonitor::bookHistograms(ib,run,c);
  if (mergeRuns_ && tevt_>0) return; // don't reset counters if merging runs

  if (tevt_==0) this->setup(ib); // create all histograms; not necessary if merging runs together
  if (mergeRuns_==false) this->reset(); // call reset at start of all runs
} // void HcalCoarsePedestalMonitor::bookHistograms()


void HcalCoarsePedestalMonitor::analyze(edm::Event const&e, edm::EventSetup const&s)
{
  HcalBaseDQMonitor::analyze(e,s);
  if (!IsAllowedCalibType()) return;
  if (LumiInOrder(e.luminosityBlock())==false) return;


  // try to get digis
  edm::Handle<HBHEDigiCollection> hbhe_digi;
  edm::Handle<HODigiCollection> ho_digi;
  edm::Handle<HFDigiCollection> hf_digi;
  edm::Handle<HcalUnpackerReport> report;

  if (!(e.getByToken(tok_hbhe_,hbhe_digi)))
    {
      edm::LogWarning("HcalCoarsePedestalMonitor")<< digiLabel_<<" hbhe_digi not available";
      return;
    }
  
  if (!(e.getByToken(tok_hf_,hf_digi)))
    {
      edm::LogWarning("HcalCoarsePedestalMonitor")<< digiLabel_<<" hf_digi not available";
      return;
    }
  if (!(e.getByToken(tok_ho_,ho_digi)))
    {
      edm::LogWarning("HcalCoarsePedestalMonitor")<< digiLabel_<<" ho_digi not available";
      return;
    }
  if (!(e.getByToken(tok_report_,report)))
    {
      edm::LogWarning("HcalCoarsePedestalMonitor")<< digiLabel_<<" unpacker report not available";
      return;
    }

  // all objects grabbed; event is good
  if (debug_>1) std::cout <<"\t<HcalCoarsePedestalMonitor::analyze>  Processing good event! event # = "<<ievt_<<std::endl;

//  HcalBaseDQMonitor::analyze(e,s); // base class increments ievt_, etc. counters

  // Digi collection was grabbed successfully; process the Event
  processEvent(*hbhe_digi, *ho_digi, *hf_digi, *report);

} //void HcalCoarsePedestalMonitor::analyze(...)


void HcalCoarsePedestalMonitor::processEvent(const HBHEDigiCollection& hbhe,
					     const HODigiCollection& ho,
					     const HFDigiCollection& hf,
					     const HcalUnpackerReport& report)
{ 
  // Skip events in which minimal good digis found -- still getting some strange (calib?) events through DQM
  
  unsigned int allgooddigis= hbhe.size()+ho.size()+hf.size();
  // bad threshold:  ignore events in which bad outnumber good by more than 100:1
  // (one RBX in HBHE seems to send valid data occasionally even on QIE resets, which is why we can't just require allgooddigis==0 when looking for events to skip)
  if ((allgooddigis==0) ||
      (1.*report.badQualityDigis()>100*allgooddigis))
    {
      return;
    }

 ///////////////////////////////////////// Loop over HBHE

  unsigned int digisize=0;
  int depth=0, iphi=0, ieta=0, binEta=-9999;

  double value=0;

  for (HBHEDigiCollection::const_iterator j=hbhe.begin(); j!=hbhe.end(); ++j)
    {
	const HBHEDataFrame digi = (const HBHEDataFrame)(*j);
	if (digi.id().subdet()==HcalBarrel)
	  {if (!HBpresent_) continue;}
	else if (digi.id().subdet()==HcalEndcap)
	  {if (!HEpresent_) continue;}
	else continue;
	digisize=digi.size();
	if (digisize<8) 
	  continue;
	
	depth=digi.id().depth();
	iphi=digi.id().iphi();
	ieta=digi.id().ieta();
	
	digi.id().subdet()==HcalBarrel ? 
	  binEta=CalcEtaBin(HcalBarrel, ieta, depth) :
	  binEta=CalcEtaBin(HcalEndcap, ieta, depth);
	  
	// 'value' is the average pedestal over the 8 time slices.
	// In the CoarsePedestal client, we will divide the summed value by Nevents (in underflow bin)
	// in order to calculate average pedestals.
	value=0;
	for (unsigned int i=0;i<8;++i)
	  value+=digi.sample(i).adc()/8.;
	pedestalsum_[binEta][iphi-1][depth-1]+=value;
	++pedestalocc_[binEta][iphi-1][depth-1];
    }


  //////////////////////////////////// Loop over HO collection
  if (HOpresent_)
    {
      for (HODigiCollection::const_iterator j=ho.begin(); j!=ho.end(); ++j)
	{
	  const HODataFrame digi = (const HODataFrame)(*j);
	  digisize=digi.size();
	  if (digisize<8) 
	    continue;
	  
	  depth=digi.id().depth();
	  iphi=digi.id().iphi();
	  ieta=digi.id().ieta();
	  binEta=CalcEtaBin(HcalOuter, ieta, depth);

	  // Don't fill cells that are part of HO ring 2 if an exclusion is applied
	  if (excludeHORing2_==true && abs(ieta)>10 && isSiPM(ieta,iphi,4)==false)
	    continue;

	  // 'value' is the average pedestal over the 8 time slices.
	  // In the CoarsePedestal client, we will divide the summed value by Nevents (in underflow bin)
	  // in order to calculate average pedestals.
	  value=0;
	  for (unsigned int i=0;i<8;++i)
	    value+=digi.sample(i).adc()/8.;
	  pedestalsum_[binEta][iphi-1][depth-1]+=value;
	  ++pedestalocc_[binEta][iphi-1][depth-1];
	} // for (HODigiCollection)
    }
  
  /////////////////////////////////////// Loop over HF collection
  if (HFpresent_)
    {
      for (HFDigiCollection::const_iterator j=hf.begin(); j!=hf.end(); ++j)
	{
	  const HFDataFrame digi = (const HFDataFrame)(*j);
	  digisize=digi.size();
	  if (digisize<8) 
	    continue;
	  digisize=digi.size();
	  if (digisize<8) 
	    continue;
	  
	  depth=digi.id().depth();
	  iphi=digi.id().iphi();
	  ieta=digi.id().ieta();
	  binEta=CalcEtaBin(HcalForward, ieta, depth);
	  // 'value' is the average pedestal over the 8 time slices.
	  // In the CoarsePedestal client, we will divide the summed value by Nevents (in underflow bin)
	  // in order to calculate average pedestals.
	  value=0;
	  for (unsigned int i=0;i<8;++i)
	    value+=digi.sample(i).adc()/8.;
	  pedestalsum_[binEta][iphi-1][depth-1]+=value;
	  ++pedestalocc_[binEta][iphi-1][depth-1];
	} // for (HFDigiCollection)
    } // if (HFpresent_)

  return;
} // void HcalCoarsePedestalMonitor::processEvent(...)


void HcalCoarsePedestalMonitor::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
					     const edm::EventSetup& c) 
{
  HcalBaseDQMonitor::beginLuminosityBlock(lumiSeg,c);
  ProblemsCurrentLB->Reset();
}

void HcalCoarsePedestalMonitor::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
					   const edm::EventSetup& c)
{
  if (LumiInOrder(lumiSeg.luminosityBlock())==false) return;
  fill_Nevents();
  return;
}
void HcalCoarsePedestalMonitor::fill_Nevents()
{

  // require minimum number of events before processing
  if (ievt_<minEvents_)
    return;


  // Set underflow bin to the number of pedestal events processed
  // (Assume that number of ped events is the same for all channels/subdetectors)
  for (unsigned int i=0;i<CoarsePedestalsSumByDepth.depth.size();++i)
    CoarsePedestalsSumByDepth.depth[i]->setBinContent(0,0,ievt_);
  for (unsigned int i=0;i<CoarsePedestalsOccByDepth.depth.size();++i)
    CoarsePedestalsOccByDepth.depth[i]->setBinContent(0,0,ievt_);


  int iphi=-1, ieta=-99, idepth=0, calcEta=-99;
  // Loop over all depths, eta, phi
  for (int d=0;d<4;++d)
    {
      idepth=d+1;
      for (int phi=0;phi<72;++phi)
	{
	  iphi=phi+1; // actual iphi value
	  for (int eta=0;eta<83;++eta)
	    {
	      ieta=eta-41; // actual ieta value;
	      if (validDetId(HcalBarrel, ieta, iphi, idepth))
		{
		  calcEta = CalcEtaBin(HcalBarrel,ieta,idepth);
		  CoarsePedestalsSumByDepth.depth[d]->Fill(ieta,iphi,pedestalsum_[calcEta][phi][d]);
		  CoarsePedestalsOccByDepth.depth[d]->Fill(ieta,iphi,pedestalocc_[calcEta][phi][d]);
		}
	      if (validDetId(HcalEndcap, ieta, iphi, idepth))
		{
		  calcEta = CalcEtaBin(HcalEndcap,ieta,idepth);
		  CoarsePedestalsSumByDepth.depth[d]->Fill(ieta,iphi,pedestalsum_[calcEta][phi][d]);
		  CoarsePedestalsOccByDepth.depth[d]->Fill(ieta,iphi,pedestalocc_[calcEta][phi][d]);
		}
	      if (validDetId(HcalOuter, ieta, iphi, idepth))
		{
		  calcEta = CalcEtaBin(HcalOuter,ieta,idepth);
		  CoarsePedestalsSumByDepth.depth[d]->Fill(ieta,iphi,pedestalsum_[calcEta][phi][d]);
		  CoarsePedestalsOccByDepth.depth[d]->Fill(ieta,iphi,pedestalocc_[calcEta][phi][d]);
		}
	      if (validDetId(HcalForward, ieta, iphi, idepth))
		{
		  calcEta = CalcEtaBin(HcalBarrel,ieta,idepth);
		  int zside=ieta/abs(ieta);
		  CoarsePedestalsSumByDepth.depth[d]->Fill(ieta+zside,iphi,pedestalsum_[calcEta][phi][d]);
		  CoarsePedestalsOccByDepth.depth[d]->Fill(ieta+zside,iphi,pedestalocc_[calcEta][phi][d]);
		}
	    }
	}
    }
  // Fill unphysical bins
  FillUnphysicalHEHFBins(CoarsePedestalsSumByDepth);
  FillUnphysicalHEHFBins(CoarsePedestalsOccByDepth);

  return;
} // void HcalCoarsePedestalMonitor::fill_Nevents()


void HcalCoarsePedestalMonitor::reset()
{
  // then reset the MonitorElements
  zeroCounters();
  CoarsePedestalsSumByDepth.Reset();
  CoarsePedestalsOccByDepth.Reset();
  return;
}

void HcalCoarsePedestalMonitor::zeroCounters()
{
  for (int i=0;i<85;++i)
    for (int j=0;j<72;++j)
      for (int k=0;k<4;++k)
	{
	  pedestalsum_[i][j][k]=0;
	  pedestalocc_[i][j][k]=0;
	}
}
DEFINE_FWK_MODULE(HcalCoarsePedestalMonitor);
               
