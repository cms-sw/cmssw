#include <DQM/HcalMonitorTasks/interface/HcalBaseDQMonitor.h>
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "CondFormats/HcalObjects/interface/HcalLogicalMap.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalLogicalMapGenerator.h"

#include <iostream>
#include <vector>

/*
 * \file HcalBaseDQMonitor.cc
 *
 * $Date: 2012/06/27 13:20:29 $
 * $Revision: 1.8 $
 * \author J Temple
 *
 * Base class for all Hcal DQM analyzers
 *

*/

// constructor

HcalBaseDQMonitor::HcalBaseDQMonitor(const edm::ParameterSet& ps)
{
  Online_                = ps.getUntrackedParameter<bool>("online",false);
  mergeRuns_             = ps.getUntrackedParameter<bool>("mergeRuns",false);
  enableCleanup_         = ps.getUntrackedParameter<bool>("enableCleanup",false);
  debug_                 = ps.getUntrackedParameter<int>("debug",0);
  prefixME_              = ps.getUntrackedParameter<std::string>("subSystemFolder","Hcal/"); 
  if (prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
    prefixME_.append("/");
  subdir_                = ps.getUntrackedParameter<std::string>("TaskFolder","Test/"); 
  if (subdir_.size()>0 && subdir_.substr(subdir_.size()-1,subdir_.size())!="/")
    subdir_.append("/");
  subdir_=prefixME_+subdir_;
  AllowedCalibTypes_     = ps.getUntrackedParameter<std::vector<int> > ("AllowedCalibTypes");
  skipOutOfOrderLS_      = ps.getUntrackedParameter<bool>("skipOutOfOrderLS",false);
  NLumiBlocks_           = ps.getUntrackedParameter<int>("NLumiBlocks",4000);
  makeDiagnostics_       = ps.getUntrackedParameter<bool>("makeDiagnostics",false);
  
  setupDone_ = false;
  logicalMap_= 0;
  needLogicalMap_=false;
  meIevt_=0;
  meLevt_=0;
  meTevtHist_=0;
  ProblemsVsLB=0;
  ProblemsVsLB_HB=0;
  ProblemsVsLB_HE=0;
  ProblemsVsLB_HF=0;
  ProblemsVsLB_HBHEHF=0;
  ProblemsVsLB_HO=0;
  ProblemsCurrentLB=0;
} //HcalBaseDQMonitor::HcalBaseDQMonitor(const ParameterSet& ps)


// destructor

HcalBaseDQMonitor::~HcalBaseDQMonitor()
{
  if (logicalMap_) delete logicalMap_;
}

void HcalBaseDQMonitor::beginJob(void)
{

  if (debug_>0) std::cout <<"HcalBaseDQMonitor::beginJob():  task =  '"<<subdir_<<"'"<<std::endl;
  dbe_ = edm::Service<DQMStore>().operator->();

  ievt_=0;
  levt_=0;
  tevt_=0;
  currenttype_=-1;
  HBpresent_=false;
  HEpresent_=false;
  HOpresent_=false;
  HFpresent_=false;


} // beginJob()

void HcalBaseDQMonitor::endJob(void)
{
  if (enableCleanup_)
    cleanup();
} // endJob()

void HcalBaseDQMonitor::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
  if (debug_>0) std::cout <<"HcalBaseDQMonitor::beginRun():  task =  '"<<subdir_<<"'"<<std::endl;
  if (! mergeRuns_)
    {
      this->setup();
      this->reset();
    }
  else if (tevt_ == 0)
    {
      this->setup();
      this->reset();
    }
} // beginRun(const edm::Run& run, const edm::EventSetup& c)

void HcalBaseDQMonitor::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  if (debug_>0) std::cout <<"HcalBaseDQMonitor::endRun:  task = "<<subdir_<<std::endl;
} //endRun(...)

void HcalBaseDQMonitor::reset(void)
{
  if (debug_>0) std::cout <<"HcalBaseDQMonitor::reset():  task = "<<subdir_<<std::endl;
  if (meIevt_) meIevt_->Fill(-1);
  ievt_=0;
  if (meLevt_) meLevt_->Fill(-1);
  levt_=0;
  if (meTevt_) meTevt_->Fill(-1);
  tevt_=0;
  if (meTevtHist_) meTevtHist_->Reset();
  if (ProblemsCurrentLB) ProblemsCurrentLB->Reset();
  HBpresent_=false;
  HEpresent_=false;
  HOpresent_=false;
  HFpresent_=false;
  currentLS=0;
  currenttype_=-1;
} //reset()

void HcalBaseDQMonitor::cleanup(void)
{

} //cleanup()

void HcalBaseDQMonitor::setup(void)
{
  if (setupDone_)
    return;
  setupDone_ = true;
  if (debug_>3) std::cout <<"<HcalBaseDQMonitor> setup in progress"<<std::endl;
  dbe_->setCurrentFolder(subdir_);
  meIevt_ = dbe_->bookInt("EventsProcessed");
  if (meIevt_) meIevt_->Fill(-1);
  meLevt_ = dbe_->bookInt("EventsProcessed_currentLS");
  if (meLevt_) meLevt_->Fill(-1);
  meTevt_ = dbe_->bookInt("EventsProcessed_Total");
  if (meTevt_) meTevt_->Fill(-1);
  meTevtHist_=dbe_->book1D("Events_Processed_Task_Histogram","Counter of Events Processed By This Task",1,0.5,1.5);
  if (meTevtHist_) meTevtHist_->Reset();
  dbe_->setCurrentFolder(subdir_+"LSvalues");
  ProblemsCurrentLB=dbe_->book2D("ProblemsThisLS","Problem Channels in current Lumi Section",
				 7,0,7,1,0,1);
  if (ProblemsCurrentLB)
    {
      (ProblemsCurrentLB->getTH2F())->GetXaxis()->SetBinLabel(1,"HB");
      (ProblemsCurrentLB->getTH2F())->GetXaxis()->SetBinLabel(2,"HE");
      (ProblemsCurrentLB->getTH2F())->GetXaxis()->SetBinLabel(3,"HO");
      (ProblemsCurrentLB->getTH2F())->GetXaxis()->SetBinLabel(4,"HF");
      (ProblemsCurrentLB->getTH2F())->GetXaxis()->SetBinLabel(5,"HO0");
      (ProblemsCurrentLB->getTH2F())->GetXaxis()->SetBinLabel(6,"HO12");
      (ProblemsCurrentLB->getTH2F())->GetXaxis()->SetBinLabel(7,"HFlumi");
      (ProblemsCurrentLB->getTH2F())->GetYaxis()->SetBinLabel(1,"Status");
      ProblemsCurrentLB->Reset();
    }
} // setup()


void HcalBaseDQMonitor::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
			  const edm::EventSetup& c)
{
  if (this->LumiInOrder(lumiSeg.luminosityBlock())==false) return;
  currentLS=lumiSeg.luminosityBlock();
  levt_=0;
  if (meLevt_!=0) meLevt_->Fill(-1);
  if (ProblemsCurrentLB)
    ProblemsCurrentLB->Reset();
}

void HcalBaseDQMonitor::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
			const edm::EventSetup& c)
{
  if (this->LumiInOrder(lumiSeg.luminosityBlock())==false) return;
  // Inherited classes do end-of-lumi functions here
}


bool HcalBaseDQMonitor::LumiInOrder(int lumisec)
{
  if (skipOutOfOrderLS_==false) return true; // don't skip out-of-order lumi sections
  // check that latest lumi section is >= last processed
  if (Online_ && lumisec<currentLS)
    return false;
  return true;
}

bool HcalBaseDQMonitor::IsAllowedCalibType()
{
  if (debug_>9) std::cout <<"<HcalBaseDQMonitor::IsAllowedCalibType>"<<std::endl;
  if (AllowedCalibTypes_.size()==0)
    {
      if (debug_>9) std::cout <<"\tNo calib types specified by user; All events allowed"<<std::endl;
      return true;
    }
  MonitorElement* me = dbe_->get((prefixME_+"HcalInfo/CURRENT_EVENT_TYPE").c_str());
  if (me) currenttype_=me->getIntValue();
  else 
    {
      if (debug_>9) std::cout <<"\tCalib Type cannot be determined from HcalMonitorModule"<<std::endl;
      return true; // is current type can't be determined, assume event is allowed
    }
  if (debug_>9) std::cout <<"\tHcalBaseDQMonitor::IsAllowedCalibType  checking if calibration type = "<<currenttype_<<" is allowed...";
  for (std::vector<int>::size_type i=0;i<AllowedCalibTypes_.size();++i)
    {
      if (AllowedCalibTypes_[i]==currenttype_)
	{
	  if (debug_>9) std::cout <<"\t Type allowed!"<<std::endl;
	  return true;
	}
    }
  if (debug_>9) std::cout <<"\t Type not allowed!"<<std::endl;
  return false;
} // bool HcalBaseDQMonitor::IsAllowedCalibType()

void HcalBaseDQMonitor::getLogicalMap(const edm::EventSetup& c) {
  if (needLogicalMap_ && logicalMap_==0) {
    edm::ESHandle<HcalTopology> pT;
    c.get<IdealGeometryRecord>().get(pT);   
    HcalLogicalMapGenerator gen;
    logicalMap_=new HcalLogicalMap(gen.createMap(&(*pT)));
  }
}
 
void HcalBaseDQMonitor::analyze(const edm::Event& e, const edm::EventSetup& c)
{
  getLogicalMap(c);

  if (debug_>5) std::cout <<"\t<HcalBaseDQMonitor::analyze>  event = "<<ievt_<<std::endl;
  eventAllowed_=true; // assume event is allowed

  // fill with total events seen (this differs from ievent, which is total # of good events)
  ++tevt_;
  if (meTevt_) meTevt_->Fill(tevt_);
  if (meTevtHist_) meTevtHist_->Fill(1);
  // skip out of order lumi events
  if (this->LumiInOrder(e.luminosityBlock())==false)
    {
      eventAllowed_=false;
      return;
    }
  // skip events of wrong calibration type
  eventAllowed_&=(this->IsAllowedCalibType());
  if (!eventAllowed_) return;

  // Event is good; count it
  ++ievt_;
  ++levt_;
  if (meIevt_) meIevt_->Fill(ievt_);
  if (meLevt_) meLevt_->Fill(levt_);


  MonitorElement* me;
  if (HBpresent_==false)
    {
      me = dbe_->get((prefixME_+"HcalInfo/HBpresent"));
      if (me==0 || me->getIntValue()>0) HBpresent_=true;
    }
  if (HEpresent_==false)
    {
      me = dbe_->get((prefixME_+"HcalInfo/HEpresent"));
      if (me==0 || me->getIntValue()>0) HEpresent_=true;
    }
  if (HOpresent_==false)
    {
      me = dbe_->get((prefixME_+"HcalInfo/HOpresent"));
      if (me==0 || me->getIntValue()>0) HOpresent_=true;
    }
  if (HFpresent_==false)
    {
      me = dbe_->get((prefixME_+"HcalInfo/HOpresent"));
      if (me ==0 || me->getIntValue()>0) HFpresent_=true;
    }


} // void HcalBaseDQMonitor::analyze(const edm::Event& e, const edm::EventSetup& c)

DEFINE_FWK_MODULE(HcalBaseDQMonitor);
