#include "DQM/HcalMonitorTasks/interface/HcalLSbyLSMonitor.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

/*
  This task looks at the output from all other tasks (as stored in their 
  ProblemsCurrentLB histograms), and combines the output into its own ProblemsCurrentLB histogram.  Its own ProblemsCurrentLB histogram, though, is marked with the setLumiFlag, so that its contents will be stored for every lumi section.
*/

// constructor
HcalLSbyLSMonitor::HcalLSbyLSMonitor(const edm::ParameterSet& ps) 
{
  Online_                = ps.getUntrackedParameter<bool>("online",false);
  mergeRuns_             = ps.getUntrackedParameter<bool>("mergeRuns",false);
  enableCleanup_         = ps.getUntrackedParameter<bool>("enableCleanup",false);
  debug_                 = ps.getUntrackedParameter<int>("debug",0);
  prefixME_              = ps.getUntrackedParameter<std::string>("subSystemFolder","Hcal/");
  if (prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
    prefixME_.append("/");
  subdir_                = ps.getUntrackedParameter<std::string>("TaskFolder","LSbyLSMonitor_Hcal"); 
  if (subdir_.size()>0 && subdir_.substr(subdir_.size()-1,subdir_.size())!="/")
    subdir_.append("/");
  subdir_=prefixME_+subdir_;
  AllowedCalibTypes_     = ps.getUntrackedParameter<std::vector<int> > ("AllowedCalibTypes");
  skipOutOfOrderLS_      = ps.getUntrackedParameter<bool>("skipOutOfOrderLS",true);
  NLumiBlocks_           = ps.getUntrackedParameter<int>("NLumiBlocks",4000);
  makeDiagnostics_       = ps.getUntrackedParameter<bool>("makeDiagnostics",false);

  // Specify the directories where the tasks to be check reside
  TaskList_              = ps.getUntrackedParameter<std::vector<std::string> >("TaskDirectories");
  // Minimum number of events per lumi section that must be present for checks to be made.  *ALL* tasks must contain at least this many events
  minEvents_             = ps.getUntrackedParameter<int>("minEvents",500);
}

HcalLSbyLSMonitor::~HcalLSbyLSMonitor()
{
} //destructor

void HcalLSbyLSMonitor::setup()
{
  // Call base class setup
  HcalBaseDQMonitor::setup();

  if (debug_>1)
    std::cout <<"<HcalLSbyLSMonitor::setup>  Setting up histograms"<<std::endl;

  dbe_->setCurrentFolder(subdir_);
  // This will cause this information to be kept for every lumi block
  if (ProblemsCurrentLB)
    ProblemsCurrentLB->setLumiFlag();
  this->reset();
}
void HcalLSbyLSMonitor::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
  if (debug_>1) std::cout <<"HcalLSbyLSMonitor::beginRun"<<std::endl;
  HcalBaseDQMonitor::beginRun(run,c);

  if (tevt_==0) this->setup(); // set up histograms if they have not been created before
  if (mergeRuns_==false)
    this->reset();

  return;
} //void HcalLSbyLSMonitor::beginRun(...)


void HcalLSbyLSMonitor::reset()
{
  HcalBaseDQMonitor::reset();
  ProblemsCurrentLB->Reset();
}  // reset function is empty for now

void HcalLSbyLSMonitor::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
					      const edm::EventSetup& c)
{
  if (LumiInOrder(lumiSeg.luminosityBlock())==false) return;
  HcalBaseDQMonitor::beginLuminosityBlock(lumiSeg,c);
  ProblemsCurrentLB->Reset();
  return;
} // beginLuminosityBlock(...)

void HcalLSbyLSMonitor::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
					    const edm::EventSetup& c)
{
  // Processing should go here
  if (!dbe_) return;
  bool enoughEvents=true;
  int Nevents=0;
  int TotalEvents=0;
  int badHB=0;
  int badHE=0;
  int badHO=0;
  int badHF=0;
  int badHO0=0;
  int badHO12=0;
  int badHFLUMI=0;
  
  for (unsigned int i=0;i<TaskList_.size();++i)
    {
      std::string name=prefixME_+TaskList_[i]+"LSvalues/";
      dbe_->setCurrentFolder(name.c_str());
      // Do we need the 'name' prefix here?
      MonitorElement *me=dbe_->get(name+"ProblemsThisLS");
      if (me==0)
	{
	  if (debug_>0) std::cout <<"<HcalLSbyLSMonitor>  Error!  Could not get histogram "<<name.c_str()<<std::endl;
	  enoughEvents=false;
	  break;
	}

      Nevents=me->getBinContent(-1);
      if (Nevents<minEvents_)
	{
	  if (debug_>0) 
	    std::cout <<"<HcalLSbyLSMonitor>  Error!  Number of events "<<Nevents<<" for histogram "<<name.c_str()<<" is less than the required minimum of "<<minEvents_<<std::endl;
	  enoughEvents=false;
	  break;
	}
      // Total events is the number of events processed in this LS
      TotalEvents=std::max(TotalEvents,Nevents);
      // errors are sum over all tests.  This WILL lead to double counting in some subdetectors!
      badHB+=me->getBinContent(1,1);
      badHE+=me->getBinContent(2,1);
      badHO+=me->getBinContent(3,1);
      badHF+=me->getBinContent(4,1);
      badHO0+=me->getBinContent(5,1);
      badHO12+=me->getBinContent(6,1);
      badHFLUMI+=me->getBinContent(7,1);
    }
  if (enoughEvents==false)  // not enough events to make a decision
    {
      return;
    }
  ProblemsCurrentLB->setBinContent(-1,-1,TotalEvents);
  ProblemsCurrentLB->setBinContent(1,1,badHB);
  ProblemsCurrentLB->setBinContent(2,1,badHE);
  ProblemsCurrentLB->setBinContent(3,1,badHO);
  ProblemsCurrentLB->setBinContent(4,1,badHF);
  ProblemsCurrentLB->setBinContent(5,1,badHO0);
  ProblemsCurrentLB->setBinContent(6,1,badHO12);
  ProblemsCurrentLB->setBinContent(7,1,badHFLUMI);
  return;
}


void HcalLSbyLSMonitor::done()
{
  // moved database dumps to client; we want to be able to sum over results in offline
  return;

} // void HcalLSbyLSMonitor::done()


void HcalLSbyLSMonitor::endJob()
{
  if (debug_>0) std::cout <<"HcalLSbyLSMonitor::endJob()"<<std::endl;
  if (enableCleanup_) cleanup(); // when do we force cleanup?
}


void HcalLSbyLSMonitor::cleanup()
{
  if (!enableCleanup_) return;
  if (dbe_)
    {
      dbe_->setCurrentFolder(subdir_);
      dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"LSvalues");
      dbe_->removeContents();
    }
}

DEFINE_FWK_MODULE(HcalLSbyLSMonitor);
