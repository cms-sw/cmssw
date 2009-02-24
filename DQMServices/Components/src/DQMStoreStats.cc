/*
 * \file DQMStoreStats.cc
 * \author Andreas Meyer
 * Last Update:
 * $Date: 2009/02/23 10:52:59 $
 * $Revision: 1.2 $
 * $Author: ameyer $
 *
 * Description: Print out statistics of histograms in DQMStore
*/

#include "DQMServices/Components/src/DQMStoreStats.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

using namespace std;
using namespace edm;

//==================================================================//
//================= Constructor and Destructor =====================//
//==================================================================//
DQMStoreStats::DQMStoreStats( const edm::ParameterSet& ps )
  : subsystem_ (""),
    subfolder_ (""),
    nbinsglobal_ (0),
    nbinssubsys_ (0),
    nmeglobal_ (0),
    nmesubsys_ (0),
    maxbinsglobal_ (0),
    maxbinssubsys_ (0),
    maxbinsmeglobal_ (""),
    maxbinsmesubsys_ (""),
    statsdepth_ (1),
    pathnamematch_ ("*"),
    verbose_ (0)
{
  parameters_ = ps;
  pathnamematch_ = ps.getUntrackedParameter<std::string>("pathNameMatch", pathnamematch_);
  statsdepth_ = ps.getUntrackedParameter<int>("statsDepth", statsdepth_);
  verbose_ = ps.getUntrackedParameter<int>("verbose", verbose_);
  runonendrun_    = ps.getUntrackedParameter<bool>("runOnEndRun",true);
  runonendjob_    = ps.getUntrackedParameter<bool>("runOnEndJob",false);
  runonendlumi_   = ps.getUntrackedParameter<bool>("runOnEndLumi",false);
  runineventloop_ = ps.getUntrackedParameter<bool>("runInEventLoop",false);
}

DQMStoreStats::~DQMStoreStats(){
}

//==================================================================//
//======================= calcstats  ===============================//
//==================================================================//
int DQMStoreStats::calcstats() {

  ////---- initialise Event and LS counters
  nbinsglobal_ = 0; 
  nbinssubsys_ = 0; 
  maxbinsglobal_ = 0; 
  maxbinssubsys_ = 0; 
  std::string path = "";
  std::string subsystemname = "";
  std::string subfoldername = "";
  int xbins=0,ybins=0,zbins=0;  
  
  std::cout << " DQMStoreStats::calcstats ==============================" << std::endl;  
    cout << "  running " ; 
    if (runonendrun_) cout << "on run end " << endl;
    if (runonendlumi_) cout << "on lumi end " << endl;
    if (runonendjob_) cout << "on job end " << endl;
    if (runineventloop_) cout << "in event loop " << endl;

    if (verbose_) { 
      cout << "  pathNameMatch = " << pathnamematch_ << endl;
      cout << "  statsDepth = " << statsdepth_ << endl;
    }

  std::vector<MonitorElement*> melist;
  melist = dbe_->getMatchingContents(pathnamematch_);

  typedef std::vector <MonitorElement*>::iterator meIt;
  for(meIt it = melist.begin(); it != melist.end(); ++it) {

   if ((*it)->kind()==MonitorElement::DQM_KIND_TH1F ||
       (*it)->kind()==MonitorElement::DQM_KIND_TH1S ||
       (*it)->kind()==MonitorElement::DQM_KIND_TPROFILE ) {
     xbins = (*it)->getNbinsX();
     ybins = 1; 
     zbins = 1;
     }
   else if ((*it)->kind()==MonitorElement::DQM_KIND_TH2F ||
       (*it)->kind()==MonitorElement::DQM_KIND_TH2S ||
       (*it)->kind()==MonitorElement::DQM_KIND_TPROFILE2D ) {
     xbins = (*it)->getNbinsX();
     ybins = (*it)->getNbinsY();
     zbins = 1;
     }
   else if ((*it)->kind()==MonitorElement::DQM_KIND_TH3F) {
     xbins = (*it)->getNbinsX();
     ybins = (*it)->getNbinsY();
     zbins = (*it)->getNbinsZ();
     }
   else {
     xbins = 1; ybins=1; zbins=1 ;
     }

   
     //  figure out subsystem name
     std::string path =  (*it)->getPathname();
     size_t start = 0;
     size_t end = path.find('/',start);
     if (end == std::string::npos) end = path.size(); 
     subsystemname=path.substr(start,end);
     
     //  go into next level subsystem folder if required
     if (statsdepth_>1 && end < std::string::npos ) {
         start=end+1;
         end = path.find('/',start);
         if (end == std::string::npos) end = path.size();
     }
     subfoldername=path.substr(start,end);
     
          
     // if new subsystem print old one and reset subsystem counters
     if (subfolder_!=subfoldername) {
        // subsystem info printout
        print();
	// reset
	subfolder_=subfoldername;
	subsystem_=subsystemname;
        maxbinssubsys_=0;
	nbinssubsys_=0;
	nmesubsys_=0;
     }

     // add up xyz and set maximumvalues
     int xyz =xbins*ybins*zbins;
     nbinsglobal_ += xyz;
     nbinssubsys_ += xyz;
     nmeglobal_++;
     nmesubsys_++;
     if (xyz > maxbinsglobal_ ) {
        maxbinsglobal_=xyz;
	maxbinsmeglobal_=(*it)->getFullname();
     }
     if (xyz > maxbinssubsys_ ) {
        maxbinssubsys_=xyz;
	maxbinsmesubsys_=(*it)->getFullname();
     }

  } 

  // subsystem printout
  print ();

  // global summary
  std::cout << std::endl;
  std::cout <<  " ----------------------------- " << std::endl;
  std::cout <<  "  Summary: ";
  std::cout <<  nmeglobal_ << " histograms with " 
            <<  nbinsglobal_  << " bins. " ; 
  if (nmeglobal_ > 0) std::cout <<  nbinsglobal_/nmeglobal_ << " bins/histogram " ;
  std::cout << std::endl;
  std::cout <<  "  Largest histogram: " << maxbinsmeglobal_ << " with " <<
		                         maxbinsglobal_ << " bins." <<  std::endl;
  
  return 0;
  
}

// -----------------------------------------------------------------//

void DQMStoreStats::print(){
  // subsystem info printout
  std::cout << " ---------- " << subsystem_ << " ---------- " << std::endl;
  std::cout <<  "  " << subfolder_ << ": " ;
  std::cout <<  nmesubsys_ << " histograms with " 
            <<  nbinssubsys_  << " bins. " ; 
  if (nmesubsys_ > 0) std::cout <<  nbinssubsys_/nmesubsys_ << " bins/histogram " ;
  std::cout << std::endl;
  std::cout <<  "  Largest histogram: " << maxbinsmesubsys_ << " with " <<
		                         maxbinssubsys_ << " bins." <<  std::endl;
}

//==================================================================//
//========================= beginJob ===============================//
//==================================================================//
void DQMStoreStats::beginJob(const EventSetup& context) {

  ////---- get DQM store interface
  dbe_ = Service<DQMStore>().operator->();

}

//==================================================================//
//========================= beginRun ===============================//
//==================================================================//
void DQMStoreStats::beginRun(const edm::Run& r, const EventSetup& context) {
}


//==================================================================//
//==================== beginLuminosityBlock ========================//
//==================================================================//
void DQMStoreStats::beginLuminosityBlock(const LuminosityBlock& lumiSeg,
					    const EventSetup& context) {
}


//==================================================================//
//==================== analyse (takes each event) ==================//
//==================================================================//
void DQMStoreStats::analyze(const Event& iEvent, const EventSetup& iSetup) {
   if (runineventloop_) calcstats();
}

//==================================================================//
//========================= endLuminosityBlock =====================//
//==================================================================//
void DQMStoreStats::endLuminosityBlock(const LuminosityBlock& lumiSeg,
					  const EventSetup& context) {
   if (runonendlumi_) calcstats();
}

//==================================================================//
//============================= endRun =============================//
//==================================================================//
void DQMStoreStats::endRun(const Run& r, const EventSetup& context) {
   if (runonendrun_) calcstats();
}

//==================================================================//
//============================= endJob =============================//
//==================================================================//
void DQMStoreStats::endJob() {
   if (runonendjob_) calcstats();
}
