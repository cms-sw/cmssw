// $Id: HLTScalersClient.cc,v 1.12 2009/11/22 13:33:05 puigh Exp $
// 
// $Log: HLTScalersClient.cc,v $
// Revision 1.12  2009/11/22 13:33:05  puigh
// clean beginJob
//
// Revision 1.11  2009/11/07 03:35:05  lorenzo
// fixed binning
//
// Revision 1.10  2009/11/04 06:00:05  lorenzo
// changed folders
//

#include <cassert>
#include <numeric>

#include "DQM/TrigXMonitorClient/interface/HLTScalersClient.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"


#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


using edm::LogInfo;
using edm::LogWarning;

#define SECS_PER_LUMI_SECTION 93.3
const int kPerHisto = 20;
const int kNumHistos = MAX_PATHS/kPerHisto; // this hasta be w/o remainders




/// Constructors
HLTScalersClient::HLTScalersClient(const edm::ParameterSet& ps):
  dbe_(0),
  nLumi_(0),
  currentRate_(0),
  currentLumiBlockNumber_(0),
  first_(true), missingPathNames_(true),
  folderName_(ps.getUntrackedParameter<std::string>("dqmFolder", 
						    "HLT/HLScalers_EvF")),
  kRateIntegWindow_(ps.getUntrackedParameter<unsigned int>("rateIntegWindow", 3)),
  processName_(ps.getParameter<std::string>("processName")),
  ignores_(),
  debug_(ps.getUntrackedParameter<bool>("debugDump", false))
{
  LogDebug("HLTScalersClient") << "constructor" ;
  if ( debug_ ) {
    textfile_.open("debug.txt");
    if ( ! textfile_ ) {
      std::cout << "constructor: can't open text file" << std::endl;
    }
  }
  // get back-end interface
  dbe_ = edm::Service<DQMStore>().operator->();
  dbe_->setVerbose(1);
  dbe_->setCurrentFolder(folderName_);

  currentRate_ = dbe_->book1D("cur_rate", 
			      "current lumi section rate per path",
			      MAX_PATHS, -0.5, MAX_PATHS-0.5);

  // REMOVE ONCE I GET THE CONFIG WORKING
  char rates_subfolder[256]; snprintf(rates_subfolder, 256, "%s/RateHistory",
				      folderName_.c_str());
  char counts_subfolder[256]; snprintf(counts_subfolder, 256, "%s/CountHistory",
				      folderName_.c_str());
  for (int i = 0; i < MAX_PATHS; ++i ) {
    rateHistories_[i] = 0; // not really needed but ...
    char name[256]; snprintf(name, 256, "rate_p%03d", i);
//     LogDebug("HLTScalersClient") << "name " << i << " is " << name ;
    dbe_->setCurrentFolder(rates_subfolder);
    rateHistories_[i] = dbe_->book1D(name, name, MAX_LUMI_SEG_HLT, 
				     -0.5, MAX_LUMI_SEG_HLT-0.5);
    dbe_->setCurrentFolder(counts_subfolder);
    snprintf(name, 256, "counts_p%03d", i);
    countHistories_[i] = dbe_->book1D(name, name, MAX_LUMI_SEG_HLT, 
					-0.5, MAX_LUMI_SEG_HLT-0.5);
  }
  dbe_->setCurrentFolder(folderName_);

  // split hlt scalers up into groups of 20, assuming total of 200 paths
  char metitle[40]; //histo name
  char mename[40]; //ME name
  for( int k = 0; k < kNumHistos; k++ ) {
    int npath_low = kPerHisto*k;
    int npath_high = kPerHisto*(k+1)-1;
    snprintf(mename, 40, "hltScalers_%0d", k);
    snprintf(metitle, 40, "HLT scalers - Paths %d to %d", npath_low, 
	     npath_high);
    hltCurrentRate_[k]= dbe_->book1D(mename, metitle, kPerHisto, 
				     -0.5 + npath_low, npath_high+0.5);
  }
  for (int i = 0; i < MAX_PATHS; ++i ) {
    for (int j = 0; j < MAX_LUMI_SEG_HLT; ++j ) {
      counts_[i][j] = 0;
    }
  }

  updates_ = dbe_->book1D("Updates", "Status of Updates", 3, 0, 3);
  updates_->setBinLabel(1, "Good Updates");
  updates_->setBinLabel(2, "Incomplete Updates");
  updates_->setBinLabel(3, "Counter Resets");

//   hltRate_ = dbe_->book1D("hltRate", "Overall HLT rate vs LS", MAX_LUMI_SEG_HLT, -0.5, 
// 			  MAX_LUMI_SEG_HLT-0.5);

} // end constructor


/// BeginJob
void HLTScalersClient::beginJob(void)
{
  LogDebug("HLTScalersClient") << "beingJob" ;
  if (dbe_) {
    dbe_->setCurrentFolder(folderName_);
  }
  first_ = true;
  missingPathNames_ = true;
}


/// beginRun
void HLTScalersClient::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
  missingPathNames_ = true;
  first_ = true;
  LogDebug("HLTScalersClient") << "beginRun, run " << run.id();
//   std::cout << "BeginRun!!!" << std::endl;
// //   if (!hltConfig_.init(processName_)) {
// //     LogDebug("HLTScalersClient") << "HLTConfigProvider failed to initialize.";
// //   }
// //   else {
// //     const unsigned int nTrigs(hltConfig_.size());
//   unsigned int nTrigs = 200;
//     // check if trigger name in (new) config
//     //	cout << "Available TriggerNames are: " << endl;
//     //	hltConfig_.dump("Triggers");
//     LogDebug("HLTScalersClient") << "HLTConfigProvider size is " << nTrigs;
//     for (unsigned int i=0; i < nTrigs; ++i) {
//       char name[256]; snprintf(name, 256, "p%d", i);
//       std::string pathname(name);
//       //std::string pathname = hltConfig_.triggerName(i);
//       std::cout << "Pathname " << i << " is " <<  pathname << std::endl;

//       //scalerCounters_[i] = 0UL;
//       rateHistories_[i] = 0; // not really needed but ...
      
//       char title[256]; snprintf(title, 256, "Rate: %s (Path %d)", 
// 				pathname.c_str(), i);
//       LogDebug("HLTScalersClient") << "name " << i << " is " << pathname ;
//       rateHistories_[i] = dbe_->book1D(pathname.c_str(), title, MAX_LUMI_SEG_HLT, 
// 				       -0.5, MAX_LUMI_SEG_HLT-0.5);
//       snprintf(name, 256, "counts_%s", pathname.c_str());
//       snprintf(title, 256, "Counts for %s (Path %d)", pathname.c_str(), i);  
//       countHistories_[i] = dbe_->book1D(name, title, MAX_LUMI_SEG_HLT, 
// 					-0.5, MAX_LUMI_SEG_HLT-0.5);
//     } // loop over trigger paths


//     //  }

} // beginRun

/// EndRun
void HLTScalersClient::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  missingPathNames_ = true;
  first_ = true;
}


/// End LumiBlock
/// DQM Client Diagnostic should be performed here
void HLTScalersClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
			const edm::EventSetup& c)
{
  nLumi_ = lumiSeg.id().luminosityBlock();
  // get raw data
  std::string scalHisto_ = folderName_ + "/hltScalers";
  // PWDEBUG
  if ( first_)
    dbe_->showDirStructure();
  // PWDEBUG END

  MonitorElement *scalers = dbe_->get(scalHisto_);
  if ( scalers == 0 ) {
    LogDebug("HLTScalersClient") << "cannot get hlt scalers histogram, "
				 << "bailing out.";
    if ( debug_ )
      std::cout << "No scalers ? " << std::endl;
    return;
  }


  int npaths = scalers->getNbinsX();
  if ( npaths > MAX_PATHS ) npaths = MAX_PATHS; // HARD CODE FOR NOW
  LogDebug("HLTScalersClient") << "I see " << npaths << " paths. ";

  // set the bin labels on the first go-through
  // I need to do this here because we don't have the paths yet
  // on begin-run. I should do this in a less ugly way (see FV?)
  if ( first_) {
    LogDebug("HLTScalersClient") << "Setting up paths on first endLumiBlock "
				 << npaths;
    recentPathCountsPerLS_.reserve(MAX_PATHS);
    for ( int i = 0; i < MAX_PATHS; ++i ) {
      // prefill the data structures
      recentPathCountsPerLS_.push_back(CountLSFifo_t());
    }
    first_ = false;
  }

  if ( missingPathNames_) {
    for ( int i = 0; i < npaths; ++i ) {
      // set up the histogram names
      int whichHisto = i/kPerHisto;
      int whichBin = i%kPerHisto + 1;
      char pname[256];
      snprintf(pname, 256, "%s/path%03d",folderName_.c_str(), i);
//       LogDebug("HLTScalersClient") << "Looking for path " << pname;
      MonitorElement *name = dbe_->get(pname);
      std::string sname;
      if ( name ) {
	sname = std::string (name->getStringValue());
	missingPathNames_ = false;
      }
      else {
	sname = std::string("unknown");
// 	// PW DEBUG
// 	std::cout << "Can't find path name for path " << i << std::endl;
// 	// PW DEBUG
      }
      hltCurrentRate_[whichHisto]->setBinLabel(whichBin, sname.c_str());
      snprintf(pname, 256, "Rate - path %s (Path # %03d)", sname.c_str(), i);
      rateHistories_[i]->setTitle(pname);
      currentRate_->setBinLabel(i+1, sname.c_str());
      scalers->setBinLabel(i+1, sname.c_str());

      snprintf(pname, 256, "Count - path %s (Path # %03d)", sname.c_str(), i);
      countHistories_[i]->setTitle(pname);
    }
    if ( ! missingPathNames_ ) {
      if ( debug_ ) 
	std::cout << "Yaay! found pathnames. " << std::endl;
    }
  }

  // END SETUP

  std::string nLumiHisto_ = folderName_ + "/nLumiBlock";
  MonitorElement *nLumi = dbe_->get(nLumiHisto_);
  int testval = (nLumi!=0?nLumi->getIntValue():-1);
  LogDebug("HLTScalersClient") << "Lumi Block from DQM: "
			<< testval
			<< ", local is " << nLumi_;
  int nL = (nLumi!=0?nLumi->getIntValue():nLumi_);
  if ( nL > MAX_LUMI_SEG_HLT ) {
    LogDebug("HLTScalersClient") << "Too many Lumi segments, "
				 << nL << " is greater than MAX_LUMI_SEG_HLT,"
				 << " wrapping to " 
				 << (nL%MAX_LUMI_SEG_HLT);
    //nL = MAX_LUMI_SEG_HLT;
    nL = nL%MAX_LUMI_SEG_HLT;
  }

  // evaluate the data
  // loop over current counts
  int localignores = 0;
  for ( int i = 1; i <= npaths; ++i ) { // bins start at 1
    double current_count = scalers->getBinContent(i);
    countHistories_[i-1]->Fill(nL, current_count); // good or bad
    if ( recentPathCountsPerLS_[i].empty() || 
	current_count >= recentPathCountsPerLS_[i].front().second) {
      // good data
      // DEBUG
      //std::cout << "new point is " << current_count << std::endl; 
      if ( ! recentPathCountsPerLS_[i].empty() ) 
	if ( debug_ ) 
	  std::cout << "\t-> good one: new => cnt, ls = " 
		    << recentPathCountsPerLS_[i].front().second << "\t"
		    << recentPathCountsPerLS_[i].front().first << "\t"
		    << std::endl;
      // END DEBUG
      recentPathCountsPerLS_[i].push_front(CountLS_t(nL,current_count));
      while ( recentPathCountsPerLS_[i].size() > kRateIntegWindow_ ) {
	recentPathCountsPerLS_[i].pop_back();
      }
      // NB: we do not fill a new entry in the rate histo if we can't 
      // calculate it
      std::pair<double,double> sl =  getSlope_(recentPathCountsPerLS_[i]);
      double slope = sl.first; double slope_err = sl.second;
      //rateHistories_[i-1]->Fill(nL,slope);
      if ( slope > -1 ) {
	rateHistories_[i-1]->setBinContent(nL,slope);
	rateHistories_[i-1]->setBinError(nL,slope_err);
	// set the current rate(s)
	hltCurrentRate_[i/kPerHisto]->setBinContent(i%kPerHisto, slope);
	hltCurrentRate_[i/kPerHisto]->setBinError(i%kPerHisto, slope_err);
	currentRate_->setBinContent(i, slope);
	currentRate_->setBinError(i, slope_err);
      }
      
    }
    else {
      ++localignores;
      if ( debug_ ) {
	std::cout << "Ignoring point " << i << " with data "
		  << current_count ;
	double cnt = recentPathCountsPerLS_[i].empty()?
	  -1:recentPathCountsPerLS_[i].front().second;
	std::cout << "(old is " << cnt << ")" << std::endl;
      }

    }
  }
  ignores_.push_front(localignores);
  while ( ignores_.size() > 3 )
    ignores_.pop_back();
  updates_->Fill(localignores>0?1:0);
  
  int totignores = std::accumulate(ignores_.begin(), ignores_.end(), 0);
  if ( debug_ ) 
    std::cout << "Tot ignores is " << totignores << std::endl;
  if ( totignores > 150 ) {
    // clear 'em out
    LogDebug("HLTScalersClient") << "Too many ignores ("
				 << totignores 
				 << "), resetting counters.";
    for ( int i = 0; i < MAX_PATHS; ++i ) {
      recentPathCountsPerLS_[i].clear();
    }
    ignores_.clear();
    updates_->Fill(2); // keep track of these resets
  }

    


  // PW DEBUG
  if ( debug_ ) {
    textfile_ << nL << "\t"
	      << npaths << "\t";
    for ( int i = 0; i < npaths ; ++i ){
      textfile_ << scalers->getBinContent(i) << " ";
    }
    textfile_ << std::endl;
  }

    

}

// unused
void HLTScalersClient::analyze(const edm::Event& e, const edm::EventSetup& c) 
{
  // nothing to do here
}

// this is probably overkill
// note that the data is in units of counts, ls number
// but we return a value in Hz...
std::pair<double,double>
HLTScalersClient::getSlope_(HLTScalersClient::CountLSFifo_t points)
{
  // this is probably total overkill
  if (points.size() < kRateIntegWindow_ ) 
    return std::pair<double,double>(-1,0);
  double xy = 0;
  double x = 0;
  double xsq = 0;
  double y = 0;
  double n = double(points.size());
  for ( CountLSFifo_t::iterator i(points.begin());
	i != points.end(); ++i ) {
    if ( debug_ ) 
      std::cout << "x = " << i->first << ", y = " << i->second 
		<< std::endl;
    xy += i->first * i->second;
    x += i->first;
    xsq += i->first*i->first;
    y += i->second;
  }
  double slope = (n*xy - x*y)/(n*xsq - x*x);

  // now get the uncertainty on the slope. Need intercept for this.
  double intercept = (xsq*y - xy*x)/(n*xsq-x*x);
  double sigma_ysq = 0;
  for ( CountLSFifo_t::iterator i(points.begin());
	i != points.end(); ++i ) {
    sigma_ysq += pow(( i->second - slope * i->first  - intercept),2.);
  }
  if ( debug_ ) 
    std::cout << "chi^2 = " << sigma_ysq << std::endl;
  sigma_ysq *= 1./(n-2.);

  double sigma_m = sqrt( n*sigma_ysq/(n*xsq - x*x));

  //  std::cout << "Slope is " << slope << std::endl;
  slope /= SECS_PER_LUMI_SECTION;
  sigma_m /= SECS_PER_LUMI_SECTION;
  if ( debug_ ) {
    std::cout << "Slope is " << slope << " +- " << sigma_m 
	      << std::endl;
//   std::cout << "intercept is " << intercept
//  	    << std::endl;
  }


  return std::pair<double,double>(slope, sigma_m);
}
