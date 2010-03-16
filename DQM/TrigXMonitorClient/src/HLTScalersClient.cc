// $Id: HLTScalersClient.cc,v 1.15 2010/02/11 23:55:18 wittich Exp $
// 
// $Log: HLTScalersClient.cc,v $
// Revision 1.15  2010/02/11 23:55:18  wittich
// - adapt to shorter Lumi Section length
// - fix bug in how history of counts was filled
//
// Revision 1.14  2010/02/02 11:44:20  wittich
// more diagnostics for online scalers
//
// Revision 1.13  2009/12/15 20:41:16  wittich
// better hlt scalers client
//
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

// I am not sure this is right at more than 10%
#define SECS_PER_LUMI_SECTION 23.3 
const int kPerHisto = 20;




/// Constructors
HLTScalersClient::HLTScalersClient(const edm::ParameterSet& ps):
  recentOverallCountsPerLS_(),
  recentNormedOverallCountsPerLS_(),
  dbe_(0),
  nLumi_(0),
  currentRate_(0),
  currentLumiBlockNumber_(0),
  first_(true), missingPathNames_(true),
  folderName_(ps.getUntrackedParameter<std::string>("dqmFolder", 
						    "HLT/HLScalers_EvF")),
  kRateIntegWindow_(ps.getUntrackedParameter<unsigned int>("rateIntegWindow", 
							   3)),
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



  updates_ = dbe_->book1D("Updates", "Status of Updates", 3, 0, 3);
  updates_->setBinLabel(1, "Good Updates");
  updates_->setBinLabel(2, "Incomplete Updates");
  updates_->setBinLabel(3, "Counter Resets");

  hltRate_ = dbe_->book1D("hltRate", "Overall HLT Accept rate vs LS", 
			  MAX_LUMI_SEG_HLT, -0.5, MAX_LUMI_SEG_HLT-0.5);
  hltNormRate_ = dbe_->book1D("hltRateNorm", 
			      "Overall HLT Accept rate per FU vs LS", 
			      MAX_LUMI_SEG_HLT, -0.5, MAX_LUMI_SEG_HLT-0.5);
  hltCount_= dbe_->book1D("hltCount", "Overall HLT Counts vs LS", 
			  MAX_LUMI_SEG_HLT, -0.5, MAX_LUMI_SEG_HLT-0.5);
  hltCountN_= dbe_->book1D("hltCountN", "Overall HLT Counts per LS vs LS", 
			  MAX_LUMI_SEG_HLT, -0.5, MAX_LUMI_SEG_HLT-0.5);
  mergeCount_= dbe_->book1D("mergeCount", "Number of merge counts vs LS", 
			    MAX_LUMI_SEG_HLT, -0.5, MAX_LUMI_SEG_HLT-0.5);



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
  std::string scalHisto = folderName_ + "/hltScalers";
  // PWDEBUG
  if ( first_ && debug_)
    dbe_->showDirStructure();
  // PWDEBUG END

  MonitorElement *scalers = dbe_->get(scalHisto);
  if ( scalers == 0 ) {
    // try second way
    scalHisto = folderName_ + "/raw/hltScalers";
    scalers = dbe_->get(scalHisto);
    if ( scalers == 0 ) {
      LogDebug("HLTScalersClient") << "cannot get hlt scalers histogram, "
				   << "bailing out.";
      if ( debug_ )
	std::cout << "No scalers ? Looking for " 
		  << scalHisto
		  << std::endl;
      return;
    }
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
    currentRate_ = dbe_->book1D("cur_rate", 
				"current lumi section rate per path",
				npaths, -0.5, npaths-0.5);
    currentNormRate_ = dbe_->book1D("cur_rate_norm", 
				    "current norm. lumi section rate per path",
				    npaths, -0.5, npaths-0.5);
    recentPathCountsPerLS_.reserve(npaths);
    recentNormedPathCountsPerLS_.reserve(npaths);
    char rates_subfolder[256]; snprintf(rates_subfolder, 256, "%s/RateHistory",
					folderName_.c_str());
    char counts_subfolder[256]; snprintf(counts_subfolder, 256, 
					 "%s/CountHistory", 
					 folderName_.c_str());

    hltCurrentRate_.    reserve(npaths);
    rateHistories_.     reserve(npaths);
    countHistories_.    reserve(npaths);
    hltCurrentNormRate_.reserve(npaths);
    rateNormHistories_. reserve(npaths);

    for (int i = 0; i < npaths; ++i ) {

      char name[256]; snprintf(name, 256, "rate_p%03d", i);
      //     LogDebug("HLTScalersClient") << "name " << i << " is " << name ;
      dbe_->setCurrentFolder(rates_subfolder);
      rateHistories_.push_back(dbe_->book1D(name, name, MAX_LUMI_SEG_HLT, 
					    -0.5, MAX_LUMI_SEG_HLT-0.5));
      snprintf(name, 256, "nrate_p%03d", i);
      rateNormHistories_.push_back(dbe_->book1D(name, name, MAX_LUMI_SEG_HLT, 
						-0.5, MAX_LUMI_SEG_HLT-0.5));
      dbe_->setCurrentFolder(counts_subfolder);
      snprintf(name, 256, "counts_p%03d", i);
      countHistories_.push_back(dbe_->book1D(name, name, MAX_LUMI_SEG_HLT, 
					     -0.5, MAX_LUMI_SEG_HLT-0.5));
      // prefill the data structures
      recentPathCountsPerLS_.push_back(CountLSFifo_t());
      recentNormedPathCountsPerLS_.push_back(CountLSFifo_t());
    }
    dbe_->setCurrentFolder(folderName_);


    // split hlt scalers up into groups of 20, assuming total of 200 paths
    char metitle[40]; //histo name
    char mename[40]; //ME name
    int numHistos = int(npaths/kPerHisto); // this hasta be w/o remainders
    
    int remainder = npaths%kPerHisto; 
    if ( remainder ) numHistos += 1;

    for( int k = 0; k < numHistos; k++ ) {
      int npath_low = kPerHisto*k;
      int npath_high = kPerHisto*(k+1)-1;
      snprintf(mename, 40, "hltScalers_%0d", k);
      snprintf(metitle, 40, "HLT scalers - Paths %d to %d", npath_low, 
	       npath_high);
      hltCurrentRate_.push_back(dbe_->book1D(mename, metitle, kPerHisto, 
					     -0.5 + npath_low, npath_high+0.5));
      snprintf(mename, 40, "hltScalersNorm_%0d", k);
      snprintf(metitle, 40, "HLT Normalized Rate - Paths %d to %d", npath_low, 
	       npath_high);
      hltCurrentNormRate_.push_back(dbe_->book1D(mename, metitle, kPerHisto, 
						 -0.5 + npath_low, 
						 npath_high+0.5));
    }

    first_ = false;



  }

  if ( missingPathNames_) {
    // first try the scalers histogram and see if the bin names are set
    for ( int i = 0; i < npaths; ++i ) {
      // needs to be i+1 since root bins start at 1
      const char* name = scalers->getTH1()->GetXaxis()->GetBinLabel(i+1);
      if ( name ) {
	if ( debug_ ) 
	  std::cout << "path " << i << " name is " << name << std::endl;
	int whichHisto = i/kPerHisto;
	int whichBin = i%kPerHisto + 1;
	char pname[256];
	hltCurrentRate_[whichHisto]->setBinLabel(whichBin, name);
	snprintf(pname, 256, "Rate - path %s (Path # %03d)", name, i);
	rateHistories_[i]->setTitle(pname);
	currentRate_->setBinLabel(i+1, name);
	currentNormRate_->setBinLabel(i+1, name);

	missingPathNames_ = false;
      }
    }

  }

  // END SETUP

  std::string nLumiHisto(folderName_ + "/nLumiBlock");
  MonitorElement *nLumi = dbe_->get(nLumiHisto);
  if ( nLumi == 0 ) {
    nLumiHisto = folderName_ + "/raw/nLumiBlock";
    nLumi = dbe_->get(nLumiHisto);
  }
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

  // merging counts
  double num_fu = -1.0;
  std::string mergeName(folderName_ + "/raw/hltMerge");
  MonitorElement *merge = dbe_->get(mergeName);
  if ( merge != 0 ) {
    num_fu = merge->getBinContent(1);
    if ( debug_ ) {
      std::cout << "Number of received entries: " << num_fu
		<< std::endl;
    }
    mergeCount_->Fill(nL,num_fu);
  }
  // end 


  // evaluate the data
  // loop over current counts
  int localignores = 0;
  for ( int i = 1; i <= npaths; ++i ) { // bins start at 1
    double current_count = scalers->getBinContent(i);
    countHistories_[i-1]->Fill(nL, current_count); // good or bad
    if ( recentPathCountsPerLS_[i-1].empty() || 
	current_count >= recentPathCountsPerLS_[i-1].front().second) {
      // good data
      // DEBUG
      //std::cout << "new point is " << current_count << std::endl; 
      if ( ! recentPathCountsPerLS_[i-1].empty() ) 
	if ( debug_ ) 
	  std::cout << i << "\t-> good one: new => cnt, ls = " 
		    << current_count << ", " << nL
		    << ", old = "
		    << recentPathCountsPerLS_[i-1].front().second << "\t"
		    << recentPathCountsPerLS_[i-1].front().first 
		    << std::endl;
      // END DEBUG
      recentPathCountsPerLS_[i-1].push_front(CountLS_t(nL,current_count));
      while ( recentPathCountsPerLS_[i-1].size() > kRateIntegWindow_ ) {
	recentPathCountsPerLS_[i-1].pop_back();
      }


      // NB: we do not fill a new entry in the rate histo if we can't 
      // calculate it
      std::pair<double,double> sl =  getSlope_(recentPathCountsPerLS_[i-1]);
      double slope = sl.first; double slope_err = sl.second;
      //rateHistories_[i-1]->Fill(nL,slope);
      if ( slope > -1 ) {
	rateHistories_[i-1]->setBinContent(nL,slope);
	// set the current rate(s)
	hltCurrentRate_[(i-1)/kPerHisto]->setBinContent(i%kPerHisto, slope);
	currentRate_->setBinContent(i, slope);
	if ( ! std::isnan(slope_err ) && (slope_err >= 0 ) ) {
	  currentRate_->setBinError(i, slope_err);
	  hltCurrentRate_[(i-1)/kPerHisto]->setBinError(i%kPerHisto, slope_err);
	  rateHistories_[i-1]->setBinError(nL,slope_err);
	}
	else {
	  std::cout << "slope_err is nan for number " << i 
		    << ", " << slope_err
		    << std::endl;
	}
      }
      
    }
    else {
      ++localignores;
      if ( debug_ ) {
	std::cout << i << "\t<<>> Ignoring point with data "
		  << current_count ;
	double cnt = recentPathCountsPerLS_[i-1].empty()?
	  -1:recentPathCountsPerLS_[i-1].front().second;
	std::cout << "(old is " << cnt << " )" << std::endl;
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
    for ( std::vector<CountLSFifo_t>::iterator 
	    i(recentPathCountsPerLS_.begin()); 
	  i != recentPathCountsPerLS_.end(); ++i ) {
      i->clear();
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
  // end DEBUG


  // ------ overall rate normalized - all data
  std::string overallScalerName(folderName_ + "/raw/hltOverallScalerN");
  MonitorElement *hltScaler = dbe_->get(overallScalerName);
  if ( hltScaler != 0 ) {
    double cnt = hltScaler->getBinContent(1);
    //hltCountN_->setBinContent(nL,cnt);
    hltCountN_->Fill(nL,cnt);
    if ( debug_ ) {
      std::cout << "Overall Norm: new => cnt, ls = " 
		<< cnt << ", " << nL
		<< ", num_fu = " << num_fu 
		<< std::endl;
    }
    recentNormedOverallCountsPerLS_.push_front(CountLS_t(nL,
							 cnt/num_fu));
    while ( recentNormedOverallCountsPerLS_.size() > 2 ) {
      recentNormedOverallCountsPerLS_.pop_back();
    }
    double slope = cnt / num_fu / SECS_PER_LUMI_SECTION;
    if ( debug_ )  {
      std::cout << "Normalized slope = " << slope << std::endl;
    }
    hltNormRate_->setBinContent(nL,slope);
  }

  // ---------------------------- overall rate, absolute counts
  overallScalerName = std::string(folderName_ + "/raw/hltOverallScaler");
  hltScaler = dbe_->get(overallScalerName);
  if ( hltScaler != 0 ) {
    double current_count = hltScaler->getBinContent(1);
    hltCount_->setBinContent(nL,current_count);
    if ( recentOverallCountsPerLS_.empty() ||
	 current_count >= recentOverallCountsPerLS_.front().second ) {
      // good data
//       // DEBUG
//       if ( debug_ ) {
// 	if ( ! recentOverallCountsPerLS_.empty() ) {
// 	  std::cout << "Overall\t-> good one: new => cnt, ls = " 
// 		    << current_count << ", " << nL
// 		    << ", old = "
// 		    << recentOverallCountsPerLS_.front().second << "\t"
// 		    << recentOverallCountsPerLS_.front().first 
// 		    << std::endl;
// 	  // END DEBUG
// 	}
//       }
      recentOverallCountsPerLS_.push_front(CountLS_t(nL,current_count));
      while ( recentOverallCountsPerLS_.size() > kRateIntegWindow_ ) {
	recentOverallCountsPerLS_.pop_back();
      }
      std::pair<double,double> sl =  getSlope_(recentOverallCountsPerLS_);
      double slope = sl.first; double slope_err = sl.second;
      if ( slope > -1 ) {
	hltRate_->setBinContent(nL,slope);
	if ( ! std::isnan(slope_err ) && (slope_err >= 0 )  )
	  hltRate_->setBinError(nL,slope_err);
      }

    } // good data
  } // found  histo
  //

    

}

// unused
void HLTScalersClient::analyze(const edm::Event& e, const edm::EventSetup& c ) 
{
  // nothing to do here
}

// this is probably overkill. Do a least-squares fit to get slope
// note that the data is in units of counts, ls number
// but we return a value in Hz...
std::pair<double,double>
HLTScalersClient::getSlope_(HLTScalersClient::CountLSFifo_t points)
{

  double slope, sigma_m;
  // just do a delta if we just want two bins
  if ( points.size() == 2 ) {
    // just do diff and be done with it 
    double delta_ls = points.front().first - points.back().first;
    double delta_cnt = points.front().second - points.back().second;
    slope = delta_cnt / delta_ls ;
    sigma_m = -1;
  }
  else { // do a fit
    double xy = 0;
    double x = 0;
    double xsq = 0;
    double y = 0;
    double n = double(points.size());
    for ( CountLSFifo_t::iterator i(points.begin());
	  i != points.end(); ++i ) {
//       if ( debug_ ) 
// 	std::cout << "x = " << i->first << ", y = " << i->second 
// 		  << std::endl;
      xy += i->first * i->second;
      x += i->first;
      xsq += i->first*i->first;
      y += i->second;
    }
    slope = (n*xy - x*y)/(n*xsq - x*x);

    // now get the uncertainty on the slope. Need intercept for this.
    double intercept = (xsq*y - xy*x)/(n*xsq-x*x);
    double sigma_ysq = 0;
    for ( CountLSFifo_t::iterator i(points.begin());
	  i != points.end(); ++i ) {
      sigma_ysq += pow(( i->second - slope * i->first  - intercept),2.);
    }
//     if ( debug_ ) 
//       std::cout << "chi^2 = " << sigma_ysq << std::endl;
    sigma_ysq *= 1./(n-2.);

    sigma_m = sqrt( n*sigma_ysq/(n*xsq - x*x));
  }

  //  std::cout << "Slope is " << slope << std::endl;
  slope /= SECS_PER_LUMI_SECTION;
  sigma_m /= SECS_PER_LUMI_SECTION;
  if ( debug_ ) {
    std::cout << "Slope = " << slope << " +- " << sigma_m 
	      << std::endl;
//   std::cout << "intercept is " << intercept
//  	    << std::endl;
  }


  return std::pair<double,double>(slope, sigma_m);
}
