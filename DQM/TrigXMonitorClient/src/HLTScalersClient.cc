// $Id: HLTScalersClient.cc,v 1.20 2012/12/06 09:49:36 eulisse Exp $
// 
// $Log: HLTScalersClient.cc,v $
// Revision 1.20  2012/12/06 09:49:36  eulisse
// Use `edm::isNotFinite` in place of `std::isnan`.
//
// * Required when running `-Ofast` builds.
//
// Revision 1.19  2010/07/20 02:58:27  wmtan
// Add missing #include files
//
// Revision 1.18  2010/04/02 20:48:12  wittich
// updates to scale entries by received number of FU's
//
// Revision 1.17  2010/03/17 20:56:19  wittich
// Check for good updates based on mergeCount values
// add code for rates normalized per FU
//
// Revision 1.16  2010/03/16 22:19:19  wittich
// updates for per-LS normalization for variable
// number of FU's sending information back to the clients.
//
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

///// THIS ONLY WILL WORK WITH THE DATA FROM THE FU'S

#include <cassert>
#include <numeric>

#include "DQM/TrigXMonitorClient/interface/HLTScalersClient.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/isFinite.h"


#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


using edm::LogInfo;
using edm::LogWarning;

// I am not sure this is right at more than 10%
#define SECS_PER_LUMI_SECTION 23.31
const int kPerHisto = 20;




/// Constructors
HLTScalersClient::HLTScalersClient(const edm::ParameterSet& ps):
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
  debug_(ps.getUntrackedParameter<bool>("debugDump", false)),
  maxFU_(ps.getUntrackedParameter<unsigned int>("maxFU", false)),
  recentOverallCountsPerLS_(kRateIntegWindow_),
  recentNormedOverallCountsPerLS_(2)
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


  std::string rawdir(folderName_ + "/raw");
  dbe_->setCurrentFolder(rawdir);
  hltRate_ = dbe_->book1D("hltRate", "Overall HLT Accept rate vs LS", 
			  MAX_LUMI_SEG_HLT, -0.5, MAX_LUMI_SEG_HLT-0.5);
  dbe_->setCurrentFolder(folderName_);
  hltNormRate_ = dbe_->book1D("hltRateNorm", 
			      "Overall HLT Accept rate vs LS, scaled", 
			      MAX_LUMI_SEG_HLT, -0.5, MAX_LUMI_SEG_HLT-0.5);
  hltCount_= dbe_->book1D("hltCount", "Overall HLT Counts vs LS", 
			  MAX_LUMI_SEG_HLT, -0.5, MAX_LUMI_SEG_HLT-0.5);
//   hltCountN_= dbe_->book1D("hltCountN", "Overall HLT Counts per LS vs LS", 
// 			  MAX_LUMI_SEG_HLT, -0.5, MAX_LUMI_SEG_HLT-0.5);
  mergeCount_= dbe_->book1D("mergeCount", "Number of merge counts vs LS", 
			    MAX_LUMI_SEG_HLT, -0.5, MAX_LUMI_SEG_HLT-0.5);



  updates_ = dbe_->book1D("updates", "Status of Updates", 2, 0, 2);
  updates_->setBinLabel(1, "Good Updates");
  updates_->setBinLabel(2, "Incomplete Updates");


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
  // PWDEBUG
  if ( first_ && debug_)
    dbe_->showDirStructure();
  // PWDEBUG END

  // get raw data
  std::string scalHisto = folderName_ + "/raw/hltScalers";
  MonitorElement *scalers = dbe_->get(scalHisto);
  if ( scalers == 0 ) {
    LogDebug("HLTScalersClient") << "cannot get hlt scalers histogram, "
				 << "bailing out.";
    if ( debug_ )
      std::cout << "No scalers ? Looking for " 
		<< scalHisto
		<< std::endl;
    return;
  }


  int npaths = scalers->getNbinsX();
  if ( npaths > MAX_PATHS ) npaths = MAX_PATHS; // HARD CODE FOR NOW
  LogDebug("HLTScalersClient") << "I see " << npaths << " paths. ";

  // set the bin labels on the first go-through
  // I need to do this here because we don't have the paths yet
  // on begin-run. I should do this in a less ugly way (see FV?)
  if ( first_) {
    std::string rawdir(folderName_ + "/raw");

    LogDebug("HLTScalersClient") << "Setting up paths on first endLumiBlock "
				 << npaths;
    dbe_->setCurrentFolder(rawdir);
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

    dbe_->setCurrentFolder(folderName_); // these belong in top-level
    for (int i = 0; i < npaths; ++i ) {
      dbe_->setCurrentFolder(std::string(rates_subfolder)+"/raw");

      char name[256]; snprintf(name, 256, "raw_rate_p%03d", i);
      //     LogDebug("HLTScalersClient") << "name " << i << " is " << name ;
      rateHistories_.push_back(dbe_->book1D(name, name, MAX_LUMI_SEG_HLT, 
					    -0.5, MAX_LUMI_SEG_HLT-0.5));
      snprintf(name, 256, "norm_rate_p%03d", i);
      dbe_->setCurrentFolder(rates_subfolder);
      rateNormHistories_.push_back(dbe_->book1D(name, name, MAX_LUMI_SEG_HLT, 
						-0.5, MAX_LUMI_SEG_HLT-0.5));
      dbe_->setCurrentFolder(counts_subfolder);
      snprintf(name, 256, "counts_p%03d", i);
      countHistories_.push_back(dbe_->book1D(name, name, MAX_LUMI_SEG_HLT, 
					     -0.5, MAX_LUMI_SEG_HLT-0.5));
      // prefill the data structures
      recentPathCountsPerLS_.push_back(CountLSFifo_t(kRateIntegWindow_));
      recentNormedPathCountsPerLS_.push_back(CountLSFifo_t(2));
    }
    dbe_->setCurrentFolder(folderName_);


    // split hlt scalers up into groups of 20
    const int maxlen = 40;
    char metitle[maxlen]; //histo name
    char mename[maxlen]; //ME name
    int numHistos = int(npaths/kPerHisto); // this hasta be w/o remainders
    
    int remainder = npaths%kPerHisto; 
    if ( remainder ) numHistos += 1;

    for( int k = 0; k < numHistos; k++ ) {
      int npath_low = kPerHisto*k;
      int npath_high = kPerHisto*(k+1)-1;
      snprintf(mename, maxlen, "hltScalers_%0d", k);
      snprintf(metitle, maxlen, "HLT scalers - Paths %d to %d", npath_low, 
	       npath_high);
      dbe_->setCurrentFolder(rawdir);
      hltCurrentRate_.push_back(dbe_->book1D(mename, metitle, kPerHisto, 
					     -0.5 + npath_low, npath_high+0.5));
      dbe_->setCurrentFolder(folderName_); // these belong in top-level
      snprintf(mename, maxlen, "hltScalersNorm_%0d", k);
      snprintf(metitle, maxlen, 
	       "HLT Rate (scaled) - Paths %d to %d", npath_low, npath_high);
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
      if ( name && (strlen(name) > 0)) {
	if ( debug_ ) {
	  std::cout << "path " << i << " name is " << name << std::endl;
	}
	int whichHisto = i/kPerHisto;
	int whichBin = i%kPerHisto + 1;
	char pname[256];
	hltCurrentRate_[whichHisto]->setBinLabel(whichBin, name);
	hltCurrentNormRate_[whichHisto]->setBinLabel(whichBin, name);
	snprintf(pname, 256, "Rate - path %s (Path # %03d)", name, i);
	rateHistories_[i] ->setTitle(pname);
	rateNormHistories_[i]->setTitle(pname);
	snprintf(pname, 256, "Counts - path %s (Path # %03d)", name, i);
	countHistories_[i]->setTitle(pname);

	currentRate_->setBinLabel(i+1, name);
	currentNormRate_->setBinLabel(i+1, name);

	missingPathNames_ = false;
      }
    }

  }
  // MEGA-HACK
  if ( missingPathNames_) {
    // if that didn't work we load 'em from a text file. damn straight.
    int ipath = 1;
    std::ifstream names("names.dat");
    if ( ! names ) {
      if ( debug_ )  {
	std::ostringstream msg;
	msg << "open of " << "names.dat";
	perror(msg.str().c_str());
      }
    } 
    else { // open succeeded
      missingPathNames_ = false;
      std::string line;
      while ( ! names.eof() ) {
	getline(names, line);
	std::istringstream fnames(line);
	std::string label; int bin;
	if ( fnames.str().find("#") == 0 )  // skip comment lines
	  continue;
	if ( fnames >> bin >> label  ) {
          if ( debug_ ) {
	    std::cout << bin << "--" << label << "(" << ipath << ")"
	  	      << std::endl;
          }
	  currentRate_->setBinLabel(ipath, label);
	  currentNormRate_->setBinLabel(ipath, label);
	  countHistories_[ipath-1]->setTitle(label);
	  rateHistories_[ipath-1]->setTitle(label);
	  rateNormHistories_[ipath-1]->setTitle(label);
	  int whichHisto = (ipath-1)/kPerHisto;
	  int whichBin = (ipath-1)%kPerHisto +1;
	  hltCurrentRate_[whichHisto]->setBinLabel(whichBin, label);
	  hltCurrentNormRate_[whichHisto]->setBinLabel(whichBin, label);
	  ++ipath;
	  if ( ipath > npaths )
	    break;
	} //
      } // loop lines
    } // open ok
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

  // this gets filled for all times. Mainly for debugging.
  for ( int i = 1; i <= npaths; ++i ) { // bins start at 1
    double current_count = scalers->getBinContent(i);
    // nb that this will now _overwrite_ some settings
    countHistories_[i-1]->setBinContent(nL, current_count); // good or bad
  }

  std::string overallScalerName(folderName_ + "/raw/hltOverallScaler");
  MonitorElement *hltScaler = dbe_->get(overallScalerName);
  if ( hltScaler != 0 ) {
    double current_count = hltScaler->getBinContent(1);
    hltCount_->setBinContent(nL,current_count);
    recentOverallCountsPerLS_.update(CountLS_t(nL,current_count));
    std::pair<double,double> sl =  getSlope_(recentOverallCountsPerLS_);
    double slope = sl.first; double slope_err = sl.second;
    if ( slope > 0 ) {
      hltRate_->setBinContent(nL,slope);
      if ( ! edm::isNotFinite(slope_err ) && (slope_err >= 0 )  )
	hltRate_->setBinError(nL,slope_err);
    }
  } // found  histo



  if ( num_fu >= 0.95*maxFU_ ) {
    if ( num_fu > maxFU_ ) {
      maxFU_ = num_fu;
      if ( debug_ ) 
	std::cout << "maxFU is now " << maxFU_ << std::endl;
    }
    // good data
    for ( int i = 1; i <= npaths; ++i ) { // bins start at 1
      double current_count = scalers->getBinContent(i);
      // DEBUG
      if ( ! recentPathCountsPerLS_[i-1].empty() && debug_ ) 
	  std::cout << i << "\t-> good one: new => cnt, ls = " 
		    << current_count << ", " << nL
		    << ", old = "
		    << recentPathCountsPerLS_[i-1].back().second << "\t"
		    << recentPathCountsPerLS_[i-1].back().first 
		    << std::endl;
      // END DEBUG
      recentPathCountsPerLS_[i-1].update(CountLS_t(nL,current_count));

      // NB: we do not fill a new entry in the rate histo if we can't 
      // calculate it
      std::pair<double,double> sl =  getSlope_(recentPathCountsPerLS_[i-1]);
      double slope = sl.first; double slope_err = sl.second;
      //rateHistories_[i-1]->Fill(nL,slope);
      if ( slope > 0 ) {
	rateHistories_[i-1]->setBinContent(nL,slope);
	// set the current rate(s)
	hltCurrentRate_[(i-1)/kPerHisto]->setBinContent(i%kPerHisto, slope);
	currentRate_->setBinContent(i, slope);
	if ( ! edm::isNotFinite(slope_err ) && (slope_err >= 0 ) ) {
	  currentRate_->setBinError(i, slope_err);
	  hltCurrentRate_[(i-1)/kPerHisto]->setBinError(i%kPerHisto, slope_err);
	  rateHistories_[i-1]->setBinError(nL,slope_err);
	}
      }
      //// HACK - normalized path rates
      //// END HACK
      
    } // loop over paths

    // ---------------------------- overall rate, absolute counts
    std::string overallScalerName(folderName_ + "/raw/hltOverallScaler");
    MonitorElement *hltScaler = dbe_->get(overallScalerName);
    if ( hltScaler != 0 ) {
      double current_count = hltScaler->getBinContent(1);
      hltCount_->setBinContent(nL,current_count);
      recentOverallCountsPerLS_.update(CountLS_t(nL,current_count));
      std::pair<double,double> sl =  getSlope_(recentOverallCountsPerLS_);
      double slope = sl.first; double slope_err = sl.second;
      if ( slope >= 0 ) {
	hltRate_->setBinContent(nL,slope);
	if ( ! edm::isNotFinite(slope_err ) && (slope_err >= 0 )  )
	  hltRate_->setBinError(nL,slope_err);
      }
    } // found  histo
    updates_->Fill(0); // good
  } // check on number of FU's - good data
  else {
    updates_->Fill(1); // missing updates
  }
  
  // PW DEBUG
  if ( debug_ ) {
    textfile_ << nL << "\t"
	      << npaths << "\t";
    for ( int i = 0; i < npaths ; ++i ) {
      textfile_ << scalers->getBinContent(i) << " ";
    }
    textfile_ << std::endl;
  }
  // end DEBUG


#ifdef LATER
  // ------ overall rate normalized - all data
  overallScalerName = std::string(folderName_ + "/raw/hltOverallScalerN");
  hltScaler = dbe_->get(overallScalerName);
  if ( hltScaler != 0 ) {
    double cnt = hltScaler->getBinContent(1);
//     hltCountN_->setBinContent(nL,cnt);
    if ( debug_ ) {
      std::cout << "Overall Norm: new => cnt, ls = " 
		<< cnt << ", " << nL
		<< ", num_fu = " << num_fu 
		<< std::endl;
    }
    recentNormedOverallCountsPerLS_.update(CountLS_t(nL, cnt/num_fu));
    cnt = recentNormedOverallCountsPerLS_.getCount(nL); // for dupes/partials
    double slope = cnt / num_fu / SECS_PER_LUMI_SECTION;
    if ( debug_ )  {
      std::cout << "Normalized slope = " << slope << std::endl;
    }
    if ( slope > 0 ) 
      hltNormRate_->setBinContent(nL,slope);
  }
  // 
  std::string scalHistoNorm = folderName_ + "/raw/hltScalersN";
  MonitorElement *scalersN = dbe_->get(scalHistoNorm);
  if ( scalersN ) {
    for (int i = 0; i < npaths ; ++i ) {
      double cnt = scalersN->getBinContent(i);
      double slope = cnt / num_fu / SECS_PER_LUMI_SECTION;
      if ( slope > 0 ) {
	rateNormHistories_[i-1]->setBinContent(nL,slope);
	// set the current rate(s)
	hltCurrentNormRate_[(i-1)/kPerHisto]->setBinContent(i%kPerHisto, slope);
	currentNormRate_->setBinContent(i, slope);
      }
    }
  }
#else // NOT LATER
  // ------ overall rate normalized - all data
  overallScalerName = std::string(folderName_ + "/raw/hltOverallScaler");
  hltScaler = dbe_->get(overallScalerName);
  if ( hltScaler != 0 ) {
    double cnt = hltScaler->getBinContent(1);
//     hltCountN_->setBinContent(nL,cnt);
    float sf = num_fu/maxFU_;
    if ( debug_ ) {
      std::cout << "Overall Norm: new => cnt, ls = " 
		<< cnt << ", " << nL
		<< ", num_fu = " << num_fu << ", sf = " << sf
		<< std::endl;
    }
    recentNormedOverallCountsPerLS_.update(CountLS_t(nL, cnt/sf));
    cnt = recentNormedOverallCountsPerLS_.getCount(nL); // for dupes/partials
    std::pair<double,double> sl =  getSlope_(recentNormedOverallCountsPerLS_);
    double slope = sl.first; double slope_err = sl.second;
    if ( debug_ )  {
      std::cout << "Normalized slope = " << slope << std::endl;
    }
    if ( slope > 0 ) {
      hltNormRate_->setBinContent(nL,slope);
      if ( cnt > 0 ) slope_err = slope*sqrt( 2./num_fu + 2./cnt);
      if ( ! edm::isNotFinite(slope_err ) && (slope_err >= 0 )  )
	hltNormRate_->setBinError(nL,slope_err);
    }
  }
  // 
  std::string scalHistoNorm = folderName_ + "/raw/hltScalers";
  MonitorElement *scalersN = dbe_->get(scalHistoNorm);
  if ( scalersN ) {
    double sf = num_fu /maxFU_;
    for (int i = 1; i <= npaths ; ++i ) {
      double cnt = scalersN->getBinContent(i);
      recentNormedPathCountsPerLS_[i-1].update(CountLS_t(nL,cnt/sf));
      std::pair<double,double> sl =  getSlope_(recentNormedPathCountsPerLS_[i-1]);
      double slope = sl.first; double slope_err = sl.second;
      if ( slope >= 0 ) {
	rateNormHistories_[i-1]->setBinContent(nL,slope);
	// set the current rate(s)
	hltCurrentNormRate_[(i-1)/kPerHisto]->setBinContent(i%kPerHisto, slope);
	currentNormRate_->setBinContent(i, slope);
	if ( slope_err <= 0 && cnt > 0) {
	  // ignores error on prev point, so scale by sqrt(2)
	  slope_err = slope*sqrt( 2./num_fu + 2./cnt);
          if ( debug_ ) {
            std::cout << "Slope err " << i << " = " << slope_err << std::endl;
          }
	}
	if ( ! edm::isNotFinite(slope_err ) && (slope_err >= 0 )  ) {
	  rateNormHistories_[i-1]->setBinError(nL,slope_err);
	  // set the current rate(s)
	  hltCurrentNormRate_[(i-1)/kPerHisto]->setBinError(i%kPerHisto, slope_err);
	  currentNormRate_->setBinError(i, slope_err);
	  
	}

      }
    }
  }

#endif // LATER
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
  if ( points.size() < points.targetSize() ) {
    return std::pair<double,double>(-1,-1);
  }
  // just do a delta if we just want two bins
  else if ( points.size() == 2 ) {
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
       if ( debug_ ) 
	 std::cout << "x = " << i->first << ", y = " << i->second 
		   << std::endl;
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
  if ( sigma_m >0 ) 
    sigma_m /= SECS_PER_LUMI_SECTION;
//   if ( debug_ ) {
//     std::cout << "Slope = " << slope << " +- " << sigma_m 
// 	      << std::endl;
//   std::cout << "intercept is " << intercept
//  	    << std::endl;
//  }


  return std::pair<double,double>(slope, sigma_m);
}
