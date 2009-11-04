// $Id: HLTScalersClient.cc,v 1.9 2009/11/04 03:44:54 lorenzo Exp $
// 
// $Log: HLTScalersClient.cc,v $
// Revision 1.9  2009/11/04 03:44:54  lorenzo
// added folder param
//
// Revision 1.8  2008/09/07 12:16:23  wittich
// protect against divide-by-zero in rate calculation
//
// Revision 1.7  2008/09/04 11:06:02  lorenzo
// changed to _EvF folder
//
// Revision 1.6  2008/09/03 02:13:48  wittich
// - bug fix in L1Scalers
// - configurable dqm directory in L1SCalers
// - other minor tweaks in HLTScalers
//
// Revision 1.5  2008/08/28 22:22:13  wittich
// - make delta_t absolute value
// - add some more LogDebug statements
// tested in full chain on .cms network
//
// Revision 1.4  2008/08/26 01:38:55  wittich
// re-add Don's 20 entry histograms with full bin labels
//
// Revision 1.3  2008/08/25 00:38:27  wittich
// Remove defunct couts
//
// Revision 1.2  2008/08/24 16:34:56  wittich
// - rate calculation cleanups
// - fix error logging with LogDebug
// - report the actual lumi segment number that we think it might be
//
// Revision 1.1  2008/08/22 20:56:57  wittich
// - add client for HLT Scalers
// - Move rate calculation to HLTScalersClient and slim down the
//   filter-farm part of HLTScalers
//

#include <cassert>

#include "DQM/TrigXMonitorClient/interface/HLTScalersClient.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"


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
  first_(true),
  folderName_(ps.getUntrackedParameter<std::string>("dqmFolder", "HLT/HLScalers_EvF"))
{
  LogDebug("Status") << "constructor" ;
  // get back-end interface
  dbe_ = edm::Service<DQMStore>().operator->();
  assert(dbe_ != 0); // blammo!
  dbe_->setCurrentFolder(folderName_);

  currentRate_ = dbe_->book1D("cur_rate", 
			      "current lumi section rate per path",
			      MAX_PATHS, -0.5, MAX_PATHS-0.5);

  for (int i = 0; i < MAX_PATHS; ++i ) {
    scalerCounters_[i] = 0UL;
    rateHistories_[i] = 0; // not really needed but ...
    char name[256]; snprintf(name, 256, "rate_p%03d", i);
    LogDebug("Parameter") << "name " << i << " is " << name ;
    rateHistories_[i] = dbe_->book1D(name, name, MAX_LUMI_SEG, 
				     -0.5, MAX_LUMI_SEG-0.5);
  }

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
}


/// BeginJob
void HLTScalersClient::beginJob(const edm::EventSetup& c)
{
  LogDebug("Status") << "beingJob" ;
  if (dbe_) {
    dbe_->setCurrentFolder(folderName_);
  }
}


/// BeginRun
void HLTScalersClient::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
}

/// EndRun
void HLTScalersClient::endRun(const edm::Run& run, const edm::EventSetup& c)
{
}


/// End LumiBlock
/// DQM Client Diagnostic should be performed here
void HLTScalersClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
			const edm::EventSetup& c)
{
  nLumi_ = lumiSeg.id().luminosityBlock();
  std::string scalHisto_ = folderName_ + "/hltScalers";
  MonitorElement *scalers = dbe_->get(scalHisto_);
  if ( scalers == 0 ) {
    LogInfo("Status") << "cannot get hlt scalers histogram, bailing out.";
    return;
  }


  int npaths = scalers->getNbinsX();
  if ( npaths > MAX_PATHS ) npaths = MAX_PATHS; // HARD CODE FOR NOW
  LogDebug("Status") << "I see " << npaths << " paths. ";

  // set the bin labels on the first go-through
  if ( first_) {
    for ( int i = 0; i < npaths; ++i ) {
      int whichHisto = i/kPerHisto;
      int whichBin = i%kPerHisto + 1;
      char pname[256];
      snprintf(pname, 256, "%s/path%03d",folderName_.c_str(), i);
      MonitorElement *name = dbe_->get(pname);
      std::string sname;
      if ( name ) {
	sname = std::string (name->getStringValue());
      }
      else {
	sname = std::string("unknown");
      }
      hltCurrentRate_[whichHisto]->setBinLabel(whichBin, sname.c_str());
      snprintf(pname, 256, "Rate - path %s (Path # %03d)", sname.c_str(), i);
      rateHistories_[i]->setTitle(pname);
    }
    first_ = false;
  }

  std::string nLumiHisto_ = folderName_ + "/nLumiBlock";
  MonitorElement *nLumi = dbe_->get(nLumiHisto_);
  int testval = (nLumi!=0?nLumi->getIntValue():-1);
  LogDebug("Parameter") << "Lumi Block from DQM: "
			<< testval
			<< ", local is " << nLumi_;
  int nL = (nLumi!=0?nLumi->getIntValue():nLumi_);
  if ( nL > MAX_LUMI_SEG ) {
    LogDebug("Status") << "Too many Lumi segments, "
		      << nL << " is greater than MAX_LUMI_SEG,"
		      << " wrapping to " 
		      << (nL%MAX_LUMI_SEG);
    //nL = MAX_LUMI_SEG;
    nL = nL%MAX_LUMI_SEG;
  }
  float delta_t = (nL - currentLumiBlockNumber_)*SECS_PER_LUMI_SECTION;
  if ( delta_t < 0 ) {
    LogDebug("Status") << " time is negative ... " << delta_t;
    delta_t = -delta_t;
  }
  else if ( nL == currentLumiBlockNumber_ ) { // divide-by-zero
    LogInfo("Status") << "divide by zero: same lumi section 2x " << nL;
    return;
  }
  // fill in the rates
  for ( int i = 1; i <= npaths; ++i ) { // bins start at 1
    float current_count = scalers->getBinContent(i);
    float rate = (current_count-scalerCounters_[i-1])/delta_t;
    if ( rate > 1E-3 ) {
      LogDebug("Parameter") << "rate path " << i << " is " << rate;
    }
    currentRate_->setBinContent(i, rate);
    hltCurrentRate_[i/kPerHisto]->setBinContent(i%kPerHisto+1, rate);
    //currentRate_->setBinError(i, error);
    scalerCounters_[i-1] = ulong(current_count);
    rateHistories_[i-1]->setBinContent(nL, rate);
  }
  currentLumiBlockNumber_ = nL;

}

// unused
void HLTScalersClient::analyze(const edm::Event& e, const edm::EventSetup& c) 
{
}
