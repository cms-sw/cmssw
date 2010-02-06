#include <cassert>
#include <sstream>

#include "DQM/TrigXMonitorClient/interface/L1ScalersClient.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"


#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


using edm::LogInfo;
using edm::LogWarning;

#define SECS_PER_LUMI_SECTION 93.3
const int kPerHisto = 20;
const int kNumAlgoHistos = MAX_ALGOS/kPerHisto; // this hasta be w/o remainders
const int kNumTTHistos = MAX_TT/kPerHisto; // this hasta be w/o remainders


/// Constructors
L1ScalersClient::L1ScalersClient(const edm::ParameterSet& ps):
  dbe_(0),
  nLumi_(0),
  l1AlgoCurrentRate_(0),
  l1TechTrigCurrentRate_(0),
  selected_(0),
  bxSelected_(0),
  algoSelected_(ps.getUntrackedParameter<std::vector<int> >("algoMonitorBits")),
  techSelected_(ps.getUntrackedParameter<std::vector<int> >("techMonitorBits")),
  folderName_(ps.getUntrackedParameter<std::string>("dqmFolder", "L1T/L1Scalers_EvF")),
  currentLumiBlockNumber_(0),
  first_algo(true),
  first_tt(true)
{
  LogDebug("Status") << "constructor" ;
  // get back-end interface
  dbe_ = edm::Service<DQMStore>().operator->();
  assert(dbe_ != 0); // blammo!
  dbe_->setCurrentFolder(folderName_);

  l1AlgoCurrentRate_ = dbe_->book1D("algo_cur_rate", 
			      "current lumi section rate per Algo Bits",
			      MAX_ALGOS, -0.5, MAX_ALGOS-0.5);

  l1TechTrigCurrentRate_ = dbe_->book1D("tt_cur_rate", 
			      "current lumi section rate per Tech. Trig.s",
			      MAX_TT, -0.5, MAX_TT-0.5);
  // ----------------------
  numSelected_ = algoSelected_.size() + techSelected_.size();
  selected_ = dbe_->book1D("l1BitsSel", "Selected L1 Algorithm"
			   " and tech Bits", numSelected_,
			   -0.5, numSelected_-0.5);
  bxSelected_ = dbe_->book2D("l1BitsBxSel", 
			     "Selected L1 Algorithm Bits vs Bx", 
			     3600, -0.5, 3599.5,
			     numSelected_, -0.5, numSelected_-0.5);
  int j = 1;
  for ( unsigned int i = 0; i < algoSelected_.size(); ++i ) {
    char title[256];
    snprintf(title, 256, "Algo %d", algoSelected_[i]);
    selected_->setBinLabel(j, title);
    bxSelected_->setBinLabel(j, title, 2);
    ++j;
  }
  for ( unsigned int i = 0; i < techSelected_.size(); ++i ) {
    char title[256];
    snprintf(title, 256, "Tech %d", techSelected_[i]);
    selected_->setBinLabel(j, title);
    bxSelected_->setBinLabel(j, title, 2);
    ++j;
  }


// book individual bit rates vs lumi for algo bits.
  totalAlgoRate_ = dbe_->book1D("totAlgoRate", "Total Algo Rate" , MAX_LUMI_SEG, 
				     -0.5, MAX_LUMI_SEG-0.5);
  totalTtRate_ = dbe_->book1D("totTtRate", "Total Tech. Trig Rate" , MAX_LUMI_SEG, 
				     -0.5, MAX_LUMI_SEG-0.5);

  totAlgoPrevCount=0UL;
  totTtPrevCount=0UL;
  for (int i = 0; i < MAX_ALGOS; ++i ) {
    l1AlgoScalerCounters_[i] = 0UL;
    l1AlgoRateHistories_[i] = 0; // not really needed but ...
    char name[256]; snprintf(name, 256, "rate_algobit%03d", i);
    LogDebug("Parameter") << "name " << i << " is " << name ;
    l1AlgoRateHistories_[i] = dbe_->book1D(name, name, MAX_LUMI_SEG, 
				     -0.5, MAX_LUMI_SEG-0.5);
  }

// book individual bit rates vs lumi for technical trigger bits.
  for (int i = 0; i < MAX_TT; ++i ) {
    l1TechTrigScalerCounters_[i] = 0UL;
    l1TechTrigRateHistories_[i] = 0; // not really needed but ...
    char name[256]; snprintf(name, 256, "rate_ttbit%03d", i);
    LogDebug("Parameter") << "name " << i << " is " << name ;
    l1TechTrigRateHistories_[i] = dbe_->book1D(name, name, MAX_LUMI_SEG, 
				     -0.5, MAX_LUMI_SEG-0.5);
  }

  // split l1 scalers up into groups of 20, assuming total of 140 bits
  char metitle1[40]; //histo name
  char mename1[40]; //ME name
  for( int k = 0; k < kNumAlgoHistos; k++ ) {
    int npath_low = kPerHisto*k;
    int npath_high = kPerHisto*(k+1)-1;
    snprintf(mename1, 40, "L1AlgoBits_%0d", k);
    snprintf(metitle1, 40, "L1 rates - Algo Bits %d to %d", npath_low, 
	     npath_high);
    l1AlgoCurrentRatePerAlgo_[k]= dbe_->book1D(mename1, metitle1, kPerHisto, 
				     -0.5 + npath_low, npath_high+0.5);
  }

  // split l1 scalers up into groups of 20, assuming total of 80 technical bits
  char metitle2[40]; //histo name
  char mename2[40]; //ME name
  for( int k = 0; k < kNumTTHistos; k++ ) {
    int npath_low = kPerHisto*k;
    int npath_high = kPerHisto*(k+1)-1;
    snprintf(mename2, 40, "L1TechBits_%0d", k);
    snprintf(metitle2, 40, "L1 rates - Tech. Trig. Bits %d to %d", npath_low, 
	     npath_high);
    l1TechTrigCurrentRatePerAlgo_[k]= dbe_->book1D(mename2, metitle2, kPerHisto, 
				     -0.5 + npath_low, npath_high+0.5);
  }

  std::ostringstream params;
  params << "Algo: ";
  for ( unsigned int i = 0; i < algoSelected_.size(); ++i ) {
    params << algoSelected_[i] << " ";
  }
  params << ", Tech: ";
  for ( unsigned int i = 0; i < techSelected_.size(); ++i ) {
    params << techSelected_[i] << " ";
  }
  LogDebug("Parameter") << "L1 bits to monitor are " << params.str();
}


/// BeginJob
void L1ScalersClient::beginJob(const edm::EventSetup& c)
{
  LogDebug("Status") << "beingJob" ;
  if (dbe_) {
    dbe_->setCurrentFolder(folderName_);
  }
}


/// BeginRun
void L1ScalersClient::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
}

/// EndRun
void L1ScalersClient::endRun(const edm::Run& run, const edm::EventSetup& c)
{
}


/// End LumiBlock
/// DQM Client Diagnostic should be performed here
void L1ScalersClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
			const edm::EventSetup& c)
{
  nLumi_ = lumiSeg.id().luminosityBlock();

  // get EvF data

  MonitorElement *algoScalers = dbe_->get(folderName_+std::string("/l1AlgoBits"));
  MonitorElement *ttScalers = dbe_->get(folderName_+std::string("/l1TechAlgoBits"));
  
  if ( algoScalers == 0 || ttScalers ==0) {
    LogInfo("Status") << "cannot get l1 scalers histogram, bailing out.";
    return;
  }


  int nalgobits = algoScalers->getNbinsX();
  int nttbits = ttScalers->getNbinsX();

  if ( nalgobits > MAX_ALGOS ) nalgobits = MAX_ALGOS; // HARD CODE FOR NOW
  if ( nttbits > MAX_TT ) nttbits = MAX_TT; // HARD CODE FOR NOW

  LogDebug("Status") << "I see " << nalgobits << " algo paths. ";
  LogDebug("Status") << "I see " << nttbits << " tt paths. ";

  // set the bin labels on the first go-through
  if ( first_algo) {
    for ( int i = 0; i < nalgobits; ++i ) {
      int whichHisto = i/kPerHisto;
      int whichBin = i%kPerHisto + 1;
      char pname[256];
      snprintf(pname, 256, "AlgoBit%03d", i);
      l1AlgoCurrentRatePerAlgo_[whichHisto]->setBinLabel(whichBin, pname);
      snprintf(pname, 256, "Rate - Algorithm Bit %03d", i);
      l1AlgoRateHistories_[i]->setTitle(pname);
    }
    first_algo = false;
  }

  // set the bin labels on the first go-through
  if ( first_tt) {
    for ( int i = 0; i < nttbits; ++i ) {
      int whichHisto = i/kPerHisto;
      int whichBin = i%kPerHisto + 1;
      char pname[256];
      snprintf(pname, 256, "TechBit%03d", i);
      l1TechTrigCurrentRatePerAlgo_[whichHisto]->setBinLabel(whichBin, pname);
      snprintf(pname, 256, "Rate - Technical Bit %03d", i);
      l1TechTrigRateHistories_[i]->setTitle(pname);
    }
    first_tt = false;
  }

  MonitorElement *nLumi = dbe_->get(folderName_+std::string("nLumiBlock"));
  
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
  // selected ---------------------  fill in the rates for th 
  int currSlot = 1; // for selected bits histogram
  MonitorElement *algoBx = dbe_->get(folderName_+
				     std::string("/l1AlgoBits_Vs_Bx"));
  // selected ---------------------  end
  for ( int i = 1; i <= nalgobits; ++i ) { // bins start at 1
    float current_count = algoScalers->getBinContent(i);
    // selected -------------------- start
    int bit = i -1; //
    if ( std::find(algoSelected_.begin(), algoSelected_.end(),bit)
	 != algoSelected_.end() ) {
      selected_->setBinContent(currSlot, current_count);
      if ( algoBx ) {
	for ( int j = 1; j <= 3600; ++j ) {
	  bxSelected_->setBinContent(j, currSlot, 
				     algoBx->getBinContent(j, i));
	}
      }
      ++currSlot;
    }
    // selected -------------------- end
    float rate = (current_count-l1AlgoScalerCounters_[i-1])/delta_t;
    if ( rate > 1E-3 ) {
      LogDebug("Parameter") << "rate path " << i << " is " << rate;
    }
    l1AlgoCurrentRate_->setBinContent(i, rate);
    l1AlgoCurrentRatePerAlgo_[i/kPerHisto]->setBinContent(i%kPerHisto+1, rate);
    //currentRate_->setBinError(i, error);
    l1AlgoScalerCounters_[i-1] = ulong(current_count);
    l1AlgoRateHistories_[i-1]->setBinContent(nL, rate);
  }
  // selected ----------------- start
  MonitorElement *techBx = dbe_->get(folderName_+std::string("/l1TechAlgoBits_Vs_Bx"));
  // selected ----------------- end

  for ( int i = 1; i <= nttbits; ++i ) { // bins start at 1
    float current_count = ttScalers->getBinContent(i);
    // selected -------------------- start
    int bit = i -1; //
    if ( std::find(techSelected_.begin(), techSelected_.end(),bit)
	 != techSelected_.end() ) {
      selected_->setBinContent(currSlot, current_count);
      if ( techBx ) {
	for ( int j = 1; j <= 3600; ++j ) {
	  bxSelected_->setBinContent(j, currSlot, 
				     techBx->getBinContent(j, i));
	}
      }
      ++currSlot;
    }
    // selected -------------------- end
    float rate = (current_count-l1TechTrigScalerCounters_[i-1])/delta_t;
    if ( rate > 1E-3 ) {
      LogDebug("Parameter") << "rate path " << i << " is " << rate;
    }
    l1TechTrigCurrentRate_->setBinContent(i, rate);
    l1TechTrigCurrentRatePerAlgo_[i/kPerHisto]->setBinContent(i%kPerHisto+1, rate);
    //currentRate_->setBinError(i, error);
    l1TechTrigScalerCounters_[i-1] = ulong(current_count);
    l1TechTrigRateHistories_[i-1]->setBinContent(nL, rate);
  }
  
  //  compute total rate
  MonitorElement *l1AlgoCounter = dbe_->get(folderName_+std::string("/l1AlgoCounter"));
  MonitorElement *l1TtCounter = dbe_->get(folderName_+std::string("/l1TtCounter"));
  if ( l1AlgoCounter != 0 && l1TtCounter != 0 ) {
    float totAlgoCount = l1AlgoCounter->getIntValue();
    float totTtCount = l1TtCounter->getIntValue();
    float totAlgRate = (totAlgoCount - totAlgoPrevCount)/delta_t;
    float totTtRate = (totTtCount - totTtPrevCount)/delta_t;
    totalAlgoRate_->setBinContent(nL, totAlgRate);
    totAlgoPrevCount = totAlgoCount;
    totalTtRate_->setBinContent(nL, totTtRate);
    totTtPrevCount = totTtCount;
  }

  
  currentLumiBlockNumber_ = nL;

}

// unused
void L1ScalersClient::analyze(const edm::Event& e, const edm::EventSetup& c) 
{
}
