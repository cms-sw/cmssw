#include "DQM/SiStripCommissioningSources/interface/FineDelayTask.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/SiStripCommon/interface/SiStripDetKey.h"

#define NBINS (500)
#define LOWBIN (-125)
#define HIGHBIN (125)


// -----------------------------------------------------------------------------
//
CommissioningTask::HistoSet FineDelayTask::timing_;
MonitorElement* FineDelayTask::mode_ = nullptr;

// -----------------------------------------------------------------------------
//
FineDelayTask::FineDelayTask( DQMStore* dqm,
			      const FedChannelConnection& conn ) :
  CommissioningTask( dqm, conn, "FineDelayTask" )
{
  LogDebug("Commissioning") << "[FineDelayTask::FineDelayTask] Constructing object...";
}

// -----------------------------------------------------------------------------
//
FineDelayTask::~FineDelayTask() {
  LogDebug("Commissioning") << "[FineDelayTask::FineDelayTask] Destructing object...";
}

// -----------------------------------------------------------------------------
//
void FineDelayTask::book() {
  LogDebug("Commissioning") << "[FineDelayTask::book]";

  std::string title;
  int nBins = NBINS;
  SiStripDetKey detkeytracker((uint32_t) 0);

  // see if the global timing histogram is already booked
  if (timing_.histo()) {
    // if already booked, use it
    LogDebug("Commissioning") << "[FineDelayTask::book] using existing histogram.";
  } else {
    // make a new histo on the tracker level if not existing yet
    LogDebug("Commissioning") << "[LatencyTask::book] booking a new histogram.";
    // construct the histo title
    title = SiStripHistoTitle( sistrip::EXPERT_HISTO,
                               sistrip::FINE_DELAY,
                               sistrip::DET_KEY,
                               detkeytracker.key(),
                               sistrip::TRACKER,
                               0,
                               sistrip::extrainfo::clusterCharge_ ).title();
    dqm()->setCurrentFolder( detkeytracker.path() );
    timing_.histo( dqm()->bookProfile( title, title,            // name and title
                                       nBins, LOWBIN, HIGHBIN,  // binning + range
                                       100, 0., -1., "s" ) );   // Y range : automatic
    timing_.vNumOfEntries_.resize(nBins,0);
    timing_.vSumOfContents_.resize(nBins,0);
    timing_.vSumOfSquares_.resize(nBins,0);
  }
  LogDebug("Commissioning") << "Binning is " << timing_.vNumOfEntries_.size();
  LogDebug("Commissioning") << "[FineDelayTask::book] done";
  if(!mode_) {
    std::string pwd = dqm()->pwd();
    std::string rootDir = pwd.substr(0,pwd.find(std::string(sistrip::root_) + "/")+(sizeof(sistrip::root_)));
    dqm()->setCurrentFolder( rootDir );
    mode_ = dqm()->bookInt("latencyCode");
  }

}

// -----------------------------------------------------------------------------
//
void FineDelayTask::fill( const SiStripEventSummary& summary,
			  const edm::DetSet<SiStripRawDigi>& digis ) {
  LogDebug("Commissioning") << "[FineDelayTask::fill]";
  // retrieve the delay from the EventSummary
  float delay = summary.ttcrx();
  uint32_t latencyCode = (summary.layerScanned()>>24)&0xff;
  LogDebug("Commissioning") << "[FineDelayTask::fill]: layerScanned() is " << summary.layerScanned();
  int latencyShift = latencyCode & 0x3f;             // number of bunch crossings between current value and start of scan... must be positive
  if(latencyShift>32) latencyShift -=64;             // allow negative values: we cover [-32,32].. should not be needed.
  if((latencyCode>>6)==2) latencyShift -= 3;         // layer in deconv, rest in peak
  if((latencyCode>>6)==1) latencyShift += 3;         // layer in peak, rest in deconv
  float correctedDelay = delay - (latencyShift*25.); // shifts the delay so that 0 corresponds to the current settings.

  LogDebug("Commissioning") << "[FineDelayTask::fill]; the delay is " << delay;
  // loop on the strips to find the (maybe) non-zero digi
  for(unsigned int strip=0;strip<digis.data.size();strip++) {
    if(digis.data[strip].adc()!=0) {
      // apply the TOF correction
      float tof = (digis.data[strip].adc()>>8)/10.;
      correctedDelay = delay - (latencyShift*25.) - tof;
      if((digis.data[strip].adc()>>8)==255) continue; // skip hit if TOF is in overflow
      // compute the bin
      float nbins = NBINS;
      float lowbin = LOWBIN;
      float highbin = HIGHBIN;
      int bin = int((correctedDelay-lowbin)/((highbin-lowbin)/nbins));
      LogDebug("Commissioning") << "[FineDelayTask::fill]; using a hit with value " << ( digis.data[strip].adc()&0xff )
                                << " at corrected delay of " << correctedDelay
				<< " in bin " << bin << "  (tof is " << tof << "( since adc = " << digis.data[strip].adc() << "))";
      updateHistoSet( timing_,bin,digis.data[strip].adc()&0xff);
      if(mode_) mode_->Fill(latencyCode);
    }
  }
}

// -----------------------------------------------------------------------------
//
void FineDelayTask::update() {
  LogDebug("Commissioning") << "[FineDelayTask::update]";
  updateHistoSet( timing_ );
}

