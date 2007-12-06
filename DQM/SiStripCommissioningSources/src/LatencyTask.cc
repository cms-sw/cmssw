#include "DQM/SiStripCommissioningSources/interface/LatencyTask.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// -----------------------------------------------------------------------------
//
std::map<std::string, CommissioningTask::HistoSet> LatencyTask::timingMap_;

// -----------------------------------------------------------------------------
//
LatencyTask::LatencyTask( DaqMonitorBEInterface* dqm,
			      const FedChannelConnection& conn ) :
  CommissioningTask( dqm, conn, "LatencyTask" ),
  dummy_(),
  timing_(dummy_)
{
  LogDebug("Commissioning") << "[LatencyTask::LatencyTask] Constructing object...";
}

// -----------------------------------------------------------------------------
//
LatencyTask::~LatencyTask() {
  LogDebug("Commissioning") << "[LatencyTask::LatencyTask] Destructing object...";
}

// -----------------------------------------------------------------------------
//
void LatencyTask::book() {
  LogDebug("Commissioning") << "[LatencyTask::book]";

  // construct the histo title
  // by setting the granularity to sistrip::TRACKER, the title will be identical for all detkeys.
  // therefore, only one histo will be booked/analyzed
  std::string title = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
					 sistrip::APV_LATENCY, 
  					 sistrip::DET_KEY, 
					 connection().detId(),
					 sistrip::TRACKER, 
					 0 ).title(); 
  // look if such an histogram is already booked
  if(timingMap_.find(title)!=timingMap_.end()) {
    // if already booked, use it
    timing_ = timingMap_[title];
  } else {
    // if not, book it
    timingMap_[title] = HistoSet();
    timing_ = timingMap_[title];
    int nBins = 100;
    timing_.histo_ = dqm()->bookProfile( title, title,    // name and title
  				         nBins, -2500, 0,   // binning + range
				         100, 0., -1. );  // Y range : automatic
  
    timing_.vNumOfEntries_.resize(nBins,0);
    timing_.vSumOfContents_.resize(nBins,0);
    timing_.vSumOfSquares_.resize(nBins,0);
  }
  LogDebug("Commissioning") << "[LatencyTask::book] done";
  
}

// -----------------------------------------------------------------------------
//
void LatencyTask::fill( const SiStripEventSummary& summary,
			  const edm::DetSet<SiStripRawDigi>& digis ) {
  LogDebug("Commissioning") << "[LatencyTask::fill]";
  // retrieve the delay from the EventSummary
  float delay = const_cast<SiStripEventSummary&>(summary).latency()*(-25);
  float correctedDelay = delay;
  LogDebug("Commissioning") << "[LatencyTask::fill]; the delay is " << delay;
  // loop on the strips to find the (maybe) non-zero digi
  for(unsigned int strip=0;strip<digis.data.size();strip++) {
    if(digis.data[strip].adc()!=0) {
      // apply the TOF correction
      correctedDelay = delay - (digis.data[strip].adc()>>8)/10.;
      if((digis.data[strip].adc()>>8)==255) continue; // skip hit if TOF is in overflow
      LogDebug("Commissioning") << "[LatencyTask::fill]; using a hit with value " << ( digis.data[strip].adc()&0xff )
                                << " at corrected delay of " << correctedDelay;
      updateHistoSet( timing_,int(correctedDelay),digis.data[strip].adc()&0xff);
      break;
    }
  }
}

// -----------------------------------------------------------------------------
//
void LatencyTask::update() {
  LogDebug("Commissioning") << "[LatencyTask::update]";
  updateHistoSet( timing_ );
}

