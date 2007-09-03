#include "DQM/SiStripCommissioningSources/interface/FineDelayTask.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// -----------------------------------------------------------------------------
//
FineDelayTask::FineDelayTask( DaqMonitorBEInterface* dqm,
			      const FedChannelConnection& conn ) :
  CommissioningTask( dqm, conn, "FineDelayTask" ),
  timing_(),
  nBins_(100) //TODO: tune this to the expected PLL step/range
{
  LogDebug("Commissioning") << "[FineDelayTask::FineDelayTask] Constructing object...";
  // compute the fiber length correction
  float length=conn.fiberLength();
  // TODO: check the units... this is subject to change.
  // convert cm to ns
  float c=30; //speed of light in cm/ns
  float refractionIndex = 1.4; // refraction index of the optical fibers
  fiberLengthCorrection_ =  length/c*refractionIndex;
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

  std::string title = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
					 sistrip::FINE_DELAY, 
  					 sistrip::DET_KEY, 
					 connection().detId(),
					 sistrip::MODULE, 
					 0 ).title(); 
  timing_.histo_ = dqm()->bookProfile( title, title,    // name and title
				       nBins_, -25, 25, // binning + range
				       100, 0., -1. );  // Y range : automatic
  
  timing_.vNumOfEntries_.resize(nBins_,0);
  timing_.vSumOfContents_.resize(nBins_,0);
  timing_.vSumOfSquares_.resize(nBins_,0);
  LogDebug("Commissioning") << "[FineDelayTask::book] done";
  
}

// -----------------------------------------------------------------------------
//
void FineDelayTask::fill( const SiStripEventSummary& summary,
			  const edm::DetSet<SiStripRawDigi>& digis ) {
  LogDebug("Commissioning") << "[FineDelayTask::fill]";
  // retrieve the delay from the EventSummary
  float delay = const_cast<SiStripEventSummary&>(summary).pllCoarse()*25+const_cast<SiStripEventSummary&>(summary).pllFine();
  float correctedDelay = delay;
  LogDebug("Commissioning") << "[FineDelayTask::fill]; the delay is " << delay;
  // loop on the strips to find the (maybe) non-zero digi
  for(unsigned int strip=0;strip<digis.data.size();strip++) {
    if(digis.data[strip].adc()!=0) {
      // apply the TOF correction
      correctedDelay = delay - (digis.data[strip].adc()>>8)/10.;
      if((digis.data[strip].adc()>>8)==255) continue; // skip hit if TOF is in overflow
      // apply the fiber length correction
      correctedDelay += fiberLengthCorrection_;
      LogDebug("Commissioning") << "[FineDelayTask::fill]; using a hit with value " << ( digis.data[strip].adc()&0xff )
                                << " at corrected delay of " << correctedDelay;
      updateHistoSet( timing_,int(correctedDelay),digis.data[strip].adc()&0xff);
      break;
    }
  }
}

// -----------------------------------------------------------------------------
//
void FineDelayTask::update() {
  LogDebug("Commissioning") << "[FineDelayTask::update]";
  updateHistoSet( timing_ );
}

