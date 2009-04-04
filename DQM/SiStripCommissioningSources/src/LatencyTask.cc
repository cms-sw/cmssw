#include "DQM/SiStripCommissioningSources/interface/LatencyTask.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include <DataFormats/SiStripDetId/interface/SiStripDetId.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#define NBINS (192)
#define LOWBIN (-4800)
#define HIGHBIN (0)

// -----------------------------------------------------------------------------
// static initialization
CommissioningTask::HistoSet LatencyTask::timing_;
CommissioningTask::HistoSet LatencyTask::cluster_;

// -----------------------------------------------------------------------------
//
LatencyTask::LatencyTask( DQMStore* dqm,
			      const FedChannelConnection& conn ) :
  CommissioningTask( dqm, conn, "LatencyTask" ),firstReading_(-1)
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

  std::string title, pwd, rootDir;
  int nBins = NBINS;

  // see if the global timing histogram is already booked
  if (timing_.histo()) {
    // if already booked, use it
    LogDebug("Commissioning") << "[LatencyTask::book] using existing histogram.";
  } else {
    // make a new histo on the tracker level if not existing yet
    LogDebug("Commissioning") << "[LatencyTask::book] booking a new histogram in " << dqm()->pwd();
    // construct the histo title
    title = SiStripHistoTitle( sistrip::EXPERT_HISTO,
                               sistrip::APV_LATENCY,
                               sistrip::DET_KEY,
                               0,
                               sistrip::TRACKER,
                               0,
                               sistrip::extrainfo::clusterCharge_ ).title();
    pwd = dqm()->pwd();
    rootDir = pwd.substr(0,pwd.find(std::string(sistrip::root_) + "/") + sizeof(sistrip::root_));
    dqm()->setCurrentFolder( rootDir );
    timing_.histo( dqm()->bookProfile( title, title,            // name and title
                                       nBins, LOWBIN, HIGHBIN,  // binning + range
                                       100, 0., -1.,"s" ) );    // Y range : automatic
    dqm()->setCurrentFolder( pwd );
    timing_.vNumOfEntries_.resize(nBins,0);
    timing_.vSumOfContents_.resize(nBins,0);
    timing_.vSumOfSquares_.resize(nBins,0);
  }
  // make a new histo on the partition level if not existing yet
  LogDebug("Commissioning") << "[LatencyTask::book] booking a new histogram in " << dqm()->pwd();
  // histo title
  title = SiStripHistoTitle( sistrip::EXPERT_HISTO,
                             sistrip::APV_LATENCY,
                             sistrip::DET_KEY,
                             connection().fecCrate(),
                             sistrip::PARTITION,
                             0,
                             sistrip::extrainfo::clusterCharge_ ).title();
  pwd = dqm()->pwd();
  rootDir = pwd.substr(0,pwd.find(std::string(sistrip::root_) + "/") + sizeof(sistrip::root_));
  rootDir += "/"; rootDir += sistrip::detectorView_;
  dqm()->setCurrentFolder( rootDir );
  timingPartition_.histo( dqm()->bookProfile( title, title,            // name and title
                                              nBins, LOWBIN, HIGHBIN,  // binning + range
                                              100, 0., -1.,"s" ) );    // Y range : automatic
  dqm()->setCurrentFolder( pwd );
  timingPartition_.vNumOfEntries_.resize(nBins,0);
  timingPartition_.vSumOfContents_.resize(nBins,0);
  timingPartition_.vSumOfSquares_.resize(nBins,0);

  // see if the global cluster histogram is already booked
  if (cluster_.histo()) {
    // if already booked, use it
    LogDebug("Commissioning") << "[LatencyTask::book] using existing histogram.";
  } else {
    // make a new histo on the tracker level if not existing yet
    LogDebug("Commissioning") << "[LatencyTask::book] booking a new histogram in " << dqm()->pwd();
    // construct the histo title
    title = SiStripHistoTitle( sistrip::EXPERT_HISTO,
                               sistrip::APV_LATENCY,
                               sistrip::DET_KEY,
                               0,
                               sistrip::TRACKER,
                               0,
                               sistrip::extrainfo::occupancy_).title();
    pwd = dqm()->pwd();
    rootDir = pwd.substr(0,pwd.find(std::string(sistrip::root_) + "/") + sizeof(sistrip::root_));
    dqm()->setCurrentFolder( rootDir );
    cluster_.histo( dqm()->book1D( title, title,               // name and title
                                   nBins, LOWBIN, HIGHBIN ));  // binning + range
    dqm()->setCurrentFolder( pwd );
    cluster_.isProfile_ = false;
    cluster_.vNumOfEntries_.resize(nBins,0);
    cluster_.vSumOfContents_.resize(nBins,0);
    cluster_.vSumOfSquares_.resize(nBins,0);
  }
  // make a new histo on the partition level if not existing yet
  LogDebug("Commissioning") << "[LatencyTask::book] booking a new histogram in " << dqm()->pwd();
  // histo title
  title = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
                             sistrip::APV_LATENCY, 
                             sistrip::DET_KEY, 
                             connection().fecCrate(),
                             sistrip::PARTITION,
                             0,
                             sistrip::extrainfo::occupancy_ ).title(); 
  pwd = dqm()->pwd();
  rootDir = pwd.substr(0,pwd.find(std::string(sistrip::root_) + "/") + sizeof(sistrip::root_));
  rootDir += "/"; rootDir += sistrip::detectorView_;
  dqm()->setCurrentFolder( rootDir );
  clusterPartition_.histo( dqm()->book1D( title, title,                // name and title
                                          nBins, LOWBIN, HIGHBIN ) );  // binning + range
  dqm()->setCurrentFolder( pwd );
  clusterPartition_.isProfile_ = false;
  clusterPartition_.vNumOfEntries_.resize(nBins,0);
  clusterPartition_.vSumOfContents_.resize(nBins,0);
  clusterPartition_.vSumOfSquares_.resize(nBins,0);

  LogDebug("Commissioning") << "[LatencyTask::book] done";
}

// -----------------------------------------------------------------------------
//
void LatencyTask::fill( const SiStripEventSummary& summary,
			const edm::DetSet<SiStripRawDigi>& digis ) {
  LogDebug("Commissioning") << "[LatencyTask::fill]";
  // retrieve the delay from the EventSummary
  int32_t delay = static_cast<int32_t>( const_cast<SiStripEventSummary&>(summary).latency() );
  if(firstReading_==-1) firstReading_ = delay;
  float correctedDelay = 0.;
  LogDebug("Commissioning") << "[LatencyTask::fill]; the delay is " << delay;
  // loop on the strips to find the (maybe) non-zero digi
  unsigned int nclusters = 0;
  for(unsigned int strip=0;strip<digis.data.size();strip++) {
    if(digis.data[strip].adc()!=0) {
      // count the "cluster"
      ++nclusters;
      // no TOF correction is applied.
      // 2 reasons: the effect is a priori to thin to be seen with 25ns steps
      // and it biases the result by one clock due to the 25bins in the HistoSet
      correctedDelay = delay*(-25.); // no TOF correction is applied. 
      // compute the bin
      int bin = int((correctedDelay-LOWBIN)/((HIGHBIN-LOWBIN)/NBINS));
      LogDebug("Commissioning") << "[LatencyTask::fill]; using a hit with value " << ( digis.data[strip].adc()&0xff )
                                << " at corrected delay of " << correctedDelay
				<< " in bin " << bin ;
      updateHistoSet( timing_,bin,digis.data[strip].adc()&0xff);
      LogDebug("Commissioning") << "HistoSet timing Updated " << strip << " " << digis.data.size();
      updateHistoSet( timingPartition_,bin,digis.data[strip].adc()&0xff);
      LogDebug("Commissioning") << "HistoSet timingPartition Updated " << strip << " " << digis.data.size();
    }
  }
  // set the occupancy
  int bin = int((delay*(-25.)-LOWBIN)/((HIGHBIN-LOWBIN)/NBINS));
  LogDebug("Commissioning") << "[LatencyTask::fill]; occupancy is " << nclusters;
  updateHistoSet( cluster_,bin,nclusters );
  updateHistoSet( clusterPartition_,bin,nclusters );
}

// -----------------------------------------------------------------------------
//
void LatencyTask::update() {
  LogDebug("Commissioning") << "[LatencyTask::update]";
  updateHistoSet( timing_ );
  updateHistoSet( timingPartition_ );
  updateHistoSet( cluster_ );
  updateHistoSet( clusterPartition_ );
}

