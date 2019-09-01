#include "DQM/SiStripCommissioningSources/interface/FedTimingTask.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace sistrip;

// -----------------------------------------------------------------------------
//
FedTimingTask::FedTimingTask(DQMStore* dqm, const FedChannelConnection& conn)
    : CommissioningTask(dqm, conn, "FedTimingTask"),
      timing_(),
      nBins_(
          40)  //@@ this should be from number of scope mode samples (mean booking in event loop and putting scope mode length in trigger fed)
{
  LogDebug("Commissioning") << "[FedTimingTask::FedTimingTask] Constructing object...";
}

// -----------------------------------------------------------------------------
//
FedTimingTask::~FedTimingTask() { LogDebug("Commissioning") << "[FedTimingTask::FedTimingTask] Destructing object..."; }

// -----------------------------------------------------------------------------
//
void FedTimingTask::book() {
  LogDebug("Commissioning") << "[FedTimingTask::book]";

  uint16_t nbins = 24 * nBins_;  // 24 "fine" pll skews possible

  std::string title;

  title = SiStripHistoTitle(sistrip::EXPERT_HISTO,
                            sistrip::FED_TIMING,
                            sistrip::FED_KEY,
                            fedKey(),
                            sistrip::LLD_CHAN,
                            connection().lldChannel())
              .title();

  timing_.histo(dqm()->bookProfile(title, title, nbins, -0.5, nbins * 1. - 0.5, 1025, 0., 1025.));

  timing_.vNumOfEntries_.resize(nbins, 0);
  timing_.vSumOfContents_.resize(nbins, 0);
  timing_.vSumOfSquares_.resize(nbins, 0);
}

// -----------------------------------------------------------------------------
//
/*
  Some notes: 
  - use all samples 
  - extract number of samples from trigger fed
  - need to book histos in event loop?
  - why only use fine skew setting when filling histos? should use coarse setting as well?
  - why do different settings every 100 events - change more freq? 
*/
void FedTimingTask::fill(const SiStripEventSummary& summary, const edm::DetSet<SiStripRawDigi>& digis) {
  LogDebug("Commissioning") << "[FedTimingTask::fill]";

  //@@ if scope mode length is in trigger fed, then
  //@@ can add check here on number of digis
  if (digis.data.size() < nBins_) {
    edm::LogWarning("Commissioning") << "[FedTimingTask::fill]"
                                     << " Unexpected number of digis! " << digis.data.size();
  } else {
    uint32_t pll_fine = summary.pllFine();
    for (uint16_t coarse = 0; coarse < nBins_ /*digis.data.size()*/; coarse++) {
      uint16_t fine = (coarse + 1) * 24 - (pll_fine + 1);
      updateHistoSet(timing_, fine, digis.data[coarse].adc());
    }
  }
}

// -----------------------------------------------------------------------------
//
void FedTimingTask::update() {
  LogDebug("Commissioning") << "[FedTimingTask::update]";
  updateHistoSet(timing_);
}
