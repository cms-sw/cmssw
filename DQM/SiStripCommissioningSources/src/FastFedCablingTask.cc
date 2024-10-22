#include "DQM/SiStripCommissioningSources/interface/FastFedCablingTask.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>
#include <sstream>
#include <iomanip>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
FastFedCablingTask::FastFedCablingTask(DQMStore* dqm, const FedChannelConnection& conn)
    : CommissioningTask(dqm, conn, "FastFedCablingTask"), histo_() {}

// -----------------------------------------------------------------------------
//
FastFedCablingTask::~FastFedCablingTask() {}

// -----------------------------------------------------------------------------
//
void FastFedCablingTask::book() {
  std::string title = SiStripHistoTitle(sistrip::EXPERT_HISTO,
                                        sistrip::FAST_CABLING,
                                        sistrip::FED_KEY,
                                        fedKey(),
                                        sistrip::LLD_CHAN,
                                        connection().lldChannel())
                          .title();

  uint16_t nbins = 34;
  histo_.histo(dqm()->bookProfile(title, title, nbins, -0.5, nbins * 1. - 0.5, 1025, 0., 1025.));

  histo_.vNumOfEntries_.resize(nbins, 0);
  histo_.vSumOfContents_.resize(nbins, 0);
  histo_.vSumOfSquares_.resize(nbins, 0);
}

// -----------------------------------------------------------------------------
//
void FastFedCablingTask::fill(const SiStripEventSummary& summary, const edm::DetSet<SiStripRawDigi>& digis) {
  if (digis.data.empty()) {
    edm::LogWarning(mlDqmSource_) << "[FastFedCablingTask::" << __func__ << "]"
                                  << " No digis found!";
    return;
  }

  uint32_t bin = summary.binNumber();
  for (uint16_t ibin = 0; ibin < digis.data.size(); ibin++) {
    updateHistoSet(histo_, bin, digis.data[ibin].adc());
  }
}

// -----------------------------------------------------------------------------
//
void FastFedCablingTask::update() { updateHistoSet(histo_); }
