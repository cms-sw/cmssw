#include "CalibTracker/SiStripAPVAnalysis/interface/ApvAnalysisFactory.h"
#include "CalibTracker/SiStripAPVAnalysis/interface/ApvFactoryService.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"
#include "DQM/SiStripCommon/interface/UpdateTProfile.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQM/SiStripCommissioningSources/interface/PedsOnlyTask.h"

using namespace sistrip;

// -----------------------------------------------------------------------------
//
PedsOnlyTask::PedsOnlyTask(DQMStore *dqm, const FedChannelConnection &rCONN)
    : CommissioningTask(dqm, rCONN, "PedsOnlyTask") {
  //@@ NOT GUARANTEED TO BE THREAD SAFE!
  pApvFactory_ = edm::Service<ApvFactoryService>().operator->()->getApvFactory();

  LogTrace(mlDqmSource_) << "[PedsOnlyTask::" << __func__ << "]"
                         << " Constructing object...";
}

// -----------------------------------------------------------------------------
//
PedsOnlyTask::~PedsOnlyTask() {
  LogTrace(mlDqmSource_) << "[PedsOnlyTask::" << __func__ << "]"
                         << " Destructing object...";

  // Have to delete pApvFactory_ manually even though we didn't create it
  // ourself. :(
  if (pApvFactory_) {
    delete pApvFactory_;
  }
}

// -----------------------------------------------------------------------------
//
void PedsOnlyTask::book() {
  LogTrace(mlDqmSource_) << "[PedsOnlyTask::" << __func__ << "]";

  const uint16_t nBINS = 256;

  {
    // Pedestals
    std::string title = SiStripHistoTitle(sistrip::EXPERT_HISTO,
                                          sistrip::PEDS_ONLY,
                                          sistrip::FED_KEY,
                                          fedKey(),
                                          sistrip::LLD_CHAN,
                                          connection().lldChannel(),
                                          sistrip::extrainfo::pedestals_)
                            .title();

    HistoSet oHSet;
    oHSet.isProfile_ = true;

    oHSet.vNumOfEntries_.resize(nBINS, 0);
    oHSet.vSumOfContents_.resize(nBINS, 0);
    oHSet.vSumOfSquares_.resize(nBINS, 0);

    oHSet.histo(dqm()->bookProfile(title, title, nBINS, -0.5, nBINS * 1. - 0.5, 1025, 0., 1025.));

    peds_.push_back(oHSet);
  }

  {
    // Raw Noise
    std::string title = SiStripHistoTitle(sistrip::EXPERT_HISTO,
                                          sistrip::PEDS_ONLY,
                                          sistrip::FED_KEY,
                                          fedKey(),
                                          sistrip::LLD_CHAN,
                                          connection().lldChannel(),
                                          sistrip::extrainfo::rawNoise_)
                            .title();

    HistoSet oHSet;
    oHSet.isProfile_ = true;

    oHSet.vNumOfEntries_.resize(nBINS, 0);
    oHSet.vSumOfContents_.resize(nBINS, 0);
    oHSet.vSumOfSquares_.resize(nBINS, 0);

    oHSet.histo(dqm()->bookProfile(title, title, nBINS, -0.5, nBINS * 1. - 0.5, 1025, 0., 1025.));

    peds_.push_back(oHSet);
  }

  pApvFactory_->instantiateApvs(connection().detId(), connection().nApvs());
}

// -----------------------------------------------------------------------------
//
void PedsOnlyTask::fill(const SiStripEventSummary &rSummary, const edm::DetSet<SiStripRawDigi> &rDigis) {
  pApvFactory_->updatePair(connection().detId(), connection().apvPairNumber(), rDigis);
}

// -----------------------------------------------------------------------------
//
void PedsOnlyTask::update() {
  static UpdateTProfile updateTProfile;

  TProfile *pedsProf = ExtractTObject<TProfile>().extract(peds_[0].histo());
  TProfile *noiseProf = ExtractTObject<TProfile>().extract(peds_[1].histo());

  for (uint16_t nLclApv = 2 * connection().apvPairNumber(), nMaxLclApv = nLclApv + 2; nMaxLclApv > nLclApv; ++nLclApv) {
    ApvAnalysis::PedestalType lclPedestals;
    ApvAnalysis::PedestalType lclNoises;

    pApvFactory_->getPedestal(connection().detId(), nLclApv, lclPedestals);
    pApvFactory_->getRawNoise(connection().detId(), nLclApv, lclNoises);

    const uint16_t nSTART_BIN = 128 * (nLclApv % 2);

    for (uint16_t nBin = 0, nAbsBin = nSTART_BIN + nBin + 1; 128 > nBin; ++nBin, ++nAbsBin) {
      updateTProfile.setBinContent(pedsProf, nAbsBin, 5, lclPedestals[nBin], lclNoises[nBin]);
      updateTProfile.setBinContent(noiseProf, nAbsBin, 5, lclNoises[nBin], 0);
    }  // End loop over BINs
  }    // End loop over Local Apvs
}
// -----------------------------------------------------------------------------
