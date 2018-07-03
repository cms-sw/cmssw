// -*- C++ -*-
//
// Description: Fill in time bias record from an old configuration file.
// Original Author:  Dmitrijus Bugelskis
//         Created:  Thu, 14 Nov 2013 17:44:11 GMT
//

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESProducts.h"

#include "CondFormats/EcalObjects/interface/EcalTimeBiasCorrections.h"

#include "CondCore/CondDB/interface/Serialization.h"
#include "CondFormats/External/interface/EcalDetID.h"
#include "CondFormats/External/interface/SMatrix.h"
#include "CondFormats/External/interface/Timestamp.h"

#include "CondFormats/DataRecord/interface/EcalTimeBiasCorrectionsRcd.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

class EcalTimeBiasCorrectionsFillInitial : public edm::EDAnalyzer {
 public:
  explicit EcalTimeBiasCorrectionsFillInitial(const edm::ParameterSet &);
  ~EcalTimeBiasCorrectionsFillInitial();

  void analyze(const edm::Event &, const edm::EventSetup &);
  void endJob();

 private:
  std::vector<double> EBtimeCorrAmplitudeBins_;
  std::vector<double> EBtimeCorrShiftBins_;
  std::vector<double> EEtimeCorrAmplitudeBins_;
  std::vector<double> EEtimeCorrShiftBins_;

  EcalTimeBiasCorrections *bias_;
};

EcalTimeBiasCorrectionsFillInitial::EcalTimeBiasCorrectionsFillInitial(
    const edm::ParameterSet &ps) {

  EBtimeCorrAmplitudeBins_ =
      ps.getParameter<std::vector<double> >("EBtimeCorrAmplitudeBins");
  EBtimeCorrShiftBins_ =
      ps.getParameter<std::vector<double> >("EBtimeCorrShiftBins");
  EEtimeCorrAmplitudeBins_ =
      ps.getParameter<std::vector<double> >("EEtimeCorrAmplitudeBins");
  EEtimeCorrShiftBins_ =
      ps.getParameter<std::vector<double> >("EEtimeCorrShiftBins");

  if (EBtimeCorrAmplitudeBins_.size() != EBtimeCorrShiftBins_.size()) {
    edm::LogError("EcalRecHitError") << "Size of EBtimeCorrAmplitudeBins "
                                        "different from EBtimeCorrShiftBins.";
  }

  if (EEtimeCorrAmplitudeBins_.size() != EEtimeCorrShiftBins_.size()) {
    edm::LogError("EcalRecHitError") << "Size of EEtimeCorrAmplitudeBins "
                                        "different from EEtimeCorrShiftBins.";
  }

  bias_ = new EcalTimeBiasCorrections();

  copy(EBtimeCorrAmplitudeBins_.begin(), EBtimeCorrAmplitudeBins_.end(),
       back_inserter(bias_->EBTimeCorrAmplitudeBins));

  copy(EBtimeCorrShiftBins_.begin(), EBtimeCorrShiftBins_.end(),
       back_inserter(bias_->EBTimeCorrShiftBins));

  copy(EEtimeCorrAmplitudeBins_.begin(), EEtimeCorrAmplitudeBins_.end(),
       back_inserter(bias_->EETimeCorrAmplitudeBins));

  copy(EEtimeCorrShiftBins_.begin(), EEtimeCorrShiftBins_.end(),
       back_inserter(bias_->EETimeCorrShiftBins));
}

EcalTimeBiasCorrectionsFillInitial::~EcalTimeBiasCorrectionsFillInitial() {}

void EcalTimeBiasCorrectionsFillInitial::analyze(
    const edm::Event &iEvent, const edm::EventSetup &iSetup) {}
void EcalTimeBiasCorrectionsFillInitial::endJob() {
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable()) {
    poolDbService->writeOne(this->bias_, poolDbService->beginOfTime(),
                            "EcalTimeBiasCorrectionsRcd");
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EcalTimeBiasCorrectionsFillInitial);
