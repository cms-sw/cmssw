/**
 * \file EcalContainmentCorrectionAnalyzer
 *
 * Analyzer to test Shower Containment Corrections
 *
 *
 */

#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/DataRecord/interface/EcalGlobalShowerContainmentCorrectionsVsEtaRcd.h"
#include "CondFormats/EcalCorrections/interface/EcalGlobalShowerContainmentCorrectionsVsEta.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"

class EcalContainmentCorrectionAnalyzer : public edm::global::EDAnalyzer<> {
public:
  explicit EcalContainmentCorrectionAnalyzer(const edm::ParameterSet &ps);
  ~EcalContainmentCorrectionAnalyzer() override = default;

  void analyze(edm::StreamID, edm::Event const &, edm::EventSetup const &) const override;

protected:
  const edm::ESGetToken<EcalGlobalShowerContainmentCorrectionsVsEta, EcalGlobalShowerContainmentCorrectionsVsEtaRcd>
      esToken_;
};

DEFINE_FWK_MODULE(EcalContainmentCorrectionAnalyzer);

EcalContainmentCorrectionAnalyzer::EcalContainmentCorrectionAnalyzer(const edm::ParameterSet &ps)
    : esToken_(esConsumes()) {}

void EcalContainmentCorrectionAnalyzer::analyze(edm::StreamID,
                                                edm::Event const &iEvent,
                                                const edm::EventSetup &iSetup) const {
  const auto &corr = iSetup.getData(esToken_);

  for (int i = 1; i < 86; ++i) {
    EBDetId aId(i, 1, EBDetId::ETAPHIMODE);
    double e3x3 = corr.correction3x3(aId);
    double e5x5 = corr.correction5x5(aId);
    edm::LogVerbatim("EcalContainmentCorrectionAnalyzer")
        << "ieta " << aId.ieta() << " " << e3x3 << " " << e5x5 << "\n";
  }
}
