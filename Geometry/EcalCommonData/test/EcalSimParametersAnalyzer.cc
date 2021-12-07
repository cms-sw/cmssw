#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CondFormats/GeometryObjects/interface/EcalSimulationParameters.h"
#include "Geometry/Records/interface/HcalParametersRcd.h"
#include <iostream>

class EcalSimParametersAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit EcalSimParametersAnalyzer(const edm::ParameterSet&);

  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  edm::ESGetToken<EcalSimulationParameters, IdealGeometryRecord> simparToken_;
  std::string name_;
};

EcalSimParametersAnalyzer::EcalSimParametersAnalyzer(const edm::ParameterSet& ic)
    : name_(ic.getUntrackedParameter<std::string>("name")) {
  simparToken_ = esConsumes<EcalSimulationParameters, IdealGeometryRecord>(edm::ESInputTag{"", name_});
}

void EcalSimParametersAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("name", "EcalHitsEB");
  descriptions.add("ecalSimulationParametersAnalyzerEB", desc);
}

void EcalSimParametersAnalyzer::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup) {
  const auto& parS = iSetup.getData(simparToken_);
  const EcalSimulationParameters* parsim = &parS;
  if (parsim != nullptr) {
    edm::LogVerbatim("EcalGeom") << "\n nxtalEta_: " << parsim->nxtalEta_ << "\n nxtalPhi_: " << parsim->nxtalPhi_
                                 << "\n phiBaskets_: " << parsim->phiBaskets_ << "\n etaBaskets_: ";
    std::ostringstream st1;
    for (const auto& it : parsim->etaBaskets_)
      st1 << it << ", ";
    st1 << "\n ncrys_: " << parsim->ncrys_ << "\n nmods_: " << parsim->nmods_ << "\n useWeight_: " << parsim->useWeight_
        << "\n depth1Name_: " << parsim->depth1Name_ << "\n depth2Name_: " << parsim->depth2Name_
        << "\n lvNames_: " << parsim->lvNames_.size() << " occurences";
    edm::LogVerbatim("EcalGeom") << st1.str();
    int kount(0);
    std::ostringstream st2;
    for (const auto& it : parsim->lvNames_) {
      st2 << it << ", ";
      ++kount;
      if (kount == 8) {
        st2 << "\n";
        kount = 0;
      }
    }
    kount = 0;
    st2 << "\n matNames_: " << parsim->matNames_.size() << " occurences";
    edm::LogVerbatim("EcalGeom") << st2.str();
    std::ostringstream st3;
    for (const auto& it : parsim->matNames_) {
      st3 << it << ", ";
      ++kount;
      if (kount == 7) {
        st3 << "\n";
        kount = 0;
      }
    }
    kount = 0;
    st3 << "\n dzs_: " << parsim->dzs_.size() << " occurences";
    edm::LogVerbatim("EcalGeom") << st3.str();
    std::ostringstream st4;
    for (const auto& it : parsim->dzs_) {
      st4 << it << ", ";
      ++kount;
      if (kount == 20) {
        st4 << "\n";
        kount = 0;
      }
    }
    edm::LogVerbatim("EcalGeom") << st4.str();
  }
}

DEFINE_FWK_MODULE(EcalSimParametersAnalyzer);
