#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/GeometryObjects/interface/MuonOffsetMap.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include <iostream>

class MuonOffsetAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit MuonOffsetAnalyzer(const edm::ParameterSet&);

  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;

private:
  edm::ESGetToken<MuonOffsetMap, IdealGeometryRecord> parToken_;
};

MuonOffsetAnalyzer::MuonOffsetAnalyzer(const edm::ParameterSet&) {
  parToken_ = esConsumes<MuonOffsetMap, IdealGeometryRecord>(edm::ESInputTag{});
}

void MuonOffsetAnalyzer::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup) {
  const auto& par = iSetup.getData(parToken_);
  const MuonOffsetMap* php = &par;

  edm::LogVerbatim("MuonGeom") << "MuonOffsetFromDD: Finds " << php->muonMap_.size() << " entries in the map";

  unsigned int k(0);
  for (auto itr = php->muonMap_.begin(); itr != php->muonMap_.end(); ++itr, ++k) {
    edm::LogVerbatim("MuonGeom") << "[" << k << "] " << itr->first << ": (" << (itr->second).first << ", "
                                 << (itr->second).second << ")";
  }
}

DEFINE_FWK_MODULE(MuonOffsetAnalyzer);
