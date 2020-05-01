#include <iostream>
#include <map>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "Geometry/MuonNumbering/interface/MuonGeometryConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class MuonGeometryConstantsTester : public edm::one::EDAnalyzer<> {
public:
  explicit MuonGeometryConstantsTester(const edm::ParameterSet&);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  edm::ESGetToken<MuonGeometryConstants, IdealGeometryRecord> token_;
};

MuonGeometryConstantsTester::MuonGeometryConstantsTester(const edm::ParameterSet&) {
  token_ = esConsumes<MuonGeometryConstants, IdealGeometryRecord>(edm::ESInputTag{});
}

void MuonGeometryConstantsTester::analyze(const edm::Event&, const edm::EventSetup& iS) {
  const auto& par = iS.getData(token_);
  const MuonGeometryConstants* parMuon = &par;
  if (parMuon != nullptr) {
    std::cout << "\n\nMuonDDDConstants found with " << parMuon->size() << " contents\n";
    for (unsigned int k = 0; k < parMuon->size(); ++k) {
      auto entry = parMuon->getEntry(k);
      std::cout << " [" << k << "] " << entry.first << " = " << entry.second << std::endl;
    }
  } else {
    std::cout << "\n\nMuonDDDConstants not found in Event Setup\n";
  }
}

DEFINE_FWK_MODULE(MuonGeometryConstantsTester);
