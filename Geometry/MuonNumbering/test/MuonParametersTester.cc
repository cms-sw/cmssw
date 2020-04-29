#include <iostream>
#include <map>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "Geometry/MuonNumbering/interface/MuonDDDParameters.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class MuonParametersTester : public edm::one::EDAnalyzer<> {
public:
  explicit MuonParametersTester(const edm::ParameterSet&);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  edm::ESGetToken<MuonDDDParameters, IdealGeometryRecord> token_;
};

MuonParametersTester::MuonParametersTester(const edm::ParameterSet&) {
  token_ = esConsumes<MuonDDDParameters, IdealGeometryRecord>(edm::ESInputTag{});
}

void MuonParametersTester::analyze(const edm::Event&, const edm::EventSetup& iS) {
  const auto& par = iS.getData(token_);
  const MuonDDDParameters* parMuon = &par;
  if (parMuon != nullptr) {
    std::cout << "\n\nMuonDDDParameters found with " << parMuon->size() << " contents\n";
    for (unsigned int k = 0; k < parMuon->size(); ++k) {
      auto entry = parMuon->getEntry(k);
      std::cout << " [" << k << "] " << entry.first << " = " << entry.second << std::endl;
    }
  } else {
    std::cout << "\n\nMuonDDDParameters not found in Event Setup\n";
  }
}

DEFINE_FWK_MODULE(MuonParametersTester);
