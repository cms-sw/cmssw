#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "Geometry/MuonNumbering/interface/DD4hep_MuonNumbering.h"

class DDTestMuonNumbering : public edm::one::EDAnalyzer<> {
public:
  explicit DDTestMuonNumbering(const edm::ParameterSet&){}

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}
};

void
DDTestMuonNumbering::analyze(const edm::Event&, const edm::EventSetup& iEventSetup)
{
  edm::LogVerbatim("Geometry") << "DDTestMuonNumbering::analyze";
  edm::ESTransientHandle<cms::MuonNumbering> numbering;
  iEventSetup.get<MuonNumberingRecord>().get(numbering);

  edm::LogVerbatim("Geometry") << "MuonNumbering size: " << numbering->values().size();
  edm::LogVerbatim("Geometry").log([&numbering](auto& log) {
      for(const auto& i: numbering->values()) {
	log << " " << i.first << " = " << i.second;
	log << '\n';
      }
    });
}

DEFINE_FWK_MODULE(DDTestMuonNumbering);
