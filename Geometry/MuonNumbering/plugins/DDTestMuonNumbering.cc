#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "Geometry/MuonNumbering/interface/DD4hep_MuonNumbering.h"

#include <iostream>

using namespace std;
using namespace cms;
using namespace edm;

class DDTestMuonNumbering : public one::EDAnalyzer<> {
public:
  explicit DDTestMuonNumbering(const ParameterSet&){}

  void beginJob() override {}
  void analyze(Event const& iEvent, EventSetup const&) override;
  void endJob() override {}
};

void
DDTestMuonNumbering::analyze(const Event&, const EventSetup& iEventSetup)
{
  LogVerbatim("Geometry") << "DDTestMuonNumbering::analyze";
  ESTransientHandle<MuonNumbering> numbering;
  iEventSetup.get<MuonNumberingRecord>().get(numbering);

  LogVerbatim("Geometry") << "MuonNumbering size: " << numbering->values().size();
  LogVerbatim("Geometry").log([&numbering](auto& log) {
      for(const auto& i: numbering->values()) {
	log << " " << i.first << " = " << i.second;
	log << '\n';
      }
    });
}

DEFINE_FWK_MODULE(DDTestMuonNumbering);
