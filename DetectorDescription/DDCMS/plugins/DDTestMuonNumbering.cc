#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DetectorDescription/DDCMS/interface/MuonNumberingRcd.h"
#include "DetectorDescription/DDCMS/interface/MuonNumbering.h"

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
  cout << "DDTestMuonNumbering::analyze\n";
  ESTransientHandle<MuonNumbering> numbering;
  iEventSetup.get<MuonNumberingRcd>().get(numbering);

  cout << "MuonNumbering size: " << numbering->values.size() << "\n";
  for(const auto& i: numbering->values) {
    cout << " " << i.first << " = " << i.second;
    cout << '\n';
  }
}

DEFINE_FWK_MODULE(DDTestMuonNumbering);
