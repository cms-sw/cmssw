#include <iostream>
#include <map>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/HcalTestBeamData/interface/HcalTB02Parameters.h"
#include "Geometry/HcalTestBeamData/interface/HcalTB06BeamParameters.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HcalTBParameterTester : public edm::one::EDAnalyzer<> {
public:
  explicit HcalTBParameterTester(const edm::ParameterSet&);
  ~HcalTBParameterTester() override = default;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  const std::string name_;
  edm::ESGetToken<HcalTB02Parameters, IdealGeometryRecord> token1_;
  edm::ESGetToken<HcalTB06BeamParameters, IdealGeometryRecord> token2_;
  const int mode_;
};

HcalTBParameterTester::HcalTBParameterTester(const edm::ParameterSet& ic)
    : name_(ic.getUntrackedParameter<std::string>("Name")),
      token1_(esConsumes<HcalTB02Parameters, IdealGeometryRecord>(edm::ESInputTag{"", name_})),
      token2_(esConsumes<HcalTB06BeamParameters, IdealGeometryRecord>(edm::ESInputTag{})),
      mode_(ic.getUntrackedParameter<int>("Mode")) {}

void HcalTBParameterTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (mode_ == 0) {
    const auto& hcp = iSetup.getData(token1_);
    const auto* php = &hcp;
    edm::LogVerbatim("HCalGeom") << "TB02Parameters for " << name_;
    edm::LogVerbatim("HCalGeom") << "Length map with " << php->lengthMap_.size() << " elements";
    std::map<std::string, double>::const_iterator itr = php->lengthMap_.begin();
    int i(0);
    for (; itr != php->lengthMap_.end(); ++itr, ++i)
      edm::LogVerbatim("HCalGeom") << "[" << i << "] " << itr->first << " " << itr->second << " mm";
  } else {
    const auto& hcp = iSetup.getData(token2_);
    const auto* php = &hcp;
    edm::LogVerbatim("HCalGeom") << "TB06BeamParameters:: Material " << php->material_;
    edm::LogVerbatim("HCalGeom") << "TB06BeamParameters:: " << php->wchambers_.size() << " wire chambers:";
    for (unsigned int k = 0; k < php->wchambers_.size(); ++k)
      edm::LogVerbatim("HCalGeom") << "[" << k << "] " << php->wchambers_[k];
  }
}

DEFINE_FWK_MODULE(HcalTBParameterTester);
