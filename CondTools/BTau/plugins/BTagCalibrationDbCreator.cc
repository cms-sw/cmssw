#include <memory>
#include <string>
#include <iostream>
#include <sstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/BTauObjects/interface/BTagCalibration.h"

class BTagCalibrationDbCreator : public edm::one::EDAnalyzer<> {
public:
  BTagCalibrationDbCreator(const edm::ParameterSet&);
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override {}
  void endJob() override {}
  ~BTagCalibrationDbCreator() override {}

private:
  std::string csvFile_;
  std::string tagger_;
};

BTagCalibrationDbCreator::BTagCalibrationDbCreator(const edm::ParameterSet& p)
    : csvFile_(p.getUntrackedParameter<std::string>("csvFile")),
      tagger_(p.getUntrackedParameter<std::string>("tagger")) {}

void BTagCalibrationDbCreator::beginJob() {
  BTagCalibration calib(tagger_, csvFile_, true);
  edm::Service<cond::service::PoolDBOutputService> s;
  if (s.isAvailable()) {
    s->writeOneIOV(calib, s->beginOfTime(), tagger_);
  } else {
    std::cout << "ERROR: DB service not available" << std::endl;
  }
}

DEFINE_FWK_MODULE(BTagCalibrationDbCreator);
