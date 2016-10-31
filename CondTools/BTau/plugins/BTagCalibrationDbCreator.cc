#include <memory>
#include <string>
#include <iostream>
#include <sstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/BTauObjects/interface/BTagCalibration.h"

class BTagCalibrationDbCreator : public edm::EDAnalyzer
{
public:
  BTagCalibrationDbCreator(const edm::ParameterSet&);
  virtual void beginJob() override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override {}
  virtual void endJob() override {}
  ~BTagCalibrationDbCreator() {}

private:
  std::string csvFile_;
  std::string tagger_;
};

BTagCalibrationDbCreator::BTagCalibrationDbCreator(const edm::ParameterSet& p):
  csvFile_(p.getUntrackedParameter<std::string>("csvFile")),
  tagger_ (p.getUntrackedParameter<std::string>("tagger" ))
{}

void BTagCalibrationDbCreator::beginJob()
{
  auto calib = new BTagCalibration(tagger_, csvFile_);
  edm::Service<cond::service::PoolDBOutputService> s;
  if (s.isAvailable()) {
    if (s->isNewTagRequest(tagger_)) {
      s->createNewIOV<BTagCalibration>(
        calib,
        s->beginOfTime(),
        s->endOfTime(),
        tagger_
      );
    } else {
      s->appendSinceTime<BTagCalibration>(
        calib,
        111,
        tagger_
      );
    }
  } else {
    std::cout << "ERROR: DB service not available" << std::endl;
  }
}

DEFINE_FWK_MODULE(BTagCalibrationDbCreator);
