#include "FWCore/Framework/interface/EDAnalyzer.h"
#include <string>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include <vector>
#include <sstream>

#include <boost/serialization/vector.hpp>

typedef std::vector<int> Payload;

class writeInt : public edm::EDAnalyzer {
public:
  explicit writeInt(const edm::ParameterSet& iConfig);
  void analyze(const edm::Event&, const edm::EventSetup&) override {}
  void endJob() override;

private:
  std::string cont;
  int me;
};

void writeInt::endJob() {
  edm::Service<cond::service::PoolDBOutputService> outdb;

  outdb->writeOne(new std::vector<int>(1, me), me, cont);
}

writeInt::writeInt(const edm::ParameterSet& iConfig) : cont("oneInt"), me(iConfig.getParameter<int>("Number")) {}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(writeInt);
