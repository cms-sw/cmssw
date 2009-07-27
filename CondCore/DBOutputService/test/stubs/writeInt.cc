#include "FWCore/Framework/interface/EDAnalyzer.h"
#include <string>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/Common/interface/GenericSummary.h"

#include<vector>
#include<sstream>

typedef std::vector<int> Payload;


namespace {

  std::string toa(int i) {
    std::ostringstream ss;
    ss << i;
    return ss.str();

  }

}

class writeInt : public edm::EDAnalyzer {
 public:
  explicit writeInt(const edm::ParameterSet& iConfig );
  virtual void analyze( const edm::Event&, const edm::EventSetup& ){}
  virtual void endJob();
 private:
  std::string cont;
  int me;
};

void
writeInt::endJob() {

  edm::Service<cond::service::PoolDBOutputService> outdb;
  
  outdb->writeOne(new std::vector<int>(1,me),
		  new cond::GenericSummary(toa(me)),
		  me,cont);

}

writeInt::writeInt(const edm::ParameterSet& iConfig ) :
  cont("oneInt"),
  me(iConfig.getParam<int>("Number")) {}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(writeInt);
