#include "FWCore/Framework/interface/EDAnalyzer.h"
#include <string>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/DBOutputService/interface/KeyedElement"
#inlucde <sstream>

class writeKeyed : public edm::EDAnalyzer {
 public:
  explicit writeKeyed(const edm::ParameterSet& iConfig );
  virtual void analyze( const edm::Event&, const edm::EventSetup& ){}
  virtual void endJob();
 private:
  std::string crapName;
};

void
writeKeyed::endJob() {

  std::vector<std::string> dict;
  size_t tot=0;
  dict.push_back("Sneezy");
  tot+=dict.back().size();
  dict.push_back("Sleepy");
  tot+=dict.back().size();
  dict.push_back("Dopey");
  tot+=dict.back().size();
  dict.push_back("Doc");
  tot+=dict.back().size();
  dict.push_back("Happy");
  tot+=dict.back().size();
  dict.push_back("Bashful");
  tot+=dict.back().size();
  dict.push_back("Grumpy");
  tot+=dict.back().size();

  char const * nums[] = {"1","2","3","4","5","6","7"};

  edm::Service<cond::service::PoolDBOutputService> outdb;

  
  for ( int i=0; i<dict.size();i++)
    for (int j=0;j<7;j++) {
      KeyedElement k(new BaseKeyed,dict[i]+nums[j]);
      outdb->writeOne(k.m_obj,k.m_sum,k.m_key,crapName);
    }
}

writeKeyed::writeKeyed(const edm::ParameterSet& iConfig ) :
  crapName("uniqueCarppyName"){}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(writeKeyed);
