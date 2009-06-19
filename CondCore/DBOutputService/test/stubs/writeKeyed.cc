#include "FWCore/Framework/interface/EDAnalyzer.h"
#include <string>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/DBOutputService/interface/KeyedElement.h"
#include "CondFormats/Calibration/interface/Conf.h"

class writeKeyed : public edm::EDAnalyzer {
 public:
  explicit writeKeyed(const edm::ParameterSet& iConfig );
  virtual void analyze( const edm::Event&, const edm::EventSetup& ){}
  virtual void endJob();
 private:
  std::string keys, names;
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


  // populated with the keyed payloads (configurations)
  for ( int i=0; i<dict.size();i++)
    for (int j=0;j<7;j++) {
      cond::BaseKeyed * bk=0;
      cond::KeyedElement k( 
			   (0==i%2) ?
			   bk = new condex::ConfI(dict[i]+nums[j],10*i+j)
			   :
			   bk = new condex::ConfF(dict[i]+nums[j],i+0.1*j),      
			   dict[i]+nums[j]);
      std::cout << (*k.m_sum) << " " << k.m_key << std::endl;
      outdb->writeOne(k.m_obj,k.m_sum,k.m_key,keys);
    }

  // populate the master payload
  int run=10;
  for (int j=0;j<7;j++) {
    std::vector<cond::Time_t> * kl = new std::vector<cond::Time_t>(dict.size());
    for ( int i=0; i<dict.size();i++)
      (*kl)[i]=cond::KeyedElement::convert(dict[i]+nums[j]);
    // outdb->writeOne(kl,new cond::GenericSummary(nums[j]),run,names);
    outdb->writeOne(kl,0,run,names);
    run+=10;
  }

}

writeKeyed::writeKeyed(const edm::ParameterSet& iConfig ) :
  keys("keys"), names("names"){}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(writeKeyed);
