#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/DBOutputService/interface/KeyedElement.h"
#include "CondFormats/Calibration/interface/Conf.h"

#include <iostream>
#include <string>

class writeKeyed : public edm::one::EDAnalyzer<> {
public:
  explicit writeKeyed(const edm::ParameterSet& iConfig);
  virtual void analyze(const edm::Event&, const edm::EventSetup&) {}
  virtual void endJob();

private:
  std::string confcont, confiov;
};

void writeKeyed::endJob() {
  std::vector<std::string> dict;
  dict.push_back("Sneezy");
  dict.push_back("Sleepy");
  dict.push_back("Dopey");
  dict.push_back("Doc");
  dict.push_back("Happy");
  dict.push_back("Bashful");
  dict.push_back("Grumpy");

  char const* nums[] = {"1", "2", "3", "4", "5", "6", "7"};

  edm::Service<cond::service::PoolDBOutputService> outdb;

  std::map<cond::Time_t, cond::BaseKeyed*> keys;
  // populated with the keyed payloads (configurations)
  for (size_t i = 0; i < dict.size(); ++i)
    for (size_t j = 0; j < 7; ++j) {
      cond::BaseKeyed* bk = 0;
      cond::KeyedElement k((0 == i % 2) ? bk = new condex::ConfI(dict[i] + nums[j], 10 * i + j)
                                        : bk = new condex::ConfF(dict[i] + nums[j], i + 0.1 * j),
                           dict[i] + nums[j]);
      std::cout << k.m_skey << " " << k.m_key << std::endl;

      keys.insert(std::make_pair(k.m_key, k.m_obj));
    }

  std::cout << "# uploading keys..." << std::endl;
  for (auto k : keys)
    outdb->writeOneIOV(k.second, k.first, confcont);

  std::cout << "# uploading master payloads..." << std::endl;
  // populate the master payload
  int run = 10;
  for (size_t j = 0; j < 7; ++j) {
    std::vector<cond::Time_t> kl;
    for (size_t i = 0; i < dict.size(); ++i)
      (kl).at(i) = cond::KeyedElement::convert(dict[i] + nums[j]);
    outdb->writeOneIOV(kl, run, confiov);
    run += 10;
  }
}

writeKeyed::writeKeyed(const edm::ParameterSet& iConfig) : confcont("confcont"), confiov("confiov") {}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(writeKeyed);
