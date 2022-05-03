#include <iomanip>
#include <iostream>

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1TMuonBarrelKalmanParamsRcd.h"
#include "CondFormats/L1TObjects/interface/L1TMuonBarrelKalmanParams.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/CondDB/interface/Session.h"

#include <iostream>
using namespace std;

class L1TMuonBarrelKalmanParamsViewer : public edm::EDAnalyzer {
private:
  std::string hash(void *buf, size_t len) const;

public:
  void analyze(const edm::Event &, const edm::EventSetup &) override;

  L1TMuonBarrelKalmanParamsViewer(const edm::ParameterSet &){};
  ~L1TMuonBarrelKalmanParamsViewer(void) override {}
};

#include "Utilities/OpenSSL/interface/openssl_init.h"
#include <cmath>
#include <iostream>
using namespace std;

std::string L1TMuonBarrelKalmanParamsViewer::hash(void *buf, size_t len) const {
  cms::openssl_init();
  EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
  const EVP_MD *md = EVP_get_digestbyname("SHA1");
  if (!EVP_DigestInit_ex(mdctx, md, nullptr))
    throw cms::Exception("L1TMuonBarrelKalmanParamsViewer::hash") << "SHA1 initialization error";

  if (!EVP_DigestUpdate(mdctx, buf, len))
    throw cms::Exception("L1TMuonBarrelKalmanParamsViewer::hash") << "SHA1 processing error";

  unsigned char hash[EVP_MAX_MD_SIZE];
  unsigned int md_len = 0;
  if (!EVP_DigestFinal_ex(mdctx, hash, &md_len))
    throw cms::Exception("L1TMuonBarrelKalmanParamsViewer::hash") << "SHA1 finalization error";

  EVP_MD_CTX_free(mdctx);

  // re-write bytes in hex
  char tmp[EVP_MAX_MD_SIZE * 2 + 1];
  if (md_len > 20)
    md_len = 20;
  for (unsigned int i = 0; i < md_len; i++)
    ::sprintf(&tmp[i * 2], "%02x", hash[i]);

  tmp[md_len * 2] = 0;
  return std::string(tmp);
}

void L1TMuonBarrelKalmanParamsViewer::analyze(const edm::Event &iEvent, const edm::EventSetup &evSetup) {
  edm::ESHandle<L1TMuonBarrelKalmanParams> handle1;
  evSetup.get<L1TMuonBarrelKalmanParamsRcd>().get(handle1);
  std::shared_ptr<L1TMuonBarrelKalmanParams> ptr(new L1TMuonBarrelKalmanParams(*(handle1.product())));

  // Get the nodes and print out
  auto pnodes = ptr->pnodes_[ptr->CONFIG];
  cout << "version    : " << ptr->version_ << endl;
  cout << "fwVersion  : " << hex << pnodes.fwVersion_ << dec << endl;
  cout << "LUTsPath   : " << pnodes.kalmanLUTsPath_ << endl;

  // typedef std::map<short, short, std::less<short> > LUT;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1TMuonBarrelKalmanParamsViewer);
