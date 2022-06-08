#include <iomanip>
#include <iostream>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1TMuonGlobalParamsRcd.h"
#include "CondFormats/L1TObjects/interface/L1TMuonGlobalParams.h"
#include "L1Trigger/L1TMuon/interface/L1TMuonGlobalParamsHelper.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/CondDB/interface/Session.h"

#include <iostream>
using namespace std;

class L1TMuonGlobalParamsViewer : public edm::one::EDAnalyzer<> {
private:
  //    bool printLayerMap;
  std::string hash(void* buf, size_t len) const;
  void printLUT(l1t::LUT* lut, const char* name) const;

  edm::ESGetToken<L1TMuonGlobalParams, L1TMuonGlobalParamsRcd> token_;

public:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  //    string hash(void *buf, size_t len) const ;

  explicit L1TMuonGlobalParamsViewer(const edm::ParameterSet& pset) : token_{esConsumes()} {
    //       printLayerMap   = pset.getUntrackedParameter<bool>("printLayerMap",  false);
  }
};

#include "Utilities/OpenSSL/interface/openssl_init.h"
#include <cmath>
#include <iostream>
using namespace std;

string L1TMuonGlobalParamsViewer::hash(void* buf, size_t len) const {
  cms::openssl_init();
  EVP_MD_CTX* mdctx = EVP_MD_CTX_new();
  const EVP_MD* md = EVP_get_digestbyname("SHA1");
  if (!EVP_DigestInit_ex(mdctx, md, nullptr))
    throw cms::Exception("L1TMuonGlobalParamsViewer::hash") << "SHA1 initialization error";

  if (!EVP_DigestUpdate(mdctx, buf, len))
    throw cms::Exception("L1TMuonGlobalParamsViewer::hash") << "SHA1 processing error";

  unsigned char hash[EVP_MAX_MD_SIZE];
  unsigned int md_len = 0;
  if (!EVP_DigestFinal_ex(mdctx, hash, &md_len))
    throw cms::Exception("L1TMuonGlobalParamsViewer::hash") << "SHA1 finalization error";

  EVP_MD_CTX_free(mdctx);

  // re-write bytes in hex
  char tmp[EVP_MAX_MD_SIZE * 2 + 1];
  if (md_len > 20)
    md_len = 20;
  for (unsigned int i = 0; i < md_len; i++)
    ::sprintf(&tmp[i * 2], "%02x", hash[i]);

  tmp[md_len * 2] = 0;
  return string(tmp);
}

void L1TMuonGlobalParamsViewer::printLUT(l1t::LUT* lut, const char* name) const {
  if (!lut->empty()) {
    cout << "  " << std::setw(24) << name << "[" << lut->maxSize() << "] " << flush;
    int pod[lut->maxSize()];
    for (unsigned int i = 0; i < lut->maxSize(); i++)
      pod[i] = lut->data(i);
    cout << hash(pod, sizeof(int) * lut->maxSize()) << endl;
  } else {
    cout << "  " << std::setw(24) << name << "[0]" << endl;
  }
}

void L1TMuonGlobalParamsViewer::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  // Pull the config from the ES
  //   has to be a copy as call non-const methods
  L1TMuonGlobalParams ptr1 = evSetup.getData(token_);

  //    cout<<"Some fields in L1TMuonGlobalParams: "<<endl;

  //    ((L1TMuonGlobalParamsHelper*)ptr1.get())->print(cout);

  printLUT(ptr1.absIsoCheckMemLUT(), "absIsoCheckMemLUT");
  printLUT(ptr1.absIsoCheckMemLUT(), "absIsoCheckMemLUT");
  printLUT(ptr1.relIsoCheckMemLUT(), "relIsoCheckMemLUT");
  printLUT(ptr1.idxSelMemPhiLUT(), "idxSelMemPhiLUT");
  printLUT(ptr1.idxSelMemEtaLUT(), "idxSelMemEtaLUT");
  //l1t::LUT* brlSingleMatchQualLUT();
  printLUT(ptr1.fwdPosSingleMatchQualLUT(), "fwdPosSingleMatchQualLUT");
  printLUT(ptr1.fwdNegSingleMatchQualLUT(), "fwdNegSingleMatchQualLUT");
  printLUT(ptr1.ovlPosSingleMatchQualLUT(), "ovlPosSingleMatchQualLUT");
  printLUT(ptr1.ovlNegSingleMatchQualLUT(), "ovlNegSingleMatchQualLUT");
  printLUT(ptr1.bOPosMatchQualLUT(), "bOPosMatchQualLUT");
  printLUT(ptr1.bONegMatchQualLUT(), "bONegMatchQualLUT");
  printLUT(ptr1.fOPosMatchQualLUT(), "fOPosMatchQualLUT");
  printLUT(ptr1.fONegMatchQualLUT(), "fONegMatchQualLUT");
  printLUT(ptr1.bPhiExtrapolationLUT(), "bPhiExtrapolationLUT");
  printLUT(ptr1.oPhiExtrapolationLUT(), "oPhiExtrapolationLUT");
  printLUT(ptr1.fPhiExtrapolationLUT(), "fPhiExtrapolationLUT");
  printLUT(ptr1.bEtaExtrapolationLUT(), "bEtaExtrapolationLUT");
  printLUT(ptr1.oEtaExtrapolationLUT(), "oEtaExtrapolationLUT");
  printLUT(ptr1.fEtaExtrapolationLUT(), "fEtaExtrapolationLUT");
  printLUT(ptr1.sortRankLUT(), "sortRankLUT");

  std::cout << "absIsoCheckMemLUTPath: " << ptr1.absIsoCheckMemLUTPath() << std::endl;
  std::cout << "relIsoCheckMemLUTPath: " << ptr1.relIsoCheckMemLUTPath() << std::endl;
  std::cout << "idxSelMemPhiLUTPath: " << ptr1.idxSelMemPhiLUTPath() << std::endl;
  std::cout << "idxSelMemEtaLUTPath: " << ptr1.idxSelMemEtaLUTPath() << std::endl;
  //std::string brlSingleMatchQualLUTPath() const    { return pnodes_[brlSingleMatchQual].sparams_.size() > spIdx::fname ? pnodes_[brlSingleMatchQual].sparams_[spIdx::fname] : ""; }
  std::cout << "fwdPosSingleMatchQualLUTPath: " << ptr1.fwdPosSingleMatchQualLUTPath() << std::endl;
  std::cout << "fwdNegSingleMatchQualLUTPath: " << ptr1.fwdNegSingleMatchQualLUTPath() << std::endl;
  std::cout << "ovlPosSingleMatchQualLUTPath: " << ptr1.ovlPosSingleMatchQualLUTPath() << std::endl;
  std::cout << "ovlNegSingleMatchQualLUTPath: " << ptr1.ovlNegSingleMatchQualLUTPath() << std::endl;
  std::cout << "bOPosMatchQualLUTPath: " << ptr1.bOPosMatchQualLUTPath() << std::endl;
  std::cout << "bONegMatchQualLUTPath: " << ptr1.bONegMatchQualLUTPath() << std::endl;
  std::cout << "fOPosMatchQualLUTPath: " << ptr1.fOPosMatchQualLUTPath() << std::endl;
  std::cout << "fONegMatchQualLUTPath: " << ptr1.fONegMatchQualLUTPath() << std::endl;
  std::cout << "bPhiExtrapolationLUTPath: " << ptr1.bPhiExtrapolationLUTPath() << std::endl;
  std::cout << "oPhiExtrapolationLUTPath: " << ptr1.oPhiExtrapolationLUTPath() << std::endl;
  std::cout << "fPhiExtrapolationLUTPath: " << ptr1.fPhiExtrapolationLUTPath() << std::endl;
  std::cout << "bEtaExtrapolationLUTPath: " << ptr1.bEtaExtrapolationLUTPath() << std::endl;
  std::cout << "oEtaExtrapolationLUTPath: " << ptr1.oEtaExtrapolationLUTPath() << std::endl;
  std::cout << "fEtaExtrapolationLUTPath: " << ptr1.fEtaExtrapolationLUTPath() << std::endl;
  std::cout << "sortRankLUTPath: " << ptr1.sortRankLUTPath() << std::endl;

  std::cout << "fwdPosSingleMatchQualLUTMaxDR: " << ptr1.fwdPosSingleMatchQualLUTMaxDR() << std::endl;
  std::cout << "fwdNegSingleMatchQualLUTMaxDR: " << ptr1.fwdNegSingleMatchQualLUTMaxDR() << std::endl;
  std::cout << "ovlPosSingleMatchQualLUTMaxDR: " << ptr1.ovlPosSingleMatchQualLUTMaxDR() << std::endl;
  std::cout << "ovlNegSingleMatchQualLUTMaxDR: " << ptr1.ovlNegSingleMatchQualLUTMaxDR() << std::endl;
  std::cout << "bOPosMatchQualLUTMaxDR: " << ptr1.bOPosMatchQualLUTMaxDR() << std::endl;
  std::cout << "bONegMatchQualLUTMaxDR: " << ptr1.bONegMatchQualLUTMaxDR() << std::endl;
  std::cout << "bOPosMatchQualLUTMaxDREtaFine: " << ptr1.bOPosMatchQualLUTMaxDREtaFine() << std::endl;
  std::cout << "bONegMatchQualLUTMaxDREtaFine: " << ptr1.bONegMatchQualLUTMaxDREtaFine() << std::endl;
  std::cout << "fOPosMatchQualLUTMaxDR: " << ptr1.fOPosMatchQualLUTMaxDR() << std::endl;
  std::cout << "fONegMatchQualLUTMaxDR: " << ptr1.fONegMatchQualLUTMaxDR() << std::endl;

  // Sort rank LUT factors for pT and quality
  std::cout << "sortRankLUTPtFactor: " << ptr1.sortRankLUTPtFactor() << std::endl;
  std::cout << "sortRankLUTQualFactor: " << ptr1.sortRankLUTQualFactor() << std::endl;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1TMuonGlobalParamsViewer);
