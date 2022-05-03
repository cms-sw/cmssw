#include <iomanip>
#include <iostream>

#include "FWCore/Framework/interface/EDAnalyzer.h"

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

class L1TMuonGlobalParamsViewer : public edm::EDAnalyzer {
private:
  //    bool printLayerMap;
  std::string hash(void* buf, size_t len) const;
  void printLUT(l1t::LUT* lut, const char* name) const;

public:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  //    string hash(void *buf, size_t len) const ;

  explicit L1TMuonGlobalParamsViewer(const edm::ParameterSet& pset) : edm::EDAnalyzer() {
    //       printLayerMap   = pset.getUntrackedParameter<bool>("printLayerMap",  false);
  }
  ~L1TMuonGlobalParamsViewer(void) override {}
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
  edm::ESHandle<L1TMuonGlobalParams> handle1;
  evSetup.get<L1TMuonGlobalParamsRcd>().get(handle1);
  std::shared_ptr<L1TMuonGlobalParams> ptr1(new L1TMuonGlobalParams(*(handle1.product())));

  //    cout<<"Some fields in L1TMuonGlobalParams: "<<endl;

  //    ((L1TMuonGlobalParamsHelper*)ptr1.get())->print(cout);

  printLUT(ptr1.get()->absIsoCheckMemLUT(), "absIsoCheckMemLUT");
  printLUT(ptr1.get()->absIsoCheckMemLUT(), "absIsoCheckMemLUT");
  printLUT(ptr1.get()->relIsoCheckMemLUT(), "relIsoCheckMemLUT");
  printLUT(ptr1.get()->idxSelMemPhiLUT(), "idxSelMemPhiLUT");
  printLUT(ptr1.get()->idxSelMemEtaLUT(), "idxSelMemEtaLUT");
  //l1t::LUT* brlSingleMatchQualLUT();
  printLUT(ptr1.get()->fwdPosSingleMatchQualLUT(), "fwdPosSingleMatchQualLUT");
  printLUT(ptr1.get()->fwdNegSingleMatchQualLUT(), "fwdNegSingleMatchQualLUT");
  printLUT(ptr1.get()->ovlPosSingleMatchQualLUT(), "ovlPosSingleMatchQualLUT");
  printLUT(ptr1.get()->ovlNegSingleMatchQualLUT(), "ovlNegSingleMatchQualLUT");
  printLUT(ptr1.get()->bOPosMatchQualLUT(), "bOPosMatchQualLUT");
  printLUT(ptr1.get()->bONegMatchQualLUT(), "bONegMatchQualLUT");
  printLUT(ptr1.get()->fOPosMatchQualLUT(), "fOPosMatchQualLUT");
  printLUT(ptr1.get()->fONegMatchQualLUT(), "fONegMatchQualLUT");
  printLUT(ptr1.get()->bPhiExtrapolationLUT(), "bPhiExtrapolationLUT");
  printLUT(ptr1.get()->oPhiExtrapolationLUT(), "oPhiExtrapolationLUT");
  printLUT(ptr1.get()->fPhiExtrapolationLUT(), "fPhiExtrapolationLUT");
  printLUT(ptr1.get()->bEtaExtrapolationLUT(), "bEtaExtrapolationLUT");
  printLUT(ptr1.get()->oEtaExtrapolationLUT(), "oEtaExtrapolationLUT");
  printLUT(ptr1.get()->fEtaExtrapolationLUT(), "fEtaExtrapolationLUT");
  printLUT(ptr1.get()->sortRankLUT(), "sortRankLUT");

  std::cout << "absIsoCheckMemLUTPath: " << ptr1.get()->absIsoCheckMemLUTPath() << std::endl;
  std::cout << "relIsoCheckMemLUTPath: " << ptr1.get()->relIsoCheckMemLUTPath() << std::endl;
  std::cout << "idxSelMemPhiLUTPath: " << ptr1.get()->idxSelMemPhiLUTPath() << std::endl;
  std::cout << "idxSelMemEtaLUTPath: " << ptr1.get()->idxSelMemEtaLUTPath() << std::endl;
  //std::string brlSingleMatchQualLUTPath() const    { return pnodes_[brlSingleMatchQual].sparams_.size() > spIdx::fname ? pnodes_[brlSingleMatchQual].sparams_[spIdx::fname] : ""; }
  std::cout << "fwdPosSingleMatchQualLUTPath: " << ptr1.get()->fwdPosSingleMatchQualLUTPath() << std::endl;
  std::cout << "fwdNegSingleMatchQualLUTPath: " << ptr1.get()->fwdNegSingleMatchQualLUTPath() << std::endl;
  std::cout << "ovlPosSingleMatchQualLUTPath: " << ptr1.get()->ovlPosSingleMatchQualLUTPath() << std::endl;
  std::cout << "ovlNegSingleMatchQualLUTPath: " << ptr1.get()->ovlNegSingleMatchQualLUTPath() << std::endl;
  std::cout << "bOPosMatchQualLUTPath: " << ptr1.get()->bOPosMatchQualLUTPath() << std::endl;
  std::cout << "bONegMatchQualLUTPath: " << ptr1.get()->bONegMatchQualLUTPath() << std::endl;
  std::cout << "fOPosMatchQualLUTPath: " << ptr1.get()->fOPosMatchQualLUTPath() << std::endl;
  std::cout << "fONegMatchQualLUTPath: " << ptr1.get()->fONegMatchQualLUTPath() << std::endl;
  std::cout << "bPhiExtrapolationLUTPath: " << ptr1.get()->bPhiExtrapolationLUTPath() << std::endl;
  std::cout << "oPhiExtrapolationLUTPath: " << ptr1.get()->oPhiExtrapolationLUTPath() << std::endl;
  std::cout << "fPhiExtrapolationLUTPath: " << ptr1.get()->fPhiExtrapolationLUTPath() << std::endl;
  std::cout << "bEtaExtrapolationLUTPath: " << ptr1.get()->bEtaExtrapolationLUTPath() << std::endl;
  std::cout << "oEtaExtrapolationLUTPath: " << ptr1.get()->oEtaExtrapolationLUTPath() << std::endl;
  std::cout << "fEtaExtrapolationLUTPath: " << ptr1.get()->fEtaExtrapolationLUTPath() << std::endl;
  std::cout << "sortRankLUTPath: " << ptr1.get()->sortRankLUTPath() << std::endl;

  std::cout << "fwdPosSingleMatchQualLUTMaxDR: " << ptr1.get()->fwdPosSingleMatchQualLUTMaxDR() << std::endl;
  std::cout << "fwdNegSingleMatchQualLUTMaxDR: " << ptr1.get()->fwdNegSingleMatchQualLUTMaxDR() << std::endl;
  std::cout << "ovlPosSingleMatchQualLUTMaxDR: " << ptr1.get()->ovlPosSingleMatchQualLUTMaxDR() << std::endl;
  std::cout << "ovlNegSingleMatchQualLUTMaxDR: " << ptr1.get()->ovlNegSingleMatchQualLUTMaxDR() << std::endl;
  std::cout << "bOPosMatchQualLUTMaxDR: " << ptr1.get()->bOPosMatchQualLUTMaxDR() << std::endl;
  std::cout << "bONegMatchQualLUTMaxDR: " << ptr1.get()->bONegMatchQualLUTMaxDR() << std::endl;
  std::cout << "bOPosMatchQualLUTMaxDREtaFine: " << ptr1.get()->bOPosMatchQualLUTMaxDREtaFine() << std::endl;
  std::cout << "bONegMatchQualLUTMaxDREtaFine: " << ptr1.get()->bONegMatchQualLUTMaxDREtaFine() << std::endl;
  std::cout << "fOPosMatchQualLUTMaxDR: " << ptr1.get()->fOPosMatchQualLUTMaxDR() << std::endl;
  std::cout << "fONegMatchQualLUTMaxDR: " << ptr1.get()->fONegMatchQualLUTMaxDR() << std::endl;

  // Sort rank LUT factors for pT and quality
  std::cout << "sortRankLUTPtFactor: " << ptr1.get()->sortRankLUTPtFactor() << std::endl;
  std::cout << "sortRankLUTQualFactor: " << ptr1.get()->sortRankLUTQualFactor() << std::endl;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1TMuonGlobalParamsViewer);
