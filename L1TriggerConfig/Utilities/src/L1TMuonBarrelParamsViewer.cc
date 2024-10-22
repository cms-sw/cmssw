#include <iomanip>
#include <iostream>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "L1Trigger/L1TMuonBarrel/interface/L1TMuonBarrelParamsHelper.h"
#include "CondFormats/DataRecord/interface/L1TMuonBarrelParamsRcd.h"
#include "CondFormats/L1TObjects/interface/L1TMuonBarrelParams.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/CondDB/interface/Session.h"

#include <iostream>
using namespace std;

class L1TMuonBarrelParamsViewer : public edm::one::EDAnalyzer<> {
private:
  std::string hash(void *buf, size_t len) const;
  edm::ESGetToken<L1TMuonBarrelParams, L1TMuonBarrelParamsRcd> token_;
  bool printPtaThreshold;

public:
  void analyze(const edm::Event &, const edm::EventSetup &) override;

  explicit L1TMuonBarrelParamsViewer(const edm::ParameterSet &) : token_{esConsumes()} { printPtaThreshold = false; }
};

#include "Utilities/OpenSSL/interface/openssl_init.h"
#include <cmath>
#include <iostream>
using namespace std;

std::string L1TMuonBarrelParamsViewer::hash(void *buf, size_t len) const {
  cms::openssl_init();
  EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
  const EVP_MD *md = EVP_get_digestbyname("SHA1");
  if (!EVP_DigestInit_ex(mdctx, md, nullptr))
    throw cms::Exception("L1TMuonBarrelParamsViewer::hash") << "SHA1 initialization error";

  if (!EVP_DigestUpdate(mdctx, buf, len))
    throw cms::Exception("L1TMuonBarrelParamsViewer::hash") << "SHA1 processing error";

  unsigned char hash[EVP_MAX_MD_SIZE];
  unsigned int md_len = 0;
  if (!EVP_DigestFinal_ex(mdctx, hash, &md_len))
    throw cms::Exception("L1TMuonBarrelParamsViewer::hash") << "SHA1 finalization error";

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

void L1TMuonBarrelParamsViewer::analyze(const edm::Event &iEvent, const edm::EventSetup &evSetup) {
  L1TMuonBarrelParams const &ptr = evSetup.getData(token_);

  L1TMuonBarrelParamsHelper *ptr1 = (L1TMuonBarrelParamsHelper *)&ptr;

  cout << "AssLUTPath: " << ptr1->AssLUTPath() << endl;

  // typedef std::map<short, short, std::less<short> > LUT;

  size_t k = 0;  // avoid using l
  for (const auto &lut : ptr1->pta_lut()) {
    if (!lut.empty()) {
      cout << "  pta_lut[" << setw(2) << k << "]=          [" << lut.size() << "] " << flush;
      int lut_[lut.size()], i = 0;
      for (const pair<const short, short> &p : lut)
        lut_[i++] = p.first * 0xFFFF + p.second;
      cout << hash(lut_, sizeof(int) * lut.size()) << endl;
    } else {
      cout << "  pta_lut[" << setw(2) << k << "]=          [0] " << endl;
    }
    k++;
  }

  cout << "  pta_threshold=        [" << ptr1->pta_threshold().size() << "] " << flush;
  int pta_threshold[ptr1->pta_threshold().size()];
  for (unsigned int i = 0; i < ptr1->pta_threshold().size(); i++) {
    pta_threshold[i] = ptr1->pta_threshold()[i];
    if (printPtaThreshold)
      cout << "   " << pta_threshold[i] << endl;
  }
  if (!ptr1->pta_threshold().empty())
    cout << hash(pta_threshold, sizeof(int) * ptr1->pta_threshold().size()) << endl;
  else
    cout << endl;

  k = 0;
  for (const auto &lut : ptr1->pta_lut()) {
    if (!lut.empty()) {
      cout << "  phi_lut[" << k << "]=           [" << lut.size() << "] " << flush;
      int lut_[lut.size()], i = 0;
      for (const pair<const short, short> &p : lut)
        lut_[i++] = p.first * 0xFFFF + p.second;
      cout << hash(lut_, sizeof(int) * lut.size()) << endl;
    } else {
      cout << "  phi_lut[" << k << "]=           [0] " << endl;
    }
    k++;
  }

  k = 0;
  for (const auto &lu : ptr1->ext_lut()) {
    const auto &lut = lu.low;
    if (!lut.empty()) {
      cout << "  ext_lut_low[" << setw(2) << k << "]=      [" << lut.size() << "] " << flush;
      int lut_[lut.size()], i = 0;
      for (const pair<const short, short> &p : lut)
        lut_[i++] = p.first * 0xFFFF + p.second;
      cout << hash(lut_, sizeof(int) * lut.size()) << endl;
    } else {
      cout << "  ext_lut_low[" << setw(2) << k << "]=      [0] " << endl;
    }
    k++;
  }

  k = 0;
  for (const auto &lu : ptr1->ext_lut()) {
    const auto &lut = lu.high;
    if (!lut.empty()) {
      cout << "  ext_lut_high[" << setw(2) << k << "]=     [" << lut.size() << "] " << flush;
      int lut_[lut.size()], i = 0;
      for (const pair<const short, short> &p : lut)
        lut_[i++] = p.first * 0xFFFF + p.second;
      cout << hash(lut_, sizeof(int) * lut.size()) << endl;
    } else {
      cout << "  ext_lut_high[" << setw(2) << k << "]=     [0] " << endl;
    }
    k++;
  }

  // typedef std::map< LUTID, LUTCONT > qpLUT;
  for (const auto &item : ptr1->qp_lut()) {
    cout << "  qp_lut[" << item.first.first << "," << item.first.second << "]= " << item.second.first << ", ["
         << item.second.second.size() << "] " << flush;
    if (!item.second.second.empty()) {
      int lut_[item.second.second.size()];
      for (size_t i = 0; i < item.second.second.size(); i++)
        lut_[i] = item.second.second[i];
      cout << hash(lut_, sizeof(int) * item.second.second.size()) << endl;
    } else {
      cout << endl;
    }
  }

  // typedef std::map<short, L1MuDTEtaPattern, std::less<short> > etaLUT;
  for (const pair<const short, L1MuDTEtaPattern> &item : ptr1->eta_lut())
    cout << "  eta_lut[" << item.first << "]= " << endl << item.second << endl;

  cout << "PT_Assignment_nbits_Phi=   " << ptr1->get_PT_Assignment_nbits_Phi() << endl;
  cout << "PT_Assignment_nbits_PhiB=  " << ptr1->get_PT_Assignment_nbits_PhiB() << endl;
  cout << "PHI_Assignment_nbits_Phi=  " << ptr1->get_PHI_Assignment_nbits_Phi() << endl;
  cout << "PHI_Assignment_nbits_PhiB= " << ptr1->get_PHI_Assignment_nbits_PhiB() << endl;
  cout << "Extrapolation_nbits_Phi=   " << ptr1->get_Extrapolation_nbits_Phi() << endl;
  cout << "Extrapolation_nbits_PhiB=  " << ptr1->get_Extrapolation_nbits_PhiB() << endl;
  cout << "BX_min=                    " << ptr1->get_BX_min() << endl;
  cout << "BX_max=                    " << ptr1->get_BX_max() << endl;
  cout << "Extrapolation_Filter=      " << ptr1->get_Extrapolation_Filter() << endl;
  cout << "OutOfTime_Filter_Window=   " << ptr1->get_OutOfTime_Filter_Window() << endl;

  cout << boolalpha;
  cout << "OutOfTime_Filter=          " << ptr1->get_OutOfTime_Filter() << endl;
  cout << "Open_LUTs=                 " << ptr1->get_Open_LUTs() << endl;
  cout << "EtaTrackFinder=            " << ptr1->get_EtaTrackFinder() << endl;
  cout << "Extrapolation_21=          " << ptr1->get_Extrapolation_21() << endl;
  cout << "DisableNewAlgo=            " << ptr1->get_DisableNewAlgo() << endl;
  cout << noboolalpha;

  // FW version
  cout << "fwVersion=                 " << ptr1->fwVersion() << endl;
  cout << "version=                   " << ptr1->version_ << endl;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1TMuonBarrelParamsViewer);
