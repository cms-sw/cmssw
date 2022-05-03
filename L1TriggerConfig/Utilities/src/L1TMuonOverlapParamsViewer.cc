#include <iomanip>
#include <iostream>

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1TMuonOverlapParamsRcd.h"
#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/CondDB/interface/Session.h"

#include <iostream>
using namespace std;

class L1TMuonOverlapParamsViewer : public edm::EDAnalyzer {
private:
  bool printLayerMap;

public:
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  string hash(void *buf, size_t len) const;

  explicit L1TMuonOverlapParamsViewer(const edm::ParameterSet &pset) : edm::EDAnalyzer() {
    printLayerMap = pset.getUntrackedParameter<bool>("printLayerMap", false);
  }
  ~L1TMuonOverlapParamsViewer(void) override {}
};

#include "Utilities/OpenSSL/interface/openssl_init.h"
#include <cmath>
#include <iostream>
using namespace std;

string L1TMuonOverlapParamsViewer::hash(void *buf, size_t len) const {
  cms::openssl_init();
  EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
  const EVP_MD *md = EVP_get_digestbyname("SHA1");
  if (!EVP_DigestInit_ex(mdctx, md, nullptr))
    throw cms::Exception("L1TCaloParamsViewer::hash") << "SHA1 initialization error";

  if (!EVP_DigestUpdate(mdctx, buf, len))
    throw cms::Exception("L1TCaloParamsViewer::hash") << "SHA1 processing error";

  unsigned char hash[EVP_MAX_MD_SIZE];
  unsigned int md_len = 0;
  if (!EVP_DigestFinal_ex(mdctx, hash, &md_len))
    throw cms::Exception("L1TCaloParamsViewer::hash") << "SHA1 finalization error";

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

void L1TMuonOverlapParamsViewer::analyze(const edm::Event &iEvent, const edm::EventSetup &evSetup) {
  // Pull the config from the ES
  edm::ESHandle<L1TMuonOverlapParams> handle1;
  evSetup.get<L1TMuonOverlapParamsRcd>().get(handle1);
  std::shared_ptr<L1TMuonOverlapParams> ptr1(new L1TMuonOverlapParams(*(handle1.product())));

  cout << "Some fields in L1TMuonOverlapParamsParams: " << endl;

  cout << " fwVersion() = " << ptr1->fwVersion() << endl;
  cout << " nPdfAddrBits() = " << ptr1->nPdfAddrBits() << endl;
  cout << " nPdfValBits() = " << ptr1->nPdfValBits() << endl;
  cout << " nHitsPerLayer() = " << ptr1->nHitsPerLayer() << endl;
  cout << " nPhiBits() = " << ptr1->nPhiBits() << endl;
  cout << " nPhiBins() = " << ptr1->nPhiBins() << endl;
  cout << " nRefHits() = " << ptr1->nRefHits() << endl;
  cout << " nTestRefHits() = " << ptr1->nTestRefHits() << endl;
  cout << " nProcessors() = " << ptr1->nProcessors() << endl;
  cout << " nLogicRegions() = " << ptr1->nLogicRegions() << endl;
  cout << " nInputs() = " << ptr1->nInputs() << endl;
  cout << " nLayers() = " << ptr1->nLayers() << endl;
  cout << " nRefLayers() = " << ptr1->nRefLayers() << endl;
  cout << " nGoldenPatterns() = " << ptr1->nGoldenPatterns() << endl;

  const std::vector<int> *gp = ptr1->generalParams();
  cout << " number of general parameters: = " << gp->size() << endl;
  cout << "  ";
  for (auto a : *gp)
    cout << a << ", ";
  cout << endl;

  const std::vector<L1TMuonOverlapParams::LayerMapNode> *lm = ptr1->layerMap();
  if (!lm->empty()) {
    cout << " layerMap() =              [" << lm->size() << "] ";
    L1TMuonOverlapParams::LayerMapNode *lm_ = new L1TMuonOverlapParams::LayerMapNode[lm->size()];
    for (unsigned int i = 0; i < lm->size(); i++)
      lm_[i] = (*lm)[i];
    cout << hash(lm_, sizeof(L1TMuonOverlapParams::LayerMapNode) * (lm->size())) << endl;
    if (printLayerMap) {
      cout << sizeof(L1TMuonOverlapParams::LayerMapNode) << endl;
      for (unsigned int i = 0; i < lm->size(); i++) {
        cout << "  [" << i << "]: hwNumber = " << lm_[i].hwNumber << " logicNumber = " << lm_[i].logicNumber
             << " bendingLayer = " << lm_[i].bendingLayer << " connectedToLayer = " << lm_[i].connectedToLayer << " ("
             << flush;
        char *n = (char *)(&(lm_[i]));
        for (unsigned int j = 0; j < sizeof(L1TMuonOverlapParams::LayerMapNode); j++)
          cout << "0x" << hex << int(*(n + j)) << dec << ", ";
        cout << endl;
      }
      // hash( (void*)(&(lm_[i])), sizeof(L1TMuonOverlapParams::LayerMapNode) ) <<endl;
    }
    delete[] lm_;
  } else {
    cout << " layerMap() =              [0] " << endl;
  }

  const std::vector<L1TMuonOverlapParams::RefLayerMapNode> *rlm = ptr1->refLayerMap();
  if (!rlm->empty()) {
    cout << " refLayerMap() =           [" << rlm->size() << "] ";
    L1TMuonOverlapParams::RefLayerMapNode *rlm_ = new L1TMuonOverlapParams::RefLayerMapNode[rlm->size()];
    for (unsigned int i = 0; i < rlm->size(); i++)
      rlm_[i] = (*rlm)[i];
    cout << hash(rlm_, sizeof(L1TMuonOverlapParams::RefLayerMapNode) * (rlm->size())) << endl;
    delete[] rlm_;
  } else {
    cout << " refLayerMap() =           [0] " << endl;
  }

  const std::vector<L1TMuonOverlapParams::RefHitNode> *rhn = ptr1->refHitMap();
  if (!rhn->empty()) {
    cout << " refHitMap() =             [" << rhn->size() << "] ";
    L1TMuonOverlapParams::RefHitNode *rhn_ = new L1TMuonOverlapParams::RefHitNode[rhn->size()];
    for (unsigned int i = 0; i < rhn->size(); i++)
      rhn_[i] = (*rhn)[i];
    cout << hash(rhn_, sizeof(L1TMuonOverlapParams::RefHitNode) * (rhn->size())) << endl;
    delete[] rhn_;
  } else {
    cout << " refHitMap() =             [0] " << endl;
  }

  const std::vector<int> *gpsm = ptr1->globalPhiStartMap();
  if (!gpsm->empty()) {
    cout << " globalPhiStartMap() =     [" << gpsm->size() << "] ";
    int gpsm_[gpsm->size()];
    for (unsigned int i = 0; i < gpsm->size(); i++)
      gpsm_[i] = (*gpsm)[i];
    cout << hash(gpsm_, sizeof(int) * (gpsm->size())) << endl;
  } else {
    cout << " globalPhiStartMap() =     [0] " << endl;
  }

  const std::vector<L1TMuonOverlapParams::LayerInputNode> *lim = ptr1->layerInputMap();
  if (!lim->empty()) {
    cout << " layerInputMap() =         [" << lim->size() << "] ";
    L1TMuonOverlapParams::LayerInputNode *lim_ = new L1TMuonOverlapParams::LayerInputNode[lim->size()];
    for (unsigned int i = 0; i < lim->size(); i++)
      lim_[i] = (*lim)[i];
    cout << hash(lim_, sizeof(L1TMuonOverlapParams::LayerInputNode) * (lim->size())) << endl;
    delete[] lim_;
  } else {
    cout << " layerInputMap() =         [0] " << endl;
  }

  const std::vector<int> *css = ptr1->connectedSectorsStart();
  if (!css->empty()) {
    cout << " connectedSectorsStart() = [" << css->size() << "] ";
    int css_[css->size()];
    for (unsigned int i = 0; i < css->size(); i++)
      css_[i] = (*css)[i];
    cout << hash(css_, sizeof(int) * (css->size())) << endl;
  } else {
    cout << " connectedSectorsStart() = [0] " << endl;
  }

  const std::vector<int> *cse = ptr1->connectedSectorsEnd();
  if (!cse->empty()) {
    cout << " connectedSectorsEnd() = [" << cse->size() << "] ";
    int cse_[cse->size()];
    for (unsigned int i = 0; i < cse->size(); i++)
      cse_[i] = (*cse)[i];
    cout << hash(cse_, sizeof(int) * (cse->size())) << endl;
  } else {
    cout << " connectedSectorsEnd() = [0] " << endl;
  }

  const l1t::LUT *clut = ptr1->chargeLUT();
  if (clut->maxSize()) {
    cout << " chargeLUT =      [" << clut->maxSize() << "] ";
    int clut_[clut->maxSize()];
    for (unsigned int i = 0; i < clut->maxSize(); i++)
      clut_[i] = clut->data(i);
    cout << hash(clut_, sizeof(int) * (clut->maxSize())) << endl;
  } else {
    cout << " chargeLUT =      [0] " << endl;
  }

  const l1t::LUT *elut = ptr1->etaLUT();
  if (elut->maxSize()) {
    cout << " etaLUT =         [" << elut->maxSize() << "] ";
    int elut_[elut->maxSize()];
    for (unsigned int i = 0; i < elut->maxSize(); i++)
      elut_[i] = elut->data(i);
    cout << hash(elut_, sizeof(int) * (elut->maxSize())) << endl;
  } else {
    cout << " chargeLUT =      [0] " << endl;
  }

  const l1t::LUT *ptlut = ptr1->ptLUT();
  if (ptlut->maxSize()) {
    cout << " ptLUT =          [" << ptlut->maxSize() << "] " << flush;
    int ptlut_[ptlut->maxSize()];
    for (unsigned int i = 0; i < ptlut->maxSize(); i++)
      ptlut_[i] = ptlut->data(i);
    cout << hash(ptlut_, sizeof(int) * (ptlut->maxSize())) << endl;
  } else {
    cout << " ptLUT =          [0] " << endl;
  }

  const l1t::LUT *plut = ptr1->pdfLUT();
  if (plut->maxSize()) {
    cout << " pdfLUT =         [" << plut->maxSize() << "] " << flush;
    int plut_[plut->maxSize()];
    for (unsigned int i = 0; i < plut->maxSize(); i++)
      plut_[i] = plut->data(i);
    cout << hash(plut_, sizeof(int) * (plut->maxSize())) << endl;
  } else {
    cout << " pdfLUT =         [0] " << endl;
  }

  const l1t::LUT *mlut = ptr1->meanDistPhiLUT();
  if (mlut->maxSize()) {
    cout << " meanDistPhiLUT = [" << mlut->maxSize() << "] " << flush;
    int mlut_[mlut->maxSize()];
    for (unsigned int i = 0; i < mlut->maxSize(); i++)
      mlut_[i] = mlut->data(i);
    cout << hash(mlut_, sizeof(int) * (mlut->maxSize())) << endl;
  } else {
    cout << " meanDistPhiLUT = [0] " << endl;
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1TMuonOverlapParamsViewer);
