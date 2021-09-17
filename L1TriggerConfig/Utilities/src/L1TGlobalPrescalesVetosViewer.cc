#include <iomanip>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1TGlobalPrescalesVetosRcd.h"
#include "CondFormats/L1TObjects/interface/L1TGlobalPrescalesVetos.h"

class L1TGlobalPrescalesVetosViewer : public edm::EDAnalyzer {
private:
  int32_t prescale_table_verbosity;
  int32_t bxmask_map_verbosity;
  int32_t veto_verbosity;

  std::string hash(void* buf, size_t len) const;

public:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  explicit L1TGlobalPrescalesVetosViewer(const edm::ParameterSet& pset) : edm::EDAnalyzer() {
    prescale_table_verbosity = pset.getUntrackedParameter<int32_t>("prescale_table_verbosity", 0);
    bxmask_map_verbosity = pset.getUntrackedParameter<int32_t>("bxmask_map_verbosity", 0);
    veto_verbosity = pset.getUntrackedParameter<int32_t>("veto_verbosity", 0);
  }

  ~L1TGlobalPrescalesVetosViewer(void) override {}
};

#include <openssl/sha.h>
#include <cmath>
#include <iostream>
using namespace std;

std::string L1TGlobalPrescalesVetosViewer::hash(void* buf, size_t len) const {
  char tmp[SHA_DIGEST_LENGTH * 2 + 1];
  bzero(tmp, sizeof(tmp));
  SHA_CTX ctx;
  if (!SHA1_Init(&ctx))
    throw cms::Exception("L1TGlobalPrescalesVetosViewer::hash") << "SHA1 initialization error";

  if (!SHA1_Update(&ctx, buf, len))
    throw cms::Exception("L1TGlobalPrescalesVetosViewer::hash") << "SHA1 processing error";

  unsigned char hash[SHA_DIGEST_LENGTH];
  if (!SHA1_Final(hash, &ctx))
    throw cms::Exception("L1TGlobalPrescalesVetosViewer::hash") << "SHA1 finalization error";

  // re-write bytes in hex
  for (unsigned int i = 0; i < 20; i++)
    ::sprintf(&tmp[i * 2], "%02x", hash[i]);

  tmp[20 * 2] = 0;
  return std::string(tmp);
}

void L1TGlobalPrescalesVetosViewer::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  edm::ESHandle<L1TGlobalPrescalesVetos> handle1;
  evSetup.get<L1TGlobalPrescalesVetosRcd>().get(handle1);
  std::shared_ptr<L1TGlobalPrescalesVetos> ptr(new L1TGlobalPrescalesVetos(*(handle1.product())));

  edm::LogInfo("") << "L1TGlobalPrescalesVetosViewer:";

  cout << endl;
  cout << "  version_        = " << ptr->version_ << endl;
  cout << "  bxmask_default_ = " << ptr->bxmask_default_ << endl;

  size_t len_prescale_table_ = 0;
  for (size_t col = 0; col < ptr->prescale_table_.size(); col++) {
    size_t nRows = (ptr->prescale_table_)[col].size();
    len_prescale_table_ += nRows;
    if (prescale_table_verbosity > 0) {
      int column[nRows];
      for (size_t row = 0; row < nRows; row++)
        column[row] = (ptr->prescale_table_)[col][row];
      cout << "  prescale_table_[" << col << "][" << nRows << "] = ";
      if (nRows)
        cout << hash(column, sizeof(int) * nRows) << endl;
      else
        cout << 0 << endl;
    }
  }
  int prescale_table_[len_prescale_table_];
  for (size_t col = 0, pos = 0; col < ptr->prescale_table_.size(); col++) {
    size_t nRows = (ptr->prescale_table_)[col].size();
    for (size_t row = 0; row < nRows; row++, pos++)
      prescale_table_[pos] = (ptr->prescale_table_)[col][row];
  }
  cout << "  prescale_table_[[" << len_prescale_table_ << "]] = ";
  if (len_prescale_table_)
    cout << hash(prescale_table_, sizeof(int) * len_prescale_table_) << endl;
  else
    cout << 0 << endl;

  if (prescale_table_verbosity > 1) {
    cout << endl << " Detailed view on the prescales * masks: " << endl;
    for (size_t col = 0; col < ptr->prescale_table_.size(); col++)
      cout << setw(8) << " Index " << col;
    cout << endl;
    size_t nRows = (ptr->prescale_table_)[0].size();
    for (size_t row = 0; row < nRows; row++) {
      for (size_t col = 0; col < ptr->prescale_table_.size(); col++)
        cout << setw(8) << (ptr->prescale_table_)[col][row];
      cout << endl;
    }
    cout << endl;
  }

  size_t len_bxmask_map_ = 0;
  for (std::map<int, std::vector<int> >::const_iterator it = (ptr->bxmask_map_).begin(); it != (ptr->bxmask_map_).end();
       it++) {
    len_bxmask_map_ += it->second.size();
    if (bxmask_map_verbosity == 1) {
      int masks[it->second.size()];
      for (size_t i = 0; i < it->second.size(); i++)
        masks[i] = it->second[i];
      cout << "  bxmask_map_[" << it->first << "][" << it->second.size() << "] = ";
      if (!it->second.empty())
        cout << hash(masks, sizeof(int) * it->second.size()) << endl;
      else
        cout << 0 << endl;
    }
    if (bxmask_map_verbosity > 1) {
      cout << "  bxmask_map_[" << it->first << "][" << it->second.size() << "] = ";
      for (size_t algo = 0; algo < it->second.size(); algo++)
        cout << it->second[algo] << ", ";
      cout << endl;
    }
  }
  int bxmask_map_[len_bxmask_map_];
  size_t pos = 0;
  for (std::map<int, std::vector<int> >::const_iterator it = (ptr->bxmask_map_).begin(); it != (ptr->bxmask_map_).end();
       it++) {
    for (size_t i = 0; i < it->second.size(); i++, pos++)
      bxmask_map_[pos] = it->second[i];
  }
  cout << "  bxmask_map_[[" << len_bxmask_map_ << "]]        = ";
  if (len_bxmask_map_)
    cout << hash(bxmask_map_, sizeof(int) * len_bxmask_map_) << endl;
  else
    cout << 0 << endl;

  int veto_[(ptr->veto_).size()];
  bool veto_allZeros = true;
  for (size_t i = 0; i < (ptr->veto_).size(); i++) {
    veto_[i] = (ptr->veto_)[i];
    if (veto_[i])
      veto_allZeros = false;
  }
  cout << "  veto_[" << (ptr->veto_).size() << "]              = ";
  if (veto_verbosity == 0) {
    if (!(ptr->veto_).empty()) {
      cout << hash(veto_, sizeof(int) * (ptr->veto_).size());
      if (veto_allZeros)
        cout << " (all zeros)" << endl;
      else
        cout << endl;
    } else
      cout << 0 << endl;
  } else
    for (size_t i = 0; i < (ptr->veto_).size(); i++)
      cout << veto_[i] << endl;

  int exp_ints_[(ptr->exp_ints_).size()];
  for (size_t i = 0; i < (ptr->exp_ints_).size(); i++)
    exp_ints_[i] = (ptr->exp_ints_)[i];
  cout << "  exp_ints_[" << (ptr->exp_ints_).size() << "]            = ";
  if (!(ptr->exp_ints_).empty())
    cout << hash(exp_ints_, sizeof(int) * (ptr->exp_ints_).size()) << endl;
  else
    cout << 0 << endl;

  int exp_doubles_[(ptr->exp_doubles_).size()];
  for (size_t i = 0; i < (ptr->exp_doubles_).size(); i++)
    exp_ints_[i] = (ptr->exp_doubles_)[i];
  cout << "  exp_doubles_[" << (ptr->exp_doubles_).size() << "]         = ";
  if (!(ptr->exp_doubles_).empty())
    cout << hash(exp_doubles_, sizeof(int) * (ptr->exp_doubles_).size()) << endl;
  else
    cout << 0 << endl;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1TGlobalPrescalesVetosViewer);
