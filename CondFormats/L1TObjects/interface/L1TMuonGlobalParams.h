///
/// \class L1TMuonGlobalParams
///
/// Description: Placeholder for MicroGMT parameters
///
/// Implementation:
///
/// \author: Thomas Reis
///

#ifndef L1TGMTParams_h
#define L1TGMTParams_h

#include <memory>
#include <iostream>
#include <vector>

#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/L1TObjects/interface/LUT.h"

class L1TMuonGlobalParams {

public:
  enum { Version = 1 };

  class Node {
  public:
    std::string type_;
    unsigned version_;
    l1t::LUT LUT_;
    std::vector<double> dparams_;
    std::vector<unsigned> uparams_;
    std::vector<int> iparams_;
    std::vector<std::string> sparams_;
    Node(){ type_="unspecified"; version_=0; }
    COND_SERIALIZABLE;
  };

  enum {absIsoCheckMem=0,
        relIsoCheckMem=1,
        idxSelMemPhi=2,
        idxSelMemEta=3,
        brlSingleMatchQual=4,
        fwdPosSingleMatchQual=5,
        fwdNegSingleMatchQual=6,
        ovlPosSingleMatchQual=7,
        ovlNegSingleMatchQual=8,
        bOPosMatchQual=9,
        bONegMatchQual=10,
        fOPosMatchQual=11,
        fONegMatchQual=12,
        bPhiExtrapolation=13,
        oPhiExtrapolation=14,
        fPhiExtrapolation=15,
        bEtaExtrapolation=16,
        oEtaExtrapolation=17,
        fEtaExtrapolation=18,
        sortRank=19,
	NUM_CALOPARAMNODES=20
  };

  L1TMuonGlobalParams() { version_=Version; pnodes_.resize(NUM_CALOPARAMNODES); }
  ~L1TMuonGlobalParams() {}

  // FW version
  unsigned fwVersion() const { return fwVersion_; }
  void setFwVersion(unsigned fwVersion) { fwVersion_ = fwVersion; }

  // LUTs
  l1t::LUT* absIsoCheckMemLUT()        { return &pnodes_[absIsoCheckMem].LUT_; }
  l1t::LUT* relIsoCheckMemLUT()        { return &pnodes_[relIsoCheckMem].LUT_; }
  l1t::LUT* idxSelMemPhiLUT()          { return &pnodes_[idxSelMemPhi].LUT_; }
  l1t::LUT* idxSelMemEtaLUT()          { return &pnodes_[idxSelMemEta].LUT_; }
  l1t::LUT* brlSingleMatchQualLUT()    { return &pnodes_[brlSingleMatchQual].LUT_; }
  l1t::LUT* fwdPosSingleMatchQualLUT() { return &pnodes_[fwdPosSingleMatchQual].LUT_; }
  l1t::LUT* fwdNegSingleMatchQualLUT() { return &pnodes_[fwdNegSingleMatchQual].LUT_; }
  l1t::LUT* ovlPosSingleMatchQualLUT() { return &pnodes_[ovlPosSingleMatchQual].LUT_; }
  l1t::LUT* ovlNegSingleMatchQualLUT() { return &pnodes_[ovlNegSingleMatchQual].LUT_; }
  l1t::LUT* bOPosMatchQualLUT()        { return &pnodes_[bOPosMatchQual].LUT_; }
  l1t::LUT* bONegMatchQualLUT()        { return &pnodes_[bONegMatchQual].LUT_; }
  l1t::LUT* fOPosMatchQualLUT()        { return &pnodes_[fOPosMatchQual].LUT_; }
  l1t::LUT* fONegMatchQualLUT()        { return &pnodes_[fONegMatchQual].LUT_; }
  l1t::LUT* bPhiExtrapolationLUT()     { return &pnodes_[bPhiExtrapolation].LUT_; }
  l1t::LUT* oPhiExtrapolationLUT()     { return &pnodes_[oPhiExtrapolation].LUT_; }
  l1t::LUT* fPhiExtrapolationLUT()     { return &pnodes_[fPhiExtrapolation].LUT_; }
  l1t::LUT* bEtaExtrapolationLUT()     { return &pnodes_[bEtaExtrapolation].LUT_; }
  l1t::LUT* oEtaExtrapolationLUT()     { return &pnodes_[oEtaExtrapolation].LUT_; }
  l1t::LUT* fEtaExtrapolationLUT()     { return &pnodes_[fEtaExtrapolation].LUT_; }
  l1t::LUT* sortRankLUT()              { return &pnodes_[sortRank].LUT_; }
  void setAbsIsoCheckMemLUT        (const l1t::LUT & lut) { pnodes_[absIsoCheckMem].type_ = "LUT"; pnodes_[absIsoCheckMem].LUT_ = lut; }
  void setRelIsoCheckMemLUT        (const l1t::LUT & lut) { pnodes_[relIsoCheckMem].type_ = "LUT"; pnodes_[relIsoCheckMem].LUT_ = lut; }
  void setIdxSelMemPhiLUT          (const l1t::LUT & lut) { pnodes_[idxSelMemPhi].type_ = "LUT"; pnodes_[idxSelMemPhi].LUT_ = lut; }
  void setIdxSelMemEtaLUT          (const l1t::LUT & lut) { pnodes_[idxSelMemEta].type_ = "LUT"; pnodes_[idxSelMemEta].LUT_ = lut; }
  void setBrlSingleMatchQualLUT    (const l1t::LUT & lut) { pnodes_[brlSingleMatchQual].type_ = "LUT"; pnodes_[brlSingleMatchQual].LUT_ = lut; }
  void setFwdPosSingleMatchQualLUT (const l1t::LUT & lut) { pnodes_[fwdPosSingleMatchQual].type_ = "LUT"; pnodes_[fwdPosSingleMatchQual].LUT_ = lut; }
  void setFwdNegSingleMatchQualLUT (const l1t::LUT & lut) { pnodes_[fwdNegSingleMatchQual].type_ = "LUT"; pnodes_[fwdNegSingleMatchQual].LUT_ = lut; }
  void setOvlPosSingleMatchQualLUT (const l1t::LUT & lut) { pnodes_[ovlPosSingleMatchQual].type_ = "LUT"; pnodes_[ovlPosSingleMatchQual].LUT_ = lut; }
  void setOvlNegSingleMatchQualLUT (const l1t::LUT & lut) { pnodes_[ovlNegSingleMatchQual].type_ = "LUT"; pnodes_[ovlNegSingleMatchQual].LUT_ = lut; }
  void setBOPosMatchQualLUT        (const l1t::LUT & lut) { pnodes_[bOPosMatchQual].type_ = "LUT"; pnodes_[bOPosMatchQual].LUT_ = lut; }
  void setBONegMatchQualLUT        (const l1t::LUT & lut) { pnodes_[bONegMatchQual].type_ = "LUT"; pnodes_[bONegMatchQual].LUT_ = lut; }
  void setFOPosMatchQualLUT        (const l1t::LUT & lut) { pnodes_[fOPosMatchQual].type_ = "LUT"; pnodes_[fOPosMatchQual].LUT_ = lut; }
  void setFONegMatchQualLUT        (const l1t::LUT & lut) { pnodes_[fONegMatchQual].type_ = "LUT"; pnodes_[fONegMatchQual].LUT_ = lut; }
  void setBPhiExtrapolationLUT     (const l1t::LUT & lut) { pnodes_[bPhiExtrapolation].type_ = "LUT"; pnodes_[bPhiExtrapolation].LUT_ = lut; }
  void setOPhiExtrapolationLUT     (const l1t::LUT & lut) { pnodes_[oPhiExtrapolation].type_ = "LUT"; pnodes_[oPhiExtrapolation].LUT_ = lut; }
  void setFPhiExtrapolationLUT     (const l1t::LUT & lut) { pnodes_[fPhiExtrapolation].type_ = "LUT"; pnodes_[fPhiExtrapolation].LUT_ = lut; }
  void setBEtaExtrapolationLUT     (const l1t::LUT & lut) { pnodes_[bEtaExtrapolation].type_ = "LUT"; pnodes_[bEtaExtrapolation].LUT_ = lut; }
  void setOEtaExtrapolationLUT     (const l1t::LUT & lut) { pnodes_[oEtaExtrapolation].type_ = "LUT"; pnodes_[oEtaExtrapolation].LUT_ = lut; }
  void setFEtaExtrapolationLUT     (const l1t::LUT & lut) { pnodes_[fEtaExtrapolation].type_ = "LUT"; pnodes_[fEtaExtrapolation].LUT_ = lut; }
  void setSortRankLUT              (const l1t::LUT & lut) { pnodes_[sortRank].type_ = "LUT"; pnodes_[sortRank].LUT_ = lut; }

  // LUT paths
  std::string absIsoCheckMemLUTPath() const        { return pnodes_[absIsoCheckMem].sparams_.size() > 0 ? pnodes_[absIsoCheckMem].sparams_[0] : ""; }
  std::string relIsoCheckMemLUTPath() const        { return pnodes_[relIsoCheckMem].sparams_.size() > 0 ? pnodes_[relIsoCheckMem].sparams_[0] : ""; }
  std::string idxSelMemPhiLUTPath() const          { return pnodes_[idxSelMemPhi].sparams_.size() > 0 ? pnodes_[idxSelMemPhi].sparams_[0] : ""; }
  std::string idxSelMemEtaLUTPath() const          { return pnodes_[idxSelMemEta].sparams_.size() > 0 ? pnodes_[idxSelMemEta].sparams_[0] : ""; }
  std::string brlSingleMatchQualLUTPath() const    { return pnodes_[brlSingleMatchQual].sparams_.size() > 0 ? pnodes_[brlSingleMatchQual].sparams_[0] : ""; }
  std::string fwdPosSingleMatchQualLUTPath() const { return pnodes_[fwdPosSingleMatchQual].sparams_.size() > 0 ? pnodes_[fwdPosSingleMatchQual].sparams_[0] : ""; }
  std::string fwdNegSingleMatchQualLUTPath() const { return pnodes_[fwdNegSingleMatchQual].sparams_.size() > 0 ? pnodes_[fwdNegSingleMatchQual].sparams_[0] : ""; }
  std::string ovlPosSingleMatchQualLUTPath() const { return pnodes_[ovlPosSingleMatchQual].sparams_.size() > 0 ? pnodes_[ovlPosSingleMatchQual].sparams_[0] : ""; }
  std::string ovlNegSingleMatchQualLUTPath() const { return pnodes_[ovlNegSingleMatchQual].sparams_.size() > 0 ? pnodes_[ovlNegSingleMatchQual].sparams_[0] : ""; }
  std::string bOPosMatchQualLUTPath() const        { return pnodes_[bOPosMatchQual].sparams_.size() > 0 ? pnodes_[bOPosMatchQual].sparams_[0] : ""; }
  std::string bONegMatchQualLUTPath() const        { return pnodes_[bONegMatchQual].sparams_.size() > 0 ? pnodes_[bONegMatchQual].sparams_[0] : ""; }
  std::string fOPosMatchQualLUTPath() const        { return pnodes_[fOPosMatchQual].sparams_.size() > 0 ? pnodes_[fOPosMatchQual].sparams_[0] : ""; }
  std::string fONegMatchQualLUTPath() const        { return pnodes_[fONegMatchQual].sparams_.size() > 0 ? pnodes_[fONegMatchQual].sparams_[0] : ""; }
  std::string bPhiExtrapolationLUTPath() const     { return pnodes_[bPhiExtrapolation].sparams_.size() > 0 ? pnodes_[bPhiExtrapolation].sparams_[0] : ""; }
  std::string oPhiExtrapolationLUTPath() const     { return pnodes_[oPhiExtrapolation].sparams_.size() > 0 ? pnodes_[oPhiExtrapolation].sparams_[0] : ""; }
  std::string fPhiExtrapolationLUTPath() const     { return pnodes_[fPhiExtrapolation].sparams_.size() > 0 ? pnodes_[fPhiExtrapolation].sparams_[0] : ""; }
  std::string bEtaExtrapolationLUTPath() const     { return pnodes_[bEtaExtrapolation].sparams_.size() > 0 ? pnodes_[bEtaExtrapolation].sparams_[0] : ""; }
  std::string oEtaExtrapolationLUTPath() const     { return pnodes_[oEtaExtrapolation].sparams_.size() > 0 ? pnodes_[oEtaExtrapolation].sparams_[0] : ""; }
  std::string fEtaExtrapolationLUTPath() const     { return pnodes_[fEtaExtrapolation].sparams_.size() > 0 ? pnodes_[fEtaExtrapolation].sparams_[0] : ""; }
  std::string sortRankLUTPath() const              { return pnodes_[sortRank].sparams_.size() > 0 ? pnodes_[sortRank].sparams_[0] : ""; }
  void setAbsIsoCheckMemLUTPath        (std::string path) { pnodes_[absIsoCheckMem].sparams_.push_back(path); }
  void setRelIsoCheckMemLUTPath        (std::string path) { pnodes_[relIsoCheckMem].sparams_.push_back(path); }
  void setIdxSelMemPhiLUTPath          (std::string path) { pnodes_[idxSelMemPhi].sparams_.push_back(path); }
  void setIdxSelMemEtaLUTPath          (std::string path) { pnodes_[idxSelMemEta].sparams_.push_back(path); }
  void setBrlSingleMatchQualLUTPath    (std::string path) { pnodes_[brlSingleMatchQual].sparams_.push_back(path); }
  void setFwdPosSingleMatchQualLUTPath (std::string path) { pnodes_[fwdPosSingleMatchQual].sparams_.push_back(path); }
  void setFwdNegSingleMatchQualLUTPath (std::string path) { pnodes_[fwdNegSingleMatchQual].sparams_.push_back(path); }
  void setOvlPosSingleMatchQualLUTPath (std::string path) { pnodes_[ovlPosSingleMatchQual].sparams_.push_back(path); }
  void setOvlNegSingleMatchQualLUTPath (std::string path) { pnodes_[ovlNegSingleMatchQual].sparams_.push_back(path); }
  void setBOPosMatchQualLUTPath        (std::string path) { pnodes_[bOPosMatchQual].sparams_.push_back(path); }
  void setBONegMatchQualLUTPath        (std::string path) { pnodes_[bONegMatchQual].sparams_.push_back(path); }
  void setFOPosMatchQualLUTPath        (std::string path) { pnodes_[fOPosMatchQual].sparams_.push_back(path); }
  void setFONegMatchQualLUTPath        (std::string path) { pnodes_[fONegMatchQual].sparams_.push_back(path); }
  void setBPhiExtrapolationLUTPath     (std::string path) { pnodes_[bPhiExtrapolation].sparams_.push_back(path); }
  void setOPhiExtrapolationLUTPath     (std::string path) { pnodes_[oPhiExtrapolation].sparams_.push_back(path); }
  void setFPhiExtrapolationLUTPath     (std::string path) { pnodes_[fPhiExtrapolation].sparams_.push_back(path); }
  void setBEtaExtrapolationLUTPath     (std::string path) { pnodes_[bEtaExtrapolation].sparams_.push_back(path); }
  void setOEtaExtrapolationLUTPath     (std::string path) { pnodes_[oEtaExtrapolation].sparams_.push_back(path); }
  void setFEtaExtrapolationLUTPath     (std::string path) { pnodes_[fEtaExtrapolation].sparams_.push_back(path); }
  void setSortRankLUTPath              (std::string path) { pnodes_[sortRank].sparams_.push_back(path); }

  // print parameters to stream:
  void print(std::ostream&) const;
  friend std::ostream& operator<<(std::ostream& o, const L1TMuonGlobalParams & p) { p.print(o); return o; }

private:
  unsigned version_;
  unsigned fwVersion_;

  std::vector<Node> pnodes_;

  COND_SERIALIZABLE;
};
#endif
