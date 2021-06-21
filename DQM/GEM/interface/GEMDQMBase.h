#ifndef DQM_GEM_INTERFACE_GEMDQMBase_h
#define DQM_GEM_INTERFACE_GEMDQMBase_h

#include <map>
#include <tuple>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "CondFormats/GEMObjects/interface/GEMeMap.h"
#include "Validation/MuonGEMHits/interface/GEMValidationUtils.h"

#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"

#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

class GEMDQMBase : public DQMEDAnalyzer {
public:
  // Borrwed from DQMOffline/Muon/interface/GEMOfflineDQMBase.h
  class BookingHelper {
  public:
    BookingHelper(DQMStore::IBooker &ibooker, const TString &name_suffix, const TString &title_suffix)
        : ibooker_(&ibooker), name_suffix_(name_suffix), title_suffix_(title_suffix) {}

    ~BookingHelper() {}

    MonitorElement *book1D(TString name,
                           TString title,
                           int nbinsx,
                           double xlow,
                           double xup,
                           TString x_title = "",
                           TString y_title = "Entries") {
      name += name_suffix_;
      title += title_suffix_ + ";" + x_title + ";" + y_title;
      return ibooker_->book1D(name, title, nbinsx, xlow, xup);
    }

    MonitorElement *book1D(TString name,
                           TString title,
                           std::vector<double> &x_binning,
                           TString x_title = "",
                           TString y_title = "Entries") {
      name += name_suffix_;
      title += title_suffix_ + ";" + x_title + ";" + y_title;
      TH1F *h_obj = new TH1F(name, title, x_binning.size() - 1, &x_binning[0]);
      return ibooker_->book1D(name, h_obj);
    }

    MonitorElement *book2D(TString name,
                           TString title,
                           int nbinsx,
                           double xlow,
                           double xup,
                           int nbinsy,
                           double ylow,
                           double yup,
                           TString x_title = "",
                           TString y_title = "") {
      name += name_suffix_;
      title += title_suffix_ + ";" + x_title + ";" + y_title;
      return ibooker_->book2D(name, title, nbinsx, xlow, xup, nbinsy, ylow, yup);
    }

  private:
    DQMStore::IBooker *ibooker_;
    const TString name_suffix_;
    const TString title_suffix_;
  };  // BookingHelper

  template <class M, class K>
  class MEMapInfT {
  public:
    MEMapInfT() : bOperating_(false){};

    MEMapInfT(
        GEMDQMBase *pDQMBase, TString strName, TString strTitle, TString strTitleX = "", TString strTitleY = "Entries")
        : pDQMBase_(pDQMBase), strName_(strName), strTitle_(strTitle), strTitleX_(strTitleX), strTitleY_(strTitleY){};

    MEMapInfT(GEMDQMBase *pDQMBase,
              TString strName,
              TString strTitle,
              Int_t nBinsX,
              Double_t dXL,
              Double_t dXH,
              TString strTitleX = "",
              TString strTitleY = "Entries")
        : pDQMBase_(pDQMBase),
          strName_(strName),
          strTitle_(strTitle),
          strTitleX_(strTitleX),
          strTitleY_(strTitleY),
          bOperating_(true),
          nBinsX_(nBinsX),
          dXL_(dXL),
          dXH_(dXH),
          nBinsY_(-1){};

    MEMapInfT(GEMDQMBase *pDQMBase,
              TString strName,
              TString strTitle,
              std::vector<Double_t> &x_binning,
              TString strTitleX = "",
              TString strTitleY = "Entries")
        : pDQMBase_(pDQMBase),
          strName_(strName),
          strTitle_(strTitle),
          strTitleX_(strTitleX),
          strTitleY_(strTitleY),
          bOperating_(true),
          nBinsX_(-1),
          nBinsY_(-1) {
      for (Int_t i = 0; i < (Int_t)x_binning.size(); i++)
        x_binning_.push_back(x_binning[i]);
    };

    MEMapInfT(GEMDQMBase *pDQMBase,
              TString strName,
              TString strTitle,
              Int_t nBinsX,
              Double_t dXL,
              Double_t dXH,
              Int_t nBinsY,
              Double_t dYL,
              Double_t dYH,
              TString strTitleX = "",
              TString strTitleY = "")
        : pDQMBase_(pDQMBase),
          strName_(strName),
          strTitle_(strTitle),
          strTitleX_(strTitleX),
          strTitleY_(strTitleY),
          bOperating_(true),
          nBinsX_(nBinsX),
          dXL_(dXL),
          dXH_(dXH),
          nBinsY_(nBinsY),
          dYL_(dYL),
          dYH_(dYH){};

    //MEMapInfT(GEMDQMBase *pDQMBase,
    //          TString strName,
    //          TString strTitle,
    //          std::vector<Double_t> &x_binning,
    //          std::vector<Double_t> &y_binning,
    //          TString strTitleX = "",
    //          TString strTitleY = "")
    //    : pDQMBase_(pDQMBase),
    //      strName_(strName),
    //      strTitle_(strTitle),
    //      strTitleX_(strTitleX),
    //      strTitleY_(strTitleY),
    //      bOperating_(true),
    //      nBinsX_(nBinsX),
    //      dXL_(dXL),
    //      dXH_(dXH),
    //      nBinsY_(nBinsY),
    //      dYL_(dYL),
    //      dYH_(dYH){};

    ~MEMapInfT(){};

    Bool_t isOperating() { return bOperating_; };
    void SetOperating(Bool_t bOperating) { bOperating_ = bOperating; };
    void turnOn() { bOperating_ = true; };
    void turnOff() { bOperating_ = false; };

    TString GetName() { return strName_; };
    void SetName(TString strName) { strName_ = strName; };
    TString GetTitle() { return strTitle_; };
    void SetTitle(TString strTitle) { strTitle_ = strTitle; };
    TString GetTitleX() { return strTitleX_; };
    void SetTitleX(TString strTitleX) { strTitleX_ = strTitleX; };
    TString GetTitleY() { return strTitleY_; };
    void SetTitleY(TString strTitleY) { strTitleY_ = strTitleY; };

    Int_t GetNbinsX() { return nBinsX_; };
    void SetNbinsX(Int_t nBinsX) { nBinsX_ = nBinsX; };
    Double_t GetBinLowEdgeX() { return dXL_; };
    void SetBinLowEdgeX(Double_t dXL) { dXL_ = dXL; };
    Double_t GetBinHighEdgeX() { return dXH_; };
    void SetBinHighEdgeX(Double_t dXH) { dXH_ = dXH; };
    // FIXME: x_binning_

    Int_t GetNbinsY() { return nBinsY_; };
    void SetNbinsY(Int_t nBinsY) { nBinsY_ = nBinsY; };
    Double_t GetBinLowEdgeY() { return dYL_; };
    void SetBinLowEdgeY(Double_t dYL) { dYL_ = dYL; };
    Double_t GetBinHighEdgeY() { return dYH_; };
    void SetBinHighEdgeY(Double_t dYH) { dYH_ = dYH; };

    void SetBinConfX(Int_t nBins, Double_t dL = 0.5, Double_t dH = -1048576.0) {
      nBinsX_ = nBins;
      dXL_ = dL;
      if (dH <= dL)
        dH = dL + nBins;
      dXH_ = dH;
    };

    void SetBinConfY(Int_t nBins, Double_t dL = 0.5, Double_t dH = -1048576.0) {
      nBinsY_ = nBins;
      dYL_ = dL;
      if (dH <= dL)
        dH = dL + nBins;
      dYH_ = dH;
    };

    M &map() { return mapHist; }
    int bookND(BookingHelper &bh, K key) {
      if (!bOperating_)
        return 0;
      if (nBinsY_ > 0 && nBinsX_ > 0) {
        mapHist[key] = bh.book2D(strName_, strTitle_, nBinsX_, dXL_, dXH_, nBinsY_, dYL_, dYH_, strTitleX_, strTitleY_);
        return 0;
      } else if (!x_binning_.empty()) {
        mapHist[key] = bh.book1D(strName_, strTitle_, nBinsX_, dXL_, dXH_, strTitleX_, strTitleY_);
        return 0;
      } else if (nBinsX_ > 0) {
        mapHist[key] = bh.book1D(strName_, strTitle_, nBinsX_, dXL_, dXH_, strTitleX_, strTitleY_);
        return 0;
      }

      return -1;
    };

    dqm::impl::MonitorElement *FindHist(K key) {
      if (mapHist.find(key) == mapHist.end()) {
        std::cout << "" << std::endl;  // FIXME: It's about sending a message
        return nullptr;
      }
      return mapHist[key];
    };

    int SetLabelForChambers(K key, Int_t nAxis, Int_t nNumBin = -1) {
      if (nNumBin <= 0) {
        if (nAxis == 1)
          nNumBin = nBinsX_;
        else if (nAxis == 2)
          nNumBin = nBinsY_;
        else
          return -1;
      }
      dqm::impl::MonitorElement *histCurr = FindHist(key);
      if (histCurr == nullptr)
        return -999;
      for (Int_t i = 1; i <= nNumBin; i++) {
        histCurr->setBinLabel(i, Form("%i", i), nAxis);
      }
      return 0;
    };

    int SetLabelForVFATs(K key, Int_t nNumEtaPartitions, Int_t nAxis, Int_t nNumBin = -1) {
      if (nNumBin <= 0) {
        if (nAxis == 1)
          nNumBin = nBinsX_;
        else if (nAxis == 2)
          nNumBin = nBinsY_;
        else
          return -1;
      }
      dqm::impl::MonitorElement *histCurr = FindHist(key);
      if (histCurr == nullptr)
        return -999;
      for (Int_t i = 0; i < nNumBin; i++) {
        Int_t nIEta = pDQMBase_->getIEtaFromVFAT(std::get<1>(key), i);
        histCurr->setBinLabel(i + 1, Form("%i (%i)", i, nIEta), nAxis);
      }
      return 0;
    };

    int Fill(K key, Double_t x) {
      if (!bOperating_)
        return 0;
      dqm::impl::MonitorElement *hist = FindHist(key);
      if (hist == nullptr)
        return -999;
      hist->Fill(x);
      return 0;
    };

    int Fill(K key, Double_t x, Double_t y) {
      if (!bOperating_)
        return 0;
      dqm::impl::MonitorElement *hist = FindHist(key);
      if (hist == nullptr)
        return -999;
      hist->Fill(x, y);
      return 0;
    };

    int FillBits(K key, Double_t x, UInt_t bits) {
      if (!bOperating_)
        return 0;
      dqm::impl::MonitorElement *hist = FindHist(key);
      if (hist == nullptr)
        return -999;

      if (nBinsY_ <= 0)
        return -888;
      UInt_t unMask = 0x1;
      for (Int_t i = 1; i <= nBinsY_; i++) {
        if ((unMask & bits) != 0)
          hist->Fill(x, i);
        unMask <<= 1;
      }

      return 0;
    };

  private:
    GEMDQMBase *pDQMBase_;

    M mapHist;
    TString strName_, strTitle_, strTitleX_, strTitleY_;
    Bool_t bOperating_;

    std::vector<double> x_binning_;
    Int_t nBinsX_;
    Double_t dXL_, dXH_;
    Int_t nBinsY_;
    Double_t dYL_, dYH_;
  };

  typedef MEMapInfT<MEMap2Ids, ME2IdsKey> MEMap2Inf;
  typedef MEMapInfT<MEMap3Ids, ME3IdsKey> MEMap3Inf;
  typedef MEMapInfT<MEMap4Ids, ME4IdsKey> MEMap4Inf;

  class MEStationInfo {
  public:
    MEStationInfo() : nNumChambers_(-1){};
    MEStationInfo(Int_t nRegion,
                  Int_t nStation,
                  Int_t nLayer,
                  Int_t nNumChambers,
                  Int_t nNumEtaPartitions,
                  Int_t nMaxVFAT,
                  Int_t nNumStrip)
        : nRegion_(nRegion),
          nStation_(nStation),
          nLayer_(nLayer),
          nNumChambers_(nNumChambers),
          nNumEtaPartitions_(nNumEtaPartitions),
          nMaxVFAT_(nMaxVFAT),
          nNumStrip_(nNumStrip){};

    bool operator==(const MEStationInfo &other) const {
      return (nRegion_ == other.nRegion_ && nStation_ == other.nStation_ && nLayer_ == other.nLayer_ &&
              nNumChambers_ == other.nNumChambers_ && nNumEtaPartitions_ == other.nNumEtaPartitions_ &&
              nMaxVFAT_ == other.nMaxVFAT_ && nNumStrip_ == other.nNumStrip_);
    };

    Int_t nRegion_;            // the region index
    Int_t nStation_;           // the station index
    Int_t nLayer_;             // the layer
    Int_t nNumChambers_;       // the number of chambers in the current station
    Int_t nNumEtaPartitions_;  // the number of eta partitions of the chambers
    Int_t nMaxVFAT_;   // the number of all VFATs in each chamber (= # of VFATs in eta partition * nNumEtaPartitions_)
    Int_t nNumStrip_;  // the number of strips of each VFAT
  };

public:
  explicit GEMDQMBase(const edm::ParameterSet &cfg);
  ~GEMDQMBase() override{};

protected:
  int initGeometry(edm::EventSetup const &iSetup);
  int loadChambers();
  int readRadiusEtaPartition(int nRegion, int nStation);

  int GenerateMEPerChamber(DQMStore::IBooker &ibooker);
  virtual int ProcessWithMEMap2(BookingHelper &bh, ME2IdsKey key) { return 0; };             // must be overrided
  virtual int ProcessWithMEMap3(BookingHelper &bh, ME3IdsKey key) { return 0; };             // must be overrided
  virtual int ProcessWithMEMap4(BookingHelper &bh, ME4IdsKey key) { return 0; };             // must be overrided
  virtual int ProcessWithMEMap3WithChamber(BookingHelper &bh, ME4IdsKey key) { return 0; };  // must be overrided

  int keyToRegion(ME2IdsKey key) { return std::get<0>(key); };
  int keyToRegion(ME3IdsKey key) { return std::get<0>(key); };
  int keyToRegion(ME4IdsKey key) { return std::get<0>(key); };
  int keyToStation(ME2IdsKey key) { return std::get<1>(key); };
  int keyToStation(ME3IdsKey key) { return std::get<1>(key); };
  int keyToStation(ME4IdsKey key) { return std::get<1>(key); };
  int keyToLayer(ME3IdsKey key) { return std::get<2>(key); };
  int keyToLayer(ME4IdsKey key) { return std::get<2>(key); };
  int keyToChamber(ME4IdsKey key) { return std::get<3>(key); };
  int keyToIEta(ME4IdsKey key) { return std::get<3>(key); };

  ME3IdsKey key4Tokey3(ME4IdsKey key) {
    auto keyNew = ME3IdsKey{keyToRegion(key), keyToStation(key), keyToLayer(key)};
    return keyNew;
  };

  int SortingLayers(std::vector<ME3IdsKey> &listLayers);
  dqm::impl::MonitorElement *CreateSummaryHist(DQMStore::IBooker &ibooker, TString strName);

  template <typename T>
  inline bool checkRefs(const std::vector<T *> &);

  int getNumEtaPartitions(const GEMStation *);
  inline int getVFATNumber(const int, const int, const int);
  inline int getVFATNumberGE11(const int, const int, const int);
  inline int getVFATNumberByStrip(const int, const int, const int);
  inline int getIEtaFromVFAT(const int station, const int vfat);
  inline int getIEtaFromVFATGE11(const int vfat);
  inline int getMaxVFAT(const int);
  inline int getDetOccXBin(const int, const int, const int);

  std::string log_category_;

  const GEMGeometry *GEMGeometry_;
  edm::ESGetToken<GEMGeometry, MuonGeometryRecord> geomToken_;

  std::vector<GEMChamber> gemChambers_;

  std::map<ME2IdsKey, bool> MEMap2Check_;
  std::map<ME3IdsKey, bool> MEMap3Check_;
  std::map<ME4IdsKey, bool> MEMap3WithChCheck_;
  std::map<ME4IdsKey, bool> MEMap4Check_;

  int nMaxNumCh_;
  std::map<ME3IdsKey, int> mapStationToIdx_;
  std::map<ME3IdsKey, MEStationInfo> mapStationInfo_;

  Int_t nNumEtaPartitionGE0_;
  Int_t nNumEtaPartitionGE11_;
  Int_t nNumEtaPartitionGE21_;
};

// Borrwed from DQMOffline/Muon/interface/GEMOfflineDQMBase.h
template <typename T>
inline bool GEMDQMBase::checkRefs(const std::vector<T *> &refs) {
  if (refs.empty())
    return false;
  if (refs.front() == nullptr)
    return false;
  return true;
}

// The 'get...' functions in the below are borrwed from DQMOffline/Muon/interface/GEMOfflineDQMBase.h
inline int GEMDQMBase::getMaxVFAT(const int station) {
  if (station == 1)
    return GEMeMap::maxVFatGE11_;
  else if (station == 2)
    return GEMeMap::maxVFatGE21_;
  else
    return -1;
}

inline int GEMDQMBase::getVFATNumber(const int station, const int ieta, const int vfat_phi) {
  if (station == 1)
    return getVFATNumberGE11(station, ieta, vfat_phi);
  return getVFATNumberGE11(station, ieta, vfat_phi);  // FIXME: What about GE21 and GE0?
}

inline int GEMDQMBase::getVFATNumberGE11(const int station, const int ieta, const int vfat_phi) {
  return vfat_phi * nNumEtaPartitionGE11_ + (8 - ieta);
}

inline int GEMDQMBase::getVFATNumberByStrip(const int station, const int ieta, const int strip) {
  const int vfat_phi = (strip % GEMeMap::maxChan_) ? strip / GEMeMap::maxChan_ : strip / GEMeMap::maxChan_ - 1;
  return getVFATNumber(station, ieta, vfat_phi);
}

inline int GEMDQMBase::getIEtaFromVFAT(const int station, const int vfat) {
  if (station == 1)
    return getIEtaFromVFATGE11(vfat);
  return getIEtaFromVFATGE11(vfat);  // FIXME: What about GE21 and GE0?
}

inline int GEMDQMBase::getIEtaFromVFATGE11(const int vfat) { return 8 - (vfat % nNumEtaPartitionGE11_); }

inline int GEMDQMBase::getDetOccXBin(const int chamber, const int layer, const int n_chambers) {
  return n_chambers * (chamber - 1) + layer;
}

#endif  // DQM_GEM_INTERFACE_GEMDQMBase_h
