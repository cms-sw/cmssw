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
  // Borrwed from DQM/GEM/interface/GEMOfflineDQMBase.h
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

    MonitorElement *bookProfile2D(TString name,
                                  TString title,
                                  int nbinsx,
                                  double xlow,
                                  double xup,
                                  int nbinsy,
                                  double ylow,
                                  double yup,
                                  double zlow,
                                  double zup,
                                  TString x_title = "",
                                  TString y_title = "") {
      name += name_suffix_;
      title += title_suffix_ + ";" + x_title + ";" + y_title;
      return ibooker_->bookProfile2D(name, title, nbinsx, xlow, xup, nbinsy, ylow, yup, zlow, zup);
    }

    DQMStore::IBooker *getBooker() { return ibooker_; }

  private:
    DQMStore::IBooker *ibooker_;
    const TString name_suffix_;
    const TString title_suffix_;
  };  // BookingHelper

  template <class M, class K>
  class MEMapInfT {
  public:
    MEMapInfT() : bOperating_(false), bIsNoUnderOverflowBin_(false){};

    MEMapInfT(
        GEMDQMBase *pDQMBase, TString strName, TString strTitle, TString strTitleX = "", TString strTitleY = "Entries")
        : pDQMBase_(pDQMBase),
          strName_(strName),
          strTitle_(strTitle),
          strTitleX_(strTitleX),
          strTitleY_(strTitleY),
          log_category_own_(pDQMBase->log_category_){};

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
          bIsProfile_(false),
          bIsNoUnderOverflowBin_(false),
          nBinsX_(nBinsX),
          dXL_(dXL),
          dXH_(dXH),
          nBinsY_(-1),
          log_category_own_(pDQMBase->log_category_){};

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
          bIsProfile_(false),
          bIsNoUnderOverflowBin_(false),
          nBinsX_(-1),
          nBinsY_(-1),
          log_category_own_(pDQMBase->log_category_) {
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
          bIsProfile_(false),
          bIsNoUnderOverflowBin_(false),
          nBinsX_(nBinsX),
          dXL_(dXL),
          dXH_(dXH),
          nBinsY_(nBinsY),
          dYL_(dYL),
          dYH_(dYH),
          dZL_(0),
          dZH_(1024),
          log_category_own_(pDQMBase->log_category_){};

    MEMapInfT(GEMDQMBase *pDQMBase,  // For TProfile2D
              TString strName,
              TString strTitle,
              Int_t nBinsX,
              Double_t dXL,
              Double_t dXH,
              Int_t nBinsY,
              Double_t dYL,
              Double_t dYH,
              Double_t dZL,
              Double_t dZH,
              TString strTitleX = "",
              TString strTitleY = "")
        : pDQMBase_(pDQMBase),
          strName_(strName),
          strTitle_(strTitle),
          strTitleX_(strTitleX),
          strTitleY_(strTitleY),
          bOperating_(true),
          bIsProfile_(true),
          bIsNoUnderOverflowBin_(false),
          nBinsX_(nBinsX),
          dXL_(dXL),
          dXH_(dXH),
          nBinsY_(nBinsY),
          dYL_(dYL),
          dYH_(dYH),
          dZL_(dZL),
          dZH_(dZH),
          log_category_own_(pDQMBase->log_category_){};

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
    //      bIsNoUnderOverflowBin_(false),
    //      nBinsX_(nBinsX),
    //      dXL_(dXL),
    //      dXH_(dXH),
    //      nBinsY_(nBinsY),
    //      dYL_(dYL),
    //      dYH_(dYH),
    //      log_category_own_(pDQMBase->log_category_){};

    ~MEMapInfT(){};

    Bool_t isOperating() { return bOperating_; };
    void SetOperating(Bool_t bOperating) { bOperating_ = bOperating; };
    void TurnOn() { bOperating_ = true; };
    void TurnOff() { bOperating_ = false; };
    void SetNoUnderOverflowBin() { bIsNoUnderOverflowBin_ = true; };

    Bool_t isProfile() { return bIsProfile_; };
    void SetProfile(Bool_t bIsProfile) { bIsProfile_ = bIsProfile; };

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

    Double_t GetBinLowEdgeZ() { return dZL_; };
    void SetBinLowEdgeZ(Double_t dZL) { dZL_ = dZL; };
    Double_t GetBinHighEdgeZ() { return dZH_; };
    void SetBinHighEdgeZ(Double_t dZH) { dZH_ = dZH; };

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

    void SetPointUOFlow() {
      dXU_ = dXL_ + (dXH_ - dXL_) / nBinsX_ * 0.5;
      dXO_ = dXL_ + (dXH_ - dXL_) / nBinsX_ * (nBinsX_ - 0.5);
      dYU_ = dYL_ + (dYH_ - dYL_) / nBinsY_ * 0.5;
      dYO_ = dYL_ + (dYH_ - dYL_) / nBinsY_ * (nBinsY_ - 0.5);
      dZU_ = dZL_ + (dZH_ - dZL_) / nBinsZ_ * 0.5;
      dZO_ = dZL_ + (dZH_ - dZL_) / nBinsZ_ * (nBinsZ_ - 0.5);
    };

    M &map() { return mapHist; }
    int bookND(BookingHelper &bh, K key) {
      if (!bOperating_)
        return 0;
      SetPointUOFlow();
      if (bIsProfile_) {
        mapHist[key] = bh.bookProfile2D(
            strName_, strTitle_, nBinsX_, dXL_, dXH_, nBinsY_, dYL_, dYH_, dZL_, dZH_, strTitleX_, strTitleY_);
      } else if (nBinsY_ > 0 && nBinsX_ > 0) {
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
        edm::LogError(log_category_own_)
            << "WARNING: Cannot find the histogram corresponing to the given key\n";  // FIXME: It's about sending a message
        return nullptr;
      }
      return mapHist[key];
    };

    int SetLabelForChambers(K key, Int_t nAxis, Int_t nNumBin = -1, Int_t nIdxStart = 1) {
      if (!bOperating_)
        return 0;
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
        histCurr->setBinLabel(i, Form("%i", nIdxStart + i - 1), nAxis);
      }
      return 0;
    };

    int SetLabelForIEta(K key, Int_t nAxis, Int_t nNumBin = -1) { return SetLabelForChambers(key, nAxis, nNumBin); };

    int SetLabelForVFATs(K key, Int_t nNumEtaPartitions, Int_t nAxis, Int_t nNumBin = -1) {
      if (!bOperating_)
        return 0;
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
      if (std::get<1>(key) == 2) {
        Int_t nNumVFATPerModule = 12;  // FIXME: A better way to get this?
        if (nNumBin > nNumVFATPerModule) {
          for (Int_t i = 0; i < nNumBin; i++) {
            Int_t nIModule = i / nNumVFATPerModule + 1;
            histCurr->setBinLabel(
                i + 1, Form((nAxis == 1 ? "#splitline{%i}{M%i}" : "%i (M%i)"), i % nNumVFATPerModule, nIModule), nAxis);
          }
        } else {
          for (Int_t i = 0; i < nNumBin; i++) {
            histCurr->setBinLabel(i + 1, Form("%i", i), nAxis);
          }
        }
      } else {
        for (Int_t i = 0; i < nNumBin; i++) {
          Int_t nIEta = pDQMBase_->getIEtaFromVFAT(std::get<1>(key), i);
          histCurr->setBinLabel(i + 1, Form("%i (%i)", i, nIEta), nAxis);
        }
      }
      return 0;
    };

    int Fill(K key, Double_t x) {
      if (!bOperating_)
        return 0;
      dqm::impl::MonitorElement *hist = FindHist(key);
      if (hist == nullptr)
        return -999;
      if (bIsNoUnderOverflowBin_) {
        if (x <= dXL_)
          x = dXU_;
        else if (x >= dXH_)
          x = dXO_;
      }
      hist->Fill(x);
      return 1;
    };

    int Fill(K key, Double_t x, Double_t y, Double_t w = 1.0) {
      if (!bOperating_)
        return 0;
      dqm::impl::MonitorElement *hist = FindHist(key);
      if (hist == nullptr)
        return -999;
      if (bIsNoUnderOverflowBin_) {
        if (x <= dXL_)
          x = dXU_;
        else if (x >= dXH_)
          x = dXO_;
        if (y <= dYL_)
          y = dYU_;
        else if (y >= dYH_)
          y = dYO_;
      }
      hist->Fill(x, y, w);
      return 1;
    };

    int FillBits(K key, Double_t x, UInt_t bits, Double_t w = 1.0) {
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
          hist->Fill(x, i, w);
        unMask <<= 1;
      }

      return 1;
    };

  private:
    GEMDQMBase *pDQMBase_;

    M mapHist;
    TString strName_, strTitle_, strTitleX_, strTitleY_;
    Bool_t bOperating_;
    Bool_t bIsProfile_;
    Bool_t bIsNoUnderOverflowBin_;

    std::vector<double> x_binning_;
    Int_t nBinsX_;
    Double_t dXL_, dXH_;
    Int_t nBinsY_;
    Double_t dYL_, dYH_;
    Int_t nBinsZ_;
    Double_t dZL_, dZH_;
    Double_t dXU_, dXO_;
    Double_t dYU_, dYO_;
    Double_t dZU_, dZO_;

    std::string log_category_own_;
  };

  typedef MEMapInfT<MEMap2Ids, ME2IdsKey> MEMap2Inf;
  typedef MEMapInfT<MEMap3Ids, ME3IdsKey> MEMap3Inf;
  typedef MEMapInfT<MEMap4Ids, ME4IdsKey> MEMap4Inf;
  typedef MEMapInfT<MEMap5Ids, ME5IdsKey> MEMap5Inf;

  class MEStationInfo {
  public:
    MEStationInfo() : nNumChambers_(-1){};
    MEStationInfo(Int_t nRegion,
                  Int_t nStation,
                  Int_t nLayer,
                  Int_t nNumChambers,
                  Int_t nNumModules,
                  Int_t nNumEtaPartitions,
                  Int_t nMaxVFAT,
                  Int_t nFirstStrip,
                  Int_t nNumDigi,
                  Int_t nMinIdxChamber,
                  Int_t nMaxIdxChamber)
        : nRegion_(nRegion),
          nStation_(nStation),
          nLayer_(nLayer),
          nNumChambers_(nNumChambers),
          nNumModules_(nNumModules),
          nNumEtaPartitions_(nNumEtaPartitions),
          nMaxVFAT_(nMaxVFAT),
          nFirstStrip_(nFirstStrip),
          nNumDigi_(nNumDigi),
          nMinIdxChamber_(nMinIdxChamber),
          nMaxIdxChamber_(nMaxIdxChamber),
          fMinPhi_(0){};

    bool operator==(const MEStationInfo &other) const {
      return (nRegion_ == other.nRegion_ && nStation_ == other.nStation_ && nLayer_ == other.nLayer_ &&
              nNumChambers_ == other.nNumChambers_ && nNumModules_ == other.nNumModules_ &&
              nNumEtaPartitions_ == other.nNumEtaPartitions_ && nMaxVFAT_ == other.nMaxVFAT_ &&
              nFirstStrip_ == other.nFirstStrip_ && nNumDigi_ == other.nNumDigi_);
    };

    Int_t nRegion_;            // the region index
    Int_t nStation_;           // the station index
    Int_t nLayer_;             // the layer
    Int_t nNumChambers_;       // the number of chambers in the current station
    Int_t nNumModules_;        // the number of modules in each chamber
    Int_t nNumEtaPartitions_;  // the number of eta partitions of the chambers
    Int_t nMaxVFAT_;     // the number of all VFATs in each chamber (= # of VFATs in eta partition * nNumEtaPartitions_)
    Int_t nFirstStrip_;  // the index of the first strip
    Int_t nNumDigi_;     // the number of digis of each VFAT

    Int_t nMinIdxChamber_;
    Int_t nMaxIdxChamber_;
    Float_t fMinPhi_;

    std::vector<Float_t> listRadiusEvenChamber_;
    std::vector<Float_t> listRadiusOddChamber_;
  };

  int readGeometryRadiusInfoChamber(const GEMStation *station, MEStationInfo &stationInfo);
  int readGeometryPhiInfoChamber(const GEMStation *station, MEStationInfo &stationInfo);

public:
  explicit GEMDQMBase(const edm::ParameterSet &cfg);
  ~GEMDQMBase() override{};

  enum {
    GEMDQM_RUNTYPE_ONLINE,
    GEMDQM_RUNTYPE_OFFLINE,
    GEMDQM_RUNTYPE_RELVAL,
    GEMDQM_RUNTYPE_ALLPLOTS,
    GEMDQM_RUNTYPE_NONE = -1
  };

  Int_t nRunType_;

  std::string log_category_;

protected:
  int initGeometry(edm::EventSetup const &iSetup);
  int loadChambers();

  int GenerateMEPerChamber(DQMStore::IBooker &ibooker);
  virtual int ProcessWithMEMap2(BookingHelper &bh, ME2IdsKey key) { return 0; };              // must be overrided
  virtual int ProcessWithMEMap2WithEta(BookingHelper &bh, ME3IdsKey key) { return 0; };       // must be overrided
  virtual int ProcessWithMEMap2AbsReWithEta(BookingHelper &bh, ME3IdsKey key) { return 0; };  // must be overrided
  virtual int ProcessWithMEMap3(BookingHelper &bh, ME3IdsKey key) { return 0; };              // must be overrided
  virtual int ProcessWithMEMap4(BookingHelper &bh, ME4IdsKey key) { return 0; };              // must be overrided
  virtual int ProcessWithMEMap5(BookingHelper &bh, ME5IdsKey key) { return 0; };              // must be overrided
  virtual int ProcessWithMEMap4WithChamber(BookingHelper &bh, ME4IdsKey key) { return 0; };   // must be overrided
  virtual int ProcessWithMEMap5WithChamber(BookingHelper &bh, ME5IdsKey key) { return 0; };   // must be overrided

  int keyToRegion(ME2IdsKey key) { return std::get<0>(key); };
  int keyToRegion(ME3IdsKey key) { return std::get<0>(key); };
  int keyToRegion(ME4IdsKey key) { return std::get<0>(key); };
  int keyToRegion(ME5IdsKey key) { return std::get<0>(key); };
  int keyToStation(ME2IdsKey key) { return std::get<1>(key); };
  int keyToStation(ME3IdsKey key) { return std::get<1>(key); };
  int keyToStation(ME4IdsKey key) { return std::get<1>(key); };
  int keyToStation(ME5IdsKey key) { return std::get<1>(key); };
  int keyToLayer(ME3IdsKey key) { return std::get<2>(key); };
  int keyToLayer(ME4IdsKey key) { return std::get<2>(key); };
  int keyToLayer(ME5IdsKey key) { return std::get<2>(key); };
  int keyToModule(ME4IdsKey key) { return std::get<3>(key); };
  int keyToModule(ME5IdsKey key) { return std::get<3>(key); };
  int keyToChamber(ME4IdsKey key) { return std::get<3>(key); };
  int keyToChamber(ME5IdsKey key) { return std::get<4>(key); };
  int keyToIEta(ME4IdsKey key) { return std::get<3>(key); };
  int keyToIEta(ME5IdsKey key) { return std::get<4>(key); };

  ME2IdsKey key3Tokey2(ME3IdsKey key) {
    auto keyNew = ME2IdsKey{keyToRegion(key), keyToStation(key)};
    return keyNew;
  };

  ME3IdsKey key4Tokey3(ME4IdsKey key) {
    auto keyNew = ME3IdsKey{keyToRegion(key), keyToStation(key), keyToLayer(key)};
    return keyNew;
  };

  ME4IdsKey key5Tokey4(ME5IdsKey key) {
    auto keyNew = ME4IdsKey{keyToRegion(key), keyToStation(key), keyToLayer(key), keyToModule(key)};
    return keyNew;
  };

  int SortingLayers(std::vector<ME4IdsKey> &listLayers);
  dqm::impl::MonitorElement *CreateSummaryHist(DQMStore::IBooker &ibooker, TString strName);

  template <typename T>
  inline bool checkRefs(const std::vector<T *> &);

  int getNumEtaPartitions(const GEMStation *);
  inline int getVFATNumber(const int, const int, const int);
  inline int getVFATNumberGE11(const int, const int, const int);
  inline int getVFATNumberByDigi(const int, const int, const int);
  inline int getIEtaFromVFAT(const int station, const int vfat);
  inline int getIEtaFromVFATGE0(const int vfat);
  inline int getIEtaFromVFATGE11(const int vfat);
  inline int getIEtaFromVFATGE21(const int vfat);
  inline int getMaxVFAT(const int);
  inline int getNumModule(const int);
  inline int getIdxModule(const int, const int);
  inline int getDetOccXBin(const int, const int, const int);
  inline Float_t restrictAngle(const Float_t fTheta, const Float_t fStart);
  inline std::string getNameDirLayer(ME3IdsKey key3);
  inline std::string getNameDirLayer(ME4IdsKey key4);

  const GEMGeometry *GEMGeometry_;
  edm::ESGetToken<GEMGeometry, MuonGeometryRecord> geomToken_;

  std::vector<GEMDetId> listChamberId_;
  std::map<GEMDetId, std::vector<const GEMEtaPartition *>> mapEtaPartition_;

  std::map<ME2IdsKey, bool> MEMap2Check_;
  std::map<ME3IdsKey, bool> MEMap2WithEtaCheck_;
  std::map<ME3IdsKey, bool> MEMap2AbsReWithEtaCheck_;
  std::map<ME3IdsKey, bool> MEMap3Check_;
  std::map<ME4IdsKey, bool> MEMap4Check_;
  std::map<ME4IdsKey, bool> MEMap4WithChCheck_;
  std::map<ME5IdsKey, bool> MEMap5WithChCheck_;
  std::map<ME5IdsKey, bool> MEMap5Check_;

  int nMaxNumCh_;
  std::map<ME3IdsKey, MEStationInfo> mapStationInfo_;
  std::map<ME4IdsKey, int> mapStationToIdx_;
};

// Borrwed from DQM/GEM/interface/GEMOfflineDQMBase.h
template <typename T>
inline bool GEMDQMBase::checkRefs(const std::vector<T *> &refs) {
  if (refs.empty())
    return false;
  if (refs.front() == nullptr)
    return false;
  return true;
}

// The 'get...' functions in the below are borrwed from DQM/GEM/interface/GEMOfflineDQMBase.h
inline int GEMDQMBase::getMaxVFAT(const int station) {
  if (station == 0)
    return GEMeMap::maxVFatGE0_;
  if (station == 1)
    return GEMeMap::maxVFatGE11_;
  else if (station == 2)
    return GEMeMap::maxVFatGE21_ / 2;
  else
    return -1;
}

inline int GEMDQMBase::getNumModule(const int station) {
  if (station == 0)
    return 1;
  if (station == 1)
    return 1;
  else if (station == 2)
    return 4;
  else
    return -1;
}

inline int GEMDQMBase::getIdxModule(const int station, const int chamberType) {
  if (station == 0)
    return 1;
  if (station == 1)
    return 1;
  else if (station == 2)
    return chamberType - 20;
  else
    return -1;
}

inline int GEMDQMBase::getVFATNumber(const int station, const int ieta, const int vfat_phi) {
  if (station == 1)
    return getVFATNumberGE11(station, ieta, vfat_phi);
  return getVFATNumberGE11(station, ieta, vfat_phi);  // FIXME: What about GE21 and GE0?
}

inline int GEMDQMBase::getVFATNumberGE11(const int station, const int ieta, const int vfat_phi) {
  return vfat_phi * GEMeMap::maxiEtaIdGE11_ + (GEMeMap::maxiEtaIdGE11_ - ieta);
}

inline int GEMDQMBase::getVFATNumberByDigi(const int station, const int ieta, const int digi) {
  const int vfat_phi = digi / GEMeMap::maxChan_;
  return getVFATNumber(station, ieta, vfat_phi);
}

inline int GEMDQMBase::getIEtaFromVFAT(const int station, const int vfat) {
  if (station == 0)
    return getIEtaFromVFATGE0(vfat);
  if (station == 1)
    return getIEtaFromVFATGE11(vfat);
  if (station == 2)
    return getIEtaFromVFATGE21(vfat);
  return getIEtaFromVFATGE11(vfat);  // FIXME: What about GE21 and GE0?
}

inline int GEMDQMBase::getIEtaFromVFATGE0(const int vfat) {
  return GEMeMap::maxiEtaIdGE0_ - (vfat % GEMeMap::maxiEtaIdGE0_);
}

inline int GEMDQMBase::getIEtaFromVFATGE11(const int vfat) {
  return GEMeMap::maxiEtaIdGE11_ - (vfat % GEMeMap::maxiEtaIdGE11_);
}

inline int GEMDQMBase::getIEtaFromVFATGE21(const int vfat) {
  return GEMeMap::maxiEtaIdGE21_ - (vfat % GEMeMap::maxiEtaIdGE21_);
}

inline int GEMDQMBase::getDetOccXBin(const int chamber, const int layer, const int n_chambers) {
  return n_chambers * (chamber - 1) + layer;
}

inline Float_t GEMDQMBase::restrictAngle(const Float_t fTheta, const Float_t fStart) {
  Float_t fLoop = (fTheta - fStart) / (2 * M_PI);
  int nLoop = (fLoop >= 0 ? (int)fLoop : (int)fLoop - 1);
  return fTheta - nLoop * 2 * M_PI;
}

inline std::string GEMDQMBase::getNameDirLayer(ME3IdsKey key3) {
  auto nStation = keyToStation(key3);
  char cRegion = (keyToRegion(key3) > 0 ? 'P' : 'M');
  auto nLayer = keyToLayer(key3);
  return std::string(Form("GE%i1-%c-L%i", nStation, cRegion, nLayer));
}

inline std::string GEMDQMBase::getNameDirLayer(ME4IdsKey key4) {
  auto nStation = keyToStation(key4);
  char cRegion = (keyToRegion(key4) > 0 ? 'P' : 'M');
  auto nLayer = keyToLayer(key4);
  if (nStation == 2) {
    auto nModule = keyToModule(key4);
    return std::string(Form("GE%i1-%c-L%i-M%i", nStation, cRegion, nLayer, nModule));
  }
  return std::string(Form("GE%i1-%c-L%i", nStation, cRegion, nLayer));
}

#endif  // DQM_GEM_INTERFACE_GEMDQMBase_h
