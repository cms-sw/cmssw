#ifndef DQM_GEM_INTERFACE_GEMDAQStatusSource_h
#define DQM_GEM_INTERFACE_GEMDAQStatusSource_h

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "Validation/MuonGEMHits/interface/GEMValidationUtils.h"

#include "CondFormats/DataRecord/interface/GEMChMapRcd.h"
#include "CondFormats/GEMObjects/interface/GEMChMap.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMVFATStatusCollection.h"
#include "DataFormats/GEMDigi/interface/GEMOHStatusCollection.h"
#include "DataFormats/GEMDigi/interface/GEMAMCStatusCollection.h"
#include "DataFormats/GEMDigi/interface/GEMAMC13StatusCollection.h"

#include "DQM/GEM/interface/GEMDQMBase.h"

#include <string>

//----------------------------------------------------------------------------------------------------

class GEMDAQStatusSource : public GEMDQMBase {
public:
  explicit GEMDAQStatusSource(const edm::ParameterSet &cfg);
  ~GEMDAQStatusSource() override{};
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

protected:
  void LoadROMap(edm::EventSetup const &iSetup);

  void dqmBeginRun(edm::Run const &, edm::EventSetup const &) override{};
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const &e, edm::EventSetup const &eSetup) override;

  void FillWithRiseErr(MonitorElement *h, Int_t nX, Int_t nY, Bool_t &bErr) {
    h->Fill(nX, nY);
    bErr = true;
  };

  void FillStatusSummaryPlot(std::map<ME5IdsKey, bool> &mapChamber,
                             MonitorElement *h2Plot,
                             std::map<ME5IdsKey, bool> *pmapSummary = nullptr) {
    for (auto const &[key5, bFlag] : mapChamber) {  // bFlag is not used
      ME4IdsKey key4 = key5Tokey4(key5);
      Int_t nChamber = keyToChamber(key5);
      h2Plot->Fill(nChamber, mapStationToIdx_[key4]);
      if (pmapSummary != nullptr)
        (*pmapSummary)[key5] = true;
    }
  };

private:
  int ProcessWithMEMap4(BookingHelper &bh, ME4IdsKey key) override;
  int ProcessWithMEMap5WithChamber(BookingHelper &bh, ME5IdsKey key) override;

  void SetLabelAMC13Status(MonitorElement *h2Status);
  void SetLabelAMCStatus(MonitorElement *h2Status);
  void SetLabelOHStatus(MonitorElement *h2Status);
  void SetLabelVFATStatus(MonitorElement *h2Status);

  const edm::ESGetToken<GEMChMap, GEMChMapRcd> gemChMapToken_;

  edm::EDGetToken tagVFAT_;
  edm::EDGetToken tagOH_;
  edm::EDGetToken tagAMC_;
  edm::EDGetToken tagAMC13_;

  Bool_t bWarnedNotFound_;

  MonitorElement *h2AMC13Status_;

  MEMap4Inf mapStatusOH_;
  MEMap4Inf mapStatusVFAT_;

  MEMap4Inf mapStatusWarnVFATPerLayer_;
  MEMap4Inf mapStatusErrVFATPerLayer_;
  MEMap5Inf mapStatusVFATPerCh_;

  MonitorElement *h2SummaryStatusAll;
  MonitorElement *h2SummaryStatusWarning;
  MonitorElement *h2SummaryStatusError;
  MonitorElement *h2SummaryStatusVFATWarning;
  MonitorElement *h2SummaryStatusVFATError;
  MonitorElement *h2SummaryStatusOHWarning;
  MonitorElement *h2SummaryStatusOHError;
  MonitorElement *h2SummaryStatusAMCWarning;
  MonitorElement *h2SummaryStatusAMCError;
  MonitorElement *h2SummaryStatusAMC13Error;

  std::string strFolderMain_;

  Bool_t bFillAMC_;

  Int_t nBXMin_, nBXMax_;

  std::map<UInt_t, int> mapFEDIdToRe_;
  std::map<UInt_t, int> mapFEDIdToSt_;
  std::map<UInt_t, int> mapFEDIdToPosition_;
  std::map<UInt_t, MonitorElement *> mapFEDIdToAMCStatus_;
  std::map<int, std::vector<GEMDetId>> mapAMC13ToListChamber_;
  std::map<std::tuple<int, int>, std::vector<GEMDetId>> mapAMCToListChamber_;
  Int_t nAMCSlots_;
  Bool_t useDBEMap_;

  int nBitAMC13_ = 10;
  int nBitAMC_ = 12;
  int nBitOH_ = 18;
  int nBitVFAT_ = 8;
};

#endif  // DQM_GEM_INTERFACE_GEMDAQStatusSource_h
