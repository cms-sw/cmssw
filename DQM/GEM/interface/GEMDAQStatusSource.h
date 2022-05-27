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

#include "CondFormats/DataRecord/interface/GEMeMapRcd.h"
#include "CondFormats/GEMObjects/interface/GEMeMap.h"
#include "CondFormats/GEMObjects/interface/GEMROMapping.h"
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

  void FillStatusSummaryPlot(std::map<ME4IdsKey, bool> &mapChamber,
                             MonitorElement *h2Plot,
                             std::map<ME4IdsKey, bool> *pmapSummary = nullptr) {
    for (auto const &[key4, bFlag] : mapChamber) {  // bFlag is not used
      ME3IdsKey key3 = key4Tokey3(key4);
      Int_t nChamber = keyToChamber(key4);
      h2Plot->Fill(nChamber, mapStationToIdx_[key3]);
      if (pmapSummary != nullptr)
        (*pmapSummary)[key4] = true;
    }
  };

private:
  int ProcessWithMEMap3(BookingHelper &bh, ME3IdsKey key) override;
  int ProcessWithMEMap3WithChamber(BookingHelper &bh, ME4IdsKey key) override;

  void SetLabelAMC13Status(MonitorElement *h2Status);
  void SetLabelAMCStatus(MonitorElement *h2Status);
  void SetLabelOHStatus(MonitorElement *h2Status);
  void SetLabelVFATStatus(MonitorElement *h2Status);

  edm::ESGetToken<GEMeMap, GEMeMapRcd> gemEMapToken_;
  //std::shared_ptr<GEMROMapping> gemROMap_;
  const GEMeMap *gemEMap_;

  edm::EDGetToken tagVFAT_;
  edm::EDGetToken tagOH_;
  edm::EDGetToken tagAMC_;
  edm::EDGetToken tagAMC13_;

  MonitorElement *h2AMC13Status_;
  MonitorElement *h2AMCStatusPos_;
  MonitorElement *h2AMCStatusNeg_;

  MEMap3Inf mapStatusOH_;
  MEMap3Inf mapStatusVFAT_;

  MEMap3Inf mapStatusWarnVFATPerLayer_;
  MEMap3Inf mapStatusErrVFATPerLayer_;
  MEMap4Inf mapStatusVFATPerCh_;

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

  Int_t nBXMin_, nBXMax_;

  std::map<UInt_t, int> mapFEDIdToRe_;
  std::map<int, std::vector<GEMDetId>> mapAMC13ToListChamber_;
  std::map<std::tuple<int, int>, std::vector<GEMDetId>> mapAMCToListChamber_;
  Int_t nAMCSlots_;

  int nBitAMC13_ = 10;
  int nBitAMC_ = 12;
  int nBitOH_ = 17;
  int nBitVFAT_ = 7;
};

#endif  // DQM_GEM_INTERFACE_GEMDAQStatusSource_h
