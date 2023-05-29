#ifndef DQM_GEM_INTERFACE_GEMDigiSource_h
#define DQM_GEM_INTERFACE_GEMDigiSource_h

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

#include "Validation/MuonGEMHits/interface/GEMValidationUtils.h"

#include "CondFormats/DataRecord/interface/GEMChMapRcd.h"
#include "CondFormats/GEMObjects/interface/GEMChMap.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"

#include "DataFormats/Scalers/interface/LumiScalers.h"

#include "DQM/GEM/interface/GEMDQMBase.h"

#include <string>

//----------------------------------------------------------------------------------------------------

class GEMDigiSource : public GEMDQMBase {
public:
  explicit GEMDigiSource(const edm::ParameterSet& cfg);
  ~GEMDigiSource() override{};
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void LoadROMap(edm::EventSetup const& iSetup);

  void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override{};
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& e, edm::EventSetup const& eSetup) override;

private:
  int ProcessWithMEMap2WithEta(BookingHelper& bh, ME3IdsKey key) override;
  int ProcessWithMEMap2(BookingHelper& bh, ME2IdsKey key) override;
  int ProcessWithMEMap3(BookingHelper& bh, ME3IdsKey key) override;
  int ProcessWithMEMap3WithChamber(BookingHelper& bh, ME4IdsKey key) override;

  const static int nNumBitDigiOcc_ = 16384;

  const edm::ESGetToken<GEMChMap, GEMChMapRcd> gemChMapToken_;

  edm::EDGetToken tagDigi_;

  edm::EDGetTokenT<LumiScalersCollection> lumiScalers_;

  std::map<ME4IdsKey, Int_t> mapChamberType_;
  std::map<ME3IdsKey, Int_t> mapStripToVFAT_;

  MEMap3Inf mapTotalDigi_layer_;
  MEMap3Inf mapDigiWheel_layer_;
  MEMap3Inf mapDigiOcc_ieta_;
  MEMap3Inf mapDigiOcc_phi_;
  MEMap3Inf mapTotalDigiPerEvtLayer_;
  MEMap3Inf mapTotalDigiPerEvtIEta_;
  MEMap2Inf mapBX_;

  MEMap4Inf mapDigiOccPerCh_;

  std::string strFolderMain_;

  Int_t nBXMin_, nBXMax_;
  Float_t fRadiusMin_;
  Float_t fRadiusMax_;
};

#endif  // DQM_GEM_INTERFACE_GEMDigiSource_h
