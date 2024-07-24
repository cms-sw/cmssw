#ifndef DQM_GEM_INTERFACE_GEMPadDigiClusterSource_h
#define DQM_GEM_INTERFACE_GEMPadDigiClusterSource_h

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

#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCluster.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiClusterCollection.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"

#include "DQM/GEM/interface/GEMDQMBase.h"

#include <string>

//----------------------------------------------------------------------------------------------------

class GEMPadDigiClusterSource : public GEMDQMBase {
public:
  explicit GEMPadDigiClusterSource(const edm::ParameterSet& cfg);
  ~GEMPadDigiClusterSource() override{};
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override{};
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& e, edm::EventSetup const& eSetup) override;

private:
  int ProcessWithMEMap2WithEta(BookingHelper& bh, ME3IdsKey key) override;
  int ProcessWithMEMap2(BookingHelper& bh, ME2IdsKey key) override;
  int ProcessWithMEMap3(BookingHelper& bh, ME3IdsKey key) override;

  int ProcessWithMEMap2WithChamber(BookingHelper& bh, ME3IdsKey key) override;

  //int ProcessWithMEMap2WithEtaCh(BookingHelper& bh, ME4IdsKey key) override;

  int ProcessWithMEMap4WithChamber(BookingHelper& bh, ME4IdsKey key) override;

  const static int nNumBitDigiOcc_ = 16384;

  edm::EDGetToken tagPadDigiCluster_;

  edm::EDGetTokenT<LumiScalersCollection> lumiScalers_;

  MEMap3Inf mapPadDiffPerCh_;
  MEMap3Inf mapBXDiffPerCh_;
  MEMap3Inf mapPadBXDiffPerCh_;

  MEMap4Inf mapPadBXDiffPerEtaCh_;
  MEMap4Inf mapPadDigiOccPerCh_;
  MEMap4Inf mapPadBxPerCh_;
  MEMap4Inf mapPadCLSPerCh_;
  MEMap4Inf mapPadDiffPerEtaCh_;
  MEMap4Inf mapBXMidPerCh_;
  MEMap4Inf mapBXCLSPerCh_;
  std::string strFolderMain_;

  Int_t nBXMin_, nBXMax_;
  Int_t nCLSMax_, nClusterSizeBinNum_;
  Float_t fRadiusMin_;
  Float_t fRadiusMax_;
};

#endif  // DQM_GEM_INTERFACE_GEMDigiSource_h
