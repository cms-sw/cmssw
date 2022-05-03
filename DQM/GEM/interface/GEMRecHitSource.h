#ifndef DQM_GEM_INTERFACE_GEMRecHitSource_h
#define DQM_GEM_INTERFACE_GEMRecHitSource_h

#include "DQM/GEM/interface/GEMDQMBase.h"

#include "DataFormats/GEMRecHit/interface/GEMRecHit.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"

#include <string>

//----------------------------------------------------------------------------------------------------

class GEMRecHitSource : public GEMDQMBase {
public:
  explicit GEMRecHitSource(const edm::ParameterSet& cfg);
  ~GEMRecHitSource() override{};
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override{};
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& e, edm::EventSetup const& eSetup) override;

private:
  int ProcessWithMEMap2WithEta(BookingHelper& bh, ME3IdsKey key) override;
  int ProcessWithMEMap2AbsReWithEta(BookingHelper& bh, ME3IdsKey key) override;
  int ProcessWithMEMap3(BookingHelper& bh, ME3IdsKey key) override;
  int ProcessWithMEMap3WithChamber(BookingHelper& bh, ME4IdsKey key) override;

  edm::EDGetToken tagRecHit_;

  int nIdxFirstDigi_;
  int nClusterSizeBinNum_;
  int nNumDivideEtaPartitionInRPhi_;

  MEMap3Inf mapRecHitXY_layer_;
  MEMap3Inf mapRecHitWheel_layer_;
  MEMap3Inf mapRecHitOcc_ieta_;
  MEMap3Inf mapRecHitOcc_phi_;
  MEMap3Inf mapTotalRecHitPerEvtLayer_;
  MEMap3Inf mapTotalRecHitPerEvtIEta_;
  MEMap3Inf mapCLSRecHit_ieta_;
  MEMap3Inf mapCLSAverage_;
  MEMap3Inf mapCLSOver5_;

  MEMap4Inf mapCLSPerCh_;

  std::string strFolderMain_;

  Int_t nCLSMax_;
  Float_t fRadiusMin_;
  Float_t fRadiusMax_;

  std::unordered_map<UInt_t, MonitorElement*> recHitME_;
  std::unordered_map<UInt_t, MonitorElement*> VFAT_vs_ClusterSize_;
  std::unordered_map<UInt_t, MonitorElement*> DigisFired_vs_eta_;
  std::unordered_map<UInt_t, MonitorElement*> rh_vs_eta_;
  std::unordered_map<UInt_t, MonitorElement*> recGlobalPos;
};

#endif  // DQM_GEM_INTERFACE_GEMRecHitSource_h
