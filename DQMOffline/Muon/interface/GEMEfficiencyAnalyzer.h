#ifndef DQMOffline_Muon_GEMEfficiencyAnalyzer_h
#define DQMOffline_Muon_GEMEfficiencyAnalyzer_h

/** \class GEMEfficiencyAnalyzer
 * 
 * DQM monitoring source for GEM efficiency and resolution
 * based on https://github.com/CPLUOS/MuonPerformance/blob/master/MuonAnalyser/plugins/SliceTestEfficiencyAnalysis.cc
 *
 * \author Seungjin Yang <seungjin.yang@cern.ch>
 */

#include "DQMOffline/Muon/interface/GEMOfflineDQMBase.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

class GEMEfficiencyAnalyzer : public GEMOfflineDQMBase {
public:
  explicit GEMEfficiencyAnalyzer(const edm::ParameterSet &);
  ~GEMEfficiencyAnalyzer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions &);

protected:
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &event, const edm::EventSetup &eventSetup) override;

private:
  struct GEMLayerData {
    GEMLayerData(Disk::DiskPointer surface, std::vector<const GEMChamber *> chambers, int region, int station, int layer)
        : surface(surface), chambers(chambers), region(region), station(station), layer(layer) {}

    Disk::DiskPointer surface;
    std::vector<const GEMChamber *> chambers;
    int region, station, layer;
  };

  MonitorElement *bookNumerator1D(DQMStore::IBooker &, MonitorElement *);
  MonitorElement *bookNumerator2D(DQMStore::IBooker &, MonitorElement *);

  void bookEfficiencyMomentum(DQMStore::IBooker &, const edm::ESHandle<GEMGeometry> &);
  void bookEfficiencyChamber(DQMStore::IBooker &, const edm::ESHandle<GEMGeometry> &);
  void bookEfficiencyEtaPartition(DQMStore::IBooker &, const edm::ESHandle<GEMGeometry> &);
  void bookResolution(DQMStore::IBooker &, const edm::ESHandle<GEMGeometry> &);
  void bookMisc(DQMStore::IBooker &, const edm::ESHandle<GEMGeometry> &);

  inline bool isInsideOut(const reco::Track &);

  std::vector<GEMLayerData> buildGEMLayers(const edm::ESHandle<GEMGeometry> &);
  const reco::Track *getTrack(const reco::Muon &);
  std::pair<TrajectoryStateOnSurface, DetId> getStartingState(const reco::TransientTrack &,
                                                              const GEMLayerData &,
                                                              const edm::ESHandle<GlobalTrackingGeometry> &);
  std::pair<TrajectoryStateOnSurface, DetId> findStartingState(const reco::TransientTrack &,
                                                               const GEMLayerData &,
                                                               const edm::ESHandle<GlobalTrackingGeometry> &);
  bool isME11(const DetId &);
  bool skipLayer(const reco::Track *, const GEMLayerData &);
  bool checkBounds(const GlobalPoint &, const Plane &);
  const GEMEtaPartition *findEtaPartition(const GlobalPoint &, const std::vector<const GEMChamber *> &);
  std::pair<const GEMRecHit *, float> findClosetHit(const GlobalPoint &,
                                                    const GEMRecHitCollection::range &,
                                                    const GEMEtaPartition *);

  // data members

  // parameters
  std::string name_;
  std::string folder_;
  edm::EDGetTokenT<GEMRecHitCollection> rechit_token_;
  edm::EDGetTokenT<edm::View<reco::Muon> > muon_token_;
  bool is_cosmics_;
  bool use_global_muon_;
  bool use_skip_layer_;
  bool use_only_me11_;
  float residual_rphi_cut_;
  bool use_prop_r_error_cut_;
  double prop_r_error_cut_;
  bool use_prop_phi_error_cut_;
  double prop_phi_error_cut_;
  std::vector<double> pt_bins_;
  int eta_nbins_;
  double eta_low_;
  double eta_up_;

  // data mebers derived from parameters
  MuonServiceProxy *muon_service_;
  double pt_clamp_max_;
  double eta_clamp_max_;

  // MonitorElement
  // efficiency
  MEMap me_muon_pt_;  // 1D, region-station
  MEMap me_muon_pt_matched_;
  MEMap me_muon_eta_;  // 1D, region-station
  MEMap me_muon_eta_matched_;
  MEMap me_muon_phi_;  // 1D, region-station
  MEMap me_muon_phi_matched_;
  MEMap me_chamber_;  // 2D, region-station-layer
  MEMap me_chamber_matched_;
  MEMap me_detector_;  // 2D, region-station
  MEMap me_detector_matched_;
  // resolution
  MEMap me_residual_rphi_;  // global
  MEMap me_residual_y_;     // local
  MEMap me_pull_y_;
  // MEs for optimizing cut values
  MonitorElement *me_prop_r_err_;    // clamped
  MonitorElement *me_prop_phi_err_;  // clamped
  MonitorElement *me_all_abs_residual_rphi_;
  MEMap me_prop_chamber_;

  // const
  const std::string kLogCategory_ = "GEMEfficiencyAnalyzer";
};

inline bool GEMEfficiencyAnalyzer::isInsideOut(const reco::Track &track) {
  return track.innerPosition().mag2() > track.outerPosition().mag2();
}

#endif  // DQMOffline_Muon_GEMEfficiencyAnalyzer_h
