#ifndef DQM_GEM_GEMEfficiencyAnalyzer_h
#define DQM_GEM_GEMEfficiencyAnalyzer_h

/** \class GEMEfficiencyAnalyzer
 *
 * DQM monitoring source for GEM efficiency and resolution
 * based on https://github.com/CPLUOS/MuonPerformance/blob/master/MuonAnalyser/plugins/SliceTestEfficiencyAnalysis.cc
 *
 * TODO muonEta{Min,Max}Cut{GE11,GE21,GE0} depending on a magnetic field for
 *      a cosmic scenario
 * TODO cscForGE21, cscForGE0
 * TODO use "StraightLinePropagator" if B=0 for a cosmic scenario
 *
 * \author Seungjin Yang <seungjin.yang@cern.ch>
 */

#include "DQM/GEM/interface/GEMDQMEfficiencySourceBase.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

class GEMEfficiencyAnalyzer : public GEMDQMEfficiencySourceBase {
public:
  explicit GEMEfficiencyAnalyzer(const edm::ParameterSet &);
  ~GEMEfficiencyAnalyzer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions &);

  // currently only for STA muons
  enum class StartingStateType {
    kOutermostMeasurementState = 0,
    kInnermostMeasurementState,
    kStateOnSurfaceWithCSCSegment,
    kAlignmentStyle,
  };

  // Define the metric as the smaller the absolute value, the better the matching.
  enum class MatchingMetric {
    kDeltaPhi = 0,  // computeDeltaPhi
    kRdPhi,         // computeRdPhi
  };

  // https://github.com/cms-sw/cmssw/blob/CMSSW_12_4_0_pre3/Configuration/Applications/python/ConfigBuilder.py#L35
  enum class ScenarioOption {
    kPP = 0,
    kCosmics,
    kHeavyIons,
  };

  struct GEMLayer {
    GEMLayer(Disk::DiskPointer disk, std::vector<const GEMChamber *> chambers, GEMDetId id)
        : disk(disk), chambers(chambers), id(id) {}
    Disk::DiskPointer disk;
    std::vector<const GEMChamber *> chambers;
    GEMDetId id;
  };

  using StartingState = std::tuple<bool, TrajectoryStateOnSurface, DetId>;

protected:
  void dqmBeginRun(edm::Run const &, edm::EventSetup const &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &event, const edm::EventSetup &eventSetup) override;

private:
  StartingStateType getStartingStateType(const std::string);
  MatchingMetric getMatchingMetric(const std::string);
  reco::Muon::MuonTrackType getMuonTrackType(const std::string);
  ScenarioOption getScenarioOption(const std::string);

  void buildGEMLayers(const GEMGeometry *);

  bool checkPropagationDirection(const reco::Track *, const GEMLayer &);

  StartingState buildStartingState(const reco::Muon &, const reco::TransientTrack &, const GEMLayer &);
  StartingState getInnermostMeasurementState(const reco::TransientTrack &);
  StartingState getOutermostMeasurementState(const reco::TransientTrack &);
  StartingState buildStateOnSurfaceWithCSCSegment(const reco::Muon &, const reco::TransientTrack &, const GEMLayer &);
  StartingState buildStartingStateAlignmentStyle(const reco::Muon &, const reco::TransientTrack &, const GEMLayer &);

  // for kStateOnSurfaceWithCSCSegment and AlignmentStyle
  const CSCSegment *findCSCSegment(const reco::Muon &, const reco::TransientTrack &, const GEMLayer &);
  const CSCSegment *findCSCSegmentBeam(const reco::TransientTrack &, const GEMLayer &);
  const CSCSegment *findCSCSegmentCosmics(const reco::Muon &, const GEMLayer &);
  bool isMuonSubdetAllowed(const DetId &, const int);
  bool isCSCAllowed(const CSCDetId &, const int);

  bool checkBounds(const Plane &, const GlobalPoint &);
  bool checkBounds(const Plane &, const GlobalPoint &, const GlobalError &, float);
  const GEMEtaPartition *findEtaPartition(const GlobalPoint &,
                                          const GlobalError &,
                                          const std::vector<const GEMChamber *> &);

  float computeRdPhi(const GlobalPoint &, const LocalPoint &, const GEMEtaPartition *);
  float computeDeltaPhi(const GlobalPoint &, const LocalPoint &, const GEMEtaPartition *);
  float computeMatchingMetric(const GlobalPoint &, const LocalPoint &, const GEMEtaPartition *);

  std::pair<const GEMRecHit *, float> findClosestHit(const GlobalPoint &,
                                                     const GEMRecHitCollection::range &,
                                                     const GEMEtaPartition *);

  // some helpers
  inline bool isInsideOut(const reco::Track &);

  //////////////////////////////////////////////////////////////////////////////
  // const data members initialized in the member initializer list
  // mainly retrieved from edm::ParameterSet
  //////////////////////////////////////////////////////////////////////////////
  // ES
  const edm::ESGetToken<GEMGeometry, MuonGeometryRecord> kGEMGeometryTokenBeginRun_;
  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> kTransientTrackBuilderToken_;
  // ED
  const edm::EDGetTokenT<GEMRecHitCollection> kGEMRecHitCollectionToken_;
  const edm::EDGetTokenT<edm::View<reco::Muon> > kMuonViewToken_;
  //
  const std::string kMuonTrackTypeName_;
  const reco::Muon::MuonTrackType kMuonTrackType_;
  const TString kMuonName_;
  const std::string kFolder_;
  const ScenarioOption kScenario_;
  // cuts
  const StartingStateType kStartingStateType_;
  const std::vector<std::vector<int> > kMuonSubdetForGEM_;
  const std::vector<std::vector<int> > kCSCForGEM_;  // when using StartingStateType::kStateOnSurfaceWithCSCSegment
  const float kMuonSegmentMatchDRCut_;               // for cosmics

  const std::vector<double> kMuonPtMinCuts_;   // station as index
  const std::vector<double> kMuonEtaMinCuts_;  // station as index
  const std::vector<double> kMuonEtaMaxCuts_;  // station as index
  const float kPropagationErrorRCut_;          // cm
  const float kPropagationErrorPhiCut_;        // degree
  const float kBoundsErrorScale_;              // TODO doc
  // matching
  const MatchingMetric kMatchingMetric_;
  const float kMatchingCut_;
  // for MinotorElement
  const std::vector<double> kMuonPtBins_;  // station as index
  const std::vector<int> kMuonEtaNbins_;   // station as index
  const std::vector<double> kMuonEtaLow_;  // station as index
  const std::vector<double> kMuonEtaUp_;   // station as index

  // const
  const bool kModeDev_;

  //////////////////////////////////////////////////////////////////////////////
  // const data members
  // FIXME static?
  //////////////////////////////////////////////////////////////////////////////
  // https://github.com/cms-sw/cmssw/blob/CMSSW_12_4_0_pre3/DataFormats/CSCRecHit/interface/CSCSegment.h#L60
  const int kCSCSegmentDimension_ = 4;

  //////////////////////////////////////////////////////////////////////////////
  // non-const data members
  //////////////////////////////////////////////////////////////////////////////
  std::unique_ptr<MuonServiceProxy> muon_service_;
  std::vector<GEMLayer> gem_layers_;

  // montitor elements
  // XXX how about introducing EffPair ?
  MEMap me_chamber_ieta_, me_chamber_ieta_matched_;
  MEMap me_muon_pt_, me_muon_pt_matched_;
  MEMap me_muon_eta_, me_muon_eta_matched_;
  MEMap me_muon_phi_, me_muon_phi_matched_;
  MEMap me_residual_phi_;  // in global
  // dev mode
  MEMap me_matching_metric_all_;
  MEMap me_matching_metric_;
  MEMap me_residual_phi_muon_;      // in global
  MEMap me_residual_phi_antimuon_;  // in global
  MEMap me_residual_x_;             // in local
  MEMap me_residual_y_;             // in global
  MEMap me_residual_strip_;
  MEMap me_prop_path_length_, me_prop_path_length_matched_;
  MEMap me_prop_err_r_, me_prop_err_r_matched_;
  MEMap me_prop_err_phi_, me_prop_err_phi_matched_;
  MEMap me_muon_pt_all_, me_muon_pt_all_matched_;
  MEMap me_muon_eta_all_, me_muon_eta_all_matched_;
  MEMap me_muon_charge_, me_muon_charge_matched_;
  MEMap me_cutflow_, me_cutflow_matched_;
};

#endif  // DQM_GEM_GEMEfficiencyAnalyzer_h
