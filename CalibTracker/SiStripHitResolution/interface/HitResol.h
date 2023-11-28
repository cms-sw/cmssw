#ifndef CalibTracker_SiStripHitResolution_HitResol_H
#define CalibTracker_SiStripHitResolution_HitResol_H

// system includes
#include <vector>

// user includes
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

// ROOT includes
#include "TTree.h"
#include "TROOT.h"
#include "TH1F.h"
#include "TH2F.h"

class TrackerTopology;

class HitResol : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit HitResol(const edm::ParameterSet& conf);
  ~HitResol() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  double checkConsistency(const StripClusterParameterEstimator::LocalValues& parameters, double xx, double xerr);
  void getSimHitRes(const GeomDetUnit* det,
                    const LocalVector& trackdirection,
                    const TrackingRecHit& recHit,
                    float& trackWidth,
                    float* pitch,
                    LocalVector& drift);
  double getSimpleRes(const TrajectoryMeasurement* traj1);
  bool getPairParameters(const MagneticField* magField_,
                         AnalyticalPropagator& propagator,
                         const TrajectoryMeasurement* traj1,
                         const TrajectoryMeasurement* traj2,
                         float& pairPath,
                         float& hitDX,
                         float& trackDX,
                         float& trackDXE,
                         float& trackParamX,
                         float& trackParamY,
                         float& trackParamDXDZ,
                         float& trackParamDYDZ,
                         float& trackParamXE,
                         float& trackParamYE,
                         float& trackParamDXDZE,
                         float& trackParamDYDZE);
  typedef std::vector<Trajectory> TrajectoryCollection;

private:
  void beginJob() override;
  void endJob() override;
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  // ----------member data ---------------------------

  // ED tokens
  const edm::EDGetTokenT<LumiScalersCollection> scalerToken_;
  const edm::EDGetTokenT<reco::TrackCollection> combinatorialTracks_token_;
  const edm::EDGetTokenT<std::vector<Trajectory> > tjToken_;
  const edm::EDGetTokenT<reco::TrackCollection> tkToken_;

  // ES tokens
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<StripClusterParameterEstimator, TkStripCPERecord> cpeToken_;
  const edm::ESGetToken<SiStripQuality, SiStripQualityRcd> siStripQualityToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;

  // configuration parameters
  const bool addLumi_;
  const bool DEBUG_;
  const bool cutOnTracks_;
  const double momentumCut_;
  const int compSettings_;
  const unsigned int usePairsOnly_;
  const unsigned int layers_;
  const unsigned int trackMultiplicityCut_;

  // output file
  TTree* reso;
  TTree* treso;
  std::map<TString, TH2F*> histos2d_;

  // conversion
  static constexpr float cmToUm = 10000.f;

  // counters
  int events, EventTrackCKF;

  // Tree declarations
  // Hit Resolution Ntuple Content
  float mymom;
  int numHits;
  int NumberOf_tracks;
  float ProbTrackChi2;
  float StripCPE1_smp_pos_error;
  float StripCPE2_smp_pos_error;
  float StripErrorSquared1;
  float StripErrorSquared2;
  float uerr2;
  float uerr2_2;
  unsigned int clusterWidth;
  unsigned int clusterWidth_2;
  unsigned int clusterCharge;
  unsigned int clusterCharge_2;
  unsigned int iidd1;
  unsigned int iidd2;
  unsigned int pairsOnly;
  float pairPath;
  float mypitch1;
  float mypitch2;
  float expWidth;
  float expWidth_2;
  float driftAlpha;
  float driftAlpha_2;
  float thickness;
  float thickness_2;
  float trackWidth;
  float trackWidth_2;
  float atEdge;
  float atEdge_2;
  float simpleRes;
  float hitDX;
  float trackDX;
  float trackDXE;
  float trackParamX;
  float trackParamY;
  float trackParamXE;
  float trackParamYE;
  float trackParamDXDZ;
  float trackParamDYDZ;
  float trackParamDXDZE;
  float trackParamDYDZE;
  float track_momentum;
  float track_pt;
  float track_eta;
  float track_width;
  float track_phi;
  float track_trackChi2;
  float N1;
  float N2;
  float N1uProj;
  float N2uProj;
  int Nstrips;
  int Nstrips_2;
};

#endif
