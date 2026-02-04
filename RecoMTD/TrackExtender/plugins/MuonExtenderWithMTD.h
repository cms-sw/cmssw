#ifndef RecoMTD_TrackExtender_MuonExtenderWithMTD_h
#define RecoMTD_TrackExtender_MuonExtenderWithMTD_h

#include <CLHEP/Units/GlobalPhysicalConstants.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

#include "RecoMTD/TrackExtender/interface/BaseExtenderWithMTD.h"
#include "RecoMTD/TransientTrackingRecHit/interface/MTDTransientTrackingRecHitBuilder.h"

#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

using namespace std;
using namespace edm;
using namespace reco;
using namespace mtdtof;

namespace mtdtof{

  struct MuonTofPidInfo {
    float tmtd;
    float tmtderror;
    float pathlength;

    float betaerror;

    float dt;
    float dterror;
    float dterror2;
    float dtchi2;

    float dt_best;
    float dterror_best;
    float dtchi2_best;

    float gammasq_mu;
    float beta_mu;
    float dt_mu;
    float sigma_dt_mu;  
  };

  const MuonTofPidInfo computeMuonTofPidInfo(float magp2,
                                             float length,
                                             TrackSegments trs,
                                             float t_mtd,
                                             float t_mtderr,
                                             float t_vtx,
                                             float t_vtx_err,
                                             bool addPIDError = true,
                                             TofCalc choice = TofCalc::kCost,
                                             SigmaTofCalc sigma_choice = SigmaTofCalc::kCost);

  bool muonPathLength(const Trajectory& traj,
                      const TrajectoryStateClosestToBeamLine& tscbl,
                      const Propagator* thePropagator,
                      float& pathlength,
                      TrackSegments& trs);

  bool muonPathLength(const Trajectory& traj,
                      const reco::BeamSpot& bs,
                      const Propagator* thePropagator,
                      float& pathlength,
                      TrackSegments& trs);    

  void find_hits_in_dets_muon(const MTDTrackingDetSetVector& hits,
                              const Trajectory& traj,
                              const DetLayer* layer,
                              const TrajectoryStateOnSurface& tsos,
                              const float pmag2,
                              const float pathlength0,
                              const TrackSegments& trs0,
                              const float vtxTime,
                              const float vtxTimeError,
                              bool useVtxConstraint,
                              const reco::BeamSpot& bs,
                              const float bsTimeSpread,
                              const Propagator* prop,
                              const MeasurementEstimator* estimator,
                              std::set<MTDHitMatchingInfo>& out);
}

class MuonBaseExtenderWithMTD : public BaseExtenderWithMTD {
  public:
    explicit MuonBaseExtenderWithMTD(const ParameterSet& iConfig);
    ~MuonBaseExtenderWithMTD() override;

    void fillMatchingHits(const DetLayer*,
                          const TrajectoryStateOnSurface&,
                          const Trajectory&,
                          const float,
                          const float,
                          const TrackSegments&,
                          const MTDTrackingDetSetVector&,
                          const Propagator*,
                          const reco::BeamSpot&,
                          const float&,
                          const float&,
                          TransientTrackingRecHit::ConstRecHitContainer&,
                          MTDHitMatchingInfo&) const override;

    reco::Track buildTrack(const reco::TrackBase::TrackAlgorithm,
                           const Trajectory&,
                           const Trajectory&,
                           const reco::BeamSpot&,
                           const MagneticField* field,
                           const Propagator* prop,
                           bool hasMTD,
                           float& pathLength,
                           float& tmtdOut,
                           float& sigmatmtdOut,
                           GlobalPoint& tmtdPosOut,
                           float& tofmu,
                           float& sigmatofmu) const;
};

template <typename T1>
class MuonExtenderWithMTDT : public edm::stream::EDProducer<> {
  public:
    MuonExtenderWithMTDT(const ParameterSet& pset);

    template <class H, class T>
    void fillValueMap(edm::Event& iEvent, const H& handle, const std::vector<T>& vec, const edm::EDPutToken& token) const;

    void produce(edm::Event& ev, const edm::EventSetup& es) final;

  private:
    edm::EDPutToken btlMatchChi2Token_;
    edm::EDPutToken etlMatchChi2Token_;
    edm::EDPutToken btlMatchTimeChi2Token_;
    edm::EDPutToken etlMatchTimeChi2Token_;
    edm::EDPutToken npixBarrelToken_;
    edm::EDPutToken npixEndcapToken_;
    edm::EDPutToken outermostHitPositionToken_;
    edm::EDPutToken pOrigTrkToken_;
    edm::EDPutToken betaOrigTrkToken_;
    edm::EDPutToken t0OrigTrkToken_;
    edm::EDPutToken sigmat0OrigTrkToken_;
    edm::EDPutToken pathLengthOrigTrkToken_;
    edm::EDPutToken tmtdOrigTrkToken_;
    edm::EDPutToken sigmatmtdOrigTrkToken_;
    edm::EDPutToken tmtdPosOrigTrkToken_;
    edm::EDPutToken tofmuOrigTrkToken_;
    edm::EDPutToken sigmatofmuOrigTrkToken_;
    edm::EDPutToken assocOrigTrkToken_;

    edm::EDGetTokenT<T1> muonToken_;
    edm::EDGetTokenT<MTDTrackingDetSetVector> hitsToken_;
    edm::EDGetTokenT<reco::BeamSpot> bsToken_;
    edm::EDGetTokenT<VertexCollection> vtxToken_;

    const bool updateTraj_, updateExtra_, updatePattern_;
    const std::string mtdRecHitBuilder_, propagator_, transientTrackBuilder_;

    std::unique_ptr<TrackTransformer> theTransformer;
    edm::ESHandle<TransientTrackBuilder> builder_;
    edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> builderToken_;
    edm::ESHandle<TransientTrackingRecHitBuilder> hitbuilder_;
    edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> hitbuilderToken_;
    edm::ESHandle<GlobalTrackingGeometry> gtg_;
    edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> gtgToken_;

    edm::ESGetToken<MTDDetLayerGeometry, MTDRecoGeometryRecord> dlgeoToken_;
    edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magfldToken_;
    edm::ESGetToken<Propagator, TrackingComponentsRecord> propToken_;
    edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> ttopoToken_;

    const bool useVertex_;

    static constexpr float trackMaxBtlEta_ = 1.5;

    std::unique_ptr<MuonBaseExtenderWithMTD> baseMTDExtender_;
};

#endif
