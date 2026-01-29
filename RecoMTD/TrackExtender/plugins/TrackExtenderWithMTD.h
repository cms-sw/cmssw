#ifndef RecoMTD_TrackExtender_TrackExtenderWithMTD_h
#define RecoMTD_TrackExtender_TrackExtenderWithMTD_h

#include <CLHEP/Units/GlobalPhysicalConstants.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "RecoMTD/TrackExtender/interface/BaseExtenderWithMTD.h"
#include "RecoMTD/TransientTrackingRecHit/interface/MTDTransientTrackingRecHitBuilder.h"

#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

using namespace std;
using namespace edm;
using namespace reco;

template <class TrackCollection>
class TrackExtenderWithMTDT : public edm::stream::EDProducer<>{
  public:
    TrackExtenderWithMTDT(const ParameterSet& pset);

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
    edm::EDPutToken tofpiOrigTrkToken_;
    edm::EDPutToken tofkOrigTrkToken_;
    edm::EDPutToken tofpOrigTrkToken_;
    edm::EDPutToken sigmatofpiOrigTrkToken_;
    edm::EDPutToken sigmatofkOrigTrkToken_;
    edm::EDPutToken sigmatofpOrigTrkToken_;
    edm::EDPutToken assocOrigTrkToken_;

    edm::EDGetTokenT<TrackCollection> tracksToken_;
    edm::EDGetTokenT<TrajTrackAssociationCollection> trajTrackAToken_;
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

    std::unique_ptr<BaseExtenderWithMTD> baseMTDExtender_;
};

#endif
