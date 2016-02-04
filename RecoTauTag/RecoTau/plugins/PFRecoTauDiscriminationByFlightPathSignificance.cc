#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "RecoTauTag/TauTagTools/interface/PFTauQualityCutWrapper.h"
#include "FWCore/Utilities/interface/InputTag.h"

/* class PFRecoTauDiscriminationByFlightPathSignificance
 * created : August 30 2010,
 * contributors : Sami Lehti (sami.lehti@cern.ch ; HIP, Helsinki)
 * based on H+ tau ID by Lauri Wendland
 */

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoBTag/SecondaryVertex/interface/SecondaryVertex.h"

#include "TLorentzVector.h"

using namespace reco;
using namespace std;
using namespace edm;

class PFRecoTauDiscriminationByFlightPathSignificance
: public PFTauDiscriminationProducerBase  {
  public:
    explicit PFRecoTauDiscriminationByFlightPathSignificance(const ParameterSet& iConfig):PFTauDiscriminationProducerBase(iConfig),
    qualityCuts_(iConfig.getParameter<ParameterSet>("qualityCuts")){  // retrieve quality cuts
      flightPathSig		= iConfig.getParameter<double>("flightPathSig");
      withPVError		= iConfig.getParameter<bool>("UsePVerror");

      PVProducer		= iConfig.getParameter<edm::InputTag>("PVProducer");

      booleanOutput 		= iConfig.getParameter<bool>("BooleanOutput");
    }

    ~PFRecoTauDiscriminationByFlightPathSignificance(){}

    void beginEvent(const edm::Event&, const edm::EventSetup&);
    double discriminate(const reco::PFTauRef&);

  private:
    double threeProngFlightPathSig(const PFTauRef&);
    double vertexSignificance(reco::Vertex&,reco::Vertex&,GlobalVector&);

    PFTauQualityCutWrapper qualityCuts_;

    double flightPathSig;
    bool withPVError;

    reco::Vertex primaryVertex;
    const TransientTrackBuilder* transientTrackBuilder;
    edm::InputTag PVProducer;

    bool booleanOutput;
};

void PFRecoTauDiscriminationByFlightPathSignificance::beginEvent(const Event& iEvent, const EventSetup& iSetup){

  //Primary vertex
  edm::Handle<edm::View<reco::Vertex> > vertexHandle;
  iEvent.getByLabel(PVProducer, vertexHandle);
  const edm::View<reco::Vertex>& vertexCollection(*vertexHandle);

  primaryVertex = *(vertexCollection.begin());

  // Transient Tracks
  edm::ESHandle<TransientTrackBuilder> builder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",builder);
  transientTrackBuilder = builder.product();

}

double PFRecoTauDiscriminationByFlightPathSignificance::discriminate(const PFTauRef& tau){

  if(booleanOutput) return ( threeProngFlightPathSig(tau) > flightPathSig ? 1. : 0. );
  return threeProngFlightPathSig(tau);
}

double PFRecoTauDiscriminationByFlightPathSignificance::threeProngFlightPathSig(
    const PFTauRef& tau){
  double flightPathSignificance = 0;

  //Secondary vertex
  const PFCandidateRefVector pfSignalCandidates = tau->signalPFChargedHadrCands();
  vector<TransientTrack> transientTracks;
  RefVector<PFCandidateCollection>::const_iterator iTrack;
  for(iTrack = pfSignalCandidates.begin(); iTrack!= pfSignalCandidates.end(); iTrack++){
    const PFCandidate& pfCand = *(iTrack->get());
    if(pfCand.trackRef().isNonnull()){
      const TransientTrack transientTrack = transientTrackBuilder->build(pfCand.trackRef());
      transientTracks.push_back(transientTrack);
    }
  }
  if(transientTracks.size() > 1){
    KalmanVertexFitter kvf(true);
    TransientVertex tv = kvf.vertex(transientTracks);

    if(tv.isValid()){
      GlobalVector tauDir(tau->px(),
          tau->py(),
          tau->pz());
      Vertex secVer = tv;
      flightPathSignificance = vertexSignificance(primaryVertex,secVer,tauDir);
    }
  }
  return flightPathSignificance;
}

double PFRecoTauDiscriminationByFlightPathSignificance::vertexSignificance(
    reco::Vertex& pv, Vertex& sv,GlobalVector& direction){
  return SecondaryVertex::computeDist3d(pv,sv,direction,withPVError).significance();
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationByFlightPathSignificance);

