#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
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
#include "RecoTauTag/RecoTau/interface/RecoTauVertexAssociator.h"

#include "TLorentzVector.h"

using namespace reco;
using namespace std;
using namespace edm;

class PFRecoTauDiscriminationByFlightPathSignificance
  : public PFTauDiscriminationProducerBase  {
  public:
    explicit PFRecoTauDiscriminationByFlightPathSignificance(const ParameterSet& iConfig)
      :PFTauDiscriminationProducerBase(iConfig){
      flightPathSig		= iConfig.getParameter<double>("flightPathSig");
      withPVError		= iConfig.getParameter<bool>("UsePVerror");
      booleanOutput 		= iConfig.getParameter<bool>("BooleanOutput");
      //      edm::ConsumesCollector iC(consumesCollector());
      vertexAssociator_ = new reco::tau::RecoTauVertexAssociator(iConfig.getParameter<ParameterSet>("qualityCuts"),consumesCollector());
    }

    ~PFRecoTauDiscriminationByFlightPathSignificance(){}

    void beginEvent(const edm::Event&, const edm::EventSetup&) override;
    double discriminate(const reco::PFTauRef&) override;

  private:
    double threeProngFlightPathSig(const PFTauRef&);
    double vertexSignificance(reco::Vertex&,reco::Vertex&,GlobalVector&);

    reco::tau::RecoTauVertexAssociator* vertexAssociator_;

    bool booleanOutput;
    double flightPathSig;
    bool withPVError;

    const TransientTrackBuilder* transientTrackBuilder;
};

void PFRecoTauDiscriminationByFlightPathSignificance::beginEvent(
    const Event& iEvent, const EventSetup& iSetup){

   vertexAssociator_->setEvent(iEvent);

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

  reco::VertexRef primaryVertex = vertexAssociator_->associatedVertex(*tau);

  if (primaryVertex.isNull()) {
    edm::LogError("FlightPathSignficance") << "Could not get vertex associated"
      << " to tau, returning -999!" << std::endl;
    return -999;
  }

  //Secondary vertex
  const vector<PFCandidatePtr>& pfSignalCandidates = tau->signalPFChargedHadrCands();
  vector<TransientTrack> transientTracks;
  vector<PFCandidatePtr>::const_iterator iTrack;
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
      // We have to un-const the PV for some reason
      reco::Vertex primaryVertexNonConst = *primaryVertex;
      flightPathSignificance = vertexSignificance(primaryVertexNonConst,secVer,tauDir);
    }
  }
  return flightPathSignificance;
}

double PFRecoTauDiscriminationByFlightPathSignificance::vertexSignificance(
    reco::Vertex& pv, Vertex& sv,GlobalVector& direction){
  return SecondaryVertex::computeDist3d(pv,sv,direction,withPVError).significance();
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationByFlightPathSignificance);

