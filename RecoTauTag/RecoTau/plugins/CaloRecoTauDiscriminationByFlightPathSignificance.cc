#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "FWCore/Utilities/interface/InputTag.h"

/* class CaloRecoTauDiscriminationByFlightPathSignificance
 * created : September 23 2010,
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

namespace { 

using namespace reco;
using namespace std;

class CaloRecoTauDiscriminationByFlightPathSignificance final : public CaloTauDiscriminationProducerBase  {
  public:
    explicit CaloRecoTauDiscriminationByFlightPathSignificance(
        const edm::ParameterSet& iConfig)
        :CaloTauDiscriminationProducerBase(iConfig) {
      flightPathSig		= iConfig.getParameter<double>("flightPathSig");
      withPVError		= iConfig.getParameter<bool>("UsePVerror");

      PVProducer		= iConfig.getParameter<edm::InputTag>("PVProducer");

      booleanOutput 		= iConfig.getParameter<bool>("BooleanOutput");
    }
    ~CaloRecoTauDiscriminationByFlightPathSignificance() override{}
    void beginEvent(const edm::Event&, const edm::EventSetup&) override;
    double discriminate(const reco::CaloTauRef&) const override;

  private:
    double threeProngFlightPathSig(const CaloTauRef&) const ;
    double vertexSignificance(reco::Vertex const&,reco::Vertex const &,GlobalVector const &) const ;

    double flightPathSig;
    bool withPVError;

    reco::Vertex primaryVertex;
    const TransientTrackBuilder* transientTrackBuilder;
    edm::InputTag PVProducer;

    bool booleanOutput;
};

void CaloRecoTauDiscriminationByFlightPathSignificance::beginEvent(
    const edm::Event& iEvent, const edm::EventSetup& iSetup){
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

double
CaloRecoTauDiscriminationByFlightPathSignificance::discriminate(
    const CaloTauRef& tau) const {
  if(booleanOutput)
    return ( threeProngFlightPathSig(tau) > flightPathSig ? 1. : 0. );
  return threeProngFlightPathSig(tau);
}

double
CaloRecoTauDiscriminationByFlightPathSignificance::threeProngFlightPathSig(
    const CaloTauRef& tau) const {
  double flightPathSignificance = 0;
  //Secondary vertex
  reco::TrackRefVector signalTracks = tau->signalTracks();
  vector<TransientTrack> transientTracks;
  for(size_t i = 0; i < signalTracks.size(); ++i){
    const TransientTrack transientTrack =
        transientTrackBuilder->build(signalTracks[i]);
    transientTracks.push_back(transientTrack);
  }
  if(transientTracks.size() > 1) {
    KalmanVertexFitter kvf(true);
    TransientVertex tv = kvf.vertex(transientTracks);
    if(tv.isValid()){
      GlobalVector tauDir(tau->px(), tau->py(), tau->pz());
      Vertex secVer = tv;
      flightPathSignificance = vertexSignificance(primaryVertex,secVer,tauDir);
    }
  }
  return flightPathSignificance;
}

double
CaloRecoTauDiscriminationByFlightPathSignificance::vertexSignificance(
    reco::Vertex const & pv, Vertex const & sv,GlobalVector const & direction) const {
  return SecondaryVertex::computeDist3d(
      pv,sv,direction,withPVError).significance();
}

}
DEFINE_FWK_MODULE(CaloRecoTauDiscriminationByFlightPathSignificance);

