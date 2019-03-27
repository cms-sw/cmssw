#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

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

    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
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

void
CaloRecoTauDiscriminationByFlightPathSignificance::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // caloRecoTauDiscriminationByFlightPathSignificance
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("CaloTauProducer", edm::InputTag("caloRecoTauProducer"));
  desc.add<double>("flightPathSig", 1.5);
  desc.add<edm::InputTag>("PVProducer", edm::InputTag("offlinePrimaryVertices"));
  desc.add<bool>("BooleanOutput", true);

  edm::ParameterSetDescription pset_signalQualityCuts;
  pset_signalQualityCuts.add<double>("maxDeltaZ", 0.4);
  pset_signalQualityCuts.add<double>("minTrackPt", 0.5);
  pset_signalQualityCuts.add<double>("minTrackVertexWeight", -1.0);
  pset_signalQualityCuts.add<double>("maxTrackChi2", 100.0);
  pset_signalQualityCuts.add<unsigned int>("minTrackPixelHits", 0);
  pset_signalQualityCuts.add<double>("minGammaEt", 1.0);
  pset_signalQualityCuts.add<unsigned int>("minTrackHits", 3);
  pset_signalQualityCuts.add<double>("minNeutralHadronEt", 30.0);
  pset_signalQualityCuts.add<double>("maxTransverseImpactParameter", 0.1);
  pset_signalQualityCuts.addOptional<bool>("useTracksInsteadOfPFHadrons");

  edm::ParameterSetDescription pset_vxAssocQualityCuts;
  pset_vxAssocQualityCuts.add<double>("minTrackPt", 0.5);
  pset_vxAssocQualityCuts.add<double>("minTrackVertexWeight", -1.0);
  pset_vxAssocQualityCuts.add<double>("maxTrackChi2", 100.0);
  pset_vxAssocQualityCuts.add<unsigned int>("minTrackPixelHits", 0);
  pset_vxAssocQualityCuts.add<double>("minGammaEt", 1.0);
  pset_vxAssocQualityCuts.add<unsigned int>("minTrackHits", 3);
  pset_vxAssocQualityCuts.add<double>("maxTransverseImpactParameter", 0.1);
  pset_vxAssocQualityCuts.addOptional<bool>("useTracksInsteadOfPFHadrons");

  edm::ParameterSetDescription pset_isolationQualityCuts;
  pset_isolationQualityCuts.add<double>("maxDeltaZ", 0.2);
  pset_isolationQualityCuts.add<double>("minTrackPt", 1.0);
  pset_isolationQualityCuts.add<double>("minTrackVertexWeight", -1.0);
  pset_isolationQualityCuts.add<double>("maxTrackChi2", 100.0);
  pset_isolationQualityCuts.add<unsigned int>("minTrackPixelHits", 0);
  pset_isolationQualityCuts.add<double>("minGammaEt", 1.5);
  pset_isolationQualityCuts.add<unsigned int>("minTrackHits", 8);
  pset_isolationQualityCuts.add<double>("maxTransverseImpactParameter", 0.03);
  pset_isolationQualityCuts.addOptional<bool>("useTracksInsteadOfPFHadrons");

  edm::ParameterSetDescription pset_qualityCuts;
  pset_qualityCuts.add<edm::ParameterSetDescription>("signalQualityCuts",    pset_signalQualityCuts);
  pset_qualityCuts.add<edm::ParameterSetDescription>("vxAssocQualityCuts",   pset_vxAssocQualityCuts);
  pset_qualityCuts.add<edm::ParameterSetDescription>("isolationQualityCuts", pset_isolationQualityCuts);
  pset_qualityCuts.add<std::string>("leadingTrkOrPFCandOption", "leadPFCand");
  pset_qualityCuts.add<std::string>("pvFindingAlgo", "closestInDeltaZ");
  pset_qualityCuts.add<edm::InputTag>("primaryVertexSrc", edm::InputTag("offlinePrimaryVertices"));
  pset_qualityCuts.add<bool>("vertexTrackFiltering", false);
  pset_qualityCuts.add<bool>("recoverLeadingTrk", false);

  desc.add<edm::ParameterSetDescription>("qualityCuts", pset_qualityCuts);

  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("BooleanOperator", "and");
    {
      edm::ParameterSetDescription psd1;
      psd1.add<double>("cut");
      psd1.add<edm::InputTag>("Producer");
      psd0.addOptional<edm::ParameterSetDescription>("leadTrack", psd1);
    }
    desc.add<edm::ParameterSetDescription>("Prediscriminants", psd0);
  }

  desc.add<bool>("UsePVerror", true);
  descriptions.add("caloRecoTauDiscriminationByFlightPathSignificance", desc);
}
DEFINE_FWK_MODULE(CaloRecoTauDiscriminationByFlightPathSignificance);

