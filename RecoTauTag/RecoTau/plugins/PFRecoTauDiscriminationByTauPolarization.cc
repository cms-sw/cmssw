#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>
#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"

/* class PFRecoTauDiscriminationByTauPolarization
 * created : May 26 2010,
 * contributors : Sami Lehti (sami.lehti@cern.ch ; HIP, Helsinki)
 */

using namespace reco;
using namespace std;
using namespace edm;

class PFRecoTauDiscriminationByTauPolarization :
  public PFTauDiscriminationProducerBase  {
  public:
    explicit PFRecoTauDiscriminationByTauPolarization(
        const ParameterSet& iConfig)
      :PFTauDiscriminationProducerBase(iConfig) {  // retrieve quality cuts
        rTauMin = iConfig.getParameter<double>("rtau");
        booleanOutput = iConfig.getParameter<bool>("BooleanOutput");
      }

    ~PFRecoTauDiscriminationByTauPolarization() override{}

    void beginEvent(const Event&, const EventSetup&) override;
    double discriminate(const PFTauRef&) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  private:
    bool booleanOutput;
    double rTauMin;
};

void PFRecoTauDiscriminationByTauPolarization::beginEvent(
    const Event& event, const EventSetup& eventSetup){}

double
PFRecoTauDiscriminationByTauPolarization::discriminate(const PFTauRef& tau) const{

  double rTau = 0;
  // rtau for PFTau has to be calculated for leading PF charged hadronic candidate
  // calculating it from leadingTrack can (and will) give rtau > 1!
  if(tau.isNonnull() && tau->p() > 0
      && tau->leadPFChargedHadrCand().isNonnull()) {
    rTau = tau->leadPFChargedHadrCand()->p()/tau->p();
  }

  if(booleanOutput) return ( rTau > rTauMin ? 1. : 0. );
  return rTau;
}

void
PFRecoTauDiscriminationByTauPolarization::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // pfRecoTauDiscriminationByTauPolarization
  edm::ParameterSetDescription desc;
  desc.add<double>("rtau", 0.8);
  desc.add<edm::InputTag>("PVProducer", edm::InputTag("offlinePrimaryVertices"));
  desc.add<bool>("BooleanOutput", true);
  desc.add<edm::InputTag>("PFTauProducer", edm::InputTag("pfRecoTauProducer"));

  {
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
  }

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

  descriptions.add("pfRecoTauDiscriminationByTauPolarization", desc);
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationByTauPolarization);
