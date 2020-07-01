#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>


        #include <memory>

#include "DataFormats/VertexReco/interface/Vertex.h"
        #include "RecoTauTag/RecoTau/interface/RecoTauQualityCuts.h"
#include "RecoTauTag/RecoTau/interface/RecoTauVertexAssociator.h"

/* class PFRecoTauDiscriminationByNProngs
 * created : August 30 2010,
 * contributors : Sami Lehti (sami.lehti@cern.ch ; HIP, Helsinki)
 * based on H+ tau ID by Lauri Wendland
 * Modified April 16 2014 by S.Lehti
 */

using namespace reco;
using namespace std;
using namespace edm;

class PFRecoTauDiscriminationByNProngs : public PFTauDiscriminationProducerBase {
public:
  explicit PFRecoTauDiscriminationByNProngs(const ParameterSet&);
  ~PFRecoTauDiscriminationByNProngs() override {}

  void beginEvent(const edm::Event&, const edm::EventSetup&) override;
  double discriminate(const reco::PFTauRef&) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  std::unique_ptr<tau::RecoTauQualityCuts> qcuts_;
  std::unique_ptr<tau::RecoTauVertexAssociator> vertexAssociator_;

  uint32_t minN, maxN;
  bool booleanOutput;
  edm::ParameterSet qualityCuts;
};

PFRecoTauDiscriminationByNProngs::PFRecoTauDiscriminationByNProngs(const ParameterSet& iConfig)
    : PFTauDiscriminationProducerBase(iConfig), qualityCuts(iConfig.getParameterSet("qualityCuts")) {
  minN = iConfig.getParameter<uint32_t>("MinN");
  maxN = iConfig.getParameter<uint32_t>("MaxN");
  booleanOutput = iConfig.getParameter<bool>("BooleanOutput");

  qcuts_ = std::make_unique<tau::RecoTauQualityCuts>(qualityCuts.getParameterSet("signalQualityCuts"));
  vertexAssociator_ = std::make_unique<tau::RecoTauVertexAssociator>(qualityCuts, consumesCollector());
}

void PFRecoTauDiscriminationByNProngs::beginEvent(const Event& iEvent, const EventSetup& iSetup) {
  vertexAssociator_->setEvent(iEvent);
}

double PFRecoTauDiscriminationByNProngs::discriminate(const PFTauRef& tau) const {
  reco::VertexRef pv = vertexAssociator_->associatedVertex(*tau);
  const CandidatePtr leadingTrack = tau->leadChargedHadrCand();

  uint np = 0;
  if (leadingTrack.isNonnull() && pv.isNonnull()) {
    qcuts_->setPV(pv);
    qcuts_->setLeadTrack(*tau->leadChargedHadrCand());

    for (auto const& cand : tau->signalChargedHadrCands()) {
      if (qcuts_->filterCandRef(cand))
        np++;
    }
  }

  bool accepted = false;
  if (maxN == 0) {
    if (np == 1 || np == 3)
      accepted = true;
  } else {
    if (np >= minN && np <= maxN)
      accepted = true;
  }

  if (!accepted)
    np = 0;
  if (booleanOutput)
    return accepted;
  return np;
}

void PFRecoTauDiscriminationByNProngs::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // pfRecoTauDiscriminationByNProngs
  edm::ParameterSetDescription desc;

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
    pset_qualityCuts.add<edm::ParameterSetDescription>("signalQualityCuts", pset_signalQualityCuts);
    pset_qualityCuts.add<edm::ParameterSetDescription>("vxAssocQualityCuts", pset_vxAssocQualityCuts);
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
    desc.add<edm::ParameterSetDescription>("Prediscriminants", psd0);
  }

  desc.add<bool>("BooleanOutput", true);
  desc.add<edm::InputTag>("PFTauProducer", edm::InputTag("combinatoricRecoTaus"));
  desc.add<unsigned int>("MinN", 1);
  desc.add<unsigned int>("MaxN", 0);
  descriptions.add("pfRecoTauDiscriminationByNProngs", desc);
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationByNProngs);
