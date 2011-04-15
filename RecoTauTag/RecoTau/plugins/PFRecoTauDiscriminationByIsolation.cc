#include <functional>
#include <boost/foreach.hpp>
#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "RecoTauTag/RecoTau/interface/RecoTauQualityCuts.h"
#include "RecoTauTag/RecoTau/interface/RecoTauVertexAssociator.h"
#include "RecoTauTag/RecoTau/interface/ConeTools.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "TMath.h"

/* class PFRecoTauDiscriminationByIsolation
 * created : Jul 23 2007,
 * revised : Thu Aug 13 14:44:40 PDT 2009
 * contributors : Ludovic Houchu (IPHC, Strasbourg),
 *               Christian Veelken (UC Davis),
*                Evan K. Friis (UC Davis)
 *               Michalis Bachtis (UW Madison)
 */

using namespace reco;
using namespace std;

class PFRecoTauDiscriminationByIsolation :
  public PFTauDiscriminationProducerBase  {
  public:
    explicit PFRecoTauDiscriminationByIsolation(const edm::ParameterSet& pset):
      PFTauDiscriminationProducerBase(pset),
      qualityCutsPSet_(pset.getParameter<edm::ParameterSet>("qualityCuts")) {

        includeTracks_ = pset.getParameter<bool>(
            "ApplyDiscriminationByTrackerIsolation");
        includeGammas_ = pset.getParameter<bool>(
            "ApplyDiscriminationByECALIsolation");

        applyOccupancyCut_ = pset.getParameter<bool>("applyOccupancyCut");
        maximumOccupancy_ = pset.getParameter<uint32_t>("maximumOccupancy");

        applySumPtCut_ = pset.getParameter<bool>("applySumPtCut");
        maximumSumPt_ = pset.getParameter<double>("maximumSumPtCut");

        applyRelativeSumPtCut_ = pset.getParameter<bool>(
            "applyRelativeSumPtCut");
        maximumRelativeSumPt_ = pset.getParameter<double>(
            "relativeSumPtCut");

        if (pset.exists("customOuterCone")) {
          customIsoCone_ = pset.getParameter<double>("customOuterCone");
        } else {
          customIsoCone_ = -1;
        }

        qcuts_.reset(new tau::RecoTauQualityCuts(
            qualityCutsPSet_.getParameter<edm::ParameterSet>(
            "isolationQualityCuts")));
        vertexAssociator_.reset(
            new tau::RecoTauVertexAssociator(qualityCutsPSet_));

        applyDeltaBeta_ = false;
        if (pset.exists("applyDeltaBetaCorrection")) {
          pileupQcuts_.reset(new tau::RecoTauQualityCuts(
              qualityCutsPSet_.getParameter<edm::ParameterSet>(
                "pileupQualityCuts")));
          applyDeltaBeta_ = pset.getParameter<bool>(
              "applyDeltaBetaCorrection");
          pfCandSrc_ = pset.getParameter<edm::InputTag>("particleFlowSrc");
          deltaBetaCollectionCone_ = pset.getParameter<double>(
              "isoConeSizeForDeltaBeta");
          deltaBetaFactor_ = pset.getParameter<double>(
              "deltaBetaFactor");
        }
      }

    ~PFRecoTauDiscriminationByIsolation(){}

    void beginEvent(const edm::Event& evt, const edm::EventSetup& evtSetup);
    double discriminate(const PFTauRef& pfTau);

  private:
    edm::ParameterSet qualityCutsPSet_;
    std::auto_ptr<tau::RecoTauQualityCuts> qcuts_;
    std::auto_ptr<tau::RecoTauQualityCuts> pileupQcuts_;
    std::auto_ptr<tau::RecoTauVertexAssociator> vertexAssociator_;

    bool includeTracks_;
    bool includeGammas_;
    bool applyOccupancyCut_;
    uint32_t maximumOccupancy_;
    bool applySumPtCut_;
    double maximumSumPt_;
    bool applyRelativeSumPtCut_;
    double maximumRelativeSumPt_;
    double customIsoCone_;


    // PU subtraction parameters
    bool applyDeltaBeta_;
    edm::InputTag pfCandSrc_;
    std::vector<reco::PFCandidateRef> chargedPFCandidatesInEvent_;
    double deltaBetaCollectionCone_;
    double deltaBetaFactor_;
  };

void PFRecoTauDiscriminationByIsolation::beginEvent(const edm::Event& event,
    const edm::EventSetup& eventSetup) {

  // NB: The use of the PV in this context is necessitated by its use in
  // applying quality cuts to the different objects in the isolation cone
  // The vertex associator contains the logic to select the appropriate vertex
  // We need to pass it the event so it can load the vertices.
  vertexAssociator_->setEvent(event);

  // If we are applying the delta beta correction, we need to get the PF
  // candidates from the event so we can find the PU tracks.
  chargedPFCandidatesInEvent_.clear();
  if (applyDeltaBeta_) {
    edm::Handle<reco::PFCandidateCollection> pfCandHandle_;
    event.getByLabel(pfCandSrc_, pfCandHandle_);
    chargedPFCandidatesInEvent_.reserve(pfCandHandle_->size());
    for (size_t i = 0; i < pfCandHandle_->size(); ++i) {
      reco::PFCandidateRef pfCand(pfCandHandle_, i);
      if (pfCand->charge() != 0)
        chargedPFCandidatesInEvent_.push_back(pfCand);
    }
  }
}

double
PFRecoTauDiscriminationByIsolation::discriminate(const PFTauRef& pfTau) {
  // collect the objects we are working with (ie tracks, tracks+gammas, etc)
  std::vector<PFCandidateRef> isoCharged;
  std::vector<PFCandidateRef> isoNeutral;
  std::vector<PFCandidateRef> isoPU;

  // Get the primary vertex associated to this tau
  reco::VertexRef pv = vertexAssociator_->associatedVertex(*pfTau);
  // Let the quality cuts know which the vertex to use when applying selections
  // on dz, etc.
  qcuts_->setPV(pv);
  if (pileupQcuts_.get()) {
    pileupQcuts_->setPV(pv);
  }

  BOOST_FOREACH(const reco::PFCandidateRef& cand,
      pfTau->isolationPFChargedHadrCands()) {
    if (qcuts_->filterRef(cand))
      isoCharged.push_back(cand);
  }
  BOOST_FOREACH(const reco::PFCandidateRef& cand,
      pfTau->isolationPFGammaCands()) {
    if (qcuts_->filterRef(cand))
      isoNeutral.push_back(cand);
  }
  typedef reco::tau::cone::DeltaRPtrFilter<PFCandidateRef> DRFilter;

  // If desired, get PU tracks.
  if (applyDeltaBeta_) {
    std::vector<PFCandidateRef> allPU =
      pileupQcuts_->filterRefs(chargedPFCandidatesInEvent_);
    // Only select PU tracks inside the isolation cone.
    DRFilter deltaBetaFilter(pfTau->p4(), 0, deltaBetaCollectionCone_);
    BOOST_FOREACH(const reco::PFCandidateRef& cand, allPU) {
      if (!deltaBetaFilter(cand)) {
        isoPU.push_back(cand);
      }
    }
  }

  // Check if we want a custom iso cone
  if (customIsoCone_ >= 0.) {
    DRFilter filter(pfTau->p4(), 0, customIsoCone_);
    // Remove all the objects not in our iso cone
    std::remove_if(isoCharged.begin(), isoCharged.end(), std::not1(filter));
    std::remove_if(isoNeutral.begin(), isoNeutral.end(), std::not1(filter));
    std::remove_if(isoPU.begin(), isoPU.end(), std::not1(filter));
  }

  bool failsOccupancyCut     = false;
  bool failsSumPtCut         = false;
  bool failsRelativeSumPtCut = false;

  //--- nObjects requirement
  int neutrals = isoNeutral.size()-TMath::Nint(deltaBetaFactor_*isoPU.size());
  if(neutrals<0) neutrals=0;

  failsOccupancyCut = ( isoCharged.size()+neutrals > maximumOccupancy_ );

  //--- Sum PT requirement
  if( applySumPtCut_ || applyRelativeSumPtCut_ ) {
    double totalPt=0.0;
    double chargedPt=0.0;
    double puPt=0.0;
    double neutralPt=0.0;
    BOOST_FOREACH(const PFCandidateRef& isoObject, isoCharged) {
      chargedPt += isoObject->pt();
    }
    BOOST_FOREACH(const PFCandidateRef& isoObject, isoNeutral) {
      neutralPt += isoObject->pt();
    }
    BOOST_FOREACH(const PFCandidateRef& isoObject, isoPU) {
      puPt += isoObject->pt();
    }
    totalPt = chargedPt+max(neutralPt-deltaBetaFactor_*puPt,0.0);

    failsSumPtCut = (totalPt > maximumSumPt_);

    //--- Relative Sum PT requirement
    failsRelativeSumPtCut = (
        (pfTau->pt() > 0 ? totalPt/pfTau->pt() : 0 ) > maximumRelativeSumPt_ );
  }

  bool fails = (applyOccupancyCut_ && failsOccupancyCut) ||
    (applySumPtCut_ && failsSumPtCut) ||
    (applyRelativeSumPtCut_ && failsRelativeSumPtCut) ;

  return (fails ? 0. : 1.);
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationByIsolation);
