#include <functional>
#include <memory>

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoTauTag/RecoTau/interface/ConeTools.h"
#include "RecoTauTag/RecoTau/interface/RecoTauQualityCuts.h"
#include "RecoTauTag/RecoTau/interface/RecoTauVertexAssociator.h"
#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

#include "TMath.h"
#include "TFormula.h"

/* class PFRecoTauDiscriminationByIsolationContainer
 * created : Jul 23 2007,
 * revised : Thu Aug 13 14:44:40 PDT 2009
 * contributors : Ludovic Houchu (IPHC, Strasbourg),
 *                Christian Veelken (UC Davis),
 *                Evan K. Friis (UC Davis)
 *                Michalis Bachtis (UW Madison)
 */

using namespace reco;
using namespace std;

class PFRecoTauDiscriminationByIsolationContainer : public PFTauDiscriminationContainerProducerBase {
public:
  enum StoredRawType { None, SumPt, PUsumPt, Occupancy, FootPrintCorrection, PhotonSumPt };
  explicit PFRecoTauDiscriminationByIsolationContainer(const edm::ParameterSet& pset)
      : PFTauDiscriminationContainerProducerBase(pset),
        moduleLabel_(pset.getParameter<std::string>("@module_label")),
        qualityCutsPSet_(pset.getParameter<edm::ParameterSet>("qualityCuts")) {
    // RIC: multiply neutral isolation by a flat factor.
    //      Useful, for instance, to combine charged and neutral isolations
    //      with different relative weights
    weightGammas_ = pset.getParameter<double>("WeightECALIsolation");

    // RIC: allow to relax the isolation completely beyond a given tau pt
    minPtForNoIso_ = pset.getParameter<double>("minTauPtForNoIso");

    // Get configs for raw values
    bool storeRawFootprintCorrection = false;
    deltaBetaNeeded_ = false;
    weightsNeeded_ = false;
    tracksNeeded_ = false;
    gammasNeeded_ = false;
    storeRawValue_.clear();
    auto const& rawDefs = pset.getParameter<std::vector<edm::ParameterSet>>("IDdefinitions");
    std::vector<std::string> idnames;
    for (auto const& rawDefsEntry : rawDefs) {
      idnames.push_back(rawDefsEntry.getParameter<std::string>("IDname"));
      // Can only store one type
      int numStoreOptions = 0;
      if (rawDefsEntry.getParameter<bool>("storeRawSumPt")) {
        storeRawValue_.push_back(SumPt);
        ++numStoreOptions;
      }
      if (rawDefsEntry.getParameter<bool>("storeRawOccupancy")) {
        storeRawValue_.push_back(Occupancy);
        ++numStoreOptions;
      }
      if (rawDefsEntry.getParameter<bool>("storeRawPUsumPt")) {
        storeRawValue_.push_back(PUsumPt);
        ++numStoreOptions;
      }
      if (rawDefsEntry.getParameter<bool>("storeRawFootprintCorrection")) {
        storeRawValue_.push_back(FootPrintCorrection);
        storeRawFootprintCorrection = true;
        ++numStoreOptions;
      }
      if (rawDefsEntry.getParameter<bool>("storeRawPhotonSumPt_outsideSignalCone")) {
        storeRawValue_.push_back(PhotonSumPt);
        ++numStoreOptions;
      }
      if (numStoreOptions != 1) {
        throw cms::Exception("BadIsoConfig")
            << "Multiple or none of 'store sum pt' and/or 'store occupancy' options are set."
            << " These options are mutually exclusive.";
      }

      includeGammas_.push_back(rawDefsEntry.getParameter<bool>("ApplyDiscriminationByECALIsolation"));
      if (includeGammas_.back())
        gammasNeeded_ = true;
      calculateWeights_.push_back(rawDefsEntry.getParameter<bool>("ApplyDiscriminationByWeightedECALIsolation"));
      if (calculateWeights_.back())
        weightsNeeded_ = true;
      includeTracks_.push_back(rawDefsEntry.getParameter<bool>("ApplyDiscriminationByTrackerIsolation"));
      if (includeTracks_.back())
        tracksNeeded_ = true;
      applyDeltaBetaCorrection_.push_back(rawDefsEntry.getParameter<bool>("applyDeltaBetaCorrection"));
      if (applyDeltaBetaCorrection_.back())
        deltaBetaNeeded_ = true;
      useAllPFCandsForWeights_.push_back(rawDefsEntry.getParameter<bool>("UseAllPFCandsForWeights"));

      // sanity check2 - can't use weighted and unweighted iso at the same time
      if (includeGammas_.back() && calculateWeights_.back()) {
        throw cms::Exception("BadIsoConfig")
            << "Both 'ApplyDiscriminationByECALIsolation' and 'ApplyDiscriminationByWeightedECALIsolation' "
            << "have been set to true. These options are mutually exclusive.";
      }
    }

    // Get configs for WPs - negative cut values are used to switch of the condition
    std::vector<edm::ParameterSet> wpDefs = pset.getParameter<std::vector<edm::ParameterSet>>("IDWPdefinitions");
    for (std::vector<edm::ParameterSet>::iterator wpDefsEntry = wpDefs.begin(); wpDefsEntry != wpDefs.end();
         ++wpDefsEntry) {
      maxAbsValue_.push_back(wpDefsEntry->getParameter<std::vector<double>>("maximumAbsoluteValues"));
      maxRelValue_.push_back(wpDefsEntry->getParameter<std::vector<double>>("maximumRelativeValues"));
      offsetRelValue_.push_back(wpDefsEntry->getParameter<std::vector<double>>("relativeValueOffsets"));
      auto refRawIDNames = wpDefsEntry->getParameter<std::vector<std::string>>("referenceRawIDNames");
      if (!maxAbsValue_.back().empty() && maxAbsValue_.back().size() != refRawIDNames.size())
        throw cms::Exception("BadIsoConfig")
            << "WP configuration: Length of 'maximumAbsoluteValues' does not match length of 'referenceRawIDNames'!";
      if (!maxRelValue_.back().empty() && maxRelValue_.back().size() != refRawIDNames.size())
        throw cms::Exception("BadIsoConfig")
            << "WP configuration: Length of 'maximumRelativeValues' does not match length of 'referenceRawIDNames'!";
      if (!offsetRelValue_.back().empty() && offsetRelValue_.back().size() != refRawIDNames.size())
        throw cms::Exception("BadIsoConfig")
            << "WP configuration: Length of 'relativeValueOffsets' does not match length of 'referenceRawIDNames'!";
      else if (offsetRelValue_.back().empty())
        offsetRelValue_.back().assign(refRawIDNames.size(), 0.0);
      rawValue_reference_.push_back(std::vector<int>(refRawIDNames.size()));
      for (size_t i = 0; i < refRawIDNames.size(); i++) {
        bool found = false;
        for (size_t j = 0; j < idnames.size(); j++) {
          if (refRawIDNames[i] == idnames[j]) {
            rawValue_reference_.back()[i] = j;
            found = true;
            break;
          }
        }
        if (!found)
          throw cms::Exception("BadIsoConfig")
              << "WP configuration: Requested raw ID '" << refRawIDNames[i] << "' not defined!";
      }
    }

    customIsoCone_ = pset.getParameter<double>("customOuterCone");

    applyFootprintCorrection_ = pset.getParameter<bool>("applyFootprintCorrection");
    if (applyFootprintCorrection_ || storeRawFootprintCorrection) {
      edm::VParameterSet cfgFootprintCorrections = pset.getParameter<edm::VParameterSet>("footprintCorrections");
      for (edm::VParameterSet::const_iterator cfgFootprintCorrection = cfgFootprintCorrections.begin();
           cfgFootprintCorrection != cfgFootprintCorrections.end();
           ++cfgFootprintCorrection) {
        std::string selection = cfgFootprintCorrection->getParameter<std::string>("selection");
        std::string offset = cfgFootprintCorrection->getParameter<std::string>("offset");
        auto footprintCorrection = std::make_unique<FootprintCorrection>(selection, offset);
        footprintCorrections_.push_back(std::move(footprintCorrection));
      }
    }

    // Get the quality cuts specific to the isolation region
    edm::ParameterSet isolationQCuts = qualityCutsPSet_.getParameterSet("isolationQualityCuts");

    qcuts_ = std::make_unique<tau::RecoTauQualityCuts>(isolationQCuts);

    vertexAssociator_ = std::make_unique<tau::RecoTauVertexAssociator>(qualityCutsPSet_, consumesCollector());

    if (deltaBetaNeeded_ || weightsNeeded_) {
      // Factorize the isolation QCuts into those that are used to
      // select PU and those that are not.
      std::pair<edm::ParameterSet, edm::ParameterSet> puFactorizedIsoQCuts =
          reco::tau::factorizePUQCuts(isolationQCuts);

      // Determine the pt threshold for the PU tracks
      // First check if the user specifies explicitly the cut.
      // For that the user has to provide a >= 0  value for the PtCutOverride.
      bool deltaBetaPUTrackPtCutOverride = pset.getParameter<bool>("deltaBetaPUTrackPtCutOverride");
      if (deltaBetaPUTrackPtCutOverride) {
        double deltaBetaPUTrackPtCutOverride_val = pset.getParameter<double>("deltaBetaPUTrackPtCutOverride_val");
        puFactorizedIsoQCuts.second.addParameter<double>("minTrackPt", deltaBetaPUTrackPtCutOverride_val);
      } else {
        // Secondly take it from the minGammaEt
        puFactorizedIsoQCuts.second.addParameter<double>("minTrackPt",
                                                         isolationQCuts.getParameter<double>("minGammaEt"));
      }

      pileupQcutsPUTrackSelection_ = std::make_unique<tau::RecoTauQualityCuts>(puFactorizedIsoQCuts.first);

      pileupQcutsGeneralQCuts_ = std::make_unique<tau::RecoTauQualityCuts>(puFactorizedIsoQCuts.second);

      pfCandSrc_ = pset.getParameter<edm::InputTag>("particleFlowSrc");
      pfCand_token = consumes<edm::View<reco::Candidate>>(pfCandSrc_);
      vertexSrc_ = pset.getParameter<edm::InputTag>("vertexSrc");
      vertex_token = consumes<reco::VertexCollection>(vertexSrc_);
      deltaBetaCollectionCone_ = pset.getParameter<double>("isoConeSizeForDeltaBeta");
      std::string deltaBetaFactorFormula = pset.getParameter<string>("deltaBetaFactor");
      deltaBetaFormula_ = std::make_unique<TFormula>("DB_corr", deltaBetaFactorFormula.c_str());
    }

    applyRhoCorrection_ = pset.getParameter<bool>("applyRhoCorrection");
    if (applyRhoCorrection_) {
      rhoProducer_ = pset.getParameter<edm::InputTag>("rhoProducer");
      rho_token = consumes<double>(rhoProducer_);
      rhoConeSize_ = pset.getParameter<double>("rhoConeSize");
      rhoUEOffsetCorrection_ = pset.getParameter<double>("rhoUEOffsetCorrection");
    }

    verbosity_ = pset.getParameter<int>("verbosity");
  }

  ~PFRecoTauDiscriminationByIsolationContainer() override {}

  void beginEvent(const edm::Event& evt, const edm::EventSetup& evtSetup) override;
  reco::SingleTauDiscriminatorContainer discriminate(const PFTauRef& pfTau) const override;

  inline double weightedSum(const std::vector<CandidatePtr>& inColl_, double eta, double phi) const {
    double out = 1.0;
    for (auto const& inObj_ : inColl_) {
      double sum = (inObj_->pt() * inObj_->pt()) / (deltaR2(eta, phi, inObj_->eta(), inObj_->phi()));
      if (sum > 1.0)
        out *= sum;
    }
    return out;
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  std::string moduleLabel_;

  edm::ParameterSet qualityCutsPSet_;
  std::unique_ptr<tau::RecoTauQualityCuts> qcuts_;

  // Inverted QCut which selects tracks with bad DZ/trackWeight
  std::unique_ptr<tau::RecoTauQualityCuts> pileupQcutsPUTrackSelection_;
  std::unique_ptr<tau::RecoTauQualityCuts> pileupQcutsGeneralQCuts_;

  std::unique_ptr<tau::RecoTauVertexAssociator> vertexAssociator_;

  bool weightsNeeded_;
  bool tracksNeeded_;
  bool gammasNeeded_;
  double weightGammas_;
  double customIsoCone_;
  // RIC:
  double minPtForNoIso_;

  bool applyFootprintCorrection_;
  struct FootprintCorrection {
    FootprintCorrection(const std::string& selection, const std::string& offset)
        : selection_(selection), offset_(offset) {}
    ~FootprintCorrection() {}
    StringCutObjectSelector<PFTau> selection_;
    StringObjectFunction<PFTau> offset_;
  };
  std::vector<std::unique_ptr<FootprintCorrection>> footprintCorrections_;

  // Options to store the raw value in the discriminator instead of boolean pass/fail flag
  std::vector<StoredRawType> storeRawValue_;
  // Options to store the boolean pass/fail flag
  std::vector<std::vector<int>> rawValue_reference_;
  std::vector<std::vector<double>> maxAbsValue_;
  std::vector<std::vector<double>> maxRelValue_;
  std::vector<std::vector<double>> offsetRelValue_;
  // Options used for both raw and WP definitions
  std::vector<bool> includeGammas_;
  std::vector<bool> calculateWeights_;
  std::vector<bool> includeTracks_;
  std::vector<bool> applyDeltaBetaCorrection_;
  std::vector<bool> useAllPFCandsForWeights_;

  /* **********************************************************************
     **** Pileup Subtraction Parameters ***********************************
     **********************************************************************/

  // Delta Beta correction
  bool deltaBetaNeeded_;
  edm::InputTag pfCandSrc_;
  edm::EDGetTokenT<edm::View<reco::Candidate>> pfCand_token;
  // Keep track of how many vertices are in the event
  edm::InputTag vertexSrc_;
  edm::EDGetTokenT<reco::VertexCollection> vertex_token;
  std::vector<reco::CandidatePtr> chargedPFCandidatesInEvent_;
  // Size of cone used to collect PU tracks
  double deltaBetaCollectionCone_;
  std::unique_ptr<TFormula> deltaBetaFormula_;
  double deltaBetaFactorThisEvent_;

  // Rho correction
  bool applyRhoCorrection_;
  edm::InputTag rhoProducer_;
  edm::EDGetTokenT<double> rho_token;
  double rhoConeSize_;
  double rhoUEOffsetCorrection_;
  double rhoCorrectionThisEvent_;
  double rhoThisEvent_;

  // Flag to enable/disable debug output
  int verbosity_;
};

void PFRecoTauDiscriminationByIsolationContainer::beginEvent(const edm::Event& event,
                                                             const edm::EventSetup& eventSetup) {
  // NB: The use of the PV in this context is necessitated by its use in
  // applying quality cuts to the different objects in the isolation cone
  // The vertex associator contains the logic to select the appropriate vertex
  // We need to pass it the event so it can load the vertices.
  vertexAssociator_->setEvent(event);

  // If we are applying the delta beta correction, we need to get the PF
  // candidates from the event so we can find the PU tracks.
  if (deltaBetaNeeded_ || weightsNeeded_) {
    // Collect all the PF pile up tracks
    edm::Handle<edm::View<reco::Candidate>> pfCandidates;
    event.getByToken(pfCand_token, pfCandidates);
    chargedPFCandidatesInEvent_.clear();
    chargedPFCandidatesInEvent_.reserve(pfCandidates->size());
    size_t numPFCandidates = pfCandidates->size();
    for (size_t i = 0; i < numPFCandidates; ++i) {
      reco::CandidatePtr pfCandidate(pfCandidates, i);
      if (pfCandidate->charge() != 0) {
        chargedPFCandidatesInEvent_.push_back(pfCandidate);
      }
    }
    // Count all the vertices in the event, to parameterize the DB
    // correction factor
    edm::Handle<reco::VertexCollection> vertices;
    event.getByToken(vertex_token, vertices);
    size_t nVtxThisEvent = vertices->size();
    deltaBetaFactorThisEvent_ = deltaBetaFormula_->Eval(nVtxThisEvent);
  }

  if (applyRhoCorrection_) {
    edm::Handle<double> rhoHandle_;
    event.getByToken(rho_token, rhoHandle_);
    rhoThisEvent_ = (*rhoHandle_ - rhoUEOffsetCorrection_) * (3.14159) * rhoConeSize_ * rhoConeSize_;
  }
}

reco::SingleTauDiscriminatorContainer PFRecoTauDiscriminationByIsolationContainer::discriminate(
    const PFTauRef& pfTau) const {
  LogDebug("discriminate") << " tau: Pt = " << pfTau->pt() << ", eta = " << pfTau->eta() << ", phi = " << pfTau->phi();
  LogDebug("discriminate") << *pfTau;

  // collect the objects we are working with (ie tracks, tracks+gammas, etc)
  std::vector<CandidatePtr> isoCharged_;
  std::vector<CandidatePtr> isoNeutral_;
  std::vector<CandidatePtr> isoPU_;
  std::vector<CandidatePtr> isoPUall_;
  CandidateCollection isoNeutralWeight_;
  CandidateCollection isoNeutralWeight_UseAllPFCands_;
  std::vector<CandidatePtr> chPV_;
  std::vector<CandidatePtr> chPVall_;
  isoCharged_.reserve(pfTau->isolationChargedHadrCands().size());
  isoNeutral_.reserve(pfTau->isolationGammaCands().size());
  isoPU_.reserve(std::min(100UL, chargedPFCandidatesInEvent_.size()));
  isoPUall_.reserve(std::min(100UL, chargedPFCandidatesInEvent_.size()));
  isoNeutralWeight_.reserve(pfTau->isolationGammaCands().size());
  isoNeutralWeight_UseAllPFCands_.reserve(pfTau->isolationGammaCands().size());

  chPV_.reserve(std::min(50UL, chargedPFCandidatesInEvent_.size()));
  chPVall_.reserve(std::min(50UL, chargedPFCandidatesInEvent_.size()));

  // Get the primary vertex associated to this tau
  reco::VertexRef pv = vertexAssociator_->associatedVertex(*pfTau);
  // Let the quality cuts know which the vertex to use when applying selections
  // on dz, etc.
  if (verbosity_) {
    if (pv.isNonnull()) {
      LogTrace("discriminate") << "pv: x = " << pv->position().x() << ", y = " << pv->position().y()
                               << ", z = " << pv->position().z();
    } else {
      LogTrace("discriminate") << "pv: N/A";
    }
    if (pfTau->leadChargedHadrCand().isNonnull()) {
      LogTrace("discriminate") << "leadPFChargedHadron:"
                               << " Pt = " << pfTau->leadChargedHadrCand()->pt() << ","
                               << " eta = " << pfTau->leadChargedHadrCand()->eta() << ","
                               << " phi = " << pfTau->leadChargedHadrCand()->phi();
    } else {
      LogTrace("discriminate") << "leadPFChargedHadron: N/A";
    }
  }

  // CV: isolation is not well defined in case primary vertex or leading charged hadron do not exist
  if (!(pv.isNonnull() && pfTau->leadChargedHadrCand().isNonnull()))
    return 0.;

  qcuts_->setPV(pv);
  qcuts_->setLeadTrack(*pfTau->leadChargedHadrCand());

  if (deltaBetaNeeded_ || weightsNeeded_) {
    pileupQcutsGeneralQCuts_->setPV(pv);
    pileupQcutsGeneralQCuts_->setLeadTrack(*pfTau->leadChargedHadrCand());
    pileupQcutsPUTrackSelection_->setPV(pv);
    pileupQcutsPUTrackSelection_->setLeadTrack(*pfTau->leadChargedHadrCand());
  }

  // Load the tracks if they are being used.
  if (tracksNeeded_) {
    for (auto const& cand : pfTau->isolationChargedHadrCands()) {
      if (qcuts_->filterCandRef(cand)) {
        LogTrace("discriminate") << "adding charged iso cand with pt " << cand->pt();
        isoCharged_.push_back(cand);
      }
    }
  }
  if (gammasNeeded_ || weightsNeeded_) {
    for (auto const& cand : pfTau->isolationGammaCands()) {
      if (qcuts_->filterCandRef(cand)) {
        LogTrace("discriminate") << "adding neutral iso cand with pt " << cand->pt();
        isoNeutral_.push_back(cand);
      }
    }
  }

  typedef reco::tau::cone::DeltaRPtrFilter<CandidatePtr> DRFilter;
  typedef reco::tau::cone::DeltaRFilter<Candidate> DRFilter2;

  // If desired, get PU tracks.
  if (deltaBetaNeeded_ || weightsNeeded_) {
    // First select by inverted the DZ/track weight cuts. True = invert
    if (verbosity_) {
      std::cout << "Initial PFCands: " << chargedPFCandidatesInEvent_.size() << std::endl;
    }

    std::vector<CandidatePtr> allPU = pileupQcutsPUTrackSelection_->filterCandRefs(chargedPFCandidatesInEvent_, true);

    std::vector<CandidatePtr> allNPU = pileupQcutsPUTrackSelection_->filterCandRefs(chargedPFCandidatesInEvent_);
    LogTrace("discriminate") << "After track cuts: " << allPU.size();

    // Now apply the rest of the cuts, like pt, and TIP, tracker hits, etc
    std::vector<CandidatePtr> cleanPU = pileupQcutsGeneralQCuts_->filterCandRefs(allPU);

    std::vector<CandidatePtr> cleanNPU = pileupQcutsGeneralQCuts_->filterCandRefs(allNPU);

    LogTrace("discriminate") << "After cleaning cuts: " << cleanPU.size();

    // Only select PU tracks inside the isolation cone.
    DRFilter deltaBetaFilter(pfTau->p4(), 0, deltaBetaCollectionCone_);
    for (auto const& cand : cleanPU) {
      if (deltaBetaFilter(cand))
        isoPU_.push_back(cand);
    }

    for (auto const& cand : cleanNPU) {
      if (deltaBetaFilter(cand))
        chPV_.push_back(cand);
    }
    LogTrace("discriminate") << "After cone cuts: " << isoPU_.size() << " " << chPV_.size();
    isoPUall_ = std::move(allPU);
    chPVall_ = std::move(allNPU);
  }

  if (weightsNeeded_) {
    for (auto const& isoObject : isoNeutral_) {
      if (isoObject->charge() != 0) {
        // weight only neutral objects
        isoNeutralWeight_.push_back(*isoObject);
        isoNeutralWeight_UseAllPFCands_.push_back(*isoObject);
        continue;
      }

      double eta = isoObject->eta();
      double phi = isoObject->phi();
      {
        double sumNPU = 0.5 * log(weightedSum(chPV_, eta, phi));

        double sumPU = 0.5 * log(weightedSum(isoPU_, eta, phi));
        LeafCandidate neutral(*isoObject);
        if ((sumNPU + sumPU) > 0)
          neutral.setP4(((sumNPU) / (sumNPU + sumPU)) * neutral.p4());

        isoNeutralWeight_.push_back(neutral);
      }
      {
        double sumNPU = 0.5 * log(weightedSum(chPVall_, eta, phi));

        double sumPU = 0.5 * log(weightedSum(isoPUall_, eta, phi));
        LeafCandidate neutral(*isoObject);
        if ((sumNPU + sumPU) > 0)
          neutral.setP4(((sumNPU) / (sumNPU + sumPU)) * neutral.p4());

        isoNeutralWeight_UseAllPFCands_.push_back(neutral);
      }
    }
  }

  // Check if we want a custom iso cone
  if (customIsoCone_ >= 0.) {
    DRFilter filter(pfTau->p4(), 0, customIsoCone_);
    DRFilter2 filter2(pfTau->p4(), 0, customIsoCone_);
    std::vector<CandidatePtr> isoCharged_filter;
    std::vector<CandidatePtr> isoNeutral_filter;
    // Remove all the objects not in our iso cone
    for (auto const& isoObject : isoCharged_) {
      if (filter(isoObject))
        isoCharged_filter.push_back(isoObject);
    }
    isoCharged_ = isoCharged_filter;
    for (auto const& isoObject : isoNeutral_) {
      if (filter(isoObject))
        isoNeutral_filter.push_back(isoObject);
    }
    isoNeutral_ = isoNeutral_filter;
    {
      CandidateCollection isoNeutralWeight_filter;
      for (auto const& isoObject : isoNeutralWeight_) {
        if (filter2(isoObject))
          isoNeutralWeight_filter.push_back(isoObject);
      }
      isoNeutralWeight_ = isoNeutralWeight_filter;
    }
    {
      CandidateCollection isoNeutralWeight_filter;
      for (auto const& isoObject : isoNeutralWeight_UseAllPFCands_) {
        if (filter2(isoObject))
          isoNeutralWeight_filter.push_back(isoObject);
      }
      isoNeutralWeight_UseAllPFCands_ = isoNeutralWeight_filter;
    }
  }

  //Now all needed incredients are ready. Loop over all ID configurations and produce output
  reco::SingleTauDiscriminatorContainer result;
  for (size_t i = 0; i < includeGammas_.size(); i++) {
    //--- nObjects requirement
    int neutrals = isoNeutral_.size();

    if (applyDeltaBetaCorrection_.at(i)) {
      neutrals -= TMath::Nint(deltaBetaFactorThisEvent_ * isoPU_.size());
    }
    if (neutrals < 0) {
      neutrals = 0;
    }

    int nOccupants = isoCharged_.size() + neutrals;

    double footprintCorrection_value = 0.;
    if (applyFootprintCorrection_ || storeRawValue_.at(i) == FootPrintCorrection) {
      for (std::vector<std::unique_ptr<FootprintCorrection>>::const_iterator footprintCorrection =
               footprintCorrections_.begin();
           footprintCorrection != footprintCorrections_.end();
           ++footprintCorrection) {
        if ((*footprintCorrection)->selection_(*pfTau)) {
          footprintCorrection_value = (*footprintCorrection)->offset_(*pfTau);
        }
      }
    }

    double totalPt = 0.;
    double puPt = 0.;
    //--- Sum PT requirement
    if (storeRawValue_.at(i) == SumPt || storeRawValue_.at(i) == PUsumPt) {
      double chargedPt = 0.;
      double neutralPt = 0.;
      double weightedNeutralPt = 0.;
      if (includeTracks_.at(i)) {
        for (auto const& isoObject : isoCharged_) {
          chargedPt += isoObject->pt();
        }
      }

      if (calculateWeights_.at(i)) {
        if (useAllPFCandsForWeights_.at(i)) {
          for (auto const& isoObject : isoNeutralWeight_UseAllPFCands_) {
            weightedNeutralPt += isoObject.pt();
          }
        } else {
          for (auto const& isoObject : isoNeutralWeight_) {
            weightedNeutralPt += isoObject.pt();
          }
        }
      } else if (includeGammas_.at(i)) {
        for (auto const& isoObject : isoNeutral_) {
          neutralPt += isoObject->pt();
        }
      }
      for (auto const& isoObject : isoPU_) {
        puPt += isoObject->pt();
      }
      LogTrace("discriminate") << "chargedPt = " << chargedPt;
      LogTrace("discriminate") << "neutralPt = " << neutralPt;
      LogTrace("discriminate") << "weighted neutral Pt = " << weightedNeutralPt;
      LogTrace("discriminate") << "puPt = " << puPt << " (delta-beta corr. = " << (deltaBetaFactorThisEvent_ * puPt)
                               << ")";

      if (calculateWeights_.at(i)) {
        neutralPt = weightedNeutralPt;
      }

      if (applyDeltaBetaCorrection_.at(i)) {
        neutralPt -= (deltaBetaFactorThisEvent_ * puPt);
      }

      if (applyFootprintCorrection_) {
        neutralPt -= footprintCorrection_value;
      }

      if (applyRhoCorrection_) {
        neutralPt -= rhoThisEvent_;
      }

      if (neutralPt < 0.) {
        neutralPt = 0.;
      }

      totalPt = chargedPt + weightGammas_ * neutralPt;
    }

    double photonSumPt_outsideSignalCone = 0.;
    if (storeRawValue_.at(i) == PhotonSumPt) {
      const std::vector<reco::CandidatePtr>& signalGammas = pfTau->signalGammaCands();
      for (std::vector<reco::CandidatePtr>::const_iterator signalGamma = signalGammas.begin();
           signalGamma != signalGammas.end();
           ++signalGamma) {
        double dR = deltaR(pfTau->eta(), pfTau->phi(), (*signalGamma)->eta(), (*signalGamma)->phi());
        if (dR > pfTau->signalConeSize())
          photonSumPt_outsideSignalCone += (*signalGamma)->pt();
      }
    }

    // We did error checking in the constructor, so this is safe.
    if (storeRawValue_.at(i) == SumPt) {
      result.rawValues.push_back(totalPt);
    } else if (storeRawValue_.at(i) == PUsumPt) {
      if (applyDeltaBetaCorrection_.at(i))
        result.rawValues.push_back(puPt);
      else if (applyRhoCorrection_)
        result.rawValues.push_back(rhoThisEvent_);
      else
        result.rawValues.push_back(0.);
    } else if (storeRawValue_.at(i) == Occupancy) {
      result.rawValues.push_back(nOccupants);
    } else if (storeRawValue_.at(i) == FootPrintCorrection) {
      result.rawValues.push_back(footprintCorrection_value);
    } else if (storeRawValue_.at(i) == PhotonSumPt) {
      result.rawValues.push_back(photonSumPt_outsideSignalCone);
    }
  }
  for (size_t i = 0; i < rawValue_reference_.size(); i++) {
    bool pass = true;
    if (minPtForNoIso_ > 0. && pfTau->pt() > minPtForNoIso_)
      LogDebug("discriminate") << "tau pt = " << pfTau->pt() << "\t  min cutoff pt = " << minPtForNoIso_;
    else {
      for (size_t j = 0; j < rawValue_reference_[i].size(); j++) {
        double rawValue = result.rawValues[rawValue_reference_[i][j]];
        LogTrace("discriminate") << "Iso sum = " << rawValue << " (max_abs = " << maxAbsValue_[i][j]
                                 << ", max_rel = " << maxRelValue_[i][j] << ", offset_rel = " << offsetRelValue_[i][j]
                                 << ")";
        if (!maxAbsValue_[i].empty() && maxAbsValue_[i][j] >= 0.0)
          pass = rawValue <= maxAbsValue_[i][j];
        if (!maxRelValue_[i].empty() && maxRelValue_[i][j] >= 0.0)
          pass = rawValue <= maxRelValue_[i][j] * (pfTau->pt() - offsetRelValue_[i][j]);
        if (!pass)
          break;  // do not pass if one of the conditions in the j list fails
      }
    }
    result.workingPoints.push_back(pass);
  }
  return result;
}

void PFRecoTauDiscriminationByIsolationContainer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // pfRecoTauDiscriminationByIsolationContainer
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("PFTauProducer", edm::InputTag("pfRecoTauProducer"));

  edm::ParameterSetDescription desc_qualityCuts;
  reco::tau::RecoTauQualityCuts::fillDescriptions(desc_qualityCuts);
  desc.add<edm::ParameterSetDescription>("qualityCuts", desc_qualityCuts);

  desc.add<double>("minTauPtForNoIso", -99.0);
  desc.add<edm::InputTag>("vertexSrc", edm::InputTag("offlinePrimaryVertices"));
  desc.add<double>("rhoConeSize", 0.5);
  desc.add<edm::InputTag>("rhoProducer", edm::InputTag("fixedGridRhoFastjetAll"));

  {
    edm::ParameterSetDescription vpsd1;
    vpsd1.add<std::string>("selection");
    vpsd1.add<std::string>("offset");
    desc.addVPSet("footprintCorrections", vpsd1, {});
  }

  desc.add<std::string>("deltaBetaFactor", "0.38");
  desc.add<bool>("applyFootprintCorrection", false);
  {
    edm::ParameterSetDescription pset_Prediscriminants;
    pset_Prediscriminants.add<std::string>("BooleanOperator", "and");
    {
      edm::ParameterSetDescription psd1;
      psd1.add<double>("cut", 0.5);
      psd1.add<edm::InputTag>("Producer", edm::InputTag("pfRecoTauDiscriminationByLeadingTrackFinding"));
      pset_Prediscriminants.addOptional<edm::ParameterSetDescription>("leadTrack", psd1);
    }
    {
      edm::ParameterSetDescription psd1;
      psd1.add<double>("cut", 0.5);
      psd1.add<edm::InputTag>("Producer", edm::InputTag("hpsPFTauDiscriminationByDecayModeFindingNewDMs"));
      pset_Prediscriminants.addOptional<edm::ParameterSetDescription>("decayMode", psd1);
    }
    {
      edm::ParameterSetDescription psd1;
      psd1.add<double>("cut", 0.5);
      psd1.add<edm::InputTag>("Producer", edm::InputTag("hpsPFTauDiscriminationByLooseChargedIsolation"));
      pset_Prediscriminants.addOptional<edm::ParameterSetDescription>("preIso", psd1);
    }
    desc.add<edm::ParameterSetDescription>("Prediscriminants", pset_Prediscriminants);
  }

  desc.add<int>("verbosity", 0);

  desc.add<bool>("deltaBetaPUTrackPtCutOverride", false);
  desc.add<bool>("applyRhoCorrection", false);

  desc.add<double>("WeightECALIsolation", 1.0);
  desc.add<double>("rhoUEOffsetCorrection", 1.0);
  desc.add<double>("deltaBetaPUTrackPtCutOverride_val", -1.5);
  desc.add<double>("isoConeSizeForDeltaBeta", 0.5);
  desc.add<double>("customOuterCone", -1.0);
  desc.add<edm::InputTag>("particleFlowSrc", edm::InputTag("particleFlow"));

  // options for various stored ID raw values
  edm::ParameterSetDescription desc_idlist;
  desc_idlist.add<string>("IDname");  //not needed by producer but required for mapping at PAT level
  desc_idlist.add<bool>("storeRawSumPt", false);
  desc_idlist.add<bool>("storeRawPUsumPt", false);
  desc_idlist.add<bool>("storeRawOccupancy", false);
  desc_idlist.add<bool>("storeRawFootprintCorrection", false);
  desc_idlist.add<bool>("storeRawPhotonSumPt_outsideSignalCone", false);
  desc_idlist.add<bool>("ApplyDiscriminationByECALIsolation", false);
  desc_idlist.add<bool>("ApplyDiscriminationByWeightedECALIsolation", false);
  desc_idlist.add<bool>("ApplyDiscriminationByTrackerIsolation", false);
  desc_idlist.add<bool>("applyDeltaBetaCorrection", false);
  desc_idlist.add<bool>("UseAllPFCandsForWeights", false);
  desc.addVPSet("IDdefinitions", desc_idlist, {});
  // options for various stored ID WPs
  edm::ParameterSetDescription desc_idwplist;
  desc_idwplist.add<string>("IDname");  //not needed by producer but required for mapping at PAT level
  desc_idwplist.add<std::vector<string>>("referenceRawIDNames")
      ->setComment(
          "List of raw IDs defined in 'IDdefinitions' to pass all respective conditions defined in "
          "'maximumAbsoluteValues', 'maximumRelativeValues' , and 'relativeValueOffsets'");
  desc_idwplist.add<std::vector<double>>("maximumAbsoluteValues", {});
  desc_idwplist.add<std::vector<double>>("maximumRelativeValues", {});
  desc_idwplist.add<std::vector<double>>("relativeValueOffsets", {});
  desc.addVPSet("IDWPdefinitions", desc_idwplist, {});

  descriptions.add("pfRecoTauDiscriminationByIsolationContainer", desc);
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationByIsolationContainer);
