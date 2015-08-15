#include <functional>
#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "RecoTauTag/RecoTau/interface/RecoTauQualityCuts.h"
#include "RecoTauTag/RecoTau/interface/RecoTauVertexAssociator.h"
#include "RecoTauTag/RecoTau/interface/ConeTools.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "TMath.h"
#include "TFormula.h"

/* class PFRecoTauDiscriminationByIsolation
 * created : Jul 23 2007,
 * revised : Thu Aug 13 14:44:40 PDT 2009
 * contributors : Ludovic Houchu (IPHC, Strasbourg),
 *                Christian Veelken (UC Davis),
 *                Evan K. Friis (UC Davis)
 *                Michalis Bachtis (UW Madison)
 */

using namespace reco;
using namespace std;

class PFRecoTauDiscriminationByIsolation : public PFTauDiscriminationProducerBase  
{
 public:
  explicit PFRecoTauDiscriminationByIsolation(const edm::ParameterSet& pset)
    : PFTauDiscriminationProducerBase(pset),
      moduleLabel_(pset.getParameter<std::string>("@module_label")),
      qualityCutsPSet_(pset.getParameter<edm::ParameterSet>("qualityCuts")) 
  {
    includeTracks_ = pset.getParameter<bool>(
      "ApplyDiscriminationByTrackerIsolation");
    includeGammas_ = pset.getParameter<bool>(
      "ApplyDiscriminationByECALIsolation");

    calculateWeights_ = pset.exists("ApplyDiscriminationByWeightedECALIsolation") ?
      pset.getParameter<bool>("ApplyDiscriminationByWeightedECALIsolation") : false;

    applyOccupancyCut_ = pset.getParameter<bool>("applyOccupancyCut");
    maximumOccupancy_ = pset.getParameter<uint32_t>("maximumOccupancy");

    applySumPtCut_ = pset.getParameter<bool>("applySumPtCut");
    maximumSumPt_ = pset.getParameter<double>("maximumSumPtCut");

    applyRelativeSumPtCut_ = pset.getParameter<bool>(
      "applyRelativeSumPtCut");
    maximumRelativeSumPt_ = pset.getParameter<double>(
      "relativeSumPtCut");
    offsetRelativeSumPt_ = pset.exists("relativeSumPtOffset") ?
      pset.getParameter<double>("relativeSumPtOffset") : 0.0;

    storeRawOccupancy_ = pset.exists("storeRawOccupancy") ?
      pset.getParameter<bool>("storeRawOccupancy") : false;
    storeRawSumPt_ = pset.exists("storeRawSumPt") ?
      pset.getParameter<bool>("storeRawSumPt") : false;
    storeRawPUsumPt_ = pset.exists("storeRawPUsumPt") ?
      pset.getParameter<bool>("storeRawPUsumPt") : false;
    storeRawFootprintCorrection_ = pset.exists("storeRawFootprintCorrection") ?
      pset.getParameter<bool>("storeRawFootprintCorrection") : false;
    storeRawPhotonSumPt_outsideSignalCone_ = pset.exists("storeRawPhotonSumPt_outsideSignalCone") ?
      pset.getParameter<bool>("storeRawPhotonSumPt_outsideSignalCone") : false;

    // Sanity check on requested options.  We can't apply cuts and store the
    // raw output at the same time
    if ( applySumPtCut_ || applyOccupancyCut_ || applyRelativeSumPtCut_ ) {
      if ( storeRawSumPt_ || storeRawOccupancy_ || storeRawPUsumPt_ ) {
	throw cms::Exception("BadIsoConfig") 
	  << "A 'store raw' and a 'apply cut' option have been set to true "
	  << "simultaneously.  These options are mutually exclusive.";
      }
    }
    
    // sanity check2 - can't use weighted and unweighted iso at the same time
    if ( includeGammas_ && calculateWeights_ ) {
      throw cms::Exception("BasIsoConfig")
	<< "Both 'ApplyDiscriminationByECALIsolation' and 'ApplyDiscriminationByWeightedECALIsolation' "
	<< "have been set to true. These options are mutually exclusive.";
    }
    
    // Can only store one type
    int numStoreOptions = 0;
    if ( storeRawSumPt_                         ) ++numStoreOptions;
    if ( storeRawOccupancy_                     ) ++numStoreOptions;
    if ( storeRawPUsumPt_                       ) ++numStoreOptions;
    if ( storeRawFootprintCorrection_           ) ++numStoreOptions;
    if ( storeRawPhotonSumPt_outsideSignalCone_ ) ++numStoreOptions;
    if ( numStoreOptions > 1 ) {
      throw cms::Exception("BadIsoConfig") 
	<< "Multiple 'store sum pt' and/or 'store occupancy' options are set."
	<< " These options are mutually exclusive.";
    }

    if ( pset.exists("customOuterCone") ) {
      customIsoCone_ = pset.getParameter<double>("customOuterCone");
    } else {
      customIsoCone_ = -1;
    }

    applyPhotonPtSumOutsideSignalConeCut_ = ( pset.exists("applyPhotonPtSumOutsideSignalConeCut") ) ?
      pset.getParameter<bool>("applyPhotonPtSumOutsideSignalConeCut") : false;
    if ( applyPhotonPtSumOutsideSignalConeCut_ ) {
      maxAbsPhotonSumPt_outsideSignalCone_ = pset.getParameter<double>("maxAbsPhotonSumPt_outsideSignalCone");
      maxRelPhotonSumPt_outsideSignalCone_ = pset.getParameter<double>("maxRelPhotonSumPt_outsideSignalCone");
    }
    
    applyFootprintCorrection_ = ( pset.exists("applyFootprintCorrection") ) ?
      pset.getParameter<bool>("applyFootprintCorrection") : false;
    if ( applyFootprintCorrection_ || storeRawFootprintCorrection_ ) {
      edm::VParameterSet cfgFootprintCorrections = pset.getParameter<edm::VParameterSet>("footprintCorrections");
      for ( edm::VParameterSet::const_iterator cfgFootprintCorrection = cfgFootprintCorrections.begin();
	    cfgFootprintCorrection != cfgFootprintCorrections.end(); ++cfgFootprintCorrection ) {
	std::string selection = cfgFootprintCorrection->getParameter<std::string>("selection");
	std::string offset = cfgFootprintCorrection->getParameter<std::string>("offset");
	std::unique_ptr<FootprintCorrection> footprintCorrection(new FootprintCorrection(selection, offset));
	footprintCorrections_.push_back(std::move(footprintCorrection));
      }
    }

    // Get the quality cuts specific to the isolation region
    edm::ParameterSet isolationQCuts = qualityCutsPSet_.getParameterSet(
      "isolationQualityCuts");

    qcuts_.reset(new tau::RecoTauQualityCuts(isolationQCuts));

    vertexAssociator_.reset(
      new tau::RecoTauVertexAssociator(qualityCutsPSet_,consumesCollector()));

    applyDeltaBeta_ = pset.exists("applyDeltaBetaCorrection") ?
      pset.getParameter<bool>("applyDeltaBetaCorrection") : false;

    if ( applyDeltaBeta_ || calculateWeights_ ) {
      // Factorize the isolation QCuts into those that are used to
      // select PU and those that are not.
      std::pair<edm::ParameterSet, edm::ParameterSet> puFactorizedIsoQCuts =
	reco::tau::factorizePUQCuts(isolationQCuts);

      // Determine the pt threshold for the PU tracks
      // First check if the user specifies explicitly the cut.
      if ( pset.exists("deltaBetaPUTrackPtCutOverride") ) {
	puFactorizedIsoQCuts.second.addParameter<double>(
	  "minTrackPt",
	  pset.getParameter<double>("deltaBetaPUTrackPtCutOverride"));
      } else {
	// Secondly take it from the minGammaEt
	puFactorizedIsoQCuts.second.addParameter<double>(
          "minTrackPt",
	  isolationQCuts.getParameter<double>("minGammaEt"));
      }

      pileupQcutsPUTrackSelection_.reset(new tau::RecoTauQualityCuts(
        puFactorizedIsoQCuts.first));

      pileupQcutsGeneralQCuts_.reset(new tau::RecoTauQualityCuts(
        puFactorizedIsoQCuts.second));

      pfCandSrc_ = pset.getParameter<edm::InputTag>("particleFlowSrc");
      pfCand_token = consumes<reco::PFCandidateCollection>(pfCandSrc_);
      vertexSrc_ = pset.getParameter<edm::InputTag>("vertexSrc");
      vertex_token = consumes<reco::VertexCollection>(vertexSrc_);
      deltaBetaCollectionCone_ = pset.getParameter<double>(
        "isoConeSizeForDeltaBeta");
      std::string deltaBetaFactorFormula =
	pset.getParameter<string>("deltaBetaFactor");
      deltaBetaFormula_.reset(
        new TFormula("DB_corr", deltaBetaFactorFormula.c_str()));
    }

    applyRhoCorrection_ = pset.exists("applyRhoCorrection") ?
      pset.getParameter<bool>("applyRhoCorrection") : false;
    if ( applyRhoCorrection_ ) {
      rhoProducer_ = pset.getParameter<edm::InputTag>("rhoProducer");
      rho_token=consumes<double>(rhoProducer_);
      rhoConeSize_ = pset.getParameter<double>("rhoConeSize");
      rhoUEOffsetCorrection_ =
	pset.getParameter<double>("rhoUEOffsetCorrection");
    }
    useAllPFCands_ = pset.exists("UseAllPFCandsForWeights") ?
      pset.getParameter<bool>("UseAllPFCandsForWeights") : false;

    verbosity_ = ( pset.exists("verbosity") ) ?
      pset.getParameter<int>("verbosity") : 0;
  }

  ~PFRecoTauDiscriminationByIsolation()
  {
  }

  void beginEvent(const edm::Event& evt, const edm::EventSetup& evtSetup) override;
  double discriminate(const PFTauRef& pfTau) const override;

  inline  double weightedSum(std::vector<PFCandidatePtr> inColl_, double eta, double phi) const {
    double out = 1.0;
    for (auto const & inObj_ : inColl_){
      double sum = (inObj_->pt()*inObj_->pt())/(deltaR2(eta,phi,inObj_->eta(),inObj_->phi()));
      if(sum > 1.0) out *= sum;
    }
    return out;
  }

 private:
  std::string moduleLabel_;

  edm::ParameterSet qualityCutsPSet_;
  std::auto_ptr<tau::RecoTauQualityCuts> qcuts_;

  // Inverted QCut which selects tracks with bad DZ/trackWeight
  std::auto_ptr<tau::RecoTauQualityCuts> pileupQcutsPUTrackSelection_;
  std::auto_ptr<tau::RecoTauQualityCuts> pileupQcutsGeneralQCuts_;
  
  std::auto_ptr<tau::RecoTauVertexAssociator> vertexAssociator_;
  
  bool includeTracks_;
  bool includeGammas_;
  bool calculateWeights_;
  bool applyOccupancyCut_;
  uint32_t maximumOccupancy_;
  bool applySumPtCut_;
  double maximumSumPt_;
  bool applyRelativeSumPtCut_;
  double maximumRelativeSumPt_;
  double offsetRelativeSumPt_;
  double customIsoCone_;
  
  bool applyPhotonPtSumOutsideSignalConeCut_;
  double maxAbsPhotonSumPt_outsideSignalCone_;
  double maxRelPhotonSumPt_outsideSignalCone_;

  bool applyFootprintCorrection_;
  struct FootprintCorrection
  {
    FootprintCorrection(const std::string& selection, const std::string& offset)
      : selection_(selection),
	offset_(offset)
    {}
    ~FootprintCorrection() {}
    StringCutObjectSelector<PFTau> selection_;
    StringObjectFunction<PFTau> offset_;
  };
  std::vector<std::unique_ptr<FootprintCorrection> > footprintCorrections_;

  // Options to store the raw value in the discriminator instead of boolean pass/fail flag
  bool storeRawOccupancy_;
  bool storeRawSumPt_;
  bool storeRawPUsumPt_;
  bool storeRawFootprintCorrection_;
  bool storeRawPhotonSumPt_outsideSignalCone_;

  /* **********************************************************************
     **** Pileup Subtraction Parameters ***********************************
     **********************************************************************/

  // Delta Beta correction
  bool applyDeltaBeta_;
  edm::InputTag pfCandSrc_;
  edm::EDGetTokenT<reco::PFCandidateCollection> pfCand_token;
  // Keep track of how many vertices are in the event
  edm::InputTag vertexSrc_;
  edm::EDGetTokenT<reco::VertexCollection> vertex_token;
  std::vector<reco::PFCandidatePtr> chargedPFCandidatesInEvent_;
  // Size of cone used to collect PU tracks
  double deltaBetaCollectionCone_;
  std::auto_ptr<TFormula> deltaBetaFormula_;
  double deltaBetaFactorThisEvent_;
  
  // Rho correction
  bool applyRhoCorrection_;
  bool useAllPFCands_;
  edm::InputTag rhoProducer_;
  edm::EDGetTokenT<double> rho_token;
  double rhoConeSize_;
  double rhoUEOffsetCorrection_;
  double rhoCorrectionThisEvent_;
  double rhoThisEvent_;

  // Flag to enable/disable debug output
  int verbosity_;
};

void PFRecoTauDiscriminationByIsolation::beginEvent(const edm::Event& event, const edm::EventSetup& eventSetup) 
{
  // NB: The use of the PV in this context is necessitated by its use in
  // applying quality cuts to the different objects in the isolation cone
  // The vertex associator contains the logic to select the appropriate vertex
  // We need to pass it the event so it can load the vertices.
  vertexAssociator_->setEvent(event);

  // If we are applying the delta beta correction, we need to get the PF
  // candidates from the event so we can find the PU tracks.
  if ( applyDeltaBeta_ || calculateWeights_ ) {
    // Collect all the PF pile up tracks
    edm::Handle<reco::PFCandidateCollection> pfCandidates;
    event.getByToken(pfCand_token, pfCandidates);
    chargedPFCandidatesInEvent_.clear();
    chargedPFCandidatesInEvent_.reserve(pfCandidates->size());
    size_t numPFCandidates = pfCandidates->size();
    for ( size_t i = 0; i < numPFCandidates; ++i ) {
      reco::PFCandidatePtr pfCandidate(pfCandidates, i);
      if ( pfCandidate->charge() != 0 ) {
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

  if ( applyRhoCorrection_ ) {
    edm::Handle<double> rhoHandle_;
    event.getByToken(rho_token, rhoHandle_);
    rhoThisEvent_ = (*rhoHandle_ - rhoUEOffsetCorrection_)*
      (3.14159)*rhoConeSize_*rhoConeSize_;
  }
}

double
PFRecoTauDiscriminationByIsolation::discriminate(const PFTauRef& pfTau) const
{
    LogDebug("discriminate") << " tau: Pt = " << pfTau->pt() << ", eta = " << pfTau->eta() << ", phi = " << pfTau->phi();
    LogDebug("discriminate") << *pfTau ;

  // collect the objects we are working with (ie tracks, tracks+gammas, etc)
  std::vector<PFCandidatePtr> isoCharged_;
  std::vector<PFCandidatePtr> isoNeutral_;
  std::vector<PFCandidatePtr> isoPU_;
  PFCandidateCollection isoNeutralWeight_;
  std::vector<PFCandidatePtr> chPV_;
  isoCharged_.reserve(pfTau->isolationPFChargedHadrCands().size());
  isoNeutral_.reserve(pfTau->isolationPFGammaCands().size());
  isoPU_.reserve(chargedPFCandidatesInEvent_.size());
  isoNeutralWeight_.reserve(pfTau->isolationPFGammaCands().size());

  chPV_.reserve(chargedPFCandidatesInEvent_.size());

  // Get the primary vertex associated to this tau
  reco::VertexRef pv = vertexAssociator_->associatedVertex(*pfTau);
  // Let the quality cuts know which the vertex to use when applying selections
  // on dz, etc.
  if ( verbosity_ ) {
    if ( pv.isNonnull() ) {
      LogTrace("discriminate") << "pv: x = " << pv->position().x() << ", y = " << pv->position().y() << ", z = " << pv->position().z() ;
    } else {
      LogTrace("discriminate") << "pv: N/A" ;
    }
    if ( pfTau->leadPFChargedHadrCand().isNonnull() ) {
      LogTrace("discriminate") << "leadPFChargedHadron:" 
		<< " Pt = " << pfTau->leadPFChargedHadrCand()->pt() << "," 
		<< " eta = " << pfTau->leadPFChargedHadrCand()->eta() << "," 
		<< " phi = " << pfTau->leadPFChargedHadrCand()->phi() ;
    } else {
      LogTrace("discriminate") << "leadPFChargedHadron: N/A" ; 
    }
  }

  // CV: isolation is not well defined in case primary vertex or leading charged hadron do not exist
  if ( !(pv.isNonnull() && pfTau->leadPFChargedHadrCand().isNonnull()) ) return 0.;

  qcuts_->setPV(pv);
  qcuts_->setLeadTrack(pfTau->leadPFChargedHadrCand());

  if ( applyDeltaBeta_ || calculateWeights_) {
    pileupQcutsGeneralQCuts_->setPV(pv);
    pileupQcutsGeneralQCuts_->setLeadTrack(pfTau->leadPFChargedHadrCand());
    pileupQcutsPUTrackSelection_->setPV(pv);
    pileupQcutsPUTrackSelection_->setLeadTrack(pfTau->leadPFChargedHadrCand());
  }

  // Load the tracks if they are being used.
  if ( includeTracks_ ) {
    for( auto const & cand : pfTau->isolationPFChargedHadrCands() ) {
      if ( qcuts_->filterCandRef(cand) ) {
	LogTrace("discriminate") << "adding charged iso cand with pt " << cand->pt() ;
        isoCharged_.push_back(cand);
      }
    }
  }
  if ( includeGammas_ || calculateWeights_ ) {
    for( auto const & cand : pfTau->isolationPFGammaCands() ) {
      if ( qcuts_->filterCandRef(cand) ) {
	LogTrace("discriminate") << "adding neutral iso cand with pt " << cand->pt() ;
        isoNeutral_.push_back(cand);
      }
    }
  }

  typedef reco::tau::cone::DeltaRPtrFilter<PFCandidatePtr> DRFilter;
  typedef reco::tau::cone::DeltaRFilter<PFCandidate> DRFilter2;

  // If desired, get PU tracks.
  if ( applyDeltaBeta_ || calculateWeights_) {
    // First select by inverted the DZ/track weight cuts. True = invert
    if ( verbosity_ ) {
      std::cout << "Initial PFCands: " << chargedPFCandidatesInEvent_.size() << std::endl;
    }

    std::vector<PFCandidatePtr> allPU =
      pileupQcutsPUTrackSelection_->filterCandRefs(
          chargedPFCandidatesInEvent_, true);

    std::vector<PFCandidatePtr> allNPU =
      pileupQcutsPUTrackSelection_->filterCandRefs(
	  chargedPFCandidatesInEvent_);
      LogTrace("discriminate") << "After track cuts: " << allPU.size() ;

    // Now apply the rest of the cuts, like pt, and TIP, tracker hits, etc
    if ( !useAllPFCands_ ) {
      std::vector<PFCandidatePtr> cleanPU =
        pileupQcutsGeneralQCuts_->filterCandRefs(allPU);

      std::vector<PFCandidatePtr> cleanNPU =
        pileupQcutsGeneralQCuts_->filterCandRefs(allNPU);

      LogTrace("discriminate") << "After cleaning cuts: " << cleanPU.size() ;

      // Only select PU tracks inside the isolation cone.
      DRFilter deltaBetaFilter(pfTau->p4(), 0, deltaBetaCollectionCone_);
      for ( auto const & cand : cleanPU ) {
	if ( deltaBetaFilter(cand) )  isoPU_.push_back(cand);
      }
      
      for ( auto const & cand : cleanNPU ) {
	if ( deltaBetaFilter(cand) ) chPV_.push_back(cand);
      }
      LogTrace("discriminate") << "After cone cuts: " << isoPU_.size() << " " << chPV_.size() ;
    } else {
      isoPU_ = allPU;
      chPV_ = allNPU;
    }
  }

  if ( calculateWeights_ ) {
    for ( auto const & isoObject : isoNeutral_ ) {
      if ( isoObject->charge() != 0 ) {
	// weight only neutral objects
	isoNeutralWeight_.push_back(*isoObject);
	continue;
      }

      double eta = isoObject->eta();
      double phi = isoObject->phi();
      double sumNPU = 0.5*log(weightedSum(chPV_, eta, phi));
      
      double sumPU = 0.5*log(weightedSum(isoPU_, eta, phi));
      PFCandidate neutral = (*isoObject);
      if ( (sumNPU + sumPU) > 0 ) neutral.setP4(((sumNPU)/(sumNPU + sumPU))*neutral.p4());
      
      isoNeutralWeight_.push_back(neutral);
    }
  }
  
  // Check if we want a custom iso cone
  if ( customIsoCone_ >= 0. ) {
    DRFilter filter(pfTau->p4(), 0, customIsoCone_);
    DRFilter2 filter2(pfTau->p4(), 0, customIsoCone_);
    std::vector<PFCandidatePtr> isoCharged_filter;
    std::vector<PFCandidatePtr> isoNeutral_filter;
    PFCandidateCollection isoNeutralWeight_filter;
    // Remove all the objects not in our iso cone
    for( auto const & isoObject : isoCharged_ ) {
      if ( filter(isoObject) ) isoCharged_filter.push_back(isoObject);
    }
    if(!calculateWeights_){
      for( auto const & isoObject : isoNeutral_ ) {
	if ( filter(isoObject) ) isoNeutral_filter.push_back(isoObject);
      }
      isoNeutral_ = isoNeutral_filter;
    }else{
      for( auto const & isoObject : isoNeutralWeight_){
	if ( filter2(isoObject) ) isoNeutralWeight_filter.push_back(isoObject);
      }
      isoNeutralWeight_ = isoNeutralWeight_filter;
    }
    isoCharged_ = isoCharged_filter;
  }

  bool failsOccupancyCut     = false;
  bool failsSumPtCut         = false;
  bool failsRelativeSumPtCut = false;

//--- nObjects requirement
  int neutrals = isoNeutral_.size();

  if ( applyDeltaBeta_ ) {
    neutrals -= TMath::Nint(deltaBetaFactorThisEvent_*isoPU_.size());
  }
  if ( neutrals < 0 ) {
    neutrals = 0;
  }

  size_t nOccupants = isoCharged_.size() + neutrals;

  failsOccupancyCut = ( nOccupants > maximumOccupancy_ );

  double footprintCorrection_value = 0.;
  if ( applyFootprintCorrection_ || storeRawFootprintCorrection_ ) {
    for ( std::vector<std::unique_ptr<FootprintCorrection> >::const_iterator footprintCorrection = footprintCorrections_.begin();
	  footprintCorrection != footprintCorrections_.end(); ++footprintCorrection ) {
      if ( (*footprintCorrection)->selection_(*pfTau) ) {
	footprintCorrection_value = (*footprintCorrection)->offset_(*pfTau);
      }
    }
  }

  double totalPt = 0.;
  double puPt = 0.;
//--- Sum PT requirement
  if ( applySumPtCut_ || applyRelativeSumPtCut_ || storeRawSumPt_ || storeRawPUsumPt_ ) {
    double chargedPt = 0.;
    double neutralPt = 0.;
    double weightedNeutralPt = 0.;
    for ( auto const & isoObject : isoCharged_ ) {
      chargedPt += isoObject->pt();
    }
    if ( !calculateWeights_ ) {
      for ( auto const & isoObject : isoNeutral_ ) {
	neutralPt += isoObject->pt();
      }
    } else {
      for ( auto const & isoObject : isoNeutralWeight_ ) {
	weightedNeutralPt += isoObject.pt();
      }
    }
    for ( auto const & isoObject : isoPU_ ) {
      puPt += isoObject->pt();
    }
    LogTrace("discriminate") << "chargedPt = " << chargedPt ;
    LogTrace("discriminate") << "neutralPt = " << neutralPt ;
    LogTrace("discriminate") << "weighted neutral Pt = " << weightedNeutralPt ;
    LogTrace("discriminate") << "puPt = " << puPt << " (delta-beta corr. = " << (deltaBetaFactorThisEvent_*puPt) << ")" ;
    
    if ( calculateWeights_ ) {
      neutralPt = weightedNeutralPt;
    }

    if ( applyDeltaBeta_ ) {
      neutralPt -= (deltaBetaFactorThisEvent_*puPt);
    }
    
    if ( applyFootprintCorrection_ ) {
      neutralPt -= footprintCorrection_value;
    }
    
    if ( applyRhoCorrection_ ) {
      neutralPt -= rhoThisEvent_;
    }

    if ( neutralPt < 0. ) {
      neutralPt = 0.;
    }

    totalPt = chargedPt + neutralPt;
    LogTrace("discriminate") << "totalPt = " << totalPt << " (cut = " << maximumSumPt_ << ")" ;

    failsSumPtCut = (totalPt > maximumSumPt_);

//--- Relative Sum PT requirement
    failsRelativeSumPtCut = (totalPt > ((pfTau->pt() - offsetRelativeSumPt_)*maximumRelativeSumPt_));
  }
    
  bool failsPhotonPtSumOutsideSignalConeCut = false;
  double photonSumPt_outsideSignalCone = 0.;
  if ( applyPhotonPtSumOutsideSignalConeCut_ || storeRawPhotonSumPt_outsideSignalCone_ ) {
    const std::vector<reco::PFCandidatePtr>& signalPFGammas = pfTau->signalPFGammaCands();
    for ( std::vector<reco::PFCandidatePtr>::const_iterator signalPFGamma = signalPFGammas.begin();
	  signalPFGamma != signalPFGammas.end(); ++signalPFGamma ) {
      double dR = deltaR(pfTau->eta(), pfTau->phi(), (*signalPFGamma)->eta(), (*signalPFGamma)->phi());
      if ( dR > pfTau->signalConeSize() ) photonSumPt_outsideSignalCone += (*signalPFGamma)->pt();
    }
    if ( photonSumPt_outsideSignalCone > maxAbsPhotonSumPt_outsideSignalCone_ || photonSumPt_outsideSignalCone > (maxRelPhotonSumPt_outsideSignalCone_*pfTau->pt()) ) {
      failsPhotonPtSumOutsideSignalConeCut = true;
    }
  }

  bool fails = (applyOccupancyCut_ && failsOccupancyCut) ||
    (applySumPtCut_ && failsSumPtCut) ||
    (applyRelativeSumPtCut_ && failsRelativeSumPtCut) ||
    (applyPhotonPtSumOutsideSignalConeCut_ && failsPhotonPtSumOutsideSignalConeCut);

  // We did error checking in the constructor, so this is safe.
  if ( storeRawSumPt_ ) {
    return totalPt;
  } else if ( storeRawPUsumPt_ ) {
    if ( applyDeltaBeta_ ) return puPt;
    else if ( applyRhoCorrection_ ) return rhoThisEvent_;
    else return 0.;
  } else if ( storeRawOccupancy_ ) {
    return nOccupants;
  } else if ( storeRawFootprintCorrection_ ) {
    return footprintCorrection_value;
  } else if ( storeRawPhotonSumPt_outsideSignalCone_ ) {
    return photonSumPt_outsideSignalCone;
  } else {
    return (fails ? 0. : 1.);
  }
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationByIsolation);
