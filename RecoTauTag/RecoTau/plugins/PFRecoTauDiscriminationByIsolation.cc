#include <functional>
#include <boost/foreach.hpp>
#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "RecoTauTag/RecoTau/interface/RecoTauQualityCuts.h"
#include "RecoTauTag/RecoTau/interface/RecoTauVertexAssociator.h"
#include "RecoTauTag/RecoTau/interface/ConeTools.h"
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

    includeNeutralHadrons_ = pset.exists("ApplyDiscriminationByHCALIsolation") ?
      pset.getParameter<bool>("ApplyDiscriminationByHCALIsolation") : false;

    calculateWeights1_ = pset.exists("ApplyDiscriminationByWeightedECALIsolation1") ?
      pset.getParameter<bool>("ApplyDiscriminationByWeightedECALIsolation1") : false;

    calculateWeights2_ = pset.exists("ApplyDiscriminationByWeightedECALIsolation2") ?
      pset.getParameter<bool>("ApplyDiscriminationByWeightedECALIsolation2") : false;

    calculateWeightsNH1_ = pset.exists("ApplyDiscriminationByWeightedHCALIsolation1") ?
      pset.getParameter<bool>("ApplyDiscriminationByWeightedHCALIsolation1") : false;

    calculateWeightsNH2_ = pset.exists("ApplyDiscriminationByWeightedHCALIsolation2") ?
      pset.getParameter<bool>("ApplyDiscriminationByWeightedHCALIsolation2") : false;


    if(calculateWeights1_ && calculateWeights2_){
      throw cms::Exception("BadIsoConfig") 
	<< "2 ways of reweighting neutral particles have been chosen "
	<< "simultaneously (ApplyDiscriminationByWeightedECALIsolation1 and "
	<< "ApplyDiscriminationByWeightedECALIsolation2). "
	<< " These options are mutually exclusive.";
    }

    if(calculateWeightsNH1_ && calculateWeightsNH2_){
      throw cms::Exception("BadIsoConfig") 
        << "2 ways of reweighting neutral particles have been chosen "
        << "simultaneously (ApplyDiscriminationByWeightedHCALIsolation1 and "
        << "ApplyDiscriminationByWeightedHCALIsolation2). "
        << " These options are mutually exclusive.";
    }
    
    applyWeightsGamma_=(calculateWeights1_ || calculateWeights2_ );
    applyWeightsNH_=(calculateWeightsNH1_ || calculateWeightsNH2_ );
    applyWeights_ = (applyWeightsGamma_ || applyWeightsNH_);


    applyOccupancyCut_ = pset.getParameter<bool>("applyOccupancyCut");
    maximumOccupancy_ = pset.getParameter<uint32_t>("maximumOccupancy");

    applySumPtCut_ = pset.getParameter<bool>("applySumPtCut");
    maximumSumPt_ = pset.getParameter<double>("maximumSumPtCut");

    applyRelativeSumPtCut_ = pset.getParameter<bool>(
      "applyRelativeSumPtCut");
    maximumRelativeSumPt_ = pset.getParameter<double>(
      "relativeSumPtCut");

    storeRawOccupancy_ = pset.exists("storeRawOccupancy") ?
      pset.getParameter<bool>("storeRawOccupancy") : false;
    storeRawSumPt_ = pset.exists("storeRawSumPt") ?
      pset.getParameter<bool>("storeRawSumPt") : false;
    storeRawPUsumPt_ = pset.exists("storeRawPUsumPt") ?
      pset.getParameter<bool>("storeRawPUsumPt") : false;

    // Sanity check on requested options.  We can't apply cuts and store the
    // raw output at the same time
    if ( applySumPtCut_ || applyOccupancyCut_ || applyRelativeSumPtCut_ ) {
      if ( storeRawSumPt_ || storeRawOccupancy_ || storeRawPUsumPt_ ) {
	throw cms::Exception("BadIsoConfig") 
	  << "A 'store raw' and a 'apply cut' option have been set to true "
	  << "simultaneously.  These options are mutually exclusive.";
      }
    }
    
    // Can only store one type
    int numStoreOptions = 0;
    if ( storeRawSumPt_     ) ++numStoreOptions;
    if ( storeRawOccupancy_ ) ++numStoreOptions;
    if ( storeRawPUsumPt_   ) ++numStoreOptions;
    if ( numStoreOptions > 1 ) {
      throw cms::Exception("BadIsoConfig") 
	<< "Both 'store sum pt' and 'store occupancy' options are set."
	<< " These options are mutually exclusive.";
    }

    if ( pset.exists("customOuterCone") ) {
      customIsoCone_ = pset.getParameter<double>("customOuterCone");
    } else {
      customIsoCone_ = -1;
    }

    // Get the quality cuts specific to the isolation region
    edm::ParameterSet isolationQCuts = qualityCutsPSet_.getParameterSet(
      "isolationQualityCuts");

    qcuts_.reset(new tau::RecoTauQualityCuts(isolationQCuts));

    vertexAssociator_.reset(
			    new tau::RecoTauVertexAssociator(qualityCutsPSet_,consumesCollector()));

    applyDeltaBeta_ = pset.exists("applyDeltaBetaCorrection") ?
      pset.getParameter<bool>("applyDeltaBetaCorrection") : false;

    if ( applyDeltaBeta_ || applyWeights_ ) {
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
      pfCand_token=consumes<reco::PFCandidateCollection>(pfCandSrc_);
      vertexSrc_ = pset.getParameter<edm::InputTag>("vertexSrc");
      vertex_token=consumes<reco::VertexCollection>(vertexSrc_);
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

  ~PFRecoTauDiscriminationByIsolation(){}

  void beginEvent(const edm::Event& evt, const edm::EventSetup& evtSetup) override;
  double discriminate(const PFTauRef& pfTau) override;

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
  bool includeNeutralHadrons_;
  bool calculateWeights1_;
  bool calculateWeights2_;
  bool calculateWeightsNH1_;
  bool calculateWeightsNH2_;
  bool applyWeights_;
  bool applyWeightsGamma_;
  bool applyWeightsNH_;
  bool applyOccupancyCut_;
  uint32_t maximumOccupancy_;
  bool applySumPtCut_;
  double maximumSumPt_;
  bool applyRelativeSumPtCut_;
  double maximumRelativeSumPt_;
  double customIsoCone_;
  
  // Options to store the raw value in the discriminator instead of boolean pass/fail flag
  bool storeRawOccupancy_;
  bool storeRawSumPt_;
  bool storeRawPUsumPt_;

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
  std::vector<PFCandidatePtr> isoCharged_;
  std::vector<PFCandidatePtr> isoNeutral_;
  std::vector<PFCandidatePtr> isoNeutralHadron_;
  std::vector<PFCandidatePtr> isoPU_;
  PFCandidateCollection isoNeutralWeight_;
  PFCandidateCollection isoNeutralHadronWeight_;
  std::vector<PFCandidatePtr> chPV_;
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
  if ( applyDeltaBeta_ || applyWeights_) {
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
PFRecoTauDiscriminationByIsolation::discriminate(const PFTauRef& pfTau) 
{
  if ( verbosity_ ) {
    std::cout << "<PFRecoTauDiscriminationByIsolation::discriminate (moduleLabel = " << moduleLabel_ <<")>:" << std::endl;
    std::cout << " tau: Pt = " << pfTau->pt() << ", eta = " << pfTau->eta() << ", phi = " << pfTau->phi() << std::endl;
    pfTau->dump(std::cout);
    std::cout << *pfTau << std::endl;
  }

  // collect the objects we are working with (ie tracks, tracks+gammas, etc)
  isoCharged_.clear();
  isoCharged_.reserve(pfTau->isolationPFChargedHadrCands().size());
  isoNeutral_.clear();
  isoNeutral_.reserve(pfTau->isolationPFGammaCands().size());
  isoNeutralHadron_.clear();
  isoNeutralHadron_.reserve(pfTau->isolationPFNeutrHadrCands().size());
  isoNeutralWeight_.clear();
  isoNeutralWeight_.reserve(pfTau->isolationPFGammaCands().size());
  isoNeutralHadronWeight_.clear();
  isoNeutralHadronWeight_.reserve(pfTau->isolationPFNeutrHadrCands().size());
  isoPU_.clear();
  isoPU_.reserve(chargedPFCandidatesInEvent_.size());
  chPV_.clear();
  chPV_.reserve(chargedPFCandidatesInEvent_.size());
  // Get the primary vertex associated to this tau
  reco::VertexRef pv = vertexAssociator_->associatedVertex(*pfTau);
  // Let the quality cuts know which the vertex to use when applying selections
  // on dz, etc.
  if ( verbosity_ ) {
    if ( pv.isNonnull() ) {
      std::cout << "pv: x = " << pv->position().x() << ", y = " << pv->position().y() << ", z = " << pv->position().z() << std::endl;
    } else {
      std::cout << "pv: N/A" << std::endl;
    }
    if ( pfTau->leadPFChargedHadrCand().isNonnull() ) {
      std::cout << "leadPFChargedHadron:" 
		<< " Pt = " << pfTau->leadPFChargedHadrCand()->pt() << "," 
		<< " eta = " << pfTau->leadPFChargedHadrCand()->eta() << "," 
		<< " phi = " << pfTau->leadPFChargedHadrCand()->phi() << std::endl;
    } else {
      std::cout << "leadPFChargedHadron: N/A" << std::endl; 
    }
  }

  // CV: isolation is not well defined in case primary vertex or leading charged hadron do not exist
  if ( !(pv.isNonnull() && pfTau->leadPFChargedHadrCand().isNonnull()) ) return 0.;

  qcuts_->setPV(pv);
  qcuts_->setLeadTrack(pfTau->leadPFChargedHadrCand());

  if ( applyDeltaBeta_ || applyWeights_ ) {
    pileupQcutsGeneralQCuts_->setPV(pv);
    pileupQcutsGeneralQCuts_->setLeadTrack(pfTau->leadPFChargedHadrCand());
    pileupQcutsPUTrackSelection_->setPV(pv);
    pileupQcutsPUTrackSelection_->setLeadTrack(pfTau->leadPFChargedHadrCand());
  }

  // Load the tracks if they are being used.
  if ( includeTracks_ ) {
    BOOST_FOREACH( const reco::PFCandidatePtr& cand, pfTau->isolationPFChargedHadrCands() ) {
      if ( qcuts_->filterCandRef(cand) ) {
        isoCharged_.push_back(cand);
      }
    }
  }
  if ( includeGammas_ || applyWeightsGamma_) {
    BOOST_FOREACH( const reco::PFCandidatePtr& cand, pfTau->isolationPFGammaCands() ) {
      if ( qcuts_->filterCandRef(cand) ) {
        isoNeutral_.push_back(cand);
      }
    }
  }
  if ( includeNeutralHadrons_ || applyWeightsNH_) {
    BOOST_FOREACH( const reco::PFCandidatePtr& cand, pfTau->isolationPFNeutrHadrCands() ) {
      if ( qcuts_->filterCandRef(cand) ) {
        isoNeutralHadron_.push_back(cand);
      }
    }
  }


  
  
  typedef reco::tau::cone::DeltaRPtrFilter<PFCandidatePtr> DRFilter;

  // If desired, get PU tracks.
  if ( applyDeltaBeta_ || applyWeights_) {
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

    if ( verbosity_ ) {
      std::cout << "After track cuts: " << allPU.size() << std::endl;
    }

    // Now apply the rest of the cuts, like pt, and TIP, tracker hits, etc
    if(!useAllPFCands_){
      std::vector<PFCandidatePtr> cleanPU =
        pileupQcutsGeneralQCuts_->filterCandRefs(allPU);

      std::vector<PFCandidatePtr> cleanNPU =
        pileupQcutsGeneralQCuts_->filterCandRefs(allNPU);
    

    if ( verbosity_ ) {
      std::cout << "After cleaning cuts: " << cleanPU.size() << std::endl;
    }

    // Only select PU tracks inside the isolation cone.
    DRFilter deltaBetaFilter(pfTau->p4(), 0, deltaBetaCollectionCone_);
    BOOST_FOREACH(const reco::PFCandidatePtr& cand, cleanPU) {
      if ( deltaBetaFilter(cand) ) {
        isoPU_.push_back(cand);
      }else if(useAllPFCands_) isoPU_.push_back(cand);
    }

      BOOST_FOREACH(const reco::PFCandidatePtr& cand, cleanNPU) {
        if ( deltaBetaFilter(cand) ) {
          chPV_.push_back(cand);
        }else if(useAllPFCands_) chPV_.push_back(cand);

      }
    }else{
      isoPU_=allPU;
      chPV_= allNPU;
    }
    if ( verbosity_ ) {
      std::cout << "After cone cuts: " << isoPU_.size() << std::endl;
    }
  }

  // weighting

  if (calculateWeights1_)
    {
      double sumNPU = 0.0;
      double sumPU = 0.0;
      BOOST_FOREACH( const PFCandidatePtr& isoObject, isoNeutral_ ) {
  	if(isoObject->charge() !=0){
  	  isoNeutralWeight_.push_back(*isoObject);
  	  continue;
  	}

  	sumNPU=0.0;
  	sumPU=0.0;
	double eta=isoObject->eta();
        double phi=isoObject->phi();
  	BOOST_FOREACH( const PFCandidatePtr& chPVObject,chPV_) {
  	    sumNPU += 1./(deltaR2(eta,phi,chPVObject->eta(),chPVObject->phi()));
  	}

        BOOST_FOREACH( const PFCandidatePtr& isoPUObject,isoPU_) {
	  sumPU += 1./(deltaR2(eta,phi,isoPUObject->eta(),isoPUObject->phi()));
        }
	PFCandidate neutral = *isoObject;
	if (sumNPU+sumPU>0)
	       neutral.setP4(((sumNPU)/(sumNPU+sumPU))*neutral.p4());
	isoNeutralWeight_.push_back(neutral);
      }
    }else if (calculateWeights2_)
    {
      double sumNPU = 0.0;
      double sumPU = 0.0;
      BOOST_FOREACH( const PFCandidatePtr& isoObject, isoNeutral_ ) {
        if(isoObject->charge() !=0){
          isoNeutralWeight_.push_back(*isoObject);
          continue;
        }

        sumNPU=1.0;
        sumPU=1.0;
	double eta=isoObject->eta();
	double phi=isoObject->phi();
        BOOST_FOREACH( const PFCandidatePtr& chPVObject,chPV_) {
	  double sum = (chPVObject->pt()*chPVObject->pt())/(deltaR2(eta,phi,chPVObject->eta(),chPVObject->phi()));
	  if(sum > 1.0) sumNPU *= sum;
        }
	sumNPU=0.5*log(sumNPU);

	BOOST_FOREACH( const PFCandidatePtr& isoPUObject,isoPU_) {
          double sum = (isoPUObject->pt()*isoPUObject->pt())/(deltaR2(eta,phi,isoPUObject->eta(),isoPUObject->phi()));
	  if(sum > 1.0) sumPU*=sum;
        }
	sumPU=0.5*log(sumPU);
        PFCandidate neutral = *isoObject;
        if (sumNPU+sumPU>0)  neutral.setP4(((sumNPU)/(sumNPU+sumPU))*neutral.p4());

        isoNeutralWeight_.push_back(neutral);
      }
    }

  // neutral hadron reweighting

  if (calculateWeightsNH1_)
    {
      double sumNPU = 0.0;
      double sumPU = 0.0;
      BOOST_FOREACH( const PFCandidatePtr& isoObject, isoNeutralHadron_ ) {
        if(isoObject->charge() !=0){
          isoNeutralHadronWeight_.push_back(*isoObject);
          continue;
        }

        sumNPU=0.0;
        sumPU=0.0;
        double eta=isoObject->eta();
        double phi=isoObject->phi();
        BOOST_FOREACH( const PFCandidatePtr& chPVObject,chPV_) {
	  sumNPU += 1./(deltaR2(eta,phi,chPVObject->eta(),chPVObject->phi()));
        }

        BOOST_FOREACH( const PFCandidatePtr& isoPUObject,isoPU_) {
          sumPU += 1./(deltaR2(eta,phi,isoPUObject->eta(),isoPUObject->phi()));
        }
        PFCandidate neutral = *isoObject;
        if (sumNPU+sumPU>0)
	  neutral.setP4(((sumNPU)/(sumNPU+sumPU))*neutral.p4());
        isoNeutralHadronWeight_.push_back(neutral);
      }
    }else if (calculateWeightsNH2_)
    {
      double sumNPU = 0.0;
      double sumPU = 0.0;
      BOOST_FOREACH( const PFCandidatePtr& isoObject, isoNeutralHadron_ ) {
        if(isoObject->charge() !=0){
          isoNeutralHadronWeight_.push_back(*isoObject);
          continue;
        }

        sumNPU=1.0;
        sumPU=1.0;
        double eta=isoObject->eta();
        double phi=isoObject->phi();
        BOOST_FOREACH( const PFCandidatePtr& chPVObject,chPV_) {
          double sum = (chPVObject->pt()*chPVObject->pt())/(deltaR2(eta,phi,chPVObject->eta(),chPVObject->phi()));
          if(sum > 1.0) sumNPU *= sum;
        }
        sumNPU=0.5*log(sumNPU);

        BOOST_FOREACH( const PFCandidatePtr& isoPUObject,isoPU_) {
          double sum = (isoPUObject->pt()*isoPUObject->pt())/(deltaR2(eta,phi,isoPUObject->eta(),isoPUObject->phi()));
          if(sum > 1.0) sumPU*=sum;
        }
        sumPU=0.5*log(sumPU);
        PFCandidate neutral = *isoObject;
        if (sumNPU+sumPU>0)  neutral.setP4(((sumNPU)/(sumNPU+sumPU))*neutral.p4());

        isoNeutralHadronWeight_.push_back(neutral);
      }
    }


  // Check if we want a custom iso cone
  if ( customIsoCone_ >= 0. ) {
    //std::cout << "<PFRecoTauDiscriminationByIsolation::discriminate (moduleLabel = " << moduleLabel_ <<")>:" << std::endl;
    //std::cout << " customIsoCone = " << customIsoCone_ << std::endl;
    DRFilter filter(pfTau->p4(), 0, customIsoCone_);
    std::vector<PFCandidatePtr> isoCharged_filter;
    std::vector<PFCandidatePtr> isoNeutral_filter;
    PFCandidateCollection isoNeutralWeight_filter;
    PFCandidateCollection isoNeutralHadronWeight_filter;
    // Remove all the objects not in our iso cone
    BOOST_FOREACH( const PFCandidatePtr& isoObject, isoCharged_ ) {
      if ( filter(isoObject) ) isoCharged_filter.push_back(isoObject);
    }
    BOOST_FOREACH( const PFCandidatePtr& isoObject, isoNeutral_ ) {
      if ( filter(isoObject) ) isoNeutral_filter.push_back(isoObject);
    }
    BOOST_FOREACH( const PFCandidate& isoObject, isoNeutralWeight_){
      if ( deltaR2(isoObject.eta(),isoObject.phi(),pfTau->eta(),pfTau->phi()) < customIsoCone_*customIsoCone_ ) isoNeutralWeight_filter.push_back(isoObject);
    }
    BOOST_FOREACH( const PFCandidate& isoObject, isoNeutralHadronWeight_){
      if ( deltaR2(isoObject.eta(),isoObject.phi(),pfTau->eta(),pfTau->phi()) < customIsoCone_*customIsoCone_ ) isoNeutralHadronWeight_filter.push_back(isoObject);
    }
    isoCharged_ = isoCharged_filter;
    isoNeutral_ = isoNeutral_filter;
    isoNeutralWeight_ = isoNeutralWeight_filter;
    isoNeutralHadronWeight_ = isoNeutralHadronWeight_filter;
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

  double totalPt = 0.0;
  double puPt = 0.0;
//--- Sum PT requirement
  if ( applySumPtCut_ || applyRelativeSumPtCut_ || storeRawSumPt_ || storeRawPUsumPt_ ) {
    double chargedPt = 0.0;
    double neutralPt = 0.0;
    double neutralHadronPt = 0.0;
    double weightedNeutralPt = 0.0;
    double weightedNeutralHadronPt = 0.0;
    BOOST_FOREACH ( const PFCandidatePtr& isoObject, isoCharged_ ) {
      chargedPt += isoObject->pt();
    }
    BOOST_FOREACH ( const PFCandidatePtr& isoObject, isoNeutral_ ) {
      neutralPt += isoObject->pt();
    }
    BOOST_FOREACH ( const PFCandidatePtr& isoObject, isoNeutralHadron_ ) {
      neutralHadronPt += isoObject->pt();
    }
    BOOST_FOREACH( const PFCandidate& isoObject, isoNeutralWeight_){
	weightedNeutralPt+=isoObject.pt();
    }
    BOOST_FOREACH( const PFCandidate& isoObject, isoNeutralHadronWeight_){
      weightedNeutralHadronPt+=isoObject.pt();
    }
    
    BOOST_FOREACH ( const PFCandidatePtr& isoObject, isoPU_ ) {
      puPt += isoObject->pt();
    }
    if ( verbosity_ ) {
      std::cout << "chargedPt = " << chargedPt << std::endl;
      std::cout << "neutralPt = " << neutralPt << std::endl;
      std::cout << "neutral hadron Pt = " << neutralHadronPt << std::endl;
      std::cout << "weighted neutral Pt = " << weightedNeutralPt << std::endl;
      std::cout << "weighted neutral hadron Pt = " << weightedNeutralHadronPt << std::endl;
      std::cout << "puPt = " << puPt << " (delta-beta corr. = " << (deltaBetaFactorThisEvent_*puPt) << ")" << std::endl;
    }
    if( applyWeightsGamma_) {
      neutralPt = weightedNeutralPt;
    }
    if( applyWeightsNH_ ){
      neutralHadronPt = weightedNeutralHadronPt;
    }
    neutralPt+=neutralHadronPt;
    
    if ( applyDeltaBeta_ ) {
      neutralPt -= deltaBetaFactorThisEvent_*puPt;
    }

    if ( applyRhoCorrection_ ) {
      neutralPt -= rhoThisEvent_;
    }
    
    if( applyWeights_) {
      neutralPt = weightedNeutralPt;
    }

    if ( neutralPt  < 0.0 ) {
      neutralPt = 0.0;
    }

    totalPt = chargedPt + neutralPt;
    if ( verbosity_ ) {
      std::cout << "totalPt = " << totalPt << " (cut = " << maximumSumPt_ << ")" << std::endl;
    }

    failsSumPtCut = (totalPt > maximumSumPt_);

//--- Relative Sum PT requirement
    failsRelativeSumPtCut = (totalPt > (pfTau->pt()*maximumRelativeSumPt_));
  }

  bool fails = (applyOccupancyCut_ && failsOccupancyCut) ||
    (applySumPtCut_ && failsSumPtCut) ||
    (applyRelativeSumPtCut_ && failsRelativeSumPtCut);

  // We did error checking in the constructor, so this is safe.
  if ( storeRawSumPt_ ) {
    return totalPt;
  } else if ( storeRawPUsumPt_ ) {
    if ( applyDeltaBeta_ ) return puPt;
    else if ( applyRhoCorrection_ ) return rhoThisEvent_;
    else return 0.;
  } else if ( storeRawOccupancy_ ) {
    return nOccupants;
  } else {
    return (fails ? 0. : 1.);
  }
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationByIsolation);
