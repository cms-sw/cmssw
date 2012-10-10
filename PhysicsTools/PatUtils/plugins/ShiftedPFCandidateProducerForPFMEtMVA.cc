#include "PhysicsTools/PatUtils/plugins/ShiftedPFCandidateProducerForPFMEtMVA.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Math/interface/deltaR.h"

ShiftedPFCandidateProducerForPFMEtMVA::ShiftedPFCandidateProducerForPFMEtMVA(const edm::ParameterSet& cfg)
  : moduleLabel_(cfg.getParameter<std::string>("@module_label"))
{
  srcPFCandidates_ = cfg.getParameter<edm::InputTag>("srcPFCandidates");
  srcUnshiftedObjects_ = cfg.getParameter<edm::InputTag>("srcUnshiftedObjects");
  srcShiftedObjects_ = cfg.getParameter<edm::InputTag>("srcShiftedObjects");

  dRmatch_PFCandidate_ = cfg.getParameter<double>("dRmatch_PFCandidate");
  dRmatch_Object_ = cfg.exists("dRmatch_Object") ?
    cfg.getParameter<double>("dRmatch_Object") : 0.1;

  produces<reco::PFCandidateCollection>();
}

ShiftedPFCandidateProducerForPFMEtMVA::~ShiftedPFCandidateProducerForPFMEtMVA()
{
// nothing to be done yet...
}

void ShiftedPFCandidateProducerForPFMEtMVA::produce(edm::Event& evt, const edm::EventSetup& es)
{
  edm::Handle<reco::PFCandidateCollection> originalPFCandidates;
  evt.getByLabel(srcPFCandidates_, originalPFCandidates);

  typedef edm::View<reco::Candidate> CandidateView;

  edm::Handle<CandidateView> unshiftedObjects;
  evt.getByLabel(srcUnshiftedObjects_, unshiftedObjects);

  edm::Handle<CandidateView> shiftedObjects;
  evt.getByLabel(srcShiftedObjects_, shiftedObjects);

  objects_.clear();
  
  for ( CandidateView::const_iterator unshiftedObject = unshiftedObjects->begin();
	unshiftedObject != unshiftedObjects->end(); ++unshiftedObject ) {
    bool isMatched_Object = false;
    double dRbestMatch_Object = 1.e+3;
    reco::Candidate::LorentzVector shiftedObjectP4_matched;
    for ( CandidateView::const_iterator shiftedObject = shiftedObjects->begin();
	shiftedObject != shiftedObjects->end(); ++shiftedObject ) {
      double dR = deltaR(unshiftedObject->p4(), shiftedObject->p4());
      if ( dR < dRmatch_Object_ && dR < dRbestMatch_Object ) {
	shiftedObjectP4_matched = shiftedObject->p4();
	isMatched_Object = true;
	dRbestMatch_Object = dR;
      }
    }
    if ( isMatched_Object ) {
      objects_.push_back(objectEntryType(shiftedObjectP4_matched, unshiftedObject->p4(), dRbestMatch_Object));
    }
  }
 
  std::auto_ptr<reco::PFCandidateCollection> shiftedPFCandidates(new reco::PFCandidateCollection);
    
  for ( reco::PFCandidateCollection::const_iterator originalPFCandidate = originalPFCandidates->begin();
	originalPFCandidate != originalPFCandidates->end(); ++originalPFCandidate ) {
    
    double shift = 0.;
    bool applyShift = false;
    double dRbestMatch_PFCandidate = 1.e+3;
    for ( std::vector<objectEntryType>::const_iterator object = objects_.begin();
	  object != objects_.end(); ++object ) {
      if ( !object->isValidMatch_ ) continue;
      double dR = deltaR(originalPFCandidate->p4(), object->unshiftedObjectP4_);
      if ( dR < dRmatch_PFCandidate_ && dR < dRbestMatch_PFCandidate ) {
	shift = object->shift_;
	applyShift = true;
	dRbestMatch_PFCandidate = dR;
      }
    }
    
    reco::Candidate::LorentzVector shiftedPFCandidateP4 = originalPFCandidate->p4();
    if ( applyShift ) shiftedPFCandidateP4 *= (1. + shift);
    
    reco::PFCandidate shiftedPFCandidate(*originalPFCandidate);      
    shiftedPFCandidate.setP4(shiftedPFCandidateP4);
    
    shiftedPFCandidates->push_back(shiftedPFCandidate);
  }
  
  evt.put(shiftedPFCandidates);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ShiftedPFCandidateProducerForPFMEtMVA);


 
