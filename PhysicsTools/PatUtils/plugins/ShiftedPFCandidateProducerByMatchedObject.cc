#include "PhysicsTools/PatUtils/plugins/ShiftedPFCandidateProducerByMatchedObject.h"

const double dRDefault = 1000;

ShiftedPFCandidateProducerByMatchedObject::ShiftedPFCandidateProducerByMatchedObject(const edm::ParameterSet& cfg)
{
  srcPFCandidates_ = consumes<reco::PFCandidateCollection>(cfg.getParameter<edm::InputTag>("srcPFCandidates"));
  srcUnshiftedObjects_ = consumes<edm::View<reco::Candidate> >(cfg.getParameter<edm::InputTag>("srcUnshiftedObjects"));
  srcShiftedObjects_ = consumes<edm::View<reco::Candidate> >(cfg.getParameter<edm::InputTag>("srcShiftedObjects"));

  dRmatch_PFCandidate_ = cfg.getParameter<double>("dRmatch_PFCandidate");
  dR2match_PFCandidate_ = dRmatch_PFCandidate_*dRmatch_PFCandidate_;
  dRmatch_Object_ = cfg.exists("dRmatch_Object") ?
    cfg.getParameter<double>("dRmatch_Object") : 0.1;
  dR2match_Object_ = dRmatch_Object_*dRmatch_Object_;
  produces<reco::PFCandidateCollection>();
}

ShiftedPFCandidateProducerByMatchedObject::~ShiftedPFCandidateProducerByMatchedObject()
{
// nothing to be done yet...
}

void ShiftedPFCandidateProducerByMatchedObject::produce(edm::Event& evt, const edm::EventSetup& es)
{
  edm::Handle<reco::PFCandidateCollection> originalPFCandidates;
  evt.getByToken(srcPFCandidates_, originalPFCandidates);

  typedef edm::View<reco::Candidate> CandidateView;

  edm::Handle<CandidateView> unshiftedObjects;
  evt.getByToken(srcUnshiftedObjects_, unshiftedObjects);

  edm::Handle<CandidateView> shiftedObjects;
  evt.getByToken(srcShiftedObjects_, shiftedObjects);

  objects_.clear();
  
  CandidateView::const_iterator shiftedObjectP4_matched;
  bool isMatched_Object = false;
  double dR2bestMatch_Object = dRDefault;
  for ( CandidateView::const_iterator unshiftedObject = unshiftedObjects->begin();
	unshiftedObject != unshiftedObjects->end(); ++unshiftedObject ) {
    isMatched_Object = false;
    dR2bestMatch_Object = dRDefault;
   
    for ( CandidateView::const_iterator shiftedObject = shiftedObjects->begin();
	shiftedObject != shiftedObjects->end(); ++shiftedObject ) {
      double dR2 = deltaR2(unshiftedObject->p4(), shiftedObject->p4());
      if ( dR2 < dR2match_Object_ && dR2 < dR2bestMatch_Object ) {
	shiftedObjectP4_matched = shiftedObject;
	isMatched_Object = true;
	dR2bestMatch_Object = dR2;
      }
    }
    if ( isMatched_Object ) {
      objects_.push_back(objectEntryType(shiftedObjectP4_matched->p4(), unshiftedObject->p4(), sqrt(dR2bestMatch_Object) ));
    }
  }
 
  std::auto_ptr<reco::PFCandidateCollection> shiftedPFCandidates(new reco::PFCandidateCollection);
    
  for ( reco::PFCandidateCollection::const_iterator originalPFCandidate = originalPFCandidates->begin();
	originalPFCandidate != originalPFCandidates->end(); ++originalPFCandidate ) {
    
    double shift = 0.;
    bool applyShift = false;
    double dR2bestMatch_PFCandidate = dRDefault;
    for ( std::vector<objectEntryType>::const_iterator object = objects_.begin();
	  object != objects_.end(); ++object ) {
      if ( !object->isValidMatch_ ) continue;
      double dR2 = deltaR2(originalPFCandidate->p4(), object->unshiftedObjectP4_);
      if ( dR2 < dR2match_PFCandidate_ && dR2 < dR2bestMatch_PFCandidate ) {
	shift = object->shift_;
	applyShift = true;
	dR2bestMatch_PFCandidate = dR2;
      }
    }
    
    reco::Candidate::LorentzVector shiftedPFCandidateP4 = originalPFCandidate->p4();
    if ( applyShift ) {
      double shiftedPx = (1. + shift)*originalPFCandidate->px();
      double shiftedPy = (1. + shift)*originalPFCandidate->py();
      double shiftedPz = (1. + shift)*originalPFCandidate->pz();
      double mass = originalPFCandidate->mass();
      double shiftedEn = sqrt(shiftedPx*shiftedPx + shiftedPy*shiftedPy + shiftedPz*shiftedPz + mass*mass);
      shiftedPFCandidateP4.SetPxPyPzE(shiftedPx, shiftedPy, shiftedPz, shiftedEn);
    }
    
    reco::PFCandidate shiftedPFCandidate(*originalPFCandidate);      
    shiftedPFCandidate.setP4(shiftedPFCandidateP4);
    
    shiftedPFCandidates->push_back(shiftedPFCandidate);
  }
  
  evt.put(shiftedPFCandidates);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ShiftedPFCandidateProducerByMatchedObject);


 
