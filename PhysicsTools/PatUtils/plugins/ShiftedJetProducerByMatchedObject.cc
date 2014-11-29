#include "PhysicsTools/PatUtils/plugins/ShiftedJetProducerByMatchedObject.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Math/interface/deltaR.h"

template <typename T>
ShiftedJetProducerByMatchedObjectT<T>::ShiftedJetProducerByMatchedObjectT(const edm::ParameterSet& cfg)
  : moduleLabel_(cfg.getParameter<std::string>("@module_label"))
{
  srcJets_ = consumes<JetCollection>(cfg.getParameter<edm::InputTag>("srcJets"));
  srcUnshiftedObjects_ = consumes<edm::View<reco::Candidate> >(cfg.getParameter<edm::InputTag>("srcUnshiftedObjects"));
  srcShiftedObjects_ = consumes<edm::View<reco::Candidate> >(cfg.getParameter<edm::InputTag>("srcShiftedObjects"));

  dRmatch_Jet_ = cfg.getParameter<double>("dRmatch_Jet");
  dRmatch_Object_ = cfg.exists("dRmatch_Object") ?
    cfg.getParameter<double>("dRmatch_Object") : 0.1;

  dR2match_Jet_ = dRmatch_Jet_*dRmatch_Jet_;
  dR2match_Object_ = dRmatch_Object_*dRmatch_Object_;

  produces<JetCollection>();
}

template <typename T>
ShiftedJetProducerByMatchedObjectT<T>::~ShiftedJetProducerByMatchedObjectT()
{
// nothing to be done yet...
}

template <typename T>
void ShiftedJetProducerByMatchedObjectT<T>::produce(edm::Event& evt, const edm::EventSetup& es)
{
  edm::Handle<JetCollection> originalJets;
  evt.getByToken(srcJets_, originalJets);

  edm::Handle<reco::CandidateView> unshiftedObjects;
  evt.getByToken(srcUnshiftedObjects_, unshiftedObjects);

  edm::Handle<reco::CandidateView> shiftedObjects;
  evt.getByToken(srcShiftedObjects_, shiftedObjects);

  objects_.clear();
  
  std::vector<bool> match(shiftedObjects->size(), false);
  int prevMatch=-1;
  int cnt = 0;

  for ( reco::CandidateView::const_iterator unshiftedObject = unshiftedObjects->begin();
	unshiftedObject != unshiftedObjects->end(); ++unshiftedObject ) {
    bool isMatched_Object = false;
    double dR2bestMatch_Object = std::numeric_limits<double>::max();
    prevMatch=-1;
    cnt = 0;
    
    reco::Candidate::LorentzVector shiftedObjectP4_matched;
    for ( reco::CandidateView::const_iterator shiftedObject = shiftedObjects->begin();
	shiftedObject != shiftedObjects->end(); ++shiftedObject ) {
      if( match[ cnt ] ) continue;

      double dR2 = deltaR2(unshiftedObject->p4(), shiftedObject->p4());
      if ( dR2 < dR2match_Object_ && dR2 < dR2bestMatch_Object ) {
	shiftedObjectP4_matched = shiftedObject->p4();
	isMatched_Object = true;
	dR2bestMatch_Object = dR2;
	
	prevMatch = cnt;
      }
      cnt++;
    }
    if ( isMatched_Object ) {
      //Ambiguity removal
      match[ prevMatch ] = true;
      objects_.push_back(objectEntryType(shiftedObjectP4_matched, unshiftedObject->p4(), sqrt(dR2bestMatch_Object)));
    }
  }
 
  
  match.assign(objects_.size(), false);

  std::auto_ptr<JetCollection> shiftedJets(new JetCollection);
    
  for ( typename JetCollection::const_iterator originalJet = originalJets->begin();
	originalJet != originalJets->end(); ++originalJet ) {
    
    double shift = 0.;
    bool applyShift = false;
    double dR2bestMatch_Jet = std::numeric_limits<double>::max();
    prevMatch=-1;
    cnt = 0;
    
    for ( typename std::vector<objectEntryType>::const_iterator object = objects_.begin();
	  object != objects_.end(); ++object ) {
      if ( !object->isValidMatch_ ) continue;
      if( match[ cnt ] ) continue;

      double dR2 = deltaR2(originalJet->p4(), object->unshiftedObjectP4_);
      if ( dR2 < dR2match_Jet_ && dR2 < dR2bestMatch_Jet ) {
	shift = object->shift_;
	applyShift = true;
	dR2bestMatch_Jet = dR2;

	prevMatch = cnt;
      }
      cnt++;
    }
    
    reco::Candidate::LorentzVector shiftedJetP4 = originalJet->p4();
    if ( applyShift ) {
      //Ambiguity removal
      match[ prevMatch ] = true;
      
      shiftedJetP4 *= (1. + shift);
    }
    
    T shiftedJet(*originalJet);      
    shiftedJet.setP4(shiftedJetP4);
    
    shiftedJets->push_back(shiftedJet);
  }
  
  evt.put(shiftedJets);
}

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"

typedef ShiftedJetProducerByMatchedObjectT<reco::CaloJet> ShiftedCaloJetProducerByMatchedObject;
typedef ShiftedJetProducerByMatchedObjectT<reco::PFJet> ShiftedPFJetProducerByMatchedObject;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ShiftedCaloJetProducerByMatchedObject);
DEFINE_FWK_MODULE(ShiftedPFJetProducerByMatchedObject);


 
