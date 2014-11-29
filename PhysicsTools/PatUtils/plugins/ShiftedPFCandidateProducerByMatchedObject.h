#ifndef PhysicsTools_PatUtils_ShiftedPFCandidateProducerByMatchedObject_h
#define PhysicsTools_PatUtils_ShiftedPFCandidateProducerByMatchedObject_h

/** \class ShiftedPFCandidateProducerByMatchedObject
 *
 * Vary energy of PFCandidates coinciding in eta-phi with selected electrons/muons/tau-jets/jets
 * by electron/muon/tau-jet/jet energy uncertainty.
 *
 * \author Christian Veelken, LLR
 *
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <string>
#include <vector>

class ShiftedPFCandidateProducerByMatchedObject : public edm::stream::EDProducer<>  
{
 public:

  explicit ShiftedPFCandidateProducerByMatchedObject(const edm::ParameterSet&);
  ~ShiftedPFCandidateProducerByMatchedObject();
    
 private:

  void produce(edm::Event&, const edm::EventSetup&);

  edm::EDGetTokenT<reco::PFCandidateCollection > srcPFCandidates_; 
  edm::EDGetTokenT<edm::View<reco::Candidate> > srcUnshiftedObjects_; 
  edm::EDGetTokenT<edm::View<reco::Candidate> > srcShiftedObjects_; 

  double dRmatch_PFCandidate_;
  double dR2match_PFCandidate_;
  double dRmatch_Object_;
  double dR2match_Object_;

  struct objectEntryType
  {
    objectEntryType(const reco::Candidate::LorentzVector& shiftedObjectP4, 
		    const reco::Candidate::LorentzVector& unshiftedObjectP4, double dRmatch)
      : shiftedObjectP4_(shiftedObjectP4),
	unshiftedObjectP4_(unshiftedObjectP4),
	dRmatch_(dRmatch),
	isValidMatch_(false)
    {
      if ( unshiftedObjectP4.energy() > 0. ) {
	shift_ = (shiftedObjectP4.energy()/unshiftedObjectP4.energy()) - 1.;
	isValidMatch_ = true;
      }
    }
    ~objectEntryType() {}
    reco::Candidate::LorentzVector shiftedObjectP4_;
    reco::Candidate::LorentzVector unshiftedObjectP4_;
    double dRmatch_;
    double shift_;
    bool isValidMatch_;
  };

  std::vector<objectEntryType> objects_;
};

#endif


 

