#ifndef PhysicsTools_PatUtils_ShiftedPFCandidateProducerForPFMEtMVA_h
#define PhysicsTools_PatUtils_ShiftedPFCandidateProducerForPFMEtMVA_h

/** \class ShiftedPFCandidateProducerForPFMEtMVA
 *
 * Vary energy of PFCandidates coinciding in eta-phi with selected electrons/muons/tau-jets/jets
 * by electron/muon/tau-jet/jet energy uncertainty.
 *
 * NOTE: Auxiliary class specific to estimating systematic uncertainty 
 *       on PFMET reconstructed by MVA-based algorithm
 *      (implemented in RecoMET/METProducers/src/PFMETProducerMVA.cc)
 *
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.1 $
 *
 * $Id: ShiftedPFCandidateProducerForPFMEtMVA.h,v 1.1 2012/08/31 10:06:14 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Candidate/interface/Candidate.h"

#include <string>
#include <vector>

class ShiftedPFCandidateProducerForPFMEtMVA : public edm::EDProducer  
{
 public:

  explicit ShiftedPFCandidateProducerForPFMEtMVA(const edm::ParameterSet&);
  ~ShiftedPFCandidateProducerForPFMEtMVA();
    
 private:

  void produce(edm::Event&, const edm::EventSetup&);

  std::string moduleLabel_;

  edm::InputTag srcPFCandidates_; 
  edm::InputTag srcUnshiftedObjects_; 
  edm::InputTag srcShiftedObjects_; 

  double dRmatch_PFCandidate_;
  double dRmatch_Object_;

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


 

