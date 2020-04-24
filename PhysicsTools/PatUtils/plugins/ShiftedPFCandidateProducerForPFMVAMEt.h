#ifndef PhysicsTools_PatUtils_ShiftedPFCandidateProducerForPFMVAMEt_h
#define PhysicsTools_PatUtils_ShiftedPFCandidateProducerForPFMVAMEt_h

/** \class ShiftedPFCandidateProducerForPFMVAMEt
 *
 * Vary energy of PFCandidates coinciding in eta-phi with selected electrons/muons/tau-jets/jets
 * by electron/muon/tau-jet/jet energy uncertainty.
 *
 * NOTE: Auxiliary class specific to estimating systematic uncertainty
 *       on PFMET reconstructed by MVA-based algorithm
 *      (implemented in RecoMET/METPUSubtraction/src/PFMETProducerMVA.cc)
 *
 * \author Christian Veelken, LLR
 *
 *
 *
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include <string>
#include <vector>

class ShiftedPFCandidateProducerForPFMVAMEt : public edm::stream::EDProducer<>
{
 public:

  explicit ShiftedPFCandidateProducerForPFMVAMEt(const edm::ParameterSet&);
  ~ShiftedPFCandidateProducerForPFMVAMEt();

 private:
  typedef edm::View<reco::Candidate> CandidateView;

  void produce(edm::Event&, const edm::EventSetup&);

  std::string moduleLabel_;

  edm::EDGetTokenT<reco::PFCandidateCollection> srcPFCandidatesToken_;
  edm::EDGetTokenT<CandidateView>               srcUnshiftedObjectsToken_;
  edm::EDGetTokenT<CandidateView>               srcShiftedObjectsToken_;

  double dRmatch_PFCandidate_;
  double dRmatch_Object_;

  double dR2match_PFCandidate_;
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




