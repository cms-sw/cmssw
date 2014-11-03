#ifndef PhysicsTools_PatUtils_ShiftedParticleMETcorrInputProducer_h
#define PhysicsTools_PatUtils_ShiftedParticleMETcorrInputProducer_h

/** \class ShiftedParticleMETcorrInputProducer
 *
 * Propagate energy variations of electrons/muons/tau-jets to MET
 *
 * \author Christian Veelken, LLR
 *
 *
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/METReco/interface/CorrMETData.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include <string>
#include <vector>

class ShiftedParticleMETcorrInputProducer : public edm::EDProducer
{
 public:

  explicit ShiftedParticleMETcorrInputProducer(const edm::ParameterSet&);
  ~ShiftedParticleMETcorrInputProducer();

 private:
  typedef edm::View<reco::Candidate> CandidateView;

  void produce(edm::Event&, const edm::EventSetup&);

  edm::EDGetTokenT<CandidateView> srcOriginalToken_;
  edm::EDGetTokenT<CandidateView> srcShiftedToken_;
};

#endif




