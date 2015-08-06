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

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/METReco/interface/CorrMETData.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include <string>
#include <vector>

class ShiftedParticleMETcorrInputProducer : public edm::global::EDProducer<>
{
 public:

  explicit ShiftedParticleMETcorrInputProducer(const edm::ParameterSet&);
  ~ShiftedParticleMETcorrInputProducer();

 private:
  typedef edm::View<reco::Candidate> CandidateView;

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const;

  const edm::EDGetTokenT<CandidateView> srcOriginalToken_;
  const edm::EDGetTokenT<CandidateView> srcShiftedToken_;
};

#endif




