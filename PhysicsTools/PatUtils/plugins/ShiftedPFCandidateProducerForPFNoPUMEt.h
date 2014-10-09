#ifndef PhysicsTools_PatUtils_ShiftedPFCandidateProducerForPFNoPUMEt_h
#define PhysicsTools_PatUtils_ShiftedPFCandidateProducerForPFNoPUMEt_h

/** \class ShiftedPFCandidateProducerForPFNoPUMEt
 *
 * Vary energy of PFCandidates which are (are not) within jets of Pt > 10 GeV
 * by jet energy uncertainty (by 10% "unclustered" energy uncertainty)
 *
 * NOTE: Auxiliary class specific to estimating systematic uncertainty
 *       on PFMET reconstructed by no-PU MET reconstruction algorithm
 *      (implemented in JetMETCorrections/Type1MET/src/PFNoPUMETProducer.cc)
 *
 *       In case all PFCandidates not within jets of Pt > 30 GeV would be varied
 *       by the 10% "unclustered" energy uncertainty, the systematic uncertainty
 *       on the reconstructed no-PU MET would be overestimated significantly !!
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
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectionUncertainty.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"

#include <string>
#include <vector>

class ShiftedPFCandidateProducerForPFNoPUMEt : public edm::EDProducer
{
 public:

  explicit ShiftedPFCandidateProducerForPFNoPUMEt(const edm::ParameterSet&);
  ~ShiftedPFCandidateProducerForPFNoPUMEt();

 private:

  void produce(edm::Event&, const edm::EventSetup&);

  std::string moduleLabel_;

  edm::EDGetTokenT<reco::PFCandidateCollection> srcPFCandidatesToken_;
  edm::EDGetTokenT<reco::PFJetCollection>       srcJetsToken_;

  edm::FileInPath jetCorrInputFileName_;
  std::string jetCorrPayloadName_;
  std::string jetCorrUncertaintyTag_;
  JetCorrectorParameters* jetCorrParameters_;
  JetCorrectionUncertainty* jecUncertainty_;

  bool jecValidFileName_;

  double minJetPt_;

  double shiftBy_;

  double unclEnUncertainty_;

};

#endif




