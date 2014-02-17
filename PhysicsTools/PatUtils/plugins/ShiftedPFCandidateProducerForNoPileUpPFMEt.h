#ifndef PhysicsTools_PatUtils_ShiftedPFCandidateProducerForNoPileUpPFMEt_h
#define PhysicsTools_PatUtils_ShiftedPFCandidateProducerForNoPileUpPFMEt_h

/** \class ShiftedPFCandidateProducerForNoPileUpPFMEt
 *
 * Vary energy of PFCandidates which are (are not) within jets of Pt > 10 GeV 
 * by jet energy uncertainty (by 10% "unclustered" energy uncertainty)
 *
 * NOTE: Auxiliary class specific to estimating systematic uncertainty 
 *       on PFMET reconstructed by no-PU MET reconstruction algorithm
 *      (implemented in JetMETCorrections/Type1MET/src/NoPileUpPFMETProducer.cc)
 *
 *       In case all PFCandidates not within jets of Pt > 30 GeV would be varied
 *       by the 10% "unclustered" energy uncertainty, the systematic uncertainty
 *       on the reconstructed no-PU MET would be overestimated significantly !!
 *
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.1 $
 *
 * $Id: ShiftedPFCandidateProducerForNoPileUpPFMEt.h,v 1.1 2012/08/31 10:06:13 veelken Exp $
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

#include <string>
#include <vector>

class ShiftedPFCandidateProducerForNoPileUpPFMEt : public edm::EDProducer  
{
 public:

  explicit ShiftedPFCandidateProducerForNoPileUpPFMEt(const edm::ParameterSet&);
  ~ShiftedPFCandidateProducerForNoPileUpPFMEt();
    
 private:

  void produce(edm::Event&, const edm::EventSetup&);

  std::string moduleLabel_;

  edm::InputTag srcPFCandidates_; 
  edm::InputTag srcJets_;

  edm::FileInPath jetCorrInputFileName_;
  std::string jetCorrPayloadName_;
  std::string jetCorrUncertaintyTag_;
  JetCorrectorParameters* jetCorrParameters_;
  JetCorrectionUncertainty* jecUncertainty_;

  double minJetPt_;

  double shiftBy_;
  
  double unclEnUncertainty_;
};

#endif


 

