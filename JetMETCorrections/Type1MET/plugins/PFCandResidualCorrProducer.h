#ifndef JetMETCorrections_Type1MET_PFCandResidualCorrProducer_h
#define JetMETCorrections_Type1MET_PFCandResidualCorrProducer_h

/** \class PFCandResidualCorrProducer
 *
 * Apply "residual" jet energy corrections to PFCandidates,
 * in order to reduce data/MC differences in "unclustered energy" response
 *
 * \authors Christian Veelken, LLR
 *
 * \version $Revision: 1.2 $
 *
 * $Id: PFCandResidualCorrProducer.h,v 1.2 2013/05/13 16:29:20 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrector.h"

#include "JetMETCorrections/Type1MET/interface/SysShiftMETcorrExtractor.h"

#include <string>

class PFCandResidualCorrProducer : public edm::EDProducer  
{
 public:

  explicit PFCandResidualCorrProducer(const edm::ParameterSet&);
  ~PFCandResidualCorrProducer();
    
 private:

  void produce(edm::Event&, const edm::EventSetup&);

  std::string moduleLabel_;

  edm::InputTag src_;

  std::string residualCorrLabel_;
  double residualCorrEtaMax_;
  double extraCorrFactor_;
  FactorizedJetCorrector* residualCorrectorFromFile_;
  bool isMC_;
  edm::InputTag srcGenPileUpSummary_;
  FactorizedJetCorrector* residualCorrectorVsNumPileUp_data_offset_;
  FactorizedJetCorrector* residualCorrectorVsNumPileUp_data_slope_;
  FactorizedJetCorrector* residualCorrectorVsNumPileUp_mc_offset_;
  FactorizedJetCorrector* residualCorrectorVsNumPileUp_mc_slope_;  
  enum { kResidualCorrFromDB, kResidualCorrFromFile, kResidualCorrVsNumPileUp };
  int mode_;

  int verbosity_;
};

#endif


 

