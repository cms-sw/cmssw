#ifndef HiggsAnalysis_HiggsToZZ4LeptonsSkim
#define HiggsAnalysis_HiggsToZZ4LeptonsSkim

/* \class HiggsTo4LeptonsSkim
 *
 *
 * Filter to select 4 lepton events based on the
 * 1 or 2 electron or 1 or 2 muon HLT trigger, 
 * and four leptons (no flavour requirement).
 * No charge requirements are applied on event.
 *
 * At this stage, the L3 trigger isn't setup, so mimic L3 trigger
 * selection using full reconstruction
 *
 * \author Dominique Fortin - UC Riverside
 *
 */

// user include files
#include <HiggsAnalysis/Skimming/interface/HiggsAnalysisSkimType.h>

#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/ParameterSet/interface/InputTag.h"


class HiggsToZZ4LeptonsSkim : public HiggsAnalysisSkimType {
  
 public:
  // Constructor
  explicit HiggsToZZ4LeptonsSkim(const edm::ParameterSet&);

  // Destructor
  virtual ~HiggsToZZ4LeptonsSkim(){};

  /// Get event properties to send to builder to fill seed collection
  virtual bool skim(edm::Event&, const edm::EventSetup& );


 private:
  bool debug;
  float muonMinPt;
  float elecMinEt;
  int nLeptonMin;

  // Reco samples
  edm::InputTag recTrackLabel;
  edm::InputTag theGLBMuonLabel;
  edm::InputTag thePixelGsfELabel;
};

#endif
