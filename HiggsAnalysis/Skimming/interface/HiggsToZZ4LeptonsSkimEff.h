#ifndef HiggsAnalysis_HiggsToZZ4LeptonsSkimEff
#define HiggsAnalysis_HiggsToZZ4LeptonsSkimEff

/* \class HiggsTo4LeptonsSkimEff
 *
 * EDAnalyzer to study the HLT and skim efficiency for signal
 * A preselection on the generaged event is built in
 *
 * \author Dominique Fortin - UC Riverside
 *
 */

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

class HiggsToZZ4LeptonsSkimEff : public edm::EDAnalyzer {
  
 public:
  // Constructor
  explicit HiggsToZZ4LeptonsSkimEff(const edm::ParameterSet&);

  // Destructor
  ~HiggsToZZ4LeptonsSkimEff();

  /// Get event properties to send to builder to fill seed collection
  virtual void analyze(const edm::Event&, const edm::EventSetup& );


 private:
  bool debug;
  float stiffMinPt;
  float softMinPt;
  int nStiffLeptonMin;
  int nLeptonMin;

  int nEvents, nSelFourE, nSelFourM, nSelTwoETwoM, nSelFourL, nSelTau;
  int nFourE, nFourM, nTwoETwoM, nFourL, nTau;

  // Reco samples
  edm::InputTag recTrackLabel;
  edm::InputTag theGLBMuonLabel;
  edm::InputTag thePixelGsfELabel;
};

#endif
