#ifndef HiggsAnalysis_HiggsToZZ4LeptonsSkimProducer
#define HiggsAnalysis_HiggsToZZ4LeptonsSkimProducer

/* \class HiggsTo4LeptonsSkimProducer
 *
 *
 * Filter to select 4 lepton events based on the
 * 1 or 2 electron or 1 or 2 muon HLT trigger, 
 * and four leptons (no flavour requirement).
 * No charge requirements are applied on event.
 *
 * \author Dominique Fortin - UC - Riverside
 * modified by N. De Filippis - LLR - Ecole Polytechnique
 */

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"

using namespace edm;
using namespace std;

class HiggsToZZ4LeptonsSkimProducer : public edm::EDProducer {
  
 public:
  // Constructor
  explicit HiggsToZZ4LeptonsSkimProducer(const edm::ParameterSet&);

  // Destructor
  ~HiggsToZZ4LeptonsSkimProducer();

 private:
  virtual void produce(edm::Event& event, const edm::EventSetup& eventSetup);
  virtual void endJob();

  bool debug;
  string aliasaccept;
  float stiffMinPt;
  float softMinPt;
  int nStiffLeptonMin;
  int nLeptonMin;

  // Reco samples
  bool isGlobalMuon;
  edm::InputTag theMuonLabel;
  edm::InputTag theGsfELabel;

};

#endif
