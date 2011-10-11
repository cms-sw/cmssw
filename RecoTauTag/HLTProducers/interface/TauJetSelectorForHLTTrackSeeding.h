#ifndef TauJetSelectorForHLTTrackSeeding_H
#define TauJetSelectorForHLTTrackSeeding_H

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class TauJetSelectorForHLTTrackSeeding : public edm::EDProducer {

public:
  explicit TauJetSelectorForHLTTrackSeeding(const edm::ParameterSet&);
  ~TauJetSelectorForHLTTrackSeeding();

private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
      
  virtual void beginRun(edm::Run&, edm::EventSetup const&);
  virtual void endRun(edm::Run&, edm::EventSetup const&);

  // ----------member data ---------------------------

  edm::InputTag inputTrackJetTag_;
  edm::InputTag inputCaloJetTag_;
  edm::InputTag inputTrackTag_;
  double ptMinCaloJet_;
  double etaMinCaloJet_;
  double etaMaxCaloJet_;
  double tauConeSize_;
  double isolationConeSize_;
  double fractionMinCaloInTauCone_;
  double fractionMaxChargedPUInCaloCone_;
  double ptTrkMaxInCaloCone_;
  int nTrkMaxInCaloCone_;
};
#endif

