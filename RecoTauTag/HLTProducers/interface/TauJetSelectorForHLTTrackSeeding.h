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
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() ;
      
  // ----------member data ---------------------------

  const edm::InputTag inputTrackJetTag_;
  const edm::InputTag inputCaloJetTag_;
  const edm::InputTag inputTrackTag_;
  const double ptMinCaloJet_;
  const double etaMinCaloJet_;
  const double etaMaxCaloJet_;
  const double tauConeSize_;
  const double isolationConeSize_;
  const double fractionMinCaloInTauCone_;
  const double fractionMaxChargedPUInCaloCone_;
  const double ptTrkMaxInCaloCone_;
  const int nTrkMaxInCaloCone_;
};
#endif

