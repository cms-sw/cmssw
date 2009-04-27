#ifndef RecoTauTag_RecoTau_PFRecoTauCorrectedInvariantMassProducer_H_
#define RecoTauTag_RecoTau_PFRecoTauCorrectedInvariantMassProducer_H_

/* 
 * class PFRecoTauCorrectedInvariantMassProducer
 * created : April 16, 2009
 * revised : ,
 * Authors : Evan K. Friis, (UC Davis), Simone Gennai (SNS)
 *
 * Associates the invariant mass reconstruced in the PFTauDecayMode production 
 * to its underlying PFTau, in PFTau discriminator format.
 *
 * The following corrections are applied in the PFTauDecayMode production
 *
 * 1) UE filtering
 * 2) Merging of PFGammas into candidate pi zeros
 * 3) Application of mass hypothesis (charged/neutral pion) to constituents
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDecayMode.h"
#include "DataFormats/TauReco/interface/PFTauDecayModeAssociation.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/TrackReco/interface/Track.h"

using namespace std; 
using namespace edm;
using namespace reco;

class PFRecoTauCorrectedInvariantMassProducer : public EDProducer {
 public:
  explicit PFRecoTauCorrectedInvariantMassProducer(const ParameterSet& iConfig){   
    PFTauProducer_              = iConfig.getParameter<InputTag>("PFTauProducer");
    PFTauDecayModeProducer_     = iConfig.getParameter<InputTag>("PFTauDecayModeProducer");
    produces<PFTauDiscriminator>();
  }
  ~PFRecoTauCorrectedInvariantMassProducer(){} 
  virtual void produce(Event&, const EventSetup&);
 private:
  InputTag PFTauProducer_;
  InputTag PFTauDecayModeProducer_;
};
#endif
