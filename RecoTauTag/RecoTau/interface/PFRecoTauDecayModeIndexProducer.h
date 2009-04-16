#ifndef RecoTauTag_RecoTau_PFRecoTauDecayModeIndexProducer_H_
#define RecoTauTag_RecoTau_PFRecoTauDecayModeIndexProducer_H_

/* 
 * class PFRecoTauDecayModeIndexProducer
 * created : April 16, 2009
 * revised : ,
 * Authors : Evan K. Friis, (UC Davis), Simone Gennai (SNS)
 *
 * Associates the decay mode index (see enum in DataFormats/TauReco/interface/PFTauDecayMode.h)
 * reconstruced in the PFTauDecayMode production to its underlying PFTau, in PFTau discriminator format
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

class PFRecoTauDecayModeIndexProducer : public EDProducer {
 public:
  explicit PFRecoTauDecayModeIndexProducer(const ParameterSet& iConfig){   
    PFTauProducer_              = iConfig.getParameter<InputTag>("PFTauProducer");
    PFTauDecayModeProducer_     = iConfig.getParameter<InputTag>("PFTauDecayModeProducer");
    produces<PFTauDiscriminator>();
  }
  ~PFRecoTauDecayModeIndexProducer(){} 
  virtual void produce(Event&, const EventSetup&);
 private:
  InputTag PFTauProducer_;
  InputTag PFTauDecayModeProducer_;
};
#endif
