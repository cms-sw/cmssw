#ifndef PFJETMAKER_H
#define PFJETMAKER_H

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/JetFloatAssociation.h"
#include "DQM/PhysicsHWW/interface/HWW.h"

class PFJetMaker {

  public:

    PFJetMaker(const edm::ParameterSet&, edm::ConsumesCollector);
    void SetVars(HWW&, const edm::Event&, const edm::EventSetup&);

  private:

    edm::EDGetTokenT<edm::View<reco::Jet> > PFJetCollection_;
    edm::EDGetTokenT<reco::JetFloatAssociation::Container> BJetTags_;

};

#endif
