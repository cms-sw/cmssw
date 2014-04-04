#ifndef PFELECTRONMAKER_H
#define PFELECTRONMAKER_H

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DQM/PhysicsHWW/interface/HWW.h"

class PFElectronMaker {

  public:

    PFElectronMaker(const edm::ParameterSet&, edm::ConsumesCollector);
    void SetVars(HWW&, const edm::Event&, const edm::EventSetup&);

  private:

    edm::EDGetTokenT<edm::ValueMap<reco::PFCandidatePtr> >    PFElectrons_;

};

#endif
