//
// $Id: PATCompositeCandidateProducer.h,v 1.6 2008/09/01 14:35:48 gpetrucc Exp $
//

#ifndef PhysicsTools_PatAlgos_PATCompositeCandidateProducer_h
#define PhysicsTools_PatAlgos_PATCompositeCandidateProducer_h

/**
  \class    pat::PATCompositeCandidateProducer PATCompositeCandidateProducer.h "PhysicsTools/PatAlgos/interface/PATCompositeCandidateProducer.h"
  \brief    Produces the pat::CompositeCandidate

   The PATCompositeCandidateProducer produces the analysis-level pat::CompositeCandidate starting from
   any collection of Candidates

  \author   Salvatore Rappoccio
  \version  $Id: PATCompositeCandidateProducer.h,v 1.6 2008/09/01 14:35:48 srappocc Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "PhysicsTools/Utilities/interface/EtComparator.h"

#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"

#include "PhysicsTools/PatAlgos/interface/PATUserDataHelper.h"
#include "PhysicsTools/PatAlgos/interface/MultiIsolator.h"
#include "PhysicsTools/PatAlgos/interface/EfficiencyLoader.h"
#include "PhysicsTools/PatAlgos/interface/VertexingHelper.h"

namespace pat {

  class PATCompositeCandidateProducer : public edm::EDProducer {

    public:

      explicit PATCompositeCandidateProducer(const edm::ParameterSet & iConfig);
      ~PATCompositeCandidateProducer();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

    private:

      // configurables
      edm::InputTag src_;     // list of reco::CompositeCandidates

      bool useUserData_;
      pat::PATUserDataHelper<pat::CompositeCandidate> userDataHelper_;

  };


}

#endif
