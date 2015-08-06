//
//

#ifndef PhysicsTools_PatAlgos_PATCompositeCandidateProducer_h
#define PhysicsTools_PatAlgos_PATCompositeCandidateProducer_h

/**
  \class    pat::PATCompositeCandidateProducer PATCompositeCandidateProducer.h "PhysicsTools/PatAlgos/interface/PATCompositeCandidateProducer.h"
  \brief    Produces the pat::CompositeCandidate

   The PATCompositeCandidateProducer produces the analysis-level pat::CompositeCandidate starting from
   any collection of Candidates

  \author   Salvatore Rappoccio
  \version  $Id: PATCompositeCandidateProducer.h,v 1.3 2009/06/25 23:49:35 gpetrucc Exp $
*/


#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "CommonTools/Utils/interface/EtComparator.h"

#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"

#include "PhysicsTools/PatAlgos/interface/PATUserDataHelper.h"
#include "PhysicsTools/PatAlgos/interface/MultiIsolator.h"
#include "PhysicsTools/PatAlgos/interface/EfficiencyLoader.h"
#include "PhysicsTools/PatAlgos/interface/KinResolutionsLoader.h"
#include "PhysicsTools/PatAlgos/interface/VertexingHelper.h"

namespace pat {

  class PATCompositeCandidateProducer : public edm::stream::EDProducer<> {

    public:

      explicit PATCompositeCandidateProducer(const edm::ParameterSet & iConfig);
      ~PATCompositeCandidateProducer();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup& iSetup) override;

    private:

      // configurables
      const edm::EDGetTokenT<edm::View<reco::CompositeCandidate> > srcToken_;     // list of reco::CompositeCandidates

      const bool useUserData_;
      pat::PATUserDataHelper<pat::CompositeCandidate> userDataHelper_;

      const bool addEfficiencies_;
      pat::helper::EfficiencyLoader efficiencyLoader_;

      const bool addResolutions_;
      pat::helper::KinResolutionsLoader resolutionLoader_;
  };


}

#endif
