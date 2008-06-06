#ifndef PhysicsTools_PatAlgos_PATGenericParticleCleaner_h
#define PhysicsTools_PatAlgos_PATGenericParticleCleaner_h
//
// $Id: PATGenericParticleCleaner.h,v 1.2 2008/06/05 20:05:13 gpetrucc Exp $
//

/**
  \class    PATGenericParticleCleaner PATGenericParticleCleaner.h "PhysicsTools/PatAlgos/interface/PATGenericParticleCleaner.h"
  \brief    Produces pat::GenericParticle's

   The PATGenericParticleCleaner produces analysis-level pat::GenericParticle's starting from
   a collection of objects of GenericParticleType.

  \author   Steven Lowette, Jeremy Andrea
  \version  $Id: PATGenericParticleCleaner.h,v 1.2 2008/06/05 20:05:13 gpetrucc Exp $
*/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "PhysicsTools/PatAlgos/plugins/CleanerHelper.h"
#include "PhysicsTools/PatAlgos/interface/MultiIsolator.h"
#include "PhysicsTools/PatAlgos/interface/OverlapHelper.h"

#include <DataFormats/Candidate/interface/CandidateFwd.h>
#include <DataFormats/Candidate/interface/Candidate.h>

#include "PhysicsTools/Utilities/interface/EtComparator.h"

namespace pat {

  class PATGenericParticleCleaner : public edm::EDProducer {
    public:
      explicit PATGenericParticleCleaner(const edm::ParameterSet & iConfig);
      ~PATGenericParticleCleaner();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);
      virtual void endJob();

    private:
      // configurables
      edm::InputTag               src_;
        
      // helper
      typedef pat::helper::CleanerHelper<reco::Candidate, // need to read CandidateCollection
                                 reco::Candidate,
                                 edm::OwnVector<reco::Candidate>, 
                                 GreaterByEt<reco::Candidate> > MyCleanerHelper;
      MyCleanerHelper helper_;

      // Isolation    
      pat::helper::MultiIsolator isolator_;

      // deltaR overlap helper
      pat::helper::OverlapHelper overlapHelper_;

  }; // class

} // namespace

#endif
