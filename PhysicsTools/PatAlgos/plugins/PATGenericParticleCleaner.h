//
// $Id: PATGenericParticleCleaner.h,v 1.1.2.4 2008/03/14 15:13:51 gpetrucc Exp $
//

#ifndef PhysicsTools_PatAlgos_PATGenericParticleCleaner_h
#define PhysicsTools_PatAlgos_PATGenericParticleCleaner_h

/**
  \class    PATGenericParticleCleaner PATGenericParticleCleaner.h "PhysicsTools/PatAlgos/interface/PATGenericParticleCleaner.h"
  \brief    Produces pat::GenericParticle's

   The PATGenericParticleCleaner produces analysis-level pat::GenericParticle's starting from
   a collection of objects of GenericParticleType.

  \author   Steven Lowette, Jeremy Andrea
  \version  $Id: PATGenericParticleCleaner.h,v 1.1.2.4 2008/03/14 15:13:51 gpetrucc Exp $
*/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "PhysicsTools/PatAlgos/plugins/CleanerHelper.h"
#include "PhysicsTools/PatAlgos/interface/MultiIsolator.h"
#include "PhysicsTools/PatAlgos/interface/OverlapHelper.h"

#include <DataFormats/Candidate/interface/Candidate.h>
//#include <DataFormats/RecoCandidate/interface/RecoCandidate.h>

#include "PhysicsTools/Utilities/interface/EtComparator.h"

namespace pat {

  template<typename GP>
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
      typedef typename pat::helper::CleanerHelper<reco::Candidate, // need to read CandidateCollection
                                 GP,
                                 std::vector<GP>, // but want to write concrete collection
                                 GreaterByEt<GP> > MyCleanerHelper;
      MyCleanerHelper helper_;

      // Isolation    
      pat::helper::MultiIsolator isolator_;

      // deltaR overlap helper
      pat::helper::OverlapHelper overlapHelper_;

  }; // class

} // namespace

#endif
