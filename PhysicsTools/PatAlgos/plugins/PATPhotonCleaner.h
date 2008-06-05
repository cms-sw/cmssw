//
// $Id: PATPhotonCleaner.h,v 1.5 2008/06/05 17:23:25 gpetrucc Exp $
//

#ifndef PhysicsTools_PatAlgos_PATPhotonCleaner_h
#define PhysicsTools_PatAlgos_PATPhotonCleaner_h

/**
  \class    pat::PATPhotonCleaner PATPhotonCleaner.h "PhysicsTools/PatAlgos/interface/PATPhotonCleaner.h"
  \brief    Produces pat::Photon's

   The PATPhotonCleaner produces analysis-level pat::Photon's starting from
   a collection of objects of PhotonType.

  \author   Steven Lowette, Jeremy Andrea
  \version  $Id: PATPhotonCleaner.h,v 1.5 2008/06/05 17:23:25 gpetrucc Exp $
*/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "PhysicsTools/PatAlgos/plugins/CleanerHelper.h"
#include "PhysicsTools/PatAlgos/interface/MultiIsolator.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "PhysicsTools/PatUtils/interface/DuplicatedPhotonRemover.h"
#include "PhysicsTools/Utilities/interface/EtComparator.h"

namespace pat {

  class PATPhotonCleaner : public edm::EDProducer {
    public:
      enum RemovalAlgo { None, BySeed, BySuperCluster };
    
      explicit PATPhotonCleaner(const edm::ParameterSet & iConfig);
      ~PATPhotonCleaner();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);
      virtual void endJob();

    private:
      // configurables
      edm::InputTag               photonSrc_;
      RemovalAlgo                 removeDuplicates_;
      RemovalAlgo                 removeElectrons_;
      std::vector<edm::InputTag>  electronsToCheck_;
        
      // helpers
      pat::helper::CleanerHelper<reco::Photon,
                                 reco::Photon,
                                 std::vector<reco::Photon>,
                                 GreaterByEt<reco::Photon> > helper_;
      pat::helper::MultiIsolator isolator_;

      // duplicate removal algo
      pat::DuplicatedPhotonRemover remover_;

      static RemovalAlgo fromString(const edm::ParameterSet & iConfig, const std::string &name);
      void removeDuplicates() ;
      void removeElectrons(const edm::Event &iEvent) ;
  };

}

#endif
