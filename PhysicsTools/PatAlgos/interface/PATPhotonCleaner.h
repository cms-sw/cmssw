//
// $Id: PATPhotonCleaner.h,v 1.1 2008/01/17 02:50:11 gpetrucc Exp $
//

#ifndef PhysicsTools_PatAlgos_PATPhotonCleaner_h
#define PhysicsTools_PatAlgos_PATPhotonCleaner_h

/**
  \class    PATPhotonCleaner PATPhotonCleaner.h "PhysicsTools/PatAlgos/interface/PATPhotonCleaner.h"
  \brief    Produces pat::Photon's

   The PATPhotonCleaner produces analysis-level pat::Photon's starting from
   a collection of objects of PhotonType.

  \author   Steven Lowette, Jeremy Andrea
  \version  $Id: PATPhotonCleaner.h,v 1.1 2008/01/17 02:50:11 gpetrucc Exp $
*/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "PhysicsTools/PatAlgos/interface/CleanerHelper.h"

#include <DataFormats/EgammaCandidates/interface/Photon.h>
#include <DataFormats/EgammaCandidates/interface/ConvertedPhoton.h>

#include "PhysicsTools/PatUtils/interface/DuplicatedPhotonRemover.h"
#include "PhysicsTools/Utilities/interface/EtComparator.h"

namespace pat {

  template<typename PhotonIn, typename PhotonOut>
  class PATPhotonCleaner : public edm::EDProducer {
    public:
      enum RemovalAlgo { None, BySeed, BySuperCluster };
    
      explicit PATPhotonCleaner(const edm::ParameterSet & iConfig);
      ~PATPhotonCleaner();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

    private:
      // configurables
      edm::InputTag               photonSrc_;
      RemovalAlgo                 removeDuplicates_;
      RemovalAlgo                 removeElectrons_;
      std::vector<edm::InputTag>  electronsToCheck_;
        
      // helper
      pat::helper::CleanerHelper<PhotonIn,
                                 PhotonOut,
                                 std::vector<PhotonOut>,
                                 GreaterByEt<PhotonOut> > helper_;
      // duplicate removal algo
      pat::DuplicatedPhotonRemover remover_;

      static RemovalAlgo fromString(const edm::ParameterSet & iConfig, const std::string &name);
      void removeDuplicates() ;
      void removeElectrons(const edm::Event &iEvent) ;
  };

  // now I'm typedeffing eveything, but I don't think we really need all them
  typedef PATPhotonCleaner<reco::Photon,reco::Photon>                   PATBasePhotonCleaner;
  typedef PATPhotonCleaner<reco::ConvertedPhoton,reco::ConvertedPhoton> PATConvertedPhotonCleaner;

}

#endif
