//
// $Id: PATPhotonProducer.h,v 1.9 2008/06/24 22:58:24 gpetrucc Exp $
//

#ifndef PhysicsTools_PatAlgos_PATPhotonProducer_h
#define PhysicsTools_PatAlgos_PATPhotonProducer_h

/**
  \class    pat::PATPhotonProducer PATPhotonProducer.h "PhysicsTools/PatAlgos/interface/PATPhotonProducer.h"
  \brief    Produces the pat::Photon

   The PATPhotonProducer produces the analysis-level pat::Photon starting from
   a collection of objects of PhotonType.

  \author   Steven Lowette
  \version  $Id: PATPhotonProducer.h,v 1.9 2008/06/24 22:58:24 gpetrucc Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/Utilities/interface/EtComparator.h"

#include "DataFormats/PatCandidates/interface/Photon.h"

#include "PhysicsTools/PatAlgos/interface/MultiIsolator.h"
#include "PhysicsTools/PatAlgos/interface/EfficiencyLoader.h"

namespace pat {


  class ObjectResolutionCalc;


  class PATPhotonProducer : public edm::EDProducer {

    public:

      explicit PATPhotonProducer(const edm::ParameterSet & iConfig);
      ~PATPhotonProducer();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

    private:

      // configurables
      edm::InputTag photonSrc_;
      bool embedSuperCluster_;

      bool addGenMatch_;
      bool embedGenMatch_;
      edm::InputTag genMatchSrc_;

      bool addTrigMatch_;
      std::vector<edm::InputTag> trigMatchSrc_;

      bool addPhotonID_;
      edm::InputTag photonIDSrc_;

      // tools
      GreaterByEt<Photon> eTComparator_;

      pat::helper::MultiIsolator isolator_; 
      pat::helper::MultiIsolator::IsolationValuePairs isolatorTmpStorage_; // better here than recreate at each event
      std::vector<std::pair<pat::IsolationKeys,edm::InputTag> > isoDepositLabels_;

      bool addEfficiencies_;
      pat::helper::EfficiencyLoader efficiencyLoader_;


  };


}

#endif
