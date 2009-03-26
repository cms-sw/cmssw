//
// $Id: PATPhotonProducer.h,v 1.15 2009/03/26 05:02:42 hegner Exp $
//

#ifndef PhysicsTools_PatAlgos_PATPhotonProducer_h
#define PhysicsTools_PatAlgos_PATPhotonProducer_h

/**
  \class    pat::PATPhotonProducer PATPhotonProducer.h "PhysicsTools/PatAlgos/interface/PATPhotonProducer.h"
  \brief    Produces the pat::Photon

   The PATPhotonProducer produces the analysis-level pat::Photon starting from
   a collection of objects of PhotonType.

  \author   Steven Lowette
  \version  $Id: PATPhotonProducer.h,v 1.15 2009/03/26 05:02:42 hegner Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/Utils/interface/EtComparator.h"

#include "DataFormats/PatCandidates/interface/Photon.h"

#include "PhysicsTools/PatAlgos/interface/MultiIsolator.h"
#include "PhysicsTools/PatAlgos/interface/EfficiencyLoader.h"


#include "DataFormats/PatCandidates/interface/UserData.h"
#include "PhysicsTools/PatAlgos/interface/PATUserDataHelper.h"

namespace pat {

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
      std::vector<edm::InputTag> genMatchSrc_;

      bool addTrigMatch_;
      std::vector<edm::InputTag> trigMatchSrc_;

      // tools
      GreaterByEt<Photon> eTComparator_;

      pat::helper::MultiIsolator isolator_; 
      pat::helper::MultiIsolator::IsolationValuePairs isolatorTmpStorage_; // better here than recreate at each event
      std::vector<std::pair<pat::IsolationKeys,edm::InputTag> > isoDepositLabels_;

      bool addEfficiencies_;
      pat::helper::EfficiencyLoader efficiencyLoader_;

      bool          addPhotonID_;
      typedef std::pair<std::string, edm::InputTag> NameTag;
      std::vector<NameTag> photIDSrcs_;

      bool useUserData_;
      pat::PATUserDataHelper<pat::Photon>      userDataHelper_;

  };


}

#endif
