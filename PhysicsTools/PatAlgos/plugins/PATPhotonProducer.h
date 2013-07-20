//
// $Id: PATPhotonProducer.h,v 1.20 2013/02/27 23:26:56 wmtan Exp $
//

#ifndef PhysicsTools_PatAlgos_PATPhotonProducer_h
#define PhysicsTools_PatAlgos_PATPhotonProducer_h

/**
  \class    pat::PATPhotonProducer PATPhotonProducer.h "PhysicsTools/PatAlgos/interface/PATPhotonProducer.h"
  \brief    Produces the pat::Photon

   The PATPhotonProducer produces the analysis-level pat::Photon starting from
   a collection of objects of PhotonType.

  \author   Steven Lowette
  \version  $Id: PATPhotonProducer.h,v 1.20 2013/02/27 23:26:56 wmtan Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/Utils/interface/EtComparator.h"

#include "DataFormats/PatCandidates/interface/Photon.h"

#include "PhysicsTools/PatAlgos/interface/MultiIsolator.h"
#include "PhysicsTools/PatAlgos/interface/EfficiencyLoader.h"
#include "PhysicsTools/PatAlgos/interface/KinResolutionsLoader.h"


#include "DataFormats/PatCandidates/interface/UserData.h"
#include "PhysicsTools/PatAlgos/interface/PATUserDataHelper.h"

namespace pat {

  class PATPhotonProducer : public edm::EDProducer {

    public:

      explicit PATPhotonProducer(const edm::ParameterSet & iConfig);
      ~PATPhotonProducer();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup& iSetup) override;

      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

    private:

      // configurables
      edm::InputTag photonSrc_;
      bool embedSuperCluster_;

      bool addGenMatch_;
      bool embedGenMatch_;
      std::vector<edm::InputTag> genMatchSrc_;

      // tools
      GreaterByEt<Photon> eTComparator_;

      pat::helper::MultiIsolator isolator_; 
      pat::helper::MultiIsolator::IsolationValuePairs isolatorTmpStorage_; // better here than recreate at each event
      std::vector<std::pair<pat::IsolationKeys,edm::InputTag> > isoDepositLabels_;

      bool addEfficiencies_;
      pat::helper::EfficiencyLoader efficiencyLoader_;
      
      bool addResolutions_;
      pat::helper::KinResolutionsLoader resolutionLoader_;

      bool          addPhotonID_;
      typedef std::pair<std::string, edm::InputTag> NameTag;
      std::vector<NameTag> photIDSrcs_;

      bool useUserData_;
      pat::PATUserDataHelper<pat::Photon>      userDataHelper_;

  };


}

#endif
