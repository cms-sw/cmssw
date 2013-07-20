//
// $Id: PATJetProducer.h,v 1.27 2013/02/27 23:26:56 wmtan Exp $
//

#ifndef PhysicsTools_PatAlgos_PATJetProducer_h
#define PhysicsTools_PatAlgos_PATJetProducer_h

/**
  \class    pat::PATJetProducer PATJetProducer.h "PhysicsTools/PatAlgos/interface/PATJetProducer.h"
  \brief    Produces pat::Jet's

   The PATJetProducer produces analysis-level pat::Jet's starting from
   a collection of objects of JetType.

  \author   Steven Lowette, Jeremy Andrea
  \version  $Id: PATJetProducer.h,v 1.27 2013/02/27 23:26:56 wmtan Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"

#include "CommonTools/Utils/interface/PtComparator.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

#include "PhysicsTools/PatAlgos/interface/EfficiencyLoader.h"
#include "PhysicsTools/PatAlgos/interface/KinResolutionsLoader.h"

#include "RecoJets/JetProducers/interface/JetIDHelper.h"

#include "DataFormats/PatCandidates/interface/UserData.h"
#include "PhysicsTools/PatAlgos/interface/PATUserDataHelper.h"

class JetFlavourIdentifier;


namespace pat {

  class PATJetProducer : public edm::EDProducer {

    public:

      explicit PATJetProducer(const edm::ParameterSet & iConfig);
      ~PATJetProducer();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup& iSetup) override;

      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

    private:

      // configurables
      edm::InputTag            jetsSrc_;
      bool                     embedCaloTowers_;
      bool                     embedPFCandidates_;
      bool                     getJetMCFlavour_;
      edm::InputTag            jetPartonMapSource_;
      bool                     addGenPartonMatch_;
      bool                     embedGenPartonMatch_;
      edm::InputTag            genPartonSrc_;
      bool                     addGenJetMatch_;
      bool                     embedGenJetMatch_;
      edm::InputTag            genJetSrc_;
      bool                     addPartonJetMatch_;
      edm::InputTag            partonJetSrc_;
      bool                     addJetCorrFactors_;
      std::vector<edm::InputTag> jetCorrFactorsSrc_;

      bool                       addBTagInfo_;
      bool                       addDiscriminators_; 
      std::vector<edm::InputTag> discriminatorTags_;
      std::vector<std::string>   discriminatorLabels_;
      bool                       addTagInfos_; 
      std::vector<edm::InputTag> tagInfoTags_;
      std::vector<std::string>   tagInfoLabels_;
      bool                       addAssociatedTracks_;
      edm::InputTag              trackAssociation_;
      bool                       addJetCharge_;
      edm::InputTag              jetCharge_;
      bool                       addJetID_;
      edm::InputTag              jetIDMapLabel_;
      // tools
      GreaterByPt<Jet>                   pTComparator_;
      GreaterByPt<CaloTower>             caloPTComparator_;

      bool addEfficiencies_;
      pat::helper::EfficiencyLoader efficiencyLoader_;

      bool                     addResolutions_;
      pat::helper::KinResolutionsLoader resolutionLoader_;

      bool useUserData_;
      pat::PATUserDataHelper<pat::Jet>      userDataHelper_;

      

  };


}

#endif
