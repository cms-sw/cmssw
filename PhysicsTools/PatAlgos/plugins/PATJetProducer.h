//
//

#ifndef PhysicsTools_PatAlgos_PATJetProducer_h
#define PhysicsTools_PatAlgos_PATJetProducer_h

/**
  \class    pat::PATJetProducer PATJetProducer.h "PhysicsTools/PatAlgos/interface/PATJetProducer.h"
  \brief    Produces pat::Jet's

   The PATJetProducer produces analysis-level pat::Jet's starting from
   a collection of objects of JetType.

  \author   Steven Lowette, Jeremy Andrea
  \version  $Id: PATJetProducer.h,v 1.26 2010/08/09 18:13:54 srappocc Exp $
*/


#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"

#include "SimDataFormats/JetMatching/interface/JetFlavourMatching.h"
#include "SimDataFormats/JetMatching/interface/JetFlavourInfoMatching.h"

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

  class PATJetProducer : public edm::stream::EDProducer<> {

    public:

      explicit PATJetProducer(const edm::ParameterSet & iConfig);
      ~PATJetProducer();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup& iSetup) override;

      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

    private:

      // configurables
      edm::EDGetTokenT<edm::View<reco::Jet> > jetsToken_;
      bool                     embedCaloTowers_;
      bool                     embedPFCandidates_;
      bool                     getJetMCFlavour_;
      bool                     useLegacyJetMCFlavour_;
      bool                     addJetFlavourInfo_;
      edm::EDGetTokenT<reco::JetFlavourMatchingCollection> jetPartonMapToken_;
      edm::EDGetTokenT<reco::JetFlavourInfoMatchingCollection> jetFlavourInfoToken_;
      bool                     addGenPartonMatch_;
      bool                     embedGenPartonMatch_;
      edm::EDGetTokenT<edm::Association<reco::GenParticleCollection> > genPartonToken_;
      bool                     addGenJetMatch_;
      bool                     embedGenJetMatch_;
      edm::EDGetTokenT<edm::Association<reco::GenJetCollection> > genJetToken_;
      bool                     addPartonJetMatch_;
//       edm::EDGetTokenT<edm::View<reco::SomePartonJetType> > partonJetToken_;
      bool                     addJetCorrFactors_;
      std::vector<edm::EDGetTokenT<edm::ValueMap<JetCorrFactors> > > jetCorrFactorsTokens_;

      bool                       addBTagInfo_;
      bool                       addDiscriminators_;
      std::vector<edm::InputTag> discriminatorTags_;
      std::vector<edm::EDGetTokenT<reco::JetFloatAssociation::Container> > discriminatorTokens_;
      std::vector<std::string>   discriminatorLabels_;
      bool                       addTagInfos_;
      std::vector<edm::InputTag> tagInfoTags_;
      std::vector<edm::EDGetTokenT<edm::View<reco::BaseTagInfo> > > tagInfoTokens_;
      std::vector<std::string>   tagInfoLabels_;
      bool                       addAssociatedTracks_;
      edm::EDGetTokenT<reco::JetTracksAssociation::Container> trackAssociationToken_;
      bool                       addJetCharge_;
      edm::EDGetTokenT<reco::JetFloatAssociation::Container> jetChargeToken_;
      bool                       addJetID_;
      edm::EDGetTokenT<reco::JetIDValueMap> jetIDMapToken_;
      // tools
      GreaterByPt<Jet>                   pTComparator_;
      GreaterByPt<CaloTower>             caloPTComparator_;

      bool addEfficiencies_;
      pat::helper::EfficiencyLoader efficiencyLoader_;

      bool                     addResolutions_;
      pat::helper::KinResolutionsLoader resolutionLoader_;

      bool useUserData_;
      pat::PATUserDataHelper<pat::Jet>      userDataHelper_;
      //
      bool printWarning_; // this is introduced to issue warnings only once per job



  };


}

#endif
