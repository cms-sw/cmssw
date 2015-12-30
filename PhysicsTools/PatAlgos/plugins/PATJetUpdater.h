//
//

#ifndef PhysicsTools_PatAlgos_PATJetUpdater_h
#define PhysicsTools_PatAlgos_PATJetUpdater_h

/**
  \class    pat::PATJetUpdater PATJetUpdater.h "PhysicsTools/PatAlgos/interface/PATJetUpdater.h"
  \brief    Produces pat::Jet's

   The PATJetUpdater produces analysis-level pat::Jet's starting from
   a collection of pat::Jet's and updates information.

  \author   Andreas Hinzmann
  \version  $Id: PATJetUpdater.h,v 1.00 2014/03/11 18:13:54 srappocc Exp $
*/


#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"

#include "CommonTools/Utils/interface/PtComparator.h"

#include "DataFormats/PatCandidates/interface/Jet.h"

#include "DataFormats/PatCandidates/interface/UserData.h"
#include "PhysicsTools/PatAlgos/interface/PATUserDataHelper.h"

namespace pat {

  class PATJetUpdater : public edm::stream::EDProducer<> {

    public:

      explicit PATJetUpdater(const edm::ParameterSet & iConfig);
      ~PATJetUpdater();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup& iSetup) override;

      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

    private:

      // configurables
      edm::EDGetTokenT<edm::View<reco::Jet> > jetsToken_;
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

      GreaterByPt<Jet> pTComparator_;

      bool useUserData_;
      pat::PATUserDataHelper<pat::Jet>      userDataHelper_;
      //
      bool printWarning_; // this is introduced to issue warnings only once per job

  };


}

#endif
