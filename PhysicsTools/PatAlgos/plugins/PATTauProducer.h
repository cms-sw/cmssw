//
//

#ifndef PhysicsTools_PatAlgos_PATTauGenericProducer_h
#define PhysicsTools_PatAlgos_PATTauGenericProducer_h

/**
  \class    pat::PATTauGenericProducer PATTauGenericProducer.h "PhysicsTools/PatAlgos/interface/PATTauGenericProducer.h"
  \brief    Produces pat::Tau's

   The PATTauGenericProducer produces analysis-level pat::Tau's starting from
   a collection of objects of TauType.

  \author   Steven Lowette, Christophe Delaere
*/


#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/Handle.h"

#include "CommonTools/Utils/interface/PtComparator.h"

#include "PhysicsTools/PatAlgos/interface/MultiIsolator.h"
#include "PhysicsTools/PatAlgos/interface/EfficiencyLoader.h"
#include "PhysicsTools/PatAlgos/interface/KinResolutionsLoader.h"

#include "DataFormats/PatCandidates/interface/Tau.h"

#include "DataFormats/PatCandidates/interface/UserData.h"
#include "PhysicsTools/PatAlgos/interface/PATUserDataHelper.h"

#include "DataFormats/TauReco/interface/CaloTauDiscriminator.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameterFwd.h"
#include "PhysicsTools/PatAlgos/interface/PATUserDataMerger.h"

#include <string>

typedef edm::AssociationVector<reco::PFTauRefProd, std::vector<reco::PFTauTransverseImpactParameterRef> > PFTauTIPAssociationByRef;
namespace pat {
  template<class TauType>
  class PATTauGenericProducer : public edm::stream::EDProducer<> {

    public:

      explicit PATTauGenericProducer(const edm::ParameterSet & iConfig);
      ~PATTauGenericProducer() override;

      void produce(edm::Event & iEvent, const edm::EventSetup& iSetup) override;

      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

    private:
      bool firstOccurence_; // used to print LogWarnings only at first occurnece in the event loop

      // configurables
      edm::EDGetTokenT<edm::View<reco::BaseTau> > baseTauToken_;
      edm::EDGetTokenT<edm::AssociationVector<edm::RefProd<std::vector<TauType> >, std::vector<reco::PFTauTransverseImpactParameterRef> >> tauTransverseImpactParameterToken_;
      edm::EDGetTokenT<std::vector<TauType> > pfTauToken_;
      edm::EDGetTokenT<reco::CaloTauCollection> caloTauToken_;
      edm::InputTag tauTransverseImpactParameterSrc_;
      bool embedIsolationTracks_;
      bool embedLeadTrack_;
      bool embedSignalTracks_;
      bool embedLeadPFCand_;
      bool embedLeadPFChargedHadrCand_;
      bool embedLeadPFNeutralCand_;
      bool embedSignalPFCands_;
      bool embedSignalPFChargedHadrCands_;
      bool embedSignalPFNeutralHadrCands_;
      bool embedSignalPFGammaCands_;
      bool embedIsolationPFCands_;
      bool embedIsolationPFChargedHadrCands_;
      bool embedIsolationPFNeutralHadrCands_;
      bool embedIsolationPFGammaCands_;

      bool          addGenMatch_;
      bool          embedGenMatch_;
      std::vector<edm::EDGetTokenT<edm::Association<reco::GenParticleCollection> > > genMatchTokens_;

      bool          addGenJetMatch_;
      bool          embedGenJetMatch_;
      edm::EDGetTokenT<edm::Association<reco::GenJetCollection> > genJetMatchToken_;

      bool          addTauJetCorrFactors_;
      std::vector<edm::EDGetTokenT<edm::ValueMap<TauJetCorrFactors> > > tauJetCorrFactorsTokens_;

      bool          addTauID_;
      typedef std::pair<std::string, edm::InputTag> NameTag;
      std::vector<NameTag> tauIDSrcs_;
      std::vector<edm::EDGetTokenT<reco::CaloTauDiscriminator> > caloTauIDTokens_;
      std::vector<edm::EDGetTokenT<typename TauType::TauDiscriminator> > pfTauIDTokens_;
      bool          skipMissingTauID_;

      // tools
      GreaterByPt<Tau>       pTTauComparator_;

      pat::helper::MultiIsolator isolator_;
      pat::helper::MultiIsolator::IsolationValuePairs isolatorTmpStorage_; // better here than recreate at each event
      std::vector<std::pair<pat::IsolationKeys,edm::InputTag> > isoDepositLabels_;
      std::vector<edm::EDGetTokenT<edm::ValueMap<IsoDeposit> > > isoDepositTokens_;

      bool addEfficiencies_;
      pat::helper::EfficiencyLoader efficiencyLoader_;

      bool addResolutions_;
      pat::helper::KinResolutionsLoader resolutionLoader_;

      bool useUserData_;
      pat::PATUserDataHelper<pat::Tau>      userDataHelper_;

      template <typename TauCollectionType, typename TauDiscrType> float getTauIdDiscriminator(const edm::Handle<TauCollectionType>&, size_t, const edm::Handle<TauDiscrType>&);
  };

}

#endif
