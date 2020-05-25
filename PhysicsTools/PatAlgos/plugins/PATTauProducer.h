//
//

#ifndef PhysicsTools_PatAlgos_PATTauProducer_h
#define PhysicsTools_PatAlgos_PATTauProducer_h

/**
  \class    pat::PATTauProducer PATTauProducer.h "PhysicsTools/PatAlgos/interface/PATTauProducer.h"
  \brief    Produces pat::Tau's

   The PATTauProducer produces analysis-level pat::Tau's starting from
   a collection of objects of TauType.

  \author   Steven Lowette, Christophe Delaere
*/

#include "FWCore/Common/interface/Provenance.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"

#include "CommonTools/Utils/interface/PtComparator.h"

#include "PhysicsTools/PatAlgos/interface/MultiIsolator.h"
#include "PhysicsTools/PatAlgos/interface/EfficiencyLoader.h"
#include "PhysicsTools/PatAlgos/interface/KinResolutionsLoader.h"

#include "DataFormats/PatCandidates/interface/Tau.h"

#include "DataFormats/PatCandidates/interface/UserData.h"
#include "PhysicsTools/PatAlgos/interface/PATUserDataHelper.h"

#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/TauReco/interface/TauDiscriminatorContainer.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameterFwd.h"
#include "PhysicsTools/PatAlgos/interface/PATUserDataMerger.h"

#include "RecoTauTag/RecoTau/interface/PositionAtECalEntranceComputer.h"

#include <string>

typedef edm::AssociationVector<reco::PFTauRefProd, std::vector<reco::PFTauTransverseImpactParameterRef> >
    PFTauTIPAssociationByRef;
namespace pat {

  class PATTauProducer : public edm::stream::EDProducer<> {
  public:
    explicit PATTauProducer(const edm::ParameterSet& iConfig);
    ~PATTauProducer() override;

    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    bool firstOccurence_;  // used to print LogWarnings only at first occurnece in the event loop

    // configurables
    edm::EDGetTokenT<edm::View<reco::BaseTau> > baseTauToken_;
    edm::EDGetTokenT<PFTauTIPAssociationByRef> tauTransverseImpactParameterToken_;
    edm::EDGetTokenT<reco::PFTauCollection> pfTauToken_;
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

    bool addGenMatch_;
    bool embedGenMatch_;
    std::vector<edm::EDGetTokenT<edm::Association<reco::GenParticleCollection> > > genMatchTokens_;

    bool addGenJetMatch_;
    bool embedGenJetMatch_;
    edm::EDGetTokenT<edm::Association<reco::GenJetCollection> > genJetMatchToken_;

    bool addTauJetCorrFactors_;
    std::vector<edm::EDGetTokenT<edm::ValueMap<TauJetCorrFactors> > > tauJetCorrFactorsTokens_;

    bool addTauID_;
    typedef std::pair<std::string, edm::InputTag> NameTag;
    typedef std::pair<std::string, std::string> WPCfg;
    typedef std::pair<WPCfg, int> WPIdx;
    typedef std::pair<std::string, WPIdx> NameWPIdx;
    typedef std::pair<edm::InputTag, std::vector<NameWPIdx> >
        IDContainerData;  //to save input module tag and corresponding pairs <working point name for the output tree, WP index in the input ID container>
    std::vector<NameTag> tauIDSrcs_;
    std::vector<std::vector<NameWPIdx> > tauIDSrcContainers_;
    std::vector<edm::EDGetTokenT<reco::PFTauDiscriminator> > pfTauIDTokens_;
    std::vector<edm::EDGetTokenT<reco::TauDiscriminatorContainer> > pfTauIDContainerTokens_;
    bool skipMissingTauID_;
    edm::ProcessHistoryID phID_;
    // tools
    GreaterByPt<Tau> pTTauComparator_;

    pat::helper::MultiIsolator isolator_;
    pat::helper::MultiIsolator::IsolationValuePairs isolatorTmpStorage_;  // better here than recreate at each event
    std::vector<std::pair<pat::IsolationKeys, edm::InputTag> > isoDepositLabels_;
    std::vector<edm::EDGetTokenT<edm::ValueMap<IsoDeposit> > > isoDepositTokens_;

    bool addEfficiencies_;
    pat::helper::EfficiencyLoader efficiencyLoader_;

    bool addResolutions_;
    pat::helper::KinResolutionsLoader resolutionLoader_;

    bool useUserData_;
    pat::PATUserDataHelper<pat::Tau> userDataHelper_;

    template <typename TauCollectionType, typename TauDiscrType>
    float getTauIdDiscriminator(const edm::Handle<TauCollectionType>&, size_t, const edm::Handle<TauDiscrType>&);
    float getTauIdDiscriminatorFromContainer(const edm::Handle<reco::PFTauCollection>&,
                                             size_t,
                                             const edm::Handle<reco::TauDiscriminatorContainer>&,
                                             int);

    PositionAtECalEntranceComputer posAtECalEntranceComputer_;
  };

}  // namespace pat

#endif
