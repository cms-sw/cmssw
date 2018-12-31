/**
  \class    pat::PATElectronSlimmer PATElectronSlimmer.h "PhysicsTools/PatAlgos/interface/PATElectronSlimmer.h"
  \brief    Slimmer of PAT Electrons 
*/

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/PatCandidates/interface/Electron.h"

#include "PhysicsTools/PatAlgos/interface/ObjectModifier.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "FWCore/Utilities/interface/isFinite.h"

namespace pat {

  class PATElectronSlimmer : public edm::stream::EDProducer<> {
  public:
    explicit PATElectronSlimmer(const edm::ParameterSet & iConfig);
    ~PATElectronSlimmer() override { }
    
    void produce(edm::Event & iEvent, const edm::EventSetup & iSetup) final;
    void beginLuminosityBlock(const edm::LuminosityBlock&, const  edm::EventSetup&) final;
    
    private:
    const edm::EDGetTokenT<edm::View<pat::Electron> > src_;
    
    const StringCutObjectSelector<pat::Electron> dropSuperClusters_, dropBasicClusters_, dropPFlowClusters_, dropPreshowerClusters_, dropSeedCluster_, dropRecHits_;
    const StringCutObjectSelector<pat::Electron> dropCorrections_,dropIsolations_,dropShapes_,dropSaturation_,dropExtrapolations_,dropClassifications_;
    
    const edm::EDGetTokenT<edm::ValueMap<std::vector<reco::PFCandidateRef> > > reco2pf_;
    const edm::EDGetTokenT<edm::Association<pat::PackedCandidateCollection> > pf2pc_;
    const edm::EDGetTokenT<pat::PackedCandidateCollection>  pc_;
    const bool linkToPackedPF_;
    const StringCutObjectSelector<pat::Electron> saveNonZSClusterShapes_;
    const edm::EDGetTokenT<EcalRecHitCollection> reducedBarrelRecHitCollectionToken_, reducedEndcapRecHitCollectionToken_;
    const bool modifyElectron_;
    std::unique_ptr<pat::ObjectModifier<pat::Electron> > electronModifier_;
  };

} // namespace

pat::PATElectronSlimmer::PATElectronSlimmer(const edm::ParameterSet & iConfig) :
    src_(consumes<edm::View<pat::Electron> >(iConfig.getParameter<edm::InputTag>("src"))),
    dropSuperClusters_(iConfig.getParameter<std::string>("dropSuperCluster")),
    dropBasicClusters_(iConfig.getParameter<std::string>("dropBasicClusters")),
    dropPFlowClusters_(iConfig.getParameter<std::string>("dropPFlowClusters")),
    dropPreshowerClusters_(iConfig.getParameter<std::string>("dropPreshowerClusters")),
    dropSeedCluster_(iConfig.getParameter<std::string>("dropSeedCluster")),
    dropRecHits_(iConfig.getParameter<std::string>("dropRecHits")),
    dropCorrections_(iConfig.getParameter<std::string>("dropCorrections")),
    dropIsolations_(iConfig.getParameter<std::string>("dropIsolations")),
    dropShapes_(iConfig.getParameter<std::string>("dropShapes")),
    dropSaturation_(iConfig.getParameter<std::string>("dropSaturation")),
    dropExtrapolations_(iConfig.getParameter<std::string>("dropExtrapolations")),
    dropClassifications_(iConfig.getParameter<std::string>("dropClassifications")),
    reco2pf_(mayConsume<edm::ValueMap<std::vector<reco::PFCandidateRef>>>(iConfig.getParameter<edm::InputTag>("recoToPFMap"))),
    pf2pc_(mayConsume<edm::Association<pat::PackedCandidateCollection>>(iConfig.getParameter<edm::InputTag>("packedPFCandidates"))),
    pc_(mayConsume<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("packedPFCandidates"))),
    linkToPackedPF_(iConfig.getParameter<bool>("linkToPackedPFCandidates")),
    saveNonZSClusterShapes_(iConfig.getParameter<std::string>("saveNonZSClusterShapes")),
    reducedBarrelRecHitCollectionToken_(consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("reducedBarrelRecHitCollection"))),
    reducedEndcapRecHitCollectionToken_(consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("reducedEndcapRecHitCollection"))),
    modifyElectron_(iConfig.getParameter<bool>("modifyElectrons"))
{
    edm::ConsumesCollector sumes(consumesCollector());
    if( modifyElectron_ ) {
      const edm::ParameterSet& mod_config = iConfig.getParameter<edm::ParameterSet>("modifierConfig");
      electronModifier_.reset(new pat::ObjectModifier<pat::Electron>(mod_config) );
      electronModifier_->setConsumes(sumes);
    } else {
      electronModifier_.reset(nullptr);
    }

    mayConsume<EcalRecHitCollection>(edm::InputTag("reducedEcalRecHitsEB"));
    mayConsume<EcalRecHitCollection>(edm::InputTag("reducedEcalRecHitsEE"));

    produces<std::vector<pat::Electron> >();
}

void 
pat::PATElectronSlimmer::beginLuminosityBlock(const edm::LuminosityBlock&, const  edm::EventSetup& iSetup) {
}

void 
pat::PATElectronSlimmer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
    using namespace edm;
    using namespace std;

    Handle<View<pat::Electron> >      src;
    iEvent.getByToken(src_, src);

    Handle<edm::ValueMap<std::vector<reco::PFCandidateRef>>> reco2pf;
    Handle<edm::Association<pat::PackedCandidateCollection>> pf2pc;
    Handle<pat::PackedCandidateCollection> pc;
    if (linkToPackedPF_) {
        iEvent.getByToken(reco2pf_, reco2pf);
        iEvent.getByToken(pf2pc_, pf2pc);
        iEvent.getByToken(pc_, pc);
    }
    noZS::EcalClusterLazyTools lazyToolsNoZS(iEvent, iSetup, reducedBarrelRecHitCollectionToken_, reducedEndcapRecHitCollectionToken_);

    auto out = std::make_unique<std::vector<pat::Electron>>();
    out->reserve(src->size());

    if( modifyElectron_ ) { electronModifier_->setEvent(iEvent); }
    if( modifyElectron_ ) electronModifier_->setEventContent(iSetup);

    std::vector<unsigned int> keys;
    for (auto elePtr : src->ptrs() ){
        out->push_back(*elePtr);
        pat::Electron & electron = out->back();
	electron.addParentRef(elePtr);
        if( modifyElectron_ ) { electronModifier_->modify(electron); }
        if (dropSuperClusters_(electron)) { electron.superCluster_.clear(); electron.embeddedSuperCluster_ = false; }
	if (dropBasicClusters_(electron)) { electron.basicClusters_.clear();  }
	if (dropSuperClusters_(electron) || dropPFlowClusters_(electron)) { electron.pflowSuperCluster_.clear(); electron.embeddedPflowSuperCluster_ = false; }
	if (dropBasicClusters_(electron) || dropPFlowClusters_(electron)) { electron.pflowBasicClusters_.clear(); }
	if (dropPreshowerClusters_(electron)) { electron.preshowerClusters_.clear();  }
	if (dropPreshowerClusters_(electron) || dropPFlowClusters_(electron)) { electron.pflowPreshowerClusters_.clear(); }
	if (dropSeedCluster_(electron)) { electron.seedCluster_.clear(); electron.embeddedSeedCluster_ = false; }
        if (dropRecHits_(electron)) { electron.recHits_ = EcalRecHitCollection(); electron.embeddedRecHits_ = false; }
        if (dropCorrections_(electron)) { electron.setCorrections(reco::GsfElectron::Corrections()); }
        if (dropIsolations_(electron)) { electron.setDr03Isolation(reco::GsfElectron::IsolationVariables()); electron.setDr04Isolation(reco::GsfElectron::IsolationVariables()); electron.setPfIsolationVariables(reco::GsfElectron::PflowIsolationVariables()); }
        if (dropShapes_(electron)) { electron.setShowerShape(reco::GsfElectron::ShowerShape()); }
        if (dropSaturation_(electron)) { electron.setSaturationInfo(reco::GsfElectron::SaturationInfo()); }
        if (dropExtrapolations_(electron)) { electron.setTrackExtrapolations(reco::GsfElectron::TrackExtrapolations());  }
        if (dropClassifications_(electron)) { electron.setClassificationVariables(reco::GsfElectron::ClassificationVariables()); electron.setClassification(reco::GsfElectron::Classification()); }
        if (linkToPackedPF_) {
            //std::cout << " PAT  electron in  " << src.id() << " comes from " << electron.refToOrig_.id() << ", " << electron.refToOrig_.key() << std::endl;
            keys.clear();
            for(auto const& pf: (*reco2pf)[electron.refToOrig_]) {
              if( pf2pc->contains(pf.id()) ) {
                keys.push_back( (*pf2pc)[pf].key());
              }
            }
            electron.setAssociatedPackedPFCandidates(edm::RefProd<pat::PackedCandidateCollection>(pc),
                                                     keys.begin(), keys.end());
            //std::cout << "Electron with pt " << electron.pt() << " associated to " << electron.associatedPackedFCandidateIndices_.size() << " PF Candidates\n";
            //if there's just one PF Cand then it's me, otherwise I have no univoque parent so my ref will be null 
            if (keys.size() == 1) {
                electron.refToOrig_ = electron.sourceCandidatePtr(0);
            } else {
                electron.refToOrig_ = reco::CandidatePtr(pc.id());
            }
        }
        if (saveNonZSClusterShapes_(electron)) {
            std::vector<float> vCov = lazyToolsNoZS.localCovariances(*( electron.superCluster()->seed()));
            electron.full5x5_setSigmaIetaIphi(vCov[1]);
        }
    }

    iEvent.put(std::move(out));
}

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat;
DEFINE_FWK_MODULE(PATElectronSlimmer);
