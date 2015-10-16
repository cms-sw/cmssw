/**
  \class    pat::PATPhotonSlimmer PATPhotonSlimmer.h "PhysicsTools/PatAlgos/interface/PATPhotonSlimmer.h"
  \brief    slimmer of PAT Taus 
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/RefToPtr.h"

#include "DataFormats/PatCandidates/interface/Photon.h"

#include "PhysicsTools/PatAlgos/interface/ObjectModifier.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "FWCore/Utilities/interface/isFinite.h"

namespace pat {

  class PATPhotonSlimmer : public edm::stream::EDProducer<> {
    public:
      explicit PATPhotonSlimmer(const edm::ParameterSet & iConfig);
      virtual ~PATPhotonSlimmer() { }

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);
      virtual void beginLuminosityBlock(const edm::LuminosityBlock&, const  edm::EventSetup&) override final;

    private:
      const edm::EDGetTokenT<edm::View<pat::Photon> > src_;

      const StringCutObjectSelector<pat::Photon> dropSuperClusters_, dropBasicClusters_, dropPreshowerClusters_, dropSeedCluster_, dropRecHits_;

      const edm::EDGetTokenT<edm::ValueMap<std::vector<reco::PFCandidateRef> > > reco2pf_;
      const edm::EDGetTokenT<edm::Association<pat::PackedCandidateCollection> > pf2pc_;
      const edm::EDGetTokenT<pat::PackedCandidateCollection>  pc_;
      const bool linkToPackedPF_;
      const StringCutObjectSelector<pat::Photon> saveNonZSClusterShapes_;
      const edm::EDGetTokenT<EcalRecHitCollection> reducedBarrelRecHitCollectionToken_, reducedEndcapRecHitCollectionToken_;
      const bool modifyPhoton_;
      std::unique_ptr<pat::ObjectModifier<pat::Photon> > photonModifier_;
  };

} // namespace

pat::PATPhotonSlimmer::PATPhotonSlimmer(const edm::ParameterSet & iConfig) :
    src_(consumes<edm::View<pat::Photon> >(iConfig.getParameter<edm::InputTag>("src"))),
    dropSuperClusters_(iConfig.getParameter<std::string>("dropSuperCluster")),
    dropBasicClusters_(iConfig.getParameter<std::string>("dropBasicClusters")),
    dropPreshowerClusters_(iConfig.getParameter<std::string>("dropPreshowerClusters")),
    dropSeedCluster_(iConfig.getParameter<std::string>("dropSeedCluster")),
    dropRecHits_(iConfig.getParameter<std::string>("dropRecHits")),
    reco2pf_(mayConsume<edm::ValueMap<std::vector<reco::PFCandidateRef> > >(iConfig.getParameter<edm::InputTag>("recoToPFMap"))),
    pf2pc_(mayConsume<edm::Association<pat::PackedCandidateCollection>>(iConfig.getParameter<edm::InputTag>("packedPFCandidates"))),
    pc_(mayConsume<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("packedPFCandidates"))),
    linkToPackedPF_(iConfig.getParameter<bool>("linkToPackedPFCandidates")),
    saveNonZSClusterShapes_(iConfig.getParameter<std::string>("saveNonZSClusterShapes")),
    reducedBarrelRecHitCollectionToken_(consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("reducedBarrelRecHitCollection"))),
    reducedEndcapRecHitCollectionToken_(consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("reducedEndcapRecHitCollection"))),
    modifyPhoton_(iConfig.getParameter<bool>("modifyPhotons"))
{
    edm::ConsumesCollector sumes(consumesCollector());
    if( modifyPhoton_ ) {
      const edm::ParameterSet& mod_config = iConfig.getParameter<edm::ParameterSet>("modifierConfig");
      photonModifier_.reset(new pat::ObjectModifier<pat::Photon>(mod_config) );
      photonModifier_->setConsumes(sumes);
    } else {
      photonModifier_.reset(nullptr);
    }
    
    mayConsume<EcalRecHitCollection>(edm::InputTag("reducedEcalRecHitsEB"));
    mayConsume<EcalRecHitCollection>(edm::InputTag("reducedEcalRecHitsEE"));

    produces<std::vector<pat::Photon> >();    
}

void 
pat::PATPhotonSlimmer::beginLuminosityBlock(const edm::LuminosityBlock&, const  edm::EventSetup& iSetup) {
}

void 
pat::PATPhotonSlimmer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
    using namespace edm;
    using namespace std;

    Handle<View<pat::Photon> >      src;
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

    auto_ptr<vector<pat::Photon> >  out(new vector<pat::Photon>());
    out->reserve(src->size());

    if( modifyPhoton_ ) { photonModifier_->setEvent(iEvent); }
    if( modifyPhoton_ ) photonModifier_->setEventContent(iSetup);

    std::vector<unsigned int> keys;
    for (View<pat::Photon>::const_iterator it = src->begin(), ed = src->end(); it != ed; ++it) {
        out->push_back(*it);
        pat::Photon & photon = out->back();

        if( modifyPhoton_ ) { photonModifier_->modify(photon); }

        if (dropSuperClusters_(photon)) { photon.superCluster_.clear(); photon.embeddedSuperCluster_ = false; }
	if (dropBasicClusters_(photon)) { photon.basicClusters_.clear(); }
	if (dropPreshowerClusters_(photon)) { photon.preshowerClusters_.clear(); }
	if (dropSeedCluster_(photon)) { photon.seedCluster_.clear(); photon.embeddedSeedCluster_ = false; }
        if (dropRecHits_(photon)) { photon.recHits_ = EcalRecHitCollection(); photon.embeddedRecHits_ = false; }

        if (linkToPackedPF_) {
            //std::cout << " PAT  photon in  " << src.id() << " comes from " << photon.refToOrig_.id() << ", " << photon.refToOrig_.key() << std::endl;
            keys.clear();
            for(auto const& pf: (*reco2pf)[photon.refToOrig_]) {
              if( pf2pc->contains(pf.id()) ) {
                keys.push_back( (*pf2pc)[pf].key());
              }
            }
            photon.setAssociatedPackedPFCandidates(edm::RefProd<pat::PackedCandidateCollection>(pc),
                                                   keys.begin(), keys.end());
            //std::cout << "Photon with pt " << photon.pt() << " associated to " << photon.associatedPackedFCandidateIndices_.size() << " PF Candidates\n";
            //if there's just one PF Cand then it's me, otherwise I have no univoque parent so my ref will be null 
            if (keys.size() == 1) {
                photon.refToOrig_ = photon.sourceCandidatePtr(0);
            } else {
                photon.refToOrig_ = reco::CandidatePtr(pc.id());
            }
        }
        if (saveNonZSClusterShapes_(photon)) {
            std::vector<float> vCov = lazyToolsNoZS.localCovariances(*( photon.superCluster()->seed()));
            float r9 = lazyToolsNoZS.e3x3( *( photon.superCluster()->seed())) / photon.superCluster()->rawEnergy() ;
            float sigmaIetaIeta = ( !edm::isNotFinite(vCov[0]) ) ? sqrt(vCov[0]) : 0;
            float sigmaIetaIphi = vCov[1];
            float sigmaIphiIphi = ( !edm::isNotFinite(vCov[2]) ) ? sqrt(vCov[2]) : 0;
            float e15o55 = lazyToolsNoZS.e1x5( *( photon.superCluster()->seed()) ) / lazyToolsNoZS.e5x5( *( photon.superCluster()->seed()) );
            photon.addUserFloat("sigmaIetaIeta_NoZS", sigmaIetaIeta);
            photon.addUserFloat("sigmaIetaIphi_NoZS", sigmaIetaIphi);
            photon.addUserFloat("sigmaIphiIphi_NoZS", sigmaIphiIphi);
            photon.addUserFloat("r9_NoZS", r9);
            photon.addUserFloat("e1x5_over_e5x5_NoZS", e15o55);
        }
     }

    iEvent.put(out);
}

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat;
DEFINE_FWK_MODULE(PATPhotonSlimmer);
