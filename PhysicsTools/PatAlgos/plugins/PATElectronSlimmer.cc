//
// $Id: PATElectronSlimmer.cc,v 1.1 2011/03/24 18:45:45 mwlebour Exp $
//

/**
  \class    pat::PATElectronSlimmer PATElectronSlimmer.h "PhysicsTools/PatAlgos/interface/PATElectronSlimmer.h"
  \brief    Matcher of reconstructed objects to L1 Muons 
            
  \author   Giovanni Petrucciani
*/
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/PatCandidates/interface/Electron.h"


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"


namespace pat {

  class PATElectronSlimmer : public edm::EDProducer {
    public:
      explicit PATElectronSlimmer(const edm::ParameterSet & iConfig);
      virtual ~PATElectronSlimmer() { }

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

    private:
      edm::EDGetTokenT<edm::View<pat::Electron> > src_;

      StringCutObjectSelector<pat::Electron> dropSuperClusters_, dropBasicClusters_, dropPFlowClusters_, dropPreshowerClusters_, dropSeedCluster_, dropRecHits_;
      StringCutObjectSelector<pat::Electron> dropCorrections_,dropIsolations_,dropShapes_,dropExtrapolations_,dropClassifications_;

      edm::EDGetTokenT<edm::ValueMap<std::vector<reco::PFCandidateRef>>> reco2pf_;
      edm::EDGetTokenT<edm::Association<pat::PackedCandidateCollection>> pf2pc_;
      edm::EDGetTokenT<pat::PackedCandidateCollection>  pc_;
      bool linkToPackedPF_;
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
    dropExtrapolations_(iConfig.getParameter<std::string>("dropExtrapolations")),
    dropClassifications_(iConfig.getParameter<std::string>("dropClassifications")),
    linkToPackedPF_(iConfig.getParameter<bool>("linkToPackedPFCandidates"))
{
    produces<std::vector<pat::Electron> >();
    if (linkToPackedPF_) {
        reco2pf_ = consumes<edm::ValueMap<std::vector<reco::PFCandidateRef>>>(iConfig.getParameter<edm::InputTag>("recoToPFMap"));
        pf2pc_   = consumes<edm::Association<pat::PackedCandidateCollection>>(iConfig.getParameter<edm::InputTag>("packedPFCandidates"));
        pc_   = consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("packedPFCandidates"));
    }
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

    auto_ptr<vector<pat::Electron> >  out(new vector<pat::Electron>());
    out->reserve(src->size());

    for (View<pat::Electron>::const_iterator it = src->begin(), ed = src->end(); it != ed; ++it) {
        out->push_back(*it);
        pat::Electron & electron = out->back();
        if (dropSuperClusters_(electron)) { electron.superCluster_.clear(); electron.embeddedSuperCluster_ = false; }
	if (dropBasicClusters_(electron)) { electron.basicClusters_.clear();  }
	if (dropSuperClusters_(electron) || dropPFlowClusters_(electron)) { electron.pflowSuperCluster_.clear(); electron.embeddedPflowSuperCluster_ = false; }
	if (dropBasicClusters_(electron) || dropPFlowClusters_(electron)) { electron.pflowBasicClusters_.clear(); }
	if (dropPreshowerClusters_(electron)) { electron.preshowerClusters_.clear(); electron.embeddedSuperCluster_ = false; }
	if (dropPreshowerClusters_(electron) || dropPFlowClusters_(electron)) { electron.pflowPreshowerClusters_.clear(); }
	if (dropSeedCluster_(electron)) { electron.seedCluster_.clear(); electron.embeddedSeedCluster_ = false; }
        if (dropRecHits_(electron)) { electron.recHits_ = EcalRecHitCollection(); electron.embeddedRecHits_ = false; }
        if (dropCorrections_(electron)) { electron.setCorrections(reco::GsfElectron::Corrections()); }
        if (dropIsolations_(electron)) { electron.setDr03Isolation(reco::GsfElectron::IsolationVariables()); electron.setDr04Isolation(reco::GsfElectron::IsolationVariables()); electron.setPfIsolationVariables(reco::GsfElectron::PflowIsolationVariables()); }
        if (dropShapes_(electron)) { electron.setPfShowerShape(reco::GsfElectron::ShowerShape()); electron.setShowerShape(reco::GsfElectron::ShowerShape());  }
        if (dropExtrapolations_(electron)) { electron.setTrackExtrapolations(reco::GsfElectron::TrackExtrapolations());  }
        if (dropClassifications_(electron)) { electron.setClassificationVariables(reco::GsfElectron::ClassificationVariables()); electron.setClassification(reco::GsfElectron::Classification()); }
        if (linkToPackedPF_) {
            electron.setPackedPFCandidateCollection(edm::RefProd<pat::PackedCandidateCollection>(pc));
            //std::cout << " PAT  electron in  " << src.id() << " comes from " << electron.refToOrig_.id() << ", " << electron.refToOrig_.key() << std::endl;
            edm::RefVector<pat::PackedCandidateCollection> origs;
            for (const reco::PFCandidateRef & pf : (*reco2pf)[electron.refToOrig_]) {
                if (pf2pc->contains(pf.id())) {
                    origs.push_back((*pf2pc)[pf]);
                } //else std::cerr << " Electron linked to a PFCand in " << pf.id() << " while we expect them in " << pf2pc->ids().front().first << "\n";
            }
            //std::cout << "Electron with pt " << electron.pt() << " associated to " << origs.size() << " PF Candidates\n";
            electron.setAssociatedPackedPFCandidates(origs);
            //if there's just one PF Cand then it's me, otherwise I have no univoque parent so my ref will be null 
            if (origs.size() == 1) {
                electron.refToOrig_ = refToPtr(origs[0]);
            } else {
                electron.refToOrig_ = reco::CandidatePtr(pc.id());
            }
       }

    }

    iEvent.put(out);
}

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat;
DEFINE_FWK_MODULE(PATElectronSlimmer);
