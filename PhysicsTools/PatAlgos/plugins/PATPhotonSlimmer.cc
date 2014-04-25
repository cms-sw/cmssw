/**
  \class    pat::PATPhotonSlimmer PATPhotonSlimmer.h "PhysicsTools/PatAlgos/interface/PATPhotonSlimmer.h"
  \brief    slimmer of PAT Taus 
*/


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/RefToPtr.h"

#include "DataFormats/PatCandidates/interface/Photon.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"

namespace pat {

  class PATPhotonSlimmer : public edm::EDProducer {
    public:
      explicit PATPhotonSlimmer(const edm::ParameterSet & iConfig);
      virtual ~PATPhotonSlimmer() { }

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

    private:
      edm::EDGetTokenT<edm::View<pat::Photon> > src_;

      StringCutObjectSelector<pat::Photon> dropSuperClusters_, dropBasicClusters_, dropPreshowerClusters_, dropSeedCluster_, dropRecHits_;

      edm::EDGetTokenT<edm::ValueMap<std::vector<reco::PFCandidateRef>>> reco2pf_;
      edm::EDGetTokenT<edm::Association<pat::PackedCandidateCollection>> pf2pc_;
      edm::EDGetTokenT<pat::PackedCandidateCollection>  pc_;
      bool linkToPackedPF_;
  };

} // namespace

pat::PATPhotonSlimmer::PATPhotonSlimmer(const edm::ParameterSet & iConfig) :
    src_(consumes<edm::View<pat::Photon> >(iConfig.getParameter<edm::InputTag>("src"))),
    dropSuperClusters_(iConfig.getParameter<std::string>("dropSuperCluster")),
    dropBasicClusters_(iConfig.getParameter<std::string>("dropBasicClusters")),
    dropPreshowerClusters_(iConfig.getParameter<std::string>("dropPreshowerClusters")),
    dropSeedCluster_(iConfig.getParameter<std::string>("dropSeedCluster")),
    dropRecHits_(iConfig.getParameter<std::string>("dropRecHits")),
    linkToPackedPF_(iConfig.getParameter<bool>("linkToPackedPFCandidates"))
{
    produces<std::vector<pat::Photon> >();
    if (linkToPackedPF_) {
        reco2pf_ = consumes<edm::ValueMap<std::vector<reco::PFCandidateRef>>>(iConfig.getParameter<edm::InputTag>("recoToPFMap"));
        pf2pc_   = consumes<edm::Association<pat::PackedCandidateCollection>>(iConfig.getParameter<edm::InputTag>("packedPFCandidates"));
        pc_   = consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("packedPFCandidates"));
    }
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

    auto_ptr<vector<pat::Photon> >  out(new vector<pat::Photon>());
    out->reserve(src->size());

    for (View<pat::Photon>::const_iterator it = src->begin(), ed = src->end(); it != ed; ++it) {
        out->push_back(*it);
        pat::Photon & photon = out->back();

        if (dropSuperClusters_(photon)) { photon.superCluster_.clear(); photon.embeddedSuperCluster_ = false; }
	if (dropBasicClusters_(photon)) { photon.basicClusters_.clear(); }
	if (dropPreshowerClusters_(photon)) { photon.preshowerClusters_.clear(); }
	if (dropSeedCluster_(photon)) { photon.seedCluster_.clear(); photon.embeddedSeedCluster_ = false; }
        if (dropRecHits_(photon)) { photon.recHits_ = EcalRecHitCollection(); photon.embeddedRecHits_ = false; }

        if (linkToPackedPF_) {
            photon.setPackedPFCandidateCollection(edm::RefProd<pat::PackedCandidateCollection>(pc));
            //std::cout << " PAT  photon in  " << src.id() << " comes from " << photon.refToOrig_.id() << ", " << photon.refToOrig_.key() << std::endl;
            edm::RefVector<pat::PackedCandidateCollection> origs;
            for (const reco::PFCandidateRef & pf : (*reco2pf)[photon.refToOrig_]) {
                if (pf2pc->contains(pf.id())) {
                    origs.push_back((*pf2pc)[pf]);
                } //else std::cerr << " Photon linked to a PFCand in " << pf.id() << " while we expect them in " << pf2pc->ids().front().first << "\n";
            }
            //std::cout << "Photon with pt " << photon.pt() << " associated to " << origs.size() << " PF Candidates\n";
            photon.setAssociatedPackedPFCandidates(origs);
            //if there's just one PF Cand then it's me, otherwise I have no univoque parent so my ref will be null 
            if (origs.size() == 1) {
                photon.refToOrig_ = refToPtr(origs[0]);
            } else {
                photon.refToOrig_ = reco::CandidatePtr(pc.id());
            }
        }
     }

    iEvent.put(out);
}

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat;
DEFINE_FWK_MODULE(PATPhotonSlimmer);
