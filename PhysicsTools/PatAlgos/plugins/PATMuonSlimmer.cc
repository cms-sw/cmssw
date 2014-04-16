/**
  \class    pat::PATMuonSlimmer PATMuonSlimmer.h "PhysicsTools/PatAlgos/interface/PATMuonSlimmer.h"
  \brief    Slimmer of PAT Muons 
*/


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/RefToPtr.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

namespace pat {

  class PATMuonSlimmer : public edm::EDProducer {
    public:
      explicit PATMuonSlimmer(const edm::ParameterSet & iConfig);
      virtual ~PATMuonSlimmer() { }

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

    private:
      edm::EDGetTokenT<pat::MuonCollection> src_;
      edm::EDGetTokenT<reco::PFCandidateCollection> pf_;
      edm::EDGetTokenT<edm::Association<pat::PackedCandidateCollection>> pf2pc_;
      bool linkToPackedPF_;
  };

} // namespace

pat::PATMuonSlimmer::PATMuonSlimmer(const edm::ParameterSet & iConfig) :
    src_(consumes<pat::MuonCollection>(iConfig.getParameter<edm::InputTag>("src"))),
    linkToPackedPF_(iConfig.getParameter<bool>("linkToPackedPFCandidates"))
{
    produces<std::vector<pat::Muon> >();
    if (linkToPackedPF_) {
        pf_    = consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("pfCandidates"));
        pf2pc_ = consumes<edm::Association<pat::PackedCandidateCollection>>(iConfig.getParameter<edm::InputTag>("packedPFCandidates"));
    }
}

void 
pat::PATMuonSlimmer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
    using namespace edm;
    using namespace std;

    Handle<pat::MuonCollection>      src;
    iEvent.getByToken(src_, src);

    auto_ptr<vector<pat::Muon> >  out(new vector<pat::Muon>());
    out->reserve(src->size());

    std::map<reco::CandidatePtr,pat::PackedCandidateRef> mu2pc;
    if (linkToPackedPF_) {
        Handle<reco::PFCandidateCollection> pf;
        Handle<edm::Association<pat::PackedCandidateCollection>> pf2pc;
        iEvent.getByToken(pf_, pf);
        iEvent.getByToken(pf2pc_, pf2pc);
        for (unsigned int i = 0, n = pf->size(); i < n; ++i) {
            const reco::PFCandidate &p = (*pf)[i];
            if (p.muonRef().isNonnull()) mu2pc[refToPtr(p.muonRef())] = (*pf2pc)[reco::PFCandidateRef(pf, i)];
        }
    }

    for (vector<pat::Muon>::const_iterator it = src->begin(), ed = src->end(); it != ed; ++it) {
        out->push_back(*it);
        pat::Muon & mu = out->back();
        if (linkToPackedPF_) {
            mu.refToOrig_ = refToPtr(mu2pc[mu.refToOrig_]);
        }
    }

    iEvent.put(out);
}

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat;
DEFINE_FWK_MODULE(PATMuonSlimmer);
