/**
  Take as input:
     - the old PFCandidate collection (needed to setup the output ValueMap)
     - one ValueMap<reco::PFCandidateRef> that maps old PF candidates into new PF candidates (in multiple collections)
     - many edm::Association<pat::PackedCandidateCollection> that map new PF candidates into packed candidates
  Produce as output:
     - one ValueMap<reco::CandidatePtr> that maps the old PF candidates into the new packed PF candidates
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace pat {
  class PackedPFCandidateRefMixer : public edm::stream::EDProducer<> {
    public:
      explicit PackedPFCandidateRefMixer(const edm::ParameterSet & iConfig);
      virtual ~PackedPFCandidateRefMixer() { }

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup) override;

    private:
      edm::EDGetTokenT<std::vector<reco::PFCandidate>> pf_;
      edm::EDGetTokenT<edm::ValueMap<reco::PFCandidateRef>> pf2pf_;
      std::vector<edm::EDGetTokenT<edm::Association<pat::PackedCandidateCollection>>> pf2pcs_;
  };

} // namespace


pat::PackedPFCandidateRefMixer::PackedPFCandidateRefMixer(const edm::ParameterSet & iConfig) :
    pf_(consumes<std::vector<reco::PFCandidate>>(iConfig.getParameter<edm::InputTag>("pf"))),
    pf2pf_(consumes<edm::ValueMap<reco::PFCandidateRef>>(iConfig.getParameter<edm::InputTag>("pf2pf")))
{
    for (edm::InputTag const& tag : iConfig.getParameter<std::vector<edm::InputTag>>("pf2packed")) {
        pf2pcs_.push_back(consumes<edm::Association<pat::PackedCandidateCollection>>(tag));
    }
    produces<edm::ValueMap<reco::CandidatePtr>>();
}


void 
pat::PackedPFCandidateRefMixer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
    edm::Handle<std::vector<reco::PFCandidate>> pf;
    edm::Handle<edm::ValueMap<reco::PFCandidateRef>> pf2pf;
    std::vector<edm::Handle<edm::Association<pat::PackedCandidateCollection>>> pf2pcs(pf2pcs_.size());
    iEvent.getByToken(pf_, pf);
    iEvent.getByToken(pf2pf_, pf2pf);
    for (unsigned int i = 0, n = pf2pcs.size(); i < n; ++i) {
        iEvent.getByToken(pf2pcs_[i], pf2pcs[i]);
    }
    std::vector<reco::CandidatePtr> outptrs;
    outptrs.reserve(pf->size());
    for (unsigned int i = 0, n = pf->size(); i < n; ++i) {
        reco::PFCandidateRef oldpfRef(pf, i);
        const auto & newpfRef = (*pf2pf)[oldpfRef];
        bool found = false;
        for (const auto & pf2pc : pf2pcs) {
            if (pf2pc->contains(newpfRef.id())) {
                outptrs.push_back(refToPtr((*pf2pc)[newpfRef]));
                found = true;
                break;
            }
        }
        if (!found) {
            edm::LogPrint("PackedPFCandidateRefMixer") << "oldpfRef: " << oldpfRef.id() << " / " << oldpfRef.key() << "\n";
            edm::LogPrint("PackedPFCandidateRefMixer") << "newpfRef: " << newpfRef.id() << " / " << newpfRef.key() << "\n";
            edm::LogPrint("PackedPFCandidateRefMixer") << "and I have " << pf2pcs.size() << " rekey maps." << "\n";
            for (const auto & pf2pc : pf2pcs) {
                edm::LogPrint("PackedPFCandidateRefMixer") << "this map has keys in: " << "\n";
                for (const auto & pair : pf2pc->ids()) { edm::LogPrint("PackedPFCandidateRefMixer") << "\t" << pair.first << "\n"; }
            }
            throw cms::Exception("LogicError") << "A packed candidate has refs that we don't understand\n";
        }
    }
    std::unique_ptr<edm::ValueMap<reco::CandidatePtr>> out(new edm::ValueMap<reco::CandidatePtr>());
    edm::ValueMap<reco::CandidatePtr>::Filler filler(*out);
    filler.insert(pf, outptrs.begin(), outptrs.end());
    filler.fill();
    iEvent.put(std::move(out));
}

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat;
DEFINE_FWK_MODULE(PackedPFCandidateRefMixer);
