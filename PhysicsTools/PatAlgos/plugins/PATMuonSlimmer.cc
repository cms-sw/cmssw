/**
  \class    pat::PATMuonSlimmer PATMuonSlimmer.h "PhysicsTools/PatAlgos/interface/PATMuonSlimmer.h"
  \brief    Slimmer of PAT Muons 
*/


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/RefToPtr.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "PhysicsTools/PatAlgos/interface/ObjectModifier.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"

namespace pat {
  
  class PATMuonSlimmer : public edm::stream::EDProducer<> {
  public:
    explicit PATMuonSlimmer(const edm::ParameterSet & iConfig);
    virtual ~PATMuonSlimmer() { }
    
    virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);
    virtual void beginLuminosityBlock(const edm::LuminosityBlock&, const  edm::EventSetup&) override final;
    
  private:
    const edm::EDGetTokenT<pat::MuonCollection> src_;
    std::vector<edm::EDGetTokenT<reco::PFCandidateCollection>> pf_;
    std::vector<edm::EDGetTokenT<edm::Association<pat::PackedCandidateCollection>>> pf2pc_;
    const bool linkToPackedPF_;
    const StringCutObjectSelector<pat::Muon> saveTeVMuons_;
    const bool modifyMuon_;
    std::unique_ptr<pat::ObjectModifier<pat::Muon> > muonModifier_;
  };

} // namespace

pat::PATMuonSlimmer::PATMuonSlimmer(const edm::ParameterSet & iConfig) :
    src_(consumes<pat::MuonCollection>(iConfig.getParameter<edm::InputTag>("src"))),
    linkToPackedPF_(iConfig.getParameter<bool>("linkToPackedPFCandidates")),
    saveTeVMuons_(iConfig.getParameter<std::string>("saveTeVMuons")),
    modifyMuon_(iConfig.getParameter<bool>("modifyMuons"))
{
    if (linkToPackedPF_) {
        const std::vector<edm::InputTag> & pf = (iConfig.existsAs<std::vector<edm::InputTag>>("pfCandidates") ?
                    iConfig.getParameter<std::vector<edm::InputTag>>("pfCandidates") :
                    std::vector<edm::InputTag>(1,iConfig.getParameter<edm::InputTag>("pfCandidates")));
        const std::vector<edm::InputTag> & pf2pc = (iConfig.existsAs<std::vector<edm::InputTag>>("packedPFCandidates") ?
                    iConfig.getParameter<std::vector<edm::InputTag>>("packedPFCandidates") :
                    std::vector<edm::InputTag>(1,iConfig.getParameter<edm::InputTag>("packedPFCandidates")));
        if (pf.size() != pf2pc.size()) throw cms::Exception("Configuration") << "Mismatching pfCandidates and packedPFCandidates\n";
        for (const edm::InputTag &tag : pf) pf_.push_back(consumes<reco::PFCandidateCollection>(tag));
        for (const edm::InputTag &tag : pf2pc) pf2pc_.push_back(consumes<edm::Association<pat::PackedCandidateCollection>>(tag));
    }

    edm::ConsumesCollector sumes(consumesCollector());
    if( modifyMuon_ ) {
      const edm::ParameterSet& mod_config = iConfig.getParameter<edm::ParameterSet>("modifierConfig");
      muonModifier_.reset(new pat::ObjectModifier<pat::Muon>(mod_config) );
      muonModifier_->setConsumes(sumes);
    } else {
      muonModifier_.reset(nullptr);
    }
    produces<std::vector<pat::Muon> >();
}

void 
pat::PATMuonSlimmer::beginLuminosityBlock(const edm::LuminosityBlock&, const  edm::EventSetup& iSetup) {
  if( modifyMuon_ ) muonModifier_->setEventContent(iSetup);
}

void 
pat::PATMuonSlimmer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
    using namespace edm;
    using namespace std;

    Handle<pat::MuonCollection>      src;
    iEvent.getByToken(src_, src);

    auto_ptr<vector<pat::Muon> >  out(new vector<pat::Muon>());
    out->reserve(src->size());

    if( modifyMuon_ ) { muonModifier_->setEvent(iEvent); }

    std::map<reco::CandidatePtr,pat::PackedCandidateRef> mu2pc;
    if (linkToPackedPF_) {
        Handle<reco::PFCandidateCollection> pf;
        Handle<edm::Association<pat::PackedCandidateCollection>> pf2pc;
        for (unsigned int ipfh = 0, npfh = pf_.size(); ipfh < npfh; ++ipfh) {
            iEvent.getByToken(pf_[ipfh], pf);
            iEvent.getByToken(pf2pc_[ipfh], pf2pc);
            const auto & pfcoll = (*pf);
            const auto & pfmap  = (*pf2pc);
            for (unsigned int i = 0, n = pf->size(); i < n; ++i) {
                const reco::PFCandidate &p = pfcoll[i];
                if (p.muonRef().isNonnull()) mu2pc[refToPtr(p.muonRef())] = pfmap[reco::PFCandidateRef(pf, i)];
            }
        }
    }

    for (vector<pat::Muon>::const_iterator it = src->begin(), ed = src->end(); it != ed; ++it) {
        out->push_back(*it);
        pat::Muon & mu = out->back();
        
        if( modifyMuon_ ) { muonModifier_->modify(mu); }

	if (saveTeVMuons_(mu)){mu.embedPickyMuon(); mu.embedTpfmsMuon(); mu.embedDytMuon();}
	if (linkToPackedPF_) {
            mu.refToOrig_ = refToPtr(mu2pc[mu.refToOrig_]);
        }
    }

    iEvent.put(out);
}

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat;
DEFINE_FWK_MODULE(PATMuonSlimmer);
