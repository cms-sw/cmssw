// system include files
#include <memory>
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Phase2L1ParticleFlow/interface/PFCandidate.h"
#include "DataFormats/L1TVertex/interface/Vertex.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "L1Trigger/Phase2L1ParticleFlow/interface/RegionMapper.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/PFAlgoBase.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/PFAlgo3.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/PFAlgo2HGC.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/PuppiAlgo.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/LinearizedPuppiAlgo.h"

//--------------------------------------------------------------------------------------------------
class L1TPFProducer : public edm::stream::EDProducer<> {
    public:
        explicit L1TPFProducer(const edm::ParameterSet&);

    private:
        int debug_;

        bool hasTracks_;
        edm::EDGetTokenT<l1t::PFTrackCollection> tkCands_;
        float trkPt_, trkMaxChi2_;
        unsigned trkMinStubs_;
        l1tpf_impl::PUAlgoBase::VertexAlgo vtxAlgo_;  
        edm::EDGetTokenT<std::vector<l1t::Vertex>> extVtx_;

        edm::EDGetTokenT<l1t::MuonBxCollection> muCands_;

        std::vector<edm::EDGetTokenT<l1t::PFClusterCollection>> emCands_;
        std::vector<edm::EDGetTokenT<l1t::PFClusterCollection>> hadCands_;

        float emPtCut_, hadPtCut_;

        l1tpf_impl::RegionMapper l1regions_;
        std::unique_ptr<l1tpf_impl::PFAlgoBase> l1pfalgo_;
        std::unique_ptr<l1tpf_impl::PUAlgoBase> l1pualgo_;

        // region of interest debugging
        float debugEta_, debugPhi_, debugR_;

        virtual void produce(edm::Event&, const edm::EventSetup&) override;
};

//
// constructors and destructor
//
L1TPFProducer::L1TPFProducer(const edm::ParameterSet& iConfig):
    debug_(iConfig.getUntrackedParameter<int>("debug",0)),
    hasTracks_(!iConfig.getParameter<edm::InputTag>("tracks").label().empty()),
    tkCands_(hasTracks_ ? consumes<l1t::PFTrackCollection>(iConfig.getParameter<edm::InputTag>("tracks")) : edm::EDGetTokenT<l1t::PFTrackCollection>()),
    trkPt_(iConfig.getParameter<double>("trkPtCut")),
    trkMaxChi2_(iConfig.getParameter<double>("trkMaxChi2")),
    trkMinStubs_(iConfig.getParameter<unsigned>("trkMinStubs")),
    muCands_(consumes<l1t::MuonBxCollection>(iConfig.getParameter<edm::InputTag>("muons"))),
    emPtCut_(iConfig.getParameter<double>("emPtCut")),
    hadPtCut_(iConfig.getParameter<double>("hadPtCut")),
    l1regions_(iConfig),
    l1pfalgo_(nullptr),
    l1pualgo_(nullptr),
    debugEta_(iConfig.getUntrackedParameter<double>("debugEta",0)),
    debugPhi_(iConfig.getUntrackedParameter<double>("debugPhi",0)),
    debugR_(iConfig.getUntrackedParameter<double>("debugR",-1))
{
    produces<l1t::PFCandidateCollection>("PF");
    produces<l1t::PFCandidateCollection>("Puppi");

    //produces<l1t::PFCandidateCollection>("RawEmCalo");
    //produces<l1t::PFCandidateCollection>("RawCalo");

    produces<l1t::PFCandidateCollection>("EmCalo");
    produces<l1t::PFCandidateCollection>("Calo");
    produces<l1t::PFCandidateCollection>("TK");
    produces<l1t::PFCandidateCollection>("TKVtx");

    produces<float>("z0");

    for (auto & tag : iConfig.getParameter<std::vector<edm::InputTag>>("emClusters")) {
        emCands_.push_back(consumes<l1t::PFClusterCollection>(tag));
    }
    for (auto & tag : iConfig.getParameter<std::vector<edm::InputTag>>("hadClusters")) {
        hadCands_.push_back(consumes<l1t::PFClusterCollection>(tag));
    }

    const std::string & algo = iConfig.getParameter<std::string>("pfAlgo");
    if (algo == "PFAlgo3") {
        l1pfalgo_.reset(new l1tpf_impl::PFAlgo3(iConfig));
    } else if (algo == "PFAlgo2HGC") {
        l1pfalgo_.reset(new l1tpf_impl::PFAlgo2HGC(iConfig));
    } else if (algo == "BitwisePF") { // FIXME add back
        throw cms::Exception("Configuration", "FIXME: recover Bitwise PF");
    } else throw cms::Exception("Configuration", "Unsupported PFAlgo");

    const std::string & pualgo = iConfig.getParameter<std::string>("puAlgo");
    if (pualgo == "Puppi") {
        l1pualgo_.reset(new l1tpf_impl::PuppiAlgo(iConfig));
    } else if (pualgo == "LinearizedPuppi") {
        l1pualgo_.reset(new l1tpf_impl::LinearizedPuppiAlgo(iConfig));
    } else throw cms::Exception("Configuration", "Unsupported PUAlgo");

    std::string vtxAlgo = iConfig.getParameter<std::string>("vtxAlgo");
    if      (vtxAlgo == "TP")  vtxAlgo_ = l1tpf_impl::PUAlgoBase::TPVtxAlgo;
    else if (vtxAlgo == "old") vtxAlgo_ = l1tpf_impl::PUAlgoBase::OldVtxAlgo;
    else if (vtxAlgo == "external") {
        vtxAlgo_ = l1tpf_impl::PUAlgoBase::ExternalVtxAlgo;
        extVtx_  = consumes<std::vector<l1t::Vertex>>(iConfig.getParameter<edm::InputTag>("vtxCollection"));
    } else throw cms::Exception("Configuration") << "Unsupported vtxAlgo " << vtxAlgo << "\n";


    for (const std::string & label : l1pualgo_->puGlobalNames()) {
        produces<float>(label); 
    }
}

// ------------ method called to produce the data  ------------
void
L1TPFProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    /// ------ READ TRACKS ----
    if (hasTracks_) {
        edm::Handle<l1t::PFTrackCollection> htracks;
        iEvent.getByToken(tkCands_, htracks);
        const auto & tracks = *htracks;
        for (unsigned int itk = 0, ntk = tracks.size(); itk < ntk; ++itk) {
            const auto & tk = tracks[itk];
            // adding objects to PF
            if (debugR_ > 0 && deltaR(tk.eta(),tk.phi(),debugEta_,debugPhi_) > debugR_) continue;
            if (tk.pt() > trkPt_ && tk.nStubs() >= trkMinStubs_ && tk.normalizedChi2() < trkMaxChi2_) {
                l1regions_.addTrack(tk, l1t::PFTrackRef(htracks,itk)); 
            }
        }
    }

    /// ------ READ MUONS ----
    edm::Handle<l1t::MuonBxCollection> muons;
    iEvent.getByToken(muCands_, muons);
    for (auto it = muons->begin(0), ed = muons->end(0); it != ed; ++it) {
        const l1t::Muon & mu = *it;
        if (debugR_ > 0 && deltaR(mu.eta(),mu.phi(),debugEta_,debugPhi_) > debugR_) continue;
        l1regions_.addMuon(mu, l1t::PFCandidate::MuonRef(muons, muons->key(it)));
    }

    // ------ READ CALOS -----
    edm::Handle<l1t::PFClusterCollection> caloHandle;
    for (const auto & tag : emCands_) {
        iEvent.getByToken(tag, caloHandle);
        const auto & calos = *caloHandle;
        for (unsigned int ic = 0, nc = calos.size(); ic < nc; ++ic) {
            const auto & calo = calos[ic];
            if (debugR_ > 0 && deltaR(calo.eta(),calo.phi(),debugEta_,debugPhi_) > debugR_) continue;
            if (calo.pt() > emPtCut_) l1regions_.addEmCalo(calo, l1t::PFClusterRef(caloHandle,ic));
        }
    }
    for (const auto & tag : hadCands_) {
        iEvent.getByToken(tag, caloHandle);
        const auto & calos = *caloHandle;
        for (unsigned int ic = 0, nc = calos.size(); ic < nc; ++ic) {
            const auto & calo = calos[ic];
            if (debugR_ > 0 && deltaR(calo.eta(),calo.phi(),debugEta_,debugPhi_) > debugR_) continue;
            if (calo.pt() > hadPtCut_) l1regions_.addCalo(calo, l1t::PFClusterRef(caloHandle,ic));
        }
    }

    // First, get a copy of the discretized and corrected inputs, and write them out
    // FIXME: to be implemented
    iEvent.put(l1regions_.fetchCalo(/*ptmin=*/0.1, /*em=*/true),  "EmCalo");
    iEvent.put(l1regions_.fetchCalo(/*ptmin=*/0.1, /*em=*/false), "Calo");
    iEvent.put(l1regions_.fetchTracks(/*ptmin=*/0.0, /*fromPV=*/false), "TK");

    // Then do the vertexing, and save it out
    float z0;
    if (vtxAlgo_ == l1tpf_impl::PUAlgoBase::ExternalVtxAlgo) {
        edm::Handle<std::vector<l1t::Vertex>> vtxHandle;
        iEvent.getByToken(extVtx_, vtxHandle);
        z0 = 0;
        double ptsum = 0;
        for (const l1t::Vertex & vtx : *vtxHandle) {
            double myptsum = 0; 
            for (const auto & tkptr : vtx.tracks()) { myptsum += std::min(tkptr->getMomentum(4).perp(), 50.f); }
            if (myptsum > ptsum) { z0 = vtx.z0(); ptsum = myptsum; }
        }
    }
    l1pualgo_->doVertexing(l1regions_.regions(), vtxAlgo_, z0);
    iEvent.put(std::make_unique<float>(z0), "z0");

    // Then also save the tracks with a vertex cut
    iEvent.put(l1regions_.fetchTracks(/*ptmin=*/0.0, /*fromPV=*/true), "TKVtx");

    // Then run PF in each region
    for (auto & l1region : l1regions_.regions()) {
        l1pfalgo_->runPF(l1region);
        l1pualgo_->runChargedPV(l1region, z0);
    }
    // save PF into the event
    iEvent.put(l1regions_.fetch(false), "PF");

    // Then get our alphas (globally)
    std::vector<float> puGlobals;
    l1pualgo_->doPUGlobals(l1regions_.regions(), -1., puGlobals); // FIXME we don't have yet an external PU estimate
    const std::vector<std::string> & puGlobalNames = l1pualgo_->puGlobalNames();
    assert(puGlobals.size() == puGlobalNames.size());
    for (unsigned int i = 0, n = puGlobalNames.size(); i < n; ++i) {
        iEvent.put(std::make_unique<float>(puGlobals[i]), puGlobalNames[i]);
    }

    // Then run puppi (regionally)
    for (auto & l1region : l1regions_.regions()) {
        l1pualgo_->runNeutralsPU(l1region, -1., puGlobals);
    }
    // and save puppi
    iEvent.put(l1regions_.fetch(true), "Puppi");

    // Then go do the multiplicities
    // FIXME: recover
    /*
    for (int i = 0; i < l1tpf_impl::Region::n_input_types; ++i) {
        auto totAndMax = l1regions_.totAndMaxInput(i);
        addUInt(totAndMax.first,  std::string("totNL1")+l1tpf_impl::Region::inputTypeName(i), iEvent);
        addUInt(totAndMax.second, std::string("maxNL1")+l1tpf_impl::Region::inputTypeName(i), iEvent);
    }
    for (int i = 0; i < l1tpf_impl::Region::n_output_types; ++i) {
        auto totAndMaxPF = l1regions_.totAndMaxOutput(i,false);
        auto totAndMaxPuppi = l1regions_.totAndMaxOutput(i,true);
        addUInt(totAndMaxPF.first,  std::string("totNL1PF")+l1tpf_impl::Region::outputTypeName(i), iEvent);
        addUInt(totAndMaxPF.second, std::string("maxNL1PF")+l1tpf_impl::Region::outputTypeName(i), iEvent);
        addUInt(totAndMaxPuppi.first,  std::string("totNL1Puppi")+l1tpf_impl::Region::outputTypeName(i), iEvent);
        addUInt(totAndMaxPuppi.second, std::string("maxNL1Puppi")+l1tpf_impl::Region::outputTypeName(i), iEvent);
    }
    */

    // finall clear the regions
    l1regions_.clear();
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TPFProducer);

