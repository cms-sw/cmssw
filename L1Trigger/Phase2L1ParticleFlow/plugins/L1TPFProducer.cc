// system include files
#include <memory>
#include <algorithm>
#include <fstream>
#include <cstdio>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "DataFormats/L1TCorrelator/interface/TkPrimaryVertex.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "L1Trigger/Phase2L1ParticleFlow/interface/RegionMapper.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/PFAlgoBase.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/PFAlgo3.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/PFAlgo2HGC.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/BitwisePFAlgo.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/PuppiAlgo.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/LinearizedPuppiAlgo.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/DiscretePFInputsIO.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/COEFile.h"

#include "DataFormats/L1TCorrelator/interface/TkMuon.h"
#include "DataFormats/L1TCorrelator/interface/TkMuonFwd.h"

//--------------------------------------------------------------------------------------------------
class L1TPFProducer : public edm::stream::EDProducer<> {
public:
  explicit L1TPFProducer(const edm::ParameterSet&);
  ~L1TPFProducer() override;

private:
  edm::ParameterSet config_;
  int debug_;

  bool useStandaloneMuons_;
  bool useTrackerMuons_;

  bool hasTracks_;
  edm::EDGetTokenT<l1t::PFTrackCollection> tkCands_;
  float trkPt_, trkMaxChi2_;
  unsigned trkMinStubs_;
  l1tpf_impl::PUAlgoBase::VertexAlgo vtxAlgo_;
  edm::EDGetTokenT<std::vector<l1t::TkPrimaryVertex>> extTkVtx_;

  edm::EDGetTokenT<l1t::MuonBxCollection> muCands_;    // standalone muons
  edm::EDGetTokenT<l1t::TkMuonCollection> tkMuCands_;  // tk muons

  std::vector<edm::EDGetTokenT<l1t::PFClusterCollection>> emCands_;
  std::vector<edm::EDGetTokenT<l1t::PFClusterCollection>> hadCands_;

  float emPtCut_, hadPtCut_;

  l1tpf_impl::RegionMapper l1regions_;
  std::unique_ptr<l1tpf_impl::PFAlgoBase> l1pfalgo_;
  std::unique_ptr<l1tpf_impl::PUAlgoBase> l1pualgo_;

  edm::EDGetTokenT<math::XYZPointF> TokGenOrigin_;

  // Region dump/coe
  const std::string regionDumpName_, regionCOEName_;
  FILE* fRegionDump_;
  std::unique_ptr<l1tpf_impl::COEFile> fRegionCOE_;
  unsigned int neventscoemax_, neventsproduced_;

  // region of interest debugging
  float debugEta_, debugPhi_, debugR_;

  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void addUInt(unsigned int value, std::string iLabel, edm::Event& iEvent);
};

//
// constructors and destructor
//
L1TPFProducer::L1TPFProducer(const edm::ParameterSet& iConfig)
    : config_(iConfig),
      debug_(iConfig.getUntrackedParameter<int>("debug", 0)),
      useStandaloneMuons_(iConfig.getParameter<bool>("useStandaloneMuons")),
      useTrackerMuons_(iConfig.getParameter<bool>("useTrackerMuons")),
      hasTracks_(!iConfig.getParameter<edm::InputTag>("tracks").label().empty()),
      tkCands_(hasTracks_ ? consumes<l1t::PFTrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"))
                          : edm::EDGetTokenT<l1t::PFTrackCollection>()),
      trkPt_(iConfig.getParameter<double>("trkPtCut")),
      trkMaxChi2_(iConfig.getParameter<double>("trkMaxChi2")),
      trkMinStubs_(iConfig.getParameter<unsigned>("trkMinStubs")),
      muCands_(consumes<l1t::MuonBxCollection>(iConfig.getParameter<edm::InputTag>("muons"))),
      tkMuCands_(consumes<l1t::TkMuonCollection>(iConfig.getParameter<edm::InputTag>("tkMuons"))),
      emPtCut_(iConfig.getParameter<double>("emPtCut")),
      hadPtCut_(iConfig.getParameter<double>("hadPtCut")),
      l1regions_(iConfig),
      l1pfalgo_(nullptr),
      l1pualgo_(nullptr),
      regionDumpName_(iConfig.getUntrackedParameter<std::string>("dumpFileName", "")),
      regionCOEName_(iConfig.getUntrackedParameter<std::string>("coeFileName", "")),
      fRegionDump_(nullptr),
      fRegionCOE_(nullptr),
      neventscoemax_(iConfig.getUntrackedParameter<unsigned int>("neventscoemax_", 0)),
      neventsproduced_(0),
      debugEta_(iConfig.getUntrackedParameter<double>("debugEta", 0)),
      debugPhi_(iConfig.getUntrackedParameter<double>("debugPhi", 0)),
      debugR_(iConfig.getUntrackedParameter<double>("debugR", -1)) {
  produces<l1t::PFCandidateCollection>("PF");
  produces<l1t::PFCandidateCollection>("Puppi");

  produces<l1t::PFCandidateCollection>("EmCalo");
  produces<l1t::PFCandidateCollection>("Calo");
  produces<l1t::PFCandidateCollection>("TK");
  produces<l1t::PFCandidateCollection>("TKVtx");

  produces<float>("z0");

  for (const auto& tag : iConfig.getParameter<std::vector<edm::InputTag>>("emClusters")) {
    emCands_.push_back(consumes<l1t::PFClusterCollection>(tag));
  }
  for (const auto& tag : iConfig.getParameter<std::vector<edm::InputTag>>("hadClusters")) {
    hadCands_.push_back(consumes<l1t::PFClusterCollection>(tag));
  }

  const std::string& algo = iConfig.getParameter<std::string>("pfAlgo");
  if (algo == "PFAlgo3") {
    l1pfalgo_ = std::make_unique<l1tpf_impl::PFAlgo3>(iConfig);
  } else if (algo == "PFAlgo2HGC") {
    l1pfalgo_ = std::make_unique<l1tpf_impl::PFAlgo2HGC>(iConfig);
  } else if (algo == "BitwisePFAlgo") {
    l1pfalgo_ = std::make_unique<l1tpf_impl::BitwisePFAlgo>(iConfig);
  } else
    throw cms::Exception("Configuration", "Unsupported PFAlgo");

  const std::string& pualgo = iConfig.getParameter<std::string>("puAlgo");
  if (pualgo == "Puppi") {
    l1pualgo_ = std::make_unique<l1tpf_impl::PuppiAlgo>(iConfig);
  } else if (pualgo == "LinearizedPuppi") {
    l1pualgo_ = std::make_unique<l1tpf_impl::LinearizedPuppiAlgo>(iConfig);
  } else
    throw cms::Exception("Configuration", "Unsupported PUAlgo");

  std::string vtxAlgo = iConfig.getParameter<std::string>("vtxAlgo");
  if (vtxAlgo == "TP")
    vtxAlgo_ = l1tpf_impl::PUAlgoBase::VertexAlgo::TP;
  else if (vtxAlgo == "old")
    vtxAlgo_ = l1tpf_impl::PUAlgoBase::VertexAlgo::Old;
  else if (vtxAlgo == "external") {
    vtxAlgo_ = l1tpf_impl::PUAlgoBase::VertexAlgo::External;
    const std::string& vtxFormat = iConfig.getParameter<std::string>("vtxFormat");
    if (vtxFormat == "TkPrimaryVertex") {
      extTkVtx_ = consumes<std::vector<l1t::TkPrimaryVertex>>(iConfig.getParameter<edm::InputTag>("vtxCollection"));
    } else
      throw cms::Exception("Configuration") << "Unsupported vtxFormat " << vtxFormat << "\n";
  } else
    throw cms::Exception("Configuration") << "Unsupported vtxAlgo " << vtxAlgo << "\n";

  for (const std::string& label : l1pualgo_->puGlobalNames()) {
    produces<float>(label);
  }

  if (!regionDumpName_.empty()) {
    TokGenOrigin_ = consumes<math::XYZPointF>(iConfig.getParameter<edm::InputTag>("genOrigin"));
  }
  for (int tot = 0; tot <= 1; ++tot) {
    for (int i = 0; i < l1tpf_impl::Region::n_input_types; ++i) {
      produces<unsigned int>(std::string(tot ? "totNL1" : "maxNL1") + l1tpf_impl::Region::inputTypeName(i));
    }
    for (int i = 0; i < l1tpf_impl::Region::n_output_types; ++i) {
      produces<unsigned int>(std::string(tot ? "totNL1PF" : "maxNL1PF") + l1tpf_impl::Region::outputTypeName(i));
      produces<unsigned int>(std::string(tot ? "totNL1Puppi" : "maxNL1Puppi") + l1tpf_impl::Region::outputTypeName(i));
    }
  }
  for (int i = 0; i < l1tpf_impl::Region::n_input_types; ++i) {
    produces<std::vector<unsigned>>(std::string("vecNL1") + l1tpf_impl::Region::inputTypeName(i));
  }
  for (int i = 0; i < l1tpf_impl::Region::n_output_types; ++i) {
    produces<std::vector<unsigned>>(std::string("vecNL1PF") + l1tpf_impl::Region::outputTypeName(i));
    produces<std::vector<unsigned>>(std::string("vecNL1Puppi") + l1tpf_impl::Region::outputTypeName(i));
  }
}

L1TPFProducer::~L1TPFProducer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  if (fRegionDump_)
    fclose(fRegionDump_);
  if (fRegionCOE_)
    fRegionCOE_->close();
}

void L1TPFProducer::beginStream(edm::StreamID id) {
  if (!regionDumpName_.empty()) {
    if (id == 0) {
      fRegionDump_ = fopen(regionDumpName_.c_str(), "wb");
    } else {
      edm::LogWarning("L1TPFProducer")
          << "Job running with multiple streams, but dump file will have only events on stream zero.";
    }
  }
  if (!regionCOEName_.empty()) {
    if (id == 0) {
      fRegionCOE_ = std::make_unique<l1tpf_impl::COEFile>(config_);
    } else {
      edm::LogWarning("L1TPFProducer")
          << "Job running with multiple streams, but COE file will dump only events on stream zero.";
    }
  }
}

// ------------ method called to produce the data  ------------
void L1TPFProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // clear the regions also at the beginning, in case one event didn't complete but the job continues on
  l1regions_.clear();

  /// ------ READ TRACKS ----
  if (hasTracks_) {
    edm::Handle<l1t::PFTrackCollection> htracks;
    iEvent.getByToken(tkCands_, htracks);
    const auto& tracks = *htracks;
    for (unsigned int itk = 0, ntk = tracks.size(); itk < ntk; ++itk) {
      const auto& tk = tracks[itk];
      // adding objects to PF
      if (debugR_ > 0 && deltaR(tk.eta(), tk.phi(), debugEta_, debugPhi_) > debugR_)
        continue;
      if (tk.pt() > trkPt_ && tk.nStubs() >= trkMinStubs_ && tk.normalizedChi2() < trkMaxChi2_) {
        l1regions_.addTrack(tk, l1t::PFTrackRef(htracks, itk));
      }
    }
  }

  /// ------ READ MUONS ----
  /// ------- first check that not more than one version of muons (standaloneMu or trackerMu) is set to be used in l1pflow
  if (useStandaloneMuons_ && useTrackerMuons_) {
    throw cms::Exception(
        "Configuration",
        "setting useStandaloneMuons=True && useTrackerMuons=True is not to be done, as it would duplicate all muons\n");
  }

  if (useStandaloneMuons_) {
    edm::Handle<l1t::MuonBxCollection> muons;
    iEvent.getByToken(muCands_, muons);
    for (auto it = muons->begin(0), ed = muons->end(0); it != ed; ++it) {
      const l1t::Muon& mu = *it;
      if (debugR_ > 0 && deltaR(mu.eta(), mu.phi(), debugEta_, debugPhi_) > debugR_)
        continue;
      l1regions_.addMuon(mu, l1t::PFCandidate::MuonRef(muons, muons->key(it)));
    }
  }

  if (useTrackerMuons_) {
    edm::Handle<l1t::TkMuonCollection> muons;
    iEvent.getByToken(tkMuCands_, muons);
    for (auto it = muons->begin(), ed = muons->end(); it != ed; ++it) {
      const l1t::TkMuon& mu = *it;
      if (debugR_ > 0 && deltaR(mu.eta(), mu.phi(), debugEta_, debugPhi_) > debugR_)
        continue;
      l1regions_.addMuon(mu);  // FIXME add a l1t::PFCandidate::MuonRef
    }
  }

  // ------ READ CALOS -----
  edm::Handle<l1t::PFClusterCollection> caloHandle;
  for (const auto& tag : emCands_) {
    iEvent.getByToken(tag, caloHandle);
    const auto& calos = *caloHandle;
    for (unsigned int ic = 0, nc = calos.size(); ic < nc; ++ic) {
      const auto& calo = calos[ic];
      if (debugR_ > 0 && deltaR(calo.eta(), calo.phi(), debugEta_, debugPhi_) > debugR_)
        continue;
      if (calo.pt() > emPtCut_)
        l1regions_.addEmCalo(calo, l1t::PFClusterRef(caloHandle, ic));
    }
  }
  for (const auto& tag : hadCands_) {
    iEvent.getByToken(tag, caloHandle);
    const auto& calos = *caloHandle;
    for (unsigned int ic = 0, nc = calos.size(); ic < nc; ++ic) {
      const auto& calo = calos[ic];
      if (debugR_ > 0 && deltaR(calo.eta(), calo.phi(), debugEta_, debugPhi_) > debugR_)
        continue;
      if (calo.pt() > hadPtCut_)
        l1regions_.addCalo(calo, l1t::PFClusterRef(caloHandle, ic));
    }
  }

  // First, get a copy of the discretized and corrected inputs, and write them out
  iEvent.put(l1regions_.fetchCalo(/*ptmin=*/0.1, /*em=*/true), "EmCalo");
  iEvent.put(l1regions_.fetchCalo(/*ptmin=*/0.1, /*em=*/false), "Calo");
  iEvent.put(l1regions_.fetchTracks(/*ptmin=*/0.0, /*fromPV=*/false), "TK");
  if (fRegionDump_) {
    uint32_t run = iEvent.id().run(), lumi = iEvent.id().luminosityBlock();
    uint64_t event = iEvent.id().event();
    fwrite(&run, sizeof(uint32_t), 1, fRegionDump_);
    fwrite(&lumi, sizeof(uint32_t), 1, fRegionDump_);
    fwrite(&event, sizeof(uint64_t), 1, fRegionDump_);
    l1tpf_impl::writeManyToFile(l1regions_.regions(), fRegionDump_);
  }

  // Then save the regions to the COE file
  // Do it here because there is some sorting going on in a later function
  if (fRegionCOE_ && fRegionCOE_->is_open() && neventsproduced_ < neventscoemax_) {
    std::vector<l1tpf_impl::Region> regions = l1regions_.regions();
    fRegionCOE_->writeTracksToFile(regions, neventsproduced_ == 0);
  }
  neventsproduced_++;

  // Then do the vertexing, and save it out
  float z0;
  if (vtxAlgo_ == l1tpf_impl::PUAlgoBase::VertexAlgo::External) {
    z0 = 0;
    double ptsum = 0;
    if (!extTkVtx_.isUninitialized()) {
      edm::Handle<std::vector<l1t::TkPrimaryVertex>> vtxHandle;
      iEvent.getByToken(extTkVtx_, vtxHandle);
      for (const l1t::TkPrimaryVertex& vtx : *vtxHandle) {
        if (ptsum == 0 || vtx.sum() > ptsum) {
          z0 = vtx.zvertex();
          ptsum = vtx.sum();
        }
      }
    } else
      throw cms::Exception("LogicError", "Inconsistent vertex configuration");
  }
  l1pualgo_->doVertexing(l1regions_.regions(), vtxAlgo_, z0);
  iEvent.put(std::make_unique<float>(z0), "z0");
  if (fRegionDump_) {
    fwrite(&z0, sizeof(float), 1, fRegionDump_);
    edm::Handle<math::XYZPointF> hGenOrigin;
    iEvent.getByToken(TokGenOrigin_, hGenOrigin);
    const math::XYZPointF& genOrigin = *hGenOrigin;
    float genZ = genOrigin.Z();
    fwrite(&genZ, sizeof(float), 1, fRegionDump_);
  }

  // Then also save the tracks with a vertex cut
  iEvent.put(l1regions_.fetchTracks(/*ptmin=*/0.0, /*fromPV=*/true), "TKVtx");

  // Then run PF in each region
  for (auto& l1region : l1regions_.regions()) {
    l1pfalgo_->runPF(l1region);
    l1pualgo_->runChargedPV(l1region, z0);
  }
  // save PF into the event
  iEvent.put(l1regions_.fetch(false), "PF");

  // Then get our alphas (globally)
  std::vector<float> puGlobals;
  l1pualgo_->doPUGlobals(l1regions_.regions(), -1., puGlobals);  // FIXME we don't have yet an external PU estimate
  const std::vector<std::string>& puGlobalNames = l1pualgo_->puGlobalNames();
  if (puGlobals.size() != puGlobalNames.size())
    throw cms::Exception("LogicError", "Mismatch in the number of global pileup inputs");
  for (unsigned int i = 0, n = puGlobalNames.size(); i < n; ++i) {
    iEvent.put(std::make_unique<float>(puGlobals[i]), puGlobalNames[i]);
  }
  if (fRegionDump_) {
    l1tpf_impl::writeManyToFile(puGlobals, fRegionDump_);
  }

  // Then run puppi (regionally)
  for (auto& l1region : l1regions_.regions()) {
    l1pualgo_->runNeutralsPU(l1region, -1., puGlobals);
  }
  // and save puppi
  iEvent.put(l1regions_.fetch(true), "Puppi");

  // Then go do the multiplicities

  for (int i = 0; i < l1tpf_impl::Region::n_input_types; ++i) {
    auto totAndMax = l1regions_.totAndMaxInput(i);
    addUInt(totAndMax.first, std::string("totNL1") + l1tpf_impl::Region::inputTypeName(i), iEvent);
    addUInt(totAndMax.second, std::string("maxNL1") + l1tpf_impl::Region::inputTypeName(i), iEvent);
    iEvent.put(l1regions_.vecInput(i), std::string("vecNL1") + l1tpf_impl::Region::inputTypeName(i));
  }
  for (int i = 0; i < l1tpf_impl::Region::n_output_types; ++i) {
    auto totAndMaxPF = l1regions_.totAndMaxOutput(i, false);
    auto totAndMaxPuppi = l1regions_.totAndMaxOutput(i, true);
    addUInt(totAndMaxPF.first, std::string("totNL1PF") + l1tpf_impl::Region::outputTypeName(i), iEvent);
    addUInt(totAndMaxPF.second, std::string("maxNL1PF") + l1tpf_impl::Region::outputTypeName(i), iEvent);
    addUInt(totAndMaxPuppi.first, std::string("totNL1Puppi") + l1tpf_impl::Region::outputTypeName(i), iEvent);
    addUInt(totAndMaxPuppi.second, std::string("maxNL1Puppi") + l1tpf_impl::Region::outputTypeName(i), iEvent);
    iEvent.put(l1regions_.vecOutput(i, false), std::string("vecNL1PF") + l1tpf_impl::Region::outputTypeName(i));
    iEvent.put(l1regions_.vecOutput(i, true), std::string("vecNL1Puppi") + l1tpf_impl::Region::outputTypeName(i));
  }

  // finally clear the regions
  l1regions_.clear();
}

void L1TPFProducer::addUInt(unsigned int value, std::string iLabel, edm::Event& iEvent) {
  iEvent.put(std::make_unique<unsigned>(value), iLabel);
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TPFProducer);
