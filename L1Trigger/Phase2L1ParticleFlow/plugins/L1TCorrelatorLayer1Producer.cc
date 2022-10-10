// system include files
#include <memory>
#include <algorithm>
#include <fstream>
#include <cstdio>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "DataFormats/L1Trigger/interface/Vertex.h"
#include "DataFormats/L1Trigger/interface/VertexWord.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "L1Trigger/Phase2L1ParticleFlow/interface/l1-converters/tkinput_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/l1-converters/muonGmtToL1ct_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/regionizer_base_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/multififo_regionizer_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/tdr_regionizer_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/pf/pfalgo2hgc_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/pf/pfalgo3_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/pf/pfalgo_dummy_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/puppi/linpuppi_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/egamma/pftkegalgo_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/pf/pfalgo_common_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/egamma/pftkegsorter_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/L1TCorrelatorLayer1PatternFileWriter.h"

#include "DataFormats/L1TCorrelator/interface/TkElectron.h"
#include "DataFormats/L1TCorrelator/interface/TkElectronFwd.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1TCorrelator/interface/TkEm.h"
#include "DataFormats/L1TCorrelator/interface/TkEmFwd.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"

//--------------------------------------------------------------------------------------------------
class L1TCorrelatorLayer1Producer : public edm::stream::EDProducer<> {
public:
  explicit L1TCorrelatorLayer1Producer(const edm::ParameterSet &);
  ~L1TCorrelatorLayer1Producer() override;

private:
  edm::ParameterSet config_;

  bool hasTracks_;
  edm::EDGetTokenT<l1t::PFTrackCollection> tkCands_;
  float trkPt_;
  bool emuTkVtx_;
  edm::EDGetTokenT<std::vector<l1t::Vertex>> extTkVtx_;
  edm::EDGetTokenT<std::vector<l1t::VertexWord>> tkVtxEmu_;
  int nVtx_;

  edm::EDGetTokenT<l1t::SAMuonCollection> muCands_;  // standalone muons

  std::vector<edm::EDGetTokenT<l1t::PFClusterCollection>> emCands_;
  std::vector<edm::EDGetTokenT<l1t::PFClusterCollection>> hadCands_;

  float emPtCut_, hadPtCut_;

  l1ct::Event event_;
  std::unique_ptr<l1ct::TrackInputEmulator> trackInput_;
  std::unique_ptr<l1ct::GMTMuonDecoderEmulator> muonInput_;
  std::unique_ptr<l1ct::RegionizerEmulator> regionizer_;
  std::unique_ptr<l1ct::PFAlgoEmulatorBase> l1pfalgo_;
  std::unique_ptr<l1ct::LinPuppiEmulator> l1pualgo_;
  std::unique_ptr<l1ct::PFTkEGAlgoEmulator> l1tkegalgo_;
  std::unique_ptr<l1ct::PFTkEGSorterEmulator> l1tkegsorter_;

  // Region dump
  const std::string regionDumpName_;
  bool writeRawHgcalCluster_;
  std::fstream fRegionDump_;
  const std::vector<edm::ParameterSet> patternWriterConfigs_;
  std::vector<std::unique_ptr<L1TCorrelatorLayer1PatternFileWriter>> patternWriters_;

  // region of interest debugging
  float debugEta_, debugPhi_, debugR_;

  // these are used to link items back
  std::unordered_map<const l1t::PFCluster *, l1t::PFClusterRef> clusterRefMap_;
  std::unordered_map<const l1t::PFTrack *, l1t::PFTrackRef> trackRefMap_;
  std::unordered_map<const l1t::SAMuon *, l1t::PFCandidate::MuonRef> muonRefMap_;

  // main methods
  void beginStream(edm::StreamID) override;
  void endStream() override;
  void produce(edm::Event &, const edm::EventSetup &) override;
  void addUInt(unsigned int value, std::string iLabel, edm::Event &iEvent);

  void initSectorsAndRegions(const edm::ParameterSet &iConfig);
  void initEvent(const edm::Event &e);
  // add object, tracking references
  void addTrack(const l1t::PFTrack &t, l1t::PFTrackRef ref);
  void addMuon(const l1t::SAMuon &t, l1t::PFCandidate::MuonRef ref);
  void addHadCalo(const l1t::PFCluster &t, l1t::PFClusterRef ref);
  void addEmCalo(const l1t::PFCluster &t, l1t::PFClusterRef ref);
  // add objects in already-decoded format
  void addDecodedTrack(l1ct::DetectorSector<l1ct::TkObjEmu> &sec, const l1t::PFTrack &t);
  void addDecodedMuon(l1ct::DetectorSector<l1ct::MuObjEmu> &sec, const l1t::SAMuon &t);
  void addDecodedHadCalo(l1ct::DetectorSector<l1ct::HadCaloObjEmu> &sec, const l1t::PFCluster &t);
  void addDecodedEmCalo(l1ct::DetectorSector<l1ct::EmCaloObjEmu> &sec, const l1t::PFCluster &t);

  void addRawHgcalCluster(l1ct::DetectorSector<ap_uint<256>> &sec, const l1t::PFCluster &c);

  // fetching outputs
  std::unique_ptr<l1t::PFCandidateCollection> fetchHadCalo() const;
  std::unique_ptr<l1t::PFCandidateCollection> fetchEmCalo() const;
  std::unique_ptr<l1t::PFCandidateCollection> fetchTracks() const;
  std::unique_ptr<l1t::PFCandidateCollection> fetchPF() const;
  std::unique_ptr<std::vector<l1t::PFTrack>> fetchDecodedTracks() const;
  void putPuppi(edm::Event &iEvent) const;

  void putEgStaObjects(edm::Event &iEvent,
                       const std::string &egLablel,
                       std::vector<edm::Ref<BXVector<l1t::EGamma>>> &egsta_refs);
  void putEgObjects(edm::Event &iEvent,
                    const bool writeEgSta,
                    const std::vector<edm::Ref<BXVector<l1t::EGamma>>> &egsta_refs,
                    const std::string &tkEmLabel,
                    const std::string &tkEmPerBoardLabel,
                    const std::string &tkEleLabel,
                    const std::string &tkElePerBoardLabel) const;

  template <typename T>
  void setRefs_(l1t::PFCandidate &pf, const T &p) const;

  void doVertexings(std::vector<float> &pvdz) const;
  // for multiplicities
  enum InputType { caloType = 0, emcaloType = 1, trackType = 2, l1muType = 3 };
  static constexpr const char *inputTypeName[l1muType + 1] = {"Calo", "EmCalo", "TK", "Mu"};
  std::unique_ptr<std::vector<unsigned>> vecSecInput(InputType i) const;
  std::unique_ptr<std::vector<unsigned>> vecRegInput(InputType i) const;
  typedef l1ct::OutputRegion::ObjType OutputType;
  std::unique_ptr<std::vector<unsigned>> vecOutput(OutputType i, bool usePuppi) const;
  std::pair<unsigned int, unsigned int> totAndMax(const std::vector<unsigned> &perRegion) const;
};

//
// constructors and destructor
//
L1TCorrelatorLayer1Producer::L1TCorrelatorLayer1Producer(const edm::ParameterSet &iConfig)
    : config_(iConfig),
      hasTracks_(!iConfig.getParameter<edm::InputTag>("tracks").label().empty()),
      tkCands_(hasTracks_ ? consumes<l1t::PFTrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"))
                          : edm::EDGetTokenT<l1t::PFTrackCollection>()),
      trkPt_(iConfig.getParameter<double>("trkPtCut")),
      muCands_(consumes<l1t::SAMuonCollection>(iConfig.getParameter<edm::InputTag>("muons"))),
      emPtCut_(iConfig.getParameter<double>("emPtCut")),
      hadPtCut_(iConfig.getParameter<double>("hadPtCut")),
      regionizer_(nullptr),
      l1pfalgo_(nullptr),
      l1pualgo_(nullptr),
      l1tkegalgo_(nullptr),
      l1tkegsorter_(nullptr),
      regionDumpName_(iConfig.getUntrackedParameter<std::string>("dumpFileName", "")),
      writeRawHgcalCluster_(iConfig.getUntrackedParameter<bool>("writeRawHgcalCluster", false)),
      patternWriterConfigs_(iConfig.getUntrackedParameter<std::vector<edm::ParameterSet>>(
          "patternWriters", std::vector<edm::ParameterSet>())),
      debugEta_(iConfig.getUntrackedParameter<double>("debugEta", 0)),
      debugPhi_(iConfig.getUntrackedParameter<double>("debugPhi", 0)),
      debugR_(iConfig.getUntrackedParameter<double>("debugR", -1)) {
  produces<l1t::PFCandidateCollection>("PF");
  produces<l1t::PFCandidateCollection>("Puppi");
  produces<l1t::PFCandidateRegionalOutput>("PuppiRegional");

  produces<l1t::PFCandidateCollection>("EmCalo");
  produces<l1t::PFCandidateCollection>("Calo");
  produces<l1t::PFCandidateCollection>("TK");
#if 0  // LATER
  produces<l1t::PFCandidateCollection>("TKVtx");
#endif
#if 0  // LATER
  produces<std::vector<l1t::PFTrack>>("DecodedTK");
#endif

  for (const auto &tag : iConfig.getParameter<std::vector<edm::InputTag>>("emClusters")) {
    emCands_.push_back(consumes<l1t::PFClusterCollection>(tag));
  }
  for (const auto &tag : iConfig.getParameter<std::vector<edm::InputTag>>("hadClusters")) {
    hadCands_.push_back(consumes<l1t::PFClusterCollection>(tag));
  }

  if (hasTracks_) {
    const std::string &tkInAlgo = iConfig.getParameter<std::string>("trackInputConversionAlgo");
    if (tkInAlgo == "Emulator") {
      trackInput_ = std::make_unique<l1ct::TrackInputEmulator>(
          iConfig.getParameter<edm::ParameterSet>("trackInputConversionParameters"));
    } else if (tkInAlgo != "Ideal")
      throw cms::Exception("Configuration", "Unsupported trackInputConversionAlgo");
  }

  const std::string &muInAlgo = iConfig.getParameter<std::string>("muonInputConversionAlgo");
  if (muInAlgo == "Emulator") {
    muonInput_ = std::make_unique<l1ct::GMTMuonDecoderEmulator>(
        iConfig.getParameter<edm::ParameterSet>("muonInputConversionParameters"));
  } else if (muInAlgo != "Ideal")
    throw cms::Exception("Configuration", "Unsupported muonInputConversionAlgo");

  const std::string &regalgo = iConfig.getParameter<std::string>("regionizerAlgo");
  if (regalgo == "Ideal") {
    regionizer_ =
        std::make_unique<l1ct::RegionizerEmulator>(iConfig.getParameter<edm::ParameterSet>("regionizerAlgoParameters"));
  } else if (regalgo == "Multififo") {
    regionizer_ = std::make_unique<l1ct::MultififoRegionizerEmulator>(
        iConfig.getParameter<edm::ParameterSet>("regionizerAlgoParameters"));
  } else if (regalgo == "MultififoBarrel") {
    const auto &pset = iConfig.getParameter<edm::ParameterSet>("regionizerAlgoParameters");
    regionizer_ =
        std::make_unique<l1ct::MultififoRegionizerEmulator>(pset.getParameter<std::string>("barrelSetup"), pset);
  } else if (regalgo == "TDR") {
    regionizer_ = std::make_unique<l1ct::TDRRegionizerEmulator>(
        iConfig.getParameter<edm::ParameterSet>("regionizerAlgoParameters"));
  } else
    throw cms::Exception("Configuration", "Unsupported regionizerAlgo");

  const std::string &algo = iConfig.getParameter<std::string>("pfAlgo");
  if (algo == "PFAlgo3") {
    l1pfalgo_ = std::make_unique<l1ct::PFAlgo3Emulator>(iConfig.getParameter<edm::ParameterSet>("pfAlgoParameters"));
  } else if (algo == "PFAlgo2HGC") {
    l1pfalgo_ = std::make_unique<l1ct::PFAlgo2HGCEmulator>(iConfig.getParameter<edm::ParameterSet>("pfAlgoParameters"));
  } else if (algo == "PFAlgoDummy") {
    l1pfalgo_ =
        std::make_unique<l1ct::PFAlgoDummyEmulator>(iConfig.getParameter<edm::ParameterSet>("pfAlgoParameters"));
  } else
    throw cms::Exception("Configuration", "Unsupported pfAlgo");

  const std::string &pualgo = iConfig.getParameter<std::string>("puAlgo");
  if (pualgo == "LinearizedPuppi") {
    l1pualgo_ = std::make_unique<l1ct::LinPuppiEmulator>(iConfig.getParameter<edm::ParameterSet>("puAlgoParameters"));
  } else
    throw cms::Exception("Configuration", "Unsupported puAlgo");

  l1tkegalgo_ = std::make_unique<l1ct::PFTkEGAlgoEmulator>(
      l1ct::PFTkEGAlgoEmuConfig(iConfig.getParameter<edm::ParameterSet>("tkEgAlgoParameters")));

  l1tkegsorter_ =
      std::make_unique<l1ct::PFTkEGSorterEmulator>(iConfig.getParameter<edm::ParameterSet>("tkEgSorterParameters"));

  if (l1tkegalgo_->writeEgSta())
    produces<BXVector<l1t::EGamma>>("L1Eg");
  produces<l1t::TkElectronCollection>("L1TkEle");
  produces<l1t::TkElectronRegionalOutput>("L1TkElePerBoard");
  produces<l1t::TkEmCollection>("L1TkEm");
  produces<l1t::TkEmRegionalOutput>("L1TkEmPerBoard");

  emuTkVtx_ = iConfig.getParameter<bool>("vtxCollectionEmulation");
  if (emuTkVtx_) {
    tkVtxEmu_ = consumes<std::vector<l1t::VertexWord>>(iConfig.getParameter<edm::InputTag>("vtxCollection"));
  } else {
    extTkVtx_ = consumes<std::vector<l1t::Vertex>>(iConfig.getParameter<edm::InputTag>("vtxCollection"));
  }
  nVtx_ = iConfig.getParameter<int>("nVtx");

  const char *iprefix[4] = {"totNReg", "maxNReg", "totNSec", "maxNSec"};
  for (int i = 0; i <= l1muType; ++i) {
    for (int ip = 0; ip < 4; ++ip) {
      produces<unsigned int>(std::string(iprefix[ip]) + inputTypeName[i]);
    }
    produces<std::vector<unsigned>>(std::string("vecNReg") + inputTypeName[i]);
    produces<std::vector<unsigned>>(std::string("vecNSec") + inputTypeName[i]);
  }
  const char *oprefix[4] = {"totNPF", "maxNPF", "totNPuppi", "maxNPuppi"};
  for (int i = 0; i < l1ct::OutputRegion::nPFTypes; ++i) {
    for (int ip = 0; ip < 4; ++ip) {
      produces<unsigned int>(std::string(oprefix[ip]) + l1ct::OutputRegion::objTypeName[i]);
    }
    produces<std::vector<unsigned>>(std::string("vecNPF") + l1ct::OutputRegion::objTypeName[i]);
    produces<std::vector<unsigned>>(std::string("vecNPuppi") + l1ct::OutputRegion::objTypeName[i]);
  }

  initSectorsAndRegions(iConfig);
}

L1TCorrelatorLayer1Producer::~L1TCorrelatorLayer1Producer() {}

void L1TCorrelatorLayer1Producer::beginStream(edm::StreamID id) {
  if (!regionDumpName_.empty()) {
    if (id == 0) {
      fRegionDump_.open(regionDumpName_.c_str(), std::ios::out | std::ios::binary);
    } else {
      edm::LogWarning("L1TCorrelatorLayer1Producer")
          << "Job running with multiple streams, but dump file will have only events on stream zero.";
    }
  }
  if (!patternWriterConfigs_.empty()) {
    if (id == 0) {
      for (const auto &pset : patternWriterConfigs_) {
        patternWriters_.emplace_back(std::make_unique<L1TCorrelatorLayer1PatternFileWriter>(pset, event_));
      }
    } else {
      edm::LogWarning("L1TCorrelatorLayer1Producer")
          << "Job running with multiple streams, but pattern files will be written only on stream zero.";
    }
  }
}

void L1TCorrelatorLayer1Producer::endStream() {
  for (auto &writer : patternWriters_) {
    writer->flush();
  }
}

// ------------ method called to produce the data  ------------
void L1TCorrelatorLayer1Producer::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // clear the regions also at the beginning, in case one event didn't complete but the job continues on
  initEvent(iEvent);

  /// ------ READ TRACKS ----
  if (hasTracks_) {
    edm::Handle<l1t::PFTrackCollection> htracks;
    iEvent.getByToken(tkCands_, htracks);
    const auto &tracks = *htracks;
    for (unsigned int itk = 0, ntk = tracks.size(); itk < ntk; ++itk) {
      const auto &tk = tracks[itk];
      // adding objects to PF
      if (debugR_ > 0 && deltaR(tk.eta(), tk.phi(), debugEta_, debugPhi_) > debugR_)
        continue;
      if (tk.pt() > trkPt_) {
        addTrack(tk, l1t::PFTrackRef(htracks, itk));
      }
    }
  }

  /// ------ READ MUONS ----
  edm::Handle<l1t::SAMuonCollection> muons;
  iEvent.getByToken(muCands_, muons);
  for (unsigned int i = 0, n = muons->size(); i < n; ++i) {
    const l1t::SAMuon &mu = (*muons)[i];
    if (debugR_ > 0 && deltaR(mu.eta(), mu.phi(), debugEta_, debugPhi_) > debugR_)
      continue;
    addMuon(mu, l1t::PFCandidate::MuonRef(muons, i));
  }
  // ------ READ CALOS -----
  edm::Handle<l1t::PFClusterCollection> caloHandle;
  for (const auto &tag : emCands_) {
    iEvent.getByToken(tag, caloHandle);
    const auto &calos = *caloHandle;
    for (unsigned int ic = 0, nc = calos.size(); ic < nc; ++ic) {
      const auto &calo = calos[ic];
      if (debugR_ > 0 && deltaR(calo.eta(), calo.phi(), debugEta_, debugPhi_) > debugR_)
        continue;
      if (calo.pt() > emPtCut_)
        addEmCalo(calo, l1t::PFClusterRef(caloHandle, ic));
    }
  }
  for (const auto &tag : hadCands_) {
    iEvent.getByToken(tag, caloHandle);
    const auto &calos = *caloHandle;
    for (unsigned int ic = 0, nc = calos.size(); ic < nc; ++ic) {
      const auto &calo = calos[ic];
      if (debugR_ > 0 && deltaR(calo.eta(), calo.phi(), debugEta_, debugPhi_) > debugR_)
        continue;
      if (calo.pt() > hadPtCut_)
        addHadCalo(calo, l1t::PFClusterRef(caloHandle, ic));
    }
  }

  regionizer_->run(event_.decoded, event_.pfinputs);

  // First, get a copy of the discretized and corrected inputs, and write them out
  iEvent.put(fetchEmCalo(), "EmCalo");
  iEvent.put(fetchHadCalo(), "Calo");
  iEvent.put(fetchTracks(), "TK");
  
  #if 0
    iEvent.put(fetchDecodedTracks(), "DecodedTK");
  #endif

  // Then do the vertexing, and save it out
  std::vector<float> z0s;
  std::vector<std::pair<float, float>> ptsums;
  float z0 = 0;
  double ptsum = 0;
  l1t::VertexWord pvwd;
  // FIXME: collections seem to be already sorted
  if (emuTkVtx_) {
    edm::Handle<std::vector<l1t::VertexWord>> vtxEmuHandle;
    iEvent.getByToken(tkVtxEmu_, vtxEmuHandle);
    for (const auto &vtx : *vtxEmuHandle) {
      ptsums.push_back(std::pair<float, float>(vtx.pt(), vtx.z0()));
      if (ptsum == 0 || vtx.pt() > ptsum) {
        ptsum = vtx.pt();
        pvwd = vtx;
      }
    }
  } else {
    edm::Handle<std::vector<l1t::Vertex>> vtxHandle;
    iEvent.getByToken(extTkVtx_, vtxHandle);
    for (const auto &vtx : *vtxHandle) {
      ptsums.push_back(std::pair<float, float>(vtx.pt(), vtx.z0()));
      if (ptsum == 0 || vtx.pt() > ptsum) {
        ptsum = vtx.pt();
        z0 = vtx.z0();
      }
    }
    pvwd = l1t::VertexWord(1, z0, 1, ptsum, 1, 1, 1);
  }
  l1ct::PVObjEmu hwpv;
  hwpv.hwZ0 = l1ct::Scales::makeZ0(pvwd.z0());
  event_.pvs.push_back(hwpv);
  event_.pvs_emu.push_back(pvwd.vertexWord());
  //Do a quick histogram vertexing to get multiple vertices (Hack for the taus)
  if (nVtx_ > 1) {
    std::stable_sort(ptsums.begin(), ptsums.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
    for (int i0 = 0; i0 < std::min(int(ptsums.size()), int(nVtx_)); i0++) {
      z0s.push_back(ptsums[i0].second);
    }
    for (unsigned int i = 1; i < z0s.size(); ++i) {
      l1ct::PVObjEmu hwpv;
      hwpv.hwZ0 = l1ct::Scales::makeZ0(z0s[i]);
      event_.pvs.push_back(hwpv);  //Skip emu
    }
  }
  // Then also save the tracks with a vertex cut
#if 0
  iEvent.put(l1regions_.fetchTracks(/*ptmin=*/0.0, /*fromPV=*/true), "TKVtx");
#endif

  // Then run PF in each region
  event_.out.resize(event_.pfinputs.size());
  for (unsigned int ir = 0, nr = event_.pfinputs.size(); ir < nr; ++ir) {
    l1pfalgo_->run(event_.pfinputs[ir], event_.out[ir]);
    l1pfalgo_->mergeNeutrals(event_.out[ir]);
    l1tkegalgo_->run(event_.pfinputs[ir], event_.out[ir]);
    l1tkegalgo_->runIso(event_.pfinputs[ir], event_.pvs, event_.out[ir]);
  }

  // Then run puppi (regionally)
  for (unsigned int ir = 0, nr = event_.pfinputs.size(); ir < nr; ++ir) {
    l1pualgo_->run(event_.pfinputs[ir], event_.pvs, event_.out[ir]);
    //l1pualgo_->runNeutralsPU(l1region, z0, -1., puGlobals);
  }

  // NOTE: This needs to happen before the EG sorting per board so that the EG objects
  // get a global reference to the EGSta before being mixed among differente regions
  std::vector<edm::Ref<BXVector<l1t::EGamma>>> egsta_refs;
  if (l1tkegalgo_->writeEgSta()) {
    putEgStaObjects(iEvent, "L1Eg", egsta_refs);
  }

  // l1tkegsorter_->setDebug(true);
  for (auto &board : event_.board_out) {
    l1tkegsorter_->run(event_.pfinputs, event_.out, board.region_index, board.egphoton);
    l1tkegsorter_->run(event_.pfinputs, event_.out, board.region_index, board.egelectron);
  }

  // save PF into the event
  iEvent.put(fetchPF(), "PF");

  // and save puppi
  putPuppi(iEvent);

  // save the EG objects
  putEgObjects(iEvent, l1tkegalgo_->writeEgSta(), egsta_refs, "L1TkEm", "L1TkEmPerBoard", "L1TkEle", "L1TkElePerBoard");

  // Then go do the multiplicities
  for (int i = 0; i <= l1muType; ++i) {
    auto vecInputs = vecSecInput(InputType(i));
    auto tm = totAndMax(*vecInputs);
    addUInt(tm.first, std::string("totNSec") + inputTypeName[i], iEvent);
    addUInt(tm.second, std::string("maxNSec") + inputTypeName[i], iEvent);
    iEvent.put(std::move(vecInputs), std::string("vecNSec") + inputTypeName[i]);
  }
  for (int i = 0; i <= l1muType; ++i) {
    auto vecInputs = vecRegInput(InputType(i));
    auto tm = totAndMax(*vecInputs);
    addUInt(tm.first, std::string("totNReg") + inputTypeName[i], iEvent);
    addUInt(tm.second, std::string("maxNReg") + inputTypeName[i], iEvent);
    iEvent.put(std::move(vecInputs), std::string("vecNReg") + inputTypeName[i]);
  }
  for (int i = 0; i < l1ct::OutputRegion::nPFTypes; ++i) {
    auto vecPF = vecOutput(OutputType(i), false);
    auto tmPF = totAndMax(*vecPF);
    addUInt(tmPF.first, std::string("totNPF") + l1ct::OutputRegion::objTypeName[i], iEvent);
    addUInt(tmPF.second, std::string("maxNPF") + l1ct::OutputRegion::objTypeName[i], iEvent);
    iEvent.put(std::move(vecPF), std::string("vecNPF") + l1ct::OutputRegion::objTypeName[i]);
    auto vecPuppi = vecOutput(OutputType(i), true);
    auto tmPuppi = totAndMax(*vecPuppi);
    addUInt(tmPuppi.first, std::string("totNPuppi") + l1ct::OutputRegion::objTypeName[i], iEvent);
    addUInt(tmPuppi.second, std::string("maxNPuppi") + l1ct::OutputRegion::objTypeName[i], iEvent);
    iEvent.put(std::move(vecPuppi), std::string("vecNPuppi") + l1ct::OutputRegion::objTypeName[i]);
  }

  if (fRegionDump_.is_open()) {
    event_.write(fRegionDump_);
  }
  for (auto &writer : patternWriters_) {
    writer->write(event_);
  }

  // finally clear the regions
  event_.clear();
}

void L1TCorrelatorLayer1Producer::addUInt(unsigned int value, std::string iLabel, edm::Event &iEvent) {
  iEvent.put(std::make_unique<unsigned>(value), iLabel);
}

void L1TCorrelatorLayer1Producer::initSectorsAndRegions(const edm::ParameterSet &iConfig) {
  // the track finder geometry is fixed
  unsigned int TF_phiSlices = 9;
  float TF_phiWidth = 2 * M_PI / TF_phiSlices;
  event_.decoded.track.clear();
  for (unsigned int ieta = 0, neta = 2; ieta < neta; ++ieta) {
    for (unsigned int iphi = 0; iphi < TF_phiSlices; ++iphi) {
      float phiCenter = reco::reduceRange(iphi * TF_phiWidth);
      event_.decoded.track.emplace_back((ieta ? 0. : -2.5), (ieta ? 2.5 : 0.0), phiCenter, TF_phiWidth);
      event_.raw.track.emplace_back((ieta ? 0. : -2.5), (ieta ? 2.5 : 0.0), phiCenter, TF_phiWidth);
    }
  }

  event_.decoded.emcalo.clear();
  event_.decoded.hadcalo.clear();
  event_.raw.hgcalcluster.clear();

  for (const edm::ParameterSet &preg : iConfig.getParameter<std::vector<edm::ParameterSet>>("caloSectors")) {
    std::vector<double> etaBoundaries = preg.getParameter<std::vector<double>>("etaBoundaries");
    if (!std::is_sorted(etaBoundaries.begin(), etaBoundaries.end()))
      throw cms::Exception("Configuration", "caloSectors.etaBoundaries not sorted\n");
    unsigned int phiSlices = preg.getParameter<uint32_t>("phiSlices");
    float phiWidth = 2 * M_PI / phiSlices;
    if (phiWidth > 2 * l1ct::Scales::maxAbsPhi())
      throw cms::Exception("Configuration", "caloSectors phi range too large for phi_t data type");
    double phiZero = preg.getParameter<double>("phiZero");
    for (unsigned int ieta = 0, neta = etaBoundaries.size() - 1; ieta < neta; ++ieta) {
      float etaWidth = etaBoundaries[ieta + 1] - etaBoundaries[ieta];
      if (etaWidth > 2 * l1ct::Scales::maxAbsEta())
        throw cms::Exception("Configuration", "caloSectors eta range too large for eta_t data type");
      for (unsigned int iphi = 0; iphi < phiSlices; ++iphi) {
        float phiCenter = reco::reduceRange(iphi * phiWidth + phiZero);
        event_.decoded.hadcalo.emplace_back(etaBoundaries[ieta], etaBoundaries[ieta + 1], phiCenter, phiWidth);
        event_.decoded.emcalo.emplace_back(etaBoundaries[ieta], etaBoundaries[ieta + 1], phiCenter, phiWidth);
        event_.raw.hgcalcluster.emplace_back(etaBoundaries[ieta], etaBoundaries[ieta + 1], phiCenter, phiWidth);
      }
    }
  }

  event_.decoded.muon.region = l1ct::PFRegionEmu(0., 0.);  // centered at (0,0)
  event_.raw.muon.region = l1ct::PFRegionEmu(0., 0.);      // centered at (0,0)

  event_.pfinputs.clear();
  for (const edm::ParameterSet &preg : iConfig.getParameter<std::vector<edm::ParameterSet>>("regions")) {
    std::vector<double> etaBoundaries = preg.getParameter<std::vector<double>>("etaBoundaries");
    if (!std::is_sorted(etaBoundaries.begin(), etaBoundaries.end()))
      throw cms::Exception("Configuration", "regions.etaBoundaries not sorted\n");
    unsigned int phiSlices = preg.getParameter<uint32_t>("phiSlices");
    float etaExtra = preg.getParameter<double>("etaExtra");
    float phiExtra = preg.getParameter<double>("phiExtra");
    float phiWidth = 2 * M_PI / phiSlices;
    for (unsigned int ieta = 0, neta = etaBoundaries.size() - 1; ieta < neta; ++ieta) {
      for (unsigned int iphi = 0; iphi < phiSlices; ++iphi) {
        float phiCenter = reco::reduceRange(iphi * phiWidth);  //align with L1 TrackFinder phi sector indexing
        event_.pfinputs.emplace_back(
            etaBoundaries[ieta], etaBoundaries[ieta + 1], phiCenter, phiWidth, etaExtra, phiExtra);
      }
    }
  }

  event_.board_out.clear();
  const std::vector<edm::ParameterSet> &board_params = iConfig.getParameter<std::vector<edm::ParameterSet>>("boards");
  event_.board_out.resize(board_params.size());
  for (unsigned int bidx = 0; bidx < board_params.size(); bidx++) {
    event_.board_out[bidx].region_index = board_params[bidx].getParameter<std::vector<unsigned int>>("regions");
    float etaBoard = 0.;
    float phiBoard = 0.;
    for (auto ridx : event_.board_out[bidx].region_index) {
      etaBoard += event_.pfinputs[ridx].region.floatEtaCenter();
      phiBoard += event_.pfinputs[ridx].region.floatPhiCenter();
    }
    event_.board_out[bidx].eta = etaBoard / event_.board_out[bidx].region_index.size();
    event_.board_out[bidx].phi = phiBoard / event_.board_out[bidx].region_index.size();
  }
}

void L1TCorrelatorLayer1Producer::initEvent(const edm::Event &iEvent) {
  event_.clear();
  event_.run = iEvent.id().run();
  event_.lumi = iEvent.id().luminosityBlock();
  event_.event = iEvent.id().event();
  clusterRefMap_.clear();
  trackRefMap_.clear();
  muonRefMap_.clear();
}

void L1TCorrelatorLayer1Producer::addTrack(const l1t::PFTrack &t, l1t::PFTrackRef ref) {
  auto &rawsectors = event_.raw.track;
  auto &sectors = event_.decoded.track;
  assert(sectors.size() == 18 && rawsectors.size() == 18);
  int isec = t.track()->phiSector() + (t.eta() >= 0 ? 9 : 0);
  rawsectors[isec].obj.push_back(t.trackWord().getTrackWord());
  addDecodedTrack(sectors[isec], t);
  trackRefMap_[&t] = ref;
}
void L1TCorrelatorLayer1Producer::addMuon(const l1t::SAMuon &mu, l1t::PFCandidate::MuonRef ref) {
  event_.raw.muon.obj.emplace_back(mu.word());
  addDecodedMuon(event_.decoded.muon, mu);
  muonRefMap_[&mu] = ref;
}
void L1TCorrelatorLayer1Producer::addHadCalo(const l1t::PFCluster &c, l1t::PFClusterRef ref) {
  int sidx = 0;
  for (auto &sec : event_.decoded.hadcalo) {
    if (sec.region.contains(c.eta(), c.phi())) {
      addDecodedHadCalo(sec, c);
      if (writeRawHgcalCluster_)
        addRawHgcalCluster(event_.raw.hgcalcluster[sidx], c);
    }
    sidx++;
  }
  clusterRefMap_[&c] = ref;
}
void L1TCorrelatorLayer1Producer::addEmCalo(const l1t::PFCluster &c, l1t::PFClusterRef ref) {
  for (auto &sec : event_.decoded.emcalo) {
    if (sec.region.contains(c.eta(), c.phi())) {
      addDecodedEmCalo(sec, c);
    }
  }
  clusterRefMap_[&c] = ref;
}

void L1TCorrelatorLayer1Producer::addDecodedTrack(l1ct::DetectorSector<l1ct::TkObjEmu> &sec, const l1t::PFTrack &t) {
  std::pair<l1ct::TkObjEmu, bool> tkAndSel;
  if (trackInput_) {
    tkAndSel = trackInput_->decodeTrack(t.trackWord().getTrackWord(), sec.region);
  } else {
    tkAndSel.first.hwPt = l1ct::Scales::makePtFromFloat(t.pt());
    tkAndSel.first.hwEta =
        l1ct::Scales::makeGlbEta(t.caloEta()) -
        sec.region.hwEtaCenter;  // important to enforce that the region boundary is on a discrete value
    tkAndSel.first.hwPhi = l1ct::Scales::makePhi(sec.region.localPhi(t.caloPhi()));
    tkAndSel.first.hwCharge = t.charge() > 0;
    tkAndSel.first.hwQuality = t.quality();
    tkAndSel.first.hwDEta = l1ct::Scales::makeEta(t.eta() - t.caloEta());
    tkAndSel.first.hwDPhi = l1ct::Scales::makePhi(std::abs(reco::deltaPhi(t.phi(), t.caloPhi())));
    tkAndSel.first.hwZ0 = l1ct::Scales::makeZ0(t.vertex().Z());
    tkAndSel.first.hwDxy = 0;
    tkAndSel.second = t.quality() > 0;
  }
  // CMSSW-only extra info
  tkAndSel.first.hwChi2 =  l1ct::Scales::makeChi2(t.chi2());
  tkAndSel.first.hwStubs = t.nStubs();
  tkAndSel.first.simPt = t.pt();
  tkAndSel.first.simCaloEta = t.caloEta();
  tkAndSel.first.simCaloPhi = t.caloPhi();
  tkAndSel.first.simVtxEta = t.eta();
  tkAndSel.first.simVtxPhi = t.phi();
  tkAndSel.first.simZ0 = t.vertex().Z();
  tkAndSel.first.simD0 = t.vertex().Rho();
  tkAndSel.first.src = &t;
  
  // If the track fails, we set its pT to zero, so that the decoded tracks are still aligned with the raw tracks
  // Downstream, the regionizer will just ignore zero-momentum tracks
  if (!tkAndSel.second)
    tkAndSel.first.hwPt = 0;
  sec.obj.push_back(tkAndSel.first);
}

void L1TCorrelatorLayer1Producer::addDecodedMuon(l1ct::DetectorSector<l1ct::MuObjEmu> &sec, const l1t::SAMuon &t) {
  l1ct::MuObjEmu mu;
  if (muonInput_) {
    mu = muonInput_->decode(t.word());
  } else {
    mu.hwPt = l1ct::Scales::makePtFromFloat(t.pt());
    mu.hwEta = l1ct::Scales::makeGlbEta(t.eta());  // IMPORTANT: input is in global coordinates!
    mu.hwPhi = l1ct::Scales::makeGlbPhi(t.phi());
    mu.hwCharge = !t.hwCharge();
    mu.hwQuality = t.hwQual() / 2;
    mu.hwDEta = 0;
    mu.hwDPhi = 0;
    mu.hwZ0 = l1ct::Scales::makeZ0(t.vertex().Z());
    mu.hwDxy = 0;  // Dxy not defined yet
  }
  mu.src = &t;
  sec.obj.push_back(mu);
}

void L1TCorrelatorLayer1Producer::addDecodedHadCalo(l1ct::DetectorSector<l1ct::HadCaloObjEmu> &sec,
                                                    const l1t::PFCluster &c) {
  l1ct::HadCaloObjEmu calo;
  calo.hwPt = l1ct::Scales::makePtFromFloat(c.pt());
  calo.hwEta = l1ct::Scales::makeGlbEta(c.eta()) -
               sec.region.hwEtaCenter;  // important to enforce that the region boundary is on a discrete value
  calo.hwPhi = l1ct::Scales::makePhi(sec.region.localPhi(c.phi()));
  calo.hwEmPt = l1ct::Scales::makePtFromFloat(c.emEt());
  calo.hwEmID = c.hwEmID();
  calo.hwSrrTot = l1ct::Scales::makeSrrTot(c.sigmaRR());
- calo.hwMeanZ = l1ct::Scales::makeMeanZ(c.absZBarycenter());
- calo.hwHoe = l1ct::Scales::makeHoe(c.hOverE());
  calo.src = &c;
  sec.obj.push_back(calo);
}

void L1TCorrelatorLayer1Producer::addRawHgcalCluster(l1ct::DetectorSector<ap_uint<256>> &sec, const l1t::PFCluster &c) {
  ap_uint<256> cwrd = 0;
  ap_uint<14> w_pt = round(c.pt() / 0.25);
  ap_uint<14> w_empt = round(c.emEt() / 0.25);
  constexpr float ETAPHI_LSB = M_PI / 720;
  ap_int<9> w_eta = round(sec.region.localEta(c.eta()) / ETAPHI_LSB);
  ap_int<9> w_phi = round(sec.region.localPhi(c.phi()) / ETAPHI_LSB);
  ap_uint<10> w_qual = c.hwQual();

  cwrd(13, 0) = w_pt;
  cwrd(27, 14) = w_empt;
  cwrd(72, 64) = w_eta;
  cwrd(81, 73) = w_phi;
  cwrd(115, 106) = w_qual;

  // FIXME: cluster-shape variables use by composite-ID need to be added here

  sec.obj.push_back(cwrd);
}

void L1TCorrelatorLayer1Producer::addDecodedEmCalo(l1ct::DetectorSector<l1ct::EmCaloObjEmu> &sec,
                                                   const l1t::PFCluster &c) {
  l1ct::EmCaloObjEmu calo;
  // set the endcap-sepcific variables to default value:
  calo.clear();
  calo.hwPt = l1ct::Scales::makePtFromFloat(c.pt());
  calo.hwEta = l1ct::Scales::makeGlbEta(c.eta()) -
               sec.region.hwEtaCenter;  // important to enforce that the region boundary is on a discrete value
  calo.hwPhi = l1ct::Scales::makePhi(sec.region.localPhi(c.phi()));
  calo.hwPtErr = l1ct::Scales::makePtFromFloat(c.ptError());
  calo.hwEmID = c.hwEmID();
  calo.src = &c;
  sec.obj.push_back(calo);
}

template <typename T>
void L1TCorrelatorLayer1Producer::setRefs_(l1t::PFCandidate &pf, const T &p) const {
  if (p.srcCluster) {
    auto match = clusterRefMap_.find(p.srcCluster);
    if (match == clusterRefMap_.end()) {
      throw cms::Exception("CorruptData") << "Invalid cluster pointer in PF candidate id " << p.intId() << " pt "
                                          << p.floatPt() << " eta " << p.floatEta() << " phi " << p.floatPhi();
    }
    pf.setPFCluster(match->second);
  }
  if (p.srcTrack) {
    auto match = trackRefMap_.find(p.srcTrack);
    if (match == trackRefMap_.end()) {
      throw cms::Exception("CorruptData") << "Invalid track pointer in PF candidate id " << p.intId() << " pt "
                                          << p.floatPt() << " eta " << p.floatEta() << " phi " << p.floatPhi();
    }
    pf.setPFTrack(match->second);
  }
  if (p.srcMu) {
    auto match = muonRefMap_.find(p.srcMu);
    if (match == muonRefMap_.end()) {
      throw cms::Exception("CorruptData") << "Invalid muon pointer in PF candidate id " << p.intId() << " pt "
                                          << p.floatPt() << " eta " << p.floatEta() << " phi " << p.floatPhi();
    }
    pf.setMuon(match->second);
  }
}

template <>
void L1TCorrelatorLayer1Producer::setRefs_<l1ct::PFNeutralObjEmu>(l1t::PFCandidate &pf,
                                                                  const l1ct::PFNeutralObjEmu &p) const {
  if (p.srcCluster) {
    auto match = clusterRefMap_.find(p.srcCluster);
    if (match == clusterRefMap_.end()) {
      throw cms::Exception("CorruptData") << "Invalid cluster pointer in PF candidate id " << p.intId() << " pt "
                                          << p.floatPt() << " eta " << p.floatEta() << " phi " << p.floatPhi();
    }
    pf.setPFCluster(match->second);
  }
}

template <>
void L1TCorrelatorLayer1Producer::setRefs_<l1ct::HadCaloObjEmu>(l1t::PFCandidate &pf,
                                                                const l1ct::HadCaloObjEmu &p) const {
  if (p.src) {
    auto match = clusterRefMap_.find(p.src);
    if (match == clusterRefMap_.end()) {
      throw cms::Exception("CorruptData") << "Invalid cluster pointer in hadcalo candidate  pt " << p.floatPt()
                                          << " eta " << p.floatEta() << " phi " << p.floatPhi();
    }
    pf.setPFCluster(match->second);
  }
}

template <>
void L1TCorrelatorLayer1Producer::setRefs_<l1ct::EmCaloObjEmu>(l1t::PFCandidate &pf,
                                                               const l1ct::EmCaloObjEmu &p) const {
  if (p.src) {
    auto match = clusterRefMap_.find(p.src);
    if (match == clusterRefMap_.end()) {
      throw cms::Exception("CorruptData") << "Invalid cluster pointer in emcalo candidate  pt " << p.floatPt()
                                          << " eta " << p.floatEta() << " phi " << p.floatPhi();
    }
    pf.setPFCluster(match->second);
  }
}

template <>
void L1TCorrelatorLayer1Producer::setRefs_<l1ct::TkObjEmu>(l1t::PFCandidate &pf, const l1ct::TkObjEmu &p) const {
  if (p.src) {
    auto match = trackRefMap_.find(p.src);
    if (match == trackRefMap_.end()) {
      throw cms::Exception("CorruptData") << "Invalid track pointer in track candidate  pt " << p.floatPt() << " eta "
                                          << p.floatEta() << " phi " << p.floatPhi();
    }
    pf.setPFTrack(match->second);
  }
}

std::unique_ptr<l1t::PFCandidateCollection> L1TCorrelatorLayer1Producer::fetchHadCalo() const {
  auto ret = std::make_unique<l1t::PFCandidateCollection>();
  for (const auto &r : event_.pfinputs) {
    const auto &reg = r.region;
    for (const auto &p : r.hadcalo) {
      if (p.hwPt == 0 || !reg.isFiducial(p))
        continue;
      reco::Particle::PolarLorentzVector p4(p.floatPt(), reg.floatGlbEtaOf(p), reg.floatGlbPhiOf(p), 0.13f);
      l1t::PFCandidate::ParticleType type = p.hwIsEM() ? l1t::PFCandidate::Photon : l1t::PFCandidate::NeutralHadron;
      ret->emplace_back(type, 0, p4, 1, p.intPt(), p.intEta(), p.intPhi());
      ret->back().setHwEmID(p.hwEmID);
      setRefs_(ret->back(), p);
    }
  }
  return ret;
}
std::unique_ptr<l1t::PFCandidateCollection> L1TCorrelatorLayer1Producer::fetchEmCalo() const {
  auto ret = std::make_unique<l1t::PFCandidateCollection>();
  for (const auto &r : event_.pfinputs) {
    const auto &reg = r.region;
    for (const auto &p : r.emcalo) {
      if (p.hwPt == 0 || !reg.isFiducial(p))
        continue;
      reco::Particle::PolarLorentzVector p4(p.floatPt(), reg.floatGlbEtaOf(p), reg.floatGlbPhiOf(p), 0.13f);
      ret->emplace_back(l1t::PFCandidate::Photon, 0, p4, 1, p.intPt(), p.intEta(), p.intPhi());
      ret->back().setHwEmID(p.hwEmID);
      setRefs_(ret->back(), p);
    }
  }
  return ret;
}
std::unique_ptr<l1t::PFCandidateCollection> L1TCorrelatorLayer1Producer::fetchTracks() const {
  auto ret = std::make_unique<l1t::PFCandidateCollection>();
  for (const auto &r : event_.pfinputs) {
    const auto &reg = r.region;
    for (const auto &p : r.track) {
      if (p.hwPt == 0 || !reg.isFiducial(p))
        continue;
      reco::Particle::PolarLorentzVector p4(
          p.floatPt(), reg.floatGlbEta(p.hwVtxEta()), reg.floatGlbPhi(p.hwVtxPhi()), 0.13f);
      ret->emplace_back(l1t::PFCandidate::ChargedHadron, p.intCharge(), p4, 1, p.intPt(), p.intEta(), p.intPhi());
      setRefs_(ret->back(), p);
    }
  }
  return ret;
}

std::unique_ptr<std::vector<l1t::PFTrack>> L1TCorrelatorLayer1Producer::fetchDecodedTracks() const {
  auto ret = std::make_unique<std::vector<l1t::PFTrack>>();
  for (const auto r : event_.decoded.track) {
    const auto &reg = r.region;
    for (const auto &p : r.obj) {
      if (p.hwPt == 0 || !reg.isFiducial(p))
        continue;
      reco::Particle::PolarLorentzVector p4(p.floatPt(), reg.floatGlbEta(p.hwVtxEta()), reg.floatGlbPhi(p.hwVtxPhi()), 0);

      reco::Particle::Point vtx(0,0,p.floatZ0());
      
      ret->emplace_back(l1t::PFTrack(p.intCharge(),
                                     reco::Particle::LorentzVector(p4),
                                     vtx,
                                     p.src->track(),
                                     0,
                                     reg.floatGlbEta(p.hwEta),
                                     reg.floatGlbPhi(p.hwPhi),
                                     -1,
                                     -1,
                                     p.hwQuality.to_int(),
                                     false,
                                     p.intPt(),
                                     p.intEta(),
                                     p.intPhi()));
    }
  }
  return ret;
}


std::unique_ptr<l1t::PFCandidateCollection> L1TCorrelatorLayer1Producer::fetchPF() const {
  auto ret = std::make_unique<l1t::PFCandidateCollection>();
  for (unsigned int ir = 0, nr = event_.pfinputs.size(); ir < nr; ++ir) {
    const auto &reg = event_.pfinputs[ir].region;
    for (const auto &p : event_.out[ir].pfcharged) {
      if (p.hwPt == 0 || !reg.isFiducial(p))
        continue;
      reco::Particle::PolarLorentzVector p4(
          p.floatPt(), reg.floatGlbEta(p.hwVtxEta()), reg.floatGlbPhi(p.hwVtxPhi()), 0.13f);
      l1t::PFCandidate::ParticleType type = l1t::PFCandidate::ChargedHadron;
      if (p.hwId.isMuon())
        type = l1t::PFCandidate::Muon;
      else if (p.hwId.isElectron())
        type = l1t::PFCandidate::Electron;
      ret->emplace_back(type, p.intCharge(), p4, 1, p.intPt(), p.intEta(), p.intPhi());
      ret->back().setZ0(p.floatZ0());
      ret->back().setDxy(p.floatDxy());
      ret->back().setHwZ0(p.hwZ0);
      ret->back().setHwDxy(p.hwDxy);
      ret->back().setHwTkQuality(p.hwTkQuality);
      ret->back().setCaloEta(reg.floatGlbEtaOf(p));
      ret->back().setCaloPhi(reg.floatGlbPhiOf(p));
      
      setRefs_(ret->back(), p);
    }
    for (const auto &p : event_.out[ir].pfneutral) {
      if (p.hwPt == 0 || !reg.isFiducial(p))
        continue;
      reco::Particle::PolarLorentzVector p4(p.floatPt(), reg.floatGlbEtaOf(p), reg.floatGlbPhiOf(p), 0.13f);
      l1t::PFCandidate::ParticleType type =
          p.hwId.isPhoton() ? l1t::PFCandidate::Photon : l1t::PFCandidate::NeutralHadron;
      ret->emplace_back(type, 0, p4, 1, p.intPt(), p.intEta(), p.intPhi());
      ret->back().setHwEmID(p.hwEmID);
      ret->back().setCaloEta(reg.floatGlbEtaOf(p));
      ret->back().setCaloPhi(reg.floatGlbPhiOf(p));
      setRefs_(ret->back(), p);
    }
  }
  return ret;
}

void L1TCorrelatorLayer1Producer::putPuppi(edm::Event &iEvent) const {
  auto refprod = iEvent.getRefBeforePut<l1t::PFCandidateCollection>("Puppi");
  auto coll = std::make_unique<l1t::PFCandidateCollection>();
  auto reg = std::make_unique<l1t::PFCandidateRegionalOutput>(refprod);
  std::vector<int> nobj;
  for (unsigned int ir = 0, nr = event_.pfinputs.size(); ir < nr; ++ir) {
    nobj.clear();
    for (const auto &p : event_.out[ir].puppi) {
      if (p.hwPt == 0)
        continue;
      // note: Puppi candidates are already in global coordinates & fiducial-only!
      l1t::PFCandidate::ParticleType type;
      float mass = 0.13f;
      if (p.hwId.charged()) {
        if (p.hwId.isMuon()) {
          type = l1t::PFCandidate::Muon;
          mass = 0.105;
        } else if (p.hwId.isElectron()) {
          type = l1t::PFCandidate::Electron;
          mass = 0.005;
        } else
          type = l1t::PFCandidate::ChargedHadron;
      } else {
        type = p.hwId.isPhoton() ? l1t::PFCandidate::Photon : l1t::PFCandidate::NeutralHadron;
        mass = p.hwId.isPhoton() ? 0.0 : 0.5;
      }
      reco::Particle::PolarLorentzVector p4(p.floatPt(), p.floatEta(), p.floatPhi(), mass);
      coll->emplace_back(type, p.intCharge(), p4, p.floatPuppiW(), p.intPt(), p.intEta(), p.intPhi());
      if (p.hwId.charged()) {
        coll->back().setZ0(p.floatZ0());
        coll->back().setDxy(p.floatDxy());
        coll->back().setHwZ0(p.hwZ0());
        coll->back().setHwDxy(p.hwDxy());
        coll->back().setHwTkQuality(p.hwTkQuality());
      } else {
        coll->back().setHwPuppiWeight(p.hwPuppiW());
        coll->back().setHwEmID(p.hwEmID());
      }
      coll->back().setEncodedPuppi64(p.pack().to_uint64());
      setRefs_(coll->back(), p);
      nobj.push_back(coll->size() - 1);
    }
    reg->addRegion(nobj, event_.pfinputs[ir].region.floatEtaCenter(), event_.pfinputs[ir].region.floatPhiCenter());
  }
  iEvent.put(std::move(coll), "Puppi");
  iEvent.put(std::move(reg), "PuppiRegional");
}

// NOTE: as a side effect we change the "sta_idx" of TkEle and TkEm objects to an index of the
// vector of refs, for this reason this is not const. We could make this more explicit via arguments
void L1TCorrelatorLayer1Producer::putEgStaObjects(edm::Event &iEvent,
                                                  const std::string &egLablel,
                                                  std::vector<edm::Ref<BXVector<l1t::EGamma>>> &egsta_refs) {
  auto egs = std::make_unique<BXVector<l1t::EGamma>>();
  edm::RefProd<BXVector<l1t::EGamma>> ref_egs = iEvent.getRefBeforePut<BXVector<l1t::EGamma>>(egLablel);

  edm::Ref<BXVector<l1t::EGamma>>::key_type idx = 0;
  // FIXME: in case more BXes are introduced shuld probably use egs->key(egs->end(bx));

  for (unsigned int ir = 0, nr = event_.pfinputs.size(); ir < nr; ++ir) {
    const auto &reg = event_.pfinputs[ir].region;

    std::vector<unsigned int> ref_pos(event_.out[ir].egsta.size());

    // EG standalone objects
    for (unsigned int ieg = 0, neg = event_.out[ir].egsta.size(); ieg < neg; ++ieg) {
      const auto &p = event_.out[ir].egsta[ieg];
      if (p.hwPt == 0 || !reg.isFiducial(p))
        continue;
      l1t::EGamma eg(
          reco::Candidate::PolarLorentzVector(p.floatPt(), reg.floatGlbEta(p.hwEta), reg.floatGlbPhi(p.hwPhi), 0.));
      eg.setHwQual(p.hwQual);
      egs->push_back(0, eg);
      egsta_refs.push_back(edm::Ref<BXVector<l1t::EGamma>>(ref_egs, idx++));
      ref_pos[ieg] = egsta_refs.size() - 1;
    }

    for (auto &egiso : event_.out[ir].egphoton) {
      if (egiso.hwPt == 0)
        continue;
      egiso.sta_idx = ref_pos[egiso.sta_idx];
    }

    for (auto &egele : event_.out[ir].egelectron) {
      if (egele.hwPt == 0)
        continue;
      egele.sta_idx = ref_pos[egele.sta_idx];
    }
  }

  iEvent.put(std::move(egs), egLablel);
}

void L1TCorrelatorLayer1Producer::putEgObjects(edm::Event &iEvent,
                                               const bool writeEgSta,
                                               const std::vector<edm::Ref<BXVector<l1t::EGamma>>> &egsta_refs,
                                               const std::string &tkEmLabel,
                                               const std::string &tkEmPerBoardLabel,
                                               const std::string &tkEleLabel,
                                               const std::string &tkElePerBoardLabel) const {
  auto tkems = std::make_unique<l1t::TkEmCollection>();
  auto tkemRefProd = iEvent.getRefBeforePut<l1t::TkEmCollection>(tkEmLabel);
  auto tkemPerBoard = std::make_unique<l1t::TkEmRegionalOutput>(tkemRefProd);
  auto tkeles = std::make_unique<l1t::TkElectronCollection>();
  auto tkeleRefProd = iEvent.getRefBeforePut<l1t::TkElectronCollection>(tkEleLabel);
  auto tkelePerBoard = std::make_unique<l1t::TkElectronRegionalOutput>(tkeleRefProd);

  // TkEG objects are written out after the per-board sorting.
  // The mapping to each board is saved into the regionalmap for further (stage-2 consumption)
  std::vector<int> nele_obj;
  std::vector<int> npho_obj;

  for (const auto &board : event_.board_out) {
    npho_obj.clear();
    for (const auto &egiso : board.egphoton) {
      if (egiso.hwPt == 0)
        continue;

      edm::Ref<BXVector<l1t::EGamma>> ref_egsta;
      if (writeEgSta) {
        ref_egsta = egsta_refs[egiso.sta_idx];
      } else {
        auto egptr = egiso.srcCluster->constituentsAndFractions()[0].first;
        ref_egsta =
            edm::Ref<BXVector<l1t::EGamma>>(egptr.id(), dynamic_cast<const l1t::EGamma *>(egptr.get()), egptr.key());
      }

      reco::Candidate::PolarLorentzVector mom(egiso.floatPt(), egiso.floatEta(), egiso.floatPhi(), 0.);

      l1t::TkEm tkem(reco::Candidate::LorentzVector(mom),
                     ref_egsta,
                     egiso.floatRelIso(l1ct::EGIsoObjEmu::IsoType::TkIso),
                     egiso.floatRelIso(l1ct::EGIsoObjEmu::IsoType::TkIsoPV));
      tkem.setHwQual(egiso.hwQual);
      tkem.setPFIsol(egiso.floatRelIso(l1ct::EGIsoObjEmu::IsoType::PfIso));
      tkem.setPFIsolPV(egiso.floatRelIso(l1ct::EGIsoObjEmu::IsoType::PfIsoPV));
      tkem.setEgBinaryWord(egiso.pack());
      tkems->push_back(tkem);
      npho_obj.push_back(tkems->size() - 1);
    }
    tkemPerBoard->addRegion(npho_obj, board.eta, board.phi);

    nele_obj.clear();
    for (const auto &egele : board.egelectron) {
      if (egele.hwPt == 0)
        continue;

      edm::Ref<BXVector<l1t::EGamma>> ref_egsta;
      if (writeEgSta) {
        ref_egsta = egsta_refs[egele.sta_idx];
      } else {
        auto egptr = egele.srcCluster->constituentsAndFractions()[0].first;
        ref_egsta =
            edm::Ref<BXVector<l1t::EGamma>>(egptr.id(), dynamic_cast<const l1t::EGamma *>(egptr.get()), egptr.key());
      }

      reco::Candidate::PolarLorentzVector mom(egele.floatPt(), egele.floatEta(), egele.floatPhi(), 0.);

      l1t::TkElectron tkele(reco::Candidate::LorentzVector(mom),
                            ref_egsta,
                            edm::refToPtr(egele.srcTrack->track()),
                            egele.floatRelIso(l1ct::EGIsoEleObjEmu::IsoType::TkIso));
      tkele.setHwQual(egele.hwQual);
      tkele.setPFIsol(egele.floatRelIso(l1ct::EGIsoEleObjEmu::IsoType::PfIso));
      tkele.setEgBinaryWord(egele.pack());
      tkele.setCompositeBdtScore(egele.bdtScore);
      tkeles->push_back(tkele);
      nele_obj.push_back(tkeles->size() - 1);
    }
    tkelePerBoard->addRegion(nele_obj, board.eta, board.phi);
  }

  iEvent.put(std::move(tkems), tkEmLabel);
  iEvent.put(std::move(tkemPerBoard), tkEmPerBoardLabel);
  iEvent.put(std::move(tkeles), tkEleLabel);
  iEvent.put(std::move(tkelePerBoard), tkElePerBoardLabel);
}

std::unique_ptr<std::vector<unsigned>> L1TCorrelatorLayer1Producer::vecSecInput(InputType t) const {
  auto v = std::make_unique<std::vector<unsigned>>();
  {
    switch (t) {
      case caloType:
        for (const auto &s : event_.decoded.hadcalo)
          v->push_back(s.size());
        break;
      case emcaloType:
        for (const auto &s : event_.decoded.emcalo)
          v->push_back(s.size());
        break;
      case trackType:
        for (const auto &s : event_.decoded.track)
          v->push_back(s.size());
        break;
      case l1muType:
        v->push_back(event_.decoded.muon.size());
        break;
    }
  }
  return v;
}

std::unique_ptr<std::vector<unsigned>> L1TCorrelatorLayer1Producer::vecRegInput(InputType t) const {
  auto v = std::make_unique<std::vector<unsigned>>();
  for (const auto &reg : event_.pfinputs) {
    switch (t) {
      case caloType:
        v->push_back(reg.hadcalo.size());
        break;
      case emcaloType:
        v->push_back(reg.emcalo.size());
        break;
      case trackType:
        v->push_back(reg.track.size());
        break;
      case l1muType:
        v->push_back(reg.muon.size());
        break;
    }
  }
  return v;
}

std::unique_ptr<std::vector<unsigned>> L1TCorrelatorLayer1Producer::vecOutput(OutputType i, bool usePuppi) const {
  auto v = std::make_unique<std::vector<unsigned>>();
  for (const auto &reg : event_.out) {
    v->push_back(reg.nObj(i, usePuppi));
  }
  return v;
}
std::pair<unsigned int, unsigned int> L1TCorrelatorLayer1Producer::totAndMax(
    const std::vector<unsigned> &perRegion) const {
  unsigned int ntot = 0, nmax = 0;
  for (unsigned ni : perRegion) {
    ntot += ni;
    nmax = std::max(nmax, ni);
  }
  return std::make_pair(ntot, nmax);
}
//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TCorrelatorLayer1Producer);
