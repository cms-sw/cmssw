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
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"

#include "DataFormats/Common/interface/RefToPtr.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "DataFormats/L1TParticleFlow/interface/PFCluster.h"
#include "DataFormats/L1Trigger/interface/Vertex.h"
#include "DataFormats/L1Trigger/interface/VertexWord.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "L1Trigger/Phase2L1ParticleFlow/interface/l1-converters/tkinput_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/l1-converters/muonGmtToL1ct_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/l1-converters/hgcalinput_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/l1-converters/gcteminput_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/l1-converters/gcthadinput_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/regionizer_base_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/multififo_regionizer_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/buffered_folded_multififo_regionizer_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/middle_buffer_multififo_regionizer_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/tdr_regionizer_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/pf/pfalgo2hgc_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/pf/pfalgo3_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/pf/pfalgo_dummy_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/puppi/linpuppi_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/egamma/pftkegalgo_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/pf/pfalgo_common_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/egamma/pftkegsorter_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/egamma/pftkegsorter_barrel_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/L1TCorrelatorLayer1PatternFileWriter.h"

#include "DataFormats/L1TCorrelator/interface/TkElectron.h"
#include "DataFormats/L1TCorrelator/interface/TkElectronFwd.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1TCorrelator/interface/TkEm.h"
#include "DataFormats/L1TCorrelator/interface/TkEmFwd.h"

#include "DataFormats/L1TCalorimeterPhase2/interface/GCTHadDigiCluster.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/GCTEmDigiCluster.h"
// #include "DataFormats/L1TCalorimeterPhase2/interface/CaloCrystalCluster.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/DigitizedClusterCorrelator.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"

// constexpr unsigned int calomapping[] = {3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8};
constexpr unsigned int calomapping[] = {9, 6, 3, 0, 10, 7, 4, 1, 11, 8, 5, 2};

//--------------------------------------------------------------------------------------------------
class L1TCorrelatorLayer1Producer : public edm::stream::EDProducer<> {
public:
  explicit L1TCorrelatorLayer1Producer(const edm::ParameterSet &);
  ~L1TCorrelatorLayer1Producer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

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

  // For calo, can give either the already converted containers, or the raw containers (for GCT only)
  // These are the already converted containers
  edm::EDGetTokenT<l1t::PFClusterCollection> hadGCTCands_;
  edm::EDGetTokenT<l1t::HGCalMulticlusterBxCollection> hadHGCalCands_;

  // can alternately give the raw containers (for GCT)
  edm::EDGetTokenT<l1tp2::GCTEmDigiClusterCollection> emGCTRawCands_;
  edm::EDGetTokenT<l1tp2::GCTHadDigiClusterCollection> hadGCTRawCands_;

  float emPtCut_, hadPtCut_;

  l1ct::Event event_;
  std::unique_ptr<l1ct::TrackInputEmulator> trackInput_;
  std::unique_ptr<l1ct::GMTMuonDecoderEmulator> muonInput_;
  std::unique_ptr<l1ct::HgcalClusterDecoderEmulator> hgcalInput_;
  std::unique_ptr<l1ct::GctHadClusterDecoderEmulator> gctHadInput_;
  std::unique_ptr<l1ct::GctEmClusterDecoderEmulator> gctEmInput_;
  std::unique_ptr<l1ct::RegionizerEmulator> regionizer_;
  std::unique_ptr<l1ct::PFAlgoEmulatorBase> l1pfalgo_;
  std::unique_ptr<l1ct::LinPuppiEmulator> l1pualgo_;
  std::unique_ptr<l1ct::PFTkEGAlgoEmulator> l1tkegalgo_;
  std::unique_ptr<l1ct::PFTkEGSorterEmulator> l1tkegsorter_;

  // Region dump
  const std::string regionDumpName_;
  std::fstream fRegionDump_;
  const edm::VParameterSet patternWriterConfigs_;
  std::vector<std::unique_ptr<L1TCorrelatorLayer1PatternFileWriter>> patternWriters_;

  // region of interest debugging
  float debugEta_, debugPhi_, debugR_;

  // these are used to link items back
  std::unordered_map<const l1t::L1Candidate *, edm::Ptr<l1t::L1Candidate>> clusterRefMap_;
  std::unordered_map<const l1t::PFTrack *, l1t::PFTrackRef> trackRefMap_;
  std::unordered_map<const l1t::SAMuon *, l1t::PFCandidate::MuonRef> muonRefMap_;

  // main methods
  void beginStream(edm::StreamID) override;
  void endStream() override;
  void produce(edm::Event &, const edm::EventSetup &) override;
  void encodeAndAddHgcalCluster(ap_uint<256> &word,
                                l1ct::DetectorSector<ap_uint<256>> &sec,
                                const l1t::HGCalMulticluster &calo) const;
  void getDecodedGCTPFCluster(l1ct::HadCaloObjEmu &calo,
                              l1ct::DetectorSector<l1ct::HadCaloObjEmu> &sec,
                              const l1t::PFCluster &cluster) const;
  void addDecodedEmCalo(l1ct::EmCaloObjEmu &decCalo,
                        const edm::Ptr<l1t::L1Candidate> &caloPtr,
                        l1ct::DetectorSector<l1ct::EmCaloObjEmu> &sec);
  void addEmPFCluster(const l1ct::EmCaloObjEmu &decCalo,
                      const l1ct::PFRegionEmu &region,
                      std::unique_ptr<l1t::PFClusterCollection> &pfClusters) const;
  void addDecodedHadCalo(l1ct::HadCaloObjEmu &decCalo,
                         const edm::Ptr<l1t::L1Candidate> &caloPtr,
                         l1ct::DetectorSector<l1ct::HadCaloObjEmu> &sec);

  void addHadPFCluster(const l1ct::HadCaloObjEmu &decCalo,
                       const l1ct::PFRegionEmu &region,
                       std::unique_ptr<l1t::PFClusterCollection> &pfClusters) const;

  void addUInt(unsigned int value, std::string iLabel, edm::Event &iEvent);

  void initSectorsAndRegions(const edm::ParameterSet &iConfig);
  void initEvent(const edm::Event &e);
  // add object, tracking references
  void addTrack(const l1t::PFTrack &t, l1t::PFTrackRef ref);
  void addMuon(const l1t::SAMuon &t, l1t::PFCandidate::MuonRef ref);
  // HGCAl input clusters
  void addHGCalHadCalo(const l1t::HGCalMulticluster &calo, const edm::Ptr<l1t::L1Candidate> &caloPtr);
  // GCT input clusters
  void addGCTHadCalo(const l1t::PFCluster &calo, const edm::Ptr<l1t::L1Candidate> &caloPtr);
  // for GCT raw calos as input
  void addGCTEmCaloRaw(const l1tp2::GCTEmDigiClusterLink &link, unsigned int linkidx, unsigned int entidx);
  void addGCTHadCaloRaw(const l1tp2::GCTHadDigiClusterLink &link, unsigned int linkidx, unsigned int entidx);
  // add objects in already-decoded format
  void addDecodedTrack(l1ct::DetectorSector<l1ct::TkObjEmu> &sec, const l1t::PFTrack &t);
  void addDecodedMuon(l1ct::DetectorSector<l1ct::MuObjEmu> &sec, const l1t::SAMuon &t);
  void addDecodedGCTEmCalo(l1ct::DetectorSector<l1ct::EmCaloObjEmu> &sec, const l1tp2::GCTEmDigiCluster &digi);
  void addDecodedGCTHadCalo(l1ct::DetectorSector<l1ct::HadCaloObjEmu> &sec, const l1tp2::GCTHadDigiCluster &digi);

  void rawHgcalClusterEncode(ap_uint<256> &cwrd,
                             const l1ct::DetectorSector<ap_uint<256>> &sec,
                             const l1t::HGCalMulticluster &c) const;

  // fetching outputs
  std::unique_ptr<l1t::PFCandidateCollection> fetchHadCalo() const;
  std::unique_ptr<l1t::PFCandidateCollection> fetchEmCalo() const;
  std::unique_ptr<l1t::PFCandidateCollection> fetchTracks() const;
  std::unique_ptr<l1t::PFCandidateCollection> fetchPF() const;
  std::unique_ptr<l1t::PFClusterCollection> fetchDecodedHadCalo() const;
  std::unique_ptr<l1t::PFClusterCollection> fetchDecodedEmCalo() const;
  std::unique_ptr<l1t::PFTrackCollection> fetchDecodedTracks() const;
  void putPuppi(edm::Event &iEvent) const;

  void putEgStaObjects(edm::Event &iEvent, const std::string &egLablel) const;
  void putEgObjects(edm::Event &iEvent,
                    const bool writeEgSta,
                    const std::string &tkEmLabel,
                    const std::string &tkEmPerBoardLabel,
                    const std::string &tkEleLabel,
                    const std::string &tkElePerBoardLabel) const;

  template <typename T>
  void setRefs_(l1t::PFCandidate &pf, const T &p) const;
  template <typename T>
  void setRefs_(l1t::PFCluster &pf, const T &p) const;
  template <typename Tm, typename Tk, typename To>
  auto findRef_(const Tm &map, const Tk *key, const To &obj) const {
    auto match = map.find(key);
    if (match == map.end()) {
      throw cms::Exception("CorruptData") << refExcepMsg_(obj);
    }
    return match->second;
  }
  template <typename T>
  std::string refExcepMsg_(const T &key) const;

  void doVertexings(std::vector<float> &pvdz) const;
  // for multiplicities
  enum InputType { caloType = 0, emcaloType = 1, trackType = 2, l1muType = 3 };
  static constexpr const char *inputTypeName[l1muType + 1] = {"Calo", "EmCalo", "TK", "Mu"};
  std::unique_ptr<std::vector<unsigned>> vecSecInput(InputType i) const;
  std::unique_ptr<std::vector<unsigned>> vecRegInput(InputType i) const;
  typedef l1ct::OutputRegion::ObjType OutputType;
  std::unique_ptr<std::vector<unsigned>> vecOutput(OutputType i, bool usePuppi) const;
  std::pair<unsigned int, unsigned int> totAndMax(const std::vector<unsigned> &perRegion) const;

  // utilities
  template <typename T>
  static edm::ParameterDescription<edm::ParameterSetDescription> getParDesc(const std::string &name) {
    return edm::ParameterDescription<edm::ParameterSetDescription>(
        name + "Parameters", T::getParameterSetDescription(), true);
  }
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
      regionDumpName_(iConfig.getUntrackedParameter<std::string>("dumpFileName")),
      patternWriterConfigs_(iConfig.getUntrackedParameter<edm::VParameterSet>("patternWriters")),
      debugEta_(iConfig.getUntrackedParameter<double>("debugEta")),
      debugPhi_(iConfig.getUntrackedParameter<double>("debugPhi")),
      debugR_(iConfig.getUntrackedParameter<double>("debugR")) {
  produces<l1t::PFCandidateCollection>("PF");
  produces<l1t::PFCandidateCollection>("Puppi");
  produces<l1t::PFCandidateRegionalOutput>("PuppiRegional");

  produces<l1t::PFCandidateCollection>("EmCalo");
  produces<l1t::PFCandidateCollection>("Calo");
  produces<l1t::PFCandidateCollection>("TK");
#if 0  // LATER
  produces<l1t::PFCandidateCollection>("TKVtx");
#endif
  produces<l1t::PFTrackCollection>("DecodedTK");
  produces<l1t::PFClusterCollection>("DecodedEmClusters");
  produces<l1t::PFClusterCollection>("DecodedHadClusters");

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

  const std::string &hgcalInAlgo = iConfig.getParameter<std::string>("hgcalInputConversionAlgo");
  const edm::InputTag hadClusters = iConfig.getParameter<edm::InputTag>("hadClusters");
  if (!hadClusters.label().empty()) {
    if (hgcalInAlgo == "Emulator") {
      hadHGCalCands_ = consumes<l1t::HGCalMulticlusterBxCollection>(hadClusters);
      hgcalInput_ = std::make_unique<l1ct::HgcalClusterDecoderEmulator>(
          iConfig.getParameter<edm::ParameterSet>("hgcalInputConversionParameters"));
    } else if (hgcalInAlgo != "None")
      throw cms::Exception("Configuration", "Unsupported hgcalInputConversionAlgo");
  }
  const std::string &gctEmInAlgo = iConfig.getParameter<std::string>("gctEmInputConversionAlgo");
  const edm::InputTag emClusters = iConfig.getParameter<edm::InputTag>("emClusters");
  if (!emClusters.label().empty()) {
    if (gctEmInAlgo == "Emulator") {
      gctEmInput_ = std::make_unique<l1ct::GctEmClusterDecoderEmulator>(
          iConfig.getParameter<edm::ParameterSet>("gctEmInputConversionParameters"));
      emGCTRawCands_ = consumes<l1tp2::GCTEmDigiClusterCollection>(emClusters);
    } else if (gctEmInAlgo != "None")
      throw cms::Exception("Configuration", "Unsupported gctEmInputConversionAlgo");
  }
  const std::string &gctHadInAlgo = iConfig.getParameter<std::string>("gctHadInputConversionAlgo");
  if (!hadClusters.label().empty()) {
    if (gctHadInAlgo == "Emulator") {
      gctHadInput_ = std::make_unique<l1ct::GctHadClusterDecoderEmulator>(
          iConfig.getParameter<edm::ParameterSet>("gctHadInputConversionParameters"));
      hadGCTRawCands_ = consumes<l1tp2::GCTHadDigiClusterCollection>(hadClusters);
    } else if (gctHadInAlgo == "Ideal") {
      hadGCTCands_ = consumes<l1t::PFClusterCollection>(hadClusters);
    } else if (gctHadInAlgo != "None")
      throw cms::Exception("Configuration", "Unsupported gctHadInputConversionAlgo");
  }

  const std::string &regalgo = iConfig.getParameter<std::string>("regionizerAlgo");
  if (regalgo == "Ideal") {
    regionizer_ =
        std::make_unique<l1ct::RegionizerEmulator>(iConfig.getParameter<edm::ParameterSet>("regionizerAlgoParameters"));
  } else if (regalgo == "Multififo") {
    regionizer_ = std::make_unique<l1ct::MultififoRegionizerEmulator>(
        iConfig.getParameter<edm::ParameterSet>("regionizerAlgoParameters"));
  } else if (regalgo == "BufferedFoldedMultififo") {
    regionizer_ = std::make_unique<l1ct::BufferedFoldedMultififoRegionizerEmulator>(
        iConfig.getParameter<edm::ParameterSet>("regionizerAlgoParameters"));
  } else if (regalgo == "MultififoBarrel") {
    const auto &pset = iConfig.getParameter<edm::ParameterSet>("regionizerAlgoParameters");
    regionizer_ =
        std::make_unique<l1ct::MultififoRegionizerEmulator>(pset.getParameter<std::string>("barrelSetup"), pset);
  } else if (regalgo == "MiddleBufferMultififo") {
    regionizer_ = std::make_unique<l1ct::MiddleBufferMultififoRegionizerEmulator>(
        iConfig.getParameter<edm::ParameterSet>("regionizerAlgoParameters"));
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

  const std::string &egsortalgo = iConfig.getParameter<std::string>("tkEgSorterAlgo");
  if (egsortalgo == "Barrel") {
    l1tkegsorter_ = std::make_unique<l1ct::PFTkEGSorterBarrelEmulator>(
        iConfig.getParameter<edm::ParameterSet>("tkEgSorterParameters"));
  } else if (egsortalgo == "Endcap") {
    l1tkegsorter_ =
        std::make_unique<l1ct::PFTkEGSorterEmulator>(iConfig.getParameter<edm::ParameterSet>("tkEgSorterParameters"));
  } else
    throw cms::Exception("Configuration", "Unsupported tkEgSorterAlgo");

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

  produces<std::vector<l1t::PFCluster>>("decodedHadPFClusters");
  produces<std::vector<l1t::PFCluster>>("decodedEmPFClusters");

  initSectorsAndRegions(iConfig);
}

L1TCorrelatorLayer1Producer::~L1TCorrelatorLayer1Producer() {}

void L1TCorrelatorLayer1Producer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  // Inputs and cuts
  desc.add<edm::InputTag>("tracks", edm::InputTag(""));
  desc.add<edm::InputTag>("muons", edm::InputTag("l1tSAMuonsGmt", "prompt"));
  desc.add<edm::InputTag>("emClusters", edm::InputTag(""));
  desc.add<edm::InputTag>("hadClusters", edm::InputTag(""));
  desc.add<edm::InputTag>("vtxCollection", edm::InputTag("l1tVertexFinderEmulator", "L1VerticesEmulation"));
  desc.add<bool>("vtxCollectionEmulation", true);
  desc.add<double>("emPtCut", 0.0);
  desc.add<double>("hadPtCut", 0.0);
  desc.add<double>("trkPtCut", 0.0);
  desc.add<int32_t>("nVtx");
  // Input conversion
  edm::EmptyGroupDescription emptyGroup;
  desc.ifValue(edm::ParameterDescription<std::string>("trackInputConversionAlgo", "Ideal", true),
               "Ideal" >> emptyGroup or "Emulator" >> getParDesc<l1ct::TrackInputEmulator>("trackInputConversion"));
  desc.ifValue(edm::ParameterDescription<std::string>("muonInputConversionAlgo", "Ideal", true),
               "Ideal" >> emptyGroup or "Emulator" >> getParDesc<l1ct::GMTMuonDecoderEmulator>("muonInputConversion"));
  desc.ifValue(
      edm::ParameterDescription<std::string>("hgcalInputConversionAlgo", "None", true),
      "None" >> emptyGroup or "Emulator" >> getParDesc<l1ct::HgcalClusterDecoderEmulator>("hgcalInputConversion"));
  desc.ifValue(
      edm::ParameterDescription<std::string>("gctEmInputConversionAlgo", "None", true),
      "None" >> emptyGroup or "Emulator" >> getParDesc<l1ct::GctEmClusterDecoderEmulator>("gctEmInputConversion"));
  desc.ifValue(edm::ParameterDescription<std::string>("gctHadInputConversionAlgo", "None", true),
               "Ideal" >> emptyGroup or "None" >> emptyGroup or
                   "Emulator" >> getParDesc<l1ct::GctHadClusterDecoderEmulator>("gctHadInputConversion"));
  // Regionizer
  auto idealRegPD = getParDesc<l1ct::RegionizerEmulator>("regionizerAlgo");
  auto tdrRegPD = getParDesc<l1ct::TDRRegionizerEmulator>("regionizerAlgo");
  auto multififoRegPD = getParDesc<l1ct::MultififoRegionizerEmulator>("regionizerAlgo");
  auto bfMultififoRegPD = getParDesc<l1ct::BufferedFoldedMultififoRegionizerEmulator>("regionizerAlgo");
  auto multififoBarrelRegPD = edm::ParameterDescription<edm::ParameterSetDescription>(
      "regionizerAlgoParameters", l1ct::MultififoRegionizerEmulator::getParameterSetDescriptionBarrel(), true);
  auto mbMultififoRegPD = getParDesc<l1ct::MiddleBufferMultififoRegionizerEmulator>("regionizerAlgo");
  desc.ifValue(edm::ParameterDescription<std::string>("regionizerAlgo", "Ideal", true),
               "Ideal" >> idealRegPD or "TDR" >> tdrRegPD or "Multififo" >> multififoRegPD or
                   "BufferedFoldedMultififo" >> bfMultififoRegPD or "MultififoBarrel" >> multififoBarrelRegPD or
                   "MiddleBufferMultififo" >> mbMultififoRegPD);
  // PF
  desc.ifValue(edm::ParameterDescription<std::string>("pfAlgo", "PFAlgo3", true),
               "PFAlgo3" >> getParDesc<l1ct::PFAlgo3Emulator>("pfAlgo") or
                   "PFAlgo2HGC" >> getParDesc<l1ct::PFAlgo2HGCEmulator>("pfAlgo") or
                   "PFAlgoDummy" >> getParDesc<l1ct::PFAlgoDummyEmulator>("pfAlgo"));
  // Puppi
  desc.ifValue(edm::ParameterDescription<std::string>("puAlgo", "LinearizedPuppi", true),
               "LinearizedPuppi" >> getParDesc<l1ct::LinPuppiEmulator>("puAlgo"));
  // EGamma
  desc.add<edm::ParameterSetDescription>("tkEgAlgoParameters", l1ct::PFTkEGAlgoEmuConfig::getParameterSetDescription());
  // EGamma sort
  desc.ifValue(edm::ParameterDescription<std::string>("tkEgSorterAlgo", "Barrel", true),
               "Barrel" >> getParDesc<l1ct::PFTkEGSorterBarrelEmulator>("tkEgSorter") or
                   "Endcap" >> getParDesc<l1ct::PFTkEGSorterEmulator>("tkEgSorter"));
  // geometry: calo sectors
  edm::ParameterSetDescription caloSectorPSD;
  caloSectorPSD.add<std::vector<double>>("etaBoundaries");
  caloSectorPSD.add<uint32_t>("phiSlices", 3);
  caloSectorPSD.add<double>("phiZero", 0.);
  desc.addVPSet("caloSectors", caloSectorPSD);
  // geometry: regions
  edm::ParameterSetDescription regionPSD;
  regionPSD.add<std::vector<double>>("etaBoundaries");
  regionPSD.add<uint32_t>("phiSlices", 9);
  regionPSD.add<double>("etaExtra", 0.25);
  regionPSD.add<double>("phiExtra", 0.25);
  desc.addVPSet("regions", regionPSD);
  // geometry: boards
  edm::ParameterSetDescription boardPSD;
  boardPSD.add<std::vector<unsigned int>>("regions");
  desc.addVPSet("boards", boardPSD);
  // dump files
  desc.addUntracked<std::string>("dumpFileName", "");
  // pattern files
  desc.addVPSetUntracked(
      "patternWriters", L1TCorrelatorLayer1PatternFileWriter::getParameterSetDescription(), edm::VParameterSet());
  // debug
  desc.addUntracked<double>("debugEta", 0.);
  desc.addUntracked<double>("debugPhi", 0.);
  desc.addUntracked<double>("debugR", -1.);
  descriptions.add("l1tCorrelatorLayer1", desc);
}

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

  // this is for parsing raw calo information
  if (!emGCTRawCands_.isUninitialized()) {
    auto caloHandle = iEvent.getHandle(emGCTRawCands_);
    const auto &links = *caloHandle;
    for (unsigned int ic = 0; ic < links.size(); ++ic) {
      const auto &link = links[ic];
      for (unsigned int ie = 0; ie < link.size(); ++ie) {
        addGCTEmCaloRaw(link, ic, ie);
      }
    }
  }

  if (!hadHGCalCands_.isUninitialized()) {
    auto caloHandle = iEvent.getHandle(hadHGCalCands_);
    if (caloHandle.isValid()) {
      const auto &calos = *caloHandle;
      for (unsigned int ic = 0, nc = calos.size(); ic < nc; ++ic) {
        const auto &calo = calos[ic];
        addHGCalHadCalo(calo, edm::Ptr<l1t::L1Candidate>(caloHandle, ic));
      }
    }
  }

  if (!hadGCTRawCands_.isUninitialized()) {
    auto caloHandle = iEvent.getHandle(hadGCTRawCands_);
    const auto &links = *caloHandle;
    for (unsigned int ic = 0; ic < links.size(); ++ic) {
      const auto &link = links[ic];
      for (unsigned int ie = 0; ie < link.size(); ++ie) {
        addGCTHadCaloRaw(link, ic, ie);
      }
    }
  }

  if (!hadGCTCands_.isUninitialized()) {
    auto caloHandle = iEvent.getHandle(hadGCTCands_);
    if (caloHandle.isValid()) {
      const auto &calos = *caloHandle;
      for (unsigned int ic = 0, nc = calos.size(); ic < nc; ++ic) {
        const auto &calo = calos[ic];
        addGCTHadCalo(calo, edm::Ptr<l1t::L1Candidate>(caloHandle, ic));
      }
    }
  }

  regionizer_->run(event_.decoded, event_.pfinputs);

  // First, get a copy of the discretized and corrected inputs, and write them out
  // FIXME: steer via config flag
  iEvent.put(fetchEmCalo(), "EmCalo");
  iEvent.put(fetchHadCalo(), "Calo");
  iEvent.put(fetchTracks(), "TK");

  iEvent.put(fetchDecodedHadCalo(), "DecodedHadClusters");
  iEvent.put(fetchDecodedEmCalo(), "DecodedEmClusters");
  iEvent.put(fetchDecodedTracks(), "DecodedTK");

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
  //get additional vertices if requested
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
    putEgStaObjects(iEvent, "L1Eg");
  }

  // l1tkegsorter_->setDebug(true);
  for (auto &board : event_.board_out) {
    l1tkegsorter_->runPho(event_.pfinputs, event_.out, board.region_index, board.egphoton);
    l1tkegsorter_->runEle(event_.pfinputs, event_.out, board.region_index, board.egelectron);
  }

  // save PF into the event
  iEvent.put(fetchPF(), "PF");

  // and save puppi
  putPuppi(iEvent);

  // save the EG objects
  putEgObjects(iEvent, l1tkegalgo_->writeEgSta(), "L1TkEm", "L1TkEmPerBoard", "L1TkEle", "L1TkElePerBoard");

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

void L1TCorrelatorLayer1Producer::rawHgcalClusterEncode(ap_uint<256> &cwrd,
                                                        const l1ct::DetectorSector<ap_uint<256>> &sec,
                                                        const l1t::HGCalMulticluster &c) const {
  cwrd = 0;

  // implemented as of interface document (version of 15/11/2025)
  ap_ufixed<14, 12, AP_RND_CONV, AP_SAT> w_pt = c.pt();
  // NOTE: We use iPt here for now despite final HGC FW implementation might be different
  ap_ufixed<14, 12, AP_RND_CONV, AP_SAT> w_empt = c.iPt(l1t::HGCalMulticluster::EnergyInterpretation::EM);

  // NOTE: this number is not consistent with the iPt interpretation nor with hoe.
  // hoe uses the total cluster em and had energies while eot computed in the showershapes only accounts for a
  // a max radius from the cluster center. This is the value used by the ID models
  ap_uint<8> w_emfrac = std::min(round(c.eot() * 256), float(255.));

  // NOTE II: we compute a second eot value to propagate the value of the total hadronic energy as in
  // the hoe computation
  float em_frac_tot = c.hOverE() < 0 ? 0. : 1. / (c.hOverE() + 1.);
  ap_uint<8> w_emfrac_tot = std::min(round(em_frac_tot * 256), float(255.));

  static constexpr float ETAPHI_LSB = M_PI / 720;
  static constexpr float SIGMAZZ_LSB = 778.098 / (1 << 7);
  static constexpr float SIGMAPHIPHI_LSB = 0.12822 / (1 << 7);
  static constexpr float SIGMAETAETA_LSB = 0.148922 / (1 << 5);

  ap_uint<10> w_eta = round(fabs(c.eta()) / ETAPHI_LSB);
  ap_int<9> w_phi = round(sec.region.localPhi(c.phi()) / ETAPHI_LSB);
  // FIXME: we keep subtracting an arbitrary number different from the HLGCal FW one
  ap_ufixed<12, 11, AP_RND_CONV, AP_SAT> w_meanz = fabs(c.zBarycenter()) - 320;  // LSB = 0.5cm

  ap_uint<6> w_showerlenght = c.showerLength();
  ap_uint<7> w_sigmazz = round(c.sigmaZZ() / SIGMAZZ_LSB);
  ap_uint<7> w_sigmaphiphi = round(c.sigmaPhiPhiTot() / SIGMAPHIPHI_LSB);
  ap_uint<6> w_coreshowerlenght = c.coreShowerLength();
  ap_uint<5> w_sigmaetaeta = round(c.sigmaEtaEtaTot() / SIGMAETAETA_LSB);
  // NOTE: this is an arbitrary choice to keep the rounding consistent with the "addDecodedHadCalo" one
  // FIXME: the scaling here is added to the encoded word...
  ap_uint<13> w_sigmarrtot = round(c.sigmaRRTot() * l1ct::Scales::SRRTOT_SCALE / l1ct::Scales::SRRTOT_LSB);

  // Word 0
  cwrd(13, 0) = w_pt.range();     // 14 bits: 13-0
  cwrd(27, 14) = w_empt.range();  // 14 bits: 27-14
  cwrd(39, 32) = w_emfrac_tot;    //  8 bits: 39-32
  cwrd(47, 40) = w_emfrac;        //  8 bits: 47-40

  // Word 1
  cwrd(64 + 9, 64 + 0) = w_eta;              // 10 bits: 9-0
  cwrd(64 + 18, 64 + 10) = w_phi;            //  9 bits: 18-10
  cwrd(64 + 30, 64 + 19) = w_meanz.range();  // 12 bits: 30-19
  // Word 2
  cwrd(128 + 18, 128 + 13) = w_showerlenght;      //  6 bits: 18-13
  cwrd(128 + 38, 128 + 32) = w_sigmazz;           //  7 bits: 38-32
  cwrd(128 + 45, 128 + 39) = w_sigmaphiphi;       //  7 bits: 45-39
  cwrd(128 + 51, 128 + 46) = w_coreshowerlenght;  //  6 bits: 51-46
  cwrd(128 + 56, 128 + 52) = w_sigmaetaeta;       //  5 bits: 56-52

  // cwrd(128+63, 128+57) = w_sigmarrtot;       //  7 bits: 63-57 // FIXME: use word3 spare bits
  // Word 3
  cwrd(213, 201) = w_sigmarrtot;  // these are spare bits for now
}

void L1TCorrelatorLayer1Producer::encodeAndAddHgcalCluster(ap_uint<256> &word,
                                                           l1ct::DetectorSector<ap_uint<256>> &sec,
                                                           const l1t::HGCalMulticluster &calo) const {
  rawHgcalClusterEncode(word, sec, calo);
  sec.obj.push_back(word);
}

void L1TCorrelatorLayer1Producer::addEmPFCluster(const l1ct::EmCaloObjEmu &decCalo,
                                                 const l1ct::PFRegionEmu &region,
                                                 std::unique_ptr<l1t::PFClusterCollection> &pfClusters) const {
  // Crete the PFCluster and add the original object as Consitutent
  pfClusters->emplace_back(decCalo.floatPt(),
                           region.floatGlbEta(decCalo.hwEta),
                           region.floatGlbPhi(decCalo.hwPhi),
                           decCalo.floatHoe(),
                           true,
                           decCalo.floatPtErr(),
                           decCalo.intPt(),
                           decCalo.intEta(),
                           decCalo.intPhi());

  // Add additional variables specialized for GCT and HGCal clusters
  pfClusters->back().setHwQual(decCalo.hwEmID.to_int());
  pfClusters->back().setCaloDigi(decCalo);
  setRefs_(pfClusters->back(), decCalo);
}

void L1TCorrelatorLayer1Producer::addDecodedEmCalo(l1ct::EmCaloObjEmu &decCalo,
                                                   const edm::Ptr<l1t::L1Candidate> &caloPtr,
                                                   l1ct::DetectorSector<l1ct::EmCaloObjEmu> &sec) {
  clusterRefMap_[caloPtr.get()] = caloPtr;
  decCalo.src = caloPtr.get();
  if (decCalo.hwPt > 0) {  // NOTE: 0 pt clusters have null ptr. Do we need to keep them?
    // FIXME: for now we extract these from the upstream object since they are not yet available in the digi format
    const l1tp2::CaloCrystalCluster *crycl = dynamic_cast<const l1tp2::CaloCrystalCluster *>(decCalo.src);
    decCalo.hwShowerShape = l1ct::shower_shape_t(crycl->e2x5() / crycl->e5x5());
    decCalo.hwRelIso = l1ct::Scales::makeRelIso(crycl->isolation() / decCalo.hwPt.to_float());
  }
  sec.obj.push_back(decCalo);
}

void L1TCorrelatorLayer1Producer::addHadPFCluster(const l1ct::HadCaloObjEmu &decCalo,
                                                  const l1ct::PFRegionEmu &region,
                                                  std::unique_ptr<l1t::PFClusterCollection> &pfClusters) const {
  // Crete the PFCluster and add the original object as Consitutent
  pfClusters->emplace_back(decCalo.floatPt(),
                           region.floatGlbEta(decCalo.hwEta),
                           region.floatGlbPhi(decCalo.hwPhi),
                           decCalo.floatHoe(),
                           decCalo.hwIsEM(),
                           0.,  // ptError
                           decCalo.intPt(),
                           decCalo.intEta(),
                           decCalo.intPhi());

  // Add additional variables specialized for GCT and HGCal clusters
  pfClusters->back().setHwQual(decCalo.hwEmID.to_int());
  pfClusters->back().setCaloDigi(decCalo);
  setRefs_(pfClusters->back(), decCalo);
}

void L1TCorrelatorLayer1Producer::addDecodedHadCalo(l1ct::HadCaloObjEmu &decCalo,
                                                    const edm::Ptr<l1t::L1Candidate> &caloPtr,
                                                    l1ct::DetectorSector<l1ct::HadCaloObjEmu> &sec) {
  clusterRefMap_[caloPtr.get()] = caloPtr;
  decCalo.src = caloPtr.get();
  sec.obj.push_back(decCalo);
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
      float phiCenter = reco::reducePhiRange(iphi * TF_phiWidth);
      event_.decoded.track.emplace_back((ieta ? 0. : -2.5), (ieta ? 2.5 : 0.0), phiCenter, TF_phiWidth);
      event_.raw.track.emplace_back((ieta ? 0. : -2.5), (ieta ? 2.5 : 0.0), phiCenter, TF_phiWidth);
    }
  }

  event_.decoded.emcalo.clear();
  event_.decoded.hadcalo.clear();
  event_.raw.hgcalcluster.clear();
  event_.raw.gctEm.clear();
  event_.raw.gctHad.clear();

  for (const edm::ParameterSet &preg : iConfig.getParameter<edm::VParameterSet>("caloSectors")) {
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
        float phiCenter = reco::reducePhiRange(iphi * phiWidth + phiZero);
        event_.decoded.hadcalo.emplace_back(etaBoundaries[ieta], etaBoundaries[ieta + 1], phiCenter, phiWidth);
        event_.decoded.emcalo.emplace_back(etaBoundaries[ieta], etaBoundaries[ieta + 1], phiCenter, phiWidth);
        event_.raw.hgcalcluster.emplace_back(etaBoundaries[ieta], etaBoundaries[ieta + 1], phiCenter, phiWidth);
        event_.raw.gctEm.emplace_back(etaBoundaries[ieta], etaBoundaries[ieta + 1], phiCenter, phiWidth);
        event_.raw.gctHad.emplace_back(etaBoundaries[ieta], etaBoundaries[ieta + 1], phiCenter, phiWidth);
      }
    }
  }

  event_.decoded.muon.region = l1ct::PFRegionEmu(0., 0.);  // centered at (0,0)
  event_.raw.muon.region = l1ct::PFRegionEmu(0., 0.);      // centered at (0,0)

  event_.pfinputs.clear();
  for (const edm::ParameterSet &preg : iConfig.getParameter<edm::VParameterSet>("regions")) {
    std::vector<double> etaBoundaries = preg.getParameter<std::vector<double>>("etaBoundaries");
    if (!std::is_sorted(etaBoundaries.begin(), etaBoundaries.end()))
      throw cms::Exception("Configuration", "regions.etaBoundaries not sorted\n");
    unsigned int phiSlices = preg.getParameter<uint32_t>("phiSlices");
    float etaExtra = preg.getParameter<double>("etaExtra");
    float phiExtra = preg.getParameter<double>("phiExtra");
    float phiWidth = 2 * M_PI / phiSlices;
    for (unsigned int ieta = 0, neta = etaBoundaries.size() - 1; ieta < neta; ++ieta) {
      for (unsigned int iphi = 0; iphi < phiSlices; ++iphi) {
        float phiCenter = reco::reducePhiRange(iphi * phiWidth);  //align with L1 TrackFinder phi sector indexing
        event_.pfinputs.emplace_back(
            etaBoundaries[ieta], etaBoundaries[ieta + 1], phiCenter, phiWidth, etaExtra, phiExtra);
      }
    }
  }

  event_.board_out.clear();
  const edm::VParameterSet &board_params = iConfig.getParameter<edm::VParameterSet>("boards");
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

void L1TCorrelatorLayer1Producer::addGCTHadCalo(const l1t::PFCluster &calo, const edm::Ptr<l1t::L1Candidate> &caloPtr) {
  for (unsigned int isec = 0; isec < event_.decoded.hadcalo.size(); isec++) {
    auto &sec = event_.decoded.hadcalo[isec];
    // Get the raw and decoded sectors
    if (sec.region.contains(calo.eta(), calo.phi())) {
      l1ct::HadCaloObjEmu decCalo;
      getDecodedGCTPFCluster(decCalo, sec, calo);
      if (decCalo.floatPt() > hadPtCut_)
        addDecodedHadCalo(decCalo, caloPtr, sec);
    }
  }
}

void L1TCorrelatorLayer1Producer::getDecodedGCTPFCluster(l1ct::HadCaloObjEmu &calo,
                                                         l1ct::DetectorSector<l1ct::HadCaloObjEmu> &sec,
                                                         const l1t::PFCluster &cluster) const {
  calo.clear();
  calo.hwPt = l1ct::Scales::makePtFromFloat(cluster.pt());
  calo.hwEta = l1ct::Scales::makeGlbEta(cluster.eta()) -
               sec.region.hwEtaCenter;  // important to enforce that the region boundary is on a discrete value
  calo.hwPhi = l1ct::Scales::makePhi(sec.region.localPhi(cluster.phi()));
  calo.hwEmPt = l1ct::Scales::makePtFromFloat(cluster.emEt());
  calo.hwEmID = cluster.hwEmID();
  calo.hwHoe = l1ct::Scales::makeHoe(cluster.hOverE());
}

void L1TCorrelatorLayer1Producer::addHGCalHadCalo(const l1t::HGCalMulticluster &calo,
                                                  const edm::Ptr<l1t::L1Candidate> &caloPtr) {
  for (unsigned int isec = 0; isec < event_.decoded.hadcalo.size(); isec++) {
    auto &sec = event_.decoded.hadcalo[isec];
    // Get the raw and decoded sectors
    if (sec.region.contains(calo.eta(), calo.phi())) {
      auto &sec_raw = event_.raw.hgcalcluster[isec];
      // Get the raw word
      ap_uint<256> cwrd = 0;
      encodeAndAddHgcalCluster(cwrd, sec_raw, calo);
      // Crete the decoded object calling the unpacker
      // Use the valid flag to reject PU clusters when creating the decoded object
      bool valid = true;
      l1ct::HadCaloObjEmu decCalo = hgcalInput_->decode(sec_raw.region, cwrd, valid);
      if (decCalo.floatPt() > hadPtCut_ && valid)
        addDecodedHadCalo(decCalo, caloPtr, sec);
    }
  }
}

// regions order:  GCT1 SLR1, GCT1 SLR3, GCT2 SLR1, GCT2 SLR3, GCT3 SLR1, GCT3SLR3
// always + then - eta for each region

// I don't think above is right. I think this is the mapping
// from reg. ord:  GCT1 SLR1+, GCT1 SLR1-, GCT1 SLR3+, GCT1 SLR3-, GCT2 SLR1+, GCT2 SLR1-,
//                 GCT2 SLR3+, GCT2 SLR3-, GCT3 SLR1+, GCT3 SLR1-, GCT3 SLR3+, GCT3 SLR3-
// to reg. order:  GCT1 SLR3-, GCT2 SLR3-, GCT3 SLR3-, GCT1 SLR3+, GCT2 SLR3+, GCT3 SLR3+,
//                 GCT1 SLR1-, GCT2 SLR1-, GCT3 SLR1-, GCT1 SLR1+, GCT2 SLR1+, GCT3 SLR1+

void L1TCorrelatorLayer1Producer::addGCTEmCaloRaw(const l1tp2::GCTEmDigiClusterLink &link,
                                                  unsigned int linkidx,
                                                  unsigned int entidx) {
  event_.raw.gctEm[calomapping[linkidx]].obj.push_back(link[entidx].data());
  addDecodedGCTEmCalo(event_.decoded.emcalo[calomapping[linkidx]], link[entidx]);
}

void L1TCorrelatorLayer1Producer::addGCTHadCaloRaw(const l1tp2::GCTHadDigiClusterLink &link,
                                                   unsigned int linkidx,
                                                   unsigned int entidx) {
  event_.raw.gctHad[calomapping[linkidx]].obj.push_back(link[entidx].data());
  addDecodedGCTHadCalo(event_.decoded.hadcalo[calomapping[linkidx]], link[entidx]);
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
  tkAndSel.first.hwChi2 = round(t.chi2() * 10);
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

void L1TCorrelatorLayer1Producer::addDecodedGCTEmCalo(l1ct::DetectorSector<l1ct::EmCaloObjEmu> &sec,
                                                      const l1tp2::GCTEmDigiCluster &digi) {
  l1ct::EmCaloObjEmu calo = gctEmInput_->decode(sec.region, digi.data());

  auto caloPtr = edm::refToPtr(digi.clusterRef());
  // FIXME: should check hwPt > 0
  addDecodedEmCalo(calo, caloPtr, sec);
}

void L1TCorrelatorLayer1Producer::addDecodedGCTHadCalo(l1ct::DetectorSector<l1ct::HadCaloObjEmu> &sec,
                                                       const l1tp2::GCTHadDigiCluster &digi) {
  l1ct::HadCaloObjEmu calo = gctHadInput_->decode(sec.region, digi.data());

  auto caloPtr = edm::refToPtr(digi.clusterRef());
  // FIXME: should check hwPt > 0
  addDecodedHadCalo(calo, caloPtr, sec);
}

template <typename T>
void L1TCorrelatorLayer1Producer::setRefs_(l1t::PFCandidate &pf, const T &p) const {
  if (p.srcCluster) {
    auto match = clusterRefMap_.find(p.srcCluster);
    if (match == clusterRefMap_.end()) {
      throw cms::Exception("CorruptData") << "Invalid cluster pointer in PF candidate id " << p.intId() << " pt "
                                          << p.floatPt() << " eta " << p.floatEta() << " phi " << p.floatPhi();
    }
    pf.setCaloPtr(match->second);
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
std::string L1TCorrelatorLayer1Producer::refExcepMsg_<l1ct::PFNeutralObjEmu>(const l1ct::PFNeutralObjEmu &key) const {
  return "Invalid pointer in Neutral PF candidate id " + std::to_string(key.intId()) + " pt " +
         std::to_string(key.floatPt()) + " eta " + std::to_string(key.floatEta()) + " phi " +
         std::to_string(key.floatPhi());
}

template <>
std::string L1TCorrelatorLayer1Producer::refExcepMsg_<l1ct::HadCaloObjEmu>(const l1ct::HadCaloObjEmu &key) const {
  return "Invalid pointer in hadcalo obj, pt " + std::to_string(key.floatPt()) + " eta " +
         std::to_string(key.floatEta()) + " phi " + std::to_string(key.floatPhi());
}

template <>
std::string L1TCorrelatorLayer1Producer::refExcepMsg_<l1ct::EmCaloObjEmu>(const l1ct::EmCaloObjEmu &key) const {
  return "Invalid pointer in emcalo obj, pt " + std::to_string(key.floatPt()) + " eta " +
         std::to_string(key.floatEta()) + " phi " + std::to_string(key.floatPhi());
}

template <>
std::string L1TCorrelatorLayer1Producer::refExcepMsg_<l1ct::TkObjEmu>(const l1ct::TkObjEmu &key) const {
  return "Invalid track pointer in track obj, pt " + std::to_string(key.floatPt()) + " eta " +
         std::to_string(key.floatEta()) + " phi " + std::to_string(key.floatPhi());
}

template <>
std::string L1TCorrelatorLayer1Producer::refExcepMsg_<l1ct::EGIsoObjEmu>(const l1ct::EGIsoObjEmu &key) const {
  return "Invalid cluster pointer in EGIso candidate, pt " + std::to_string(key.floatPt()) + " eta " +
         std::to_string(key.floatEta()) + " phi " + std::to_string(key.floatPhi());
}

template <>
std::string L1TCorrelatorLayer1Producer::refExcepMsg_<l1ct::EGIsoEleObjEmu>(const l1ct::EGIsoEleObjEmu &key) const {
  return "Invalid cluster pointer in EGEleIso candidate, pt " + std::to_string(key.floatPt()) + " eta " +
         std::to_string(key.floatEta()) + " phi " + std::to_string(key.floatPhi());
}

template <>
void L1TCorrelatorLayer1Producer::setRefs_<l1ct::PFNeutralObjEmu>(l1t::PFCandidate &pf,
                                                                  const l1ct::PFNeutralObjEmu &p) const {
  if (p.srcCluster) {
    pf.setCaloPtr(findRef_(clusterRefMap_, p.srcCluster, p));
  }
}

template <>
void L1TCorrelatorLayer1Producer::setRefs_<l1ct::HadCaloObjEmu>(l1t::PFCandidate &pf,
                                                                const l1ct::HadCaloObjEmu &p) const {
  if (p.src) {
    pf.setCaloPtr(findRef_(clusterRefMap_, p.src, p));
  }
}

template <>
void L1TCorrelatorLayer1Producer::setRefs_<l1ct::EmCaloObjEmu>(l1t::PFCandidate &pf,
                                                               const l1ct::EmCaloObjEmu &p) const {
  if (p.src) {
    pf.setCaloPtr(findRef_(clusterRefMap_, p.src, p));
  }
}

template <>
void L1TCorrelatorLayer1Producer::setRefs_<l1ct::TkObjEmu>(l1t::PFCandidate &pf, const l1ct::TkObjEmu &p) const {
  if (p.src) {
    pf.setPFTrack(findRef_(trackRefMap_, p.src, p));
  }
}

template <typename T>
void L1TCorrelatorLayer1Producer::setRefs_(l1t::PFCluster &pf, const T &p) const {
  if (p.src) {
    pf.addConstituent(findRef_(clusterRefMap_, p.src, p));
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

std::unique_ptr<l1t::PFClusterCollection> L1TCorrelatorLayer1Producer::fetchDecodedHadCalo() const {
  auto ret = std::make_unique<l1t::PFClusterCollection>();
  for (const auto &r : event_.pfinputs) {
    const auto &reg = r.region;
    for (const auto &p : r.hadcalo) {
      if (p.hwPt == 0 || !reg.isFiducial(p))
        continue;
      addHadPFCluster(p, reg, ret);
    }
  }
  return ret;
}

std::unique_ptr<l1t::PFClusterCollection> L1TCorrelatorLayer1Producer::fetchDecodedEmCalo() const {
  auto ret = std::make_unique<l1t::PFClusterCollection>();
  for (const auto &r : event_.pfinputs) {
    const auto &reg = r.region;
    for (const auto &p : r.emcalo) {
      if (p.hwPt == 0 || !reg.isFiducial(p))
        continue;
      addEmPFCluster(p, reg, ret);
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

std::unique_ptr<l1t::PFTrackCollection> L1TCorrelatorLayer1Producer::fetchDecodedTracks() const {
  auto ret = std::make_unique<l1t::PFTrackCollection>();
  for (const auto &r : event_.decoded.track) {
    const auto &reg = r.region;
    for (const auto &p : r.obj) {
      if (p.hwPt == 0)
        continue;
      reco::Particle::PolarLorentzVector p4(
          p.floatPt(), reg.floatGlbEta(p.hwVtxEta()), reg.floatGlbPhi(p.hwVtxPhi()), 0);

      reco::Particle::Point vtx(0, 0, p.floatZ0());

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

void L1TCorrelatorLayer1Producer::putEgStaObjects(edm::Event &iEvent, const std::string &egLablel) const {
  auto egs = std::make_unique<BXVector<l1t::EGamma>>();
  // FIXME: in case more BXes are introduced shuld probably use egs->key(egs->end(bx));

  for (unsigned int ir = 0, nr = event_.pfinputs.size(); ir < nr; ++ir) {
    const auto &reg = event_.pfinputs[ir].region;

    // EG standalone objects
    for (unsigned int ieg = 0, neg = event_.out[ir].egsta.size(); ieg < neg; ++ieg) {
      const auto &p = event_.out[ir].egsta[ieg];
      if (p.hwPt == 0 || !reg.isFiducial(p))
        continue;
      l1t::EGamma eg(
          reco::Candidate::PolarLorentzVector(p.floatPt(), reg.floatGlbEta(p.hwEta), reg.floatGlbPhi(p.hwPhi), 0.));
      eg.setHwQual(p.hwQual);
      egs->push_back(0, eg);
    }
  }

  iEvent.put(std::move(egs), egLablel);
}

void L1TCorrelatorLayer1Producer::putEgObjects(edm::Event &iEvent,
                                               const bool writeEgSta,
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

      reco::Candidate::PolarLorentzVector mom(egiso.floatPt(), egiso.floatEta(), egiso.floatPhi(), 0.);

      l1t::TkEm tkem(reco::Candidate::LorentzVector(mom),
                     findRef_(clusterRefMap_, egiso.srcCluster, egiso),
                     egiso.floatRelIso(l1ct::EGIsoObjEmu::IsoType::TkIso),
                     egiso.floatRelIso(l1ct::EGIsoObjEmu::IsoType::TkIsoPV));
      tkem.setHwQual(egiso.hwQual);
      tkem.setPFIsol(egiso.floatRelIso(l1ct::EGIsoObjEmu::IsoType::PfIso));
      tkem.setPFIsolPV(egiso.floatRelIso(l1ct::EGIsoObjEmu::IsoType::PfIsoPV));
      tkem.setEgBinaryWord(egiso.pack(), l1t::TkEm::HWEncoding::CT);
      tkems->push_back(tkem);
      npho_obj.push_back(tkems->size() - 1);
    }
    tkemPerBoard->addRegion(npho_obj, board.eta, board.phi);

    nele_obj.clear();
    for (const auto &egele : board.egelectron) {
      if (egele.hwPt == 0)
        continue;

      reco::Candidate::PolarLorentzVector mom(egele.floatPt(), egele.floatVtxEta(), egele.floatVtxPhi(), 0.);

      l1t::TkElectron tkele(reco::Candidate::LorentzVector(mom),
                            findRef_(clusterRefMap_, egele.srcCluster, egele),
                            edm::refToPtr(egele.srcTrack->track()),
                            egele.floatRelIso(l1ct::EGIsoEleObjEmu::IsoType::TkIso));
      tkele.setHwQual(egele.hwQual);
      tkele.setPFIsol(egele.floatRelIso(l1ct::EGIsoEleObjEmu::IsoType::PfIso));
      tkele.setEgBinaryWord(egele.pack(), l1t::TkElectron::HWEncoding::CT);
      tkele.setIdScore(egele.floatIDScore());
      tkele.setCharge(egele.intCharge());
      tkele.setTrkzVtx(egele.floatZ0());
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
