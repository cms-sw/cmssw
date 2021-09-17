#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TObjString.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1Trigger/interface/Vertex.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "L1Trigger/VertexFinder/interface/AnalysisSettings.h"
#include "L1Trigger/VertexFinder/interface/InputData.h"
#include "L1Trigger/VertexFinder/interface/L1TrackTruthMatched.h"
#include "L1Trigger/VertexFinder/interface/RecoVertex.h"
#include "L1Trigger/VertexFinder/interface/selection.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTTrackAssociationMap.h"

#include "TTree.h"

#include <map>
#include <string>
#include <vector>
#include <iostream>

using namespace std;

namespace l1tVertexFinder {

  class VertexNTupler : public edm::EDAnalyzer {
  public:
    explicit VertexNTupler(const edm::ParameterSet&);
    ~VertexNTupler() override;

  private:
    struct GenJetsBranchData {
      std::vector<float> energy;
      std::vector<float> pt;
      std::vector<float> eta;
      std::vector<float> phi;

      void clear() {
        energy.clear();
        pt.clear();
        eta.clear();
        phi.clear();
      }
    };

    struct GenParticlesBranchData {
      std::vector<float> energy;
      std::vector<float> pt;
      std::vector<float> eta;
      std::vector<float> phi;
      std::vector<int> pdgId;
      std::vector<int> status;

      void clear() {
        energy.clear();
        pt.clear();
        eta.clear();
        phi.clear();
        pdgId.clear();
        status.clear();
      }
    };

    struct RecoVerticesBranchData {
      std::vector<unsigned> numTracks;
      std::vector<std::vector<unsigned>> trackIdxs;
      std::vector<float> z0;
      std::vector<float> sumPt;

      void clear() {
        numTracks.clear();
        trackIdxs.clear();
        z0.clear();
        sumPt.clear();
      }
    };

    struct RecoTracksBranchData {
      std::vector<float> pt;
      std::vector<float> eta;
      std::vector<float> phi;
      std::vector<float> z0;
      std::vector<unsigned> numStubs;
      std::vector<float> chi2dof;
      std::vector<int> trueMatchIdx;
      std::vector<int> truthMapMatchIdx;
      std::vector<float> truthMapIsGenuine;
      std::vector<float> truthMapIsLooselyGenuine;
      std::vector<float> truthMapIsCombinatoric;
      std::vector<float> truthMapIsUnknown;

      void clear() {
        pt.clear();
        eta.clear();
        phi.clear();
        z0.clear();
        numStubs.clear();
        chi2dof.clear();
        trueMatchIdx.clear();
        truthMapMatchIdx.clear();
        truthMapIsGenuine.clear();
        truthMapIsLooselyGenuine.clear();
        truthMapIsCombinatoric.clear();
        truthMapIsUnknown.clear();
      }
    };

    struct TrueTracksBranchData {
      std::vector<float> pt;
      std::vector<float> eta;
      std::vector<float> phi;
      std::vector<float> z0;
      std::vector<int> pdgId;
      std::vector<float> physCollision;
      std::vector<float> use;
      std::vector<float> useForEff;
      std::vector<float> useForAlgEff;
      std::vector<float> useForVertexReco;

      void clear() {
        pt.clear();
        eta.clear();
        phi.clear();
        z0.clear();
        pdgId.clear();
        physCollision.clear();
        use.clear();
        useForEff.clear();
        useForAlgEff.clear();
        useForVertexReco.clear();
      }
    };

    void beginJob() override;
    void analyze(const edm::Event& evt, const edm::EventSetup& setup) override;
    void endJob() override;

    // define types for stub-related classes
    typedef TTTrackAssociationMap<Ref_Phase2TrackerDigi_> TTTrackAssMap;
    typedef edm::View<TTTrack<Ref_Phase2TrackerDigi_>> TTTrackCollectionView;

    // references to tags containing information relevant to perofrmance analysis
    const edm::EDGetTokenT<l1tVertexFinder::InputData> inputDataToken_;
    const edm::EDGetTokenT<std::vector<PileupSummaryInfo>> pileupSummaryToken_;
    const edm::EDGetTokenT<edm::View<reco::GenParticle>> genParticlesToken_;
    const edm::EDGetTokenT<std::vector<reco::GenJet>> genJetsToken_;
    const edm::EDGetTokenT<std::vector<l1tVertexFinder::TP>> allMatchedTPsToken_;
    const edm::EDGetTokenT<edm::ValueMap<l1tVertexFinder::TP>> vTPsToken_;
    std::map<std::string, edm::EDGetTokenT<TTTrackCollectionView>> l1TracksTokenMap_;
    std::map<std::string, edm::EDGetTokenT<TTTrackAssociationMap<Ref_Phase2TrackerDigi_>>> l1TracksMapTokenMap_;
    std::map<std::string, edm::EDGetTokenT<std::vector<l1t::Vertex>>> l1VerticesTokenMap_;
    std::map<std::string, edm::EDGetTokenT<std::vector<l1t::Vertex>>> l1VerticesExtraTokenMap_;
    std::vector<edm::EDGetTokenT<std::vector<l1t::Vertex>>> l1VerticesExtraTokens_;

    TTree* outputTree_;

    const bool printResults_;

    // storage class for configuration parameters
    AnalysisSettings settings_;

    //edm::Service<TFileService> fs_;

    // Histograms for Vertex Reconstruction

    float numTrueInteractions_, hepMCVtxZ0_, genVtxZ0_;
    int numPileupVertices_;

    GenJetsBranchData genJetsBranchData_;
    TrueTracksBranchData trueTracksBranchData_;
    std::vector<float> truePileUpVtxZ0_;
    GenParticlesBranchData genParticlesHardOutgoingBranchData_;

    std::map<std::string, RecoTracksBranchData> l1TracksBranchData_;
    std::map<std::string, RecoVerticesBranchData> l1VerticesBranchData_;
    std::map<std::string, std::string> l1VerticesInputMap_;

    std::unordered_map<std::string, std::vector<unsigned>> l1Vertices_branchMap_numTracks_;
    std::unordered_map<std::string, std::vector<float>> l1Vertices_branchMap_z0_;
    std::unordered_map<std::string, std::vector<float>> l1Vertices_branchMap_z0_etaWeighted_;
    std::unordered_map<std::string, std::vector<float>> l1Vertices_branchMap_sumPt_;

    std::vector<std::vector<unsigned>> l1Vertices_extra_numTracks_;
    std::vector<std::vector<float>> l1Vertices_extra_z0_;
    std::vector<std::vector<float>> l1Vertices_extra_z0_etaWeighted_;
    std::vector<std::vector<float>> l1Vertices_extra_sumPt_;

    bool available_;  // ROOT file for histograms is open.
  };

  VertexNTupler::VertexNTupler(const edm::ParameterSet& iConfig)
      : inputDataToken_(consumes<l1tVertexFinder::InputData>(iConfig.getParameter<edm::InputTag>("inputDataInputTag"))),
        pileupSummaryToken_(consumes<std::vector<PileupSummaryInfo>>(edm::InputTag("addPileupInfo"))),
        genParticlesToken_(
            consumes<edm::View<reco::GenParticle>>(iConfig.getParameter<edm::InputTag>("genParticleInputTag"))),
        genJetsToken_(consumes<std::vector<reco::GenJet>>(iConfig.getParameter<edm::InputTag>("genJetsInputTag"))),
        allMatchedTPsToken_(
            consumes<std::vector<l1tVertexFinder::TP>>(iConfig.getParameter<edm::InputTag>("l1TracksTPInputTags"))),
        vTPsToken_(consumes<edm::ValueMap<l1tVertexFinder::TP>>(
            iConfig.getParameter<edm::InputTag>("l1TracksTPValueMapInputTags"))),
        //outputTree_(fs_->make<TTree>("l1VertexReco", "L1 vertex-related info")),
        printResults_(iConfig.getParameter<bool>("printResults")),
        settings_(iConfig) {
    const std::vector<std::string> trackBranchNames(
        iConfig.getParameter<std::vector<std::string>>("l1TracksBranchNames"));
    const std::vector<edm::InputTag> trackInputTags(
        iConfig.getParameter<std::vector<edm::InputTag>>("l1TracksInputTags"));
    const std::vector<edm::InputTag> trackMapInputTags(
        iConfig.getParameter<std::vector<edm::InputTag>>("l1TracksTruthMapInputTags"));

    edm::Service<TFileService> fs_;
    available_ = fs_.isAvailable();
    if (not available_)
      return;  // No ROOT file open.

    outputTree_ = fs_->make<TTree>("l1VertexReco", "L1 vertex-related info");

    if (trackBranchNames.size() != trackInputTags.size())
      throw cms::Exception("The number of track branch names (" + std::to_string(trackBranchNames.size()) +
                           ") specified in the config does not match the number of input tags (" +
                           std::to_string(trackInputTags.size()) + ")");
    if (trackBranchNames.size() != trackMapInputTags.size())
      throw cms::Exception("The number of track branch names (" + std::to_string(trackBranchNames.size()) +
                           ") specified in the config does not match the number of track map input tags (" +
                           std::to_string(trackMapInputTags.size()) + ")");

    const std::vector<std::string> vertexBranchNames(
        iConfig.getParameter<std::vector<std::string>>("l1VertexBranchNames"));
    const std::vector<edm::InputTag> vertexInputTags(
        iConfig.getParameter<std::vector<edm::InputTag>>("l1VertexInputTags"));
    const std::vector<std::string> vertexTrackNames(
        iConfig.getParameter<std::vector<std::string>>("l1VertexTrackInputs"));

    if (vertexBranchNames.size() != vertexInputTags.size())
      throw cms::Exception("The number of vertex branch names (" + std::to_string(vertexBranchNames.size()) +
                           ") specified in the config does not match the number of input tags (" +
                           std::to_string(vertexInputTags.size()) + ")");
    if (vertexBranchNames.size() != vertexTrackNames.size())
      throw cms::Exception(
          "The number of vertex branch names (" + std::to_string(vertexBranchNames.size()) +
          ") specified in the config does not match the number of associated input track collection names (" +
          std::to_string(vertexTrackNames.size()) + ")");

    outputTree_->Branch("genJets_energy", &genJetsBranchData_.energy);
    outputTree_->Branch("genJets_pt", &genJetsBranchData_.pt);
    outputTree_->Branch("genJets_eta", &genJetsBranchData_.eta);
    outputTree_->Branch("genJets_phi", &genJetsBranchData_.phi);
    outputTree_->Branch("genParticles_hardProcOutgoing_energy", &genParticlesHardOutgoingBranchData_.energy);
    outputTree_->Branch("genParticles_hardProcOutgoing_pt", &genParticlesHardOutgoingBranchData_.pt);
    outputTree_->Branch("genParticles_hardProcOutgoing_eta", &genParticlesHardOutgoingBranchData_.eta);
    outputTree_->Branch("genParticles_hardProcOutgoing_phi", &genParticlesHardOutgoingBranchData_.phi);
    outputTree_->Branch("genParticles_hardProcOutgoing_pdgId", &genParticlesHardOutgoingBranchData_.pdgId);
    outputTree_->Branch("genParticles_hardProcOutgoing_status", &genParticlesHardOutgoingBranchData_.status);
    outputTree_->Branch("genVertex_z0", &genVtxZ0_);
    outputTree_->Branch("hepMCVertex_z0", &hepMCVtxZ0_);
    outputTree_->Branch("pileupSummary_trueNumInteractions", &numTrueInteractions_);
    outputTree_->Branch("pileupSummary_numPileupVertices", &numPileupVertices_);

    std::vector<std::string>::const_iterator trackBranchNameIt = trackBranchNames.begin();
    std::vector<edm::InputTag>::const_iterator trackInputTagIt = trackInputTags.begin();
    std::vector<edm::InputTag>::const_iterator trackMapInputTagIt = trackMapInputTags.begin();
    for (; trackBranchNameIt != trackBranchNames.end(); trackBranchNameIt++, trackInputTagIt++, trackMapInputTagIt++) {
      l1TracksTokenMap_[*trackBranchNameIt] = consumes<TTTrackCollectionView>(*trackInputTagIt);
      l1TracksMapTokenMap_[*trackBranchNameIt] =
          consumes<TTTrackAssociationMap<Ref_Phase2TrackerDigi_>>(*trackMapInputTagIt);

      RecoTracksBranchData& branchData = l1TracksBranchData_[*trackBranchNameIt];

      outputTree_->Branch(("recoTracks_" + *trackBranchNameIt + "_pt").c_str(), &branchData.pt);
      outputTree_->Branch(("recoTracks_" + *trackBranchNameIt + "_eta").c_str(), &branchData.eta);
      outputTree_->Branch(("recoTracks_" + *trackBranchNameIt + "_phi").c_str(), &branchData.phi);
      outputTree_->Branch(("recoTracks_" + *trackBranchNameIt + "_z0").c_str(), &branchData.z0);
      outputTree_->Branch(("recoTracks_" + *trackBranchNameIt + "_numStubs").c_str(), &branchData.numStubs);
      outputTree_->Branch(("recoTracks_" + *trackBranchNameIt + "_chi2dof").c_str(), &branchData.chi2dof);
      outputTree_->Branch(("recoTracks_" + *trackBranchNameIt + "_trueMatchIdx").c_str(), &branchData.trueMatchIdx);
      outputTree_->Branch(("recoTracks_" + *trackBranchNameIt + "_truthMap_matchIdx").c_str(),
                          &branchData.truthMapMatchIdx);
      outputTree_->Branch(("recoTracks_" + *trackBranchNameIt + "_truthMap_isGenuine").c_str(),
                          &branchData.truthMapIsGenuine);
      outputTree_->Branch(("recoTracks_" + *trackBranchNameIt + "_truthMap_isLooselyGenuine").c_str(),
                          &branchData.truthMapIsLooselyGenuine);
      outputTree_->Branch(("recoTracks_" + *trackBranchNameIt + "_truthMap_isCombinatoric").c_str(),
                          &branchData.truthMapIsCombinatoric);
      outputTree_->Branch(("recoTracks_" + *trackBranchNameIt + "_truthMap_isUnknown").c_str(),
                          &branchData.truthMapIsUnknown);
    }

    std::vector<std::string>::const_iterator branchNameIt = vertexBranchNames.begin();
    std::vector<edm::InputTag>::const_iterator inputTagIt = vertexInputTags.begin();
    std::vector<std::string>::const_iterator l1VertexTrackNameIt = vertexTrackNames.begin();
    for (; branchNameIt != vertexBranchNames.end(); branchNameIt++, inputTagIt++, l1VertexTrackNameIt++) {
      l1VerticesTokenMap_[*branchNameIt] = consumes<std::vector<l1t::Vertex>>(*inputTagIt);

      l1VerticesBranchData_[*branchNameIt] = RecoVerticesBranchData();
      RecoVerticesBranchData& branchData = l1VerticesBranchData_.at(*branchNameIt);
      l1VerticesInputMap_[*branchNameIt] = *l1VertexTrackNameIt;

      if (l1TracksTokenMap_.count(*l1VertexTrackNameIt) == 0)
        throw cms::Exception("Invalid track collection name '" + *l1VertexTrackNameIt +
                             "' specified as input to vertex collection '" + *branchNameIt + "'");

      outputTree_->Branch(("recoVertices_" + *branchNameIt + "_numTracks").c_str(), &branchData.numTracks);
      outputTree_->Branch(("recoVertices_" + *branchNameIt + "_trackIdxs").c_str(), &branchData.trackIdxs);
      outputTree_->Branch(("recoVertices_" + *branchNameIt + "_z0").c_str(), &branchData.z0);
      outputTree_->Branch(("recoVertices_" + *branchNameIt + "_sumPt").c_str(), &branchData.sumPt);
    }

    outputTree_->Branch("truePileUpVertices_z0", &truePileUpVtxZ0_);
    outputTree_->Branch("trueTracks_pt", &trueTracksBranchData_.pt);
    outputTree_->Branch("trueTracks_eta", &trueTracksBranchData_.eta);
    outputTree_->Branch("trueTracks_phi", &trueTracksBranchData_.phi);
    outputTree_->Branch("trueTracks_z0", &trueTracksBranchData_.z0);
    outputTree_->Branch("trueTracks_pdgId", &trueTracksBranchData_.pdgId);
    outputTree_->Branch("trueTracks_physCollision", &trueTracksBranchData_.physCollision);
    outputTree_->Branch("trueTracks_use", &trueTracksBranchData_.use);
    outputTree_->Branch("trueTracks_useForEff", &trueTracksBranchData_.useForEff);
    outputTree_->Branch("trueTracks_useForAlgEff", &trueTracksBranchData_.useForAlgEff);
    outputTree_->Branch("trueTracks_useForVtxReco", &trueTracksBranchData_.useForVertexReco);

    const std::vector<edm::InputTag> extraVertexInputTags(
        iConfig.getParameter<std::vector<edm::InputTag>>("extraL1VertexInputTags"));
    const std::vector<std::string> extraVertexDescriptions(
        iConfig.getParameter<std::vector<std::string>>("extraL1VertexDescriptions"));
    for (const auto& inputTag : extraVertexInputTags)
      l1VerticesExtraTokens_.push_back(consumes<std::vector<l1t::Vertex>>(inputTag));
    TObjArray* descriptionArray = new TObjArray();
    for (const auto& description : extraVertexDescriptions)
      descriptionArray->Add(new TObjString(description.c_str()));
    outputTree_->GetUserInfo()->Add(descriptionArray);
    outputTree_->Branch("vertices_extra_numTracks", &l1Vertices_extra_numTracks_);
    outputTree_->Branch("vertices_extra_z0", &l1Vertices_extra_z0_);
    outputTree_->Branch("vertices_extra_z0_etaWeighted", &l1Vertices_extra_z0_etaWeighted_);
    outputTree_->Branch("vertices_extra_sumPt", &l1Vertices_extra_sumPt_);
  }

  void VertexNTupler::beginJob() {}

  std::ostream& operator<<(std::ostream& out, const reco::GenParticle& particle) {
    const bool positive = (particle.pdgId() < 0);
    const size_t absId = abs(particle.pdgId());
    switch (absId) {
      case 1:
        return (out << (positive ? "d" : "anti-d"));
      case 2:
        return (out << (positive ? "u" : "anti-u"));
      case 3:
        return (out << (positive ? "s" : "anti-s"));
      case 4:
        return (out << (positive ? "c" : "anti-c"));
      case 5:
        return (out << (positive ? "b" : "anti-b"));
      case 6:
        return (out << (positive ? "t" : "anti-t"));
      case 11:
        return (out << (positive ? "e+" : "e-"));
      case 12:
        return (out << (positive ? "nu_e" : "anti-nu_e"));
      case 13:
        return (out << (positive ? "mu+" : "mu-"));
      case 14:
        return (out << (positive ? "nu_mu" : "anti-nu_mu"));
      case 15:
        return (out << (positive ? "tau+" : "tau-"));
      case 16:
        return (out << (positive ? "nu_tau" : "anti-nu_tau"));
      case 21:
        return (out << "g");
      case 22:
        return (out << "photon");
      case 23:
        return (out << "Z");
      case 24:
        return (out << (positive ? "W-" : "W+"));
      default:
        if ((((absId / 1000) % 10) != 0) and (((absId / 10) % 10) == 0))
          return (out << "diquark<" << particle.pdgId() << ">");
        else
          return (out << "unknown<" << particle.pdgId() << ">");
    }
  }

  void VertexNTupler::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    // Note useful info about MC truth particles and about reconstructed stubs
    edm::Handle<l1tVertexFinder::InputData> inputDataHandle;
    iEvent.getByToken(inputDataToken_, inputDataHandle);
    InputData inputData = *inputDataHandle;

    edm::Handle<edm::ValueMap<TP>> tpValueMapHandle;
    iEvent.getByToken(vTPsToken_, tpValueMapHandle);
    edm::ValueMap<TP> tpValueMap = *tpValueMapHandle;

    // Create collections to hold the desired objects
    std::map<std::string, std::vector<L1TrackTruthMatched>> l1TrackCollections;
    std::map<std::string, edm::Handle<TTTrackAssMap>> truthAssocMapHandles;

    // Create iterators for the maps
    auto tokenMapEntry = l1TracksTokenMap_.begin(), tokenMapEntryEnd = l1TracksTokenMap_.end();
    auto mapTokenMapEntry = l1TracksMapTokenMap_.begin(), mapTokenMapEntryEnd = l1TracksMapTokenMap_.end();

    // Iterate over the maps
    for (; (tokenMapEntry != tokenMapEntryEnd) && (mapTokenMapEntry != mapTokenMapEntryEnd);
         ++tokenMapEntry, ++mapTokenMapEntry) {
      edm::Handle<TTTrackCollectionView> l1TracksHandle;
      edm::Handle<TTTrackAssMap>& mcTruthTTTrackHandle = truthAssocMapHandles[mapTokenMapEntry->first];
      iEvent.getByToken(tokenMapEntry->second, l1TracksHandle);
      iEvent.getByToken(l1TracksMapTokenMap_.at(mapTokenMapEntry->first), mcTruthTTTrackHandle);

      std::vector<L1TrackTruthMatched>& l1Tracks = l1TrackCollections[tokenMapEntry->first];
      l1Tracks.reserve(l1TracksHandle->size());
      for (const auto& track : l1TracksHandle->ptrs()) {
        l1Tracks.push_back(L1TrackTruthMatched(track, inputData.getTPPtrToRefMap(), tpValueMap, mcTruthTTTrackHandle));
      }
    }

    // create a map for associating fat reco tracks with their underlying
    // TTTrack pointers
    std::map<std::string, std::map<const edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>>, const L1TrackTruthMatched*>>
        edmL1TrackMaps;

    // get a list of reconstructed tracks with references to their TPs
    for (const auto& entry : l1TrackCollections) {
      auto& edmL1Map = edmL1TrackMaps[entry.first];
      for (const auto& track : entry.second) {
        edmL1Map.insert(std::pair<const edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>>, const L1TrackTruthMatched*>(
            track.getTTTrackPtr(), &track));
      }
    }

    ////////////////////////////////////////////////////////

    // Pile-up summary info
    edm::Handle<std::vector<PileupSummaryInfo>> pileupHandle;
    iEvent.getByToken(pileupSummaryToken_, pileupHandle);

    for (auto bxIt = pileupHandle->begin(); bxIt != pileupHandle->end(); bxIt++) {
      if (bxIt->getBunchCrossing() == 0) {
        numTrueInteractions_ = bxIt->getTrueNumInteractions();
        numPileupVertices_ = bxIt->getPU_NumInteractions();
      }
    }

    // True track info
    trueTracksBranchData_.clear();
    edm::Handle<std::vector<l1tVertexFinder::TP>> allMatchedTPsHandle;
    iEvent.getByToken(allMatchedTPsToken_, allMatchedTPsHandle);

    for (const auto& tp : *allMatchedTPsHandle) {
      trueTracksBranchData_.pt.push_back(tp->pt());
      trueTracksBranchData_.eta.push_back(tp->eta());
      trueTracksBranchData_.phi.push_back(tp->phi());
      trueTracksBranchData_.z0.push_back(tp->z0());
      trueTracksBranchData_.pdgId.push_back(tp->pdgId());
      trueTracksBranchData_.physCollision.push_back(tp.physicsCollision() ? 1.0 : 0.0);
      trueTracksBranchData_.use.push_back(tp.use() ? 1.0 : 0.0);
      trueTracksBranchData_.useForEff.push_back(tp.useForEff() ? 1.0 : 0.0);
      trueTracksBranchData_.useForAlgEff.push_back(tp.useForAlgEff() ? 1.0 : 0.0);
      trueTracksBranchData_.useForVertexReco.push_back(tp.useForVertexReco() ? 1.0 : 0.0);
    }

    // True pile-up vertex info
    truePileUpVtxZ0_.clear();
    for (const Vertex& vtx : inputData.getPileUpVertices())
      truePileUpVtxZ0_.push_back(vtx.z0());

    // Generator level vertex info
    hepMCVtxZ0_ = inputData.getHepMCVertex().vz();
    genVtxZ0_ = inputData.getGenVertex().vz();

    // Gen particles
    genParticlesHardOutgoingBranchData_.clear();
    edm::Handle<edm::View<reco::GenParticle>> genParticlesH;
    iEvent.getByToken(genParticlesToken_, genParticlesH);
    for (const auto& p : *genParticlesH) {
      genParticlesHardOutgoingBranchData_.energy.push_back(p.energy());
      genParticlesHardOutgoingBranchData_.pt.push_back(p.pt());
      genParticlesHardOutgoingBranchData_.eta.push_back(p.eta());
      genParticlesHardOutgoingBranchData_.phi.push_back(p.phi());
      genParticlesHardOutgoingBranchData_.pdgId.push_back(p.pdgId());
      genParticlesHardOutgoingBranchData_.status.push_back(p.status());
    }

    // Gen jet (AK4) branches
    genJetsBranchData_.clear();
    edm::Handle<std::vector<reco::GenJet>> genJetsHandle;
    iEvent.getByToken(genJetsToken_, genJetsHandle);
    for (const auto& genJet : *genJetsHandle) {
      genJetsBranchData_.energy.push_back(genJet.energy());
      genJetsBranchData_.pt.push_back(genJet.pt());
      genJetsBranchData_.eta.push_back(genJet.eta());
      genJetsBranchData_.phi.push_back(genJet.phi());
    }

    for (const auto& entry : l1TrackCollections) {
      const auto& l1Tracks = entry.second;
      RecoTracksBranchData& branchData = l1TracksBranchData_.at(entry.first);

      const TTTrackAssociationMap<Ref_Phase2TrackerDigi_>& truthAssocMap = *truthAssocMapHandles.at(entry.first);

      // Reco track branches
      branchData.clear();
      for (const L1TrackTruthMatched& track : l1Tracks) {
        branchData.pt.push_back(track.pt());
        branchData.eta.push_back(track.eta());
        branchData.phi.push_back(track.phi0());
        branchData.z0.push_back(track.z0());
        branchData.numStubs.push_back(track.getNumStubs());
        branchData.chi2dof.push_back(track.chi2dof());
        branchData.trueMatchIdx.push_back(track.getMatchedTPidx());

        edm::Ptr<TrackingParticle> matchedTP = truthAssocMap.findTrackingParticlePtr(track.getTTTrackPtr());
        if (matchedTP.isNull())
          branchData.truthMapMatchIdx.push_back(-1);
        else {
          auto it = std::find_if(allMatchedTPsHandle->begin(),
                                 allMatchedTPsHandle->end(),
                                 [&matchedTP](auto const& tp) { return tp.getTrackingParticle() == matchedTP; });
          assert(it != allMatchedTPsHandle->end());
          branchData.truthMapMatchIdx.push_back(std::distance(allMatchedTPsHandle->begin(), it));
        }
        branchData.truthMapIsGenuine.push_back(truthAssocMap.isGenuine(track.getTTTrackPtr()) ? 1.0 : 0.0);
        branchData.truthMapIsLooselyGenuine.push_back(truthAssocMap.isLooselyGenuine(track.getTTTrackPtr()) ? 1.0
                                                                                                            : 0.0);
        branchData.truthMapIsCombinatoric.push_back(truthAssocMap.isCombinatoric(track.getTTTrackPtr()) ? 1.0 : 0.0);
        branchData.truthMapIsUnknown.push_back(truthAssocMap.isUnknown(track.getTTTrackPtr()) ? 1.0 : 0.0);
      }
    }

    // Reco vertex branches
    for (const auto& tokenMapEntry : l1VerticesTokenMap_) {
      RecoVerticesBranchData& branchData = l1VerticesBranchData_.at(tokenMapEntry.first);

      edm::Handle<std::vector<l1t::Vertex>> handle;
      iEvent.getByToken(tokenMapEntry.second, handle);
      std::vector<std::shared_ptr<const RecoVertexWithTP>> recoVertices;
      recoVertices.reserve(handle->size());
      for (unsigned int i = 0; i < handle->size(); ++i) {
        recoVertices.push_back(std::shared_ptr<const RecoVertexWithTP>(
            new RecoVertexWithTP(handle->at(i), edmL1TrackMaps.at(l1VerticesInputMap_.at(tokenMapEntry.first)))));
      }

      branchData.clear();
      std::vector<L1TrackTruthMatched>& l1Tracks = l1TrackCollections.at(l1VerticesInputMap_.at(tokenMapEntry.first));
      for (const std::shared_ptr<const RecoVertexWithTP>& vtx : recoVertices) {
        branchData.numTracks.push_back(vtx->numTracks());
        branchData.trackIdxs.push_back(std::vector<unsigned>());
        for (const L1TrackTruthMatched* track : vtx->tracks())
          branchData.trackIdxs.back().push_back(track - l1Tracks.data());
        branchData.z0.push_back(vtx->z0());
        branchData.sumPt.push_back(vtx->pt());
      }

      if (printResults_) {
        edm::LogInfo("VertexNTupler") << "analyze::" << recoVertices.size() << " '" << tokenMapEntry.first
                                      << "' vertices were found ... ";
        for (const auto& vtx : recoVertices) {
          edm::LogInfo("VertexNTupler") << "analyze::"
                                        << "  * z0 = " << vtx->z0() << "; contains " << vtx->numTracks()
                                        << " tracks ...";
          for (const auto& trackPtr : vtx->tracks())
            edm::LogInfo("VertexNTupler") << "analyze::"
                                          << "     - z0 = " << trackPtr->z0() << "; pt = " << trackPtr->pt()
                                          << ", eta = " << trackPtr->eta() << ", phi = " << trackPtr->phi0();
        }
      }
    }

    l1Vertices_extra_numTracks_.resize(l1VerticesExtraTokens_.size());
    l1Vertices_extra_z0_.resize(l1VerticesExtraTokens_.size());
    l1Vertices_extra_z0_etaWeighted_.resize(l1VerticesExtraTokens_.size());
    l1Vertices_extra_sumPt_.resize(l1VerticesExtraTokens_.size());

    for (size_t i = 0; i < l1VerticesExtraTokens_.size(); i++) {
      edm::Handle<std::vector<l1t::Vertex>> vertexHandle;
      iEvent.getByToken(l1VerticesExtraTokens_.at(i), vertexHandle);

      const std::vector<l1t::Vertex>& vertices = *vertexHandle;

      l1Vertices_extra_numTracks_.at(i).clear();
      l1Vertices_extra_z0_.at(i).clear();
      l1Vertices_extra_z0_etaWeighted_.at(i).clear();
      l1Vertices_extra_sumPt_.at(i).clear();

      for (const auto& vertex : vertices) {
        l1Vertices_extra_numTracks_.at(i).push_back(vertex.tracks().size());
        l1Vertices_extra_z0_.at(i).push_back(vertex.z0());

        float sumPt = 0.0;
        float etaWeightedSumZ0 = 0.0;
        float etaWeightSum = 0.0;
        for (const auto& track : vertex.tracks()) {
          sumPt += track->momentum().transverse();
          // const float zRes = 0.133616 * track->momentum().eta() * track->momentum().eta() - 0.0522353 * std::abs(track->momentum().eta()) + 0.109918;
          const float zRes = 0.223074 * track->momentum().eta() * track->momentum().eta() -
                             0.050231 * abs(track->momentum().eta()) + 0.209719;
          etaWeightedSumZ0 += track->POCA().z() / (zRes * zRes);
          etaWeightSum += 1.0 / (zRes * zRes);
        }

        l1Vertices_extra_sumPt_.at(i).push_back(sumPt);
        l1Vertices_extra_z0_etaWeighted_.at(i).push_back(etaWeightedSumZ0 / etaWeightSum);
      }
    }

    outputTree_->Fill();
    /////////////////////////////

    if (settings_.debug() > 2)
      edm::LogInfo("VertexNTupler") << "analyze::================ End of Event ==============";
  }

  void VertexNTupler::endJob() {}

  VertexNTupler::~VertexNTupler() {}

}  // namespace l1tVertexFinder

using namespace l1tVertexFinder;

// define this as a plug-in
DEFINE_FWK_MODULE(VertexNTupler);
