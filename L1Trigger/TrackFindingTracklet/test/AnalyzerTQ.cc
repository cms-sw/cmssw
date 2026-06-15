#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "SimDataFormats/Associations/interface/TTTypes.h"
#include "SimDataFormats/Associations/interface/StubAssociation.h"
#include "L1Trigger/TrackTrigger/interface/Associator.h"
#include "L1Trigger/TrackFindingTracklet/interface/DataFormats.h"
#include "SimDataFormats/Associations/interface/TTTrackAssociationMap.h"

#include <TProfile.h>
#include <TTree.h>
#include <TH1F.h>

#include <vector>
#include <deque>
#include <map>
#include <set>
#include <cmath>
#include <numeric>
#include <sstream>

namespace trklet {

  /*! \class  trklet::AnalyzerTQ
   *  \brief  Class to analyze emulated Track Quality 
   *  \author Thomas Schuh
   *  \date   2025, Aug
   */
  class AnalyzerTQ : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
  public:
    AnalyzerTQ(const edm::ParameterSet& iConfig);
    void beginJob() override {}
    void beginRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) override;
    void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
    void endRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) override {}
    void endJob() override;

  private:

    struct TrackAssociationResult {
      bool isGenuine;
      TrackingParticlePtr associatedTP;
      int nUnknownStubs;
      int nMatchingTPs;
    };

    TrackAssociationResult associateTrack(const std::vector<TTStubRef>& theseStubs,
                                          const TTClusterAssociationMap<Ref_Phase2TrackerDigi_>& clusterMap,
                                          const TTStubAssociationMap<Ref_Phase2TrackerDigi_>& stubMap);

    // ED input token of stubs
    edm::EDGetTokenT<tt::StreamsStub> edGetTokenStubs_;
    // ED input token of tracks
    edm::EDGetTokenT<tt::StreamsTrack> edGetTokenTracks_;
    // ED output token for stub association for fake rate
    edm::EDGetTokenT<tt::StubAssociation> edGetTokenFake_;
    // ED output token for stub association for tracking efficiency
    edm::EDGetTokenT<tt::StubAssociation> edGetTokenEff_;
    // Associator token
    edm::ESGetToken<tt::Associator, trackerDTC::SetupRcd> esGetTokenAssociator_;
    // Setup token
    edm::ESGetToken<Setup, trackerDTC::SetupRcd> esGetTokenSetup_;
    // DataFormats token
    edm::ESGetToken<DataFormats, trackerDTC::SetupRcd> esGetTokenDataFormats_;
    // TTTrackAssociationMap Token (Failed Attempt to Work with Those below)
    edm::EDGetTokenT<TTTrackAssociationMap<Ref_Phase2TrackerDigi_>> ttTrackMCTruthToken_;
    // TTTracks Token
    edm::EDGetTokenT<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>> ttTrackToken_;
    edm::EDGetTokenT<TTStubAssociationMap<Ref_Phase2TrackerDigi_> > ttStubTruthToken_;
    edm::EDGetTokenT<TTClusterAssociationMap<Ref_Phase2TrackerDigi_> > ttClusterTruthToken_;
    //
    int numMVA_ = 8;
    //
    bool looseMatching_;
    //
    int nEvents_ = 0;
    // Histograms
    std::vector<TProfile*> prof_;
    // printout
    std::stringstream log_;
    // private
    edm::InputTag MCTruthTrackInputTag;
    edm::InputTag L1TrackInputTag;
    // tree
    bool training_run;
    TTree* tree_;
    std::vector<double> f_zT;
    std::vector<double> f_cot;
    std::vector<double> f_chi20;
    std::vector<double> f_chi21;
    std::vector<int> f_hitpattern;
    std::vector<int> f_nstubs;
    std::vector<int> f_ngaps;
    std::vector<int> matched_;
    std::vector<int> matchtp_pdgid;

    // private helper methods
    std::vector<TTStubRef> getStubRefs(int region, int frame, int numRegions, const tt::StreamsStub& streamsStub);
  };

  AnalyzerTQ::AnalyzerTQ(const edm::ParameterSet& iConfig)
      : looseMatching_(iConfig.getParameter<bool>("LooseMatching")) {
    L1TrackInputTag = iConfig.getParameter<edm::InputTag>("L1TrackInputTag");
    MCTruthTrackInputTag = iConfig.getParameter<edm::InputTag>("MCTruthTrackInputTag");
    training_run = iConfig.getParameter<bool>("TrainingMode");
    usesResource("TFileService");
    // book in- and output ED products
    const std::string& labelKF = iConfig.getParameter<std::string>("OutputLabelKF");
    const std::string& labelTQ = iConfig.getParameter<std::string>("OutputLabelTQ");
    const std::string& branchStubs = iConfig.getParameter<std::string>("BranchStubs");
    const std::string& branchTracks = iConfig.getParameter<std::string>("BranchTracks");
    edGetTokenStubs_ = consumes(edm::InputTag(labelKF, branchStubs));
    edGetTokenTracks_ = consumes(edm::InputTag(labelTQ, branchTracks));
    const std::string& labelMC = iConfig.getParameter<std::string>("LabelMC");
    const std::string& branchFake = iConfig.getParameter<std::string>("BranchFake");
    const std::string& branchEff = iConfig.getParameter<std::string>("BranchEff");
    edGetTokenFake_ = consumes(edm::InputTag(labelMC, branchFake));
    edGetTokenEff_ = consumes(edm::InputTag(labelMC, branchEff));
    ttTrackToken_ = consumes<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>>(L1TrackInputTag);
    ttTrackMCTruthToken_ = consumes<TTTrackAssociationMap<Ref_Phase2TrackerDigi_>>(MCTruthTrackInputTag);
    ttClusterTruthToken_ = consumes<TTClusterAssociationMap<Ref_Phase2TrackerDigi_> >(iConfig.getParameter<edm::InputTag>("TTClusterTruth"));
    ttStubTruthToken_ = consumes<TTStubAssociationMap<Ref_Phase2TrackerDigi_> >(iConfig.getParameter<edm::InputTag>("TTStubTruth"));
    // book ES products
    esGetTokenAssociator_ = esConsumes();
    esGetTokenSetup_ = esConsumes();
    esGetTokenDataFormats_ = esConsumes();
    // log config
    log_.setf(std::ios::fixed, std::ios::floatfield);
    log_.precision(4);
  }

  void AnalyzerTQ::beginRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) {
    // book histograms
    edm::Service<TFileService> fs;
    TFileDirectory dir;
    dir = fs->mkdir("TQ");
    prof_ = std::vector<TProfile*>(numMVA_);
    for (int mva = 0; mva < numMVA_; mva++) {
      prof_[mva] = dir.make<TProfile>(("Counts for MVA" + std::to_string(mva)).c_str(), ";", 4, 0.5, 4.5);
      prof_[mva]->GetXaxis()->SetBinLabel(1, "All TPs");
      prof_[mva]->GetXaxis()->SetBinLabel(2, "All Tracks");
      prof_[mva]->GetXaxis()->SetBinLabel(3, "Matched to any Tracks");
      prof_[mva]->GetXaxis()->SetBinLabel(4, "Found Perfect TPs");
    }

    if (training_run) {
      tree_ = fs->make<TTree>("TrackQualityAttributes", "Track Quality Analysis");
      // main 
      tree_->Branch("trk_feature_1", &f_nstubs);
      tree_->Branch("trk_feature_2", &f_zT);
      tree_->Branch("trk_feature_3", &f_cot);
      tree_->Branch("trk_feature_4", &f_chi20);
      tree_->Branch("trk_feature_5", &f_chi21);
      tree_->Branch("trk_feature_6", &f_ngaps);
      tree_->Branch("trk_hitpattern", &f_hitpattern);
      tree_->Branch("trk_matchtp_eventtype", &matched_);
      tree_->Branch("matchtp_pdgid", &matchtp_pdgid);
    }
  }

  void AnalyzerTQ::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

    // TSCHUH PART
    // read in tracks and stubs products
    const tt::StreamsStub& streamsStub = iEvent.get(edGetTokenStubs_);
    const tt::StreamsTrack& streamsTrack = iEvent.get(edGetTokenTracks_);
    const Setup* setup = &iSetup.getData(esGetTokenSetup_);
    const DataFormats* df = &iSetup.getData(esGetTokenDataFormats_);
    // read in MCTruth
    tt::Associator forFake = iSetup.getData(esGetTokenAssociator_);
    tt::Associator forEff = iSetup.getData(esGetTokenAssociator_);
    forFake.consume(iEvent.get(edGetTokenFake_));
    forEff.consume(iEvent.get(edGetTokenEff_));
    for (TProfile* prof : prof_)
      prof->Fill(1, forEff.numTPs());
    // analyze and associate tracks with TrackingParticles per mva categorie
    for (int mva = 0; mva < numMVA_; mva++) {
      std::set<TPPtr> tpPtrsPerfect;
      int nTracks(0);
      int nMatched(0);
      for (int region = 0; region < setup->sysNumRegion(); region++) {
        const tt::StreamTrack& streamTrack = streamsTrack[region * 2 + 1];
        const int numFrames = streamTrack.size();
        for (int frame = 0; frame < numFrames; frame++) {
          if (streamTrack[frame].first.isNull())
            continue;
          const TrackTQ trackTQ(streamTrack[frame], df);
          if (trackTQ.mva() < mva)
            continue;
          nTracks++;
          const int offset = region * setup->kfNumLayers();
          std::vector<TTStubRef> ttStubRefs;
          ttStubRefs.reserve(setup->kfNumLayers());
          for (int layer = 0; layer < setup->kfNumLayers(); layer++) {
            const TTStubRef& ttStubRef = streamsStub[offset + layer][frame].first;
            if (ttStubRef.isNonnull())
              ttStubRefs.push_back(ttStubRef);
          }
          const std::vector<TPPtr>& any =
              looseMatching_ ? forFake.associate(ttStubRefs) : forFake.associateFinal(ttStubRefs);
          if (any.empty())
            continue;
          nMatched++;
          const std::vector<TPPtr> perfect = forEff.associateFinal(ttStubRefs);
          tpPtrsPerfect.insert(perfect.begin(), perfect.end());
        }
      }
      prof_[mva]->Fill(2, nTracks);
      prof_[mva]->Fill(3, nMatched);
      prof_[mva]->Fill(4, tpPtrsPerfect.size());
    }
    // END OF TSCHUH PART

    // AMASTRON PART
    if (training_run) {

      // Attempt to work with those failed miserably.
      edm::Handle<TTTrackAssociationMap<Ref_Phase2TrackerDigi_>> MCTruthTTTrackHandle;
      iEvent.getByToken(ttTrackMCTruthToken_, MCTruthTTTrackHandle);
      edm::Handle<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>> TTTrackHandle;
      iEvent.getByToken(ttTrackToken_, TTTrackHandle);

      // Pull in Stub and Cluster Truth
      edm::Handle<TTClusterAssociationMap<Ref_Phase2TrackerDigi_>> ttClusterAssociationMapHandle;
      iEvent.getByToken(ttClusterTruthToken_, ttClusterAssociationMapHandle);
      edm::Handle<TTStubAssociationMap<Ref_Phase2TrackerDigi_>> ttStubAssociationMapHandle;
      iEvent.getByToken(ttStubTruthToken_, ttStubAssociationMapHandle);

      // Build a map from stub set to L1Track index for fast lookup
      std::map<std::set<TTStubRef>, int> stubSetToTrackIndex;

      if (TTTrackHandle.isValid()) {
          // std::cout << "TTTrackHandle is valid, size = " << TTTrackHandle->size() << std::endl;
          for (size_t idx = 0; idx < TTTrackHandle->size(); ++idx) {
              const auto& ttTrack = (*TTTrackHandle)[idx];
              std::set<TTStubRef> trackStubSet;
              for (const auto& stub : ttTrack.getStubRefs()) {
                  if (stub.isNonnull())
                      trackStubSet.insert(stub);
              }
              stubSetToTrackIndex[trackStubSet] = idx;
              // std::cout << "L1Track " << idx << " has " << trackStubSet.size() << " stubs" << std::endl;
          }
      } else {
          std::cout << "[AnalyzerTQ] TTTrackHandle is NOT valid" << std::endl;
      }

      // clean from previous event
      f_zT.clear();
      f_cot.clear();
      f_chi20.clear();
      f_chi21.clear();
      f_hitpattern.clear();
      f_nstubs.clear();
      f_ngaps.clear();
      matched_.clear();
      matchtp_pdgid.clear();

      for (int region = 0; region < setup->sysNumRegion(); region++) {

        const tt::StreamTrack& streamTrack = streamsTrack[region * 2 + 1];
        const int numFrames = streamTrack.size();

        for (int frame = 0; frame < numFrames; frame++) {

          if (streamTrack[frame].first.isNull())
            continue;

          const TrackTQ trackTQ (streamTrack[frame], df);
          const DataFormat& dfChi20 = df->format(Variable::chi20, Process::tq);
          const DataFormat& dfChi21 = df->format(Variable::chi21, Process::tq);
          const DataFormat& dfZT = df->format(Variable::z0, Process::tq);
          const DataFormat& dfCot = df->format(Variable::cot, Process::tq);

          auto stubRefs = getStubRefs(region, frame, setup->kfNumLayers(), streamsStub);

          std::set<TTStubRef> tqStubSet;
          for (const auto& stub : stubRefs) {
            if (stub.isNonnull())
              tqStubSet.insert(stub);
          }
          
          bool matched = false;
          int matchedIndex = -1;
          auto it = stubSetToTrackIndex.find(tqStubSet);
          if (it != stubSetToTrackIndex.end()) {
            matched = true;
            matchedIndex = it->second;
          }
          
          // Print results
          // std::cout << "TQ Track " << frame << ": " 
          //           << tqStubSet.size() << " stubs, "
          //           << "matched = " << (matched ? "YES" : "NO");
          // if (matched) {
          //     std::cout << " (L1Track index " << matchedIndex << ")";
          // }
          // std::cout << std::endl;

          if (!matched) continue;

          const auto& matchedTrack = (*TTTrackHandle)[matchedIndex];

          const double z0_scale = std::pow(2, 3);
          const double tanL_scale = std::pow(2, 7);
          const double feature_1 = stubRefs.size();
          const double feature_2 = dfZT.integer(trackTQ.z0()) / z0_scale;
          const double feature_3 = dfCot.integer(trackTQ.cot()) / tanL_scale;
          const double feature_4 = dfChi20.integer(trackTQ.chi20()); // leave as is
          const double feature_5 = dfChi21.integer(trackTQ.chi21()); // leave as is
          const TTBV hitPattern = trackTQ.hitPattern();
          const double feature_6 = hitPattern.count(hitPattern.plEncode(), hitPattern.pmEncode(), false);
          
          edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>> l1track_ptr(TTTrackHandle, matchedIndex);
          int tmp_trk_genuine = 0;
          int tmp_matchtp_pdgid = -999;
          
          if (MCTruthTTTrackHandle->isGenuine(l1track_ptr))
            tmp_trk_genuine = 1;
          int tmp_matchtp_eventtype = -999;
          if (tmp_trk_genuine) {
            edm::Ptr<TrackingParticle> my_tp = MCTruthTTTrackHandle->findTrackingParticlePtr(l1track_ptr);
            if (my_tp.isNull())
              assert(false);
            int tmp_eventid = my_tp->eventId().event();
            if (tmp_eventid > 0)
              tmp_matchtp_eventtype = 2;
            else
              tmp_matchtp_eventtype = 1;
            
            tmp_matchtp_pdgid = my_tp->pdgId();
          }
          matched_.push_back(tmp_matchtp_eventtype);
          f_nstubs.push_back(feature_1);
          f_zT.push_back(feature_2);
          f_cot.push_back(feature_3);
          f_chi20.push_back(feature_4);
          f_chi21.push_back(feature_5);
          f_ngaps.push_back(feature_6);
          f_hitpattern.push_back(matchedTrack.hitPattern());
          matchtp_pdgid.push_back(tmp_matchtp_pdgid);
        }
      }
    }
    // END OF AMASTRON PART
    if (training_run) tree_->Fill();
    nEvents_++;
  }

  void AnalyzerTQ::endJob() {
    if (nEvents_ == 0)
      return;
    // printout summary
    log_ << "                         TQ  SUMMARY                         " << std::endl;
    for (int mva = 0; mva < numMVA_; mva++) {
      const double allTracks = prof_[mva]->GetBinContent(2);
      const double allMatched = prof_[mva]->GetBinContent(3);
      const double numPerfect = prof_[mva]->GetBinContent(4);
      const double allTPs = prof_[mva]->GetBinContent(1);
      const double fracFake = (allTracks - allMatched) / allTracks;
      const double effPerfect = numPerfect / allTPs;
      log_ << "mva " << mva << " (effi: " << effPerfect << ", fake rate: " << fracFake << ") numMatched "
           << std::setw(3) << (int)std::round(allMatched) << " numFake " << std::setw(3)
           << (int)std::round(allTracks - allMatched) << std::endl;
    }
    log_ << "=============================================================";
    edm::LogPrint(moduleDescription().moduleName()) << log_.str();
  }

  std::vector<TTStubRef> AnalyzerTQ::getStubRefs(int region, int frame, int numKFLayers, const tt::StreamsStub& streamsStub) {
    const int offset = region * numKFLayers;
    std::vector<TTStubRef> ttStubRefs;
    ttStubRefs.reserve(numKFLayers);
    for (int layer = 0; layer < numKFLayers; layer++) {
      const TTStubRef& ttStubRef = streamsStub[offset + layer][frame].first;
      if (ttStubRef.isNonnull())
        ttStubRefs.push_back(ttStubRef);
    }
    return ttStubRefs;
    }

AnalyzerTQ::TrackAssociationResult AnalyzerTQ::associateTrack(const std::vector<TTStubRef>& theseStubs, const TTClusterAssociationMap<Ref_Phase2TrackerDigi_>& clusterMap, const TTStubAssociationMap<Ref_Phase2TrackerDigi_>& stubMap) {
      
    std::map<const TrackingParticle*, TrackingParticlePtr> auxMap;
    int mayCombinUnknown = 0;
    
    for (const TTStubRef& stub : theseStubs) {
        for (unsigned int ic = 0; ic < 2; ic++) {
            const std::vector<TrackingParticlePtr>& tempTPs =
                clusterMap.findTrackingParticlePtrs(stub->clusterRef(ic));
            
            for (const TrackingParticlePtr& testTP : tempTPs) {
                if (testTP.isNull())
                    continue;
                
                if (auxMap.find(testTP.get()) == auxMap.end()) {
                    auxMap.emplace(testTP.get(), testTP);
                }
            }
        }
        
        if (stubMap.isUnknown(stub))
            ++mayCombinUnknown;
    }
    
    if (mayCombinUnknown > 0) {
        return {false, TrackingParticlePtr(), mayCombinUnknown, 0};
    }
    
    std::vector<const TrackingParticle*> tpInAllStubs;
    
    for (const auto& auxPair : auxMap) {
        const std::vector<TTStubRef>& tempStubs = 
            stubMap.findTTStubRefs(auxPair.second);
        
        int nnotfound = 0;
        for (const TTStubRef& stub : theseStubs) {
            if (std::find(tempStubs.begin(), tempStubs.end(), stub) == tempStubs.end()) {
                ++nnotfound;
                break;
            }
        }
        
        if (nnotfound > 0)
            continue;
        
        tpInAllStubs.push_back(auxPair.first);
    }
    
    unsigned int nTPs = tpInAllStubs.size();
    
    // Strict matching logic:
    // - If exactly 1 TP appears in ALL stubs: GENUINE
    // - If 0 or >= 2 TP: FAKE
    if (nTPs != 1) {
        return {false, TrackingParticlePtr(), mayCombinUnknown, static_cast<int>(nTPs)};
    }
    
    // This track is genuine (strict matching) - return the associated TP
    const TrackingParticle* bestTPptr = tpInAllStubs.at(0);
    TrackingParticlePtr bestTP = auxMap.find(bestTPptr)->second;
    
    return {true, bestTP, mayCombinUnknown, 1};
  }

}  // namespace trklet

DEFINE_FWK_MODULE(trklet::AnalyzerTQ);
