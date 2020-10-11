//
// Package:         RecoTracker/FinalTrackSelectors
// Class:           TrackListMerger
//
// Description:     Hit Dumper
//
// Original Author: Steve Wagner, stevew@pizero.colorado.edu
// Created:         Sat Jan 14 22:00:00 UTC 2006
//
//

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "RecoTracker/FinalTrackSelectors/interface/TrackAlgoPriorityOrder.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

class dso_hidden TrackListMerger : public edm::stream::EDProducer<> {
public:
  explicit TrackListMerger(const edm::ParameterSet& conf);

  ~TrackListMerger() override;

  void produce(edm::Event& e, const edm::EventSetup& c) override;

private:
  using MVACollection = std::vector<float>;
  using QualityMaskCollection = std::vector<unsigned char>;

  std::unique_ptr<reco::TrackCollection> outputTrks;
  std::unique_ptr<reco::TrackExtraCollection> outputTrkExtras;
  std::unique_ptr<TrackingRecHitCollection> outputTrkHits;
  std::unique_ptr<std::vector<Trajectory>> outputTrajs;
  std::unique_ptr<TrajTrackAssociationCollection> outputTTAss;
  std::unique_ptr<TrajectorySeedCollection> outputSeeds;

  reco::TrackRefProd refTrks;
  reco::TrackExtraRefProd refTrkExtras;
  TrackingRecHitRefProd refTrkHits;
  edm::RefProd<std::vector<Trajectory>> refTrajs;
  edm::RefProd<TrajectorySeedCollection> refTrajSeeds;

  bool copyExtras_;
  bool makeReKeyedSeeds_;

  edm::ESGetToken<TrackAlgoPriorityOrder, CkfComponentsRecord> priorityToken;

  struct TkEDGetTokenss {
    edm::InputTag tag;
    edm::EDGetTokenT<reco::TrackCollection> tk;
    edm::EDGetTokenT<std::vector<Trajectory>> traj;
    edm::EDGetTokenT<TrajTrackAssociationCollection> tass;
    edm::EDGetTokenT<edm::ValueMap<int>> tsel;
    edm::EDGetTokenT<edm::ValueMap<float>> tmva;
    TkEDGetTokenss() {}
    TkEDGetTokenss(const edm::InputTag& tag_,
                   edm::EDGetTokenT<reco::TrackCollection>&& tk_,
                   edm::EDGetTokenT<std::vector<Trajectory>>&& traj_,
                   edm::EDGetTokenT<TrajTrackAssociationCollection>&& tass_,
                   edm::EDGetTokenT<edm::ValueMap<int>>&& tsel_,
                   edm::EDGetTokenT<edm::ValueMap<float>>&& tmva_)
        : tag(tag_), tk(tk_), traj(traj_), tass(tass_), tsel(tsel_), tmva(tmva_) {}
  };
  TkEDGetTokenss edTokens(const edm::InputTag& tag, const edm::InputTag& seltag, const edm::InputTag& mvatag) {
    return TkEDGetTokenss(tag,
                          consumes<reco::TrackCollection>(tag),
                          consumes<std::vector<Trajectory>>(tag),
                          consumes<TrajTrackAssociationCollection>(tag),
                          consumes<edm::ValueMap<int>>(seltag),
                          consumes<edm::ValueMap<float>>(mvatag));
  }
  TkEDGetTokenss edTokens(const edm::InputTag& tag, const edm::InputTag& mvatag) {
    return TkEDGetTokenss(tag,
                          consumes<reco::TrackCollection>(tag),
                          consumes<std::vector<Trajectory>>(tag),
                          consumes<TrajTrackAssociationCollection>(tag),
                          edm::EDGetTokenT<edm::ValueMap<int>>(),
                          consumes<edm::ValueMap<float>>(mvatag));
  }
  std::vector<TkEDGetTokenss> trackProducers_;

  std::string priorityName_;

  double maxNormalizedChisq_;
  double minPT_;
  unsigned int minFound_;
  float epsilon_;
  float shareFrac_;
  float foundHitBonus_;
  float lostHitPenalty_;
  std::vector<double> indivShareFrac_;

  std::vector<std::vector<int>> listsToMerge_;
  std::vector<bool> promoteQuality_;
  std::vector<int> hasSelector_;
  bool copyMVA_;

  bool allowFirstHitShare_;
  reco::TrackBase::TrackQuality qualityToSet_;
  bool use_sharesInput_;
  bool trkQualMod_;
};

#include <memory>
#include <string>
#include <iostream>
#include <cmath>
#include <vector>

#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1DCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

//#include "DataFormats/TrackReco/src/classes.h"

#include "TrackingTools/PatternTools/interface/ClusterRemovalRefSetter.h"

#ifdef STAT_TSB
#include <x86intrin.h>
#endif

namespace {
#ifdef STAT_TSB
  inline volatile unsigned long long rdtsc() { return __rdtsc(); }

  struct StatCount {
    float maxDP = 0.;
    float maxDE = 0.;
    unsigned long long st;
    long long totBegin = 0;
    long long totPre = 0;
    long long totEnd = 0;
    unsigned long long timeNo;  // no-overlap
    unsigned long long timeOv;  // overlap
    void begin(int tt) { totBegin += tt; }
    void start() { st = rdtsc(); }
    void noOverlap() { timeNo += (rdtsc() - st); }
    void overlap() { timeOv += (rdtsc() - st); }
    void pre(int tt) { totPre += tt; }
    void end(int tt) { totEnd += tt; }
    void de(float d) {
      if (d > maxDE)
        maxDE = d;
    }
    void dp(float d) {
      if (d > maxDP)
        maxDP = d;
    }

    void print() const {
      std::cout << "TrackListMerger stat\nBegin/Pre/End/maxDPhi/maxDEta/Overlap/NoOverlap " << totBegin << '/' << totPre
                << '/' << totEnd << '/' << maxDP << '/' << maxDE << '/' << timeOv / 1000 << '/' << timeNo / 1000
                << std::endl;
    }
    StatCount() {}
    ~StatCount() { print(); }
  };
  StatCount statCount;

#else
  struct StatCount {
    void begin(int) {}
    void pre(int) {}
    void end(int) {}
    void start() {}
    void noOverlap() {}
    void overlap() {}
    void de(float) {}
    void dp(float) {}
  };
  CMS_THREAD_SAFE StatCount statCount;
#endif

}  // namespace

namespace {
  edm::ProductID clusterProductB(const TrackingRecHit* hit) {
    return reinterpret_cast<const BaseTrackerRecHit*>(hit)->firstClusterRef().id();
  }
}  // namespace

TrackListMerger::TrackListMerger(edm::ParameterSet const& conf) {
  copyExtras_ = conf.getUntrackedParameter<bool>("copyExtras", true);
  priorityName_ = conf.getParameter<std::string>("trackAlgoPriorityOrder");

  std::vector<edm::InputTag> trackProducerTags(conf.getParameter<std::vector<edm::InputTag>>("TrackProducers"));
  //which of these do I need to turn into vectors?
  maxNormalizedChisq_ = conf.getParameter<double>("MaxNormalizedChisq");
  minPT_ = conf.getParameter<double>("MinPT");
  minFound_ = (unsigned int)conf.getParameter<int>("MinFound");
  epsilon_ = conf.getParameter<double>("Epsilon");
  shareFrac_ = conf.getParameter<double>("ShareFrac");
  allowFirstHitShare_ = conf.getParameter<bool>("allowFirstHitShare");
  foundHitBonus_ = conf.getParameter<double>("FoundHitBonus");
  lostHitPenalty_ = conf.getParameter<double>("LostHitPenalty");
  indivShareFrac_ = conf.getParameter<std::vector<double>>("indivShareFrac");
  std::string qualityStr = conf.getParameter<std::string>("newQuality");
  priorityToken = esConsumes<TrackAlgoPriorityOrder, CkfComponentsRecord>(edm::ESInputTag("", priorityName_));

  if (!qualityStr.empty()) {
    qualityToSet_ = reco::TrackBase::qualityByName(conf.getParameter<std::string>("newQuality"));
  } else
    qualityToSet_ = reco::TrackBase::undefQuality;

  use_sharesInput_ = true;
  if (epsilon_ > 0.0)
    use_sharesInput_ = false;

  edm::VParameterSet setsToMerge = conf.getParameter<edm::VParameterSet>("setsToMerge");

  for (unsigned int i = 0; i < setsToMerge.size(); i++) {
    listsToMerge_.push_back(setsToMerge[i].getParameter<std::vector<int>>("tLists"));
    promoteQuality_.push_back(setsToMerge[i].getParameter<bool>("pQual"));
  }
  hasSelector_ = conf.getParameter<std::vector<int>>("hasSelector");
  copyMVA_ = conf.getParameter<bool>("copyMVA");

  std::vector<edm::InputTag> selectors(conf.getParameter<std::vector<edm::InputTag>>("selectedTrackQuals"));
  std::vector<edm::InputTag> mvaStores;
  if (conf.exists("mvaValueTags")) {
    mvaStores = conf.getParameter<std::vector<edm::InputTag>>("mvaValueTags");
  } else {
    for (int i = 0; i < (int)selectors.size(); i++) {
      edm::InputTag ntag(selectors[i].label(), "MVAVals");
      mvaStores.push_back(ntag);
    }
  }
  unsigned int numTrkColl = trackProducerTags.size();
  if (numTrkColl != hasSelector_.size() || numTrkColl != selectors.size()) {
    throw cms::Exception("Inconsistent size") << "need same number of track collections and selectors";
  }
  if (numTrkColl != hasSelector_.size() || numTrkColl != mvaStores.size()) {
    throw cms::Exception("Inconsistent size") << "need same number of track collections and MVA stores";
  }
  for (unsigned int i = indivShareFrac_.size(); i < numTrkColl; i++) {
    //      edm::LogWarning("TrackListMerger") << "No indivShareFrac for " << trackProducersTags <<". Using default value of 1";
    indivShareFrac_.push_back(1.0);
  }

  trkQualMod_ = conf.getParameter<bool>("writeOnlyTrkQuals");
  if (trkQualMod_) {
    bool ok = true;
    for (unsigned int i = 1; i < numTrkColl; i++) {
      if (!(trackProducerTags[i] == trackProducerTags[0]))
        ok = false;
    }
    if (!ok) {
      throw cms::Exception("Bad input") << "to use writeOnlyTrkQuals=True all input InputTags must be the same";
    }
    produces<edm::ValueMap<int>>();
    produces<QualityMaskCollection>("QualityMasks");
  } else {
    produces<reco::TrackCollection>();

    makeReKeyedSeeds_ = conf.getUntrackedParameter<bool>("makeReKeyedSeeds", false);
    if (makeReKeyedSeeds_) {
      copyExtras_ = true;
      produces<TrajectorySeedCollection>();
    }

    if (copyExtras_) {
      produces<reco::TrackExtraCollection>();
      produces<TrackingRecHitCollection>();
    }
    produces<std::vector<Trajectory>>();
    produces<TrajTrackAssociationCollection>();
  }
  produces<edm::ValueMap<float>>("MVAVals");
  produces<MVACollection>("MVAValues");

  // Do all the consumes
  trackProducers_.resize(numTrkColl);
  for (unsigned int i = 0; i < numTrkColl; ++i) {
    trackProducers_[i] = hasSelector_[i] > 0 ? edTokens(trackProducerTags[i], selectors[i], mvaStores[i])
                                             : edTokens(trackProducerTags[i], mvaStores[i]);
  }
}

// Virtual destructor needed.
TrackListMerger::~TrackListMerger() {}

// Functions that gets called by framework every event
void TrackListMerger::produce(edm::Event& e, const edm::EventSetup& es) {
  // extract tracker geometry
  //
  //edm::ESHandle<TrackerGeometry> theG;
  //es.get<TrackerDigiGeometryRecord>().get(theG);

  //    using namespace reco;

  edm::ESHandle<TrackAlgoPriorityOrder> priorityH = es.getHandle(priorityToken);
  auto const& trackAlgoPriorityOrder = *priorityH;

  // get Inputs
  // if 1 input list doesn't exist, make an empty list, issue a warning, and continue
  // this allows TrackListMerger to be used as a cleaner only if handed just one list
  // if both input lists don't exist, will issue 2 warnings and generate an empty output collection
  //
  static const reco::TrackCollection s_empty;

  std::vector<const reco::TrackCollection*> trackColls;
  std::vector<edm::Handle<reco::TrackCollection>> trackHandles(trackProducers_.size());
  for (unsigned int i = 0; i < trackProducers_.size(); i++) {
    trackColls.push_back(nullptr);
    //edm::Handle<reco::TrackCollection> trackColl;
    e.getByToken(trackProducers_[i].tk, trackHandles[i]);
    if (trackHandles[i].isValid()) {
      trackColls[i] = trackHandles[i].product();
    } else {
      edm::LogWarning("TrackListMerger") << "TrackCollection " << trackProducers_[i].tag << " not found";
      trackColls[i] = &s_empty;
    }
  }

  unsigned int collsSize = trackColls.size();
  unsigned int rSize = 0;
  unsigned int trackCollSizes[collsSize];
  unsigned int trackCollFirsts[collsSize];
  for (unsigned int i = 0; i != collsSize; i++) {
    trackCollSizes[i] = trackColls[i]->size();
    trackCollFirsts[i] = rSize;
    rSize += trackCollSizes[i];
  }

  statCount.begin(rSize);

  //
  //  quality cuts first
  //
  int i = -1;

  int selected[rSize];
  int indexG[rSize];
  bool trkUpdated[rSize];
  int trackCollNum[rSize];
  int trackQuals[rSize];
  float trackMVAs[rSize];
  reco::TrackBase::TrackAlgorithm oriAlgo[rSize];
  std::vector<reco::TrackBase::AlgoMask> algoMask(rSize);
  for (unsigned int j = 0; j < rSize; j++) {
    indexG[j] = -1;
    selected[j] = 1;
    trkUpdated[j] = false;
    trackCollNum[j] = 0;
    trackQuals[j] = 0;
    trackMVAs[j] = -998.0;
    oriAlgo[j] = reco::TrackBase::undefAlgorithm;
  }

  int ngood = 0;
  for (unsigned int j = 0; j != collsSize; j++) {
    const reco::TrackCollection* tC1 = trackColls[j];

    edm::Handle<edm::ValueMap<int>> trackSelColl;
    edm::Handle<edm::ValueMap<float>> trackMVAStore;
    if (copyMVA_)
      e.getByToken(trackProducers_[j].tmva, trackMVAStore);
    if (hasSelector_[j] > 0) {
      e.getByToken(trackProducers_[j].tsel, trackSelColl);
    }

    if (!tC1->empty()) {
      unsigned int iC = 0;
      for (reco::TrackCollection::const_iterator track = tC1->begin(); track != tC1->end(); track++) {
        i++;
        trackCollNum[i] = j;
        trackQuals[i] = track->qualityMask();
        oriAlgo[i] = track->originalAlgo();
        algoMask[i] = track->algoMask();

        reco::TrackRef trkRef = reco::TrackRef(trackHandles[j], iC);
        if (copyMVA_)
          if ((*trackMVAStore).contains(trkRef.id()))
            trackMVAs[i] = (*trackMVAStore)[trkRef];
        if (hasSelector_[j] > 0) {
          int qual = (*trackSelColl)[trkRef];
          if (qual < 0) {
            selected[i] = 0;
            iC++;
            continue;
          } else {
            trackQuals[i] = qual;
          }
        }
        iC++;
        selected[i] = trackQuals[i] + 10;  //10 is magic number used throughout...
        if ((short unsigned)track->ndof() < 1) {
          selected[i] = 0;
          continue;
        }
        if (track->normalizedChi2() > maxNormalizedChisq_) {
          selected[i] = 0;
          continue;
        }
        if (track->found() < minFound_) {
          selected[i] = 0;
          continue;
        }
        if (track->pt() < minPT_) {
          selected[i] = 0;
          continue;
        }
        // good!
        indexG[i] = ngood++;
        //if ( beVerb) std::cout << "inverb " << track->pt() << " " << selected[i] << std::endl;
      }  //end loop over tracks
    }    //end more than 0 track
  }      // loop over trackcolls

  statCount.pre(ngood);

  //cache the id and rechits of valid hits
  typedef std::pair<unsigned int, const TrackingRecHit*> IHit;
  std::vector<std::vector<IHit>> rh1(ngood);  // "not an array" of vectors!
  //const TrackingRecHit*  fh1[ngood];  // first hit...
  reco::TrackBase::TrackAlgorithm algo[ngood];
  float score[ngood];

  for (unsigned int j = 0; j < rSize; j++) {
    if (selected[j] == 0)
      continue;
    int i = indexG[j];
    assert(i >= 0);
    unsigned int collNum = trackCollNum[j];
    unsigned int trackNum = j - trackCollFirsts[collNum];
    const reco::Track* track = &((trackColls[collNum])->at(trackNum));

    algo[i] = track->algo();
    int validHits = track->numberOfValidHits();
    int validPixelHits = track->hitPattern().numberOfValidPixelHits();
    int lostHits = track->numberOfLostHits();
    score[i] =
        foundHitBonus_ * validPixelHits + foundHitBonus_ * validHits - lostHitPenalty_ * lostHits - track->chi2();

    rh1[i].reserve(validHits);
    auto compById = [](IHit const& h1, IHit const& h2) { return h1.first < h2.first; };
    for (trackingRecHit_iterator it = track->recHitsBegin(); it != track->recHitsEnd(); ++it) {
      const TrackingRecHit* hit = (*it);
      unsigned int id = hit->rawId();
      if (hit->geographicalId().subdetId() > 2)
        id &= (~3);  // mask mono/stereo in strips...
      if LIKELY (hit->isValid()) {
        rh1[i].emplace_back(id, hit);
        std::push_heap(rh1[i].begin(), rh1[i].end(), compById);
      }
    }
    std::sort_heap(rh1[i].begin(), rh1[i].end(), compById);
  }

  //DL here
  if LIKELY (ngood > 1 && collsSize > 1)
    for (unsigned int ltm = 0; ltm < listsToMerge_.size(); ltm++) {
      int saveSelected[rSize];
      bool notActive[collsSize];
      for (unsigned int cn = 0; cn != collsSize; ++cn)
        notActive[cn] = find(listsToMerge_[ltm].begin(), listsToMerge_[ltm].end(), cn) == listsToMerge_[ltm].end();

      for (unsigned int i = 0; i < rSize; i++)
        saveSelected[i] = selected[i];

      //DL protect against 0 tracks?
      for (unsigned int i = 0; i < rSize - 1; i++) {
        if (selected[i] == 0)
          continue;
        unsigned int collNum = trackCollNum[i];

        //check that this track is in one of the lists for this iteration
        if (notActive[collNum])
          continue;

        int k1 = indexG[i];
        unsigned int nh1 = rh1[k1].size();
        int qualityMaskT1 = trackQuals[i];

        int nhit1 = nh1;  // validHits[k1];
        float score1 = score[k1];

        // start at next collection
        for (unsigned int j = i + 1; j < rSize; j++) {
          if (selected[j] == 0)
            continue;
          unsigned int collNum2 = trackCollNum[j];
          if ((collNum == collNum2) && indivShareFrac_[collNum] > 0.99)
            continue;
          //check that this track is in one of the lists for this iteration
          if (notActive[collNum2])
            continue;

          int k2 = indexG[j];

          int newQualityMask = -9;  //avoid resetting quality mask if not desired 10+ -9 =1
          if (promoteQuality_[ltm]) {
            int maskT1 = saveSelected[i] > 1 ? saveSelected[i] - 10 : qualityMaskT1;
            int maskT2 = saveSelected[j] > 1 ? saveSelected[j] - 10 : trackQuals[j];
            newQualityMask = (maskT1 | maskT2);  // take OR of trackQuality
          }
          unsigned int nh2 = rh1[k2].size();
          int nhit2 = nh2;

          auto share = use_sharesInput_ ? [](const TrackingRecHit* it, const TrackingRecHit* jt, float) -> bool {
            return it->sharesInput(jt, TrackingRecHit::some);
          }
          : [](const TrackingRecHit* it, const TrackingRecHit* jt, float eps) -> bool {
              float delta = std::abs(it->localPosition().x() - jt->localPosition().x());
              return (it->geographicalId() == jt->geographicalId()) && (delta < eps);
            };

          statCount.start();

          //loop over rechits
          int noverlap = 0;
          int firstoverlap = 0;
          // check first hit  (should use REAL first hit?)
          if UNLIKELY (allowFirstHitShare_ && rh1[k1][0].first == rh1[k2][0].first) {
            const TrackingRecHit* it = rh1[k1][0].second;
            const TrackingRecHit* jt = rh1[k2][0].second;
            if (share(it, jt, epsilon_))
              firstoverlap = 1;
          }

          // exploit sorting
          unsigned int jh = 0;
          unsigned int ih = 0;
          while (ih != nh1 && jh != nh2) {
            // break if not enough to go...
            // if ( nprecut-noverlap+firstoverlap > int(nh1-ih)) break;
            // if ( nprecut-noverlap+firstoverlap > int(nh2-jh)) break;
            auto const id1 = rh1[k1][ih].first;
            auto const id2 = rh1[k2][jh].first;
            if (id1 < id2)
              ++ih;
            else if (id2 < id1)
              ++jh;
            else {
              // in case of split-hit do full conbinatorics
              auto li = ih;
              while ((++li) != nh1 && id1 == rh1[k1][li].first) {
              }
              auto lj = jh;
              while ((++lj) != nh2 && id2 == rh1[k2][lj].first) {
              }
              for (auto ii = ih; ii != li; ++ii)
                for (auto jj = jh; jj != lj; ++jj) {
                  const TrackingRecHit* it = rh1[k1][ii].second;
                  const TrackingRecHit* jt = rh1[k2][jj].second;
                  if (share(it, jt, epsilon_))
                    noverlap++;
                }
              jh = lj;
              ih = li;
            }  // equal ids

          }  //loop over ih & jh

          bool dupfound =
              (collNum != collNum2)
                  ? (noverlap - firstoverlap) > (std::min(nhit1, nhit2) - firstoverlap) * shareFrac_
                  : (noverlap - firstoverlap) > (std::min(nhit1, nhit2) - firstoverlap) * indivShareFrac_[collNum];

          auto seti = [&](unsigned int ii, unsigned int jj) {
            selected[jj] = 0;
            selected[ii] = 10 + newQualityMask;  // add 10 to avoid the case where mask = 1
            trkUpdated[ii] = true;
            if (trackAlgoPriorityOrder.priority(oriAlgo[jj]) < trackAlgoPriorityOrder.priority(oriAlgo[ii]))
              oriAlgo[ii] = oriAlgo[jj];
            algoMask[ii] |= algoMask[jj];
            algoMask[jj] = algoMask[ii];  // in case we keep discarded
          };

          if (dupfound) {
            float score2 = score[k2];
            constexpr float almostSame =
                0.01f;  // difference rather than ratio due to possible negative values for score
            if (score1 - score2 > almostSame) {
              seti(i, j);
            } else if (score2 - score1 > almostSame) {
              seti(j, i);
            } else {
              // If tracks from both iterations are virtually identical, choose the one with the best quality or with lower algo
              if ((trackQuals[j] &
                   (1 << reco::TrackBase::loose | 1 << reco::TrackBase::tight | 1 << reco::TrackBase::highPurity)) ==
                  (trackQuals[i] &
                   (1 << reco::TrackBase::loose | 1 << reco::TrackBase::tight | 1 << reco::TrackBase::highPurity))) {
                //same quality, pick earlier algo
                if (trackAlgoPriorityOrder.priority(algo[k1]) <= trackAlgoPriorityOrder.priority(algo[k2])) {
                  seti(i, j);
                } else {
                  seti(j, i);
                }
              } else if ((trackQuals[j] & (1 << reco::TrackBase::loose | 1 << reco::TrackBase::tight |
                                           1 << reco::TrackBase::highPurity)) <
                         (trackQuals[i] & (1 << reco::TrackBase::loose | 1 << reco::TrackBase::tight |
                                           1 << reco::TrackBase::highPurity))) {
                seti(i, j);
              } else {
                seti(j, i);
              }
            }  //end fi < fj
            statCount.overlap();
            /*
	    if (at0[k1]&&at0[k2]) {
	      statCount.dp(dphi);
	      if (dz<1.f) statCount.de(deta);
	    }
	    */
          }  //end got a duplicate
          else {
            statCount.noOverlap();
          }
          //stop if the ith track is now unselected
          if (selected[i] == 0)
            break;
        }  //end track2 loop
      }    //end track loop
    }      //end loop over track list sets

  auto vmMVA = std::make_unique<edm::ValueMap<float>>();
  edm::ValueMap<float>::Filler fillerMVA(*vmMVA);

  // special case - if just doing the trkquals
  if (trkQualMod_) {
    unsigned int tSize = trackColls[0]->size();
    auto vm = std::make_unique<edm::ValueMap<int>>();
    edm::ValueMap<int>::Filler filler(*vm);

    std::vector<int> finalQuals(tSize, -1);  //default is unselected
    for (unsigned int i = 0; i < rSize; i++) {
      unsigned int tNum = i % tSize;

      if (selected[i] > 1) {
        finalQuals[tNum] = selected[i] - 10;
        if (trkUpdated[i])
          finalQuals[tNum] = (finalQuals[tNum] | (1 << qualityToSet_));
      }
      if (selected[i] == 1)
        finalQuals[tNum] = trackQuals[i];
    }

    filler.insert(trackHandles[0], finalQuals.begin(), finalQuals.end());
    filler.fill();

    e.put(std::move(vm));
    for (auto& q : finalQuals)
      q = std::max(q, 0);
    auto quals = std::make_unique<QualityMaskCollection>(finalQuals.begin(), finalQuals.end());
    e.put(std::move(quals), "QualityMasks");

    std::vector<float> mvaVec(tSize, -99);

    for (unsigned int i = 0; i < rSize; i++) {
      unsigned int tNum = i % tSize;
      mvaVec[tNum] = trackMVAs[tNum];
    }

    fillerMVA.insert(trackHandles[0], mvaVec.begin(), mvaVec.end());
    fillerMVA.fill();
    if (copyMVA_) {
      e.put(std::move(vmMVA), "MVAVals");
      auto mvas = std::make_unique<MVACollection>(mvaVec.begin(), mvaVec.end());
      e.put(std::move(mvas), "MVAValues");
    }
    return;
  }

  //
  //  output selected tracks - if any
  //

  std::vector<reco::TrackRef> trackRefs(rSize);
  std::vector<edm::RefToBase<TrajectorySeed>> seedsRefs(rSize);

  unsigned int nToWrite = 0;
  for (unsigned int i = 0; i < rSize; i++)
    if (selected[i] != 0)
      nToWrite++;

  std::vector<float> mvaVec;

  outputTrks = std::make_unique<reco::TrackCollection>();
  outputTrks->reserve(nToWrite);
  refTrks = e.getRefBeforePut<reco::TrackCollection>();

  if (copyExtras_) {
    outputTrkExtras = std::make_unique<reco::TrackExtraCollection>();
    outputTrkExtras->reserve(nToWrite);
    refTrkExtras = e.getRefBeforePut<reco::TrackExtraCollection>();
    outputTrkHits = std::make_unique<TrackingRecHitCollection>();
    outputTrkHits->reserve(nToWrite * 25);
    refTrkHits = e.getRefBeforePut<TrackingRecHitCollection>();
    if (makeReKeyedSeeds_) {
      outputSeeds = std::make_unique<TrajectorySeedCollection>();
      outputSeeds->reserve(nToWrite);
      refTrajSeeds = e.getRefBeforePut<TrajectorySeedCollection>();
    }
  }

  outputTrajs = std::make_unique<std::vector<Trajectory>>();
  outputTrajs->reserve(rSize);

  for (unsigned int i = 0; i < rSize; i++) {
    if (selected[i] == 0) {
      trackRefs[i] = reco::TrackRef();
      continue;
    }

    unsigned int collNum = trackCollNum[i];
    unsigned int trackNum = i - trackCollFirsts[collNum];
    const reco::Track* track = &((trackColls[collNum])->at(trackNum));
    outputTrks->push_back(reco::Track(*track));
    mvaVec.push_back(trackMVAs[i]);
    if (selected[i] > 1) {
      outputTrks->back().setQualityMask(selected[i] - 10);
      if (trkUpdated[i])
        outputTrks->back().setQuality(qualityToSet_);
    }
    //might duplicate things, but doesnt hurt
    if (selected[i] == 1)
      outputTrks->back().setQualityMask(trackQuals[i]);
    outputTrks->back().setOriginalAlgorithm(oriAlgo[i]);
    outputTrks->back().setAlgoMask(algoMask[i]);

    // if ( beVerb ) std::cout << "selected " << outputTrks->back().pt() << " " << outputTrks->back().qualityMask() << " " << selected[i] << std::endl;

    //fill the TrackCollection
    if (copyExtras_) {
      edm::RefToBase<TrajectorySeed> origSeedRef = track->seedRef();
      //creating a seed with rekeyed clusters if required
      if (makeReKeyedSeeds_) {
        bool doRekeyOnThisSeed = false;

        edm::InputTag clusterRemovalInfos("");
        //grab on of the hits of the seed
        if (origSeedRef->nHits() != 0) {
          TrackingRecHit const& hit = *origSeedRef->recHits().begin();
          if (hit.isValid()) {
            edm::ProductID pID = clusterProductB(&hit);
            // the cluster collection either produced a removalInfo or mot
            //get the clusterremoval info from the provenance: will rekey if this is found
            edm::Handle<reco::ClusterRemovalInfo> CRIh;
            edm::Provenance prov = e.getProvenance(pID);
            clusterRemovalInfos = edm::InputTag(prov.moduleLabel(), prov.productInstanceName(), prov.processName());
            doRekeyOnThisSeed = e.getByLabel(clusterRemovalInfos, CRIh);
          }  //valid hit
        }    //nhit!=0

        if (doRekeyOnThisSeed && !(clusterRemovalInfos == edm::InputTag(""))) {
          ClusterRemovalRefSetter refSetter(e, clusterRemovalInfos);
          TrajectorySeed::RecHitContainer newRecHitContainer;
          newRecHitContainer.reserve(origSeedRef->nHits());
          for (auto const& recHit : origSeedRef->recHits()) {
            newRecHitContainer.push_back(recHit);
            refSetter.reKey(&newRecHitContainer.back());
          }
          outputSeeds->push_back(
              TrajectorySeed(origSeedRef->startingState(), newRecHitContainer, origSeedRef->direction()));
        }
        //doRekeyOnThisSeed=true
        else {
          //just copy the one we had before
          outputSeeds->push_back(TrajectorySeed(*origSeedRef));
        }
        edm::Ref<TrajectorySeedCollection> pureRef(refTrajSeeds, outputSeeds->size() - 1);
        origSeedRef = edm::RefToBase<TrajectorySeed>(pureRef);
      }  //creating a new seed and rekeying it rechit clusters.

      // Fill TrackExtra collection
      outputTrkExtras->push_back(reco::TrackExtra(track->outerPosition(),
                                                  track->outerMomentum(),
                                                  track->outerOk(),
                                                  track->innerPosition(),
                                                  track->innerMomentum(),
                                                  track->innerOk(),
                                                  track->outerStateCovariance(),
                                                  track->outerDetId(),
                                                  track->innerStateCovariance(),
                                                  track->innerDetId(),
                                                  track->seedDirection(),
                                                  origSeedRef));
      seedsRefs[i] = origSeedRef;
      outputTrks->back().setExtra(reco::TrackExtraRef(refTrkExtras, outputTrkExtras->size() - 1));
      reco::TrackExtra& tx = outputTrkExtras->back();
      tx.setResiduals(track->residuals());

      // fill TrackingRecHits
      unsigned nh1 = track->recHitsSize();
      tx.setHits(refTrkHits, outputTrkHits->size(), nh1);
      tx.setTrajParams(track->extra()->trajParams(), track->extra()->chi2sX5());
      assert(tx.trajParams().size() == tx.recHitsSize());
      for (auto hh = track->recHitsBegin(), eh = track->recHitsEnd(); hh != eh; ++hh) {
        outputTrkHits->push_back((*hh)->clone());
      }
    }
    trackRefs[i] = reco::TrackRef(refTrks, outputTrks->size() - 1);

  }  //end faux loop over tracks

  //Fill the trajectories, etc. for 1st collection
  refTrajs = e.getRefBeforePut<std::vector<Trajectory>>();

  outputTTAss = std::make_unique<TrajTrackAssociationCollection>(refTrajs, refTrks);

  for (unsigned int ti = 0; ti < trackColls.size(); ti++) {
    edm::Handle<std::vector<Trajectory>> hTraj1;
    edm::Handle<TrajTrackAssociationCollection> hTTAss1;
    e.getByToken(trackProducers_[ti].traj, hTraj1);
    e.getByToken(trackProducers_[ti].tass, hTTAss1);

    if (hTraj1.failedToGet() || hTTAss1.failedToGet())
      continue;

    for (size_t i = 0, n = hTraj1->size(); i < n; ++i) {
      edm::Ref<std::vector<Trajectory>> trajRef(hTraj1, i);
      TrajTrackAssociationCollection::const_iterator match = hTTAss1->find(trajRef);
      if (match != hTTAss1->end()) {
        const edm::Ref<reco::TrackCollection>& trkRef = match->val;
        uint32_t oldKey = trackCollFirsts[ti] + static_cast<uint32_t>(trkRef.key());
        if (trackRefs[oldKey].isNonnull()) {
          outputTrajs->push_back(*trajRef);
          //if making extras and the seeds at the same time, change the seed ref on the trajectory
          if (copyExtras_ && makeReKeyedSeeds_)
            outputTrajs->back().setSeedRef(seedsRefs[oldKey]);
          outputTTAss->insert(edm::Ref<std::vector<Trajectory>>(refTrajs, outputTrajs->size() - 1), trackRefs[oldKey]);
        }
      }
    }
  }

  statCount.end(outputTrks->size());

  edm::ProductID nPID = refTrks.id();
  edm::TestHandle<reco::TrackCollection> outHandle(outputTrks.get(), nPID);
  fillerMVA.insert(outHandle, mvaVec.begin(), mvaVec.end());
  fillerMVA.fill();

  e.put(std::move(outputTrks));
  if (copyMVA_) {
    e.put(std::move(vmMVA), "MVAVals");
    auto mvas = std::make_unique<MVACollection>(mvaVec.begin(), mvaVec.end());
    e.put(std::move(mvas), "MVAValues");
  }
  if (copyExtras_) {
    e.put(std::move(outputTrkExtras));
    e.put(std::move(outputTrkHits));
    if (makeReKeyedSeeds_)
      e.put(std::move(outputSeeds));
  }
  e.put(std::move(outputTrajs));
  e.put(std::move(outputTTAss));
  return;

}  //end produce

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TrackListMerger);
