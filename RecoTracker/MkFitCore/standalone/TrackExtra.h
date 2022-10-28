#ifndef RecoTracker_MkFitCore_standalone_TrackExtra_h
#define RecoTracker_MkFitCore_standalone_TrackExtra_h

#include "RecoTracker/MkFitCore/interface/Track.h"

#include <array>
#include <map>
#include <unordered_map>
#include <unordered_set>

namespace mkfit {
  //==============================================================================
  // ReducedTrack
  //==============================================================================

  struct ReducedTrack  // used for cmssw reco track validation
  {
  public:
    ReducedTrack() {}
    ReducedTrack(const int label, const int seedID, const SVector2& params, const float phi, const HitLayerMap& hitmap)
        : label_(label), seedID_(seedID), parameters_(params), phi_(phi), hitLayerMap_(hitmap) {}

    int label() const { return label_; }
    int seedID() const { return seedID_; }
    const SVector2& parameters() const { return parameters_; }
    float momPhi() const { return phi_; }
    const HitLayerMap& hitLayerMap() const { return hitLayerMap_; }

    int label_;
    int seedID_;
    SVector2 parameters_;
    float phi_;
    HitLayerMap hitLayerMap_;
  };
  typedef std::vector<ReducedTrack> RedTrackVec;

  typedef std::map<int, std::map<int, std::vector<int>>> LayIdxIDVecMapMap;
  typedef std::map<int, std::unordered_set<int>> TrkIDLaySetMap;
  typedef std::array<int, 2> PairIdx;
  typedef std::vector<PairIdx> PairIdxVec;
  typedef std::array<int, 3> TripletIdx;
  typedef std::vector<TripletIdx> TripletIdxVec;

  //==============================================================================
  // TrackExtra
  //==============================================================================

  class TrackExtra;
  typedef std::vector<TrackExtra> TrackExtraVec;

  class TrackExtra {
  public:
    TrackExtra() : seedID_(std::numeric_limits<int>::max()) {}
    TrackExtra(int seedID) : seedID_(seedID) {}

    int modifyRefTrackID(const int foundHits,
                         const int minHits,
                         const TrackVec& reftracks,
                         const int trueID,
                         const int duplicate,
                         int refTrackID);
    void setMCTrackIDInfo(const Track& trk,
                          const std::vector<HitVec>& layerHits,
                          const MCHitInfoVec& globalHitInfo,
                          const TrackVec& simtracks,
                          const bool isSeed,
                          const bool isPure);
    void setCMSSWTrackIDInfoByTrkParams(const Track& trk,
                                        const std::vector<HitVec>& layerHits,
                                        const TrackVec& cmsswtracks,
                                        const RedTrackVec& redcmsswtracks,
                                        const bool isBkFit);
    void setCMSSWTrackIDInfoByHits(const Track& trk,
                                   const LayIdxIDVecMapMap& cmsswHitIDMap,
                                   const TrackVec& cmsswtracks,
                                   const TrackExtraVec& cmsswextras,
                                   const RedTrackVec& redcmsswtracks,
                                   const int cmsswlabel);
    int mcTrackID() const { return mcTrackID_; }
    int nHitsMatched() const { return nHitsMatched_; }
    float fracHitsMatched() const { return fracHitsMatched_; }
    int seedID() const { return seedID_; }
    bool isDuplicate() const { return isDuplicate_; }
    int duplicateID() const { return duplicateID_; }
    void setDuplicateInfo(int duplicateID, bool isDuplicate) {
      duplicateID_ = duplicateID;
      isDuplicate_ = isDuplicate;
    }
    int cmsswTrackID() const { return cmsswTrackID_; }
    float helixChi2() const { return helixChi2_; }
    float dPhi() const { return dPhi_; }
    void findMatchingSeedHits(const Track& reco_trk, const Track& seed_trk, const std::vector<HitVec>& layerHits);
    bool isSeedHit(const int lyr, const int idx) const;
    int nMatchedSeedHits() const { return matchedSeedHits_.size(); }

    void setmcTrackID(int mcTrackID) { mcTrackID_ = mcTrackID; }
    void setseedID(int seedID) { seedID_ = seedID; }

    void addAlgo(int algo) { seedAlgos_.push_back(algo); }
    const std::vector<int> seedAlgos() const { return seedAlgos_; }

  private:
    friend class Track;

    int mcTrackID_;
    int nHitsMatched_;
    float fracHitsMatched_;
    int seedID_;
    int duplicateID_;
    bool isDuplicate_;
    int cmsswTrackID_;
    float helixChi2_;
    float dPhi_;
    HoTVec matchedSeedHits_;
    std::vector<int> seedAlgos_;
  };

  typedef std::vector<TrackState> TSVec;
  typedef std::vector<TSVec> TkIDToTSVecVec;
  typedef std::vector<std::pair<int, TrackState>> TSLayerPairVec;
  typedef std::vector<std::pair<int, float>> FltLayerPairVec;  // used exclusively for debugtree

  // Map typedefs needed for mapping different sets of tracks to another
  typedef std::unordered_map<int, int> TkIDToTkIDMap;
  typedef std::unordered_map<int, std::vector<int>> TkIDToTkIDVecMap;
  typedef std::unordered_map<int, TrackState> TkIDToTSMap;
  typedef std::unordered_map<int, TSLayerPairVec> TkIDToTSLayerPairVecMap;

}  // namespace mkfit

#endif
