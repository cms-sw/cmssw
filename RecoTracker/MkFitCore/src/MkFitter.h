#ifndef RecoTracker_MkFitCore_src_MkFitter_h
#define RecoTracker_MkFitCore_src_MkFitter_h

#include "MkBase.h"
#include "RecoTracker/MkFitCore/interface/Track.h"
#include "RecoTracker/MkFitCore/interface/HitStructures.h"

namespace mkfit {

  class CandCloner;

  static constexpr int MPlexHitIdxMax = 16;
  using MPlexHitIdx = Matriplex::Matriplex<int, MPlexHitIdxMax, 1, NN>;
  using MPlexQHoT = Matriplex::Matriplex<HitOnTrack, 1, 1, NN>;

  class MkFitter : public MkBase {
  public:
    MkFitter() : m_Nhits(0) {}

    // Copy-in timing tests.
    MPlexLS& refErr0() { return m_Err[0]; }
    MPlexLV& refPar0() { return m_Par[0]; }

    //----------------------------------------------------------------------------

    void checkAlignment();

    void printPt(int idx);

    void setNhits(int newnhits) { m_Nhits = std::min(newnhits, Config::nMaxTrkHits - 1); }

    int countValidHits(int itrack, int end_hit) const;
    int countInvalidHits(int itrack, int end_hit) const;
    int countValidHits(int itrack) const { return countValidHits(itrack, m_Nhits); }
    int countInvalidHits(int itrack) const { return countInvalidHits(itrack, m_Nhits); }

    void inputTracksAndHits(const std::vector<Track>& tracks, const std::vector<HitVec>& layerHits, int beg, int end);
    void inputTracksAndHits(const std::vector<Track>& tracks,
                            const std::vector<LayerOfHits>& layerHits,
                            int beg,
                            int end);
    void slurpInTracksAndHits(const std::vector<Track>& tracks, const std::vector<HitVec>& layerHits, int beg, int end);
    void inputTracksAndHitIdx(const std::vector<Track>& tracks, int beg, int end, bool inputProp);
    void inputTracksAndHitIdx(const std::vector<std::vector<Track> >& tracks,
                              const std::vector<std::pair<int, int> >& idxs,
                              int beg,
                              int end,
                              bool inputProp);
    void inputSeedsTracksAndHits(const std::vector<Track>& seeds,
                                 const std::vector<Track>& tracks,
                                 const std::vector<HitVec>& layerHits,
                                 int beg,
                                 int end);

    void inputTracksForFit(const std::vector<Track>& tracks, int beg, int end);
    void fitTracksWithInterSlurp(const std::vector<HitVec>& layersohits, int N_proc);

    void outputTracks(std::vector<Track>& tracks, int beg, int end, int iCP) const;

    void outputFittedTracks(std::vector<Track>& tracks, int beg, int end) const {
      return outputTracks(tracks, beg, end, iC);
    }

    void outputPropagatedTracks(std::vector<Track>& tracks, int beg, int end) const {
      return outputTracks(tracks, beg, end, iP);
    }

    void outputFittedTracksAndHitIdx(std::vector<Track>& tracks, int beg, int end, bool outputProp) const;

    //----------------------------------------------------------------------------

  private:
    MPlexQF m_Chi2;

    MPlexHS m_msErr[Config::nMaxTrkHits];
    MPlexHV m_msPar[Config::nMaxTrkHits];

    MPlexQI m_Label;    //this is the seed index in global seed vector (for MC truth match)
    MPlexQI m_SeedIdx;  //this is the seed index in local thread (for bookkeeping at thread level)
    MPlexQI m_CandIdx;  //this is the candidate index for the given seed (for bookkeeping of clone engine)

    MPlexQHoT m_HoTArr[Config::nMaxTrkHits];

    // Hold hit indices to explore at current layer.
    MPlexQI m_XHitSize;
    MPlexHitIdx m_XHitArr;

    int m_Nhits;
  };

}  // end namespace mkfit

#endif
