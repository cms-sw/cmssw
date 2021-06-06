#ifndef RecoLocalMuon_GEMSegment_GE0SegAlgoRU_h
#define RecoLocalMuon_GEMSegment_GE0SegAlgoRU_h

/**
 * \class GE0SegAlgoRU
 * adapted from CSC to ME0 bt Marcello Maggi
 *  and ME0 to GE0 by Ian J. Watson
 *
 * This is the original algorithm for building endcap muon track segments
 * out of the rechit's in a GE0Chamber
 * 'RU' = 'RUssia' = Road Usage
 *
 * A GEMSegment is a RecSegment4D, and is built from
 * GEMRecHit2D objects, each of which is a RecHit2DLocalPos. <BR>
 *
 * This class is used by the GEMSegmentAlgorithm. <BR>
 * Alternative algorithms can be used for the segment building
 * by writing classes like this, and then selecting which one is actually
 * used via the GEMSegmentBuilder. <BR>
 *
 * developed and implemented by Vladimir Palichik <Vladimir.Paltchik@cern.ch>
 *                          and Nikolay Voytishin <nikolay.voytishin@cern.ch>
 */

#include <RecoLocalMuon/GEMSegment/plugins/GEMSegmentAlgorithmBase.h>
#include <DataFormats/GEMRecHit/interface/GEMRecHit.h>
#include "MuonSegFit.h"

#include <vector>

class MuonSegFit;
class GE0SegAlgoRU : public GEMSegmentAlgorithmBase {
public:
  struct SegmentParameters {
    float maxETASeeds;
    float maxPhiSeeds;
    float maxPhiAdditional;
    float maxChi2Additional;
    float maxChi2Prune;
    float maxChi2GoodSeg;
    bool requireCentralBX;
    unsigned int minNumberOfHits;
    unsigned int maxNumberOfHits;
    unsigned int maxNumberOfHitsPerLayer;
    bool requireBeamConstr;
  };

  // originally from ME0SegmentAlgorithmBase
  struct HitAndPosition {
    HitAndPosition(const GEMRecHit* rh, const LocalPoint& lp, const GlobalPoint& gp, unsigned int idx)
        : rh(rh), lp(lp), gp(gp), layer(rh->gemId().layer()), idx(idx) {}
    const GEMRecHit* rh;
    LocalPoint lp;
    GlobalPoint gp;
    unsigned int layer;
    unsigned int idx;
  };
  typedef std::vector<HitAndPosition> HitAndPositionContainer;
  typedef std::vector<const HitAndPosition*> HitAndPositionPtrContainer;

  // We need to be able to flag a hit as 'used' and so need a container of bool's.
  typedef std::vector<bool> BoolContainer;
  typedef std::vector<std::pair<float, HitAndPositionPtrContainer> > SegmentByMetricContainer;

  /// Constructor
  explicit GE0SegAlgoRU(const edm::ParameterSet& ps);
  /// Destructor
  ~GE0SegAlgoRU() override{};

  /**
   * Here we must implement the algorithm
   */
  std::vector<GEMSegment> run(const GEMSuperChamber* chamber, const HitAndPositionContainer& rechits);
  // The original ME0SegAlgoRU used the run(..) function above, we
  // implement a small wrapper to use with GEMSegments fairly
  // transparently
  std::vector<GEMSegment> run(const GEMEnsemble& ensemble, const std::vector<const GEMRecHit*>& rechits) override;

private:
  //Look for segments that have at least "n_seg_min" consituents and following the associated paramters
  void lookForSegments(const SegmentParameters& params,
                       const unsigned int n_seg_min,
                       const HitAndPositionContainer& rechits,
                       const std::vector<unsigned int>& recHits_per_layer,
                       BoolContainer& used,
                       std::vector<GEMSegment>& segments) const;
  //Look for any hits between the two seed hits consistent with a segment
  void tryAddingHitsToSegment(const float maxETA,
                              const float maxPhi,
                              const float maxChi2,
                              std::unique_ptr<MuonSegFit>& current_fit,
                              HitAndPositionPtrContainer& proto_segment,
                              const BoolContainer& used,
                              HitAndPositionContainer::const_iterator i1,
                              HitAndPositionContainer::const_iterator i2) const;
  //Remove extra hits until the segment passes "maxChi2"
  void pruneBadHits(const float maxChi2,
                    HitAndPositionPtrContainer& proto_segment,
                    std::unique_ptr<MuonSegFit>& fit,
                    const unsigned int n_seg_min) const;
  //Remove any overlapping segments by which has the lowset chi2
  void addUniqueSegments(SegmentByMetricContainer& proto_segments,
                         std::vector<GEMSegment>& segments,
                         BoolContainer& used) const;

  //Are the two seed hits consistent spatially?
  bool areHitsCloseInEta(const float maxETA, const bool beamConst, const GlobalPoint& h1, const GlobalPoint& h2) const;
  bool areHitsCloseInGlobalPhi(const float maxPHI,
                               const unsigned int nLayDisp,
                               const GlobalPoint& h1,
                               const GlobalPoint& h2) const;

  //Add a hit to a segment
  std::unique_ptr<MuonSegFit> addHit(HitAndPositionPtrContainer& proto_segment, const HitAndPosition& aHit) const;
  //Does the segment have any hits on this layer?
  bool hasHitOnLayer(const HitAndPositionPtrContainer& proto_segment, const unsigned int layer) const;
  //Produce a new fit
  std::unique_ptr<MuonSegFit> makeFit(const HitAndPositionPtrContainer& proto_segment) const;

  //Is this new hit btw the seeds near the segment fit?
  bool isHitNearSegment(const float maxETA,
                        const float maxPHI,
                        const std::unique_ptr<MuonSegFit>& fit,
                        const HitAndPositionPtrContainer& proto_segment,
                        const HitAndPosition& h) const;
  //Return a chi2 for a hit and a predicted segment extrapolation
  float getHitSegChi2(const std::unique_ptr<MuonSegFit>& fit, const GEMRecHit& hit) const;
  //Global point of a segment extrapolated to a Z value
  GlobalPoint globalAtZ(const std::unique_ptr<MuonSegFit>& fit, float z) const;

  //Try adding a hit instead of another and return new or old, depending on which has the smallest chi2
  void compareProtoSegment(std::unique_ptr<MuonSegFit>& current_fit,
                           HitAndPositionPtrContainer& current_proto_segment,
                           const HitAndPosition& new_hit) const;
  //Try adding this hit to the segment, dont if the new chi2 is too big
  void increaseProtoSegment(const float maxChi2,
                            std::unique_ptr<MuonSegFit>& current_fit,
                            HitAndPositionPtrContainer& current_proto_segment,
                            const HitAndPosition& new_hit) const;

  const std::string myName;
  bool doCollisions;
  bool allowWideSegments;

  SegmentParameters stdParameters;
  SegmentParameters displacedParameters;
  SegmentParameters wideParameters;

  //Objects used to produce the segments
  const GEMSuperChamber* theChamber;
};

#endif
