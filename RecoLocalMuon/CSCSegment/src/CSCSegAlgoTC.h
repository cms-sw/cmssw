#ifndef CSCSegment_CSCSegAlgoTC_h
#define CSCSegment_CSCSegAlgoTC_h

/**
 * \class CSCSegAlgoTC
 *
 * This is an alternative algorithm for building endcap muon track segments
 * out of the rechit's in a CSCChamber. cf. CSCSegmentizerSK.<BR>
 * 'TC' = 'Tim Cox' = Try (all) Combinations <BR>
 *
 * A CSCSegment isa BasicRecHit4D, and is built from
 * CSCRhit objects, each of which isa BasicRecHit2D. <BR>
 *
 * This class is used by the CSCSegmentRecDet. <BR>
 * Alternative algorithms can be used for the segment building
 * by writing classes like this, and then selecting which one is actually
 * used via the CSCSegmentizerBuilder in CSCDetector. <BR>
 *
 * Ported to CMSSW 2006-04-03: Matteo.Sani@cern.ch <BR>
 *
 * Replaced least-squares fit by CSCSegFit - Feb 2015 <BR>
 *
 */

#include <RecoLocalMuon/CSCSegment/src/CSCSegmentAlgorithm.h>

#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>

#include <deque>
#include <vector>

class CSCSegFit;

class CSCSegAlgoTC : public CSCSegmentAlgorithm {
 public:
  
  /// Typedefs
  typedef std::vector<int> LayerIndex;
  typedef std::vector<const CSCRecHit2D*> ChamberHitContainer;
  typedef ChamberHitContainer::const_iterator ChamberHitContainerCIt;
  
  // We need to be able to flag a hit as 'used' and so need a container
  // of bool's. Naively, this would be vector<bool>... but AVOID that since it's
  // non-standard i.e. packed-bit implementation which is not a standard STL container. 
  // We don't need what it offers and it could lead to unexpected trouble in the future
  
  typedef std::deque<bool> BoolContainer;
  
  /// Constructor
  explicit CSCSegAlgoTC(const edm::ParameterSet& ps);
  /// Destructor
  virtual ~CSCSegAlgoTC() {};
  
  /**
   * Build track segments in this chamber (this is where the actual
   * segment-building algorithm hides.)
   */
  std::vector<CSCSegment> buildSegments(const ChamberHitContainer& rechits);
  
  /**
   * Here we must implement the algorithm
   */
  std::vector<CSCSegment> run(const CSCChamber* aChamber, const ChamberHitContainer& rechits);
  
 private:
  
  /// Utility functions 
  
  bool addHit(const CSCRecHit2D* aHit, int layer);
  bool replaceHit(const CSCRecHit2D* h, int layer);
  void compareProtoSegment(const CSCRecHit2D* h, int layer);
  void increaseProtoSegment(const CSCRecHit2D* h, int layer);
  
  /**
   * Return true if the difference in (local) x of two hits is < dRPhiMax
   */
  bool areHitsCloseInLocalX(const CSCRecHit2D* h1, const CSCRecHit2D* h2) const;
  
  /**
   * Return true if the difference in (global) phi of two hits is < dPhiMax
   */
  bool areHitsCloseInGlobalPhi(const CSCRecHit2D* h1, const CSCRecHit2D* h2) const;
  
  bool hasHitOnLayer(int layer) const;
  
  /**
   * Return true if hit is near segment.
   * 'Near' means deltaphi and rxy*deltaphi are within ranges
   * specified by config parameters dPhiFineMax and dRPhiFineMax,
   * where rxy = sqrt(x**2+y**2) of the hit in global coordinates.
   */
  bool isHitNearSegment(const CSCRecHit2D* h) const;
  
  /**
   * Dump global and local coordinate of each rechit in chamber after sort in z
   */
  void dumpHits(const ChamberHitContainer& rechits) const;
  
  /**
   * Try adding non-used hits to segment
   */
  void tryAddingHitsToSegment(const ChamberHitContainer& rechits, 
			      const ChamberHitContainerCIt i1, 
			      const ChamberHitContainerCIt i2);
  
  /**
   * Return true if segment is good.
   * In this algorithm, this means it shares no hits with any other segment.
   * If "SegmentSort=2" also require a minimal chi2 probability of "chi2ndfProbMin".
   */
  bool isSegmentGood(std::vector<CSCSegFit*>::iterator is, 
		     const ChamberHitContainer& rechitsInChamber,  
		     BoolContainer& used) const;
  
  /**
   * Flag hits on segment as used
   */
  void flagHitsAsUsed(std::vector<CSCSegFit*>::iterator is, 
		      const ChamberHitContainer& rechitsInChamber, BoolContainer& used) const;
  
  /**
   * Order segments by quality (chi2/#hits) and select the best,
   * requiring that they have unique hits.
   */
  void pruneTheSegments(const ChamberHitContainer& rechitsInChamber);
  /**
   * Sort criterion for segment quality, for use in pruneTheSegments.
   */   
  void segmentSort(void);  
  
  float phiAtZ(float z) const;  

  void updateParameters(void);

  void dumpSegment( const CSCSegment& seg ) const;
  
  /// Member variables
  // ================
  
  const CSCChamber* theChamber;

  ChamberHitContainer proto_segment;

  // Pointer to most recent candidate fit
  CSCSegFit* sfit_;

  // Store pointers to set of candidate fits
  std::vector<CSCSegFit*> candidates;
  
  /** max segment chi squared
   */
  float chi2Max;
  
  /** min segment chi squared probability.
   *  Used ONLY if SegmentSorting is chosen to be 2 
   */
  float chi2ndfProbMin;

  /** max hit deviation in r-phi from the segment axis.
   *  Function hitNearSegment requires rxy*abs(deltaphi) < dRPhiFineMax.
   */
  float dRPhiFineMax;
  
  /** max hit deviation in global phi from the segment axis.
   *  Function hitNearSegment requires abs(deltaphi) < dPhiFineMax.
   */
  float dPhiFineMax;
  
  /** max distance in local x between hits in one segment
   * @@ The name is historical!
   */
  float dRPhiMax;
  
  /** max distance in global phi between hits in one segment
   */
  float dPhiMax;
  
  /** Require end-points of segment are at least minLayersApart 
   */
  int minLayersApart;
  
  /** Select which segment sorting to use (the higher the segment 
   *  is in the list, the better the segment is supposed to be): 
   *  if value is ==1: Sort segments by Chi2/(#hits on segment)
   *  if value is ==2: Sort segments first by #hits on segment,
   *  then by Chi2Probability(Chi2/ndf)
   */
  int SegmentSorting;

  /** Name of this class
   */
  const std::string myName;
  bool debugInfo;

};

#endif
