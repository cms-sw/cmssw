#ifndef CSCSegment_CSCSegAlgoDF_h
#define CSCSegment_CSCSegAlgoDF_h

/**
 * \class CSCSegAlgoDF
 *
 * This is a modified version of the SK algorithm for building endcap 
 * muon track segments out of the rechit's in a CSCChamber.<BR>
 *
 * A CSCSegment is a RecSegment4D, and is built from
 * CSCRecHit2D objects, each of which is a RecHit2DLocalPos. <BR>
 *
 * This builds segments by first creating proto-segments from at least 3 hits.   
 * We intend to try all possible pairs of hits to start segment building. 'All possible'
 * means each hit lies on different layers in the chamber.  Once a hit has been assigned
 * to a segment, we don't consider it again, THAT IS, FOR THE FIRST PASS ONLY !
 * In fact, this is one of the possible flaw with the SK algorithms as it sometimes manages
 * to build segments with the wrong starting points.  In the DF algorithm, the endpoints
 * are tested as the best starting points in a 2nd and 3rd loop.<BR>
 *
 * Another difference with the from the SK algorithm is that rechits can be added to proto
 * segments if they fall within n sigmas of the projected track within a given layer.
 * Hence, a cylinder isn't used as in the SK algorimthm, which allows for pseudo 2D hits
 * built from wire or strip only hits to be used in segment reconstruction.<BR>
 *
 * Also, only a certain muonsPerChamberMax maximum number of segments can be produced in the 
 * chamber. [Seems to be hardwired rather than using this variable?] <BR>
 *
 * Alternative algorithms can be used for the segment building
 * by writing classes like this, and then selecting which one is actually
 * used via the CSCSegmentBuilder. <BR>
 *
 *
 *  \author Dominique Fortin - UCR
 *
 */

#include <RecoLocalMuon/CSCSegment/src/CSCSegmentAlgorithm.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>

#include <deque>
#include <vector>

class CSCSegAlgoPreClustering;
class CSCSegAlgoShowering;
class CSCSegFit;

class CSCSegAlgoDF : public CSCSegmentAlgorithm {

public:

    
  /// Typedefs
    
  typedef std::vector<int> LayerIndex;
  typedef std::vector<const CSCRecHit2D*> ChamberHitContainer;
  typedef std::vector<const CSCRecHit2D*>::const_iterator ChamberHitContainerCIt;
  typedef std::deque<bool> BoolContainer;
    
  /// Constructor
  explicit CSCSegAlgoDF(const edm::ParameterSet& ps);

  /// Destructor
  virtual ~CSCSegAlgoDF();

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

  /**
   * Try adding non-used hits to segment<BR>
   * Skip the layers containing the segment endpoints on first 2 passes, but then       <BR>
   * try hits on layer containing the segment starting points on 2nd and/or 3rd pass    <BR>
   * if segment has >2 hits.  Test each hit on the other layers to see if it is near    <BR>
   * the segment using rechit error matrix.                                             <BR>
   * If it is, see whether there is already a hit on the segment from the same layer    <BR>
   *    - if so, and there are more than 2 hits on the segment, copy the segment,       <BR>
   *      replace the old hit with the new hit. If the new segment chi2 is better       <BR>
   *      then replace the original segment with the new one                            <BR>
   *    - if not, copy the segment, add the hit if it's within a certain range.         <BR>
   */
  void tryAddingHitsToSegment( const ChamberHitContainer& rechitsInChamber,
                               const ChamberHitContainerCIt i1, 
                               const ChamberHitContainerCIt i2,
                               const LayerIndex& layerIndex);

  /**
   * Flag hits on segment as used
   */
  void flagHitsAsUsed(const ChamberHitContainer& rechitsInChamber);
 
  /** 
   * Prune bad segment from the worse hit based on residuals
   */
  void pruneFromResidual(void);

  bool isHitNearSegment(const CSCRecHit2D* h) const;
  bool addHit(const CSCRecHit2D* hit, int layer);
  void updateParameters(void);
  bool hasHitOnLayer(int layer) const;
  void compareProtoSegment(const CSCRecHit2D* h, int layer);
  void dumpSegment( const CSCSegment& seg ) const;

  // Member variables
  const std::string myName; 
  const CSCChamber* theChamber;
  BoolContainer usedHits;

  ChamberHitContainer closeHits;

  ChamberHitContainer protoSegment;
  ChamberHitContainer secondSeedHits;

  // input from .cfi file
  bool   debug;
  bool   preClustering;
  int    minHitsForPreClustering;
  //  bool   testSeg;
  bool   Pruning;
  int    minLayersApart;
  int    nHitsPerClusterIsShower;
  //  float  nSigmaFromSegment;
  int    minHitsPerSegment;
  //  int    muonsPerChamberMax;
  double dRPhiFineMax;
  double dPhiFineMax;
  float tanPhiMax;
  float tanThetaMax;
  float chi2Max;
  float maxRatioResidual;

  CSCSegAlgoPreClustering* preCluster_;
  CSCSegAlgoShowering* showering_;
  CSCSegFit* sfit_;

};

#endif
