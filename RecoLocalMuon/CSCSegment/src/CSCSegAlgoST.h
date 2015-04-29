#ifndef CSCSegment_CSCSegAlgoST_h
#define CSCSegment_CSCSegAlgoST_h

/**
 * \class CSCSegAlgoST
 *
 * This algorithm is based on the Minimum Spanning Tree (ST) approach 
 * for building endcap muon track segments out of the rechit's in a CSCChamber.<BR>
 *
 * A CSCSegment is a RecSegment4D, and is built from
 * CSCRecHit2D objects, each of which is a RecHit2DLocalPos. <BR>
 *
 * This builds segments consisting of at least 3 hits.
 * Segments can share a common rechit, but only one.
 * 
 *  \authors S. Stoynev  - NWU
 *           I. Bloch    - FNAL
 *           E. James    - FNAL
 *           A. Sakharov - WSU (extensive revision to handle weird segments)
 *           ... ... ...
 *           T. Cox      - UC Davis (struggling to handle this monster)
 *
 */

#include <RecoLocalMuon/CSCSegment/src/CSCSegmentAlgorithm.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <deque>
#include <vector>

class CSCSegAlgoShowering;
class CSCSegAlgoST : public CSCSegmentAlgorithm {


public:

  /// Typedefs

  typedef std::vector<const CSCRecHit2D*> ChamberHitContainer;
  typedef std::vector < std::vector<const CSCRecHit2D* > > Segments;
  typedef std::deque<bool> BoolContainer;

  /// Constructor
  explicit CSCSegAlgoST(const edm::ParameterSet& ps);

  /// Destructor
  virtual ~CSCSegAlgoST();

  /**
   * Build track segments in this chamber (this is where the actual
   * segment-building algorithm hides.)
   */
  std::vector<CSCSegment> buildSegments(const ChamberHitContainer& rechits);

  /**
   * Build track segments in this chamber (this is where the actual
   * segment-building algorithm hides.)
   */
  std::vector<CSCSegment> buildSegments2(const ChamberHitContainer& rechits);

  /**
   * Build segments for all desired groups of hits
   */
  std::vector<CSCSegment> run(const CSCChamber* aChamber, const ChamberHitContainer& rechits); 

  /**
   * Build groups of rechits that are separated in x and y to save time on the segment finding
   */
  std::vector< std::vector<const CSCRecHit2D*> > clusterHits(const CSCChamber* aChamber, const ChamberHitContainer & rechits);


   /* Build groups of rechits that are separated in strip numbers and Z to save time on the segment finding
   */
     std::vector< std::vector<const CSCRecHit2D*> > chainHits(const CSCChamber* aChamber, const ChamberHitContainer & rechits);


  /**
   * Remove bad hits from found segments based not only on chi2, but also on charge and 
   * further "low level" chamber information.
   */
  std::vector< CSCSegment > prune_bad_hits(const CSCChamber* aChamber, std::vector< CSCSegment > & segments);

private:

  // Retrieve pset
  const edm::ParameterSet& pset(void) const { return ps_;}

  // Adjust covariance matrix?
  bool adjustCovariance(void) { return adjustCovariance_;}

  /// Utility functions 
  double theWeight(double coordinate_1, double coordinate_2, double coordinate_3, float layer_1, float layer_2, float layer_3);

  void ChooseSegments(void);

  // Return the segment with the smallest weight
  void ChooseSegments2a(std::vector< ChamberHitContainer > & best_segments, int best_seg);
  // Version of ChooseSegments for the case without fake hits
  void ChooseSegments2(int best_seg);

  // Choose routine with reduce nr of loops
  void ChooseSegments3(int best_seg);
  void ChooseSegments3(std::vector< ChamberHitContainer > & best_segments, std::vector< float > & best_weight, int best_seg);
  //

  // Find duplicates in ME1/1a, if it has ganged strips (i.e. pre-LS1)
  void findDuplicates(std::vector<CSCSegment>  & segments );

  bool isGoodToMerge(bool isME11a, ChamberHitContainer & newChain, ChamberHitContainer & oldChain);

  void dumpSegment( const CSCSegment& seg ) const;
  const CSCChamber* chamber() const {return theChamber;}

  // Member variables
  const std::string myName_; 
  const edm::ParameterSet ps_;
  CSCSegAlgoShowering* showering_;

  const CSCChamber* theChamber;
  Segments GoodSegments;

  ChamberHitContainer PAhits_onLayer[6];
  ChamberHitContainer Psegments_hits;

  std::vector< ChamberHitContainer > Psegments;
  std::vector< ChamberHitContainer > Psegments_noLx;
  std::vector< ChamberHitContainer > Psegments_noL1;
  std::vector< ChamberHitContainer > Psegments_noL2;
  std::vector< ChamberHitContainer > Psegments_noL3;
  std::vector< ChamberHitContainer > Psegments_noL4;
  std::vector< ChamberHitContainer > Psegments_noL5;
  std::vector< ChamberHitContainer > Psegments_noL6;
  std::vector< ChamberHitContainer > chosen_Psegments;
  std::vector< float > weight_A;
  std::vector< float > weight_noLx_A;
  std::vector< float > weight_noL1_A;
  std::vector< float > weight_noL2_A;
  std::vector< float > weight_noL3_A;
  std::vector< float > weight_noL4_A;
  std::vector< float > weight_noL5_A;
  std::vector< float > weight_noL6_A;
  std::vector< float > chosen_weight_A;
  std::vector< float > curv_A;
  std::vector< float > curv_noL1_A;
  std::vector< float > curv_noL2_A;
  std::vector< float > curv_noL3_A;
  std::vector< float > curv_noL4_A;
  std::vector< float > curv_noL5_A;
  std::vector< float > curv_noL6_A;
  std::vector< float > weight_B;
  std::vector< float > weight_noL1_B;
  std::vector< float > weight_noL2_B;
  std::vector< float > weight_noL3_B;
  std::vector< float > weight_noL4_B;
  std::vector< float > weight_noL5_B;
  std::vector< float > weight_noL6_B;

  ChamberHitContainer protoSegment;

  // input from .cfi file
  bool    debug;
  //  int     minLayersApart;
  //  double  nSigmaFromSegment;
  int     minHitsPerSegment;
  //  int     muonsPerChamberMax;
  //  double  chi2Max;
  double  dXclusBoxMax;
  double  dYclusBoxMax;
  int     maxRecHitsInCluster;
  bool    preClustering;
  bool    preClustering_useChaining;
  bool    Pruning;
  bool    BrutePruning;
  double  BPMinImprovement;
  bool    onlyBestSegment;
  bool    useShowering;

  double  hitDropLimit4Hits;
  double  hitDropLimit5Hits;
  double  hitDropLimit6Hits;

  float a_yweightPenaltyThreshold[5][5];

  double  yweightPenaltyThreshold;
  double  yweightPenalty;

  double  curvePenaltyThreshold;
  double  curvePenalty;


  bool adjustCovariance_;       /// Flag whether to 'improve' covariance matrix

  bool condpass1, condpass2;

  double chi2Norm_3D_;           /// Chi^2 normalization for the initial fit

  bool prePrun_;                 /// Allow to prune a (rechit in a) segment in segment buld method
                                 /// once it passed through Chi^2-X and  chi2uCorrection is big.
  double prePrunLimit_;          /// The upper limit of protoChiUCorrection to apply prePrun

};

#endif
