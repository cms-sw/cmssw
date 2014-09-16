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
 * This builds segments consisting of at least 3 hits. It is allowed for segments to have 
 * a common (only one) rechit.  
 * 
 * The program is under construction/testing, as it has been for many years?!
 * As of Aug-2014 replace CLHEP matrices by ROOT::SMatrix in a minimal way.
 *
 *  \authors S. Stoynev  - NWU
 *           I. Bloch    - FNAL
 *           E. James    - FNAL
 *           A. Sakharov - WSU (extensive revision to handle weird segments)
 *           T. Cox      - UC Davis (struggling to handle this monster)
 *
 */

#include <RecoLocalMuon/CSCSegment/src/CSCSegmentAlgorithm.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>

#include <Math/Functions.h>
#include <Math/SVector.h>
#include <Math/SMatrix.h>

#include <deque>
#include <vector>

class CSCSegAlgoShowering;
class CSCSegAlgoST : public CSCSegmentAlgorithm {


public:

  /// Typedefs

  typedef std::vector<const CSCRecHit2D*> ChamberHitContainer;
  typedef std::vector < std::vector<const CSCRecHit2D* > > Segments;
  typedef std::deque<bool> BoolContainer;

  // 12 x12 Symmetric
  typedef ROOT::Math::SMatrix<double,12,12,ROOT::Math::MatRepSym<double,12> > SMatrixSym12;

  // 12 x 4
  typedef ROOT::Math::SMatrix<double,12,4 > SMatrix12by4;

  // 4 x 4 General + Symmetric
  typedef ROOT::Math::SMatrix<double, 4 > SMatrix4;
  typedef ROOT::Math::SMatrix<double,4,4,ROOT::Math::MatRepSym<double,4> > SMatrixSym4;

  // 2 x 2 Symmetric
  typedef ROOT::Math::SMatrix<double,2,2,ROOT::Math::MatRepSym<double,2> > SMatrixSym2;

  // 4-dim vector
  typedef ROOT::Math::SVector<double,4> SVector4;


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
  void fitSlopes(void);
  void fillChiSquared(void);
  void fillLocalDirection(void);
  void doSlopesAndChi2(void);

  // Find duplicates in ME1/1a, if it has ganged strips (i.e. pre-LS1)
  void findDuplicates(std::vector<CSCSegment>  & segments );

  bool isGoodToMerge(bool isME11a, ChamberHitContainer & newChain, ChamberHitContainer & oldChain);

  // THE FOLLOWING BLITHELY PASS MATRICES AROUND LIKE CONFETTI
  // (And that must be fixed!)

  SMatrix12by4 derivativeMatrix(void) const;
  SMatrixSym12 weightMatrix(void) const;
  SMatrixSym4 calculateError(void) const;
  AlgebraicSymMatrix flipErrors(const SMatrixSym4&) const;
  void correctTheCovMatrix(SMatrixSym2& IC);
  void correctTheCovX(void);

  void dumpSegment( const CSCSegment& seg ) const;
  const CSCChamber* chamber() const {return theChamber;}

  // Member variables
  const std::string myName; 
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

  //ibl

  ChamberHitContainer protoSegment;
  float       protoSlope_u;
  float       protoSlope_v;
  LocalPoint  protoIntercept;		
  double      protoChi2;
  double      protoNDF;
  LocalVector protoDirection;

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
  CSCSegAlgoShowering* showering_;

  /// Correct the Error Matrix
  bool correctCov_;              /// Allow to correct the error matrix
  double protoChiUCorrection;
  std::vector<double> e_Cxx;
  double chi2Norm_2D_;           /// Chi^2 normalization for the corrected fit
  double chi2Norm_3D_;           /// Chi^2 normalization for the initial fit
  unsigned maxContrIndex;        /// The index of the worst x RecHit in Chi^2-X method
  bool prePrun_;                 /// Allow to prune a (rechit in a) segment in segment buld method
                                 /// once it passed through Chi^2-X and  protoChiUCorrection is big.
  double prePrunLimit_;          /// The upper limit of protoChiUCorrection to apply prePrun

  /// Correct the error matrix for the condition number
  double condSeed1_, condSeed2_; /// The correction parameters
  bool covToAnyNumber_;          /// Allow to use any number for covariance (by hand)
  bool covToAnyNumberAll_;       /// Allow to use any number for covariance for all RecHits
  double covAnyNumber_;          /// The number to force the Covariance
  bool passCondNumber;           /// Passage the condition number calculations
  bool passCondNumber_2;         /// Passage the condition number calculations
};

#endif
