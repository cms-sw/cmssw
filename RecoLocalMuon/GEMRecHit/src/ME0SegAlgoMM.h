#ifndef GEMRecHit_ME0SegAlgoMM_h
#define GEMRecHit_ME0SegAlgoMM_h

/**
 * \class ME0SegAlgoMM
 *
 * This algorithm is very basic no attemp to deal with ambiguities , noise etc.
 * The ME0 track segments is built out of the rechit's in a the 6 ME0 Layer denoted
 * as the ME0 Ensabmle .<BR>
 *
 *  \authors Marcello Maggi 
 *
 */

#include <RecoLocalMuon/GEMRecHit/src/ME0SegmentAlgorithm.h>
#include <DataFormats/GEMRecHit/interface/ME0RecHit.h>

#include <deque>
#include <vector>

class ME0SegAlgoMM : public ME0SegmentAlgorithm {


public:

  /// Typedefs

  typedef std::vector<const ME0RecHit*> EnsambleHitContainer;
  typedef std::vector<EnsambleHitContainer> ProtoSegments;
  typedef std::deque<bool> BoolContainer;

  /// Constructor
  explicit ME0SegAlgoMM(const edm::ParameterSet& ps);
  /// Destructor
  virtual ~ME0SegAlgoMM();

  /**
   * Build segments for all desired groups of hits
   */
  std::vector<ME0Segment> run(ME0Ensamble ensamble, const EnsambleHitContainer& rechits); 

private:
  /// Utility functions 

  //  Build groups of rechits that are separated in x and y to save time on the segment finding
  ProtoSegments clusterHits(const EnsambleHitContainer & rechits);

  // Build groups of rechits that are separated in strip numbers and Z to save time on the segment finding
  ProtoSegments chainHits(const EnsambleHitContainer & rechits);

  bool isGoodToMerge(EnsambleHitContainer & newChain, EnsambleHitContainer & oldChain);

  // Build track segments in this chamber (this is where the actual segment-building algorithm hides.)
  std::vector<ME0Segment> buildSegments(const EnsambleHitContainer& rechits);

  void doSlopesAndChi2();
  void fitSlopes();
  void fillChiSquared();
  void fillLocalDirection();
  CLHEP::HepMatrix derivativeMatrix(void);
  AlgebraicSymMatrix weightMatrix(void);
  AlgebraicSymMatrix calculateError(void);
  void flipErrors(AlgebraicSymMatrix& protoErrors);

  // Member variables
 private:
  const std::string myName; 

  // input from .cfi file
 private:
  bool    debug;
  unsigned int     minHitsPerSegment;
  bool    preClustering;
  double  dXclusBoxMax;
  double  dYclusBoxMax;
  bool    preClustering_useChaining;
  double  dPhiChainBoxMax;
  double  dEtaChainBoxMax;
  int     maxRecHitsInCluster;
  
 private:
  EnsambleHitContainer proto_segment;
  ME0Ensamble theEnsamble;
  LocalPoint protoIntercept;
  float protoSlope_u;
  float protoSlope_v;
  double protoChi2;
  double protoNDF;
  LocalVector protoDirection;
  double protoChiUCorrection;

  std::vector<double> e_Cxx;
};

#endif
