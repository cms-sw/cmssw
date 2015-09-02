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

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <RecoLocalMuon/GEMSegment/plugins/ME0SegmentAlgorithm.h>
#include <DataFormats/GEMRecHit/interface/ME0RecHit.h>

#include <deque>
#include <vector>

class ME0SegFit;

class ME0SegAlgoMM : public ME0SegmentAlgorithm {


public:

  /// Typedefs
  typedef std::vector<const ME0RecHit*> EnsembleHitContainer;
  typedef std::vector<EnsembleHitContainer> ProtoSegments;

  /// Constructor
  explicit ME0SegAlgoMM(const edm::ParameterSet& ps);
  /// Destructor
  virtual ~ME0SegAlgoMM();

  /**
   * Build segments for all desired groups of hits
   */
  std::vector<ME0Segment> run(const ME0Ensemble& ensemble, const EnsembleHitContainer& rechits); 

private:
  /// Utility functions 

  //  Build groups of rechits that are separated in x and y to save time on the segment finding
  ProtoSegments clusterHits(const EnsembleHitContainer& rechits);

  // Build groups of rechits that are separated in strip numbers and Z to save time on the segment finding
  ProtoSegments chainHits(const EnsembleHitContainer& rechits);

  bool isGoodToMerge(const EnsembleHitContainer& newChain, const EnsembleHitContainer& oldChain);

  // Build track segments in this chamber (this is where the actual segment-building algorithm hides.)
  std::vector<ME0Segment> buildSegments(const EnsembleHitContainer& rechits);

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
  EnsembleHitContainer proto_segment;
  ME0Ensemble theEnsemble;

  static constexpr float running_max=999999.;
  std::unique_ptr<ME0SegFit> sfit_;

};

#endif
