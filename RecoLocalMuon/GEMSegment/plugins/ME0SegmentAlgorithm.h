#ifndef GEMRecHit_ME0SegmentAlgorithm_h
#define GEMRecHit_ME0SegmentAlgorithm_h

/**
 * \class ME0SegmentAlgorithm
 *
 * This algorithm is very basic no attemp to deal with ambiguities , noise etc.
 * The ME0 track segments is built out of the rechit's in a the 6 ME0 Layer denoted
 * as the ME0 Ensabmle .<BR>
 *
 *  \authors Marcello Maggi 
 *
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoLocalMuon/GEMSegment/plugins/ME0SegmentAlgorithmBase.h"
#include "DataFormats/GEMRecHit/interface/ME0RecHit.h"

#include <deque>
#include <vector>

class MuonSegFit;

class ME0SegmentAlgorithm : public ME0SegmentAlgorithmBase {


public:

  /// Typedefs
  typedef std::vector<const ME0RecHit*> EnsembleHitContainer;
  typedef std::vector<EnsembleHitContainer> ProtoSegments;

  /// Constructor
  explicit ME0SegmentAlgorithm(const edm::ParameterSet& ps);
  /// Destructor
  virtual ~ME0SegmentAlgorithm();

  /**
   * Build segments for all desired groups of hits
   */
  std::vector<ME0Segment> run(const ME0Ensemble& ensemble, const EnsembleHitContainer& rechits); 

private:
  /// Utility functions 

  //  Build groups of rechits that are separated in x and y to save time on the segment finding
  ProtoSegments clusterHits(const EnsembleHitContainer& rechits);

  // Build groups of rechits that are separated in strip numbers and Z to save time on the segment finding
  ProtoSegments chainHits(const ME0Ensemble& ensemble, const EnsembleHitContainer& rechits);

  bool isGoodToMerge(const ME0Ensemble& ensemble, const EnsembleHitContainer& newChain, const EnsembleHitContainer& oldChain);

  // Build track segments in this chamber (this is where the actual segment-building algorithm hides.)
  void buildSegments(const ME0Ensemble& ensemble, const EnsembleHitContainer& rechits, std::vector<ME0Segment>& me0segs);

  // Member variables
  const std::string myName; 

  // input from .cfi file
  bool    debug;
  unsigned int     minHitsPerSegment;
  bool    preClustering;
  double  dXclusBoxMax;
  double  dYclusBoxMax;
  bool    preClustering_useChaining;
  double  dPhiChainBoxMax;
  double  dEtaChainBoxMax;
  double  dTimeChainBoxMax;
  int     maxRecHitsInCluster;
  
  EnsembleHitContainer proto_segment;

  static constexpr float running_max=std::numeric_limits<float>::max();
  std::unique_ptr<MuonSegFit> sfit_;

};

#endif
