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
  typedef std::vector<HitAndPositionPtrContainer> ProtoSegments;

  /// Constructor
  explicit ME0SegmentAlgorithm(const edm::ParameterSet& ps);
  /// Destructor
  ~ME0SegmentAlgorithm() override;

  /**
   * Build segments for all desired groups of hits
   */
  std::vector<ME0Segment> run(const ME0Chamber * chamber, const HitAndPositionContainer& rechits) override;

private:
  /// Utility functions 

  //  Build groups of rechits that are separated in x and y to save time on the segment finding
  ProtoSegments clusterHits(const HitAndPositionContainer& rechits);

  // Build groups of rechits that are separated in strip numbers and Z to save time on the segment finding
  ProtoSegments chainHits(const ME0Chamber * chamber, const HitAndPositionContainer& rechits);

  bool isGoodToMerge(const ME0Chamber * chamber, const HitAndPositionPtrContainer& newChain, const HitAndPositionPtrContainer& oldChain);

  // Build track segments in this chamber (this is where the actual segment-building algorithm hides.)
  void buildSegments(const ME0Chamber * chamber, const HitAndPositionPtrContainer& rechits, std::vector<ME0Segment>& me0segs);

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
  
  static constexpr float running_max=std::numeric_limits<float>::max();
  std::unique_ptr<MuonSegFit> sfit_;

};

#endif
