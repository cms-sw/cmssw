#ifndef GEMRecHit_GEMSegmentAlgorithm_h
#define GEMRecHit_GEMSegmentAlgorithm_h

/**
 * \class GEMSegmentAlgorithm
 *
 * This algorithm is very basic no attemp to deal with ambiguities , noise etc.
 * The GEM track segments (actually more correct would be: GEM correlated hits)
 * is built out of the rechits in two GEM layers in GE1/1 or GE2/1
 * as the GEM Ensabmle .<BR>
 *
 *  \authors Piet Verwilligen 
 *  updated by Jason Lee to use general segment fitter, MuonSegFit
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoLocalMuon/GEMSegment/plugins/GEMSegmentAlgorithmBase.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHit.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <deque>
#include <vector>

class MuonSegFit;

class GEMSegmentAlgorithm : public GEMSegmentAlgorithmBase {

public:

  /// Typedefs
  typedef std::vector<const GEMRecHit*> EnsembleHitContainer;
  typedef std::vector<EnsembleHitContainer> ProtoSegments;
  
  /// Constructor
  explicit GEMSegmentAlgorithm(const edm::ParameterSet& ps);
  /// Destructor
  ~GEMSegmentAlgorithm() override;

  /**
   * Build segments for all desired groups of hits
   */
  std::vector<GEMSegment> run(const GEMEnsemble& ensemble, const EnsembleHitContainer& rechits) override; 

private:
  /// Utility functions 

  //  Build groups of rechits that are separated in x and y to save time on the segment finding
  ProtoSegments clusterHits(const GEMEnsemble& ensemble, const EnsembleHitContainer& rechits);

  // Build groups of rechits that are separated in strip numbers and Z to save time on the segment finding
  ProtoSegments chainHits(const GEMEnsemble& ensemble, const EnsembleHitContainer& rechits);

  bool isGoodToMerge(const GEMEnsemble& ensemble, const EnsembleHitContainer& newChain, const EnsembleHitContainer& oldChain);

  // Build track segments in this chamber (this is where the actual segment-building algorithm hides.)
  void buildSegments(const GEMEnsemble& ensemble, const EnsembleHitContainer& rechits, std::vector<GEMSegment>& gemsegs);

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
  int     maxRecHitsInCluster;
  bool    clusterOnlySameBXRecHits;
  
  EnsembleHitContainer proto_segment;
  GEMDetId    theChamberId;

  static constexpr float running_max=std::numeric_limits<float>::max();
  std::unique_ptr<MuonSegFit> sfit_;

};

#endif
