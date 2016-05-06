#ifndef GEMRecHit_GEMSegAlgo_h
#define GEMRecHit_GEMSegAlgo_h

/**
 * \class GEMSegAlgo
 *
 * This algorithm is very basic no attemp to deal with ambiguities , noise etc.
 * The GEM track segments (actually more correct would be: GEM correlated hits)
 * is built out of the rechits in two GEM layers (GE1/1) or three GEM layers (GE2/1)
 * as the GEM Ensabmle .<BR>
 *
 *  \authors Piet Verwilligen 
 *  updated by Jason Lee to use general segment fitter, MuonSegFit
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <RecoLocalMuon/GEMSegment/plugins/GEMSegmentAlgorithm.h>
#include <DataFormats/GEMRecHit/interface/GEMRecHit.h>
#include "FWCore/Utilities/interface/Exception.h"
#include <deque>
#include <vector>

class MuonSegFit;

class GEMSegAlgo : public GEMSegmentAlgorithm {

public:

  /// Typedefs
  typedef std::vector<const GEMRecHit*> EnsembleHitContainer;
  typedef std::vector<EnsembleHitContainer> ProtoSegments;
  typedef std::vector<const TrackingRecHit*> MuonRecHitContainer;

  /// Constructor
  explicit GEMSegAlgo(const edm::ParameterSet& ps);
  /// Destructor
  virtual ~GEMSegAlgo();

  /**
   * Build segments for all desired groups of hits
   */
  std::vector<GEMSegment> run(const GEMEnsemble& ensemble, const EnsembleHitContainer& rechits); 

private:
  /// Utility functions 

  //  Build groups of rechits that are separated in x and y to save time on the segment finding
  ProtoSegments clusterHits(const EnsembleHitContainer& rechits);

  // Build groups of rechits that are separated in strip numbers and Z to save time on the segment finding
  ProtoSegments chainHits(const EnsembleHitContainer& rechits);

  bool isGoodToMerge(const EnsembleHitContainer& newChain, const EnsembleHitContainer& oldChain);

  // Build track segments in this chamber (this is where the actual segment-building algorithm hides.)
  std::vector<GEMSegment> buildSegments(const EnsembleHitContainer& rechits);

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
  bool    clusterOnlySameBXRecHits;
  // bool    useGE21Short;
  
 private:
  EnsembleHitContainer proto_segment;
  GEMEnsemble theEnsemble;
  GEMDetId    theChamberId;

  static constexpr float running_max=999999.;
  std::unique_ptr<MuonSegFit> sfit_;

};

#endif
