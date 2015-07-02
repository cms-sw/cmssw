#ifndef GEMRecHit_GEMCSCSegAlgoRR_h
#define GEMRecHit_GEMCSCSegAlgoRR_h

/**
 * \class GEMCSCSegAlgoRR
 *
 * This algorithm is very basic, there is no attempt to deal with ambiguities, such as noise etc.
 * The GEMCSC track segment is built starting from a CSC segment and trying to match GEM rechits.
 * The GEM rechits are searched inside a conigurable (eta,phi) window and associated to the segment.
 * This collection of GEM rechits and CSC semgent is called the GEMCSC Ensemble. <BR>
 *
 *  \authors Raffaella Radogna 
 *
 */

#include <RecoLocalMuon/GEMCSCSegment/plugins/GEMCSCSegmentAlgorithm.h>
#include <DataFormats/GEMRecHit/interface/GEMRecHit.h>
#include <DataFormats/CSCRecHit/interface/CSCSegment.h>

#include <deque>
#include <vector>


class GEMCSCSegFit;

class GEMCSCSegAlgoRR : public GEMCSCSegmentAlgorithm {


public:

  /// Constructor
  explicit GEMCSCSegAlgoRR(const edm::ParameterSet& ps);

  /// Destructor
  ~GEMCSCSegAlgoRR();

  /**
   * Build segments for all desired groups of hits
   */
  std::vector<GEMCSCSegment> run( const std::map<uint32_t, const CSCLayer*>& csclayermap, const std::map<uint32_t, const GEMEtaPartition*>& gemrollmap,
				  const std::vector<const CSCSegment*>& cscsegments, const std::vector<const GEMRecHit*>& gemrechits);
private:

  /// Utility functions 
  /**
   * Search for GEMHits inside a Box around the position extrapolated from the CSC segment.
   */
  std::vector<const TrackingRecHit*> chainHitsToSegm(const CSCSegment* cscsegment, const std::vector<const GEMRecHit*>& gemrechits);

    
  /**
   * Build the GEMCSCSegment.
   */
  std::vector<GEMCSCSegment> buildSegments(const CSCSegment* cscsegment, const std::vector<const TrackingRecHit*>& rechits);


  /// Configuration parameters
  bool         debug;
  unsigned int minHitsPerSegment;
  bool         preClustering;
  double       dXclusBoxMax;
  double       dYclusBoxMax;
  bool         preClustering_useChaining;
  double       dPhiChainBoxMax;
  double       dThetaChainBoxMax;
  double       dRChainBoxMax;
  int          maxRecHitsInCluster;
  
  /// Member variables
  const std::string myName; // name of the algorithm, here: GEMCSCSegAlgoRR

  std::map<uint32_t, const CSCLayer*>        theCSCLayers_;
  std::map<uint32_t, const GEMEtaPartition*> theGEMEtaParts_;
  GEMCSCSegFit*                              sfit_;
};

#endif
