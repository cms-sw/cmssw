#ifndef RecoMuon_MuonSeedBuilder_H
#define RecoMuon_MuonSeedBuilder_H

/** \class MuonSeedBuilder
 *
 * Algorith to build TrajectorySeed for muon standalone reconstruction.
 * The segments are sorted out to make a protoTrack (vector of matching segments in different 
 * stations a.k.a. layers), for DT+overlap and CSC regions, in that order.  
 * The protoTrack is then passed to the seed creator to create CSC, overlap and/or DT seeds.
 *
 * \author Dominique Fortin - UCR
 *
 */

#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h>
#include <DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h>

//muon service
#include <RecoMuon/TrackingTools/interface/MuonServiceProxy.h>

#include<vector>

class DetLayer; 
class MuonDetLayerGeometry;
class MagneticField;
class MuonSeedCreator;

class MuonSeedBuilder
{
 public:

  typedef MuonTransientTrackingRecHit::MuonRecHitContainer SegmentContainer;
  typedef std::deque<bool> BoolContainer;

  /// Constructor
  explicit MuonSeedBuilder(const edm::ParameterSet&);
  
  /// Destructor
  ~MuonSeedBuilder();
  
  // Operations

  /// Cache pointer to geometry
  void setGeometry( const MuonDetLayerGeometry* lgeom ) {muonLayers = lgeom;}

  /// Cache pointer to Magnetic field
  void setBField( const MagneticField* theField ) {BField = theField;}

  /// Build seed collection
  int build( edm::Event& event, const edm::EventSetup& eventSetup, TrajectorySeedCollection& seeds );

  std::vector<int> badSeedLayer;


 private:

  /// Find segment which matches protoTrack for endcap only
  bool foundMatchingSegment( int type, SegmentContainer& protoTrack, SegmentContainer& segments,
			     BoolContainer& usedSeg, float& eta_temp, float& phi_temp );

  /// cleaning the seeds 
  void seedCleaner(const edm::EventSetup& eventSetup, std::vector<TrajectorySeed>& seeds );   

  /// calculate the eta error from global R error
  double etaError(const GlobalPoint gp, double rErr);

  // This Producer private debug flag
  bool debug;

  // Enable the DT measurement
  bool enableDTMeasurement;

  // Enable the CSC measurement
  bool enableCSCMeasurement;

  // Minimum # of hits to consider a CSC Segment;
  int minCSCHitsPerSegment;

  // Minimum # of hits to consider a DT Segment;
  int minDTHitsPerSegment;

  // Maximum distance from initial segment
  float maxDeltaEtaCSC;
  float maxDeltaPhiCSC;
  float maxDeltaEtaOverlap;
  float maxDeltaPhiOverlap;
  float maxDeltaEtaDT;
  float maxDeltaPhiDT;

  /// Name of the DT segment collection
  edm::InputTag theDTSegmentLabel;

  /// Name of the CSC segment collection
  edm::InputTag theCSCSegmentLabel;

  /// Create seed according to region (CSC, DT, Overlap)
  MuonSeedCreator* muonSeedCreate_;

  // Cache geometry for current event
  const MuonDetLayerGeometry* muonLayers;

  // Cache Magnetic Field for current event
  const MagneticField* BField;
 
  // muon service
  MuonServiceProxy* theService;

  // Minimum separation when we can distinguish between 2 muon seeds
  // (to suppress combinatorics)
  float maxEtaResolutionDT;
  float maxEtaResolutionCSC;
  float maxPhiResolutionDT;
  float maxPhiResolutionCSC;
  float theMinMomentum;

};
#endif

