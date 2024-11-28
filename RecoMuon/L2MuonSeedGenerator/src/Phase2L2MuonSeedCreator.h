#ifndef RecoMuon_L2MuonSeedCreator_Phase2L2MuonSeedCreator_H
#define RecoMuon_L2MuonSeedCreator_Phase2L2MuonSeedCreator_H

/** \class Phase2L2MuonSeedCreator
 * 
 *  Standalone Muon seeds for Phase-2
 *  This class takes in input the full L1 Tracker Muon collection
 *  and the collections of (DT/CSC) segments in the muon chambers.
 *  For each L1 Tracker Muon, the stubs used to produce it are 
 *  matched with segments in the muon chambers looking, in order,
 *  at deltaPhi, number of hits, and deltaTheta. All matched segments
 *  are added to the seed, together with the pT information from the
 *  tracker muon itself. Specifically for the barrel region and in 
 *  stations where no stub is found, a simple extrapolation is 
 *  attempted from nearby stations with a match (e.g no stub found in 
 *  station 2, attempt to match segments extrapolating from station
 *  1, 3, and 4 in this order).
 * 
 *  The logic allows a single-step extension to seed displaced muons
 *  (currently not implemented)
 * 
 *  \author Luca Ferragina (INFN BO), 2024
 */

#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeed.h"
#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/L1TMuonPhase2/interface/MuonStub.h"
#include "DataFormats/CSCRecHit/interface/CSCSegment.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include <map>
#include <utility>

class RecHit;
class Plane;
class GeomDet;
class MagneticField;
class MuonTransientTrackingRecHit;

enum Type { barrel, overlap, endcap };

class Phase2L2MuonSeedCreator : public edm::stream::EDProducer<> {
public:
  typedef MuonTransientTrackingRecHit::MuonRecHitContainer SegmentContainer;

  // Constructor
  explicit Phase2L2MuonSeedCreator(const edm::ParameterSet& pset);

  // Destructor
  ~Phase2L2MuonSeedCreator() override = default;

  // Operations
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  // Tokens
  const edm::EDGetTokenT<l1t::TrackerMuonCollection> l1TkMuCollToken_;
  const edm::EDGetTokenT<CSCSegmentCollection> cscSegmentCollToken_;
  const edm::EDGetTokenT<DTRecSegment4DCollection> dtSegmentCollToken_;

  const edm::ESGetToken<CSCGeometry, MuonGeometryRecord> cscGeometryToken_;
  const edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeometryToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldToken_;

  // Miminum and maximum pt momentum of a track
  const double minMomentum_;
  const double maxMomentum_;

  // Parameters to match L1 stubs to DT/CSC segments
  const double matchingPhiWindow_;
  const double matchingThetaWindow_;

  // Parameters to extrapolate matches in nearby stations
  const double extrapolationDeltaPhiClose_;
  const double extrapolationDeltaPhiFar_;

  const double maxEtaBarrel_;   // barrel with |eta| < 0.7
  const double maxEtaOverlap_;  // overlap with |eta| < 1.3, endcap after that

  // Handles
  edm::ESHandle<MagneticField> magneticField_;
  edm::ESHandle<CSCGeometry> cscGeometry_;
  edm::ESHandle<DTGeometry> dtGeometry_;

  std::unique_ptr<MuonServiceProxy> service_;
  std::unique_ptr<MeasurementEstimator> estimator_;

  const std::string propagatorName_;

  // In DT station 4 the top and bottom sectors are made of two chambers
  // due to material requirements. Online is not split:
  // Online sector 4 == offline sector 4 or 10, Online sector 10 == offline sector 10 or 14
  const std::vector<DTChamberId> matchingIds(const DTChamberId& stubId) const;

  // Match online-level CSCDetIds with offline
  const std::vector<CSCDetId> matchingIds(const CSCDetId& stubId) const;

  // Logic to match L1 stubs to DT segments
  const std::pair<int, int> matchingStubSegment(const DTChamberId& stubId,
                                                const l1t::MuonStubRef stub,
                                                const DTRecSegment4DCollection& segments,
                                                const l1t::TrackerMuonRef l1TkMuRef) const;

  // Logic to match L1 stubs to CSC segments
  const std::pair<int, int> matchingStubSegment(const CSCDetId& stubId,
                                                const l1t::MuonStubRef stub,
                                                const CSCSegmentCollection& segments,
                                                const l1t::TrackerMuonRef l1TkMuRef) const;

  // Logic to extrapolate from nearby stations in the barrel
  const std::pair<int, int> extrapolateToNearbyStation(const int endingStation,
                                                       const std::map<DTChamberId, std::pair<int, int>>& matchesInBarrel,
                                                       const DTRecSegment4DCollection& segments) const;

  const std::pair<int, int> extrapolateMatch(const int bestStartingSegIndex,
                                             const int endingStation,
                                             const DTRecSegment4DCollection& segments) const;
};
#endif
