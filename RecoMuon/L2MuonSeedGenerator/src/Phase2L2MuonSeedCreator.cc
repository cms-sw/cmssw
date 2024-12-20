/**  \class phase2L2MuonSeedCreator
 *   See header file for a description of this class
 *   \author Luca Ferragina (INFN BO), 2024
 */

#include "RecoMuon/L2MuonSeedGenerator/src/Phase2L2MuonSeedCreator.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeedCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vdt/atan.h>
#include <vdt/exp.h>
#include <vdt/sincos.h>

#include <vector>

// Constructor
Phase2L2MuonSeedCreator::Phase2L2MuonSeedCreator(const edm::ParameterSet& pset)
    : l1TkMuCollToken_{consumes(pset.getParameter<edm::InputTag>("inputObjects"))},
      cscSegmentCollToken_{consumes(pset.getParameter<edm::InputTag>("cscRecSegmentLabel"))},
      dtSegmentCollToken_{consumes(pset.getParameter<edm::InputTag>("dtRecSegmentLabel"))},
      cscGeometryToken_{esConsumes<CSCGeometry, MuonGeometryRecord>()},
      dtGeometryToken_{esConsumes<DTGeometry, MuonGeometryRecord>()},
      magneticFieldToken_{esConsumes<MagneticField, IdealMagneticFieldRecord>()},
      minMomentum_{pset.getParameter<double>("minPL1Tk")},
      maxMomentum_{pset.getParameter<double>("maxPL1Tk")},
      matchingPhiWindow_{pset.getParameter<double>("stubMatchDPhi")},
      matchingThetaWindow_{pset.getParameter<double>("stubMatchDTheta")},
      extrapolationDeltaPhiClose_{pset.getParameter<double>("extrapolationWindowClose")},
      extrapolationDeltaPhiFar_{pset.getParameter<double>("extrapolationWindowFar")},
      maxEtaBarrel_{pset.getParameter<double>("maximumEtaBarrel")},
      maxEtaOverlap_{pset.getParameter<double>("maximumEtaOverlap")},
      propagatorName_{pset.getParameter<string>("propagator")} {
  // Service parameters
  edm::ParameterSet serviceParameters = pset.getParameter<edm::ParameterSet>("serviceParameters");
  // Services
  service_ = std::make_unique<MuonServiceProxy>(serviceParameters, consumesCollector());
  estimator_ = std::make_unique<Chi2MeasurementEstimator>(pset.getParameter<double>("estimatorMaxChi2"));
  produces<L2MuonTrajectorySeedCollection>();
}

void Phase2L2MuonSeedCreator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputObjects", edm::InputTag("l1tTkMuonsGmt"));
  desc.add<edm::InputTag>("cscRecSegmentLabel", edm::InputTag("hltCscSegments"));
  desc.add<edm::InputTag>("dtRecSegmentLabel", edm::InputTag("hltDt4DSegments"));
  desc.add<double>("minPL1Tk", 3.5);
  desc.add<double>("maxPL1Tk", 200);
  desc.add<double>("stubMatchDPhi", 0.05);
  desc.add<double>("stubMatchDTheta", 0.1);
  desc.add<double>("extrapolationWindowClose", 0.1);
  desc.add<double>("extrapolationWindowFar", 0.05);
  desc.add<double>("maximumEtaBarrel", 0.7);
  desc.add<double>("maximumEtaOverlap", 1.3);
  desc.add<string>("propagator", "SteppingHelixPropagatorAny");

  // Service parameters
  edm::ParameterSetDescription psd0;
  psd0.addUntracked<std::vector<std::string>>("Propagators", {"SteppingHelixPropagatorAny"});
  psd0.add<bool>("RPCLayers", true);
  psd0.addUntracked<bool>("UseMuonNavigation", true);
  desc.add<edm::ParameterSetDescription>("serviceParameters", psd0);
  desc.add<double>("estimatorMaxChi2", 1000.0);
  descriptions.add("Phase2L2MuonSeedCreator", desc);
}

void Phase2L2MuonSeedCreator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const std::string metname = "RecoMuon|Phase2L2MuonSeedCreator";

  auto output = std::make_unique<L2MuonTrajectorySeedCollection>();

  auto const l1TkMuColl = iEvent.getHandle(l1TkMuCollToken_);

  auto cscHandle = iEvent.getHandle(cscSegmentCollToken_);
  auto cscSegments = *cscHandle;
  auto dtHandle = iEvent.getHandle(dtSegmentCollToken_);
  auto dtSegments = *dtHandle;

  cscGeometry_ = iSetup.getHandle(cscGeometryToken_);
  dtGeometry_ = iSetup.getHandle(dtGeometryToken_);

  auto magneticFieldHandle = iSetup.getHandle(magneticFieldToken_);

  LogDebug(metname) << "Number of L1 Tracker Muons in the Event: " << l1TkMuColl->size();

  // Loop on all L1TkMu in event
  for (size_t l1TkMuIndex = 0; l1TkMuIndex != l1TkMuColl->size(); ++l1TkMuIndex) {
    l1t::TrackerMuonRef l1TkMuRef(l1TkMuColl, l1TkMuIndex);

    // Physical info of L1TkMu
    const float pt = l1TkMuRef->phPt();
    if (pt < minMomentum_) {
      continue;
    }
    const float eta = l1TkMuRef->phEta();
    const float phi = l1TkMuRef->phPhi();
    const int charge = l1TkMuRef->phCharge();

    // Calculate theta once to use it for stub-segment matching
    // theta == 2 * atan(e^(-eta))
    const float theta = 2 * vdt::fast_atanf(vdt::fast_expf(-eta));

    // Precompute trig functions for theta
    float sinTheta, cosTheta;
    vdt::fast_sincosf(theta, sinTheta, cosTheta);

    // Precompute trig functions for phi
    float sinPhi, cosPhi;
    vdt::fast_sincosf(phi, sinPhi, cosPhi);

    LogDebug(metname) << "L1TKMu pT: " << pt << ", eta: " << eta << ", phi: " << phi;
    Type muonType = overlap;
    if (std::abs(eta) < maxEtaBarrel_) {
      muonType = barrel;
      LogDebug(metname) << "L1TkMu found in the barrel";
    } else if (std::abs(eta) > maxEtaOverlap_) {
      muonType = endcap;
      LogDebug(metname) << "L1TkMu found in the endcap";
    }

    // Starting seed creation
    LogDebug(metname) << "Start seed creation";

    l1t::MuonStubRefVector stubRefs = l1TkMuRef->stubs();

    LogDebug(metname) << "Number of stubs per L1TkMu: " << stubRefs.size();
    LogDebug(metname) << "Number of DT segments in event: " << dtSegments.size();
    LogDebug(metname) << "Number of CSC segments in event: " << cscSegments.size();

    // Pairs segIndex, segQuality for matches in Barrel/Overlap/Endcap
    std::map<DTChamberId, std::pair<int, int>> matchesInBarrel;
    std::map<CSCDetId, std::pair<int, int>> matchesInEndcap;

    // Matching info
    bool atLeastOneMatch = false;
    bool bestInDt = false;

    // Variables for matches in Overlap
    int totalBarrelQuality = 0;
    int totalEndcapQuality = 0;
    unsigned int nDtHits = 0;
    unsigned int nCscHits = 0;

    // Loop on L1TkMu stubs to find best association to DT/CSC segments
    for (auto stub : stubRefs) {
#ifdef EDM_ML_DEBUG
      stub->print();
#endif
      // Separate barrel, endcap and overlap cases
      switch (muonType) {
        case barrel: {
          if (!stub->isBarrel()) {
            continue;  // skip all non-barrel stubs
          }
          // Create detId for stub
          DTChamberId stubId = DTChamberId(stub->etaRegion(),       // wheel
                                           stub->depthRegion(),     // station
                                           stub->phiRegion() + 1);  // sector, online to offline
          LogDebug(metname) << "Stub DT detId: " << stubId << ". RawId: " << stubId.rawId();

          auto& tmpMatch = matchingStubSegment(stubId, stub, dtSegments, theta);

          // Found a match -> update matching info
          if (tmpMatch.first != -1) {
            matchesInBarrel.emplace(stubId, tmpMatch);
            atLeastOneMatch = true;
            bestInDt = true;
          }

#ifdef EDM_ML_DEBUG
          LogDebug(metname) << "BARREL best segments:";
          for (const auto& [detId, matchingPair] : matchesInBarrel) {
            LogDebug(metname) << "Station " << detId.station() << " (" << matchingPair.first << ", "
                              << matchingPair.second << ")";
          }
#endif
          break;
        }  // End barrel

        case endcap: {
          if (!stub->isEndcap()) {
            continue;  // skip all non-endcap stubs
          }
          // Create detId for stub
          int endcap = (eta > 0) ? 1 : 2;  // CSC DetId endcap (1 -> Forward, 2 -> Backwards)
          CSCDetId stubId = CSCDetId(endcap,
                                     stub->depthRegion(),              // station
                                     6 - std::abs(stub->etaRegion()),  // ring, online to offline
                                     stub->phiRegion());               // chamber
          LogDebug(metname) << "Stub CSC detId: " << stubId << ". RawId: " << stubId.rawId();

          auto& tmpMatch = matchingStubSegment(stubId, stub, cscSegments, theta);

          // Found a match -> update matching info
          if (tmpMatch.first != -1) {
            matchesInEndcap.emplace(stubId, tmpMatch);
            atLeastOneMatch = true;
          }

#ifdef EDM_ML_DEBUG
          LogDebug(metname) << "ENDCAP best segments:";
          for (const auto& [detId, matchingPair] : matchesInEndcap) {
            LogDebug(metname) << "Station " << detId.station() << " (" << matchingPair.first << ", "
                              << matchingPair.second << ")";
          }
#endif
          break;
        }  // End endcap

        case overlap: {
          // Overlap runs on both DTs and CSCs and picks the best overall match
          if (stub->isBarrel()) {
            // Check DTs
            LogDebug(metname) << "OVERLAP stub in DTs, checking " << dtSegments.size() << " DT segments";
            // Create detId for stub
            DTChamberId stubId = DTChamberId(stub->etaRegion(),       // wheel
                                             stub->depthRegion(),     // station
                                             stub->phiRegion() + 1);  // sector, online to offline
            LogDebug(metname) << "Stub DT detId: " << stubId << ". RawId: " << stubId.rawId();

            auto& tmpMatch = matchingStubSegment(stubId, stub, dtSegments, theta);
            totalBarrelQuality += tmpMatch.second;

            // Found a match -> update matching info
            if (tmpMatch.first != -1) {
              matchesInBarrel.emplace(stubId, tmpMatch);
              atLeastOneMatch = true;
              auto dtSegment = dtSegments.begin() + tmpMatch.first;
              nDtHits += (dtSegment->hasPhi() ? dtSegment->phiSegment()->recHits().size() : 0);
              nDtHits += (dtSegment->hasZed() ? dtSegment->zSegment()->recHits().size() : 0);
            }

#ifdef EDM_ML_DEBUG
            LogDebug(metname) << "OVERLAP best segments in DTs:";
            for (auto& [detId, matchingPair] : matchesInBarrel) {
              LogDebug(metname) << "Station " << detId.station() << " (" << matchingPair.first << ", "
                                << matchingPair.second << ")";
            }
#endif
          } else if (stub->isEndcap()) {
            // Check CSCs
            LogDebug(metname) << "OVERLAP stub in CSCs, checking " << cscSegments.size() << " CSC segments";
            int endcap = (eta > 0) ? 1 : 2;  // CSC DetId endcap (1 -> Forward, 2 -> Backwards)
            CSCDetId stubId = CSCDetId(endcap,
                                       stub->depthRegion(),              // station
                                       6 - std::abs(stub->etaRegion()),  // ring, online to offline
                                       stub->phiRegion());               // chamber
            LogDebug(metname) << "Stub CSC detId: " << stubId << ". RawId: " << stubId.rawId();

            auto& tmpMatch = matchingStubSegment(stubId, stub, cscSegments, theta);
            totalEndcapQuality += tmpMatch.second;

            // Found a match -> update matching info
            if (tmpMatch.first != -1) {
              matchesInEndcap.emplace(stubId, tmpMatch);
              atLeastOneMatch = true;
              auto cscSegment = cscSegments.begin() + tmpMatch.first;
              nCscHits += cscSegment->nRecHits();
            }

#ifdef EDM_ML_DEBUG
            LogDebug(metname) << "OVERLAP best segments in CSCs:";
            for (auto& [detId, matchingPair] : matchesInEndcap) {
              LogDebug(metname) << "Station " << detId.station() << " (" << matchingPair.first << ", "
                                << matchingPair.second << ")";
            }
#endif
          }

          LogDebug(metname) << "OVERLAP comparing total qualities. DT: " << totalBarrelQuality
                            << ", CSC: " << totalEndcapQuality;

          // Pick segments in DTs / CSCs based on quality
          bestInDt = (totalBarrelQuality > totalEndcapQuality) ? true : false;

          // Same qualities, pick higher number of hits
          if (totalBarrelQuality == totalEndcapQuality and totalBarrelQuality > -1) {
            LogDebug(metname) << "Same quality " << totalBarrelQuality << ". Checking total number of hits";
            LogDebug(metname) << "DT hits: " << nDtHits << ", CSC hits: " << nCscHits;
            LogDebug(metname) << (nDtHits > nCscHits ? "More hits in DT segment" : "More hits in CSC segment");
            bestInDt = (nDtHits >= nCscHits) ? true : false;
          }
#ifdef EDM_ML_DEBUG
          LogDebug(metname) << "OVERLAP best segments:";
          if (bestInDt) {
            LogDebug(metname) << "OVERLAP best match in DTs:";
            for (auto& [detId, matchingPair] : matchesInBarrel) {
              LogDebug(metname) << "Station " << detId.station() << " (" << matchingPair.first << ", "
                                << matchingPair.second << ")";
            }
          } else if (!bestInDt) {
            LogDebug(metname) << "OVERLAP best match in CSCs:";
            for (auto& [detId, matchingPair] : matchesInEndcap) {
              LogDebug(metname) << "Station " << detId.station() << " (" << matchingPair.first << ", "
                                << matchingPair.second << ")";
            }
          }
#endif
          break;
        }  // End overlap

        default:
          edm::LogError("L1TkMu must be either barrel, endcap or overlap");
          break;
      }
    }  // End loop on stubs

    // Emplace seeds in output
    if (!atLeastOneMatch) {
      LogDebug(metname) << "No matching stub found, skipping seed";
      continue;  // skip unmatched L1TkMu
    } else {
      // Info for propagation to MB2 or ME2
      service_->update(iSetup);
      const DetLayer* detLayer = nullptr;
      float radius = 0.;

      DetId propagateToId;

      edm::OwnVector<TrackingRecHit> container;
      if (bestInDt) {
        // Found at least one matching segment in DT -> propagate to Muon Barrel 2 (MB2) for track finding
        LogDebug(metname) << "Found matching segment(s) in DTs, propagating L1TkMu info to MB2 to seed";
        // MB2 detId
        propagateToId = DTChamberId(0, 2, 0);
        detLayer = service_->detLayerGeometry()->idToLayer(propagateToId);
        const BoundSurface* sur = &(detLayer->surface());
        const BoundCylinder* bc = dynamic_cast<const BoundCylinder*>(sur);
        radius = std::abs(bc->radius() / sinTheta);

        // Propagate matched segments to the seed and try to extrapolate in unmatched chambers
        std::vector<int> matchedStations;
        matchedStations.reserve(4);
        for (auto& [detId, matchingPair] : matchesInBarrel) {
          // Add matched segments to the seed
          LogDebug(metname) << "Adding matched DT segment in station " << detId.station() << " to the seed";
          container.push_back(dtSegments[matchingPair.first]);
          matchedStations.push_back(detId.station());
        }
        // Loop over all barrel muon stations (1-4)
        for (int station = DTChamberId::minStationId; station <= DTChamberId::maxStationId; ++station) {
          if (std::find(matchedStations.begin(), matchedStations.end(), station) == matchedStations.end()) {
            // Try to extrapolate from stations with a match to the ones without
            LogDebug(metname) << "No matching DT segment found in station " << station;
            auto extrapolatedMatch = extrapolateToNearbyStation(station, matchesInBarrel, dtSegments);
            if (extrapolatedMatch.first != -1) {
              LogDebug(metname) << "Adding extrapolated DT segment " << extrapolatedMatch.first << " with quality "
                                << extrapolatedMatch.second << " found in station " << station << " to the seed";
              container.push_back(dtSegments[extrapolatedMatch.first]);
            }
          }
        }
      } else if (!bestInDt) {
        // Found a matching segment in CSC -> propagate to Muon Endcap 2 (ME2) for track finding
        LogDebug(metname) << "Found matching segment(s) in CSCs, propagating L1TkMu info to ME2 to seed";
        // ME2 detId
        propagateToId = eta > 0 ? CSCDetId(1, 2, 0, 0, 0) : CSCDetId(2, 2, 0, 0, 0);
        detLayer = service_->detLayerGeometry()->idToLayer(propagateToId);
        radius = std::abs(detLayer->position().z() / cosTheta);

        // Fill seed with matched segment(s)
        for (auto& [detId, matchingPair] : matchesInEndcap) {
          LogDebug(metname) << "Adding matched CSC segment in station " << detId.station() << " to the seed";
          container.push_back(cscSegments[matchingPair.first]);
        }
      }
      // Get Global point and direction
      GlobalPoint pos(radius * cosPhi * sinTheta, radius * sinPhi * sinTheta, radius * cosTheta);
      GlobalVector mom(pt * cosPhi, pt * sinPhi, pt * cosTheta / sinTheta);

      GlobalTrajectoryParameters param(pos, mom, charge, &*magneticFieldHandle);

      AlgebraicSymMatrix55 mat;

      mat[0][0] = bestInDt ? (0.25 / pt) * (0.25 / pt) : (0.4 / pt) * (0.4 / pt);  // sigma^2(charge/abs_momentum)
      mat[1][1] = 0.05 * 0.05;                                                     // sigma^2(lambda)
      mat[2][2] = 0.2 * 0.2;                                                       // sigma^2(phi)
      mat[3][3] = 20. * 20.;                                                       // sigma^2(x_transverse)
      mat[4][4] = 20. * 20.;                                                       // sigma^2(y_transverse)

      CurvilinearTrajectoryError error(mat);

      const FreeTrajectoryState state(param, error);

      // Create the TrajectoryStateOnSurface
      TrajectoryStateOnSurface tsos = service_->propagator(propagatorName_)->propagate(state, detLayer->surface());
      // Find valid detectors with states
      auto detsWithStates = detLayer->compatibleDets(tsos, *service_->propagator(propagatorName_), *estimator_);
      // Check that at least one valid detector was found
      if (detsWithStates.size() > 0) {
        // Update the detId with the one from the first valid detector with measurments found
        propagateToId = detsWithStates.front().first->geographicalId();
        // Create the Trajectory State on that detector's surface
        tsos = detsWithStates.front().second;
      } else if (detsWithStates.empty() and bestInDt) {
        // Propagation to MB2 failed, fallback to ME2 (might be an overlap edge case)
        LogDebug(metname) << "Warning: detsWithStates collection is empty for a barrel collection. Falling back to ME2";
        // Get ME2 DetLayer
        DetId fallback_id = eta > 0 ? CSCDetId(1, 2, 0, 0, 0) : CSCDetId(2, 2, 0, 0, 0);
        const DetLayer* ME2DetLayer = service_->detLayerGeometry()->idToLayer(fallback_id);
        // Trajectory state on ME2 DetLayer
        tsos = service_->propagator(propagatorName_)->propagate(state, ME2DetLayer->surface());
        // Find the detectors with states on ME2
        detsWithStates = ME2DetLayer->compatibleDets(tsos, *service_->propagator(propagatorName_), *estimator_);
      }
      // Use the valid detector found to produce the persistentState for the seed
      if (!detsWithStates.empty()) {
        LogDebug(metname) << "Found a compatible detWithStates";
        TrajectoryStateOnSurface newTSOS = detsWithStates.front().second;
        const GeomDet* newTSOSDet = detsWithStates.front().first;
        LogDebug(metname) << "Most compatible detector: " << newTSOSDet->geographicalId().rawId();
        if (newTSOS.isValid()) {
          LogDebug(metname) << "pos: (r=" << newTSOS.globalPosition().mag()
                            << ", phi=" << newTSOS.globalPosition().phi() << ", eta=" << newTSOS.globalPosition().eta()
                            << ")";
          LogDebug(metname) << "mom: (q*pt=" << newTSOS.charge() * newTSOS.globalMomentum().perp()
                            << ", phi=" << newTSOS.globalMomentum().phi() << ", eta=" << newTSOS.globalMomentum().eta()
                            << ")";
          // Transform the TrajectoryStateOnSurface in a Persistent TrajectoryStateOnDet
          const PTrajectoryStateOnDet& seedTSOS =
              trajectoryStateTransform::persistentState(newTSOS, newTSOSDet->geographicalId().rawId());

          // Emplace seed in output
          LogDebug(metname) << "Emplacing seed in output";
          output->emplace_back(L2MuonTrajectorySeed(seedTSOS, container, alongMomentum, l1TkMuRef));
        }
      }
    }  // End seed emplacing (one seed per L1TkMu)
  }  // End loop on L1TkMu
  LogDebug(metname) << "All L1TkMu in event processed";
  iEvent.put(std::move(output));
}

// In DT station 4 the top and bottom sectors are made of two chambers
// due to material requirements. Online is not split:
// Online sector 4 == offline sector 4 or 13, Online sector 10 == offline sector 10 or 14
const std::vector<DTChamberId> Phase2L2MuonSeedCreator::matchingIds(const DTChamberId& stubId) const {
  std::vector<DTChamberId> matchingDtIds;
  matchingDtIds.reserve(2);
  matchingDtIds.push_back(stubId);
  if (stubId.station() == 4) {
    if (stubId.sector() == 4) {
      matchingDtIds.emplace_back(DTChamberId(stubId.wheel(), stubId.station(), 13));
    }
    if (stubId.sector() == 10) {
      matchingDtIds.emplace_back(DTChamberId(stubId.wheel(), stubId.station(), 14));
    }
  }
  return matchingDtIds;
}

// Pair bestSegIndex, quality for DT segments matching
const std::pair<int, int> Phase2L2MuonSeedCreator::matchingStubSegment(const DTChamberId& stubId,
                                                                       const l1t::MuonStubRef stub,
                                                                       const DTRecSegment4DCollection& segments,
                                                                       const float l1TkMuTheta) const {
  const std::string metname = "RecoMuon|Phase2L2MuonSeedCreator";

  int bestSegIndex = -1;
  int quality = -1;
  unsigned int nHitsPhiBest = 0;
  unsigned int nHitsThetaBest = 0;

  LogDebug(metname) << "Matching stub with DT segment";
  int nMatchingIds = 0;

  for (DTChamberId id : matchingIds(stubId)) {
    DTRecSegment4DCollection::range segmentsInChamber = segments.get(id);
    for (DTRecSegment4DCollection::const_iterator segment = segmentsInChamber.first;
         segment != segmentsInChamber.second;
         ++segment) {
      ++nMatchingIds;
      DTChamberId segId = segment->chamberId();
      LogDebug(metname) << "Segment DT detId: " << segId << ". RawId: " << segId.rawId();

      // Global position of the segment
      GlobalPoint segPos = dtGeometry_->idToDet(segId)->toGlobal(segment->localPosition());

      // Check delta phi
      double deltaPhi = std::abs(segPos.phi() - stub->offline_coord1());
      LogDebug(metname) << "deltaPhi: " << deltaPhi;

      double deltaTheta = std::abs(segPos.theta() - l1TkMuTheta);
      LogDebug(metname) << "deltaTheta: " << deltaTheta;

      // Skip segments outside phi window or very far in the theta view
      if (deltaPhi > matchingPhiWindow_ or deltaTheta > 4 * matchingThetaWindow_) {
        continue;
      }

      // Inside phi window -> check hit multiplicity
      unsigned int nHitsPhi = (segment->hasPhi() ? segment->phiSegment()->recHits().size() : 0);
      unsigned int nHitsTheta = (segment->hasZed() ? segment->zSegment()->recHits().size() : 0);
      LogDebug(metname) << "DT found match in deltaPhi: " << std::distance(segments.begin(), segment) << " with "
                        << nHitsPhi << " hits in phi and " << nHitsTheta << " hits in theta";

      if (nHitsPhi == nHitsPhiBest and segment->hasZed()) {
        // Same phi hit multiplicity -> check delta theta
        LogDebug(metname) << "DT found segment with same hits in phi as previous best (" << nHitsPhiBest
                          << "), checking theta window";

        // More precise check in theta window
        if (deltaTheta > matchingThetaWindow_) {
          continue;  // skip segments outside theta window
        }

        LogDebug(metname) << "DT found match in deltaTheta: " << std::distance(segments.begin(), segment) << " with "
                          << nHitsPhi << " hits in phi and " << nHitsTheta << " hits in theta";

        // Inside theta window -> check hit multiplicity (theta)
        if (nHitsTheta > nHitsThetaBest) {
          // More hits in theta -> update bestSegment and quality
          LogDebug(metname) << "DT found segment with more hits in theta than previous best";
          bestSegIndex = std::distance(segments.begin(), segment);
          quality = 2;
          LogDebug(metname) << "DT updating bestSegIndex (nHitsTheta): " << bestSegIndex << " with "
                            << nHitsPhi + nHitsTheta << ">" << nHitsPhiBest + nHitsThetaBest
                            << " total hits and quality " << quality;
          nHitsThetaBest = nHitsTheta;
        }
      } else if (nHitsPhi > nHitsPhiBest) {
        // More hits in phi -> update bestSegment and quality
        LogDebug(metname) << "DT found segment with more hits in phi than previous best";
        bestSegIndex = std::distance(segments.begin(), segment);
        quality = 1;
        LogDebug(metname) << "DT updating bestSegIndex (nHitsPhi): " << bestSegIndex << " with " << nHitsPhi << ">"
                          << nHitsPhiBest << " hits in phi, " << nHitsTheta << " hits in theta and quality " << quality;
        nHitsPhiBest = nHitsPhi;
        nHitsThetaBest = nHitsTheta;
      }
    }  // End loop on segments
  }

  LogDebug(metname) << "DT looped over " << nMatchingIds << (nMatchingIds > 1 ? " segments" : " segment")
                    << " with same DT detId as stub";

  if (quality < 0) {
    LogDebug(metname) << "DT proposed match: " << bestSegIndex << " with quality " << quality << ". Not good enough!";
    return std::make_pair(-1, -1);
  } else {
    LogDebug(metname) << "Found DT segment match";
    LogDebug(metname) << "New DT segment: " << bestSegIndex << " with " << nHitsPhiBest + nHitsThetaBest
                      << " total hits and quality " << quality;
    return std::make_pair(bestSegIndex, quality);
  }
}

// Match online-level CSCDetIds to offline labels
const std::vector<CSCDetId> Phase2L2MuonSeedCreator::matchingIds(const CSCDetId& stubId) const {
  std::vector<CSCDetId> matchingCscIds;
  matchingCscIds.push_back(stubId);

  if (stubId.station() == 1 and stubId.ring() == 1) {
    matchingCscIds.emplace_back(CSCDetId(stubId.endcap(), stubId.station(), 4, stubId.chamber()));
  }

  return matchingCscIds;
}

// Pair bestSegIndex, quality for CSC segments matching
const std::pair<int, int> Phase2L2MuonSeedCreator::matchingStubSegment(const CSCDetId& stubId,
                                                                       const l1t::MuonStubRef stub,
                                                                       const CSCSegmentCollection& segments,
                                                                       const float l1TkMuTheta) const {
  const std::string metname = "RecoMuon|Phase2L2MuonSeedCreator";

  int bestSegIndex = -1;
  int quality = -1;
  unsigned int nHitsBest = 0;

  LogDebug(metname) << "Matching stub with CSC segment";
  int nMatchingIds = 0;
  for (CSCDetId id : matchingIds(stubId)) {
    CSCSegmentCollection::range segmentsInChamber = segments.get(id);
    for (CSCSegmentCollection::const_iterator segment = segmentsInChamber.first; segment != segmentsInChamber.second;
         ++segment) {
      ++nMatchingIds;
      CSCDetId segId = segment->cscDetId();
      LogDebug(metname) << "Segment CSC detId: " << segId << ". RawId: " << segId.rawId();

      // Global position of the segment
      GlobalPoint segPos = cscGeometry_->idToDet(segId)->toGlobal(segment->localPosition());

      // Check delta phi
      double deltaPhi = std::abs(segPos.phi() - stub->offline_coord1());
      LogDebug(metname) << "deltaPhi: " << deltaPhi;

      double deltaTheta = std::abs(segPos.theta() - l1TkMuTheta);
      LogDebug(metname) << "deltaTheta: " << deltaTheta;

      // Theta mainly used in cases where multiple matches are found
      // to keep only the best one. Still skip segments way outside
      // a reasonable window
      const double roughThetaWindow = 0.4;
      if (deltaPhi > matchingPhiWindow_ or deltaTheta > roughThetaWindow) {
        continue;  // skip segments outside phi window
      }

      // Inside phi window -> check hit multiplicity
      unsigned int nHits = segment->nRecHits();
      LogDebug(metname) << "CSC found match in deltaPhi: " << std::distance(segments.begin(), segment) << " with "
                        << nHits << " hits";

      if (nHits == nHitsBest) {
        // Same hit multiplicity -> check delta theta
        LogDebug(metname) << "Found CSC segment with same hits (" << nHitsBest
                          << ") as previous best, checking theta window";

        if (deltaTheta > matchingThetaWindow_) {
          continue;  // skip segments outside theta window
        }

        // Inside theta window -> update bestSegment and quality
        bestSegIndex = std::distance(segments.begin(), segment);
        quality = 1;
        LogDebug(metname) << "CSC found match in deltaTheta: " << bestSegIndex << " with " << nHits
                          << " hits and quality " << quality;
      } else if (nHits > nHitsBest) {
        // More hits -> update bestSegment and quality
        bestSegIndex = std::distance(segments.begin(), segment);
        quality = 2;
        LogDebug(metname) << "Found CSC segment with more hits. Index: " << bestSegIndex << " with " << nHits << ">"
                          << nHitsBest << " hits and quality " << quality;
        nHitsBest = nHits;
      }
    }  // End loop on segments
  }

  LogDebug(metname) << "CSC looped over " << nMatchingIds << (nMatchingIds != 1 ? " segments" : " segment")
                    << " with same CSC detId as stub";

  if (quality < 0) {
    LogDebug(metname) << "CSC proposed match: " << bestSegIndex << " with quality " << quality << ". Not good enough!";
    return std::make_pair(-1, -1);
  } else {
    LogDebug(metname) << "Found CSC segment match";
    LogDebug(metname) << "New CSC segment: " << bestSegIndex << " with " << nHitsBest << " hits and quality "
                      << quality;
    return std::make_pair(bestSegIndex, quality);
  }
}

const std::pair<int, int> Phase2L2MuonSeedCreator::extrapolateToNearbyStation(
    const int endingStation,
    const std::map<DTChamberId, std::pair<int, int>>& matchesInBarrel,
    const DTRecSegment4DCollection& segments) const {
  const std::string metname = "RecoMuon|Phase2L2MuonSeedCreator";

  std::pair<int, int> extrapolatedMatch = std::make_pair(-1, -1);
  bool foundExtrapolatedMatch = false;
  switch (endingStation) {
    case 1: {
      // Station 1. Extrapolate 2->1 or 3->1 (4->1)
      int startingStation = 2;
      while (startingStation < 5) {
        for (auto& [detId, matchingPair] : matchesInBarrel) {
          if (detId.station() == startingStation) {
            LogDebug(metname) << "Extrapolating from station " << startingStation << " to station " << endingStation;
            extrapolatedMatch = extrapolateMatch(matchingPair.first, endingStation, segments);
            if (extrapolatedMatch.first != -1) {
              LogDebug(metname) << "Found extrapolated match in station " << endingStation << " from station "
                                << startingStation;
              foundExtrapolatedMatch = true;
              break;
            }
          }
        }
        if (foundExtrapolatedMatch) {
          break;
        }
        ++startingStation;
      }
      break;
    }
    case 2: {
      // Station 2. Extrapolate 1->2 or 3->2 (4->2)
      int startingStation = 1;
      while (startingStation < 5) {
        for (auto& [detId, matchingPair] : matchesInBarrel) {
          if (detId.station() == startingStation) {
            LogDebug(metname) << "Extrapolating from station " << startingStation << " to station " << endingStation;
            extrapolatedMatch = extrapolateMatch(matchingPair.first, endingStation, segments);
            if (extrapolatedMatch.first != -1) {
              LogDebug(metname) << "Found extrapolated match in station " << endingStation << " from station "
                                << startingStation;
              foundExtrapolatedMatch = true;
              break;
            }
          }
        }
        if (foundExtrapolatedMatch) {
          break;
        }
        startingStation = startingStation == 1 ? startingStation + 2 : startingStation + 1;
      }
      break;
    }
    case 3: {
      // Station 3. Extrapolate 2->3 or 4->3 (1->3)
      int startingStation = 2;
      while (startingStation > 0) {
        for (auto& [detId, matchingPair] : matchesInBarrel) {
          if (detId.station() == startingStation) {
            LogDebug(metname) << "Extrapolating from station " << startingStation << " to station " << endingStation;
            extrapolatedMatch = extrapolateMatch(matchingPair.first, endingStation, segments);
            if (extrapolatedMatch.first != -1) {
              LogDebug(metname) << "Found extrapolated match in station " << endingStation << " from station "
                                << startingStation;
              foundExtrapolatedMatch = true;
              break;
            }
          }
        }
        if (foundExtrapolatedMatch) {
          break;
        }
        startingStation = startingStation == 2 ? startingStation + 2 : startingStation - 3;
      }
      break;
    }
    case 4: {
      // Station 4. Extrapolate 2->4 or 3->4 (1->4)
      int startingStation = 2;
      while (startingStation > 0) {
        for (auto& [detId, matchingPair] : matchesInBarrel) {
          if (detId.station() == startingStation) {
            LogDebug(metname) << "Extrapolating from station " << startingStation << " to station " << endingStation;
            extrapolatedMatch = extrapolateMatch(matchingPair.first, endingStation, segments);
            if (extrapolatedMatch.first != -1) {
              LogDebug(metname) << "Found extrapolated match in station " << endingStation << " from station "
                                << startingStation;
              foundExtrapolatedMatch = true;
              break;
            }
          }
        }
        if (foundExtrapolatedMatch) {
          break;
        }
        startingStation = startingStation == 2 ? startingStation + 1 : startingStation - 2;
      }
      break;
    }
    default:
      std::cerr << "Muon stations only go from 1 to 4" << std::endl;
      break;
  }  // end endingStation switch
  return extrapolatedMatch;
}

const std::pair<int, int> Phase2L2MuonSeedCreator::extrapolateMatch(const int bestStartingSegIndex,
                                                                    const int endingStation,
                                                                    const DTRecSegment4DCollection& segments) const {
  const std::string metname = "RecoMuon|Phase2L2MuonSeedCreator";

  const auto& segmentInStartingStation = segments.begin() + bestStartingSegIndex;
  auto matchId = segmentInStartingStation->chamberId();
  GlobalPoint matchPos = dtGeometry_->idToDet(matchId)->toGlobal(segmentInStartingStation->localPosition());

  int bestSegIndex = -1;
  int quality = -1;
  unsigned int nHitsPhiBest = 0;
  unsigned int nHitsThetaBest = 0;

  // Find possible extrapolation from startingStation to endingStation
  for (DTRecSegment4DCollection::const_iterator segment = segments.begin(), last = segments.end(); segment != last;
       ++segment) {
    auto segId = segment->chamberId();

    if (segId.station() != endingStation) {
      continue;  // skip segments outside of endingStation
    }

    // Global positions of the segment
    GlobalPoint segPos = dtGeometry_->idToDet(segId)->toGlobal(segment->localPosition());

    double deltaPhi = std::abs(segPos.phi() - matchPos.phi());
    LogDebug(metname) << "Extrapolation deltaPhi: " << deltaPhi;

    double deltaTheta = std::abs(segPos.theta() - matchPos.theta());
    LogDebug(metname) << "Extrapolation deltaTheta: " << deltaTheta;

    double matchingDeltaPhi =
        std::abs(matchId.station() - endingStation) == 1 ? extrapolationDeltaPhiClose_ : extrapolationDeltaPhiFar_;

    // Theta mainly used in cases where multiple matches are found
    // to keep only the best one. Still skip segments way outside
    // a reasonable window
    const double roughThetaWindow = 0.4;
    if (deltaPhi > matchingDeltaPhi or deltaTheta > roughThetaWindow) {
      continue;
    }

    // Inside phi window -> check hit multiplicity
    unsigned int nHitsPhi = (segment->hasPhi() ? segment->phiSegment()->recHits().size() : 0);
    unsigned int nHitsTheta = (segment->hasZed() ? segment->zSegment()->recHits().size() : 0);
    LogDebug(metname) << "Extrapolation found match in deltaPhi: " << std::distance(segments.begin(), segment)
                      << " with " << nHitsPhi << " hits in phi and " << nHitsTheta << " hits in theta";

    if (nHitsPhi == nHitsPhiBest and segment->hasZed()) {
      // Same phi hit multiplicity -> check delta theta
      LogDebug(metname) << "Extrapolation found segment with same hits in phi as previous best (" << nHitsPhiBest
                        << "), checking theta window";
      double deltaTheta = std::abs(segPos.theta() - matchPos.theta());
      LogDebug(metname) << "Extrapolation deltaTheta: " << deltaTheta;

      if (deltaTheta > matchingThetaWindow_) {
        continue;  // skip segments outside theta window
      }

      LogDebug(metname) << "Extrapolation found match in deltaTheta: " << std::distance(segments.begin(), segment)
                        << " with " << nHitsPhi << " hits in phi and " << nHitsTheta << " hits in theta";

      // Inside theta window -> check hit multiplicity (theta)
      if (nHitsTheta > nHitsThetaBest) {
        // More hits in theta -> update bestSegment and quality
        LogDebug(metname) << "Extrapolation found segment with more hits in theta than previous best";
        bestSegIndex = std::distance(segments.begin(), segment);
        quality = 2;
        LogDebug(metname) << "Extrapolation updating bestSegIndex (nHitsTheta): " << bestSegIndex << " with "
                          << nHitsPhi + nHitsTheta << ">" << nHitsPhiBest + nHitsThetaBest << " total hits and quality "
                          << quality;
        nHitsThetaBest = nHitsTheta;
      }
    } else if (nHitsPhi > nHitsPhiBest) {
      // More hits in phi -> update bestSegment and quality
      LogDebug(metname) << "Extrapolation found segment with more hits in phi than previous best";
      bestSegIndex = std::distance(segments.begin(), segment);
      quality = 1;
      LogDebug(metname) << "Extrapolation updating bestSegIndex (nHitsPhi): " << bestSegIndex << " with " << nHitsPhi
                        << ">" << nHitsPhiBest << " hits in phi, " << nHitsTheta << " hits in theta and quality "
                        << quality;
      nHitsPhiBest = nHitsPhi;
      nHitsThetaBest = nHitsTheta;
    }
  }  // end loop on segments
  return std::make_pair(bestSegIndex, quality);
}

DEFINE_FWK_MODULE(Phase2L2MuonSeedCreator);
