/*
 * $Id: $ 
 */

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsFromTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonDT13ChamberResidual.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonDT2ChamberResidual.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonCSCChamberResidual.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonTrackDT13ChamberResidual.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonTrackDT2ChamberResidual.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonTrackCSCChamberResidual.h"

#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"
#include "TrackingTools/TrackAssociator/interface/DetIdAssociator.h"

#include "TDecompChol.h"
#include <cmath>

edm::ESInputTag MuonResidualsFromTrack::builderESInputTag() { return edm::ESInputTag("", "WithTrackAngle"); }

MuonResidualsFromTrack::MuonResidualsFromTrack(edm::ESHandle<TransientTrackingRecHitBuilder> trackerRecHitBuilder,
                                               edm::ESHandle<MagneticField> magneticField,
                                               edm::ESHandle<GlobalTrackingGeometry> globalGeometry,
                                               edm::ESHandle<DetIdAssociator> muonDetIdAssociator_,
                                               edm::ESHandle<Propagator> prop,
                                               const Trajectory* traj,
                                               const reco::Track* recoTrack,
                                               AlignableNavigator* navigator,
                                               double maxResidual)
    : m_recoTrack(recoTrack) {
  bool m_debug = false;

  if (m_debug) {
    std::cout << "BEGIN MuonResidualsFromTrack" << std::endl;
    const std::string metname = " *** MuonResidualsFromTrack *** ";
    LogTrace(metname) << "Tracking Component changed!";
  }

  clear();

  reco::TransientTrack track(*m_recoTrack, &*magneticField, globalGeometry);
  TransientTrackingRecHit::ConstRecHitContainer recHitsForRefit;
  int iT = 0, iM = 0;
  for (auto const& hit : m_recoTrack->recHits()) {
    if (hit->isValid()) {
      DetId hitId = hit->geographicalId();
      if (hitId.det() == DetId::Tracker) {
        iT++;
        if (m_debug)
          std::cout << "Tracker Hit " << iT << " is found. Add to refit. Dimension: " << hit->dimension() << std::endl;

        recHitsForRefit.push_back(trackerRecHitBuilder->build(&*hit));
      } else if (hitId.det() == DetId::Muon) {
        //        if ( hit->geographicalId().subdetId() == 3 && !theRPCInTheFit ) {
        //          LogTrace("Reco|TrackingTools|TrackTransformer") << "RPC Rec Hit discarged";
        //          continue;
        //        }
        iM++;
        if (m_debug)
          std::cout << "Muon Hit " << iM
                    << " is found. We do not add muon hits to refit. Dimension: " << hit->dimension() << std::endl;
        if (hitId.subdetId() == MuonSubdetId::DT) {
          const DTChamberId chamberId(hitId.rawId());
          if (m_debug)
            std::cout << "Muon Hit in DT wheel " << chamberId.wheel() << " station " << chamberId.station()
                      << " sector " << chamberId.sector() << "." << std::endl;
        } else if (hitId.subdetId() == MuonSubdetId::CSC) {
          const CSCDetId cscDetId(hitId.rawId());
          if (m_debug)
            std::cout << "Muon hit in CSC endcap " << cscDetId.endcap() << " station " << cscDetId.station() << " ring "
                      << cscDetId.ring() << " chamber " << cscDetId.chamber() << "." << std::endl;
        } else if (hitId.subdetId() == MuonSubdetId::RPC) {
          if (m_debug)
            std::cout << "Muon Hit in RPC" << std::endl;
        } else {
          if (m_debug)
            std::cout << "Warning! Muon Hit not in DT or CSC or RPC" << std::endl;
        }
        //        recHitsForRefit.push_back(theMuonRecHitBuilder->build(&*hit));
      }
    }
  }

  //  TrackTransformer trackTransformer();
  //  std::vector<Trajectory> vTrackerTrajectory = trackTransformer.transform(track, recHitsForReFit);
  //  std::cout << "Tracker trajectories size " << vTrackerTrajectory.size() << std::endl;

  TrajectoryStateOnSurface lastTrackerTsos;
  double lastTrackerTsosGlobalPositionR = 0.0;

  std::vector<TrajectoryMeasurement> vTrajMeasurement = traj->measurements();
  if (m_debug)
    std::cout << "  Size of vector of TrajectoryMeasurements: " << vTrajMeasurement.size() << std::endl;
  int nTrajMeasurement = 0;
  for (std::vector<TrajectoryMeasurement>::const_iterator iTrajMeasurement = vTrajMeasurement.begin();
       iTrajMeasurement != vTrajMeasurement.end();
       ++iTrajMeasurement) {
    nTrajMeasurement++;
    if (m_debug)
      std::cout << "    TrajectoryMeasurement #" << nTrajMeasurement << std::endl;

    const TrajectoryMeasurement& trajMeasurement = *iTrajMeasurement;

    TrajectoryStateOnSurface tsos =
        m_tsoscomb(trajMeasurement.forwardPredictedState(), trajMeasurement.backwardPredictedState());
    const TrajectoryStateOnSurface& tsosF = trajMeasurement.forwardPredictedState();
    const TrajectoryStateOnSurface& tsosB = trajMeasurement.backwardPredictedState();
    const TrajectoryStateOnSurface& tsosU = trajMeasurement.updatedState();
    if (m_debug)
      std::cout << "      TrajectoryMeasurement TSOS validity: " << tsos.isValid() << std::endl;
    if (tsos.isValid()) {
      double tsosGlobalPositionR = sqrt(tsos.globalPosition().x() * tsos.globalPosition().x() +
                                        tsos.globalPosition().y() * tsos.globalPosition().y());
      if (m_debug) {
        std::cout << "         TrajectoryMeasurement TSOS localPosition"
                  << " x: " << tsos.localPosition().x() << " y: " << tsos.localPosition().y()
                  << " z: " << tsos.localPosition().z() << std::endl;
        std::cout << "         TrajectoryMeasurement TSOS globalPosition"
                  << " x: " << tsos.globalPosition().x() << " y: " << tsos.globalPosition().y()
                  << " R: " << tsosGlobalPositionR << " z: " << tsos.globalPosition().z() << std::endl;
      }
      if (tsosGlobalPositionR > lastTrackerTsosGlobalPositionR) {
        lastTrackerTsos = tsos;
        lastTrackerTsosGlobalPositionR = tsosGlobalPositionR;
      }
    }

    const TransientTrackingRecHit* trajMeasurementHit = &(*trajMeasurement.recHit());
    if (m_debug)
      std::cout << "      TrajectoryMeasurement hit validity: " << trajMeasurementHit->isValid() << std::endl;
    if (trajMeasurementHit->isValid()) {
      DetId trajMeasurementHitId = trajMeasurementHit->geographicalId();
      int trajMeasurementHitDim = trajMeasurementHit->dimension();
      if (trajMeasurementHitId.det() == DetId::Tracker) {
        if (m_debug)
          std::cout << "      TrajectoryMeasurement hit Det: Tracker" << std::endl;
        if (m_debug)
          std::cout << "      TrajectoryMeasurement hit dimension: " << trajMeasurementHitDim << std::endl;
        m_tracker_numHits++;
        double xresid = tsos.localPosition().x() - trajMeasurementHit->localPosition().x();
        double xresiderr2 = tsos.localError().positionError().xx() + trajMeasurementHit->localPositionError().xx();
        m_tracker_chi2 += xresid * xresid / xresiderr2;

        if (trajMeasurementHitId.subdetId() == StripSubdetector::TID ||
            trajMeasurementHitId.subdetId() == StripSubdetector::TEC) {
          m_contains_TIDTEC = true;
        }
        // YP I add false here. No trajectory measurments in Muon system if we corrected TrackTransformer accordingly
      } else if (false && trajMeasurementHitId.det() == DetId::Muon) {
        //AR: I removed the false criteria for cosmic tests
        // } else if (trajMeasurementHitId.det() == DetId::Muon ) {

        if (m_debug)
          std::cout << "      TrajectoryMeasurement hit Det: Muon" << std::endl;

        if (trajMeasurementHitId.subdetId() == MuonSubdetId::DT) {
          const DTChamberId chamberId(trajMeasurementHitId.rawId());
          if (m_debug)
            std::cout << "        TrajectoryMeasurement hit subDet: DT wheel " << chamberId.wheel() << " station "
                      << chamberId.station() << " sector " << chamberId.sector() << std::endl;

          //          double gChX = globalGeometry->idToDet(chamberId)->position().x();
          //          double gChY = globalGeometry->idToDet(chamberId)->position().y();
          //          double gChZ = globalGeometry->idToDet(chamberId)->position().z();
          //          std::cout << "           The chamber position in global frame x: " << gChX << " y: " << gChY << " z: " << gChZ << std::endl;

          const GeomDet* geomDet = muonDetIdAssociator_->getGeomDet(chamberId);
          double chamber_width = geomDet->surface().bounds().width();
          double chamber_length = geomDet->surface().bounds().length();

          double hitX = trajMeasurementHit->localPosition().x();
          double hitY = trajMeasurementHit->localPosition().y();
          // double hitZ = trajMeasurementHit->localPosition().z();

          double tsosX = tsos.localPosition().x();
          double tsosY = tsos.localPosition().y();
          // double tsosZ = tsos.localPosition().z();

          int residualDT13IsAdded = false;
          int residualDT2IsAdded = false;

          // have we seen this chamber before?
          if (m_dt13.find(chamberId) == m_dt13.end() && m_dt2.find(chamberId) == m_dt2.end()) {
            if (m_debug)
              std::cout << "AR: pushing back chamber: " << chamberId << std::endl;
            m_chamberIds.push_back(chamberId);
            //addTrkCovMatrix(chamberId, tsos); // only for the 1st hit
          }
          if (m_debug)
            std::cout << "AR: size of chamberId: " << m_chamberIds.size() << std::endl;

          if (m_debug)
            std::cout << "        TrajectoryMeasurement hit dimension: " << trajMeasurementHitDim << std::endl;
          if (trajMeasurementHitDim > 1) {
            std::vector<const TrackingRecHit*> vDTSeg2D = trajMeasurementHit->recHits();
            if (m_debug)
              std::cout << "          vDTSeg2D size: " << vDTSeg2D.size() << std::endl;
            for (std::vector<const TrackingRecHit*>::const_iterator itDTSeg2D = vDTSeg2D.begin();
                 itDTSeg2D != vDTSeg2D.end();
                 ++itDTSeg2D) {
              std::vector<const TrackingRecHit*> vDTHits1D = (*itDTSeg2D)->recHits();
              if (m_debug)
                std::cout << "            vDTHits1D size: " << vDTHits1D.size() << std::endl;
              for (std::vector<const TrackingRecHit*>::const_iterator itDTHits1D = vDTHits1D.begin();
                   itDTHits1D != vDTHits1D.end();
                   ++itDTHits1D) {
                const TrackingRecHit* hit = *itDTHits1D;
                if (m_debug)
                  std::cout << "              hit dimension: " << hit->dimension() << std::endl;

                DetId hitId = hit->geographicalId();
                const DTSuperLayerId superLayerId(hitId.rawId());
                const DTLayerId layerId(hitId.rawId());
                if (m_debug)
                  std::cout << "              hit superLayerId: " << superLayerId.superLayer() << std::endl;
                if (m_debug)
                  std::cout << "              hit layerId: " << layerId.layer() << std::endl;

                if (superLayerId.superlayer() == 2 && vDTHits1D.size() >= 3) {
                  if (m_dt2.find(chamberId) == m_dt2.end()) {
                    AlignableDetOrUnitPtr chamberAlignable = navigator->alignableFromDetId(chamberId);
                    m_dt2[chamberId] =
                        new MuonDT2ChamberResidual(globalGeometry, navigator, chamberId, chamberAlignable);
                    if (m_debug)
                      std::cout << "              This is first appearance of the DT with hits in superlayer 2"
                                << std::endl;
                  }
                  m_dt2[chamberId]->addResidual(prop, &tsos, hit, chamber_width, chamber_length);
                  residualDT2IsAdded = true;

                } else if ((superLayerId.superlayer() == 1 || superLayerId.superlayer() == 3) &&
                           vDTHits1D.size() >= 6) {
                  if (m_dt13.find(chamberId) == m_dt13.end()) {
                    AlignableDetOrUnitPtr chamberAlignable = navigator->alignableFromDetId(chamberId);
                    m_dt13[chamberId] =
                        new MuonDT13ChamberResidual(globalGeometry, navigator, chamberId, chamberAlignable);
                    if (m_debug)
                      std::cout << "              This is first appearance of the DT with hits in superlayers 1 and 3"
                                << std::endl;
                  }
                  m_dt13[chamberId]->addResidual(prop, &tsos, hit, chamber_width, chamber_length);
                  residualDT13IsAdded = true;
                }
              }
            }
          }

          if (residualDT13IsAdded == true && residualDT2IsAdded == true && chamberId.wheel() == 0 &&
              chamberId.station() == 2 && chamberId.sector() == 7) {
            if (m_debug) {
              std::cout << "MYMARK " << tsosX << " " << hitX << " " << tsosX - hitX << " "
                        << m_dt13[chamberId]->trackx() << " " << m_dt13[chamberId]->residual() << " " << tsosY << " "
                        << hitY << " " << tsosY - hitY << " " << m_dt2[chamberId]->tracky() << " "
                        << m_dt2[chamberId]->residual() << " " << tsosF.localPosition().x() << " "
                        << tsosF.localPosition().y() << " " << tsosF.localPosition().z() << " "
                        << tsosB.localPosition().x() << " " << tsosB.localPosition().y() << " "
                        << tsosB.localPosition().z() << " " << tsosU.localPosition().x() << " "
                        << tsosU.localPosition().y() << " " << tsosU.localPosition().z() << std::endl;
            }
          }

          // http://cmslxr.fnal.gov/lxr/source/DataFormats/TrackReco/src/HitPattern.cc#101
          // YP I add false here. No trajectory measurments in Muon system if we corrected TrackTransformer accordingly
        } else if (false && trajMeasurementHitId.subdetId() == MuonSubdetId::CSC) {
          const CSCDetId cscDetId(trajMeasurementHitId.rawId());
          const CSCDetId chamberId2(cscDetId.endcap(), cscDetId.station(), cscDetId.ring(), cscDetId.chamber());
          if (m_debug)
            std::cout << "        TrajectoryMeasurement hit subDet: CSC endcap " << cscDetId.endcap() << " station "
                      << cscDetId.station() << " ring " << cscDetId.ring() << " chamber " << cscDetId.chamber()
                      << std::endl;
          if (m_debug)
            std::cout << "        TrajectoryMeasurement hit dimension: " << trajMeasurementHitDim << std::endl;

          if (trajMeasurementHitDim == 4) {
            std::vector<const TrackingRecHit*> vCSCHits2D = trajMeasurementHit->recHits();
            if (m_debug)
              std::cout << "          vCSCHits2D size: " << vCSCHits2D.size() << std::endl;
            if (vCSCHits2D.size() >= 5) {
              for (std::vector<const TrackingRecHit*>::const_iterator itCSCHits2D = vCSCHits2D.begin();
                   itCSCHits2D != vCSCHits2D.end();
                   ++itCSCHits2D) {
                const TrackingRecHit* cscHit2D = *itCSCHits2D;
                if (m_debug)
                  std::cout << "            cscHit2D dimension: " << cscHit2D->dimension() << std::endl;
                const TrackingRecHit* hit = cscHit2D;
                if (m_debug)
                  std::cout << "              hit dimension: " << hit->dimension() << std::endl;

                DetId hitId = hit->geographicalId();
                const CSCDetId cscDetId(hitId.rawId());
                if (m_debug)
                  std::cout << "              hit layer: " << cscDetId.layer() << std::endl;

                // not sure why we sometimes get layer == 0
                if (cscDetId.layer() == 0)
                  continue;

                // have we seen this chamber before?
                if (m_csc.find(chamberId2) == m_csc.end()) {
                  m_chamberIds.push_back(chamberId2);
                  //addTrkCovMatrix(chamberId, tsos); // only for the 1st hit
                  AlignableDetOrUnitPtr chamberAlignable = navigator->alignableFromDetId(chamberId2);
                  m_csc[chamberId2] =
                      new MuonCSCChamberResidual(globalGeometry, navigator, chamberId2, chamberAlignable);
                  if (m_debug)
                    std::cout << "              This is first appearance of the CSC with hits QQQ" << std::endl;
                }

                m_csc[chamberId2]->addResidual(prop, &tsos, hit, 250.0, 250.0);
              }
            }
          }
        } else {
          if (m_debug)
            std::cout << "        TrajectoryMeasurement hit subDet: UNKNOWN" << std::endl;
          if (m_debug)
            std::cout << "AR: trajMeasurementHitId.det(): " << trajMeasurementHitId.subdetId() << std::endl;
        }
      } else {
        if (m_debug)
          std::cout << "      TrajectoryMeasurement hit det: UNKNOWN" << std::endl;
        if (m_debug)
          std::cout << "AR: trajMeasurementHitId.det(): " << trajMeasurementHitId.det() << std::endl;
        if (m_debug)
          std::cout << "DetId::Tracker: " << DetId::Tracker << std::endl;
      }
    }
  }

  int iT2 = 0, iM2 = 0;
  for (auto const& hit2 : m_recoTrack->recHits()) {
    if (hit2->isValid()) {
      DetId hitId2 = hit2->geographicalId();
      if (hitId2.det() == DetId::Tracker) {
        iT2++;
        if (m_debug)
          std::cout << "Tracker Hit " << iT2 << " is found. We don't calcualte Tsos for it" << std::endl;
      } else if (hitId2.det() == DetId::Muon) {
        //        if ( (*hit)->geographicalId().subdetId() == 3 && !theRPCInTheFit ) {
        //          LogTrace("Reco|TrackingTools|TrackTransformer") << "RPC Rec Hit discarged";
        //          continue;
        //        }
        iM2++;
        if (m_debug)
          std::cout << "Muon Hit " << iM2 << " is found. Dimension: " << hit2->dimension() << std::endl;
        if (hitId2.subdetId() == MuonSubdetId::DT) {
          const DTChamberId chamberId(hitId2.rawId());
          if (m_debug)
            std::cout << "Muon Hit in DT wheel " << chamberId.wheel() << " station " << chamberId.station()
                      << " sector " << chamberId.sector() << std::endl;

          const GeomDet* geomDet = muonDetIdAssociator_->getGeomDet(chamberId);
          double chamber_width = geomDet->surface().bounds().width();
          double chamber_length = geomDet->surface().bounds().length();

          if (hit2->dimension() > 1) {
            // std::vector<const TrackingRecHit*> vDTSeg2D = hit2->recHits();
            std::vector<TrackingRecHit*> vDTSeg2D = hit2->recHits();

            if (m_debug)
              std::cout << "          vDTSeg2D size: " << vDTSeg2D.size() << std::endl;

            // for ( std::vector<const TrackingRecHit*>::const_iterator itDTSeg2D =  vDTSeg2D.begin();
            //                                                          itDTSeg2D != vDTSeg2D.end();
            //                                                        ++itDTSeg2D ) {

            for (std::vector<TrackingRecHit*>::const_iterator itDTSeg2D = vDTSeg2D.begin(); itDTSeg2D != vDTSeg2D.end();
                 ++itDTSeg2D) {
              // std::vector<const TrackingRecHit*> vDTHits1D =  (*itDTSeg2D)->recHits();
              std::vector<TrackingRecHit*> vDTHits1D = (*itDTSeg2D)->recHits();
              if (m_debug)
                std::cout << "            vDTHits1D size: " << vDTHits1D.size() << std::endl;
              // for ( std::vector<const TrackingRecHit*>::const_iterator itDTHits1D =  vDTHits1D.begin();
              //                                                          itDTHits1D != vDTHits1D.end();
              //                                                        ++itDTHits1D ) {
              for (std::vector<TrackingRecHit*>::const_iterator itDTHits1D = vDTHits1D.begin();
                   itDTHits1D != vDTHits1D.end();
                   ++itDTHits1D) {
                //const TrackingRecHit* hit = *itDTHits1D;
                TrackingRecHit* hit = *itDTHits1D;
                if (m_debug)
                  std::cout << "              hit dimension: " << hit->dimension() << std::endl;

                DetId hitId = hit->geographicalId();
                const DTSuperLayerId superLayerId(hitId.rawId());
                const DTLayerId layerId(hitId.rawId());
                if (m_debug)
                  std::cout << "              hit superLayerId: " << superLayerId.superLayer() << std::endl;
                if (m_debug)
                  std::cout << "              hit layerId: " << layerId.layer() << std::endl;

                if (superLayerId.superlayer() == 2 && vDTHits1D.size() >= 3) {
                  if (m_dt2.find(chamberId) == m_dt2.end()) {
                    AlignableDetOrUnitPtr chamberAlignable = navigator->alignableFromDetId(chamberId);
                    m_dt2[chamberId] =
                        new MuonDT2ChamberResidual(globalGeometry, navigator, chamberId, chamberAlignable);
                    if (m_debug)
                      std::cout << "              This is first appearance of the DT with hits in superlayer 2"
                                << std::endl;

                    // have we seen this chamber before? check if it was in dt13
                    if (m_dt13.find(chamberId) == m_dt13.end()) {
                      m_chamberIds.push_back(chamberId);
                    }
                  }

                  TrajectoryStateOnSurface extrapolation;
                  extrapolation = prop->propagate(lastTrackerTsos, globalGeometry->idToDet(hitId)->surface());

                  if (extrapolation.isValid()) {
                    if (m_debug) {
                      std::cout << " extrapolation localPosition()"
                                << " x: " << extrapolation.localPosition().x()
                                << " y: " << extrapolation.localPosition().y()
                                << " z: " << extrapolation.localPosition().z() << std::endl;
                    }
                    m_dt2[chamberId]->addResidual(prop, &extrapolation, hit, chamber_width, chamber_length);
                  }
                  //            	    residualDT2IsAdded = true;

                } else if ((superLayerId.superlayer() == 1 || superLayerId.superlayer() == 3) &&
                           vDTHits1D.size() >= 6) {
                  if (m_dt13.find(chamberId) == m_dt13.end()) {
                    AlignableDetOrUnitPtr chamberAlignable = navigator->alignableFromDetId(chamberId);
                    m_dt13[chamberId] =
                        new MuonDT13ChamberResidual(globalGeometry, navigator, chamberId, chamberAlignable);
                    if (m_debug)
                      std::cout << "              This is first appearance of the DT with hits in superlayers 1 and 3"
                                << std::endl;

                    // have we seen this chamber before? check if it was in dt2
                    if (m_dt2.find(chamberId) == m_dt2.end()) {
                      m_chamberIds.push_back(chamberId);
                    }
                  }

                  TrajectoryStateOnSurface extrapolation;
                  extrapolation = prop->propagate(lastTrackerTsos, globalGeometry->idToDet(hitId)->surface());

                  if (extrapolation.isValid()) {
                    if (m_debug) {
                      std::cout << " extrapolation localPosition()"
                                << " x: " << extrapolation.localPosition().x()
                                << " y: " << extrapolation.localPosition().y()
                                << " z: " << extrapolation.localPosition().z() << std::endl;
                    }
                    m_dt13[chamberId]->addResidual(prop, &extrapolation, hit, chamber_width, chamber_length);
                  }
                  //            	    residualDT13IsAdded = true;
                }
              }
            }
          }

          //          std::cout << "Extrapolate last Tracker TSOS to muon hit" << std::endl;
          //            TrajectoryStateOnSurface extrapolation;
          //            extrapolation = prop->propagate( lastTrackerTsos, globalGeometry->idToDet(hitId2)->surface() );
          //
          //            if ( chamberId2.wheel() == 0 && chamberId2.station() == 2 && chamberId2.sector() == 7 ) {
          //
          //            double hitX2 = hit2->localPosition().x();
          //          double hitY2 = hit2->localPosition().y();
          //          double hitZ2 = hit2->localPosition().z();
          //
          //          double tsosX2 = extrapolation.localPosition().x();
          //          double tsosY2 = extrapolation.localPosition().y();
          //          double tsosZ2 = extrapolation.localPosition().z();
          //
          //            std::cout << "MYMARK " << tsosX2 << " " << hitX2 << " " << tsosX2 - hitX2 << " " << "0" << " " << "0"
          //                            << " " << tsosY2 << " " << hitY2 << " " << tsosY2 - hitY2 << " " << "0"  << " " << "0"
          //                            << " 0 0 0 0 0 0 0 0 0 " << std::endl;
          ////                            << " " << tsosF.localPosition().x() << " " << tsosF.localPosition().y() << " " << tsosF.localPosition().z()
          ////                            << " " << tsosB.localPosition().x() << " " << tsosB.localPosition().y() << " " << tsosB.localPosition().z()
          ////                            << " " << tsosU.localPosition().x() << " " << tsosU.localPosition().y() << " " << tsosU.localPosition().z() << std::endl;
          //            }

        } else if (hitId2.subdetId() == MuonSubdetId::CSC) {
          const CSCDetId cscDetId2(hitId2.rawId());
          const CSCDetId chamberId(cscDetId2.endcap(), cscDetId2.station(), cscDetId2.ring(), cscDetId2.chamber());
          if (m_debug)
            std::cout << "Muon hit in CSC endcap " << cscDetId2.endcap() << " station " << cscDetId2.station()
                      << " ring " << cscDetId2.ring() << " chamber " << cscDetId2.chamber() << "." << std::endl;

          if (hit2->dimension() == 4) {
            // std::vector<const TrackingRecHit*> vCSCHits2D = hit2->recHits();
            std::vector<TrackingRecHit*> vCSCHits2D = hit2->recHits();
            if (m_debug)
              std::cout << "          vCSCHits2D size: " << vCSCHits2D.size() << std::endl;
            if (vCSCHits2D.size() >= 5) {
              // for ( std::vector<const TrackingRecHit*>::const_iterator itCSCHits2D =  vCSCHits2D.begin();
              //                                                          itCSCHits2D != vCSCHits2D.end();
              //                                                        ++itCSCHits2D ) {

              for (std::vector<TrackingRecHit*>::const_iterator itCSCHits2D = vCSCHits2D.begin();
                   itCSCHits2D != vCSCHits2D.end();
                   ++itCSCHits2D) {
                // const TrackingRecHit* cscHit2D = *itCSCHits2D;
                TrackingRecHit* cscHit2D = *itCSCHits2D;
                if (m_debug)
                  std::cout << "            cscHit2D dimension: " << cscHit2D->dimension() << std::endl;
                // const TrackingRecHit* hit = cscHit2D;
                TrackingRecHit* hit = cscHit2D;
                if (m_debug)
                  std::cout << "              hit dimension: " << hit->dimension() << std::endl;

                DetId hitId = hit->geographicalId();
                const CSCDetId cscDetId(hitId.rawId());

                if (m_debug) {
                  std::cout << "              hit layer: " << cscDetId.layer() << std::endl;

                  std::cout << " hit localPosition"
                            << " x: " << hit->localPosition().x() << " y: " << hit->localPosition().y()
                            << " z: " << hit->localPosition().z() << std::endl;
                  std::cout << " hit globalPosition"
                            << " x: " << globalGeometry->idToDet(hitId)->toGlobal(hit->localPosition()).x()
                            << " y: " << globalGeometry->idToDet(hitId)->toGlobal(hit->localPosition()).y()
                            << " z: " << globalGeometry->idToDet(hitId)->toGlobal(hit->localPosition()).z()
                            << std::endl;
                }

                // not sure why we sometimes get layer == 0
                if (cscDetId.layer() == 0)
                  continue;

                // have we seen this chamber before?
                if (m_debug)
                  std::cout << "Have we seen this chamber before?";
                if (m_csc.find(chamberId) == m_csc.end()) {
                  if (m_debug)
                    std::cout << " NO. m_csc.count() = " << m_csc.count(chamberId) << std::endl;
                  AlignableDetOrUnitPtr chamberAlignable = navigator->alignableFromDetId(chamberId);
                  m_csc[chamberId] = new MuonCSCChamberResidual(globalGeometry, navigator, chamberId, chamberAlignable);
                  if (m_debug)
                    std::cout << "              This is first appearance of the CSC with hits m_csc.count() = "
                              << m_csc.count(chamberId) << std::endl;
                  m_chamberIds.push_back(chamberId);
                  //addTrkCovMatrix(chamberId, tsos); // only for the 1st hit
                } else {
                  if (m_debug)
                    std::cout << " YES. m_csc.count() = " << m_csc.count(chamberId) << std::endl;
                }

                if (m_debug) {
                  std::cout << " lastTrackerTsos localPosition"
                            << " x: " << lastTrackerTsos.localPosition().x()
                            << " y: " << lastTrackerTsos.localPosition().y()
                            << " z: " << lastTrackerTsos.localPosition().z() << std::endl;
                  std::cout << " lastTrackerTsos globalPosition"
                            << " x: " << lastTrackerTsos.globalPosition().x()
                            << " y: " << lastTrackerTsos.globalPosition().y()
                            << " z: " << lastTrackerTsos.globalPosition().z() << std::endl;
                  std::cout << " Do extrapolation from lastTrackerTsos to hit surface" << std::endl;
                }
                TrajectoryStateOnSurface extrapolation;
                extrapolation = prop->propagate(lastTrackerTsos, globalGeometry->idToDet(hitId)->surface());
                if (m_debug)
                  std::cout << " extrapolation.isValid() = " << extrapolation.isValid() << std::endl;

                if (extrapolation.isValid()) {
                  if (m_debug) {
                    std::cout << " extrapolation localPosition()"
                              << " x: " << extrapolation.localPosition().x()
                              << " y: " << extrapolation.localPosition().y()
                              << " z: " << extrapolation.localPosition().z() << std::endl;
                  }
                  m_csc[chamberId]->addResidual(prop, &extrapolation, hit, 250.0, 250.0);
                }
              }
            }
          }

        } else if (hitId2.subdetId() == MuonSubdetId::RPC) {
          if (m_debug)
            std::cout << "Muon Hit in RPC" << std::endl;
        } else {
          if (m_debug)
            std::cout << "Warning! Muon Hit not in DT or CSC or RPC" << std::endl;
        }
        //        recHitsForRefit.push_back(theMuonRecHitBuilder->build(&**hit));
        if (hitId2.subdetId() == MuonSubdetId::DT || hitId2.subdetId() == MuonSubdetId::CSC) {
        }
      }
    }
  }

  if (m_debug)
    std::cout << "END MuonResidualsFromTrack" << std::endl << std::endl;
}

MuonResidualsFromTrack::MuonResidualsFromTrack(edm::ESHandle<GlobalTrackingGeometry> globalGeometry,
                                               const reco::Muon* recoMuon,
                                               AlignableNavigator* navigator,
                                               double maxResidual)
    : m_recoMuon(recoMuon) {
  bool m_debug = false;

  clear();
  assert(m_recoMuon->isTrackerMuon() && m_recoMuon->innerTrack().isNonnull());
  m_recoTrack = m_recoMuon->innerTrack().get();

  m_tracker_chi2 = m_recoMuon->innerTrack()->chi2();
  m_tracker_numHits = m_recoMuon->innerTrack()->ndof() + 5;
  m_tracker_numHits = m_tracker_numHits > 0 ? m_tracker_numHits : 0;

  /*
                   for(auto const& hit : m_recoMuon->innerTrack()->recHits())
                   {
                   DetId id = hit->geographicalId();
                   if (id.det() == DetId::Tracker)
                   {
                   m_tracker_numHits++;
                   if (id.subdetId() == StripSubdetector::TID  ||  id.subdetId() == StripSubdetector::TEC) m_contains_TIDTEC = true;
                   }
                   }
                   */

  for (std::vector<reco::MuonChamberMatch>::const_iterator chamberMatch = m_recoMuon->matches().begin();
       chamberMatch != m_recoMuon->matches().end();
       chamberMatch++) {
    if (chamberMatch->id.det() != DetId::Muon)
      continue;

    for (std::vector<reco::MuonSegmentMatch>::const_iterator segMatch = chamberMatch->segmentMatches.begin();
         segMatch != chamberMatch->segmentMatches.end();
         ++segMatch) {
      // select the only segment that belongs to track and is the best in station by dR
      if (!(segMatch->isMask(reco::MuonSegmentMatch::BestInStationByDR) &&
            segMatch->isMask(reco::MuonSegmentMatch::BelongsToTrackByDR)))
        continue;

      if (chamberMatch->id.subdetId() == MuonSubdetId::DT) {
        const DTChamberId chamberId(chamberMatch->id.rawId());

        DTRecSegment4DRef segmentDT = segMatch->dtSegmentRef;
        const DTRecSegment4D* segment = segmentDT.get();
        if (segment == nullptr)
          continue;

        if (segment->hasPhi() && fabs(chamberMatch->x - segMatch->x) > maxResidual)
          continue;
        if (segment->hasZed() && fabs(chamberMatch->y - segMatch->y) > maxResidual)
          continue;

        // have we seen this chamber before?
        if (m_dt13.find(chamberId) == m_dt13.end() && m_dt2.find(chamberId) == m_dt2.end()) {
          m_chamberIds.push_back(chamberId);
        }

        if (segment->hasZed()) {
          if (m_dt2.find(chamberId) == m_dt2.end()) {
            AlignableDetOrUnitPtr chamberAlignable = navigator->alignableFromDetId(chamberId);
            // YP
            //            m_dt2[chamberId] = new MuonTrackDT2ChamberResidual(globalGeometry, navigator, chamberId, chamberAlignable);
          } else if (m_debug)
            std::cout << "multi segment match to tmuon: dt2  -- should not happen!" << std::endl;
          m_dt2[chamberId]->setSegmentResidual(&(*chamberMatch), &(*segMatch));
        }
        if (segment->hasPhi()) {
          if (m_dt13.find(chamberId) == m_dt13.end()) {
            AlignableDetOrUnitPtr chamberAlignable = navigator->alignableFromDetId(chamberId);
            // YP
            //            m_dt13[chamberId] = new MuonTrackDT13ChamberResidual(globalGeometry, navigator, chamberId, chamberAlignable);
          } else if (m_debug)
            std::cout << "multi segment match to tmuon: dt13  -- should not happen!" << std::endl;
          m_dt13[chamberId]->setSegmentResidual(&(*chamberMatch), &(*segMatch));
        }
      }

      else if (chamberMatch->id.subdetId() == MuonSubdetId::CSC) {
        const CSCDetId cscDetId(chamberMatch->id.rawId());
        const CSCDetId chamberId(cscDetId.chamberId());

        if (fabs(chamberMatch->x - segMatch->x) > maxResidual)
          continue;

        // have we seen this chamber before?
        if (m_csc.find(chamberId) == m_csc.end()) {
          m_chamberIds.push_back(chamberId);
          AlignableDetOrUnitPtr chamberAlignable = navigator->alignableFromDetId(chamberId);
          // YP
          //          m_csc[chamberId] = new MuonTrackCSCChamberResidual(globalGeometry, navigator, chamberId, chamberAlignable);
        } else if (m_debug)
          std::cout << "multi segment match to tmuon: csc  -- should not happen!" << std::endl;
        m_csc[chamberId]->setSegmentResidual(&(*chamberMatch), &(*segMatch));
      }
    }
  }
}

// This is destructor
// It deletes all chambers residulas
MuonResidualsFromTrack::~MuonResidualsFromTrack() {
  for (std::map<DetId, MuonChamberResidual*>::const_iterator residual = m_dt13.begin(); residual != m_dt13.end();
       ++residual) {
    delete residual->second;
  }
  for (std::map<DetId, MuonChamberResidual*>::const_iterator residual = m_dt2.begin(); residual != m_dt2.end();
       ++residual) {
    delete residual->second;
  }
  for (std::map<DetId, MuonChamberResidual*>::const_iterator residual = m_csc.begin(); residual != m_csc.end();
       ++residual) {
    delete residual->second;
  }
}

void MuonResidualsFromTrack::clear() {
  m_tracker_numHits = 0;
  m_tracker_chi2 = 0.;
  m_contains_TIDTEC = false;
  m_chamberIds.clear();
  m_dt13.clear();
  m_dt2.clear();
  m_csc.clear();
  m_trkCovMatrix.clear();
}

double MuonResidualsFromTrack::trackerRedChi2() const {
  if (m_tracker_numHits > 5)
    return m_tracker_chi2 / double(m_tracker_numHits - 5);
  else
    return -1.;
}

double MuonResidualsFromTrack::normalizedChi2() const {
  if (m_recoMuon)
    return m_recoTrack->normalizedChi2();
  return trackerRedChi2();
}

MuonChamberResidual* MuonResidualsFromTrack::chamberResidual(DetId chamberId, int type) {
  if (type == MuonChamberResidual::kDT13) {
    if (m_dt13.find(chamberId) == m_dt13.end())
      return nullptr;
    return m_dt13[chamberId];
  } else if (type == MuonChamberResidual::kDT2) {
    if (m_dt2.find(chamberId) == m_dt2.end())
      return nullptr;
    return m_dt2[chamberId];
  } else if (type == MuonChamberResidual::kCSC) {
    if (m_csc.find(chamberId) == m_csc.end())
      return nullptr;
    return m_csc[chamberId];
  } else
    return nullptr;
}

void MuonResidualsFromTrack::addTrkCovMatrix(DetId chamberId, TrajectoryStateOnSurface& tsos) {
  const AlgebraicSymMatrix55 cov55 = tsos.localError().matrix();
  TMatrixDSym cov44(4);
  // change indices from q/p,dxdz,dydz,x,y   to   x,y,dxdz,dydz
  int subs[4] = {3, 4, 1, 2};
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      cov44(i, j) = cov55(subs[i], subs[j]);
  m_trkCovMatrix[chamberId] = cov44;
}

TMatrixDSym MuonResidualsFromTrack::covMatrix(DetId chamberId) {
  bool m_debug = false;

  TMatrixDSym result(4);
  if (m_debug)
    std::cout << "MuonResidualsFromTrack:: cov initial:" << std::endl;
  result.Print();
  if (m_trkCovMatrix.find(chamberId) == m_trkCovMatrix.end()) {
    if (m_debug)
      std::cout << "MuonResidualsFromTrack:: cov does not exist!" << std::endl;
    return result;
  }
  result = m_trkCovMatrix[chamberId];

  if (m_debug)
    std::cout << "MuonResidualsFromTrack:: cov before:" << std::endl;
  result.Print();

  // add segment's errors in quadratures to track's covariance matrix
  double r_err;
  if (m_csc.find(chamberId) == m_csc.end()) {
    r_err = m_csc[chamberId]->residual_error();
    result(0, 0) += r_err * r_err;
    r_err = m_csc[chamberId]->resslope_error();
    result(2, 2) += r_err * r_err;
  }
  if (m_dt13.find(chamberId) == m_dt13.end()) {
    r_err = m_dt13[chamberId]->residual_error();
    result(0, 0) += r_err * r_err;
    r_err = m_dt13[chamberId]->resslope_error();
    result(2, 2) += r_err * r_err;
  }
  if (m_dt2.find(chamberId) == m_dt2.end()) {
    r_err = m_dt2[chamberId]->residual_error();
    result(1, 1) += r_err * r_err;
    r_err = m_dt2[chamberId]->resslope_error();
    result(3, 3) += r_err * r_err;
  }
  if (m_debug)
    std::cout << "MuonResidualsFromTrack:: cov after:" << std::endl;
  result.Print();

  return result;
}

TMatrixDSym MuonResidualsFromTrack::corrMatrix(DetId chamberId) {
  bool m_debug = false;

  TMatrixDSym result(4);
  TMatrixDSym cov44 = covMatrix(chamberId);

  // invert it using cholesky decomposition
  TDecompChol decomp(cov44);
  bool ok = decomp.Invert(result);
  if (m_debug)
    std::cout << "MuonResidualsFromTrack:: corr after:" << std::endl;
  result.Print();

  if (!ok && m_debug)
    std::cout << "MuonResidualsFromTrack:: cov inversion failed!" << std::endl;
  return result;
}

TMatrixD MuonResidualsFromTrack::choleskyCorrMatrix(DetId chamberId) {
  bool m_debug = false;

  TMatrixD result(4, 4);
  TMatrixDSym corr44 = corrMatrix(chamberId);

  // get an upper triangular matrix U such that corr = U^T * U
  TDecompChol decomp(corr44);
  bool ok = decomp.Decompose();
  result = decomp.GetU();

  if (m_debug)
    std::cout << "MuonResidualsFromTrack:: corr cholesky after:" << std::endl;
  result.Print();

  if (!ok && m_debug)
    std::cout << "MuonResidualsFromTrack:: corr decomposition failed!" << std::endl;
  return result;
}
