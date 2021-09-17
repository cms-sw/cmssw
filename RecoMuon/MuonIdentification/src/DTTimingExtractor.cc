// -*- C++ -*-
//
// Package:    MuonIdentification
// Class:      DTTimingExtractor
//
/**\class DTTimingExtractor DTTimingExtractor.cc RecoMuon/MuonIdentification/src/DTTimingExtractor.cc
 *
 * Description: Produce timing information for a muon track using 1D DT hits from segments used to build the track
 *
 */
//
// Original Author:  Traczyk Piotr
//         Created:  Thu Oct 11 15:01:28 CEST 2007
//
//

#include "RecoMuon/MuonIdentification/interface/DTTimingExtractor.h"

// user include files
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"
#include "DataFormats/DTRecHit/interface/DTSLRecSegment2D.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2D.h"

#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2D.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2DCollection.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"

// system include files
#include <memory>
#include <vector>
#include <string>
#include <iostream>

namespace edm {
  class ParameterSet;
  class EventSetup;
  class InputTag;
}  // namespace edm

class MuonServiceProxy;

//
// constructors and destructor
//
DTTimingExtractor::DTTimingExtractor(const edm::ParameterSet& iConfig,
                                     MuonSegmentMatcher* segMatcher,
                                     edm::ConsumesCollector& iC)
    : theHitsMin_(iConfig.getParameter<int>("HitsMin")),
      thePruneCut_(iConfig.getParameter<double>("PruneCut")),
      theTimeOffset_(iConfig.getParameter<double>("DTTimeOffset")),
      theError_(iConfig.getParameter<double>("HitError")),
      useSegmentT0_(iConfig.getParameter<bool>("UseSegmentT0")),
      doWireCorr_(iConfig.getParameter<bool>("DoWireCorr")),
      dropTheta_(iConfig.getParameter<bool>("DropTheta")),
      requireBothProjections_(iConfig.getParameter<bool>("RequireBothProjections")),
      debug(iConfig.getParameter<bool>("debug")) {
  edm::ParameterSet serviceParameters = iConfig.getParameter<edm::ParameterSet>("ServiceParameters");
  theService = std::make_unique<MuonServiceProxy>(serviceParameters, edm::ConsumesCollector(iC));
  theMatcher = segMatcher;
}

DTTimingExtractor::~DTTimingExtractor() {}

//
// member functions
//
void DTTimingExtractor::fillTiming(TimeMeasurementSequence& tmSequence,
                                   const std::vector<const DTRecSegment4D*>& segments,
                                   reco::TrackRef muonTrack,
                                   const edm::Event& iEvent,
                                   const edm::EventSetup& iSetup) {
  if (debug)
    std::cout << " *** DT Timimng Extractor ***" << std::endl;

  theService->update(iSetup);

  const GlobalTrackingGeometry* theTrackingGeometry = &*theService->trackingGeometry();

  // get the DT geometry
  edm::ESHandle<DTGeometry> theDTGeom;
  iSetup.get<MuonGeometryRecord>().get(theDTGeom);

  // get the propagator
  edm::ESHandle<Propagator> propagator;
  iSetup.get<TrackingComponentsRecord>().get("SteppingHelixPropagatorAny", propagator);
  const Propagator* propag = propagator.product();

  math::XYZPoint pos = muonTrack->innerPosition();
  math::XYZVector mom = muonTrack->innerMomentum();
  GlobalPoint posp(pos.x(), pos.y(), pos.z());
  GlobalVector momv(mom.x(), mom.y(), mom.z());
  FreeTrajectoryState muonFTS(posp, momv, (TrackCharge)muonTrack->charge(), theService->magneticField().product());

  // create a collection on TimeMeasurements for the track
  std::vector<TimeMeasurement> tms;
  for (const auto& rechit : segments) {
    // Create the ChamberId
    DetId id = rechit->geographicalId();
    DTChamberId chamberId(id.rawId());
    int station = chamberId.station();
    if (debug)
      std::cout << "Matched DT segment in station " << station << std::endl;

    // use only segments with both phi and theta projections present (optional)
    bool bothProjections = ((rechit->hasPhi()) && (rechit->hasZed()));
    if (requireBothProjections_ && !bothProjections)
      continue;

    // loop over (theta, phi) segments
    for (int phi = 0; phi < 2; phi++) {
      if (dropTheta_ && !phi)
        continue;

      const DTRecSegment2D* segm;
      if (phi) {
        segm = dynamic_cast<const DTRecSegment2D*>(rechit->phiSegment());
        if (debug)
          std::cout << " *** Segment t0: " << segm->t0() << std::endl;
      } else
        segm = dynamic_cast<const DTRecSegment2D*>(rechit->zSegment());

      if (segm == nullptr)
        continue;
      if (segm->specificRecHits().empty())
        continue;

      const GeomDet* geomDet = theTrackingGeometry->idToDet(segm->geographicalId());
      const std::vector<DTRecHit1D>& hits1d(segm->specificRecHits());

      // store all the hits from the segment
      for (const auto& hiti : hits1d) {
        const GeomDet* dtcell = theTrackingGeometry->idToDet(hiti.geographicalId());
        TimeMeasurement thisHit;

        std::pair<TrajectoryStateOnSurface, double> tsos;
        tsos = propag->propagateWithPath(muonFTS, dtcell->surface());

        double dist;
        double dist_straight = dtcell->toGlobal(hiti.localPosition()).mag();
        if (tsos.first.isValid()) {
          dist = tsos.second + posp.mag();
          //	  std::cout << "Propagate distance: " << dist << " ( innermost: " << posp.mag() << ")" << std::endl;
        } else {
          dist = dist_straight;
          //	  std::cout << "Geom distance: " << dist << std::endl;
        }

        thisHit.driftCell = hiti.geographicalId();
        if (hiti.lrSide() == DTEnums::Left)
          thisHit.isLeft = true;
        else
          thisHit.isLeft = false;
        thisHit.isPhi = phi;
        thisHit.posInLayer = geomDet->toLocal(dtcell->toGlobal(hiti.localPosition())).x();
        thisHit.distIP = dist;
        thisHit.station = station;
        if (useSegmentT0_ && segm->ist0Valid())
          thisHit.timeCorr = segm->t0();
        else
          thisHit.timeCorr = 0.;
        thisHit.timeCorr += theTimeOffset_;

        // signal propagation along the wire correction for unmached theta or phi segment hits
        if (doWireCorr_ && !bothProjections && tsos.first.isValid()) {
          const DTLayer* layer = theDTGeom->layer(hiti.wireId());
          float propgL = layer->toLocal(tsos.first.globalPosition()).y();
          float wirePropCorr = propgL / 24.4 * 0.00543;  // signal propagation speed along the wire
          if (thisHit.isLeft)
            wirePropCorr = -wirePropCorr;
          thisHit.posInLayer += wirePropCorr;
          const DTSuperLayer* sl = layer->superLayer();
          float tofCorr = sl->surface().position().mag() - tsos.first.globalPosition().mag();
          tofCorr = (tofCorr / 29.979) * 0.00543;
          if (thisHit.isLeft)
            tofCorr = -tofCorr;
          thisHit.posInLayer += tofCorr;
        } else {
          // non straight-line path correction
          float slCorr = (dist_straight - dist) / 29.979 * 0.00543;
          if (thisHit.isLeft)
            slCorr = -slCorr;
          thisHit.posInLayer += slCorr;
        }

        if (debug)
          std::cout << " dist: " << dist << "  t0: " << thisHit.posInLayer << std::endl;

        tms.push_back(thisHit);
      }
    }  // phi = (0,1)
  }    // rechit

  bool modified = false;
  std::vector<double> dstnc, local_t0, hitWeightTimeVtx, hitWeightInvbeta, left;
  double totalWeightInvbeta = 0;
  double totalWeightTimeVtx = 0;

  // Now loop over the measurements, calculate 1/beta and time at vertex and cut away outliers
  do {
    modified = false;
    dstnc.clear();
    local_t0.clear();
    hitWeightTimeVtx.clear();
    hitWeightInvbeta.clear();
    left.clear();

    std::vector<int> hit_idx;
    totalWeightInvbeta = 0;
    totalWeightTimeVtx = 0;

    // Rebuild segments
    for (int sta = 1; sta < 5; sta++)
      for (int phi = 0; phi < 2; phi++) {
        std::vector<TimeMeasurement> seg;
        std::vector<int> seg_idx;
        int tmpos = 0;
        for (auto& tm : tms) {
          if ((tm.station == sta) && (tm.isPhi == phi)) {
            seg.push_back(tm);
            seg_idx.push_back(tmpos);
          }
          tmpos++;
        }

        unsigned int segsize = seg.size();
        if (segsize < theHitsMin_)
          continue;

        double a = 0, b = 0;
        std::vector<double> hitxl, hitxr, hityl, hityr;

        for (auto& tm : seg) {
          DetId id = tm.driftCell;
          const GeomDet* dtcell = theTrackingGeometry->idToDet(id);
          DTChamberId chamberId(id.rawId());
          const GeomDet* dtcham = theTrackingGeometry->idToDet(chamberId);

          double celly = dtcham->toLocal(dtcell->position()).z();

          if (tm.isLeft) {
            hitxl.push_back(celly);
            hityl.push_back(tm.posInLayer);
          } else {
            hitxr.push_back(celly);
            hityr.push_back(tm.posInLayer);
          }
        }

        if (!fitT0(a, b, hitxl, hityl, hitxr, hityr)) {
          if (debug)
            std::cout << "     t0 = zero, Left hits: " << hitxl.size() << " Right hits: " << hitxr.size() << std::endl;
          continue;
        }

        // a segment must have at least one left and one right hit
        if ((hitxl.empty()) || (hityl.empty()))
          continue;

        int segidx = 0;
        for (const auto& tm : seg) {
          DetId id = tm.driftCell;
          const GeomDet* dtcell = theTrackingGeometry->idToDet(id);
          DTChamberId chamberId(id.rawId());
          const GeomDet* dtcham = theTrackingGeometry->idToDet(chamberId);

          double layerZ = dtcham->toLocal(dtcell->position()).z();
          double segmLocalPos = b + layerZ * a;
          double hitLocalPos = tm.posInLayer;
          int hitSide = -tm.isLeft * 2 + 1;
          double t0_segm = (-(hitSide * segmLocalPos) + (hitSide * hitLocalPos)) / 0.00543 + tm.timeCorr;

          if (debug)
            std::cout << "   Segm hit.  dstnc: " << tm.distIP << "   t0: " << t0_segm << std::endl;

          dstnc.push_back(tm.distIP);
          local_t0.push_back(t0_segm);
          left.push_back(hitSide);
          hitWeightInvbeta.push_back(((double)seg.size() - 2.) * tm.distIP * tm.distIP /
                                     ((double)seg.size() * 30. * 30. * theError_ * theError_));
          hitWeightTimeVtx.push_back(((double)seg.size() - 2.) / ((double)seg.size() * theError_ * theError_));
          hit_idx.push_back(seg_idx.at(segidx));
          segidx++;
          totalWeightInvbeta += ((double)seg.size() - 2.) * tm.distIP * tm.distIP /
                                ((double)seg.size() * 30. * 30. * theError_ * theError_);
          totalWeightTimeVtx += ((double)seg.size() - 2.) / ((double)seg.size() * theError_ * theError_);
        }
      }

    if (totalWeightInvbeta == 0)
      break;

    // calculate the value and error of 1/beta and timeVtx from the complete set of 1D hits
    if (debug)
      std::cout << " Points for global fit: " << dstnc.size() << std::endl;

    double invbeta = 0, invbetaErr = 0;
    double timeVtx = 0, timeVtxErr = 0;

    for (unsigned int i = 0; i < dstnc.size(); i++) {
      invbeta += (1. + local_t0.at(i) / dstnc.at(i) * 30.) * hitWeightInvbeta.at(i) / totalWeightInvbeta;
      timeVtx += local_t0.at(i) * hitWeightTimeVtx.at(i) / totalWeightTimeVtx;
    }

    double chimax = 0.;
    std::vector<TimeMeasurement>::iterator tmmax;

    // Calculate the inv beta and time at vertex dispersion
    double diff_ibeta, diff_tvtx;
    for (unsigned int i = 0; i < dstnc.size(); i++) {
      diff_ibeta = (1. + local_t0.at(i) / dstnc.at(i) * 30.) - invbeta;
      diff_ibeta = diff_ibeta * diff_ibeta * hitWeightInvbeta.at(i);
      diff_tvtx = local_t0.at(i) - timeVtx;
      diff_tvtx = diff_tvtx * diff_tvtx * hitWeightTimeVtx.at(i);
      invbetaErr += diff_ibeta;
      timeVtxErr += diff_tvtx;

      // decide if we cut away time at vertex outliers or inverse beta outliers
      // currently not configurable.
      if (diff_tvtx > chimax) {
        tmmax = tms.begin() + hit_idx.at(i);
        chimax = diff_tvtx;
      }
    }

    // cut away the outliers
    if (chimax > thePruneCut_) {
      tms.erase(tmmax);
      modified = true;
    }

    if (debug) {
      double cf = 1. / (dstnc.size() - 1);
      invbetaErr = sqrt(invbetaErr / totalWeightInvbeta * cf);
      timeVtxErr = sqrt(timeVtxErr / totalWeightTimeVtx * cf);
      std::cout << " Measured 1/beta: " << invbeta << " +/- " << invbetaErr << std::endl;
      std::cout << " Measured time: " << timeVtx << " +/- " << timeVtxErr << std::endl;
    }

  } while (modified);

  for (unsigned int i = 0; i < dstnc.size(); i++) {
    tmSequence.dstnc.push_back(dstnc.at(i));
    tmSequence.local_t0.push_back(local_t0.at(i));
    tmSequence.weightInvbeta.push_back(hitWeightInvbeta.at(i));
    tmSequence.weightTimeVtx.push_back(hitWeightTimeVtx.at(i));
  }

  tmSequence.totalWeightInvbeta = totalWeightInvbeta;
  tmSequence.totalWeightTimeVtx = totalWeightTimeVtx;
}

// ------------ method called to produce the data  ------------
void DTTimingExtractor::fillTiming(TimeMeasurementSequence& tmSequence,
                                   reco::TrackRef muonTrack,
                                   const edm::Event& iEvent,
                                   const edm::EventSetup& iSetup) {
  // get the DT segments that were used to construct the muon
  std::vector<const DTRecSegment4D*> range = theMatcher->matchDT(*muonTrack, iEvent);

  if (debug)
    std::cout << " The muon track matches " << range.size() << " segments." << std::endl;
  fillTiming(tmSequence, range, muonTrack, iEvent, iSetup);
}

double DTTimingExtractor::fitT0(double& a,
                                double& b,
                                const std::vector<double>& xl,
                                const std::vector<double>& yl,
                                const std::vector<double>& xr,
                                const std::vector<double>& yr) {
  double sx = 0, sy = 0, sxy = 0, sxx = 0, ssx = 0, ssy = 0, s = 0, ss = 0;

  for (unsigned int i = 0; i < xl.size(); i++) {
    sx += xl[i];
    sy += yl[i];
    sxy += xl[i] * yl[i];
    sxx += xl[i] * xl[i];
    s++;
    ssx += xl[i];
    ssy += yl[i];
    ss++;
  }

  for (unsigned int i = 0; i < xr.size(); i++) {
    sx += xr[i];
    sy += yr[i];
    sxy += xr[i] * yr[i];
    sxx += xr[i] * xr[i];
    s++;
    ssx -= xr[i];
    ssy -= yr[i];
    ss--;
  }

  double delta = ss * ss * sxx + s * sx * sx + s * ssx * ssx - s * s * sxx - 2 * ss * sx * ssx;

  double t0_corr = 0.;

  if (delta) {
    a = (ssy * s * ssx + sxy * ss * ss + sy * sx * s - sy * ss * ssx - ssy * sx * ss - sxy * s * s) / delta;
    b = (ssx * sy * ssx + sxx * ssy * ss + sx * sxy * s - sxx * sy * s - ssx * sxy * ss - sx * ssy * ssx) / delta;
    t0_corr = (ssx * s * sxy + sxx * ss * sy + sx * sx * ssy - sxx * s * ssy - sx * ss * sxy - ssx * sx * sy) / delta;
  }

  // convert drift distance to time
  t0_corr /= -0.00543;

  return t0_corr;
}

//define this as a plug-in
//DEFINE_FWK_MODULE(DTTimingExtractor);
