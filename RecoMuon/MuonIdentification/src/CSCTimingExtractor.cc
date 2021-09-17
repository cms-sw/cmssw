// -*- C++ -*-
//
// Package:    MuonIdentification
// Class:      CSCTimingExtractor
//
/**\class CSCTimingExtractor CSCTimingExtractor.cc RecoMuon/MuonIdentification/src/CSCTimingExtractor.cc
 *
 * Description: Produce timing information for a muon track using CSC hits from segments used to build the track
 *
 */
//
// Original Author:  Traczyk Piotr
//         Created:  Thu Oct 11 15:01:28 CEST 2007
//
//

#include "RecoMuon/MuonIdentification/interface/CSCTimingExtractor.h"

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

#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/CSCRecHit/interface/CSCSegment.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2D.h"
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
CSCTimingExtractor::CSCTimingExtractor(const edm::ParameterSet& iConfig,
                                       MuonSegmentMatcher* segMatcher,
                                       edm::ConsumesCollector& iC)
    : thePruneCut_(iConfig.getParameter<double>("PruneCut")),
      theStripTimeOffset_(iConfig.getParameter<double>("CSCStripTimeOffset")),
      theWireTimeOffset_(iConfig.getParameter<double>("CSCWireTimeOffset")),
      theStripError_(iConfig.getParameter<double>("CSCStripError")),
      theWireError_(iConfig.getParameter<double>("CSCWireError")),
      UseWireTime(iConfig.getParameter<bool>("UseWireTime")),
      UseStripTime(iConfig.getParameter<bool>("UseStripTime")),
      debug(iConfig.getParameter<bool>("debug")) {
  edm::ParameterSet serviceParameters = iConfig.getParameter<edm::ParameterSet>("ServiceParameters");
  theService = std::make_unique<MuonServiceProxy>(serviceParameters, edm::ConsumesCollector(iC));
  theMatcher = segMatcher;
}

CSCTimingExtractor::~CSCTimingExtractor() {}

//
// member functions
//

void CSCTimingExtractor::fillTiming(TimeMeasurementSequence& tmSequence,
                                    const std::vector<const CSCSegment*>& segments,
                                    reco::TrackRef muonTrack,
                                    const edm::Event& iEvent,
                                    const edm::EventSetup& iSetup) {
  theService->update(iSetup);

  const GlobalTrackingGeometry* theTrackingGeometry = &*theService->trackingGeometry();

  // get the propagator
  edm::ESHandle<Propagator> propagator;
  iSetup.get<TrackingComponentsRecord>().get("SteppingHelixPropagatorAny", propagator);
  const Propagator* propag = propagator.product();

  math::XYZPoint pos = muonTrack->innerPosition();
  math::XYZVector mom = muonTrack->innerMomentum();
  if (sqrt(muonTrack->innerPosition().mag2()) > sqrt(muonTrack->outerPosition().mag2())) {
    pos = muonTrack->outerPosition();
    mom = -1 * muonTrack->outerMomentum();
  }
  GlobalPoint posp(pos.x(), pos.y(), pos.z());
  GlobalVector momv(mom.x(), mom.y(), mom.z());
  FreeTrajectoryState muonFTS(posp, momv, (TrackCharge)muonTrack->charge(), theService->magneticField().product());

  // create a collection on TimeMeasurements for the track
  std::vector<TimeMeasurement> tms;
  for (const auto& rechit : segments) {
    // Create the ChamberId
    DetId id = rechit->geographicalId();
    CSCDetId chamberId(id.rawId());
    //    int station = chamberId.station();

    if (rechit->specificRecHits().empty())
      continue;

    const std::vector<CSCRecHit2D>& hits2d(rechit->specificRecHits());

    // store all the hits from the segment
    for (const auto& hiti : hits2d) {
      const GeomDet* cscDet = theTrackingGeometry->idToDet(hiti.geographicalId());
      TimeMeasurement thisHit;

      std::pair<TrajectoryStateOnSurface, double> tsos;
      tsos = propag->propagateWithPath(muonFTS, cscDet->surface());

      double dist;
      if (tsos.first.isValid())
        dist = tsos.second + posp.mag();
      else
        dist = cscDet->toGlobal(hiti.localPosition()).mag();

      thisHit.distIP = dist;
      if (UseStripTime) {
        thisHit.weightInvbeta = dist * dist / (theStripError_ * theStripError_ * 30. * 30.);
        thisHit.weightTimeVtx = 1. / (theStripError_ * theStripError_);
        thisHit.timeCorr = hiti.tpeak() - theStripTimeOffset_;
        tms.push_back(thisHit);
      }

      if (UseWireTime) {
        thisHit.weightInvbeta = dist * dist / (theWireError_ * theWireError_ * 30. * 30.);
        thisHit.weightTimeVtx = 1. / (theWireError_ * theWireError_);
        thisHit.timeCorr = hiti.wireTime() - theWireTimeOffset_;
        tms.push_back(thisHit);
      }

      //      std::cout << " CSC Hit. Dist= " << dist << "    Time= " << thisHit.timeCorr
      //           << "   invBeta= " << (1.+thisHit.timeCorr/dist*30.) << std::endl;
    }

  }  // rechit

  bool modified = false;
  std::vector<double> dstnc, local_t0, hitWeightInvbeta, hitWeightTimeVtx;
  double totalWeightInvbeta = 0;
  double totalWeightTimeVtx = 0;

  // Now loop over the measurements, calculate 1/beta and cut away outliers
  do {
    modified = false;
    dstnc.clear();
    local_t0.clear();
    hitWeightInvbeta.clear();
    hitWeightTimeVtx.clear();

    totalWeightInvbeta = 0;
    totalWeightTimeVtx = 0;

    for (auto& tm : tms) {
      dstnc.push_back(tm.distIP);
      local_t0.push_back(tm.timeCorr);
      hitWeightInvbeta.push_back(tm.weightInvbeta);
      hitWeightTimeVtx.push_back(tm.weightTimeVtx);
      totalWeightInvbeta += tm.weightInvbeta;
      totalWeightTimeVtx += tm.weightTimeVtx;
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
        tmmax = tms.begin() + i;
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
void CSCTimingExtractor::fillTiming(TimeMeasurementSequence& tmSequence,
                                    reco::TrackRef muonTrack,
                                    const edm::Event& iEvent,
                                    const edm::EventSetup& iSetup) {
  if (debug)
    std::cout << " *** CSC Timimng Extractor ***" << std::endl;

  // get the CSC segments that were used to construct the muon
  std::vector<const CSCSegment*> range = theMatcher->matchCSC(*muonTrack, iEvent);

  fillTiming(tmSequence, range, muonTrack, iEvent, iSetup);
}

//define this as a plug-in
//DEFINE_FWK_MODULE(CSCTimingExtractor);
