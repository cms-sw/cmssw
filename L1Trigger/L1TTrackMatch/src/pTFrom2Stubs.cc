#include "L1Trigger/L1TTrackMatch/interface/pTFrom2Stubs.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "Geometry/CommonTopologies/interface/Topology.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>

static constexpr float local_c_light = 0.00299792;
static constexpr float B_field = 3.8;

namespace pTFrom2Stubs {

  //====================
  float rInvFrom2(std::vector<TTTrack<Ref_Phase2TrackerDigi_> >::const_iterator trk,
                  const TrackerGeometry* tkGeometry) {
    //vector of R, r and phi for each stub
    std::vector<std::vector<float> > riPhiStubs(0);
    //get stub reference
    std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_> >, TTStub<Ref_Phase2TrackerDigi_> > >
        vecStubRefs = trk->getStubRefs();

    //loop over L1Track's stubs
    int rsize = vecStubRefs.size();
    for (int j = 0; j < rsize; ++j) {
      edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_> >, TTStub<Ref_Phase2TrackerDigi_> > stubRef =
          vecStubRefs.at(j);
      const TTStub<Ref_Phase2TrackerDigi_>* stub = &(*stubRef);
      MeasurementPoint localPos = stub->clusterRef(0)->findAverageLocalCoordinates();

      DetId detid = stub->clusterRef(0)->getDetId();

      if (detid.det() != DetId::Detector::Tracker)
        continue;
      if (detid.subdetId() != StripSubdetector::TOB && detid.subdetId() != StripSubdetector::TID)
        continue;
      const GeomDet* geomDet = tkGeometry->idToDet(detid);
      if (geomDet) {
        const GeomDetUnit* gDetUnit = tkGeometry->idToDetUnit(detid);
        GlobalPoint stubPosition = geomDet->surface().toGlobal(gDetUnit->topology().localPosition(localPos));

        std::vector<float> tmp(0);
        float Rad = sqrt(stubPosition.x() * stubPosition.x() + stubPosition.y() * stubPosition.y() + stubPosition.z() +
                         stubPosition.z());
        float r_i = sqrt(stubPosition.x() * stubPosition.x() + stubPosition.y() * stubPosition.y());
        float phi_i = stubPosition.phi();

        tmp.push_back(Rad);
        tmp.push_back(r_i);
        tmp.push_back(phi_i);

        riPhiStubs.push_back(tmp);
      }
    }

    std::sort(riPhiStubs.begin(), riPhiStubs.end());
    //now calculate the curvature from first 2 stubs
    float nr1 = (riPhiStubs[0])[1];
    float nphi1 = (riPhiStubs[0])[2];

    float nr2 = (riPhiStubs[1])[1];
    float nphi2 = (riPhiStubs[1])[2];

    float dPhi = reco::deltaPhi(nphi1, nphi2);

    float ndist = sqrt(nr2 * nr2 + nr1 * nr1 - 2 * nr1 * nr2 * cos(dPhi));

    float curvature = 2 * sin(dPhi) / ndist;
    return curvature;
  }
  //====================
  float pTFrom2(std::vector<TTTrack<Ref_Phase2TrackerDigi_> >::const_iterator trk, const TrackerGeometry* tkGeometry) {
    float rinv = rInvFrom2(trk, tkGeometry);
    return std::abs(local_c_light * B_field / rinv);
  }
  //====================
}  // namespace pTFrom2Stubs
