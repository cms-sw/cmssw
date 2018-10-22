#include <iostream>
#include <memory>

#include "L1Trigger/TrackTrigger/interface/StubPtConsistency.h"

namespace StubPtConsistency {

  float getConsistency(TTTrack< Ref_Phase2TrackerDigi_ > aTrack, const TrackerGeometry* theTrackerGeom, const TrackerTopology* tTopo, int nPar) {
    double trk_bendchi2 = 0.0;

    if ( !(nPar==4 || nPar==5)) {
      std::cerr << "Not a valid nPar option!" << std::endl;
      return trk_bendchi2;
    }

    double bend_resolution = 0.463;
    float trk_signedPt = 0.3*3.811202/100.0/(aTrack.getRInv(nPar));

    // loop over stubs
    std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_Phase2TrackerDigi_ > >, TTStub< Ref_Phase2TrackerDigi_ > > > stubRefs = aTrack.getStubRefs();
    int nStubs = stubRefs.size();

    for (int is=0; is<nStubs; is++) {
      DetId detIdStub = theTrackerGeom->idToDet( (stubRefs.at(is)->getClusterRef(0))->getDetId() )->geographicalId();
      MeasurementPoint coords = stubRefs.at(is)->getClusterRef(0)->findAverageLocalCoordinatesCentered();
      const GeomDet* theGeomDet = theTrackerGeom->idToDet(detIdStub);
      Global3DPoint posStub = theGeomDet->surface().toGlobal( theGeomDet->topology().localPosition(coords) );

      float stub_r = posStub.perp();
      float stub_z = posStub.z();

      bool isBarrel = false;
      if ( detIdStub.subdetId()==StripSubdetector::TOB ) {
        isBarrel = true;
      }
      else if ( detIdStub.subdetId()==StripSubdetector::TID ) {
        isBarrel = false;
      }

      float pitch = 0.089;

      // if stub is PS module
      if (theTrackerGeom->getDetectorType(detIdStub)==TrackerGeometry::ModuleType::Ph2PSP){
        pitch = 0.099;
      }

      //input tilted module correction
      const GeomDetUnit* det0 = theTrackerGeom->idToDetUnit( detIdStub );
      const GeomDetUnit* det1 = theTrackerGeom->idToDetUnit( tTopo->partnerDetId( detIdStub ) );
      bool tiltedBarrel = (isBarrel && tTopo->tobSide(detIdStub)!=3);

      float modMinR = std::min(det0->position().perp(),det1->position().perp());
      float modMaxR = std::max(det0->position().perp(),det1->position().perp());
      float modMinZ = std::min(det0->position().z(),det1->position().z());
      float modMaxZ = std::max(det0->position().z(),det1->position().z());

      float sensorSpacing = sqrt((modMaxR-modMinR)*(modMaxR-modMinR) + (modMaxZ-modMinZ)*(modMaxZ-modMinZ));
      float correction;
      if (tiltedBarrel) correction= 0.886454*fabs(stub_z)/stub_r+0.504148;
      else if (isBarrel) correction=1;
      else correction= fabs(stub_z)/stub_r;

      float stubBend = stubRefs.at(is)->getTriggerBend();
      if (!isBarrel && stub_z<0.0) stubBend=-stubBend;
      float trackBend = -(sensorSpacing*0.57*stub_r/10)/(pitch*trk_signedPt*correction);
      float bendDiff = trackBend-stubBend;

      trk_bendchi2 += (bendDiff*bendDiff)/(bend_resolution*bend_resolution);
    }// end loop over stubs

    float bendchi2 = trk_bendchi2/nStubs;
    return bendchi2;
  } //end getConsistency()
}
