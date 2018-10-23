#include <iostream>
#include <memory>

#include "L1Trigger/TrackTrigger/interface/StubPtConsistency.h"

namespace StubPtConsistency {

  float getConsistency(TTTrack< Ref_Phase2TrackerDigi_ > aTrack, const TrackerGeometry* theTrackerGeom, const TrackerTopology* tTopo, double mMagneticFieldStrength, int nPar) {
    double trk_bendchi2 = 0.0;

    if ( !(nPar==4 || nPar==5)) {
      std::cerr << "Not a valid nPar option!" << std::endl;
      return trk_bendchi2;
    }

    double bend_resolution = 0.483;
    // Need the pT signed in order to determine if bend is positive or negative
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

      //input tilted module correction
      const GeomDetUnit* det0 = theTrackerGeom->idToDetUnit( detIdStub );
      const GeomDetUnit* det1 = theTrackerGeom->idToDetUnit( tTopo->partnerDetId( detIdStub ) );
      const PixelGeomDetUnit* unit = reinterpret_cast<const PixelGeomDetUnit*>( det0 );
      const PixelTopology& topo = unit->specificTopology();

      //Calculation of snesor spacing obtained from TMTT: https://github.com/CMS-TMTT/cmssw/blob/TMTT_938/L1Trigger/TrackFindingTMTT/src/Stub.cc#L138-L146
      float stripPitch = topo.pitch().first;

      float modMinR = std::min(det0->position().perp(),det1->position().perp());
      float modMaxR = std::max(det0->position().perp(),det1->position().perp());
      float modMinZ = std::min(det0->position().z(),det1->position().z());
      float modMaxZ = std::max(det0->position().z(),det1->position().z());
      float sensorSpacing = sqrt((modMaxR-modMinR)*(modMaxR-modMinR) + (modMaxZ-modMinZ)*(modMaxZ-modMinZ));

      //Approximation of phiOverBendCorrection, from TMTT: https://github.com/CMS-TMTT/cmssw/blob/TMTT_938/L1Trigger/TrackFindingTMTT/src/Stub.cc#L440-L448
      bool tiltedBarrel = (isBarrel && tTopo->tobSide(detIdStub)!=3);
      float correction;
      if (tiltedBarrel) correction= 0.886454*fabs(stub_z)/stub_r+0.504148;
      else if (isBarrel) correction=1;
      else correction= fabs(stub_z)/stub_r;

      float stubBend = stubRefs.at(is)->getTriggerBend();
      if (!isBarrel && stub_z<0.0) stubBend=-stubBend;
      float trackBend = -(sensorSpacing*stub_r*mMagneticFieldStrength*(3.0E8/2.0E11))/(stripPitch*trk_signedPt*correction);
      float bendDiff = trackBend-stubBend;

      trk_bendchi2 += (bendDiff*bendDiff)/(bend_resolution*bend_resolution);
    }// end loop over stubs

    float bendchi2 = trk_bendchi2/nStubs;
    return bendchi2;
  } //end getConsistency()
}
