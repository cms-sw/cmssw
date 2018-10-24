#include <iostream>
#include <memory>

#include "L1Trigger/TrackTrigger/interface/StubPtConsistency.h"
#include "CLHEP/Units/PhysicalConstants.h"

namespace StubPtConsistency {

  float getConsistency(TTTrack< Ref_Phase2TrackerDigi_ > aTrack, const TrackerGeometry* theTrackerGeom, const TrackerTopology* tTopo, double mMagneticFieldStrength, int nPar) {

    if( !(nPar==4 || nPar==5) ) throw cms::Exception("IncorrectInput") << "Not a valid nPar option!";

    double trk_bendchi2 = 0.0;
    double bend_resolution = 0.483;
    float speedOfLightConverted = CLHEP::c_light/1.0E5; // B*c/2E11 - converts q/pt to track angle at some radius from beamline

    // Need the pT signed in order to determine if bend is positive or negative
    float trk_signedPt = speedOfLightConverted*mMagneticFieldStrength/aTrack.getRInv(nPar); // P(MeV/c) = (c/10^9)·Q·B(kG)·R(cm)

    // loop over stubs
    const auto& stubRefs = aTrack.getStubRefs();
    int nStubs = stubRefs.size();

    for ( const auto& stubRef : stubRefs) {
      DetId detIdStub = theTrackerGeom->idToDet( (stubRef->getClusterRef(0))->getDetId() )->geographicalId();
      MeasurementPoint coords = stubRef->getClusterRef(0)->findAverageLocalCoordinatesCentered();
      const GeomDet* theGeomDet = theTrackerGeom->idToDet(detIdStub);
      Global3DPoint posStub = theGeomDet->surface().toGlobal( theGeomDet->topology().localPosition(coords) );

      float stub_r = posStub.perp();
      float stub_z = posStub.z();

      bool isBarrel = (detIdStub.subdetId()==StripSubdetector::TOB);

      const GeomDetUnit* det0 = theTrackerGeom->idToDetUnit( detIdStub );
      const GeomDetUnit* det1 = theTrackerGeom->idToDetUnit( tTopo->partnerDetId( detIdStub ) );
      const PixelGeomDetUnit* unit = reinterpret_cast<const PixelGeomDetUnit*>( det0 );
      const PixelTopology& topo = unit->specificTopology();

      // Calculation of snesor spacing obtained from TMTT: https://github.com/CMS-TMTT/cmssw/blob/TMTT_938/L1Trigger/TrackFindingTMTT/src/Stub.cc#L138-L146
      float stripPitch = topo.pitch().first;

      float modMinR = std::min(det0->position().perp(),det1->position().perp());
      float modMaxR = std::max(det0->position().perp(),det1->position().perp());
      float modMinZ = std::min(det0->position().z(),det1->position().z());
      float modMaxZ = std::max(det0->position().z(),det1->position().z());
      float sensorSpacing = sqrt((modMaxR-modMinR)*(modMaxR-modMinR) + (modMaxZ-modMinZ)*(modMaxZ-modMinZ));

      // Approximation of phiOverBendCorrection, from TMTT: https://github.com/CMS-TMTT/cmssw/blob/TMTT_938/L1Trigger/TrackFindingTMTT/src/Stub.cc#L440-L448
      bool tiltedBarrel = (isBarrel && tTopo->tobSide(detIdStub)!=3);
      float gradient = 0.886454;
      float intercept = 0.504148;
      float correction;
      if (tiltedBarrel) correction = gradient*fabs(stub_z)/stub_r + intercept;
      else if (isBarrel) correction = 1;
      else correction = fabs(stub_z)/stub_r;

      float stubBend = stubRef->getTriggerBend();
      if (!isBarrel && stub_z<0.0) stubBend=-stubBend; // flip sign of bend if in negative end cap

      float trackBend = -(sensorSpacing*stub_r*mMagneticFieldStrength*(speedOfLightConverted/2))/(stripPitch*trk_signedPt*correction);
      float bendDiff = trackBend-stubBend;

      trk_bendchi2 += (bendDiff*bendDiff)/(bend_resolution*bend_resolution);
    }// end loop over stubs

    float bendchi2 = trk_bendchi2/nStubs;
    return bendchi2;
  } //end getConsistency()
}
