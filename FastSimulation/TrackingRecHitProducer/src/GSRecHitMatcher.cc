#include "FastSimulation/TrackingRecHitProducer/interface/GSRecHitMatcher.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonTopologies/interface/RadialStripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include <cfloat>

SiTrackerGSRecHit2D * GSRecHitMatcher::match( const SiTrackerGSRecHit2D *monoRH,
					      const SiTrackerGSRecHit2D *stereoRH,
					      const GluedGeomDet* gluedDet,
					            LocalVector& trackdirection) const
{
  // stripdet = mono
  // partnerstripdet = stereo
  const GeomDetUnit* stripdet = gluedDet->monoDet();
  const GeomDetUnit* partnerstripdet = gluedDet->stereoDet();
  const StripTopology& topol=(const StripTopology&)stripdet->topology();

  LocalPoint position;    

  // position of the initial and final point of the strip (RPHI cluster) in local strip coordinates
  MeasurementPoint RPHIpoint=topol.measurementPosition(monoRH->localPosition());
  MeasurementPoint RPHIpointini=MeasurementPoint(RPHIpoint.x(),-0.5);
  MeasurementPoint RPHIpointend=MeasurementPoint(RPHIpoint.x(),0.5);
  
  // position of the initial and final point of the strip in local coordinates (mono det)
  StripPosition stripmono=StripPosition(topol.localPosition(RPHIpointini),topol.localPosition(RPHIpointend));

  if(trackdirection.mag2()<FLT_MIN){// in case of no track hypothesis assume a track from the origin through the center of the strip
    LocalPoint lcenterofstrip=monoRH->localPosition();
    GlobalPoint gcenterofstrip=(stripdet->surface()).toGlobal(lcenterofstrip);
    GlobalVector gtrackdirection=gcenterofstrip-GlobalPoint(0,0,0);
    trackdirection=(gluedDet->surface()).toLocal(gtrackdirection);
  }
 
  //project mono hit on glued det
  StripPosition projectedstripmono=project(stripdet,gluedDet,stripmono,trackdirection);
  const StripTopology& partnertopol=(const StripTopology&)partnerstripdet->topology();

  //error calculation (the part that depends on mono RH only)
  LocalVector  RPHIpositiononGluedendvector=projectedstripmono.second-projectedstripmono.first;
  double c1=sin(RPHIpositiononGluedendvector.phi()); 
  double s1=-cos(RPHIpositiononGluedendvector.phi());
  MeasurementError errormonoRH=topol.measurementError(monoRH->localPosition(),monoRH->localPositionError());
  double pitch=topol.localPitch(monoRH->localPosition());
  double sigmap12=errormonoRH.uu()*pitch*pitch;

  // position of the initial and final point of the strip (STEREO cluster)
  MeasurementPoint STEREOpoint=partnertopol.measurementPosition(stereoRH->localPosition());

  MeasurementPoint STEREOpointini=MeasurementPoint(STEREOpoint.x(),-0.5);
  MeasurementPoint STEREOpointend=MeasurementPoint(STEREOpoint.x(),0.5);

  // position of the initial and final point of the strip in local coordinates (stereo det)
  StripPosition stripstereo(partnertopol.localPosition(STEREOpointini),partnertopol.localPosition(STEREOpointend));

  //project stereo hit on glued det
  StripPosition projectedstripstereo=project(partnerstripdet,gluedDet,stripstereo,trackdirection);

  //perform the matching
  //(x2-x1)(y-y1)=(y2-y1)(x-x1)
  AlgebraicMatrix22 m; AlgebraicVector2 c, solution;
  m(0,0)=-(projectedstripmono.second.y()-projectedstripmono.first.y()); m(0,1)=(projectedstripmono.second.x()-projectedstripmono.first.x());
  m(1,0)=-(projectedstripstereo.second.y()-projectedstripstereo.first.y()); m(1,1)=(projectedstripstereo.second.x()-projectedstripstereo.first.x());
  c(0)=m(0,1)*projectedstripmono.first.y()+m(0,0)*projectedstripmono.first.x();
  c(1)=m(1,1)*projectedstripstereo.first.y()+m(1,0)*projectedstripstereo.first.x();
  m.Invert(); solution = m * c;
  position=LocalPoint(solution(0),solution(1));


  //
  // temporary fix by tommaso
  //


  LocalError tempError (100,0,100);

  // calculate the error
  LocalVector  stereopositiononGluedendvector=projectedstripstereo.second-projectedstripstereo.first;
  double c2=sin(stereopositiononGluedendvector.phi()); double s2=-cos(stereopositiononGluedendvector.phi());
  MeasurementError errorstereoRH=partnertopol.measurementError(stereoRH->localPosition(), stereoRH->localPositionError());
  pitch=partnertopol.localPitch(stereoRH->localPosition());
  double sigmap22=errorstereoRH.uu()*pitch*pitch;
  double diff=(c1*s2-c2*s1);
  double invdet2=1/(diff*diff);
  float xx=invdet2*(sigmap12*s2*s2+sigmap22*s1*s1);
  float xy=-invdet2*(sigmap12*c2*s2+sigmap22*c1*s1);
  float yy=invdet2*(sigmap12*c2*c2+sigmap22*c1*c1);
  LocalError error=LocalError(xx,xy,yy);

 //  if((gluedDet->surface()).bounds().inside(position,error,3)){
//     std::cout<<"  ERROR ok "<< std::endl;
//   }
//   else {
//     std::cout<<" ERROR not ok " << std::endl;
//   }
  
  return new SiTrackerGSRecHit2D(position, error,gluedDet->geographicalId(), monoRH->simhitId(), 
				 monoRH->simtrackId(), monoRH->eeId(), monoRH->simMultX(), monoRH->simMultY() );
    
  
  
}


GSRecHitMatcher::StripPosition 
GSRecHitMatcher::project(const GeomDetUnit *det,
			 const GluedGeomDet* glueddet,
			 const StripPosition& strip,
			 const LocalVector& trackdirection)const
{

  GlobalPoint globalpointini=(det->surface()).toGlobal(strip.first);
  GlobalPoint globalpointend=(det->surface()).toGlobal(strip.second);

  // position of the initial and final point of the strip in glued local coordinates
  LocalPoint positiononGluedini=(glueddet->surface()).toLocal(globalpointini);
  LocalPoint positiononGluedend=(glueddet->surface()).toLocal(globalpointend);

  //correct the position with the track direction

  float scale=-positiononGluedini.z()/trackdirection.z();

  LocalPoint projpositiononGluedini= positiononGluedini + scale*trackdirection;
  LocalPoint projpositiononGluedend= positiononGluedend + scale*trackdirection;

  return StripPosition(projpositiononGluedini,projpositiononGluedend);
}



SiTrackerGSRecHit2D * GSRecHitMatcher::projectOnly( const SiTrackerGSRecHit2D *monoRH,
						    const GeomDet * monoDet,
						    const GluedGeomDet* gluedDet,
					            LocalVector& ldir) const
{
  LocalPoint position(monoRH->localPosition().x(), 0.,0.);
  const BoundPlane& gluedPlane = gluedDet->surface();
  const BoundPlane& hitPlane = monoDet->surface();

  double delta = gluedPlane.localZ( hitPlane.position());

  LocalPoint lhitPos = gluedPlane.toLocal( monoDet->surface().toGlobal(position ) );
  LocalPoint projectedHitPos = lhitPos - ldir * delta/ldir.z();

  LocalVector hitXAxis = gluedPlane.toLocal( hitPlane.toGlobal( LocalVector(1,0,0)));
  LocalError hitErr = monoRH->localPositionError();

  if (gluedPlane.normalVector().dot( hitPlane.normalVector()) < 0) {
    // the two planes are inverted, and the correlation element must change sign
    hitErr = LocalError( hitErr.xx(), -hitErr.xy(), hitErr.yy());
  }
  LocalError rotatedError = hitErr.rotate( hitXAxis.x(), hitXAxis.y());
 
  return new SiTrackerGSRecHit2D(projectedHitPos, rotatedError, gluedDet->geographicalId(), monoRH->simhitId(),  monoRH->simtrackId(), monoRH->eeId(), monoRH->simMultX(), monoRH->simMultY() );
}

