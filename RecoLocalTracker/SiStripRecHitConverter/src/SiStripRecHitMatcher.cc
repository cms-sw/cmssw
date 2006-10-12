// File: SiStripRecHitMatcher.cc
// Description:  Matches into rechits
// Author:  C.Genta

#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

  SiStripRecHitMatcher::SiStripRecHitMatcher(const edm::ParameterSet& conf){   
    scale_=conf.getParameter<double>("NSigmaInside");  
  };



//match a single hit
const SiStripMatchedRecHit2D& SiStripRecHitMatcher::match(const SiStripRecHit2D *monoRH, 
					    const SiStripRecHit2D *stereoRH,
					    const GluedGeomDet* gluedDet,
					    LocalVector trackdirection) const{
  SimpleHitCollection stereoHits;
  stereoHits.push_back(stereoRH);
  //const StripGeomDetUnit* monoDet = dynamic_cast< const StripGeomDetUnit*>(gluedDet->monoDet());
  //  const GeomDetUnit* stereoDet = gluedDet->stereoDet();
  edm::OwnVector<SiStripMatchedRecHit2D> collection;
  collection= match( monoRH,
		     stereoHits.begin(), stereoHits.end(), 
		     gluedDet,trackdirection);
  return *collection.begin();
}


edm::OwnVector<SiStripMatchedRecHit2D> 
SiStripRecHitMatcher::match( const  SiStripRecHit2D *monoRH,
			     RecHitIterator &begin, RecHitIterator &end, 
			     const GluedGeomDet* gluedDet,
			     LocalVector trackdirection) const
{
  SimpleHitCollection stereoHits;
  for (RecHitIterator i=begin; i != end; ++i) {
    stereoHits.push_back( &(*i)); // convert to simple pointer
  }
  return match( monoRH,
		stereoHits.begin(), stereoHits.end(), 
		gluedDet,trackdirection);
}

edm::OwnVector<SiStripMatchedRecHit2D> 
SiStripRecHitMatcher::match( const  SiStripRecHit2D *monoRH,
			     SimpleHitIterator begin, SimpleHitIterator end,
			     const GluedGeomDet* gluedDet,
			     LocalVector trackdirection) const
{
  // stripdet = mono
  // partnerstripdet = stereo
  const GeomDetUnit* stripdet = gluedDet->monoDet();
  const GeomDetUnit* partnerstripdet = gluedDet->stereoDet();
  const StripTopology& topol=(const StripTopology&)stripdet->topology();
  edm::OwnVector<SiStripMatchedRecHit2D> collector;
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
  double c1=sin(RPHIpositiononGluedendvector.phi()); double s1=-cos(RPHIpositiononGluedendvector.phi());
  MeasurementError errormonoRH=topol.measurementError(monoRH->localPosition(),monoRH->localPositionError());
  double sigmap12=errormonoRH.uu()*pow(topol.localPitch(monoRH->localPosition()),2);
  SimpleHitIterator seconditer;  

  for(seconditer=begin;seconditer!=end;++seconditer){//iterate on stereo rechits

    // position of the initial and final point of the strip (STEREO cluster)
    MeasurementPoint STEREOpoint=partnertopol.measurementPosition((*seconditer)->localPosition());
    MeasurementPoint STEREOpointini=MeasurementPoint(STEREOpoint.x(),-0.5);
    MeasurementPoint STEREOpointend=MeasurementPoint(STEREOpoint.x(),0.5);

    // position of the initial and final point of the strip in local coordinates (stereo det)
    StripPosition stripstereo(partnertopol.localPosition(STEREOpointini),partnertopol.localPosition(STEREOpointend));
 
    //project stereo hit on glued det
    StripPosition projectedstripstereo=project(partnerstripdet,gluedDet,stripstereo,trackdirection);

    //perform the matching
   //(x2-x1)(y-y1)=(y2-y1)(x-x1)
    AlgebraicMatrix m(2,2); AlgebraicVector c(2), solution(2);
    m(1,1)=-(projectedstripmono.second.y()-projectedstripmono.first.y()); m(1,2)=(projectedstripmono.second.x()-projectedstripmono.first.x());
    m(2,1)=-(projectedstripstereo.second.y()-projectedstripstereo.first.y()); m(2,2)=(projectedstripstereo.second.x()-projectedstripstereo.first.x());
    c(1)=m(1,2)*projectedstripmono.first.y()+m(1,1)*projectedstripmono.first.x();
    c(2)=m(2,2)*projectedstripstereo.first.y()+m(2,1)*projectedstripstereo.first.x();
    solution=solve(m,c);
    position=LocalPoint(solution(1),solution(2));


    // then calculate the error
    LocalVector  stereopositiononGluedendvector=projectedstripstereo.second-projectedstripstereo.first;
    double c2=sin(stereopositiononGluedendvector.phi()); double s2=-cos(stereopositiononGluedendvector.phi());
    MeasurementError errorstereoRH=partnertopol.measurementError((*seconditer)->localPosition(),(*seconditer)->localPositionError());
    double sigmap22=errorstereoRH.uu()*pow(partnertopol.localPitch((*seconditer)->localPosition()),2);
    double invdet2=1/pow((c1*s2-c2*s1),2);
    float xx=invdet2*(sigmap12*s2*s2+sigmap22*s1*s1);
    float xy=-invdet2*(sigmap12*c2*s2+sigmap22*c1*s1);
    float yy=invdet2*(sigmap12*c2*c2+sigmap22*c1*c1);
    LocalError error=LocalError(xx,xy,yy);

    if((gluedDet->surface()).bounds().inside(position,error,scale_)){ //if it is inside the gluedet bonds
      //Change NSigmaInside in the configuration file to accept more hits
      //...and add it to the Rechit collection 

      const SiStripRecHit2D* secondHit = *seconditer;
      collector.push_back(new SiStripMatchedRecHit2D(position, error,gluedDet->geographicalId() ,
							     monoRH,secondHit));
    }
  }
  return collector;
}


SiStripRecHitMatcher::StripPosition SiStripRecHitMatcher::project(const GeomDetUnit *det,const GluedGeomDet* glueddet,StripPosition strip,LocalVector trackdirection)const
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
