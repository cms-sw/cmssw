// File: SiStripRecHitMatcher.cc
// Description:  Matches into rechits
// Author:  C.Genta
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

#include "TrackingTools/TransientTrackingRecHit/interface/HelpertRecHit2DLocalPos.h"
#include<boost/bind.hpp>

#include <DataFormats/TrackingRecHit/interface/AlignmentPositionError.h>




SiStripRecHitMatcher::SiStripRecHitMatcher(const edm::ParameterSet& conf):
  scale_(conf.getParameter<double>("NSigmaInside")),
  preFilter_(conf.existsAs<bool>("PreFilter") ? conf.getParameter<bool>("PreFilter") : false)  
  {}

SiStripRecHitMatcher::SiStripRecHitMatcher(const double theScale):
  scale_(theScale){}  



namespace {
  // FIXME for c++0X
  inline void pb1(std::vector<SiStripMatchedRecHit2D*> & v, SiStripMatchedRecHit2D* h) {
    v.push_back(h);
  }
  inline void pb2(SiStripRecHitMatcher::CollectorMatched & v, const SiStripMatchedRecHit2D & h) {
    v.push_back(h);
  }

}


// needed by the obsolete version still in use on some architectures
void
SiStripRecHitMatcher::match( const SiStripRecHit2D *monoRH,
			     SimpleHitIterator begin, SimpleHitIterator end,
			     edm::OwnVector<SiStripMatchedRecHit2D> & collector, 
			     const GluedGeomDet* gluedDet,
			     LocalVector trackdirection) const {

  std::vector<SiStripMatchedRecHit2D*> result;
  result.reserve(end-begin);
  match(monoRH,begin,end,result,gluedDet,trackdirection);
  for (std::vector<SiStripMatchedRecHit2D*>::iterator p=result.begin(); p!=result.end();
       p++) collector.push_back(*p);
}


void
SiStripRecHitMatcher::match( const SiStripRecHit2D *monoRH,
			     SimpleHitIterator begin, SimpleHitIterator end,
			     std::vector<SiStripMatchedRecHit2D*> & collector, 
			     const GluedGeomDet* gluedDet,
			     LocalVector trackdirection) const {
  Collector result(boost::bind(&pb1,boost::ref(collector),
			     boost::bind(&SiStripMatchedRecHit2D::clone,_1)));
  match(monoRH,begin,end,result,gluedDet,trackdirection);
}

void
SiStripRecHitMatcher::match( const SiStripRecHit2D *monoRH,
			     SimpleHitIterator begin, SimpleHitIterator end,
			     CollectorMatched & collector,
			     const GluedGeomDet* gluedDet,
			     LocalVector trackdirection) const {

  Collector result(boost::bind(pb2,boost::ref(collector),_1));
  match(monoRH,begin,end,result,gluedDet,trackdirection);
  
}



// this is the one used by the RecHitConverter
void
SiStripRecHitMatcher::match( const SiStripRecHit2D *monoRH,
			     RecHitIterator begin, RecHitIterator end,
			     CollectorMatched & collector,
			     const GluedGeomDet* gluedDet,
			     LocalVector trackdirection) const {
  
  // is this really needed now????
  SimpleHitCollection stereoHits;
  stereoHits.reserve(end-begin);
  for (RecHitIterator i=begin; i != end; ++i) 
    stereoHits.push_back( &(*i)); // convert to simple pointer
  
  return match( monoRH,
		stereoHits.begin(), stereoHits.end(),
		collector,
		gluedDet,trackdirection);
}



// the "real implementation"
void
SiStripRecHitMatcher::match( const SiStripRecHit2D *monoRH,
			     SimpleHitIterator begin, SimpleHitIterator end,
			     Collector & collector, 
			     const GluedGeomDet* gluedDet,
			     LocalVector trackdirection) const {
  // stripdet = mono
  // partnerstripdet = stereo
  const GeomDetUnit* stripdet = gluedDet->monoDet();
  const GeomDetUnit* partnerstripdet = gluedDet->stereoDet();
  const StripTopology& topol=(const StripTopology&)stripdet->topology();

  // position of the initial and final point of the strip (RPHI cluster) in local strip coordinates
  double RPHIpointX = topol.measurementPosition(monoRH->localPositionFast()).x();
  MeasurementPoint RPHIpointini(RPHIpointX,-0.5);
  MeasurementPoint RPHIpointend(RPHIpointX,0.5);

  // position of the initial and final point of the strip in local coordinates (mono det)
  StripPosition stripmono=StripPosition(topol.localPosition(RPHIpointini),topol.localPosition(RPHIpointend));

  if(trackdirection.mag2()<FLT_MIN){// in case of no track hypothesis assume a track from the origin through the center of the strip
    LocalPoint lcenterofstrip=monoRH->localPositionFast();
    GlobalPoint gcenterofstrip=(stripdet->surface()).toGlobal(lcenterofstrip);
    GlobalVector gtrackdirection=gcenterofstrip-GlobalPoint(0,0,0);
    trackdirection=(gluedDet->surface()).toLocal(gtrackdirection);
  }

  //project mono hit on glued det
  StripPosition projectedstripmono=project(stripdet,gluedDet,stripmono,trackdirection);
  const StripTopology& partnertopol=(const StripTopology&)partnerstripdet->topology();

  double m00 = -(projectedstripmono.second.y()-projectedstripmono.first.y()); 
  double m01 =  (projectedstripmono.second.x()-projectedstripmono.first.x());
  double c0  =  m01*projectedstripmono.first.y()   + m00*projectedstripmono.first.x();
 
  //error calculation (the part that depends on mono RH only)
  //  LocalVector  RPHIpositiononGluedendvector=projectedstripmono.second-projectedstripmono.first;
  /*
  double l1 = 1./RPHIpositiononGluedendvector.perp2();
  double c1 = RPHIpositiononGluedendvector.y();
  double s1 =-RPHIpositiononGluedendvector.x();
  */
  double c1 = -m00;
  double s1 = -m01;
  double l1 = 1./(c1*c1+s1*s1);

 
  float sigmap12 = sigmaPitch(monoRH->localPosition(), monoRH->localPositionError(),topol);
  // auto sigmap12 = monoRH->sigmaPitch();
  // assert(sigmap12>=0);


  SimpleHitIterator seconditer;  

  for(seconditer=begin;seconditer!=end;++seconditer){//iterate on stereo rechits

    // position of the initial and final point of the strip (STEREO cluster)
    double STEREOpointX=partnertopol.measurementPosition((*seconditer)->localPositionFast()).x();
    MeasurementPoint STEREOpointini(STEREOpointX,-0.5);
    MeasurementPoint STEREOpointend(STEREOpointX,0.5);

    // position of the initial and final point of the strip in local coordinates (stereo det)
    StripPosition stripstereo(partnertopol.localPosition(STEREOpointini),partnertopol.localPosition(STEREOpointend));
 
    //project stereo hit on glued det
    StripPosition projectedstripstereo=project(partnerstripdet,gluedDet,stripstereo,trackdirection);

 
    double m10=-(projectedstripstereo.second.y()-projectedstripstereo.first.y()); 
    double m11=(projectedstripstereo.second.x()-projectedstripstereo.first.x());

   //perform the matching
   //(x2-x1)(y-y1)=(y2-y1)(x-x1)
    AlgebraicMatrix22 m; AlgebraicVector2 c; // FIXME understand why moving this initializer out of the loop changes the output!
    m(0,0)=m00; 
    m(0,1)=m01;
    m(1,0)=m10;
    m(1,1)=m11;
    c(0)=c0;
    c(1)=m11*projectedstripstereo.first.y()+m10*projectedstripstereo.first.x();
    m.Invert(); 
    AlgebraicVector2 solution = m * c;
    LocalPoint position(solution(0),solution(1));

    /*
    {
      double m00 = -(projectedstripmono.second.y()-projectedstripmono.first.y()); 
      double m01 =  (projectedstripmono.second.x()-projectedstripmono.first.x());
      double m10 = -(projectedstripstereo.second.y()-projectedstripstereo.first.y()); 
      double m11 =  (projectedstripstereo.second.x()-projectedstripstereo.first.x());
      double c0  =  m01*projectedstripmono.first.y()   + m00*projectedstripmono.first.x();
      double c1  =  m11*projectedstripstereo.first.y() + m10*projectedstripstereo.first.x();
      
      double invDet = 1./(m00*m11-m10*m01);
    }
    */

    //
    // temporary fix by tommaso
    //


    LocalError tempError (100,0,100);
    if (!((gluedDet->surface()).bounds().inside(position,tempError,scale_))) continue;                                                       

    // then calculate the error
    /*
    LocalVector  stereopositiononGluedendvector=projectedstripstereo.second-projectedstripstereo.first;
    double l2 = 1./stereopositiononGluedendvector.perp2();
    double c2 = stereopositiononGluedendvector.y(); 
    double s2 =-stereopositiononGluedendvector.x();
    */

    double c2 = -m10;
    double s2 = -m11;
    double l2 = 1./(c2*c2+s2*s2);

   float sigmap22 = sigmaPitch((*seconditer)->localPosition(),(*seconditer)->localPositionError(),partnertopol);
   // auto sigmap22 = (*seconditer)->sigmaPitch();
    // assert(sigmap22>=0);

    double diff=(c1*s2-c2*s1);
    double invdet2=1/(diff*diff*l1*l2);
    float xx= invdet2*(sigmap12*s2*s2*l2+sigmap22*s1*s1*l1);
    float xy=-invdet2*(sigmap12*c2*s2*l2+sigmap22*c1*s1*l1);
    float yy= invdet2*(sigmap12*c2*c2*l2+sigmap22*c1*c1*l1);
    LocalError error(xx,xy,yy);

    if((gluedDet->surface()).bounds().inside(position,error,scale_)){ //if it is inside the gluedet bonds
      //Change NSigmaInside in the configuration file to accept more hits
      //...and add it to the Rechit collection 

      const SiStripRecHit2D* secondHit = *seconditer;
      collector(SiStripMatchedRecHit2D(position, error,*gluedDet,
				       monoRH,secondHit));
    }
  }
}


SiStripRecHitMatcher::StripPosition 
SiStripRecHitMatcher::project(const GeomDetUnit *det,const GluedGeomDet* glueddet,
			      StripPosition strip,LocalVector trackdirection) const {

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



//match a single hit
SiStripMatchedRecHit2D * 
SiStripRecHitMatcher::match(const SiStripRecHit2D *monoRH, 
			    const SiStripRecHit2D *stereoRH,
			    const GluedGeomDet* gluedDet,
			    LocalVector trackdirection, bool force) const {
  // stripdet = mono
  // partnerstripdet = stereo
  const GeomDetUnit* stripdet = gluedDet->monoDet();
  const GeomDetUnit* partnerstripdet = gluedDet->stereoDet();
  const StripTopology& topol=(const StripTopology&)stripdet->topology();

  // position of the initial and final point of the strip (RPHI cluster) in local strip coordinates
  auto RPHIpointX = topol.measurementPosition(monoRH->localPositionFast()).x();
  MeasurementPoint RPHIpointini(RPHIpointX,-0.5f);
  MeasurementPoint RPHIpointend(RPHIpointX,0.5f);

  // position of the initial and final point of the strip in local coordinates (mono det)
  StripPosition stripmono=StripPosition(topol.localPosition(RPHIpointini),topol.localPosition(RPHIpointend));

  if(trackdirection.mag2()<float(FLT_MIN)){// in case of no track hypothesis assume a track from the origin through the center of the strip
    LocalPoint lcenterofstrip=monoRH->localPositionFast();
    GlobalPoint gcenterofstrip=(stripdet->surface()).toGlobal(lcenterofstrip);
    GlobalVector gtrackdirection=gcenterofstrip-GlobalPoint(0,0,0);
    trackdirection=(gluedDet->surface()).toLocal(gtrackdirection);
  }

  //project mono hit on glued det
  StripPosition projectedstripmono=project(stripdet,gluedDet,stripmono,trackdirection);
  const StripTopology& partnertopol=(const StripTopology&)partnerstripdet->topology();

  double m00 = -(projectedstripmono.second.y()-projectedstripmono.first.y()); 
  double m01 =  (projectedstripmono.second.x()-projectedstripmono.first.x());
  double c0  =  m01*projectedstripmono.first.y()   + m00*projectedstripmono.first.x();
 
  //error calculation (the part that depends on mono RH only)
  //  LocalVector  RPHIpositiononGluedendvector=projectedstripmono.second-projectedstripmono.first;
  /*
  double l1 = 1./RPHIpositiononGluedendvector.perp2();
  double c1 = RPHIpositiononGluedendvector.y();
  double s1 =-RPHIpositiononGluedendvector.x();
  */
  double c1 = -m00;
  double s1 = -m01;
  double l1 = 1./(c1*c1+s1*s1);


  float sigmap12 = sigmaPitch(monoRH->localPosition(), monoRH->localPositionError(),topol);
  // auto sigmap12 = monoRH->sigmaPitch();
  // assert(sigmap12>=0);




    // position of the initial and final point of the strip (STEREO cluster)
  auto STEREOpointX=partnertopol.measurementPosition(stereoRH->localPositionFast()).x();
  MeasurementPoint STEREOpointini(STEREOpointX,-0.5f);
  MeasurementPoint STEREOpointend(STEREOpointX,0.5f);
  
  // position of the initial and final point of the strip in local coordinates (stereo det)
  StripPosition stripstereo(partnertopol.localPosition(STEREOpointini),partnertopol.localPosition(STEREOpointend));
  
  //project stereo hit on glued det
  StripPosition projectedstripstereo=project(partnerstripdet,gluedDet,stripstereo,trackdirection);
  
  
  double m10=-(projectedstripstereo.second.y()-projectedstripstereo.first.y()); 
  double m11=(projectedstripstereo.second.x()-projectedstripstereo.first.x());
  
  //perform the matching
  //(x2-x1)(y-y1)=(y2-y1)(x-x1)
  AlgebraicMatrix22 m; AlgebraicVector2 c;
  m(0,0)=m00; 
  m(0,1)=m01;
  m(1,0)=m10;
  m(1,1)=m11;
  c(0)=c0;
  c(1)=m11*projectedstripstereo.first.y()+m10*projectedstripstereo.first.x();
  m.Invert(); 
  AlgebraicVector2 solution = m * c;
  Local2DPoint position(solution(0),solution(1));
  

  if ((!force) &&  (!((gluedDet->surface()).bounds().inside(position,10.f*scale_))) ) return nullptr;                                                       
  

  double c2 = -m10;
  double s2 = -m11;
  double l2 = 1./(c2*c2+s2*s2);
  
  
  float sigmap22 = sigmaPitch(stereoRH->localPosition(),stereoRH->localPositionError(),partnertopol);
  // auto sigmap22 = stereoRH->sigmaPitch();
  // assert (sigmap22>0);

  double diff=(c1*s2-c2*s1);
  double invdet2=1/(diff*diff*l1*l2);
  float xx= invdet2*(sigmap12*s2*s2*l2+sigmap22*s1*s1*l1);
  float xy=-invdet2*(sigmap12*c2*s2*l2+sigmap22*c1*s1*l1);
  float yy= invdet2*(sigmap12*c2*c2*l2+sigmap22*c1*c1*l1);
  LocalError error(xx,xy,yy);
 

  //if it is inside the gluedet bonds
  //Change NSigmaInside in the configuration file to accept more hits
  if(force || (gluedDet->surface()).bounds().inside(position,error,scale_)) 
    return new SiStripMatchedRecHit2D(LocalPoint(position), error, *gluedDet, monoRH,stereoRH);
  return nullptr;
}

