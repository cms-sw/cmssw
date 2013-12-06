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




SiStripRecHitMatcher::SiStripRecHitMatcher(const edm::ParameterSet& conf){   
  scale_=conf.getParameter<double>("NSigmaInside");  
}

SiStripRecHitMatcher::SiStripRecHitMatcher(const double theScale){   
  scale_=theScale;  
}


//match a single hit
SiStripMatchedRecHit2D * SiStripRecHitMatcher::match(const SiStripRecHit2D *monoRH, 
						     const SiStripRecHit2D *stereoRH,
						     const GluedGeomDet* gluedDet,
						     LocalVector trackdirection) const{
  SimpleHitCollection stereoHits(1,stereoRH);
  std::vector<SiStripMatchedRecHit2D*>  collection;
  match( monoRH,
	 stereoHits.begin(), stereoHits.end(),
	 collection,
	 gluedDet,trackdirection);
  
  return collection.empty() ? (SiStripMatchedRecHit2D*)(0) : collection.front();
}

//repeat matching for an already  a single hit
SiStripMatchedRecHit2D* SiStripRecHitMatcher::match(const SiStripMatchedRecHit2D *origRH, 
						    const GluedGeomDet* gluedDet,
						    LocalVector trackdirection) const{
  
  throw "SiStripRecHitMatcher::match(const SiStripMatchedRecHit2D *,..) is obsoltete since 5.2.0"; 

  /*
  const SiStripRecHit2D* theMonoRH   = origRH->monoHit();
  // const SiStripRecHit2D* theStereoRH = origRH->stereoHit();
  SimpleHitCollection theStereoHits(1, origRH->stereoHit());
  // theStereoHits.push_back(theStereoRH);
  
  std::vector<SiStripMatchedRecHit2D*>  collection;
  match( theMonoRH,
	 theStereoHits.begin(), theStereoHits.end(),
	 collection,
	 gluedDet,trackdirection);
  
  return collection.empty() ? (SiStripMatchedRecHit2D*)(0) : collection.front();
  */

  return nullptr;
}


edm::OwnVector<SiStripMatchedRecHit2D> 
SiStripRecHitMatcher::match( const  SiStripRecHit2D *monoRH,
			     RecHitIterator begin, RecHitIterator end, 
			     const GluedGeomDet* gluedDet,
			     LocalVector trackdirection) const
{
  SimpleHitCollection stereoHits;
  stereoHits.reserve(end-begin);

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
			     LocalVector trackdirection) const {
  edm::OwnVector<SiStripMatchedRecHit2D> collector;
  collector.reserve(end-begin); // a resonable estimate of its size... 
  match(monoRH,begin,end,collector,gluedDet,trackdirection);
  return collector;
}


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

namespace {
  // FIXME for c++0X
  inline void pb1(std::vector<SiStripMatchedRecHit2D*> & v, SiStripMatchedRecHit2D* h) {
    v.push_back(h);
  }
  inline void pb2(SiStripRecHitMatcher::CollectorMatched & v, const SiStripMatchedRecHit2D & h) {
    v.push_back(h);
  }

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

void
SiStripRecHitMatcher::match( const SiStripRecHit2D *monoRH,
			     SimpleHitIterator begin, SimpleHitIterator end,
			     CollectorMatched & collector,
			     const GluedGeomDet* gluedDet,
			     LocalVector trackdirection) const {

  Collector result(boost::bind(pb2,boost::ref(collector),_1));
  match(monoRH,begin,end,result,gluedDet,trackdirection);
  
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

 
  // FIXME: here for test...
  double sigmap12 = monoRH->sigmaPitch();
  if (sigmap12<0) {
    //AlgebraicSymMatrix tmpMatrix = monoRH->parametersError();
    /*
    std::cout << "DEBUG START" << std::endl;
    std::cout << "APE mono,stereo,glued : " 
	      << stripdet->alignmentPositionError()->globalError().cxx()  << " , "
	      << partnerstripdet->alignmentPositionError()->globalError().cxx()  << " , "
	      << gluedDet->alignmentPositionError()->globalError().cxx()  << std::endl;
    */
    LocalError tmpError(monoRH->localPositionErrorFast());
    HelpertRecHit2DLocalPos::updateWithAPE(tmpError,*stripdet);
    MeasurementError errormonoRH=topol.measurementError(monoRH->localPositionFast(),tmpError);
    /*
    std::cout << "localPosError.xx(), helper.xx(), param.xx(): "
	 << monoRH->localPositionError().xx() << " , "
	 << monoRH->parametersError()[0][0] << " , "
	 << tmpMatrix[0][0] << std::endl;
    */
    //MeasurementError errormonoRH=topol.measurementError(monoRH->localPosition(),monoRH->localPositionError());
    double pitch=topol.localPitch(monoRH->localPositionFast());
    monoRH->setSigmaPitch(sigmap12=errormonoRH.uu()*pitch*pitch);
  }

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


    // FIXME: here for test...
    double sigmap22 = (*seconditer)->sigmaPitch();
    if (sigmap22<0) {
      //AlgebraicSymMatrix tmpMatrix = (*seconditer)->parametersError();
      LocalError tmpError((*seconditer)->localPositionErrorFast());
      HelpertRecHit2DLocalPos::updateWithAPE(tmpError, *partnerstripdet);
      MeasurementError errorstereoRH=partnertopol.measurementError((*seconditer)->localPositionFast(),tmpError);
      //MeasurementError errorstereoRH=partnertopol.measurementError((*seconditer)->localPosition(),(*seconditer)->localPositionError());
      double pitch=partnertopol.localPitch((*seconditer)->localPositionFast());
      (*seconditer)->setSigmaPitch(sigmap22=errorstereoRH.uu()*pitch*pitch);
    }

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
      collector(SiStripMatchedRecHit2D(position, error,gluedDet->geographicalId() ,
				       monoRH,secondHit));
    }
  }
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
