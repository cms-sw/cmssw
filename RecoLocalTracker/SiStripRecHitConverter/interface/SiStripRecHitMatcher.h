#ifndef RECOLOCALTRACKER_SISTRIPCLUSTERIZER_SISTRIPRECHITMATCH_H
#define RECOLOCALTRACKER_SISTRIPCLUSTERIZER_SISTRIPRECHITRMATCH_H
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPos.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPosCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DMatchedLocalPosCollection.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "Geometry/CommonDetAlgo/interface/MeasurementPoint.h"
#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"
#include "Geometry/Surface/interface/LocalError.h"
#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include <cfloat>
class SiStripCluster;

class SiStripRecHitMatcher {
public:
  
  typedef  SiStripRecHit2DLocalPosCollection::const_iterator RecHitIterator;
  SiStripRecHitMatcher(){};
  template<class T>  
    edm::OwnVector<SiStripRecHit2DMatchedLocalPos> match(const  SiStripRecHit2DLocalPos *monoRH,RecHitIterator &begin, RecHitIterator &end, const DetId &detId, const T &topol,const GeomDetUnit* stripdet,const GeomDetUnit * partnerstripdet){
    return match(monoRH,begin, end, detId, topol, stripdet,partnerstripdet,LocalVector(0.,0.,0.));
  }
  template<class T>  
    edm::OwnVector<SiStripRecHit2DMatchedLocalPos> match(const  SiStripRecHit2DLocalPos *monoRH,RecHitIterator &begin, RecHitIterator &end, const DetId &detId, const T &topol,const GeomDetUnit* stripdet,const GeomDetUnit * partnerstripdet, LocalVector trackdirection){
    // stripdet = mono
    // partnerstripdet = stereo
    edm::OwnVector<SiStripRecHit2DMatchedLocalPos> collector;
    LocalPoint position;    
    // position of the initial and final point of the strip (RPHI cluster)
    MeasurementPoint RPHIpointini=MeasurementPoint(monoRH->cluster().front()->barycenter(),-0.5);
    MeasurementPoint RPHIpointend=MeasurementPoint(monoRH->cluster().front()->barycenter(),0.5);
    // position of the initial and final point of the strip in local coordinates (RPHI cluster)
    LocalPoint RPHIpositionini=topol.localPosition(RPHIpointini); 
    LocalPoint RPHIpositionend=topol.localPosition(RPHIpointend); 
    //cout<<"LocalPosition of monohit on monodet INI: "<<RPHIpositionini.x()<<" "<<RPHIpositionini.y()<<endl;
    //cout<<"LocalPosition of monohit on monodet END: "<<RPHIpositionend.x()<<" "<<RPHIpositionend.y()<<endl;
    if(trackdirection.mag2()<FLT_MIN){// in case of no track hypothesis assume a track from the origin through the center of the strip
      LocalPoint lcenterofstrip=topol.localPosition(monoRH->cluster().front()->barycenter());
      GlobalPoint gcenterofstrip=(stripdet->surface()).toGlobal(lcenterofstrip);
      GlobalVector gtrackdirection=gcenterofstrip-GlobalPoint(0,0,0);
      trackdirection=(stripdet->surface()).toLocal(gtrackdirection);
    }
    //compute the distance of two detectors
    const T& partnertopol=(T&)partnerstripdet->topology();
    GlobalPoint gdetmono=(stripdet->surface()).toGlobal(LocalPoint(0,0,0));
    GlobalPoint gdetstereo=(partnerstripdet->surface()).toGlobal(LocalPoint(0,0,0));
    GlobalVector gdist=gdetstereo-gdetmono;
    //    std::cout<<"gdist= "<<gdist.mag()<<std::endl;
    LocalVector ldist=(stripdet->surface()).toLocal(gdist);
    //    std::cout<<"ldist= "<<ldist.x()<<" "<<ldist.y()<<" "<<ldist.z()<<std::endl;
    //std::cout<<"phi= "<<trackdirection.phi()*180/3.14<<" theta= "<<trackdirection.theta()*180/3.14<<std::endl;
    LocalVector shift=LocalVector(ldist.z()*tan(trackdirection.theta())*cos(trackdirection.phi()),ldist.z()*tan(trackdirection.theta())*sin(trackdirection.phi()),0);
    //std::cout<<"xshift= "<<shift.x()<<std::endl;
    //std::cout<<"yshift= "<<shift.y()<<std::endl;
    RPHIpositionini+=shift; RPHIpositionend+=shift;
    //cout<<"LocalPosition of monohit on monodet INI corrected: "<<RPHIpositionini.x()<<" "<<RPHIpositionini.y()<<endl;
    //cout<<"LocalPosition of monohit on monodet END corrected: "<<RPHIpositionend.x()<<" "<<RPHIpositionend.y()<<endl;
    // position of the initial and final point of the strip in global coordinates (RPHI cluster)
    GlobalPoint rphiglobalpointini=(stripdet->surface()).toGlobal(RPHIpositionini);
    GlobalPoint rphiglobalpointend=(stripdet->surface()).toGlobal(RPHIpositionend);
    // position of the initial and final point of the strip in stereo local coordinates (RPHI cluster)
    LocalPoint RPHIpositiononStereoini=(partnerstripdet->surface()).toLocal(rphiglobalpointini);
    LocalPoint RPHIpositiononStereoend=(partnerstripdet->surface()).toLocal(rphiglobalpointend);
    //cout<<"LocalPosition of monohit on stereodet INI: "<<RPHIpositiononStereoini.x()<<" "<<RPHIpositiononStereoini.y()<<endl;
    //cout<<"LocalPosition of monohit on stereodet END: "<<RPHIpositiononStereoend.x()<<" "<<RPHIpositiononStereoend.y()<<endl;
    //to calculate the error:
    LocalVector  RPHIpositiononStereoendvector=RPHIpositiononStereoend-LocalPoint(0.,0.,0.);
    double c1=abs(sin(RPHIpositiononStereoendvector.phi())); double s1=abs(cos(RPHIpositiononStereoendvector.phi()));
    MeasurementError errormonoRH=topol.measurementError(monoRH->localPosition(),monoRH->localPositionError());
    double sigmap12=errormonoRH.uu()*pow(topol.localPitch(monoRH->localPosition()),2);
    RecHitIterator seconditer;  
    for(seconditer=begin;seconditer!=end;++seconditer){
      // position of the initial and final point of the strip (STEREO cluster)
      MeasurementPoint STEREOpointini=MeasurementPoint(seconditer->cluster().front()->barycenter(),-0.5);
      MeasurementPoint STEREOpointend=MeasurementPoint(seconditer->cluster().front()->barycenter(),0.5);
      LocalPoint STEREOpositionini=partnertopol.localPosition(STEREOpointini); 
      LocalPoint STEREOpositionend=partnertopol.localPosition(STEREOpointend); 
      // cout<<"LocalPosition of stereohit on stereodet INI: "<<STEREOpositionini.x()<<" "<<STEREOpositionini.y()<<endl;
      //cout<<"LocalPosition of stereohit on stereodet END: "<<STEREOpositionend.x()<<" "<<STEREOpositionend.y()<<endl;
      //(x2-x1)(y-y1)=(y2-y1)(x-x1)
      AlgebraicMatrix m(2,2); AlgebraicVector c(2), solution(2);
      m(1,1)=-(RPHIpositiononStereoend.y()-RPHIpositiononStereoini.y()); m(1,2)=(RPHIpositiononStereoend.x()-RPHIpositiononStereoini.x());
      m(2,1)=-(STEREOpositionend.y()-STEREOpositionini.y());m(2,2)=(STEREOpositionend.x()-STEREOpositionini.x());
      c(1)=m(1,2)*RPHIpositiononStereoini.y()+m(1,1)*RPHIpositiononStereoini.x();
      c(2)=m(2,2)*STEREOpositionini.y()+m(2,1)*STEREOpositionini.x();
      solution=solve(m,c);
      //cout<<"LocalPosition of matched on stereodet: "<<solution(1)<<" "<<solution(2)<<endl;
      if(solution(2)>-(partnertopol.localStripLength(seconditer->localPosition())/2)&&solution(2)<partnertopol.localStripLength(seconditer->localPosition())/2){//(to be modified)
	position=LocalPoint(solution(1),solution(2));
	// then calculate the error
	double c2=cos(partnertopol.stripAngle(seconditer->cluster().front()->barycenter())); double s2=sin(partnertopol.stripAngle(seconditer->cluster().front()->barycenter()));
	MeasurementError errorstereoRH=partnertopol.measurementError(seconditer->localPosition(),seconditer->localPositionError());
	double sigmap22=errorstereoRH.uu()*pow(partnertopol.localPitch(seconditer->localPosition()),2);
	double invdet2=1/pow((c1*s2-c2*s1),2);
	float xx=invdet2*(sigmap12*s2*s2+sigmap22*s1*s1);
	float xy=-invdet2*(sigmap12+c2*s2+invdet2*sigmap22*c1*s1);
	float yy=invdet2*(sigmap12*c2*c2+sigmap22*c1*c1);
	LocalError error=LocalError(xx,xy,yy);
	//...and add it to the Rechit collection 
	//	SiStripRecHit2DLocalPos secondcluster=*seconditer;
	collector.push_back(new SiStripRecHit2DMatchedLocalPos(position, error,detId,monoRH,&(*seconditer)));
      }
    }
    return collector;
  };
};
#endif
