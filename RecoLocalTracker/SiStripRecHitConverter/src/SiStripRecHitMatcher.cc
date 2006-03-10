#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "Geometry/Vector/interface/GlobalPoint.h"

edm::OwnVector<SiStripRecHit2DMatchedLocalPos> SiStripRecHitMatcher::match(const  SiStripRecHit2DLocalPos *monoRH,RecHitIterator &begin, RecHitIterator &end, const DetId &detId, const StripTopology &topol,const GeomDetUnit* stripdet,const GeomDetUnit * partnerstripdet, LocalVector trackdirection){
  // stripdet = mono
  // partnerstripdet = stereo
  edm::OwnVector<SiStripRecHit2DMatchedLocalPos> collector;
  LocalPoint position;    
  // position of the initial and final point of the strip (RPHI cluster)
  MeasurementPoint RPHIpoint=topol.measurementPosition(monoRH->localPosition());
  MeasurementPoint RPHIpointini=MeasurementPoint(RPHIpoint.x(),-0.5);
  MeasurementPoint RPHIpointend=MeasurementPoint(RPHIpoint.x(),0.5);
  // position of the initial and final point of the strip in local coordinates (RPHI cluster)
  LocalPoint RPHIpositionini=topol.localPosition(RPHIpointini); 
  LocalPoint RPHIpositionend=topol.localPosition(RPHIpointend); 
  //cout<<"LocalPosition of monohit on monodet INI: "<<RPHIpositionini.x()<<" "<<RPHIpositionini.y()<<endl;
  //cout<<"LocalPosition of monohit on monodet END: "<<RPHIpositionend.x()<<" "<<RPHIpositionend.y()<<endl;
  if(trackdirection.mag2()<FLT_MIN){// in case of no track hypothesis assume a track from the origin through the center of the strip
    LocalPoint lcenterofstrip=monoRH->localPosition();
    GlobalPoint gcenterofstrip=(stripdet->surface()).toGlobal(lcenterofstrip);
    GlobalVector gtrackdirection=gcenterofstrip-GlobalPoint(0,0,0);
    trackdirection=(stripdet->surface()).toLocal(gtrackdirection);
  }
    //compute the distance of two detectors
  const StripTopology& partnertopol=(const StripTopology&)partnerstripdet->topology();
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
  double c1=fabs(sin(RPHIpositiononStereoendvector.phi())); double s1=fabs(cos(RPHIpositiononStereoendvector.phi()));
  MeasurementError errormonoRH=topol.measurementError(monoRH->localPosition(),monoRH->localPositionError());
    double sigmap12=errormonoRH.uu()*pow(topol.localPitch(monoRH->localPosition()),2);
    RecHitIterator seconditer;  
    for(seconditer=begin;seconditer!=end;++seconditer){
      // position of the initial and final point of the strip (STEREO cluster)
      MeasurementPoint STEREOpoint=partnertopol.measurementPosition(seconditer->localPosition());
      MeasurementPoint STEREOpointini=MeasurementPoint(STEREOpoint.x(),-0.5);
      MeasurementPoint STEREOpointend=MeasurementPoint(STEREOpoint.x(),0.5);
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
	double c2=cos(partnertopol.stripAngle(STEREOpoint.x())); double s2=sin(partnertopol.stripAngle(STEREOpoint.x()));
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
}
