#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

  SiStripRecHitMatcher::SiStripRecHitMatcher(const edm::ParameterSet& conf){   
    scale_=conf.getParameter<double>("NSigmaInside");  
  };

const SiStripRecHit2DMatchedLocalPos& SiStripRecHitMatcher::match(const SiStripRecHit2DLocalPos *monoRH, 
					    const SiStripRecHit2DLocalPos *stereoRH,
					    const GluedGeomDet* gluedDet,
					    LocalVector trackdirection){
  SimpleHitCollection stereoHits;
  stereoHits.push_back(stereoRH);
  //const StripGeomDetUnit* monoDet = dynamic_cast< const StripGeomDetUnit*>(gluedDet->monoDet());
  //  const GeomDetUnit* stereoDet = gluedDet->stereoDet();
  edm::OwnVector<SiStripRecHit2DMatchedLocalPos> collection;
  collection= match( monoRH,
		     stereoHits.begin(), stereoHits.end(), 
		     gluedDet,trackdirection);
  return *collection.begin();
}

//edm::OwnVector<SiStripRecHit2DMatchedLocalPos> 
//SiStripRecHitMatcher::match( const SiStripRecHit2DLocalPos *monoRH, 
//			     SimpleHitIterator begin, SimpleHitIterator end,
//			     const GluedGeomDet* gluedDet,
//			     LocalVector trackdirection) 
//{
//  //  const StripGeomDetUnit* monoDet = dynamic_cast< const StripGeomDetUnit*>(gluedDet->monoDet());
//  //const GeomDetUnit* stereoDet = gluedDet->stereoDet();
//
//  return match( monoRH, begin, end,
//		gluedDet, trackdirection);
//}



edm::OwnVector<SiStripRecHit2DMatchedLocalPos> 
SiStripRecHitMatcher::match( const  SiStripRecHit2DLocalPos *monoRH,
			     RecHitIterator &begin, RecHitIterator &end, 
			     const GluedGeomDet* gluedDet,
			     LocalVector trackdirection)
{
  SimpleHitCollection stereoHits;
  for (RecHitIterator i=begin; i != end; ++i) {
    stereoHits.push_back( &(*i)); // convert to simple pointer
  }
  return match( monoRH,
		stereoHits.begin(), stereoHits.end(), 
		gluedDet,trackdirection);
}

edm::OwnVector<SiStripRecHit2DMatchedLocalPos> 
SiStripRecHitMatcher::match( const  SiStripRecHit2DLocalPos *monoRH,
			     SimpleHitIterator begin, SimpleHitIterator end,
			     const GluedGeomDet* gluedDet,
			     LocalVector trackdirection)
{
  // stripdet = mono
  // partnerstripdet = stereo
  const GeomDetUnit* stripdet = gluedDet->monoDet();
  const GeomDetUnit* partnerstripdet = gluedDet->stereoDet();
  const StripTopology& topol=(const StripTopology&)stripdet->topology();
  edm::OwnVector<SiStripRecHit2DMatchedLocalPos> collector;
  LocalPoint position;    
  // position of the initial and final point of the strip (RPHI cluster)
  MeasurementPoint RPHIpoint=topol.measurementPosition(monoRH->localPosition());
  MeasurementPoint RPHIpointini=MeasurementPoint(RPHIpoint.x(),-0.5);
  MeasurementPoint RPHIpointend=MeasurementPoint(RPHIpoint.x(),0.5);
  // position of the initial and final point of the strip in local coordinates (RPHI cluster)
  //LocalPoint RPHIpositionini=topol.localPosition(RPHIpointini); 
  //LocalPoint RPHIpositionend=topol.localPosition(RPHIpointend); 
  StripPosition stripmono=StripPosition(topol.localPosition(RPHIpointini),topol.localPosition(RPHIpointend));
  //std::cout<<"LocalPosition of monohit on monodet INI: "<<stripmono.first.x()<<" "<<stripmono.first.y()<<std::endl;
  //std::cout<<"LocalPosition of monohit on monodet END: "<<stripmono.second.x()<<" "<<stripmono.second.y()<<std::endl;
  if(trackdirection.mag2()<FLT_MIN){// in case of no track hypothesis assume a track from the origin through the center of the strip
    LocalPoint lcenterofstrip=monoRH->localPosition();
    GlobalPoint gcenterofstrip=(stripdet->surface()).toGlobal(lcenterofstrip);
    GlobalVector gtrackdirection=gcenterofstrip-GlobalPoint(0,0,0);
    trackdirection=(gluedDet->surface()).toLocal(gtrackdirection);
  }
  StripPosition projectedstripmono=project(stripdet,gluedDet,stripmono,trackdirection);
  const StripTopology& partnertopol=(const StripTopology&)partnerstripdet->topology();
  LocalVector  RPHIpositiononGluedendvector=projectedstripmono.second-projectedstripmono.first;
  double c1=sin(RPHIpositiononGluedendvector.phi()); double s1=cos(RPHIpositiononGluedendvector.phi());
  MeasurementError errormonoRH=topol.measurementError(monoRH->localPosition(),monoRH->localPositionError());
  double sigmap12=errormonoRH.uu()*pow(topol.localPitch(monoRH->localPosition()),2);
  SimpleHitIterator seconditer;  
  for(seconditer=begin;seconditer!=end;++seconditer){
    // position of the initial and final point of the strip (STEREO cluster)
    MeasurementPoint STEREOpoint=partnertopol.measurementPosition((*seconditer)->localPosition());
    MeasurementPoint STEREOpointini=MeasurementPoint(STEREOpoint.x(),-0.5);
    MeasurementPoint STEREOpointend=MeasurementPoint(STEREOpoint.x(),0.5);
    StripPosition stripstereo(partnertopol.localPosition(STEREOpointini),partnertopol.localPosition(STEREOpointend));
    //std::cout<<"LocalPosition of stereohit on stereodet INI: "<<stripstereo.first.x()<<" "<<stripstereo.first.y()<<std::endl;
    //std::cout<<"LocalPosition of stereohit on stereodet END: "<<stripstereo.second.x()<<" "<<stripstereo.second.y()<<std::endl;
    StripPosition projectedstripstereo=project(partnerstripdet,gluedDet,stripstereo,trackdirection);
    //(x2-x1)(y-y1)=(y2-y1)(x-x1)
    AlgebraicMatrix m(2,2); AlgebraicVector c(2), solution(2);
    m(1,1)=-(projectedstripmono.second.y()-projectedstripmono.first.y()); m(1,2)=(projectedstripmono.second.x()-projectedstripmono.first.x());
    m(2,1)=-(projectedstripstereo.second.y()-projectedstripstereo.first.y()); m(2,2)=(projectedstripstereo.second.x()-projectedstripstereo.first.x());
    c(1)=m(1,2)*projectedstripmono.first.y()+m(1,1)*projectedstripmono.first.x();
    c(2)=m(2,2)*projectedstripstereo.first.y()+m(2,1)*projectedstripstereo.first.x();
    solution=solve(m,c);
    //std::cout<<"LocalPosition of matched on stereodet: "<<solution(1)<<" "<<solution(2)<<std::endl;
    position=LocalPoint(solution(1),solution(2));
    // then calculate the error
    
    LocalVector  stereopositiononGluedendvector=projectedstripstereo.second-projectedstripstereo.first;
    double c2=sin(stereopositiononGluedendvector.phi()); double s2=cos(stereopositiononGluedendvector.phi());
    MeasurementError errorstereoRH=partnertopol.measurementError((*seconditer)->localPosition(),(*seconditer)->localPositionError());
    double sigmap22=errorstereoRH.uu()*pow(partnertopol.localPitch((*seconditer)->localPosition()),2);
    double invdet2=1/pow((c1*s2-c2*s1),2);
    float xx=invdet2*(sigmap12*s2*s2+sigmap22*s1*s1);
    float xy=-invdet2*(sigmap12*c2*s2+sigmap22*c1*s1);
    float yy=invdet2*(sigmap12*c2*c2+sigmap22*c1*c1);
    //std::cout<<"sp1="<<sigmap12<<" sp2="<<sigmap12<<std::endl;
    //std::cout<<"c1="<<c1<<" s1="<<s1<<std::endl;
    //std::cout<<"c2="<<c2<<" s2="<<s2<<std::endl;
    //std::cout<<"xx= "<<xx<<" xy="<<xy<<" yy="<<yy<<std::endl;
    LocalError error=LocalError(xx,xy,yy);
    if((gluedDet->surface()).bounds().inside(position,error,scale_)){
      //...and add it to the Rechit collection 
      //	SiStripRecHit2DLocalPos secondcluster=*seconditer;
      const SiStripRecHit2DLocalPos* secondHit = *seconditer;
      collector.push_back(new SiStripRecHit2DMatchedLocalPos(position, error,gluedDet->geographicalId() ,
							     monoRH,secondHit));
    }
  }
  return collector;
}


SiStripRecHitMatcher::StripPosition SiStripRecHitMatcher::project(const GeomDetUnit *det,const GluedGeomDet* glueddet,StripPosition strip,LocalVector trackdirection)
{

  GlobalPoint globalpointini=(det->surface()).toGlobal(strip.first);
  GlobalPoint globalpointend=(det->surface()).toGlobal(strip.second);

  // position of the initial and final point of the strip in stereo local coordinates (RPHI cluster)
  LocalPoint positiononGluedini=(glueddet->surface()).toLocal(globalpointini);
  LocalPoint positiononGluedend=(glueddet->surface()).toLocal(globalpointend);

  float scale=-positiononGluedini.z()/trackdirection.z();

  LocalPoint projpositiononGluedini= positiononGluedini + scale*trackdirection;
  LocalPoint projpositiononGluedend= positiononGluedend + scale*trackdirection;

  return StripPosition(projpositiononGluedini,projpositiononGluedend);
}
