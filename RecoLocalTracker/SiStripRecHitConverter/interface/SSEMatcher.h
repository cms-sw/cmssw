#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "TrackingTools/TransientTrackingRecHit/interface/HelpertRecHit2DLocalPos.h"

namespace matcherDetails {  
  
  struct StereoInfo {
    mathSSE::Vec2D c1vec;
    const SiStripRecHit2D * secondHit;
    double sigmap22;
    double m10, m11;
  };
  
}

template<typename MonoIterator, typename StereoIterator,  typename CollectorHelper>
void SiStripRecHitMatcher::doubleMatch(MonoIterator monoRHiter, MonoIterator monoRHend,
				       StereoIterator seconditer, StereoIterator seconditerend,
				       const GluedGeomDet* gluedDet,	LocalVector trdir, 
				       CollectorHelper & collectorHelper) const{
  
  using  matcherDetails::StereoInfo;  
  using  mathSSE::Vec3F;
  using  mathSSE::Vec2D;
  using  mathSSE::Vec3D;
  using  mathSSE::Rot3F;
  typedef  GloballyPositioned<float> ToGlobal;
  typedef  typename GloballyPositioned<float>::ToLocal ToLocal;
  
  // hits in both mono and stero
  // match
  bool notk = trdir.mag2()<FLT_MIN;
  // FIXME we shall find a faster approximation for trdir: not useful to compute it each time for each strip
  
  // stripdet = mono
  // partnerstripdet = stereo
  const GeomDetUnit* stripdet = gluedDet->monoDet();
  const GeomDetUnit* partnerstripdet = gluedDet->stereoDet();
  const StripTopology& topol=(const StripTopology&)stripdet->topology();
  const StripTopology& partnertopol=(const StripTopology&)partnerstripdet->topology();
  
  // toGlobal is fast,  toLocal is slow
  ToGlobal const & stripDetTrans =  stripdet->surface();
  ToGlobal const & partnerStripDetTrans = partnerstripdet->surface();
  ToLocal          gluedDetInvTrans(gluedDet->surface());
  
  
  
  StereoInfo cache[std::distance(seconditer,seconditerend)];
  //iterate on stereo rechits
  // fill cache with relevant info
  int  cacheSize=0;
  for (;seconditer!=seconditerend; ++seconditer){
    
    const SiStripRecHit2D & secondHit = CollectorHelper::stereoHit(seconditer);
    
    double sigmap22 =secondHit.sigmaPitch();
    if (sigmap22<0) {
      LocalError tmpError( secondHit.localPositionErrorFast());
      HelpertRecHit2DLocalPos::updateWithAPE(tmpError, *partnerstripdet);
      MeasurementError errorstereoRH=partnertopol.measurementError(secondHit.localPositionFast(),tmpError);
      
      double pitch=partnertopol.localPitch(secondHit.localPositionFast());
      secondHit.setSigmaPitch(sigmap22=errorstereoRH.uu()*pitch*pitch);
    }
    
    
    double STEREOpointX=partnertopol.measurementPosition( secondHit.localPositionFast()).x();
    MeasurementPoint STEREOpointini(STEREOpointX,-0.5);
    MeasurementPoint STEREOpointend(STEREOpointX,0.5);
    
    LocalPoint locp1 = partnertopol.localPosition(STEREOpointini);
    LocalPoint locp2 = partnertopol.localPosition(STEREOpointend);
    
    GlobalPoint globalpointini=partnerStripDetTrans.toGlobal(locp1);
    GlobalPoint globalpointend=partnerStripDetTrans.toGlobal(locp2);
    
    // position of the initial and final point of the strip in glued local coordinates
    LocalPoint positiononGluedini=gluedDetInvTrans.toLocal(globalpointini);
    LocalPoint positiononGluedend=gluedDetInvTrans.toLocal(globalpointend); 
    
   // in case of no track hypothesis assume a track from the origin through the center of the strip
    if(notk){
      LocalPoint lcenterofstrip=secondHit.localPositionFast();
      GlobalPoint gcenterofstrip= partnerStripDetTrans.toGlobal(lcenterofstrip);
      GlobalVector gtrackdirection=gcenterofstrip-GlobalPoint(0,0,0);
      trdir=gluedDetInvTrans.toLocal(gtrackdirection);
    }
  

    Vec3F offset = trdir.basicVector().v * positiononGluedini.basicVector().v.get1<2>()/trdir.basicVector().v.get1<2>();
    
    
    Vec3F ret1 = positiononGluedini.basicVector().v - offset;
    Vec3F ret2 = positiononGluedend.basicVector().v - offset;
    
    double m10=-(ret2.arr[1] - ret1.arr[1]); 
    double m11=  ret2.arr[0] - ret1.arr[0];
    
    Vec2D c1vec; c1vec.set1(m11*ret1.arr[1] + m10 * ret1.arr[0]);
    
    // store
    StereoInfo info = {c1vec,&secondHit,sigmap22,m10,m11};
    cache[cacheSize++] = info;
  }
  
  
  
  for (;monoRHiter != monoRHend; ++monoRHiter) {
    
    SiStripRecHit2D const & monoRH = CollectorHelper::monoHit(monoRHiter);
    
    // position of the initial and final point of the strip (RPHI cluster) in local strip coordinates
    double RPHIpointX = topol.measurementPosition(monoRH.localPositionFast()).x();
    MeasurementPoint RPHIpointini(RPHIpointX,-0.5);
    MeasurementPoint RPHIpointend(RPHIpointX,0.5);
    
    // position of the initial and final point of the strip in local coordinates (mono det)
    //StripPosition stripmono=StripPosition(topol.localPosition(RPHIpointini),topol.localPosition(RPHIpointend));
    LocalPoint locp1o = topol.localPosition(RPHIpointini);
    LocalPoint locp2o = topol.localPosition(RPHIpointend);
    
    
    // in case of no track hypothesis assume a track from the origin through the center of the strip
    if(notk){
      LocalPoint lcenterofstrip=monoRH.localPositionFast();
      GlobalPoint gcenterofstrip= stripDetTrans.toGlobal(lcenterofstrip);
      GlobalVector gtrackdirection=gcenterofstrip-GlobalPoint(0,0,0);
      trdir=gluedDetInvTrans.toLocal(gtrackdirection);
    }
    
    
    //project mono hit on glued det
    //StripPosition projectedstripmono=project(stripdet,gluedDet,stripmono,trackdirection);
    
    
    GlobalPoint globalpointini=stripDetTrans.toGlobal(locp1o);
    GlobalPoint globalpointend=stripDetTrans.toGlobal(locp2o);
    
    // position of the initial and final point of the strip in glued local coordinates
    LocalPoint positiononGluedini=gluedDetInvTrans.toLocal(globalpointini);
    LocalPoint positiononGluedend=gluedDetInvTrans.toLocal(globalpointend); 
    
    Vec3F offset = trdir.basicVector().v * positiononGluedini.basicVector().v.get1<2>()/trdir.basicVector().v.get1<2>();
    
    
    Vec3F projini= positiononGluedini.basicVector().v - offset;
    Vec3F projend = positiononGluedend.basicVector().v -offset;
    
    // ret1o = ret1o + (trdir * (ret1o.getSimd(2) / trdirz));
    // ret2o = ret2o + (trdir * (ret2o.getSimd(2) / trdirz));
    
    double m00 = -(projend.arr[1] - projini.arr[1]);//-(projectedstripmono.second.y()-projectedstripmono.first.y()); 
    double m01 =  (projend.arr[0] - projini.arr[0]); // (projectedstripmono.second.x()-projectedstripmono.first.x());
    double c0  =  m01*projini.arr[1] + m00*projini.arr[0];//m01*projectedstripmono.first.y()   + m00*projectedstripmono.first.x();
    
    Vec2D c0vec(c0,c0);
    Vec2D minv00(-m01, m00);
    
    //error calculation (the part that depends on mono RH only)
    double c1 = -m00;
    double s1 = -m01;
    double l1 = 1./(c1*c1+s1*s1);
    
    // FIXME: here for test...
    double sigmap12 = monoRH.sigmaPitch();
    if (sigmap12<0) {
      
      LocalError tmpError(monoRH.localPositionErrorFast());
      HelpertRecHit2DLocalPos::updateWithAPE(tmpError,*stripdet);
      MeasurementError errormonoRH=topol.measurementError(monoRH.localPositionFast(),tmpError);
      
      double pitch=topol.localPitch(monoRH.localPositionFast());
      monoRH.setSigmaPitch(sigmap12=errormonoRH.uu()*pitch*pitch);
    }

    //float code
    float fc1(c1), fs1(s1);
    Vec3F scc1(fs1, fc1, fc1, 0.f);
    Vec3F ssc1(fs1, fs1, fc1, 0.f);
    Vec3F l1vec; l1vec.set1(l1);
    const Vec3F cslsimd = scc1 * ssc1 * l1vec;
    Vec3F sigmap12simd; sigmap12simd.set1(sigmap12);
    
    for (int i=0; i!=cacheSize; ++i) {
      StereoInfo const si = cache[i];
      
      // match 
      Vec2D minv10(si.m11, -si.m10);
      Vec2D mult; mult.set1(1./(m00*si.m11 - m01*si.m10));
      Vec2D resultmatmul = mult * (minv10 * c0vec + minv00 * si.c1vec);
      
      Local2DPoint position(resultmatmul.arr[0], resultmatmul.arr[1]);
      
      // LocalError tempError (100,0,100);
      if (!((gluedDet->surface()).bounds().inside(position,10.f*scale_))) continue;                                                       
      
      double c2 = -si.m10;
      double s2 = -si.m11;
      double l2 = 1./(c2*c2+s2*s2);
      
      double diff=(c1*s2-c2*s1);
      double invdet2 = 1./(diff*diff*l1*l2);
      
      float fc2(c2), fs2(s2), fid2(invdet2);	
      Vec3F invdet2simd(fid2, -fid2, fid2, 0.f);
      Vec3F ccssimd(fs2, fc2, fc2, 0.f);
      Vec3F csssimd(fs2, fs2, fc2, 0.f);
      Vec3F l2simd; l2simd.set1(l2);
      Vec3F sigmap22simd; sigmap22simd.set1(si.sigmap22);
      Vec3F result = invdet2simd * (sigmap22simd * cslsimd + sigmap12simd * ccssimd * csssimd * l2simd);
      
      
      LocalError error(result.arr[0], result.arr[1], result.arr[2]);
      
      
      if((gluedDet->surface()).bounds().inside(position,error,scale_)){ //if it is inside the gluedet bonds
	
	//Change NSigmaInside in the configuration file to accept more hits
	//...and add it to the Rechit collection 
	
	collectorHelper.collector()(SiStripMatchedRecHit2D(LocalPoint(position), error,gluedDet->geographicalId() ,
							   &monoRH,si.secondHit));
      }
      
    } // loop on cache info
    
    collectorHelper.closure(monoRHiter);
  } // loop on mono hit
  
}
