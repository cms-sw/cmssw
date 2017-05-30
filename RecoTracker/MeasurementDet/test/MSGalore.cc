#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <RecoTracker/MeasurementDet/interface/MeasurementTracker.h>
#include <RecoTracker/Record/interface/CkfComponentsRecord.h>
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "MagneticField/VolumeGeometry/interface/MagVolumeOutsideValidity.h"
#include "DataFormats/GeometrySurface/interface/PlaneBuilder.h"


#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackPropagation/RungeKutta/interface/defaultRKPropagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"


namespace {
inline
Surface::RotationType rotation( const GlobalVector& zDir)
{
  GlobalVector zAxis = zDir.unit();
  GlobalVector yAxis( zAxis.y(), -zAxis.x(), 0); 
  GlobalVector xAxis = yAxis.cross( zAxis);
  return Surface::RotationType( xAxis, yAxis, zAxis);
}

}


struct MSData {
  int stid;
  int lid;
  float z;
  float uerr;
  float verr;
};
inline
std::ostream & operator<<(std::ostream & os, MSData d) {
  os <<  d.stid<<'>' <<d.lid <<'|'<<d.z<<':'<<d.uerr<<'/'<<d.verr;
  return os;
}

class MSGalore final : public edm::EDAnalyzer {
public:
  explicit MSGalore(const edm::ParameterSet&);


private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

  std::string theMeasurementTrackerName;
  std::string theNavigationSchoolName;
};


MSGalore::MSGalore(const edm::ParameterSet& iConfig): 
   theMeasurementTrackerName(iConfig.getParameter<std::string>("measurementTracker"))
  ,theNavigationSchoolName(iConfig.getParameter<std::string>("navigationSchool")){}


void MSGalore::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  //get the measurementtracker
  edm::ESHandle<MeasurementTracker> measurementTracker;
  edm::ESHandle<NavigationSchool>   navSchool;

  iSetup.get<CkfComponentsRecord>().get(theMeasurementTrackerName, measurementTracker);
  iSetup.get<NavigationSchoolRecord>().get(theNavigationSchoolName, navSchool);

  auto const & geom = *(TrackerGeometry const *)(*measurementTracker).geomTracker();
  auto const & searchGeom = *(*measurementTracker).geometricSearchTracker();
  auto const & dus = geom.detUnits();
  
  auto firstBarrel = geom.offsetDU(GeomDetEnumerators::tkDetEnum[1]);
  auto firstForward = geom.offsetDU(GeomDetEnumerators::tkDetEnum[2]);
 
  std::cout << "number of dets " << dus.size() << std::endl;
  std::cout << "Bl/Fw loc " << firstBarrel<< '/' << firstForward << std::endl;

  edm::ESHandle<MagneticField> magfield;
  iSetup.get<IdealMagneticFieldRecord>().get(magfield);

  edm::ESHandle<Propagator>             propagatorHandle;
  iSetup.get<TrackingComponentsRecord>().get("PropagatorWithMaterial", propagatorHandle);
  auto const & ANprop = *propagatorHandle;  

  // error (very very small)
  ROOT::Math::SMatrixIdentity id;
  AlgebraicSymMatrix55 C(id);
  C *= 1.e-16;

  CurvilinearTrajectoryError err(C);

  Chi2MeasurementEstimator estimator(30.,-3.0,0.5,2.0,0.5,1.e12);  // same as defauts....
  KFUpdator kfu;
  LocalError he(0.01*0.01,0,0.02*0.02);

  /*
  auto bsz = searchGeom.pixelBarrelLayers().size();
  auto fsz = searchGeom.posPixelForwardLayers().size();
  auto nl=bsz+fsz-2;
  DetLayer const * layers[nl];
  for (decltype(bsz) i=0;i<bsz-1;++i) layers[i]=searchGeom.pixelBarrelLayers()[i];
  for (decltype(fsz) i=0;i<fsz-1;++i) layers[i+bsz-1]=searchGeom.posPixelForwardLayers()[i];
  */

  for (int from=0; from<3; ++from) {
  std::cout << "from layer "<< from << std::endl; 
  for (float tl = 0.0f; tl<12.0f; tl+=0.1f) {

  float p = 1.0f;
  float phi = 0.1415f;
  float tanlmd = tl; // 0.1f;
  auto sinth2 = 1.f/(1.f+tanlmd*tanlmd);
  auto sinth = std::sqrt(sinth2);
  auto costh = tanlmd*sinth;

  // std::cout << tl << ' ' << sinth << ' ' << costh << std::endl;


  GlobalVector startingMomentum(p*std::sin(phi)*sinth,p*std::cos(phi)*sinth,p*costh);

  std::vector<MSData> mserr[3];

  float lastzz=-18;
  float lastbz=-18;
  bool goFw=false;
  std::string loc=" Barrel";
  for (int iz=0;iz<2; ++iz) {
  if (iz>0) goFw=true;
  for (float zz=lastzz; zz<18.1; zz+=0.1) {
  float z = zz;
  GlobalPoint startingPosition(0,0,z);

  // make TSOS happy
  //Define starting plane
  PlaneBuilder pb;
  auto rot = rotation(startingMomentum);
  auto startingPlane = pb.plane( startingPosition, rot);

  TrajectoryStateOnSurface startingStateP( GlobalTrajectoryParameters(startingPosition, 
	  				         startingMomentum, 1, magfield.product()), 
				                  err, *startingPlane);
  auto tsos = startingStateP;

  DetLayer const * layer = searchGeom.pixelBarrelLayers()[0];
  if (goFw) layer = searchGeom.posPixelForwardLayers()[0];
  int stid = layer->seqNum();
  /*
  {
    auto it = layer;
    std::cout << "first layer " << (it->isBarrel() ? " Barrel" : " Forward") << " layer " << it->seqNum() << " SubDet " << it->subDetector()<< std::endl;
  }
 */
  auto const & detWithState = layer->compatibleDets(tsos,ANprop,estimator);
  if(!detWithState.size()) {
    // std::cout << "no det on first layer" << layer->seqNum() << std::endl;
    continue;
  }
  tsos = detWithState.front().second;
  // std::cout << "arrived at " << int(detWithState.front().first->geographicalId()) << ' ' << tsos.globalPosition() << ' ' << tsos.localError().positionError() << std::endl;

  // for barrel
  float z1 = tsos.globalPosition().z();
  if (goFw) {
    loc = " Forward";
    z1 = tsos.globalPosition().perp();
  }
  for (int il=1; il<4;	++il) {

  auto const & compLayers = (*navSchool).nextLayers(*layer,*tsos.freeState(),alongMomentum);
  layer = nullptr;
  for(auto it : compLayers){
    if (it->basicComponents().empty()) {
	   //this should never happen. but better protect for it
	   std::cout <<"a detlayer with no components: I cannot figure out a DetId from this layer. please investigate." << std::endl;
	   continue;
     }
    //std::cout << il << (it->isBarrel() ? " Barrel" : " Forward") << " layer " << it->seqNum() << " SubDet " << it->subDetector()<< std::endl;
    auto const & detWithState = it->compatibleDets(tsos,ANprop,estimator);
    if(!detWithState.size()) { 
      // std::cout << "no det on this layer" << it->seqNum() << std::endl; 
      continue;
    }
    layer = it;
    //auto did = detWithState.front().first->geographicalId();
    //std::cout << "arrived at " << int(did) << std::endl;
    tsos = detWithState.front().second;
    // std::cout << tsos.globalPosition() << ' ' << tsos.localError().positionError() << std::endl;

    if (from==il) {
      // std::cout << tsos.globalPosition() << ' ' << tsos.globalDirection() << ' ' << tsos.localError().positionError() << std::endl;
      // constrain it to this location (relevant for layer other than very first)
      SiPixelRecHit::ClusterRef pref;
      SiPixelRecHit   hitpx(tsos.localPosition(),he,1.,*detWithState.front().first,pref);
      tsos = kfu.update(tsos, hitpx);
      // std::cout << tsos.globalPosition() << ' ' << tsos.globalDirection() << ' ' << tsos.localError().positionError() << std::endl;
      z1 = layer->isBarrel() ? tsos.globalPosition().z() : tsos.globalPosition().perp();
      stid = layer->seqNum();
    }

    //if (il>from) 
    {
      float xerr = std::sqrt(tsos.localError().matrix()(3,3));
      float zerr = std::sqrt(tsos.localError().matrix()(4,4));
      //  std::cout << tanlmd << ' ' << z1 << ' ' << it->seqNum() << ':' << xerr <<'/'<<zerr << std::endl;    
      if (mserr[il-1].empty()) mserr[il-1].emplace_back(MSData{stid,it->seqNum(),z1,xerr,zerr});
      else if ( stid!=mserr[il-1].back().stid ||  std::abs(xerr-mserr[il-1].back().uerr)>0.1f*xerr || std::abs(zerr-mserr[il-1].back().verr)>0.1f*zerr) mserr[il-1].emplace_back(MSData{stid,it->seqNum(),z1,xerr,zerr});
    }
    break;
   }
   if (!layer) break;
   if (!goFw) lastbz=z1;
   lastzz=zz;


  } // layer loop
  }} // loop on z
   if (mserr[0].empty()) continue;
   std::cout << "tl " << tanlmd << loc << ' ' <<from<< std::endl;
   for (auto il=0; il<3; ++il) { std::cout << il << ' ';
   for ( auto const & e : mserr[il]) std::cout << e << '-' <<e.uerr/sinth <<'/'<<e.verr/sinth <<' ';
   std::cout << std::endl;
  }
   std::cout << tanlmd << ' ' << lastbz << std::endl;
  } // loop  on tanLa
 } // loop on from
}

//define this as a plug-in
DEFINE_FWK_MODULE(MSGalore);
