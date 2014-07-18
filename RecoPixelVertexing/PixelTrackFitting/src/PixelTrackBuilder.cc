#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackBuilder.h"


#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryError.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/BasicTrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"

#include <sstream>
using namespace std;
using namespace reco;


template <class T> T inline sqr( T t) {return t*t;}

namespace {
  std::string print(
		    const Measurement1D & pt,
		    const Measurement1D & phi,
		    const Measurement1D & cotTheta,
		    const Measurement1D & tip,
		    const Measurement1D & zip,
		    float chi2,
		    int   charge)
  {
    ostringstream str;
    str <<"\t pt: "  << pt.value() <<"+/-"<<pt.error()  
        <<"\t phi: " << phi.value() <<"+/-"<<phi.error()
        <<"\t cot: " << cotTheta.value() <<"+/-"<<cotTheta.error()
        <<"\t tip: " << tip.value() <<"+/-"<<tip.error()
        <<"\t zip: " << zip.value() <<"+/-"<<zip.error()
        <<"\t charge: " << charge;
    return str.str();
  }

  std::string print(const reco::Track & track, const GlobalPoint & origin)
  {
    
    math::XYZPoint bs(origin.x(), origin.y(), origin.z());
    
    Measurement1D phi( track.phi(), track.phiError());
    
    float theta = track.theta();
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    float errLambda2 = sqr( track.lambdaError() );
    Measurement1D cotTheta(cosTheta/sinTheta, sqrt(errLambda2)/sqr(sinTheta));
    
    float pt_v = track.pt();
    float errInvP2 = sqr(track.qoverpError());
    float covIPtTheta = track.covariance(TrackBase::i_qoverp, TrackBase::i_lambda);
    float errInvPt2 = (   errInvP2
			  + sqr(cosTheta/pt_v)*errLambda2
			  + 2*(cosTheta/pt_v)*covIPtTheta
			  ) / sqr(sinTheta);
    Measurement1D pt(pt_v, sqr(pt_v)*sqrt(errInvPt2));
    
    Measurement1D tip(track.dxy(bs), track.d0Error());
    
    Measurement1D zip(track.dz(bs), track.dzError());
    
    return print(pt, phi, cotTheta, tip, zip, track.chi2(),  track.charge());
  }
  
  std::string print(const  BasicTrajectoryStateOnSurface & state)
  {
    // TrajectoryStateOnSurface state(bstate);
    float pt_v = state.globalMomentum().perp();
    float phi_v = state.globalMomentum().phi();
    float theta_v = state.globalMomentum().theta();
    
    CurvilinearTrajectoryError curv = state.curvilinearError();
    float errPhi2 = curv.matrix()(3,3);
    float errLambda2 = curv.matrix()(2,2);
    float errInvP2 = curv.matrix()(1,1);
    float covIPtTheta = curv.matrix()(1,2);
    float cosTheta = cos(theta_v);
    float sinTheta = sin(theta_v);
    float errInvPt2 = (   errInvP2
			  + sqr(cosTheta/pt_v)*errLambda2
			  + 2*(cosTheta/pt_v)*covIPtTheta) / sqr(sinTheta);
    float errCotTheta = sqrt(errLambda2)/sqr(sinTheta) ;
    Measurement1D pt(pt_v, sqr(pt_v) * sqrt(errInvPt2));
    Measurement1D phi(phi_v, sqrt(errPhi2) );
    Measurement1D cotTheta(cosTheta/sinTheta, errCotTheta);
    
    float zip_v = state.globalPosition().z();
    float zip_e = sqrt( state.localError().matrix()(4,4));
    Measurement1D zip(zip_v, zip_e);
    
    float tip_v  = state.localPosition().x(); 
    int tip_sign = (state.localMomentum().y()*cotTheta.value() > 0) ? -1 : 1;
    float tip_e  = sqrt( state.localError().matrix()(3,3) );
    Measurement1D tip( tip_sign*tip_v, tip_e);
    
    return print(pt, phi, cotTheta, tip, zip, 0., state.charge());
  }


  inline void checkState(const  BasicTrajectoryStateOnSurface & bstate, const MagneticField* mf, const GlobalPoint & origin)
  {
    TrajectoryStateOnSurface state(bstate.clone());
    
    LogTrace("")<<" *** PixelTrackBuilder::checkState: ";
    LogTrace("")<<"INPUT,  ROTATION" << endl<<state.surface().rotation();
    LogTrace("")<<"INPUT,  TSOS:"<<endl<<state;
    
    TransverseImpactPointExtrapolator tipe(mf);
    TrajectoryStateOnSurface test= tipe.extrapolate(state, origin);
    LogTrace("")<<"CHECK-1 ROTATION" << endl<<"\n"<<test.surface().rotation();
    LogTrace("")<<"CHECK-1 TSOS" << endl<<test;
    
    TSCPBuilderNoMaterial tscpBuilder;
    TrajectoryStateClosestToPoint tscp =
      tscpBuilder(*(state.freeState()), origin);
    FreeTrajectoryState fs = tscp.theState();
    LogTrace("")<<"CHECK-2 FTS: " << fs;
  }

}


reco::Track * PixelTrackBuilder::build(
      const Measurement1D & pt,
      const Measurement1D & phi, 
      const Measurement1D & cotTheta,
      const Measurement1D & tip,  
      const Measurement1D & zip,
      float chi2,
      int   charge,
      const std::vector<const TrackingRecHit* >& hits,
      const MagneticField * mf,
      const GlobalPoint   & origin) const 
{

  LogDebug("PixelTrackBuilder::build");
  LogTrace("")<<"reconstructed TRIPLET kinematics:\n"<<print(pt,phi,cotTheta,tip,zip,chi2,charge);

  double sinTheta = 1/std::sqrt(1+sqr(cotTheta.value()));
  double cosTheta = cotTheta.value()*sinTheta;
  int tipSign = tip.value() > 0 ? 1 : -1;

  AlgebraicSymMatrix55 m;
  double invPtErr = 1./sqr(pt.value()) * pt.error();
  m(0,0) = sqr(sinTheta) * (
              sqr(invPtErr)
            + sqr(cotTheta.error()/pt.value()*cosTheta * sinTheta)
            );
  m(0,2) = sqr( cotTheta.error()) * cosTheta * sqr(sinTheta) / pt.value();
  m(1,1) = sqr( phi.error() );
  m(2,2) = sqr( cotTheta.error());
  m(3,3) = sqr( tip.error() );
  m(4,4) = sqr( zip.error() );
  LocalTrajectoryError error(m);

  LocalTrajectoryParameters lpar(
    LocalPoint(tipSign*tip.value(), -tipSign*zip.value(), 0),
    LocalVector(0., -tipSign*pt.value()*cotTheta.value(), pt.value()),
    charge);

  
  float sp = std::sin(phi.value());
  float cp = std::cos(phi.value());
  Surface::RotationType rot(
			    sp*tipSign, -cp*tipSign,           0,
			    0         ,           0,    -tipSign,
			    cp        ,  sp        ,           0);

  // BTSOS hold Surface in a shared pointer and  will be autodeleted when BTSOS goes out of scope...
  // to avoid memory churn we allocate it locally and just avoid it be deleted by refcount... 
  Plane impPointPlane(origin, rot);
  // (twice just to be sure!)
  impPointPlane.addReference(); impPointPlane.addReference();
  // use Base (to avoid a useless new) 
  BasicTrajectoryStateOnSurface impactPointState( lpar , error, impPointPlane, mf, 1.0);
  
  //checkState(impactPointState,mf);
  LogTrace("")<<"constructed TSOS :\n"<<print(impactPointState);

  int ndof = 2*hits.size()-5;
  GlobalPoint vv = impactPointState.globalPosition();
  math::XYZPoint  pos( vv.x(), vv.y(), vv.z() );
  GlobalVector pp = impactPointState.globalMomentum();
  math::XYZVector mom( pp.x(), pp.y(), pp.z() );

  reco::Track * track = new reco::Track( chi2, ndof, pos, mom, 
        impactPointState.charge(), impactPointState.curvilinearError());

  LogTrace("") <<"RECONSTRUCTED TRACK (0,0,0):\n"<< print(*track,GlobalPoint(0,0,0))<<std::endl;
  LogTrace("") <<"RECONSTRUCTED TRACK "<<origin<<"\n"<< print(*track,origin)<<std::endl;

  return track;
}

