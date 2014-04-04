#include "Math/VectorUtil.h"
#include <math.h>
#include "DQM/PhysicsHWW/interface/trackSelections.h"

namespace HWWFunctions {

  // return a pair of d0, d0err of a ctf track with respect to a primary vertex
  std::pair<double, double> trks_d0_pv (HWW& hww, int itrk, int ipv)
  {
      // assume the layout of the covariance matrix is (Vxx, Vxy, Vxz)
      //						      (Vyx, Vyy, ...)
      const double bx  = hww.vtxs_position().at(ipv).x()   ;
      const double by  = hww.vtxs_position().at(ipv).y()   ;
      const double vxx = hww.vtxs_covMatrix().at(ipv).at(0);
      const double vxy = hww.vtxs_covMatrix().at(ipv).at(1);
      const double vyy = hww.vtxs_covMatrix().at(ipv).at(4);

      const double phi      = hww.trks_trk_p4().at(itrk).phi();
      const double d0vtx    = hww.trks_d0().at(itrk) - bx * sin(phi) + by * cos(phi);
      const double d0err    = hww.trks_d0Err().at(itrk);
      const double phierr   = hww.trks_phiErr().at(itrk);
      const double d0phicov = hww.trks_d0phiCov().at(itrk);

      // we will let the optimizer take care of subexpression
      // elimination for this one...
      const double d0err2vtx = d0err * d0err 
          - 2 * (bx * cos(phi) + by * sin(phi)) * d0phicov
          + (bx * cos(phi) + by * sin(phi)) * (bx * cos(phi) + by * sin(phi)) * phierr * phierr
          + sin(phi) * sin(phi) * vxx + cos(phi) * cos(phi) * vyy
          - 2 * sin(phi) * cos(phi) * vxy;
      if (d0err2vtx >= 0) 
          return std::pair<double, double>(d0vtx, sqrt(d0err2vtx));

      edm::LogError("NegativeValue") << "Oh no!  sigma^2(d0corr) < 0!";
      return std::pair<double, double>(d0vtx, -sqrt(-d0err2vtx));
  }

  // return a pair of d0, d0err of a gsf track with respect to a primary vertex
  std::pair<double , double> gsftrks_d0_pv (HWW& hww, int itrk, int ipv)
  {
      // assume the layout of the covariance matrix is (Vxx, Vxy, Vxz)
      //						      (Vyx, Vyy, ...)
      const double bx  = hww.vtxs_position().at(ipv).x()   ;
      const double by  = hww.vtxs_position().at(ipv).y()   ;
      const double vxx = hww.vtxs_covMatrix().at(ipv).at(0);
      const double vxy = hww.vtxs_covMatrix().at(ipv).at(1);
      const double vyy = hww.vtxs_covMatrix().at(ipv).at(4);

      const double phi      = hww.gsftrks_p4().at(itrk).phi();
      const double d0vtx    = hww.gsftrks_d0().at(itrk) - bx * sin(phi) + by * cos(phi);
      const double d0err    = hww.gsftrks_d0Err().at(itrk);
      const double phierr   = hww.gsftrks_phiErr().at(itrk);
      const double d0phicov = hww.gsftrks_d0phiCov().at(itrk);

      // we will let the optimizer take care of subexpression
      // elimination for this one...
      const double d0err2vtx = d0err * d0err 
          - 2 * (bx * cos(phi) + by * sin(phi)) * d0phicov
          + (bx * cos(phi) + by * sin(phi)) * (bx * cos(phi) + by * sin(phi)) * phierr * phierr
          + sin(phi) * sin(phi) * vxx + cos(phi) * cos(phi) * vyy
          - 2 * sin(phi) * cos(phi) * vxy;
      if (d0err2vtx >= 0) 
          return std::pair<double, double>(d0vtx, sqrt(d0err2vtx));

      edm::LogError("NegativeValue") << "Oh no!  sigma^2(d0corr) < 0!";
      return std::pair<double, double>(d0vtx, -sqrt(-d0err2vtx));
  }

  // return a pair of dz, dzerr of a ctf track with respect to a primary vertex
  std::pair<double, double> trks_dz_pv (HWW& hww, int itrk, int ipv)
  {


      LorentzVector pv = hww.vtxs_position().at(ipv);
      double pvxErr    = hww.vtxs_xError().at(ipv)  ;
      double pvyErr    = hww.vtxs_yError().at(ipv)  ;
      double pvzErr    = hww.vtxs_zError().at(ipv)  ;
      double phi        = hww.trks_trk_p4().at(itrk).phi();
      double theta      = hww.trks_trk_p4().at(itrk).theta();
      double ddzdpvx    = cos(phi)*1./tan(theta);
      double ddzdpvy    = sin(phi)*1./tan(theta);
      double ddzdphi    = -1*pv.x()*sin(phi)*1./tan(theta) + pv.y()*cos(phi)*1./tan(theta);
      double ddzdtheta  = -1*1/sin(theta)*1/sin(theta) * (pv.x()*cos(phi) + pv.y()*sin(phi));

      ddzdpvx   *= ddzdpvx;
      ddzdpvy   *= ddzdpvy;
      ddzdphi   *= ddzdphi;
      ddzdtheta *= ddzdtheta;

      double z0Err    = hww.trks_z0Err().at(itrk);
      double phiErr   = hww.trks_phiErr().at(itrk);
      double thetaErr = hww.trks_etaErr().at(itrk)*sin(theta);

      z0Err    *= z0Err;
      phiErr   *= phiErr;
      thetaErr *= thetaErr;
      pvxErr   *= pvxErr;
      pvyErr   *= pvyErr;
      pvzErr   *= pvzErr;

      double value = hww.trks_z0().at(itrk) - pv.z() + (pv.x()*cos(phi) + pv.y()*sin(phi) )*1./tan(theta);

      //note that the error does not account for correlations since we do not store the track covariance matrix
      double error = sqrt(z0Err + pvzErr + ddzdpvx*pvxErr + ddzdpvy*pvyErr + ddzdphi*phiErr + ddzdtheta*thetaErr);

      return std::pair<double, double>(value, error);
  }

  std::pair<double, double> gsftrks_dz_pv (HWW& hww, int itrk, int ipv)
  {
      LorentzVector pv = hww.vtxs_position().at(ipv);
      double pvxErr    = hww.vtxs_xError().at(ipv)  ;
      double pvyErr    = hww.vtxs_yError().at(ipv)  ;
      double pvzErr    = hww.vtxs_zError().at(ipv)  ;

      //LorentzVector tkp = hww.gsftrks_p4().at(itrk);
      //LorentzVector tkv = hww.gsftrks_vertex_p4().at(itrk);

      double phi   = hww.gsftrks_p4().at(itrk).phi();
      double theta = hww.gsftrks_p4().at(itrk).theta();

      double ddzdpvx   = cos(phi)*1./tan(theta);
      double ddzdpvy   = sin(phi)*1./tan(theta);
      double ddzdphi   = -1*pv.x()*sin(phi)*1./tan(theta) + pv.y()*cos(phi)*1./tan(theta);
      double ddzdtheta = -1*1/sin(theta)*1/sin(theta) * (pv.x()*cos(phi) + pv.y()*sin(phi));

      ddzdpvx   *= ddzdpvx;
      ddzdpvy   *= ddzdpvy;
      ddzdphi   *= ddzdphi;
      ddzdtheta *= ddzdtheta;

      double z0Err    = hww.gsftrks_z0Err().at(itrk);
      double phiErr   = hww.gsftrks_phiErr().at(itrk);
      double thetaErr = hww.gsftrks_etaErr().at(itrk)*sin(theta);

      z0Err    *= z0Err;
      phiErr   *= phiErr;
      thetaErr *= thetaErr;
      pvxErr   *= pvxErr;
      pvyErr   *= pvyErr;
      pvzErr   *= pvzErr;

      double value = hww.gsftrks_z0().at(itrk) - pv.z() + (pv.x()*cos(phi) + pv.y()*sin(phi) )*1./tan(theta);

      //note that the error does not account for correlations since we do not store the track covariance matrix
      double error = sqrt(z0Err + pvzErr + ddzdpvx*pvxErr + ddzdpvy*pvyErr + ddzdphi*phiErr + ddzdtheta*thetaErr);

      return std::pair<double, double>(value, error);
  }

}
