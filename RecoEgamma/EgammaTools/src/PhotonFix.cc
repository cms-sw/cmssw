#include <cmath>
#include <cassert>
#include <fstream>
#include <iomanip>

#ifndef LOCAL_FITTING_PROCEDURE
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/EcalBarrelGeometryRecord.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/Records/interface/EcalEndcapGeometryRecord.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "../interface/PhotonFix.h"

#else
#include "PhotonFix.h"
#endif

#ifndef LOCAL_FITTING_PROCEDURE
PhotonFix::PhotonFix(const reco::Photon &p):
  _e(p.energy()), 
  _eta(p.superCluster()->eta()) , 
  _phi(p.superCluster()->phi()),
  _r9(p.r9()) {

  setup(true);
}

PhotonFix::PhotonFix(double eta, double phi):
  _e(0.0), 
  _eta(eta) , 
  _phi(phi),
  _r9(1.0) {

  setup(true);
}
#endif

PhotonFix::PhotonFix(double e, double eta, double phi, double r9, double aC, double aS, double aM, double bC, double bS, double bM):
  _e(e),
  _eta(eta),
  _phi(phi),
  _r9(r9),
  _aC(aC),
  _aS(aS),
  _aM(aM),
  _bC(bC),
  _bS(bS),
  _bM(bM) {

  setup(false);
}

void PhotonFix::setup(bool doGeom){

  #ifndef LOCAL_FITTING_PROCEDURE
  // Check constants have been set up
  assert(_initialisedGeom);
  #endif

  // Determine if EB or EE
  _be=(fabs(_eta)<1.48?0:1);
  
  // Determine if high or low R9
  if(_be==0) _hl=(_r9>=0.94?0:1);
  else       _hl=(_r9>=0.95?0:1);
  
  if (doGeom){
    // Coordinates relative to cracks
    double r2Min;
    if(_be==0) {
      
      r2Min=1.0e6;
      for(unsigned i(0);i<169;i++) {
        for(unsigned j(0);j<360;j++) {
          double de(_eta-_barrelCGap[i][j][0]);
          double df(dPhi(_phi,_barrelCGap[i][j][1]));
          double r2(de*de+df*df);
    
          if(r2<r2Min) {
            r2Min=r2;
            if(i>=84) {
              _aC= de;
              _bC=-df;
            } 
            else {
              _aC=-de;
              _bC= df;
            }
          }
        }
      }
        
      r2Min=1.0e6;
      for(unsigned i(0);i<33;i++) {
        for(unsigned j(0);j<180;j++) {
          double de(_eta-_barrelSGap[i][j][0]);
          double df(dPhi(_phi,_barrelSGap[i][j][1]));
          double r2(de*de+df*df);
            
          if(r2<r2Min) {
            r2Min=r2;
            if(i>=16) {
              _aS= de;
              _bS=-df;
            } 
            else {
              _aS=-de;
              _bS= df;
            }
          }
        }
      }
        
      r2Min=1.0e6;
      for(unsigned i(0);i<7;i++) {
        for(unsigned j(0);j<18;j++) {
          double de(_eta-_barrelMGap[i][j][0]);
          double df(dPhi(_phi,_barrelMGap[i][j][1]));
          double r2(de*de+df*df);
            
          if(r2<r2Min) {
            r2Min=r2;
            if(i>=3) {
              _aM= de;
              _bM=-df;
            } 
            else {
              _aM=-de;
              _bM= df;
            }
          }
        }
      }

    } 
    else {
      unsigned iz(_eta>=0.0?0:1);
      double r[2]={xZ(),yZ()};

      r2Min=1.0e6;
      for(unsigned i(0);i<7080;i++) {
        double dx(r[0]-_endcapCGap[iz][i][0]);
        double dy(r[1]-_endcapCGap[iz][i][1]);
        double r2(dx*dx+dy*dy);

        if(r2<r2Min) {
          r2Min=r2;
          if(r[0]>0.0) _aC= dx;
          else         _aC=-dx;
          if(r[1]>0.0) _bC= dy;
          else         _bC=-dy;
        }
      }

      r2Min=1.0e6;
      for(unsigned i(0);i<264;i++) {
        double dx(r[0]-_endcapSGap[iz][i][0]);
        double dy(r[1]-_endcapSGap[iz][i][1]);
        double r2(dx*dx+dy*dy);

        if(r2<r2Min) {
          r2Min=r2;
          if(r[0]>0.0) _aS= dx;
          else         _aS=-dx;
          if(r[1]>0.0) _bS= dy;
          else         _bS=-dy;
        }
      }

      r2Min=1.0e6;
      for(unsigned i(0);i<1;i++) {
        double dx(r[0]-_endcapMGap[iz][i][0]);
        double dy(r[1]-_endcapMGap[iz][i][1]);
        double r2(dx*dx+dy*dy);

        if(r2<r2Min) {
          r2Min=r2;
          if(iz==0) {_aM= dx;_bM= dy;}
          else      {_aM=-dx;_bM=-dy;}
        }
      }
    }
  }
}

double PhotonFix::fixedEnergy() const {
  
  #ifndef LOCAL_FITTING_PROCEDURE
  // Check constants have been set up
  assert(_initialisedParams);
  #endif  
  
  double f(0.0);
  
  // Overall scale and energy(T) dependence
  f =_meanScale[_be][_hl][0];
  f+=_meanScale[_be][_hl][1]*_e;
  f+=_meanScale[_be][_hl][2]*_e/cosh(_eta);
  f+=_meanScale[_be][_hl][3]*cosh(_eta)/_e;
  
  // General eta or zeta dependence
  if(_be==0) {
    f+=_meanAT[_be][_hl][0]*_eta*_eta;
    f+=expCorrection(_eta,_meanBT[_be][_hl]);
  } 
  else {
    f+=_meanAT[_be][_hl][0]*xZ()*xZ();
    f+=_meanBT[_be][_hl][0]*yZ()*yZ();
  }
  
  // Eta or x crystal, submodule and module dependence
  f+=expCorrection(_aC,_meanAC[_be][_hl]);
  f+=expCorrection(_aS,_meanAS[_be][_hl]);
  f+=expCorrection(_aM,_meanAM[_be][_hl]);
  
  // Phi or y crystal, submodule and module dependence
  f+=expCorrection(_bC,_meanBC[_be][_hl]);
  f+=expCorrection(_bS,_meanBS[_be][_hl]);
  f+=expCorrection(_bM,_meanBM[_be][_hl]);
  
  // R9 dependence
  if(_hl==0) {
    f+=_meanR9[_be][_hl][1]*(_r9-_meanR9[_be][_hl][0])*(_r9-_meanR9[_be][_hl][0])
      +_meanR9[_be][_hl][2]*(_r9-_meanR9[_be][_hl][0])*(_r9-_meanR9[_be][_hl][0])*(_r9-_meanR9[_be][_hl][0]);
  } 
  else {
    f+=_meanR9[_be][_hl][0]*_r9+_meanR9[_be][_hl][1]*_r9*_r9+_meanR9[_be][_hl][2]*_r9*_r9*_r9;
  }
  
  return _e*f;
}

double PhotonFix::sigmaEnergy() const {
  
  #ifndef LOCAL_FITTING_PROCEDURE
  // Check constants have been set up
  assert(_initialisedParams);
  #endif  
  
  // Overall resolution scale vs energy
  double sigma;
  if(_be==0) {
    sigma =_sigmaScale[_be][_hl][0]*_sigmaScale[_be][_hl][0];
    //std::cout << "PhotonFix::sigmaEnergy 1 sigma = " << sigma << std::endl;
    sigma+=_sigmaScale[_be][_hl][1]*_sigmaScale[_be][_hl][1]*_e;
    //std::cout << "PhotonFix::sigmaEnergy 2 sigma = " << sigma << std::endl;
    sigma+=_sigmaScale[_be][_hl][2]*_sigmaScale[_be][_hl][2]*_e*_e;
    //std::cout << "PhotonFix::sigmaEnergy 3 sigma = " << sigma << std::endl;
  } 
  else {
    sigma =_sigmaScale[_be][_hl][0]*_sigmaScale[_be][_hl][0]*cosh(_eta)*cosh(_eta);
    sigma+=_sigmaScale[_be][_hl][1]*_sigmaScale[_be][_hl][1]*_e;
    sigma+=_sigmaScale[_be][_hl][2]*_sigmaScale[_be][_hl][2]*_e*_e;
  }
  sigma=sqrt(sigma);
  
  double f(1.0);
  
  // General eta or zeta dependence
  if(_be==0) {
    f+=_sigmaAT[_be][_hl][0]*_eta*_eta;
    //std::cout << "PhotonFix::sigmaEnergy 4 f = " << f << std::endl;
    f+=expCorrection(_eta,_sigmaBT[_be][_hl]);
    //std::cout << "PhotonFix::sigmaEnergy 5 f = " << f << std::endl;
  } 
  else {
    f+=_sigmaAT[_be][_hl][0]*xZ()*xZ();
    f+=_sigmaBT[_be][_hl][0]*yZ()*yZ();
  }
  
  // Eta or x crystal, submodule and module dependence
  f+=expCorrection(_aC,_sigmaAC[_be][_hl]);
  //std::cout << "PhotonFix::sigmaEnergy 6 f = " << f << std::endl;
  f+=expCorrection(_aS,_sigmaAS[_be][_hl]);
  //std::cout << "PhotonFix::sigmaEnergy 7 f = " << f << std::endl;
  f+=expCorrection(_aM,_sigmaAM[_be][_hl]);
  //std::cout << "PhotonFix::sigmaEnergy 8 f = " << f << std::endl;
  
  // Phi or y crystal, submodule and module dependence
  f+=expCorrection(_bC,_sigmaBC[_be][_hl]);
  //std::cout << "PhotonFix::sigmaEnergy 9 f = " << f << std::endl;
  f+=expCorrection(_bS,_sigmaBS[_be][_hl]);
  //std::cout << "PhotonFix::sigmaEnergy 10 f = " << f << std::endl;
  f+=expCorrection(_bM,_sigmaBM[_be][_hl]);
  //std::cout << "PhotonFix::sigmaEnergy 11 f = " << f << std::endl;
  
  // R9 dependence
  if(_hl==0) {
    f+=_sigmaR9[_be][_hl][1]*(_r9-_sigmaR9[_be][_hl][0])*(_r9-_sigmaR9[_be][_hl][0])
      +_sigmaR9[_be][_hl][2]*(_r9-_sigmaR9[_be][_hl][0])*(_r9-_sigmaR9[_be][_hl][0])*(_r9-_sigmaR9[_be][_hl][0]);
    //std::cout << "PhotonFix::sigmaEnergy 12 f = " << f << std::endl;
  } 
  else {
    f+=_sigmaR9[_be][_hl][0]*_r9+_sigmaR9[_be][_hl][1]*_r9*_r9+_sigmaR9[_be][_hl][2]*_r9*_r9*_r9;
    //std::cout << "PhotonFix::sigmaEnergy 13 f = " << f << std::endl;
  }
  
  return sigma*f;
}

double PhotonFix::rawEnergy() const {
  return _e;
}

double PhotonFix::eta() const {
  return _eta;
}

double PhotonFix::phi() const {
  return _phi;
}

double PhotonFix::r9() const {
  return _r9;
}

double PhotonFix::etaC() const {
  assert(_be==0);
  return _aC;
}

double PhotonFix::etaS() const {
  assert(_be==0);
  return _aS;
}

double PhotonFix::etaM() const {
  assert(_be==0);
  return _aM;
}

double PhotonFix::phiC() const {
  assert(_be==0);
  return _bC;
}

double PhotonFix::phiS() const {
  assert(_be==0);
  return _bS;
}

double PhotonFix::phiM() const {
  assert(_be==0);
  return _bM;
}

double PhotonFix::xZ() const {
  assert(_be==1);
  return asinh(cos(_phi)/sinh(_eta));
}

double PhotonFix::xC() const {
  assert(_be==1);
  return _aC;
}

double PhotonFix::xS() const {
  assert(_be==1);
  return _aS;
}

double PhotonFix::xM() const {
  assert(_be==1);
  return _aM;
}

double PhotonFix::yZ() const {
  assert(_be==1);
  return asinh(sin(_phi)/sinh(_eta));
}

double PhotonFix::yC() const {
  assert(_be==1);
  return _bC;
}

double PhotonFix::yS() const {
  assert(_be==1);
  return _bS;
}

double PhotonFix::yM() const {
  assert(_be==1);
  return _bM;
}

double PhotonFix::aC() const {
  return _aC;
}

double PhotonFix::aS() const {
  return _aS;
}

double PhotonFix::aM() const {
  return _aM;
}

double PhotonFix::bC() const {
  return _bC;
}

double PhotonFix::bS() const {
  return _bS;
}

double PhotonFix::bM() const {
  return _bM;
}

void PhotonFix::barrelCGap(unsigned i, unsigned j, unsigned k, double c){
  _barrelCGap[i][j][k] = c;
}

void PhotonFix::barrelSGap(unsigned i, unsigned j, unsigned k, double c){
  _barrelSGap[i][j][k] = c;
}

void PhotonFix::barrelMGap(unsigned i, unsigned j, unsigned k, double c){
  _barrelMGap[i][j][k] = c;
}

void PhotonFix::endcapCrystal(unsigned i, unsigned j, bool c){
  _endcapCrystal[i][j] = c;
}

void PhotonFix::endcapCGap(unsigned i, unsigned j, unsigned k, double c){
  _endcapCGap[i][j][k] = c;
}

void PhotonFix::endcapSGap(unsigned i, unsigned j, unsigned k, double c){
  _endcapSGap[i][j][k] = c;
}

void PhotonFix::endcapMGap(unsigned i, unsigned j, unsigned k, double c){
  _endcapMGap[i][j][k] = c;
}

void PhotonFix::print() const {
  std::cout << "PhotonFix:  e,eta,phi,r9 = " << _e << ", " << _eta << ", " << _phi << ", " << _r9 << ", gaps "
	    << _aC << ", " << _aS << ", " << _aM << ", "
	    << _bC << ", " << _bS << ", " << _bM << std::endl;
}

void PhotonFix::setParameters(unsigned be, unsigned hl, const double *p) {
  for(unsigned i(0);i<4;i++) {
    _meanScale[be][hl][i] =p[i+ 0*4];
    _meanAT[be][hl][i]    =p[i+ 1*4];
    _meanAC[be][hl][i]    =p[i+ 2*4];
    _meanAS[be][hl][i]    =p[i+ 3*4];
    _meanAM[be][hl][i]    =p[i+ 4*4];
    _meanBT[be][hl][i]    =p[i+ 5*4];
    _meanBC[be][hl][i]    =p[i+ 6*4];
    _meanBS[be][hl][i]    =p[i+ 7*4];
    _meanBM[be][hl][i]    =p[i+ 8*4];
    _meanR9[be][hl][i]    =p[i+ 9*4];
    
    _sigmaScale[be][hl][i]=p[i+10*4];
    _sigmaAT[be][hl][i]   =p[i+11*4];
    _sigmaAC[be][hl][i]   =p[i+12*4];
    _sigmaAS[be][hl][i]   =p[i+13*4];
    _sigmaAM[be][hl][i]   =p[i+14*4];
    _sigmaBT[be][hl][i]   =p[i+15*4];
    _sigmaBC[be][hl][i]   =p[i+16*4];
    _sigmaBS[be][hl][i]   =p[i+17*4];
    _sigmaBM[be][hl][i]   =p[i+18*4];
    _sigmaR9[be][hl][i]   =p[i+19*4];
  }
}

void PhotonFix::getParameters(unsigned be, unsigned hl, double *p) {
  for(unsigned i(0);i<4;i++) {
    p[i+ 0*4]=_meanScale[be][hl][i];
    p[i+ 1*4]=_meanAT[be][hl][i];
    p[i+ 2*4]=_meanAC[be][hl][i];
    p[i+ 3*4]=_meanAS[be][hl][i];
    p[i+ 4*4]=_meanAM[be][hl][i];
    p[i+ 5*4]=_meanBT[be][hl][i];
    p[i+ 6*4]=_meanBC[be][hl][i];
    p[i+ 7*4]=_meanBS[be][hl][i];
    p[i+ 8*4]=_meanBM[be][hl][i];
    p[i+ 9*4]=_meanR9[be][hl][i];
    
    p[i+10*4]=_sigmaScale[be][hl][i];
    p[i+11*4]=_sigmaAT[be][hl][i];
    p[i+12*4]=_sigmaAC[be][hl][i];
    p[i+13*4]=_sigmaAS[be][hl][i];
    p[i+14*4]=_sigmaAM[be][hl][i];
    p[i+15*4]=_sigmaBT[be][hl][i];
    p[i+16*4]=_sigmaBC[be][hl][i];
    p[i+17*4]=_sigmaBS[be][hl][i];
    p[i+18*4]=_sigmaBM[be][hl][i];
    p[i+19*4]=_sigmaR9[be][hl][i];
  }
}
void PhotonFix::dumpConfigParameters(std::ostream &o) {
  o << std::setprecision(9);

  o << "import FWCore.ParameterSet.Config as cms" << std::endl;
  o << " " << std::endl;
  o << "PhotonFixParameters = cms.Pset(" << std::endl;
  
  o << "meanScale = cms.vdouble(";
  for(unsigned be(0);be<2;be++) {
    for(unsigned hl(0);hl<2;hl++) {
      for(unsigned i(0);i<4;i++) {
        if (be==1 && hl==1 && i==3) o << _meanScale[be][hl][i] << ")," << std::endl;
        else o << _meanScale[be][hl][i] << ",";
      }
    }
  }
  o << "meanAT = cms.vdouble(";
  for(unsigned be(0);be<2;be++) {
    for(unsigned hl(0);hl<2;hl++) {
      for(unsigned i(0);i<4;i++) {
        if (be==1 && hl==1 && i==3) o << _meanAT[be][hl][i] << ")," << std::endl;
        else o << _meanAT[be][hl][i] << ",";
      }
    }
  }
  o << "meanAC = cms.vdouble(";
  for(unsigned be(0);be<2;be++) {
    for(unsigned hl(0);hl<2;hl++) {
      for(unsigned i(0);i<4;i++) {
        if (be==1 && hl==1 && i==3) o << _meanAC[be][hl][i] << ")," << std::endl;
        else o << _meanAC[be][hl][i] << ",";
      }
    }
  }
  o << "meanAS = cms.vdouble(";
  for(unsigned be(0);be<2;be++) {
    for(unsigned hl(0);hl<2;hl++) {
      for(unsigned i(0);i<4;i++) {
        if (be==1 && hl==1 && i==3) o << _meanAS[be][hl][i] << ")," << std::endl;
        else o << _meanAS[be][hl][i] << ",";
      }
    }
  }
  o << "meanAM = cms.vdouble(";
  for(unsigned be(0);be<2;be++) {
    for(unsigned hl(0);hl<2;hl++) {
      for(unsigned i(0);i<4;i++) {
        if (be==1 && hl==1 && i==3) o << _meanAM[be][hl][i] << ")," << std::endl;
        else o << _meanAM[be][hl][i] << ",";
      }
    }
  }
  o << "meanBT = cms.vdouble(";
  for(unsigned be(0);be<2;be++) {
    for(unsigned hl(0);hl<2;hl++) {
      for(unsigned i(0);i<4;i++) {
        if (be==1 && hl==1 && i==3) o << _meanBT[be][hl][i] << ")," << std::endl;
        else o << _meanBT[be][hl][i] << ",";
      }
    }
  }
  o << "meanBC = cms.vdouble(";
  for(unsigned be(0);be<2;be++) {
    for(unsigned hl(0);hl<2;hl++) {
      for(unsigned i(0);i<4;i++) {
        if (be==1 && hl==1 && i==3) o << _meanBC[be][hl][i] << ")," << std::endl;
        else o << _meanBC[be][hl][i] << ",";
      }
    }
  }
  o << "meanBS = cms.vdouble(";
  for(unsigned be(0);be<2;be++) {
    for(unsigned hl(0);hl<2;hl++) {
      for(unsigned i(0);i<4;i++) {
        if (be==1 && hl==1 && i==3) o << _meanBS[be][hl][i] << ")," << std::endl;
        else o << _meanBS[be][hl][i] << ",";
      }
    }
  }
  o << "meanBM = cms.vdouble(";
  for(unsigned be(0);be<2;be++) {
    for(unsigned hl(0);hl<2;hl++) {
      for(unsigned i(0);i<4;i++) {
        if (be==1 && hl==1 && i==3) o << _meanBM[be][hl][i] << ")," << std::endl;
        else o << _meanBM[be][hl][i] << ",";
      }
    }
  }
  o << "meanR9 = cms.vdouble(";
  for(unsigned be(0);be<2;be++) {
    for(unsigned hl(0);hl<2;hl++) {
      for(unsigned i(0);i<4;i++) {
        if (be==1 && hl==1 && i==3) o << _meanR9[be][hl][i] << ")," << std::endl;
        else o << _meanR9[be][hl][i] << ",";
      }
    }
  }
  o << "sigmaScale = cms.vdouble(";
  for(unsigned be(0);be<2;be++) {
    for(unsigned hl(0);hl<2;hl++) {
      for(unsigned i(0);i<4;i++) {
        if (be==1 && hl==1 && i==3) o << _sigmaScale[be][hl][i] << ")," << std::endl;
        else o << _sigmaScale[be][hl][i] << ",";
      }
    }
  }
  o << "sigmaAT = cms.vdouble(";
  for(unsigned be(0);be<2;be++) {
    for(unsigned hl(0);hl<2;hl++) {
      for(unsigned i(0);i<4;i++) {
        if (be==1 && hl==1 && i==3) o << _sigmaAT[be][hl][i] << ")," << std::endl;
        else o << _sigmaAT[be][hl][i] << ",";
      }
    }
  }
  o << "sigmaAC = cms.vdouble(";
  for(unsigned be(0);be<2;be++) {
    for(unsigned hl(0);hl<2;hl++) {
      for(unsigned i(0);i<4;i++) {
        if (be==1 && hl==1 && i==3) o << _sigmaAC[be][hl][i] << ")," << std::endl;
        else o << _sigmaAC[be][hl][i] << ",";
      }
    }
  }
  o << "sigmaAS = cms.vdouble(";
  for(unsigned be(0);be<2;be++) {
    for(unsigned hl(0);hl<2;hl++) {
      for(unsigned i(0);i<4;i++) {
        if (be==1 && hl==1 && i==3) o << _sigmaAS[be][hl][i] << ")," << std::endl;
        else o << _sigmaAS[be][hl][i] << ",";
      }
    }
  }
  o << "sigmaAM = cms.vdouble(";
  for(unsigned be(0);be<2;be++) {
    for(unsigned hl(0);hl<2;hl++) {
      for(unsigned i(0);i<4;i++) {
        if (be==1 && hl==1 && i==3) o << _sigmaAM[be][hl][i] << ")," << std::endl;
        else o << _sigmaAM[be][hl][i] << ",";
      }
    }
  }
  o << "sigmaBT = cms.vdouble(";
  for(unsigned be(0);be<2;be++) {
    for(unsigned hl(0);hl<2;hl++) {
      for(unsigned i(0);i<4;i++) {
        if (be==1 && hl==1 && i==3) o << _sigmaBT[be][hl][i] << ")," << std::endl;
        else o << _sigmaBT[be][hl][i] << ",";
      }
    }
  }
  o << "sigmaBC = cms.vdouble(";
  for(unsigned be(0);be<2;be++) {
    for(unsigned hl(0);hl<2;hl++) {
      for(unsigned i(0);i<4;i++) {
        if (be==1 && hl==1 && i==3) o << _sigmaBC[be][hl][i] << ")," << std::endl;
        else o << _sigmaBC[be][hl][i] << ",";
      }
    }
  }
  o << "sigmaBS = cms.vdouble(";
  for(unsigned be(0);be<2;be++) {
    for(unsigned hl(0);hl<2;hl++) {
      for(unsigned i(0);i<4;i++) {
        if (be==1 && hl==1 && i==3) o << _sigmaBS[be][hl][i] << ")," << std::endl;
        else o << _sigmaBS[be][hl][i] << ",";
      }
    }
  }
  o << "sigmaBM = cms.vdouble(";
  for(unsigned be(0);be<2;be++) {
    for(unsigned hl(0);hl<2;hl++) {
      for(unsigned i(0);i<4;i++) {
        if (be==1 && hl==1 && i==3) o << _sigmaBM[be][hl][i] << ")," << std::endl;
        else o << _sigmaBM[be][hl][i] << ",";
      }
    }
  }
  o << "sigmaR9 = cms.vdouble(";
  for(unsigned be(0);be<2;be++) {
    for(unsigned hl(0);hl<2;hl++) {
      for(unsigned i(0);i<4;i++) {
        if (be==1 && hl==1 && i==3) o << _sigmaR9[be][hl][i] << ")" << std::endl;
        else o << _sigmaR9[be][hl][i] << ",";
      }
    }
  }
  o << ")" << std::endl;
}

void PhotonFix::readConfigParameters(std::istream &i) {
  std::string word;
  char c;

  // Read initial lines
  for(unsigned n(0);n<8;n++) i >> word;

  // Read each type of variable
  for(unsigned j(0);j<20;j++) {
    for(unsigned n(0);n<3;n++) i >> word;
    for(unsigned n(0);n<11;n++) i >> c;

    for(unsigned be(0);be<2;be++) {
      for(unsigned hl(0);hl<2;hl++) {
	for(unsigned n(0);n<4;n++) {
	  if(j== 0) i >> c >> _meanScale[be][hl][n];
	  if(j== 1) i >> c >> _meanAT[be][hl][n];
	  if(j== 2) i >> c >> _meanAC[be][hl][n];
	  if(j== 3) i >> c >> _meanAS[be][hl][n];
	  if(j== 4) i >> c >> _meanAM[be][hl][n];
	  if(j== 5) i >> c >> _meanBT[be][hl][n];
	  if(j== 6) i >> c >> _meanBC[be][hl][n];
	  if(j== 7) i >> c >> _meanBS[be][hl][n];
	  if(j== 8) i >> c >> _meanBM[be][hl][n];
	  if(j== 9) i >> c >> _meanR9[be][hl][n];

	  if(j==10) i >> c >> _sigmaScale[be][hl][n];
	  if(j==11) i >> c >> _sigmaAT[be][hl][n];
	  if(j==12) i >> c >> _sigmaAC[be][hl][n];
	  if(j==13) i >> c >> _sigmaAS[be][hl][n];
	  if(j==14) i >> c >> _sigmaAM[be][hl][n];
	  if(j==15) i >> c >> _sigmaBT[be][hl][n];
	  if(j==16) i >> c >> _sigmaBC[be][hl][n];
	  if(j==17) i >> c >> _sigmaBS[be][hl][n];
	  if(j==18) i >> c >> _sigmaBM[be][hl][n];
	  if(j==19) i >> c >> _sigmaR9[be][hl][n];
	}
      }
    }
  }
  i >> word;
}

void PhotonFix::dumpParameters(std::ostream &o) {
  o << std::setprecision(9);

  for(unsigned be(0);be<2;be++) {
    for(unsigned hl(0);hl<2;hl++) {
      for(unsigned i(0);i<4;i++) {
	o << "    _meanScale[" << be << "][" << hl << "][" << i << "]=" << _meanScale[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << "    _meanAT[" << be << "][" << hl << "][" << i << "]=" << _meanAT[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << "    _meanAC[" << be << "][" << hl << "][" << i << "]=" << _meanAC[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << "    _meanAS[" << be << "][" << hl << "][" << i << "]=" << _meanAS[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << "    _meanAM[" << be << "][" << hl << "][" << i << "]=" << _meanAM[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << "    _meanBT[" << be << "][" << hl << "][" << i << "]=" << _meanBT[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << "    _meanBC[" << be << "][" << hl << "][" << i << "]=" << _meanBC[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << "    _meanBS[" << be << "][" << hl << "][" << i << "]=" << _meanBS[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << "    _meanBM[" << be << "][" << hl << "][" << i << "]=" << _meanBM[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << "    _meanR9[" << be << "][" << hl << "][" << i << "]=" << _meanR9[be][hl][i] << ";" << std::endl;
      }
      o << std::endl;
      
      for(unsigned i(0);i<4;i++) {
	o << "    _sigmaScale[" << be << "][" << hl << "][" << i << "]=" << _sigmaScale[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << "    _sigmaAT[" << be << "][" << hl << "][" << i << "]=" << _sigmaAT[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << "    _sigmaAC[" << be << "][" << hl << "][" << i << "]=" << _sigmaAC[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << "    _sigmaAS[" << be << "][" << hl << "][" << i << "]=" << _sigmaAS[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << "    _sigmaAM[" << be << "][" << hl << "][" << i << "]=" << _sigmaAM[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << "    _sigmaBT[" << be << "][" << hl << "][" << i << "]=" << _sigmaBT[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << "    _sigmaBC[" << be << "][" << hl << "][" << i << "]=" << _sigmaBC[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << "    _sigmaBS[" << be << "][" << hl << "][" << i << "]=" << _sigmaBS[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << "    _sigmaBM[" << be << "][" << hl << "][" << i << "]=" << _sigmaBM[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << "    _sigmaR9[" << be << "][" << hl << "][" << i << "]=" << _sigmaR9[be][hl][i] << ";" << std::endl;
      }
      o << std::endl;
    }
  }
}

void PhotonFix::printParameters(std::ostream &o) {
  o << "PhotonFix::printParameters()" << std::endl;
  
  for(unsigned be(0);be<2;be++) {
    for(unsigned hl(0);hl<2;hl++) {
      o << " Parameters for " << (be==0?"barrel":"endcap")
	<< ", " << (hl==0?"high":"low") << " R9" << std::endl;
      
      o << "  Mean  scaling        ";
      for(unsigned i(0);i<4;i++) o << std::setw(14) << _meanScale[be][hl][i];
      o << std::endl;
      o << "  Mean  " << (be==0?"Eta  ":"ZetaX") << " total    ";
      for(unsigned i(0);i<4;i++) o << std::setw(14) << _meanAT[be][hl][i];
      o << std::endl;
      o << "  Mean  " << (be==0?"Eta  ":"ZetaX") << " crystal  ";
      for(unsigned i(0);i<4;i++) o << std::setw(14) << _meanAC[be][hl][i];
      o << std::endl;
      o << "  Mean  " << (be==0?"Eta  ":"ZetaX") << " submodule";
      for(unsigned i(0);i<4;i++) o << std::setw(14) << _meanAS[be][hl][i];
      o << std::endl;
      o << "  Mean  " << (be==0?"Eta  ":"ZetaX") << " module   ";
      for(unsigned i(0);i<4;i++) o << std::setw(14) << _meanAM[be][hl][i];
      o << std::endl;
      o << "  Mean  " << (be==0?"Eta   zero     ":"ZetaY total    ");
      for(unsigned i(0);i<4;i++) o << std::setw(14) << _meanBT[be][hl][i];
      o << std::endl;
      o << "  Mean  " << (be==0?"Phi  ":"ZetaY") << " crystal  ";
      for(unsigned i(0);i<4;i++) o << std::setw(14) << _meanBC[be][hl][i];
      o << std::endl;
      o << "  Mean  " << (be==0?"Phi  ":"ZetaY") << " submodule";
      for(unsigned i(0);i<4;i++) o << std::setw(14) << _meanBS[be][hl][i];
      o << std::endl;
      o << "  Mean  " << (be==0?"Phi  ":"ZetaY") << " module   ";
      for(unsigned i(0);i<4;i++) o << std::setw(14) << _meanBM[be][hl][i];
      o << std::endl;
      o << "  Mean  R9             ";
      for(unsigned i(0);i<4;i++) o << std::setw(14) << _meanR9[be][hl][i];
      o << std::endl;
      
      o << "  Sigma scaling        ";
      for(unsigned i(0);i<4;i++) o << std::setw(14) << _sigmaScale[be][hl][i];
      o << std::endl;
      o << "  Sigma " << (be==0?"Eta  ":"ZetaX") << " total    ";
      for(unsigned i(0);i<4;i++) o << std::setw(14) << _sigmaAT[be][hl][i];
      o << std::endl;
      o << "  Sigma " << (be==0?"Eta  ":"ZetaX") << " crystal  ";
      for(unsigned i(0);i<4;i++) o << std::setw(14) << _sigmaAC[be][hl][i];
      o << std::endl;
      o << "  Sigma " << (be==0?"Eta  ":"ZetaX") << " submodule";
      for(unsigned i(0);i<4;i++) o << std::setw(14) << _sigmaAS[be][hl][i];
      o << std::endl;
      o << "  Sigma " << (be==0?"Eta  ":"ZetaX") << " module   ";
      for(unsigned i(0);i<4;i++) o << std::setw(14) << _sigmaAM[be][hl][i];
      o << std::endl;
      o << "  Sigma " << (be==0?"Eta  ":"ZetaY") << " total    ";
      for(unsigned i(0);i<4;i++) o << std::setw(14) << _sigmaBT[be][hl][i];
      o << std::endl;
      o << "  Sigma " << (be==0?"Eta  ":"ZetaY") << " crystal  ";
      for(unsigned i(0);i<4;i++) o << std::setw(14) << _sigmaBC[be][hl][i];
      o << std::endl;
      o << "  Sigma " << (be==0?"Phi  ":"ZetaY") << " submodule";
      for(unsigned i(0);i<4;i++) o << std::setw(14) << _sigmaBS[be][hl][i];
      o << std::endl;
      o << "  Sigma " << (be==0?"Phi  ":"ZetaY") << " module   ";
      for(unsigned i(0);i<4;i++) o << std::setw(14) << _sigmaBM[be][hl][i];
      o << std::endl;
      o << "  Sigma R9             ";
      for(unsigned i(0);i<4;i++) o << std::setw(14) << _sigmaR9[be][hl][i];
      o << std::endl;
    }
  }
}

void PhotonFix::dumpGaps(std::ostream &o) {
  o << std::setprecision(15);
  
  for(unsigned i(0);i<169;i++) {
    for(unsigned j(0);j<360;j++) {
      for(unsigned k(0);k<2;k++) {
	o << _barrelCGap[i][j][k] << std::endl;
      }
    }
  }
  
  for(unsigned i(0);i<33;i++) {
    for(unsigned j(0);j<180;j++) {
      for(unsigned k(0);k<2;k++) {
	o << _barrelSGap[i][j][k] << std::endl;
      }
    }
  }
  
  for(unsigned i(0);i<7;i++) {
    for(unsigned j(0);j<18;j++) {
      for(unsigned k(0);k<2;k++) {
	o << _barrelMGap[i][j][k] << std::endl;
      }
    }
  }
  
  for(unsigned i(0);i<100;i++) {
    for(unsigned j(0);j<100;j++) {
      if(_endcapCrystal[i][j]) o << 0 << std::endl;
      else                     o << 1 << std::endl;
    }
  }
  
  for(unsigned i(0);i<2;i++) {
    for(unsigned j(0);j<7080;j++) {
      for(unsigned k(0);k<2;k++) {
	o << _endcapCGap[i][j][k] << std::endl;
      }
    }
  }
  
  for(unsigned i(0);i<2;i++) {
    for(unsigned j(0);j<264;j++) {
      for(unsigned k(0);k<2;k++) {
	o << _endcapSGap[i][j][k] << std::endl;
      }
    }
  }
  
  for(unsigned i(0);i<2;i++) {
    for(unsigned j(0);j<1;j++) {
      for(unsigned k(0);k<2;k++) {
	o << _endcapMGap[i][j][k] << std::endl;
      }
    }
  }
}

double PhotonFix::asinh(double s) {
  if(s>=0.0) return  log(sqrt(s*s+1.0)+s);
  else       return -log(sqrt(s*s+1.0)-s);
}

double PhotonFix::dPhi(double f0, double f1) {
  double df(f0-f1);
  if(df> _onePi) df-=_twoPi;
  if(df<-_onePi) df+=_twoPi;
  return df;
}

double PhotonFix::aPhi(double f0, double f1) {
  double af(0.5*(f0+f1));
  if(fabs(dPhi(af,f0))>0.5*_onePi) {
    if(af>=0.0) af-=_onePi;
    else        af+=_onePi;
  }
  
  assert(fabs(dPhi(af,f0))<0.5*_onePi);
  assert(fabs(dPhi(af,f1))<0.5*_onePi);
  
  return af;
}

double PhotonFix::expCorrection(double a, const double *p) {
  if(p[1]==0.0 || p[2]==0.0 || p[3]==0.0) return 0.0;
  
  double b(a-p[0]);
  if(b>=0.0) return p[1]*exp(-fabs(p[2])*b);
  else       return p[1]*exp( fabs(p[3])*b);
}

double PhotonFix::gausCorrection(double a, const double *p) {
  if(p[1]==0.0 || p[2]==0.0 || p[3]==0.0) return 0.0;
  
  double b(a-p[0]);
  if(b>=0.0) return p[1]*exp(-0.5*p[2]*p[2]*b*b);
  else       return p[1]*exp(-0.5*p[3]*p[3]*b*b);
}

#ifndef LOCAL_FITTING_PROCEDURE
bool PhotonFix::initialiseParameters(const edm::ParameterSet &iConfig){
  
  if(_initialisedParams) return false;

  edm::ParameterSet fitParameters;
  fitParameters = iConfig.getParameter<edm::ParameterSet>("PFParameters");

  std::vector<double> meanScale = fitParameters.getParameter<std::vector<double> >("meanScale");
  std::vector<double> meanAT = fitParameters.getParameter<std::vector<double> >("meanAT");
  std::vector<double> meanAC = fitParameters.getParameter<std::vector<double> >("meanAC");
  std::vector<double> meanAS = fitParameters.getParameter<std::vector<double> >("meanAS");
  std::vector<double> meanAM = fitParameters.getParameter<std::vector<double> >("meanAM");
  std::vector<double> meanBT = fitParameters.getParameter<std::vector<double> >("meanBT");
  std::vector<double> meanBC = fitParameters.getParameter<std::vector<double> >("meanBC");
  std::vector<double> meanBS = fitParameters.getParameter<std::vector<double> >("meanBS");
  std::vector<double> meanBM = fitParameters.getParameter<std::vector<double> >("meanBM");
  std::vector<double> meanR9 = fitParameters.getParameter<std::vector<double> >("meanR9");
  std::vector<double> sigmaScale = fitParameters.getParameter<std::vector<double> >("sigmaScale");
  std::vector<double> sigmaAT = fitParameters.getParameter<std::vector<double> >("sigmaAT");
  std::vector<double> sigmaAC = fitParameters.getParameter<std::vector<double> >("sigmaAC");
  std::vector<double> sigmaAS = fitParameters.getParameter<std::vector<double> >("sigmaAS");
  std::vector<double> sigmaAM = fitParameters.getParameter<std::vector<double> >("sigmaAM");
  std::vector<double> sigmaBT = fitParameters.getParameter<std::vector<double> >("sigmaBT");
  std::vector<double> sigmaBC = fitParameters.getParameter<std::vector<double> >("sigmaBC");
  std::vector<double> sigmaBS = fitParameters.getParameter<std::vector<double> >("sigmaBS");
  std::vector<double> sigmaBM = fitParameters.getParameter<std::vector<double> >("sigmaBM");
  std::vector<double> sigmaR9 = fitParameters.getParameter<std::vector<double> >("sigmaR9");

  for(unsigned be(0);be<2;be++) {
    for(unsigned hl(0);hl<2;hl++) {
      for(unsigned i(0);i<4;i++) {
        _meanScale[be][hl][i] = meanScale.at((8*be)+(4*hl)+i);
        _meanAT[be][hl][i] = meanAT.at((8*be)+(4*hl)+i);
        _meanAC[be][hl][i] = meanAC.at((8*be)+(4*hl)+i);
        _meanAS[be][hl][i] = meanAS.at((8*be)+(4*hl)+i);
        _meanAM[be][hl][i] = meanAM.at((8*be)+(4*hl)+i);
        _meanBT[be][hl][i] = meanBT.at((8*be)+(4*hl)+i);
        _meanBC[be][hl][i] = meanBC.at((8*be)+(4*hl)+i);
        _meanBS[be][hl][i] = meanBS.at((8*be)+(4*hl)+i);
        _meanBM[be][hl][i] = meanBM.at((8*be)+(4*hl)+i);
        _meanR9[be][hl][i] = meanR9.at((8*be)+(4*hl)+i);
        
        _sigmaScale[be][hl][i] = sigmaScale.at((8*be)+(4*hl)+i);
        _sigmaAT[be][hl][i] = sigmaAT.at((8*be)+(4*hl)+i);
        _sigmaAC[be][hl][i] = sigmaAC.at((8*be)+(4*hl)+i);
        _sigmaAS[be][hl][i] = sigmaAS.at((8*be)+(4*hl)+i);
        _sigmaAM[be][hl][i] = sigmaAM.at((8*be)+(4*hl)+i);
        _sigmaBT[be][hl][i] = sigmaBT.at((8*be)+(4*hl)+i);
        _sigmaBC[be][hl][i] = sigmaBC.at((8*be)+(4*hl)+i);
        _sigmaBS[be][hl][i] = sigmaBS.at((8*be)+(4*hl)+i);
        _sigmaBM[be][hl][i] = sigmaBM.at((8*be)+(4*hl)+i);
        _sigmaR9[be][hl][i] = sigmaR9.at((8*be)+(4*hl)+i);
      }
    }
  }
  _initialisedParams=true;
  
  assert(_initialisedParams);
  return true;
}

bool PhotonFix::initialiseGeometry(const edm::EventSetup &iSetup) {
 
  if (_initialisedGeom) return false;
  // Get ECAL geometry
  edm::ESHandle<CaloGeometry> geoHandle;
  iSetup.get<CaloGeometryRecord>().get(geoHandle);
  const CaloGeometry& geometry = *geoHandle;
  
  // EB
  const CaloSubdetectorGeometry *barrelGeometry = geometry.getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  
  double bc[170][360][2];
  for(int iz(0);iz<2;iz++) {
    for(int ie(0);ie<85;ie++) {
      int id = ie+1;
      if (iz==0) id = ie-85; 
      for(int ip(0);ip<360;ip++) {
        EBDetId eb(id,ip+1);	
        const CaloCellGeometry *cellGeometry = barrelGeometry->getGeometry(eb);
        GlobalPoint crystalPos = cellGeometry->getPosition();
        bc[85*iz+ie][ip][0]=crystalPos.eta();
        bc[85*iz+ie][ip][1]=crystalPos.phi();
      }
    }
  }
  
  for(unsigned i(0);i<169;i++) {
    for(unsigned j(0);j<360;j++) {
      unsigned k((j+1)%360);
     
      double eta = 0.25*(    bc[i][j][0]+bc[i+1][j][0]+
				     bc[i][k][0]+bc[i+1][k][0]);
      double phi = aPhi(aPhi(bc[i][j][1],bc[i+1][j][1]),
				aPhi(bc[i][k][1],bc[i+1][k][1]));

      barrelCGap(i,j,0,eta);
      barrelCGap(i,j,1,phi);
      
      if((i%5)==4 && (j%2)==1) {
        barrelSGap(i/5,j/2,0,eta);
        barrelSGap(i/5,j/2,1,phi);	
      }
      
      if((j%20)==19) {
        if(i== 19) {barrelMGap(0,j/20,0,eta); barrelMGap(0,j/20,1,phi);}
        if(i== 39) {barrelMGap(1,j/20,0,eta); barrelMGap(1,j/20,1,phi);}
        if(i== 59) {barrelMGap(2,j/20,0,eta); barrelMGap(2,j/20,1,phi);}
        if(i== 84) {barrelMGap(3,j/20,0,eta); barrelMGap(3,j/20,1,phi);}
        if(i==109) {barrelMGap(4,j/20,0,eta); barrelMGap(4,j/20,1,phi);}
        if(i==129) {barrelMGap(5,j/20,0,eta); barrelMGap(5,j/20,1,phi);}
        if(i==149) {barrelMGap(6,j/20,0,eta); barrelMGap(6,j/20,1,phi);}
      
      }
    }
  }
  
  // EE
  const CaloSubdetectorGeometry *endcapGeometry = geometry.getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
  
  double ec[2][100][100][2];
  bool valid[100][100];
  int val_count=0;
  for(int iz(0);iz<2;iz++) {
    for(int ix(0);ix<100;ix++) {
      for(int iy(0);iy<100;iy++) {
        valid[ix][iy] = EEDetId::validDetId(ix+1,iy+1,2*iz-1);
        if(iz==0) endcapCrystal(ix,iy,valid[ix][iy]);
	      if(valid[ix][iy]) {
	        EEDetId ee(ix+1,iy+1,2*iz-1);
          val_count+=1;
          
          const CaloCellGeometry *cellGeometry = endcapGeometry->getGeometry(ee);
          GlobalPoint crystalPos = cellGeometry->getPosition();
          ec[iz][ix][iy][0]=asinh(crystalPos.x()/fabs(crystalPos.z()));
          ec[iz][ix][iy][1]=asinh(crystalPos.y()/fabs(crystalPos.z()));
	      }
      }
    }
  }
  //  std::cout << "GG valid " << val_count << std::endl;
  double c[2];
  for(unsigned iz(0);iz<2;iz++) {
    unsigned nC(0),nS(0);
    for(unsigned i(0);i<99;i++) {
      for(unsigned j(0);j<99;j++) {
	      if(valid[i][j  ] && valid[i+1][j  ] && 
	        valid[i][j+1] && valid[i+1][j+1]) {
	        for(unsigned k(0);k<2;k++) {

            c[k] = 0.25*(ec[iz][i][j][k]+ec[iz][i+1][j][k]+ec[iz][i][j+1][k]+ec[iz][i+1][j+1][k]);

	          endcapCGap(iz,nC,k,c[k]);	 
          }
	  
	        if((i%5)==4 && (j%5)==4) {
	          for(unsigned k(0);k<2;k++) {
	            endcapSGap(iz,nS,k,c[k]);	 
	          }
	          nS++;
	        }
	        nC++;
	      }
      }
    }
    //    std::cout << "Endcap number of crystal, submodule boundaries = "
    //	      << nC << ", " << nS << std::endl;
  }
  
  // Hardcode EE D-module gap to 0,0
	endcapMGap(0,0,0,0.0);	 
	endcapMGap(0,0,1,0.0);	 
	endcapMGap(1,0,0,0.0);	 
	endcapMGap(1,0,1,0.0);	 
  
  _initialisedGeom = true;
  assert(_initialisedGeom);
  return true;
}
#endif

const double PhotonFix::_onePi(acos(-1.0));
const double PhotonFix::_twoPi(2.0*acos(-1.0));

bool   PhotonFix::_initialisedGeom=false;
bool   PhotonFix::_initialisedParams=false;

double PhotonFix::_meanScale[2][2][4];
double PhotonFix::_meanAT[2][2][4];
double PhotonFix::_meanAC[2][2][4];
double PhotonFix::_meanAS[2][2][4];
double PhotonFix::_meanAM[2][2][4];
double PhotonFix::_meanBT[2][2][4];
double PhotonFix::_meanBC[2][2][4];
double PhotonFix::_meanBS[2][2][4];
double PhotonFix::_meanBM[2][2][4];
double PhotonFix::_meanR9[2][2][4];

double PhotonFix::_sigmaScale[2][2][4];
double PhotonFix::_sigmaAT[2][2][4];
double PhotonFix::_sigmaAC[2][2][4];
double PhotonFix::_sigmaAS[2][2][4];
double PhotonFix::_sigmaAM[2][2][4];
double PhotonFix::_sigmaBT[2][2][4];
double PhotonFix::_sigmaBC[2][2][4];
double PhotonFix::_sigmaBS[2][2][4];
double PhotonFix::_sigmaBM[2][2][4];
double PhotonFix::_sigmaR9[2][2][4];

double PhotonFix::_barrelCGap[169][360][2];
double PhotonFix::_barrelSGap[33][180][2];
double PhotonFix::_barrelMGap[7][18][2];

bool   PhotonFix::_endcapCrystal[100][100];
double PhotonFix::_endcapCGap[2][7080][2];
double PhotonFix::_endcapSGap[2][264][2];
double PhotonFix::_endcapMGap[2][1][2];
