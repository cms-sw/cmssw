#include <cmath>
#include <cassert>
#include <fstream>
#include <iomanip>

// ensure that this include points to the appropriate location for PhotonFix.h
#include "../interface/PhotonFix.h"

PhotonFix::PhotonFix(double e, double eta, double phi, double r9) :
  _e(e), _eta(eta), _phi(phi), _r9(r9) {

  setup();
}
 
void PhotonFix::setup(){
  // Check constants have been set up
  assert(_initialised);

  // Determine if EB or EE
  _be=(fabs(_eta)<1.48?0:1);
  
  // Determine if high or low R9
  if(_be==0) _hl=(_r9>=0.94?0:1);
  else       _hl=(_r9>=0.95?0:1);
  
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
	  } else {
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
	  } else {
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
	  } else {
	    _aM=-de;
	    _bM= df;
	  }
	}
      }
    }

  } else {
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

double PhotonFix::fixedEnergy() const {
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
  } else {
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
  } else {
    f+=_meanR9[_be][_hl][0]*_r9+_meanR9[_be][_hl][1]*_r9*_r9+_meanR9[_be][_hl][2]*_r9*_r9*_r9;
  }
  
  return _e*f;
}

double PhotonFix::sigmaEnergy() const {
  
  // Overall resolution scale vs energy
  double sigma;
  if(_be==0) {
    sigma =_sigmaScale[_be][_hl][0]*_sigmaScale[_be][_hl][0];
    //std::cout << "PhotonFix::sigmaEnergy 1 sigma = " << sigma << std::endl;
    sigma+=_sigmaScale[_be][_hl][1]*_sigmaScale[_be][_hl][1]*_e;
    //std::cout << "PhotonFix::sigmaEnergy 2 sigma = " << sigma << std::endl;
    sigma+=_sigmaScale[_be][_hl][2]*_sigmaScale[_be][_hl][2]*_e*_e;
    //std::cout << "PhotonFix::sigmaEnergy 3 sigma = " << sigma << std::endl;
  } else {
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
  } else {
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
  } else {
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

double PhotonFix::GetaPhi(double f0, double f1){
  return aPhi(f0,f1);
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

double PhotonFix::asinh(double s) {
  if(s>=0.0) return  log(sqrt(s*s+1.0)+s);
  else       return -log(sqrt(s*s+1.0)-s);
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
bool PhotonFix::initialised() {
  return _initialised;
}
bool PhotonFix::initialise(const std::string &s) {
  if(_initialised) return false;

  
  initialiseParameters(s);
  initialiseGeometry(s);
  return true;
}

bool PhotonFix::initialiseParameters(const std::string &s) {
  _initialised=false;
  
  if(s=="Nominal") {
    for(unsigned be(0);be<2;be++) {
      for(unsigned hl(0);hl<2;hl++) {
	for(unsigned i(0);i<4;i++) {
	  _meanScale[be][hl][i]=0;
	  _meanAT[be][hl][i]=0;
	  _meanAC[be][hl][i]=0;
	  _meanAS[be][hl][i]=0;
	  _meanAM[be][hl][i]=0;
	  _meanBT[be][hl][i]=0;
	  _meanBC[be][hl][i]=0;
	  _meanBS[be][hl][i]=0;
	  _meanBM[be][hl][i]=0;
	  _meanR9[be][hl][i]=0;
	  
	  _sigmaScale[be][hl][i]=0;
	  _sigmaAT[be][hl][i]=0;
	  _sigmaAC[be][hl][i]=0;
	  _sigmaAS[be][hl][i]=0;
	  _sigmaAM[be][hl][i]=0;
	  _sigmaBT[be][hl][i]=0;
	  _sigmaBC[be][hl][i]=0;
	  _sigmaBS[be][hl][i]=0;
	  _sigmaBM[be][hl][i]=0;
	  _sigmaR9[be][hl][i]=0;
	}
	
	_meanScale[be][hl][0]=1.0;
	if(be==0) {
	  _sigmaScale[be][hl][0]=0.2;
	  _sigmaScale[be][hl][1]=0.03;
	  _sigmaScale[be][hl][2]=0.006;
	} else {
	  _sigmaScale[be][hl][0]=0.25;
	  _sigmaScale[be][hl][1]=0.05;
	  _sigmaScale[be][hl][2]=0.010;
	}
      }
    }
    
    _initialised=true;
  }
  
  if(s=="3_8") {
    _meanScale[0][0][0]=0.994724;
    _meanScale[0][0][1]=1.98102e-06;
    _meanScale[0][0][2]=1.43015e-05;
    _meanScale[0][0][3]=-0.0908525;
    _meanAT[0][0][0]=0.0;
    _meanAT[0][0][1]=0.0;
    _meanAT[0][0][2]=0.0;
    _meanAT[0][0][3]=0.0;
    _meanAC[0][0][0]=-0.00352041;
    _meanAC[0][0][1]=0.00982015;
    _meanAC[0][0][2]=434.32;
    _meanAC[0][0][3]=529.508;
    _meanAS[0][0][0]=-1.1;
    _meanAS[0][0][1]=0.00135995;
    _meanAS[0][0][2]=295.712;
    _meanAS[0][0][3]=5.13202e+07;
    _meanAM[0][0][0]=-0.00140562;
    _meanAM[0][0][1]=0.156322;
    _meanAM[0][0][2]=263.097;
    _meanAM[0][0][3]=222.294;
    _meanBT[0][0][0]=0.0;
    _meanBT[0][0][1]=0.0;
    _meanBT[0][0][2]=0.0;
    _meanBT[0][0][3]=0.0;
    _meanBC[0][0][0]=-0.00294295;
    _meanBC[0][0][1]=0.011533;
    _meanBC[0][0][2]=562.905;
    _meanBC[0][0][3]=421.097;
    _meanBS[0][0][0]=-0.00204373;
    _meanBS[0][0][1]=0.00347592;
    _meanBS[0][0][2]=36.5614;
    _meanBS[0][0][3]=1265.25;
    _meanBM[0][0][0]=-0.00275381;
    _meanBM[0][0][1]=0.0812447;
    _meanBM[0][0][2]=216.885;
    _meanBM[0][0][3]=264.754;
    _meanR9[0][0][0]=0.952584;
    _meanR9[0][0][1]=22.7119;
    _meanR9[0][0][2]=402.816;
    _meanR9[0][0][3]=0;
    
    _sigmaScale[0][0][0]=0.167184;
    _sigmaScale[0][0][1]=6.14323e-11;
    _sigmaScale[0][0][2]=0.00769693;
    _sigmaScale[0][0][3]=0;
    _sigmaAT[0][0][0]=0.228255;
    _sigmaAT[0][0][1]=0;
    _sigmaAT[0][0][2]=0;
    _sigmaAT[0][0][3]=0;
    _sigmaAC[0][0][0]=-0.00411906;
    _sigmaAC[0][0][1]=0.077799;
    _sigmaAC[0][0][2]=23.1033;
    _sigmaAC[0][0][3]=-3e+17;
    _sigmaAS[0][0][0]=0;
    _sigmaAS[0][0][1]=0;
    _sigmaAS[0][0][2]=0;
    _sigmaAS[0][0][3]=0;
    _sigmaAM[0][0][0]=-0.000130695;
    _sigmaAM[0][0][1]=11.2121;
    _sigmaAM[0][0][2]=468.535;
    _sigmaAM[0][0][3]=407.652;
    _sigmaBT[0][0][0]=1.33384e-05;
    _sigmaBT[0][0][1]=8.77098;
    _sigmaBT[0][0][2]=324.048;
    _sigmaBT[0][0][3]=239.868;
    _sigmaBC[0][0][0]=-0.00281964;
    _sigmaBC[0][0][1]=0.125811;
    _sigmaBC[0][0][2]=538.949;
    _sigmaBC[0][0][3]=1358.76;
    _sigmaBS[0][0][0]=0;
    _sigmaBS[0][0][1]=0;
    _sigmaBS[0][0][2]=0;
    _sigmaBS[0][0][3]=0;
    _sigmaBM[0][0][0]=-0.00293676;
    _sigmaBM[0][0][1]=8.88276;
    _sigmaBM[0][0][2]=350.032;
    _sigmaBM[0][0][3]=580.354;
    _sigmaR9[0][0][0]=0.955876;
    _sigmaR9[0][0][1]=2254.5;
    _sigmaR9[0][0][2]=14627;
    _sigmaR9[0][0][3]=0;
    
    _meanScale[0][1][0]=0.888348;
    _meanScale[0][1][1]=1.20452e-05;
    _meanScale[0][1][2]=-1.04458e-05;
    _meanScale[0][1][3]=-0.542383;
    _meanAT[0][1][0]=0.0;
    _meanAT[0][1][1]=0.0;
    _meanAT[0][1][2]=0.0;
    _meanAT[0][1][3]=0.0;
    _meanAC[0][1][0]=-0.00320856;
    _meanAC[0][1][1]=0.0240109;
    _meanAC[0][1][2]=115.145;
    _meanAC[0][1][3]=205.859;
    _meanAS[0][1][0]=0.0349736;
    _meanAS[0][1][1]=-0.00232864;
    _meanAS[0][1][2]=318.584;
    _meanAS[0][1][3]=1.4e+09;
    _meanAM[0][1][0]=-0.00104798;
    _meanAM[0][1][1]=0.208249;
    _meanAM[0][1][2]=297.049;
    _meanAM[0][1][3]=220.609;
    _meanBT[0][1][0]=0.0;
    _meanBT[0][1][1]=0.0;
    _meanBT[0][1][2]=0.0;
    _meanBT[0][1][3]=0.0;
    _meanBC[0][1][0]=-0.00420429;
    _meanBC[0][1][1]=0.00203991;
    _meanBC[0][1][2]=172.278;
    _meanBC[0][1][3]=410.677;
    _meanBS[0][1][0]=-0.0430854;
    _meanBS[0][1][1]=0.0961883;
    _meanBS[0][1][2]=0.196958;
    _meanBS[0][1][3]=11442.2;
    _meanBM[0][1][0]=-0.00389457;
    _meanBM[0][1][1]=0.0449086;
    _meanBM[0][1][2]=78.9252;
    _meanBM[0][1][3]=103.237;
    _meanR9[0][1][0]=0.0182102;
    _meanR9[0][1][1]=-0.03752;
    _meanR9[0][1][2]=0.0198881;
    _meanR9[0][1][3]=0;
    
    _sigmaScale[0][1][0]=0.386681;
    _sigmaScale[0][1][1]=0.0913412;
    _sigmaScale[0][1][2]=0.00119232;
    _sigmaScale[0][1][3]=0;
    _sigmaAT[0][1][0]=1.36562;
    _sigmaAT[0][1][1]=0;
    _sigmaAT[0][1][2]=0;
    _sigmaAT[0][1][3]=0;
    _sigmaAC[0][1][0]=-0.00504613;
    _sigmaAC[0][1][1]=-1.09115;
    _sigmaAC[0][1][2]=8.57406;
    _sigmaAC[0][1][3]=57.1351;
    _sigmaAS[0][1][0]=0;
    _sigmaAS[0][1][1]=0;
    _sigmaAS[0][1][2]=0;
    _sigmaAS[0][1][3]=0;
    _sigmaAM[0][1][0]=-0.00014319;
    _sigmaAM[0][1][1]=5.39527;
    _sigmaAM[0][1][2]=432.566;
    _sigmaAM[0][1][3]=265.165;
    _sigmaBT[0][1][0]=-0.040161;
    _sigmaBT[0][1][1]=2.65711;
    _sigmaBT[0][1][2]=-0.398357;
    _sigmaBT[0][1][3]=-0.440649;
    _sigmaBC[0][1][0]=0.00580015;
    _sigmaBC[0][1][1]=-0.631833;
    _sigmaBC[0][1][2]=18594.3;
    _sigmaBC[0][1][3]=4.00955e+08;
    _sigmaBS[0][1][0]=0;
    _sigmaBS[0][1][1]=0;
    _sigmaBS[0][1][2]=0;
    _sigmaBS[0][1][3]=0;
    _sigmaBM[0][1][0]=-0.00376665;
    _sigmaBM[0][1][1]=3.74316;
    _sigmaBM[0][1][2]=102.72;
    _sigmaBM[0][1][3]=157.396;
    _sigmaR9[0][1][0]=-3.12696;
    _sigmaR9[0][1][1]=1.75114;
    _sigmaR9[0][1][2]=0;
    _sigmaR9[0][1][3]=0;
    
    _meanScale[1][0][0]=0.999461;
    _meanScale[1][0][1]=4.37414e-06;
    _meanScale[1][0][2]=4.92078e-06;
    _meanScale[1][0][3]=-0.121609;
    _meanAT[1][0][0]=0.0;
    _meanAT[1][0][1]=0.0;
    _meanAT[1][0][2]=0.0;
    _meanAT[1][0][3]=0.0;
    _meanAC[1][0][0]=-0.000396058;
    _meanAC[1][0][1]=0.0144837;
    _meanAC[1][0][2]=1374.93;
    _meanAC[1][0][3]=945.634;
    _meanAS[1][0][0]=-0.000871036;
    _meanAS[1][0][1]=0.0442747;
    _meanAS[1][0][2]=645.709;
    _meanAS[1][0][3]=962.845;
    _meanAM[1][0][0]=0.000434298;
    _meanAM[1][0][1]=0.0658628;
    _meanAM[1][0][2]=1928.49;
    _meanAM[1][0][3]=728.522;
    _meanBT[1][0][0]=0.0;
    _meanBT[1][0][1]=0.0;
    _meanBT[1][0][2]=0.0;
    _meanBT[1][0][3]=0.0;
    _meanBC[1][0][0]=-0.000452212;
    _meanBC[1][0][1]=0.0129968;
    _meanBC[1][0][2]=1056.08;
    _meanBC[1][0][3]=759.102;
    _meanBS[1][0][0]=-0.000786157;
    _meanBS[1][0][1]=0.0346555;
    _meanBS[1][0][2]=592.239;
    _meanBS[1][0][3]=854.285;
    _meanBM[1][0][0]=-0.0665038;
    _meanBM[1][0][1]=-0.00211713;
    _meanBM[1][0][2]=4.84395;
    _meanBM[1][0][3]=11.6644;
    _meanR9[1][0][0]=0.971355;
    _meanR9[1][0][1]=47.2751;
    _meanR9[1][0][2]=536.907;
    _meanR9[1][0][3]=0;
    
    _sigmaScale[1][0][0]=0.254641;
    _sigmaScale[1][0][1]=0.00264818;
    _sigmaScale[1][0][2]=0.0114953;
    _sigmaScale[1][0][3]=0;
    _sigmaAT[1][0][0]=0.935839;
    _sigmaAT[1][0][1]=0;
    _sigmaAT[1][0][2]=0;
    _sigmaAT[1][0][3]=0;
    _sigmaAC[1][0][0]=-0.00476475;
    _sigmaAC[1][0][1]=2.14548;
    _sigmaAC[1][0][2]=29937;
    _sigmaAC[1][0][3]=2.6e+11;
    _sigmaAS[1][0][0]=-8.17285e-05;
    _sigmaAS[1][0][1]=1.5821;
    _sigmaAS[1][0][2]=1928.83;
    _sigmaAS[1][0][3]=902.519;
    _sigmaAM[1][0][0]=0.0278577;
    _sigmaAM[1][0][1]=0.58439;
    _sigmaAM[1][0][2]=43.3575;
    _sigmaAM[1][0][3]=19.7836;
    _sigmaBT[1][0][0]=-0.456051;
    _sigmaBT[1][0][1]=0;
    _sigmaBT[1][0][2]=0;
    _sigmaBT[1][0][3]=0;
    _sigmaBC[1][0][0]=-0.00264527;
    _sigmaBC[1][0][1]=0.696043;
    _sigmaBC[1][0][2]=7.49509e+12;
    _sigmaBC[1][0][3]=96843;
    _sigmaBS[1][0][0]=0.000258933;
    _sigmaBS[1][0][1]=1.28387;
    _sigmaBS[1][0][2]=1668.71;
    _sigmaBS[1][0][3]=730.716;
    _sigmaBM[1][0][0]=0.00121506;
    _sigmaBM[1][0][1]=0.938541;
    _sigmaBM[1][0][2]=9003.57;
    _sigmaBM[1][0][3]=288.897;
    _sigmaR9[1][0][0]=1.01207;
    _sigmaR9[1][0][1]=-816.244;
    _sigmaR9[1][0][2]=-16283.8;
    _sigmaR9[1][0][3]=0;
    
    _meanScale[1][1][0]=0.324634;
    _meanScale[1][1][1]=9.48206e-05;
    _meanScale[1][1][2]=1.0e-12;
    _meanScale[1][1][3]=1.0e-12;
    _meanAT[1][1][0]=0.0;
    _meanAT[1][1][1]=0.0;
    _meanAT[1][1][2]=0.0;
    _meanAT[1][1][3]=0.0;
    _meanAC[1][1][0]=-0.00158311;
    _meanAC[1][1][1]=0.0106161;
    _meanAC[1][1][2]=338.964;
    _meanAC[1][1][3]=797.172;
    _meanAS[1][1][0]=-0.00960269;
    _meanAS[1][1][1]=-0.00496491;
    _meanAS[1][1][2]=934.472;
    _meanAS[1][1][3]=8.32667e-16;
    _meanAM[1][1][0]=-0.00219814;
    _meanAM[1][1][1]=0.653906;
    _meanAM[1][1][2]=0.0949848;
    _meanAM[1][1][3]=0.0977831;
    _meanBT[1][1][0]=0.0;
    _meanBT[1][1][1]=0.0;
    _meanBT[1][1][2]=0.0;
    _meanBT[1][1][3]=0.0;
    _meanBC[1][1][0]=-0.00423472;
    _meanBC[1][1][1]=0.0279695;
    _meanBC[1][1][2]=28073.7;
    _meanBC[1][1][3]=118612;
    _meanBS[1][1][0]=-0.0012476;
    _meanBS[1][1][1]=0.02744;
    _meanBS[1][1][2]=390.697;
    _meanBS[1][1][3]=727.861;
    _meanBM[1][1][0]=-1.36573e-05;
    _meanBM[1][1][1]=0.0667504;
    _meanBM[1][1][2]=-80154.4;
    _meanBM[1][1][3]=576.637;
    _meanR9[1][1][0]=0.113317;
    _meanR9[1][1][1]=0.0142669;
    _meanR9[1][1][2]=-0.125721;
    _meanR9[1][1][3]=0;

    _sigmaScale[1][1][0]=0.471767;
    _sigmaScale[1][1][1]=0.211196;
    _sigmaScale[1][1][2]=0.0240124;
    _sigmaScale[1][1][3]=0;
    _sigmaAT[1][1][0]=0.404395;
    _sigmaAT[1][1][1]=0;
    _sigmaAT[1][1][2]=0;
    _sigmaAT[1][1][3]=0;
    _sigmaAC[1][1][0]=0.00173151;
    _sigmaAC[1][1][1]=-0.479291;
    _sigmaAC[1][1][2]=11583.5;
    _sigmaAC[1][1][3]=-7e+09;
    _sigmaAS[1][1][0]=0.000450387;
    _sigmaAS[1][1][1]=0.662978;
    _sigmaAS[1][1][2]=924.051;
    _sigmaAS[1][1][3]=448.417;
    _sigmaAM[1][1][0]=0.00335603;
    _sigmaAM[1][1][1]=0.648407;
    _sigmaAM[1][1][2]=134.672;
    _sigmaAM[1][1][3]=27.4139;
    _sigmaBT[1][1][0]=0.602402;
    _sigmaBT[1][1][1]=0;
    _sigmaBT[1][1][2]=0;
    _sigmaBT[1][1][3]=0;
    _sigmaBC[1][1][0]=-0.00256192;
    _sigmaBC[1][1][1]=2.01276;
    _sigmaBC[1][1][2]=114558;
    _sigmaBC[1][1][3]=2.15421e+06;
    _sigmaBS[1][1][0]=0.00151576;
    _sigmaBS[1][1][1]=0.359084;
    _sigmaBS[1][1][2]=329.414;
    _sigmaBS[1][1][3]=154.509;
    _sigmaBM[1][1][0]=-0.0452587;
    _sigmaBM[1][1][1]=1.26253;
    _sigmaBM[1][1][2]=1.9e+09;
    _sigmaBM[1][1][3]=1058.76;
    _sigmaR9[1][1][0]=4.59667;
    _sigmaR9[1][1][1]=-5.14404;
    _sigmaR9[1][1][2]=0;
    _sigmaR9[1][1][3]=0;

    _initialised=true;
  }

  if(s=="3_11") {
    _meanScale[0][0][0]=0.994363;
    _meanScale[0][0][1]=4.84904e-07;
    _meanScale[0][0][2]=1.54475e-05;
    _meanScale[0][0][3]=-0.103309;
    _meanAT[0][0][0]=0.0;
    _meanAT[0][0][1]=0.0;
    _meanAT[0][0][2]=0.0;
    _meanAT[0][0][3]=0.0;
    _meanAC[0][0][0]=-0.00360057;
    _meanAC[0][0][1]=0.00970858;
    _meanAC[0][0][2]=409.406;
    _meanAC[0][0][3]=527.952;
    _meanAS[0][0][0]=-1.1;
    _meanAS[0][0][1]=0.00135995;
    _meanAS[0][0][2]=295.712;
    _meanAS[0][0][3]=5.13202e+07;
    _meanAM[0][0][0]=-0.00129854;
    _meanAM[0][0][1]=0.151466;
    _meanAM[0][0][2]=261.828;
    _meanAM[0][0][3]=214.662;
    _meanBT[0][0][0]=0.0;
    _meanBT[0][0][1]=0.0;
    _meanBT[0][0][2]=0.0;
    _meanBT[0][0][3]=0.0;
    _meanBC[0][0][0]=-0.00286864;
    _meanBC[0][0][1]=0.0114118;
    _meanBC[0][0][2]=563.962;
    _meanBC[0][0][3]=412.922;
    _meanBS[0][0][0]=-0.00210996;
    _meanBS[0][0][1]=0.00327867;
    _meanBS[0][0][2]=23.617;
    _meanBS[0][0][3]=1018.45;
    _meanBM[0][0][0]=-0.002287;
    _meanBM[0][0][1]=0.0848984;
    _meanBM[0][0][2]=235.575;
    _meanBM[0][0][3]=260.773;
    _meanR9[0][0][0]=0.951724;
    _meanR9[0][0][1]=23.7181;
    _meanR9[0][0][2]=177.34;
    _meanR9[0][0][3]=0;

    _sigmaScale[0][0][0]=0.187578;
    _sigmaScale[0][0][1]=-0.000901045;
    _sigmaScale[0][0][2]=0.00673186;
    _sigmaScale[0][0][3]=0;
    _sigmaAT[0][0][0]=0.183777;
    _sigmaAT[0][0][1]=0;
    _sigmaAT[0][0][2]=0;
    _sigmaAT[0][0][3]=0;
    _sigmaAC[0][0][0]=-0.00430202;
    _sigmaAC[0][0][1]=0.122501;
    _sigmaAC[0][0][2]=51.9772;
    _sigmaAC[0][0][3]=-3e+17;
    _sigmaAS[0][0][0]=0;
    _sigmaAS[0][0][1]=0;
    _sigmaAS[0][0][2]=0;
    _sigmaAS[0][0][3]=0;
    _sigmaAM[0][0][0]=0.00101883;
    _sigmaAM[0][0][1]=11.2009;
    _sigmaAM[0][0][2]=593.111;
    _sigmaAM[0][0][3]=345.433;
    _sigmaBT[0][0][0]=-6.02356e-05;
    _sigmaBT[0][0][1]=6.99896;
    _sigmaBT[0][0][2]=235.996;
    _sigmaBT[0][0][3]=196;
    _sigmaBC[0][0][0]=-0.00282254;
    _sigmaBC[0][0][1]=0.18764;
    _sigmaBC[0][0][2]=509.825;
    _sigmaBC[0][0][3]=1400.14;
    _sigmaBS[0][0][0]=0;
    _sigmaBS[0][0][1]=0;
    _sigmaBS[0][0][2]=0;
    _sigmaBS[0][0][3]=0;
    _sigmaBM[0][0][0]=-0.00252199;
    _sigmaBM[0][0][1]=39.1544;
    _sigmaBM[0][0][2]=612.481;
    _sigmaBM[0][0][3]=905.994;
    _sigmaR9[0][0][0]=0.95608;
    _sigmaR9[0][0][1]=2203.31;
    _sigmaR9[0][0][2]=-22454.2;
    _sigmaR9[0][0][3]=0;

    _meanScale[0][1][0]=0.889415;
    _meanScale[0][1][1]=1.21788e-05;
    _meanScale[0][1][2]=-4.3438e-06;
    _meanScale[0][1][3]=-0.629968;
    _meanAT[0][1][0]=0.0;
    _meanAT[0][1][1]=0.0;
    _meanAT[0][1][2]=0.0;
    _meanAT[0][1][3]=0.0;
    _meanAC[0][1][0]=-0.00313701;
    _meanAC[0][1][1]=0.0227998;
    _meanAC[0][1][2]=128.653;
    _meanAC[0][1][3]=234.333;
    _meanAS[0][1][0]=0.0346198;
    _meanAS[0][1][1]=-0.00261336;
    _meanAS[0][1][2]=177.983;
    _meanAS[0][1][3]=1.19839e+14;
    _meanAM[0][1][0]=-0.00100745;
    _meanAM[0][1][1]=0.264247;
    _meanAM[0][1][2]=337.255;
    _meanAM[0][1][3]=251.454;
    _meanBT[0][1][0]=0.0;
    _meanBT[0][1][1]=0.0;
    _meanBT[0][1][2]=0.0;
    _meanBT[0][1][3]=0.0;
    _meanBC[0][1][0]=-0.00397794;
    _meanBC[0][1][1]=0.00219079;
    _meanBC[0][1][2]=176.842;
    _meanBC[0][1][3]=450.29;
    _meanBS[0][1][0]=-2e+07;
    _meanBS[0][1][1]=0.0957598;
    _meanBS[0][1][2]=-8.88573e-27;
    _meanBS[0][1][3]=11442.2;
    _meanBM[0][1][0]=-0.00366315;
    _meanBM[0][1][1]=0.0622186;
    _meanBM[0][1][2]=94.5155;
    _meanBM[0][1][3]=126.404;
    _meanR9[0][1][0]=0.00636789;
    _meanR9[0][1][1]=0.000336062;
    _meanR9[0][1][2]=-0.0092699;
    _meanR9[0][1][3]=0;

    _sigmaScale[0][1][0]=0.685096;
    _sigmaScale[0][1][1]=0.129065;
    _sigmaScale[0][1][2]=-0.00212486;
    _sigmaScale[0][1][3]=0;
    _sigmaAT[0][1][0]=0.898865;
    _sigmaAT[0][1][1]=0;
    _sigmaAT[0][1][2]=0;
    _sigmaAT[0][1][3]=0;
    _sigmaAC[0][1][0]=-0.00492979;
    _sigmaAC[0][1][1]=-1.20123;
    _sigmaAC[0][1][2]=2.89231;
    _sigmaAC[0][1][3]=18.2059;
    _sigmaAS[0][1][0]=0;
    _sigmaAS[0][1][1]=0;
    _sigmaAS[0][1][2]=0;
    _sigmaAS[0][1][3]=0;
    _sigmaAM[0][1][0]=-0.000727825;
    _sigmaAM[0][1][1]=8.42395;
    _sigmaAM[0][1][2]=512.032;
    _sigmaAM[0][1][3]=415.962;
    _sigmaBT[0][1][0]=-0.0336364;
    _sigmaBT[0][1][1]=2.45182;
    _sigmaBT[0][1][2]=-0.284353;
    _sigmaBT[0][1][3]=-0.31679;
    _sigmaBC[0][1][0]=0.00510553;
    _sigmaBC[0][1][1]=-0.953869;
    _sigmaBC[0][1][2]=113872;
    _sigmaBC[0][1][3]=1.35966e+09;
    _sigmaBS[0][1][0]=0;
    _sigmaBS[0][1][1]=0;
    _sigmaBS[0][1][2]=0;
    _sigmaBS[0][1][3]=0;
    _sigmaBM[0][1][0]=-0.0034071;
    _sigmaBM[0][1][1]=4.19719;
    _sigmaBM[0][1][2]=128.952;
    _sigmaBM[0][1][3]=180.604;
    _sigmaR9[0][1][0]=-3.38988;
    _sigmaR9[0][1][1]=2.0714;
    _sigmaR9[0][1][2]=0;
    _sigmaR9[0][1][3]=0;

    _meanScale[1][0][0]=1.0009;
    _meanScale[1][0][1]=-4.79805e-06;
    _meanScale[1][0][2]=3.34625e-05;
    _meanScale[1][0][3]=-0.194267;
    _meanAT[1][0][0]=0.0;
    _meanAT[1][0][1]=0.0;
    _meanAT[1][0][2]=0.0;
    _meanAT[1][0][3]=0.0;
    _meanAC[1][0][0]=-0.000177563;
    _meanAC[1][0][1]=0.0122839;
    _meanAC[1][0][2]=1798.92;
    _meanAC[1][0][3]=776.856;
    _meanAS[1][0][0]=-0.000533039;
    _meanAS[1][0][1]=0.0642604;
    _meanAS[1][0][2]=969.596;
    _meanAS[1][0][3]=1004.15;
    _meanAM[1][0][0]=0.000163185;
    _meanAM[1][0][1]=0.085936;
    _meanAM[1][0][2]=1593.17;
    _meanAM[1][0][3]=681.623;
    _meanBT[1][0][0]=0.0;
    _meanBT[1][0][1]=0.0;
    _meanBT[1][0][2]=0.0;
    _meanBT[1][0][3]=0.0;
    _meanBC[1][0][0]=-0.000518186;
    _meanBC[1][0][1]=0.0121868;
    _meanBC[1][0][2]=1112.53;
    _meanBC[1][0][3]=933.281;
    _meanBS[1][0][0]=-0.000750734;
    _meanBS[1][0][1]=0.03859;
    _meanBS[1][0][2]=547.579;
    _meanBS[1][0][3]=775.887;
    _meanBM[1][0][0]=-0.190395;
    _meanBM[1][0][1]=-0.00362647;
    _meanBM[1][0][2]=5.25687;
    _meanBM[1][0][3]=-2.8e+08;
    _meanR9[1][0][0]=0.972346;
    _meanR9[1][0][1]=53.9185;
    _meanR9[1][0][2]=1354.5;
    _meanR9[1][0][3]=0;

    _sigmaScale[1][0][0]=0.348019;
    _sigmaScale[1][0][1]=-6.43731e-11;
    _sigmaScale[1][0][2]=0.0158647;
    _sigmaScale[1][0][3]=0;
    _sigmaAT[1][0][0]=0.215239;
    _sigmaAT[1][0][1]=0;
    _sigmaAT[1][0][2]=0;
    _sigmaAT[1][0][3]=0;
    _sigmaAC[1][0][0]=-0.00492298;
    _sigmaAC[1][0][1]=-3.40058;
    _sigmaAC[1][0][2]=17263.9;
    _sigmaAC[1][0][3]=2.6e+11;
    _sigmaAS[1][0][0]=-0.000237998;
    _sigmaAS[1][0][1]=3.0258;
    _sigmaAS[1][0][2]=1811.25;
    _sigmaAS[1][0][3]=1846.79;
    _sigmaAM[1][0][0]=0.0210134;
    _sigmaAM[1][0][1]=0.328359;
    _sigmaAM[1][0][2]=22.49;
    _sigmaAM[1][0][3]=14.5021;
    _sigmaBT[1][0][0]=-0.495072;
    _sigmaBT[1][0][1]=0;
    _sigmaBT[1][0][2]=0;
    _sigmaBT[1][0][3]=0;
    _sigmaBC[1][0][0]=-0.00265007;
    _sigmaBC[1][0][1]=0.970549;
    _sigmaBC[1][0][2]=-6.89119e+07;
    _sigmaBC[1][0][3]=180110;
    _sigmaBS[1][0][0]=0.00045833;
    _sigmaBS[1][0][1]=2.16342;
    _sigmaBS[1][0][2]=3582.4;
    _sigmaBS[1][0][3]=1100.36;
    _sigmaBM[1][0][0]=0.00188871;
    _sigmaBM[1][0][1]=1.66177;
    _sigmaBM[1][0][2]=3.2e+08;
    _sigmaBM[1][0][3]=2163.81;
    _sigmaR9[1][0][0]=-220.415;
    _sigmaR9[1][0][1]=5.19136e-08;
    _sigmaR9[1][0][2]=3.04028e-10;
    _sigmaR9[1][0][3]=0;

    _meanScale[1][1][0]=0.338011;
    _meanScale[1][1][1]=9.47815e-05;
    _meanScale[1][1][2]=-0.000238735;
    _meanScale[1][1][3]=-0.846414;
    _meanAT[1][1][0]=0.0;
    _meanAT[1][1][1]=0.0;
    _meanAT[1][1][2]=0.0;
    _meanAT[1][1][3]=0.0;
    _meanAC[1][1][0]=-0.00125367;
    _meanAC[1][1][1]=0.013324;
    _meanAC[1][1][2]=203.988;
    _meanAC[1][1][3]=431.951;
    _meanAS[1][1][0]=0.000282607;
    _meanAS[1][1][1]=0.0307431;
    _meanAS[1][1][2]=343.509;
    _meanAS[1][1][3]=274.957;
    _meanAM[1][1][0]=0.0020258;
    _meanAM[1][1][1]=0.643913;
    _meanAM[1][1][2]=0.0693877;
    _meanAM[1][1][3]=0.0816029;
    _meanBT[1][1][0]=0.0;
    _meanBT[1][1][1]=0.0;
    _meanBT[1][1][2]=0.0;
    _meanBT[1][1][3]=0.0;
    _meanBC[1][1][0]=-0.00513833;
    _meanBC[1][1][1]=5.94424e+08;
    _meanBC[1][1][2]=-62814.9;
    _meanBC[1][1][3]=118612;
    _meanBS[1][1][0]=-0.00152129;
    _meanBS[1][1][1]=0.0234694;
    _meanBS[1][1][2]=186.483;
    _meanBS[1][1][3]=754.201;
    _meanBM[1][1][0]=-0.000404987;
    _meanBM[1][1][1]=0.156384;
    _meanBM[1][1][2]=-1.7e+08;
    _meanBM[1][1][3]=1793.83;
    _meanR9[1][1][0]=0.0645278;
    _meanR9[1][1][1]=0.161614;
    _meanR9[1][1][2]=-0.215822;
    _meanR9[1][1][3]=0;

    _sigmaScale[1][1][0]=1.07376;
    _sigmaScale[1][1][1]=7.47238e-13;
    _sigmaScale[1][1][2]=0.0289594;
    _sigmaScale[1][1][3]=0;
    _sigmaAT[1][1][0]=-0.520907;
    _sigmaAT[1][1][1]=0;
    _sigmaAT[1][1][2]=0;
    _sigmaAT[1][1][3]=0;
    _sigmaAC[1][1][0]=0.00165941;
    _sigmaAC[1][1][1]=-0.351422;
    _sigmaAC[1][1][2]=8968.94;
    _sigmaAC[1][1][3]=-7e+09;
    _sigmaAS[1][1][0]=0.000490279;
    _sigmaAS[1][1][1]=0.554531;
    _sigmaAS[1][1][2]=469.111;
    _sigmaAS[1][1][3]=457.541;
    _sigmaAM[1][1][0]=0.00102079;
    _sigmaAM[1][1][1]=0.628055;
    _sigmaAM[1][1][2]=53.9452;
    _sigmaAM[1][1][3]=72.911;
    _sigmaBT[1][1][0]=-0.461542;
    _sigmaBT[1][1][1]=0;
    _sigmaBT[1][1][2]=0;
    _sigmaBT[1][1][3]=0;
    _sigmaBC[1][1][0]=-0.00219303;
    _sigmaBC[1][1][1]=0.874327;
    _sigmaBC[1][1][2]=71353.2;
    _sigmaBC[1][1][3]=2.09924e+08;
    _sigmaBS[1][1][0]=0.00104021;
    _sigmaBS[1][1][1]=0.236098;
    _sigmaBS[1][1][2]=482.954;
    _sigmaBS[1][1][3]=191.984;
    _sigmaBM[1][1][0]=-0.000116086;
    _sigmaBM[1][1][1]=2.4438;
    _sigmaBM[1][1][2]=1.9e+09;
    _sigmaBM[1][1][3]=-700.271;
    _sigmaR9[1][1][0]=4.59374;
    _sigmaR9[1][1][1]=-5.06202;
    _sigmaR9[1][1][2]=0;
    _sigmaR9[1][1][3]=0;

    _initialised=true;
  }

  if(s=="4_2") {
    _meanScale[0][0][0]=0.995941423;
    _meanScale[0][0][1]=-1.41986304e-05;
    _meanScale[0][0][2]=3.66129541e-05;
    _meanScale[0][0][3]=-0.0774047233;
    _meanAT[0][0][0]=0.000720281545;
    _meanAT[0][0][1]=0;
    _meanAT[0][0][2]=0;
    _meanAT[0][0][3]=0;
    _meanAC[0][0][0]=-0.00344862444;
    _meanAC[0][0][1]=0.0101395802;
    _meanAC[0][0][2]=466.112225;
    _meanAC[0][0][3]=507.628173;
    _meanAS[0][0][0]=0;
    _meanAS[0][0][1]=0;
    _meanAS[0][0][2]=0;
    _meanAS[0][0][3]=0;
    _meanAM[0][0][0]=-0.000871553792;
    _meanAM[0][0][1]=0.141419889;
    _meanAM[0][0][2]=281.104504;
    _meanAM[0][0][3]=195.875679;
    _meanBT[0][0][0]=0;
    _meanBT[0][0][1]=0.026344491;
    _meanBT[0][0][2]=-104.20518;
    _meanBT[0][0][3]=-176099;
    _meanBC[0][0][0]=-0.00272095949;
    _meanBC[0][0][1]=0.012411788;
    _meanBC[0][0][2]=587.318903;
    _meanBC[0][0][3]=381.415059;
    _meanBS[0][0][0]=-0.00201265145;
    _meanBS[0][0][1]=0.00372948657;
    _meanBS[0][0][2]=41.2773112;
    _meanBS[0][0][3]=748.890936;
    _meanBM[0][0][0]=-0.00168471013;
    _meanBM[0][0][1]=0.0685484442;
    _meanBM[0][0][2]=217.983503;
    _meanBM[0][0][3]=207.660928;
    _meanR9[0][0][0]=0.946581139;
    _meanR9[0][0][1]=20.6034189;
    _meanR9[0][0][2]=187.28856;
    _meanR9[0][0][3]=0;

    _sigmaScale[0][0][0]=0.206349443;
    _sigmaScale[0][0][1]=0.0206592338;
    _sigmaScale[0][0][2]=0.00653752299;
    _sigmaScale[0][0][3]=0;
    _sigmaAT[0][0][0]=0.178629422;
    _sigmaAT[0][0][1]=0;
    _sigmaAT[0][0][2]=0;
    _sigmaAT[0][0][3]=0;
    _sigmaAC[0][0][0]=-0.00335501889;
    _sigmaAC[0][0][1]=0.0997921532;
    _sigmaAC[0][0][2]=93.6397821;
    _sigmaAC[0][0][3]=1519.43272;
    _sigmaAS[0][0][0]=0;
    _sigmaAS[0][0][1]=0;
    _sigmaAS[0][0][2]=0;
    _sigmaAS[0][0][3]=0;
    _sigmaAM[0][0][0]=0.000927325527;
    _sigmaAM[0][0][1]=10.2678389;
    _sigmaAM[0][0][2]=619.975988;
    _sigmaAM[0][0][3]=285.190815;
    _sigmaBT[0][0][0]=0;
    _sigmaBT[0][0][1]=0.895041707;
    _sigmaBT[0][0][2]=94.6834192;
    _sigmaBT[0][0][3]=62.3012502;
    _sigmaBC[0][0][0]=-0.00169896783;
    _sigmaBC[0][0][1]=0.323973706;
    _sigmaBC[0][0][2]=1234.03309;
    _sigmaBC[0][0][3]=907.352988;
    _sigmaBS[0][0][0]=0;
    _sigmaBS[0][0][1]=0;
    _sigmaBS[0][0][2]=0;
    _sigmaBS[0][0][3]=0;
    _sigmaBM[0][0][0]=-0.00249508825;
    _sigmaBM[0][0][1]=57.8982306;
    _sigmaBM[0][0][2]=665.068952;
    _sigmaBM[0][0][3]=1075.1094;
    _sigmaR9[0][0][0]=0.952890416;
    _sigmaR9[0][0][1]=1958.37946;
    _sigmaR9[0][0][2]=21612.0219;
    _sigmaR9[0][0][3]=0;

    _meanScale[0][1][0]=0.982680412;
    _meanScale[0][1][1]=3.13860176e-05;
    _meanScale[0][1][2]=-2.89107109e-05;
    _meanScale[0][1][3]=-0.458678502;
    _meanAT[0][1][0]=-0.00204222443;
    _meanAT[0][1][1]=0;
    _meanAT[0][1][2]=0;
    _meanAT[0][1][3]=0;
    _meanAC[0][1][0]=-0.00329797061;
    _meanAC[0][1][1]=0.0212879256;
    _meanAC[0][1][2]=135.879912;
    _meanAC[0][1][3]=238.247576;
    _meanAS[0][1][0]=0;
    _meanAS[0][1][1]=0;
    _meanAS[0][1][2]=0;
    _meanAS[0][1][3]=0;
    _meanAM[0][1][0]=-0.000512006976;
    _meanAM[0][1][1]=0.124281288;
    _meanAM[0][1][2]=480.326634;
    _meanAM[0][1][3]=286.165783;
    _meanBT[0][1][0]=0;
    _meanBT[0][1][1]=0.204384889;
    _meanBT[0][1][2]=303.764745;
    _meanBT[0][1][3]=408.14741;
    _meanBC[0][1][0]=-0.0035698745;
    _meanBC[0][1][1]=0.00402323151;
    _meanBC[0][1][2]=980.296598;
    _meanBC[0][1][3]=869.711616;
    _meanBS[0][1][0]=0;
    _meanBS[0][1][1]=0;
    _meanBS[0][1][2]=0;
    _meanBS[0][1][3]=0;
    _meanBM[0][1][0]=-0.00321305828;
    _meanBM[0][1][1]=0.0454848819;
    _meanBM[0][1][2]=147.827487;
    _meanBM[0][1][3]=227.625382;
    _meanR9[0][1][0]=0.0253777359;
    _meanR9[0][1][1]=-0.0420810898;
    _meanR9[0][1][2]=0.0181966013;
    _meanR9[0][1][3]=0;

    _sigmaScale[0][1][0]=1.53707929;
    _sigmaScale[0][1][1]=0.0946423194;
    _sigmaScale[0][1][2]=-0.00765920151;
    _sigmaScale[0][1][3]=0;
    _sigmaAT[0][1][0]=0.808880052;
    _sigmaAT[0][1][1]=0;
    _sigmaAT[0][1][2]=0;
    _sigmaAT[0][1][3]=0;
    _sigmaAC[0][1][0]=-0.00195542375;
    _sigmaAC[0][1][1]=-2.09949949;
    _sigmaAC[0][1][2]=4.30292193;
    _sigmaAC[0][1][3]=5.09475964;
    _sigmaAS[0][1][0]=0;
    _sigmaAS[0][1][1]=0;
    _sigmaAS[0][1][2]=0;
    _sigmaAS[0][1][3]=0;
    _sigmaAM[0][1][0]=-0.00105652021;
    _sigmaAM[0][1][1]=5.83420851;
    _sigmaAM[0][1][2]=506.986527;
    _sigmaAM[0][1][3]=468.330744;
    _sigmaBT[0][1][0]=0;
    _sigmaBT[0][1][1]=2.83411417;
    _sigmaBT[0][1][2]=-0.211242292;
    _sigmaBT[0][1][3]=-0.198231087;
    _sigmaBC[0][1][0]=0.00580038243;
    _sigmaBC[0][1][1]=0.165505659;
    _sigmaBC[0][1][2]=4133.45418;
    _sigmaBC[0][1][3]=375000000;
    _sigmaBS[0][1][0]=0;
    _sigmaBS[0][1][1]=0;
    _sigmaBS[0][1][2]=0;
    _sigmaBS[0][1][3]=0;
    _sigmaBM[0][1][0]=-0.00269993666;
    _sigmaBM[0][1][1]=3.42390459;
    _sigmaBM[0][1][2]=171.300481;
    _sigmaBM[0][1][3]=284.718025;
    _sigmaR9[0][1][0]=-3.75255938;
    _sigmaR9[0][1][1]=4.3849733;
    _sigmaR9[0][1][2]=-1.81745726;
    _sigmaR9[0][1][3]=0;

    _meanScale[1][0][0]=0.990082016;
    _meanScale[1][0][1]=-3.75802712e-06;
    _meanScale[1][0][2]=2.56693516e-05;
    _meanScale[1][0][3]=-0.0492813428;
    _meanAT[1][0][0]=0.072352478;
    _meanAT[1][0][1]=0;
    _meanAT[1][0][2]=0;
    _meanAT[1][0][3]=0;
    _meanAC[1][0][0]=-0.0002936899;
    _meanAC[1][0][1]=0.0160546814;
    _meanAC[1][0][2]=1183.48593;
    _meanAC[1][0][3]=761.29774;
    _meanAS[1][0][0]=-0.000462243216;
    _meanAS[1][0][1]=0.0795658256;
    _meanAS[1][0][2]=887.080242;
    _meanAS[1][0][3]=1067.72442;
    _meanAM[1][0][0]=0.000354495505;
    _meanAM[1][0][1]=0.516700576;
    _meanAM[1][0][2]=4376.14811;
    _meanAM[1][0][3]=2093.33478;
    _meanBT[1][0][0]=0.077752944;
    _meanBT[1][0][1]=0;
    _meanBT[1][0][2]=0;
    _meanBT[1][0][3]=0;
    _meanBC[1][0][0]=-0.000411367107;
    _meanBC[1][0][1]=0.0161135906;
    _meanBC[1][0][2]=1414.07982;
    _meanBC[1][0][3]=951.556042;
    _meanBS[1][0][0]=8.51070829e-05;
    _meanBS[1][0][1]=0.0699037982;
    _meanBS[1][0][2]=1565.72963;
    _meanBS[1][0][3]=841.509573;
    _meanBM[1][0][0]=-0.00252281385;
    _meanBM[1][0][1]=0.00600665031;
    _meanBM[1][0][2]=268.761304;
    _meanBM[1][0][3]=46.5945865;
    _meanR9[1][0][0]=0.964231565;
    _meanR9[1][0][1]=30.1631606;
    _meanR9[1][0][2]=414.510458;
    _meanR9[1][0][3]=0;

    _sigmaScale[1][0][0]=0.218991853;
    _sigmaScale[1][0][1]=6.93889e-18;
    _sigmaScale[1][0][2]=0.00939222285;
    _sigmaScale[1][0][3]=0;
    _sigmaAT[1][0][0]=1.61339852;
    _sigmaAT[1][0][1]=0;
    _sigmaAT[1][0][2]=0;
    _sigmaAT[1][0][3]=0;
    _sigmaAC[1][0][0]=0.00019476922;
    _sigmaAC[1][0][1]=0.697650974;
    _sigmaAC[1][0][2]=-0.000125668382;
    _sigmaAC[1][0][3]=12.8659982;
    _sigmaAS[1][0][0]=-1.68218147e-05;
    _sigmaAS[1][0][1]=6.57794255;
    _sigmaAS[1][0][2]=1555.93015;
    _sigmaAS[1][0][3]=1401.542;
    _sigmaAM[1][0][0]=0.0570038229;
    _sigmaAM[1][0][1]=0.633551691;
    _sigmaAM[1][0][2]=9.59639e+11;
    _sigmaAM[1][0][3]=16.4637695;
    _sigmaBT[1][0][0]=-0.0591443023;
    _sigmaBT[1][0][1]=0;
    _sigmaBT[1][0][2]=0;
    _sigmaBT[1][0][3]=0;
    _sigmaBC[1][0][0]=-0.00320070019;
    _sigmaBC[1][0][1]=25.5502578;
    _sigmaBC[1][0][2]=7.49509e+12;
    _sigmaBC[1][0][3]=3798165.72;
    _sigmaBS[1][0][0]=9.63685051e-05;
    _sigmaBS[1][0][1]=6.91673581;
    _sigmaBS[1][0][2]=2447.68053;
    _sigmaBS[1][0][3]=1721.11327;
    _sigmaBM[1][0][0]=0.00148006;
    _sigmaBM[1][0][1]=28;
    _sigmaBM[1][0][2]=5400000;
    _sigmaBM[1][0][3]=-9000000;
    _sigmaR9[1][0][0]=187.987786;
    _sigmaR9[1][0][1]=-1.91777372e-07;
    _sigmaR9[1][0][2]=8.29820105e-09;
    _sigmaR9[1][0][3]=0;

    _meanScale[1][1][0]=0.331585644;
    _meanScale[1][1][1]=-4.97323079e-05;
    _meanScale[1][1][2]=0.000208912195;
    _meanScale[1][1][3]=-1.36032052;
    _meanAT[1][1][0]=-0.0640673292;
    _meanAT[1][1][1]=0;
    _meanAT[1][1][2]=0;
    _meanAT[1][1][3]=0;
    _meanAC[1][1][0]=-0.00129027954;
    _meanAC[1][1][1]=0.00733510902;
    _meanAC[1][1][2]=182.714706;
    _meanAC[1][1][3]=621.652554;
    _meanAS[1][1][0]=-0.000490574173;
    _meanAS[1][1][1]=0.0308208884;
    _meanAS[1][1][2]=385.372647;
    _meanAS[1][1][3]=492.313289;
    _meanAM[1][1][0]=-0.0064828927;
    _meanAM[1][1][1]=0.649443452;
    _meanAM[1][1][2]=0.0573092773;
    _meanAM[1][1][3]=0.0743069;
    _meanBT[1][1][0]=-0.147343956;
    _meanBT[1][1][1]=0;
    _meanBT[1][1][2]=0;
    _meanBT[1][1][3]=0;
    _meanBC[1][1][0]=-0.00503351921;
    _meanBC[1][1][1]=-57691.5085;
    _meanBC[1][1][2]=46202.9758;
    _meanBC[1][1][3]=118612;
    _meanBS[1][1][0]=-0.000793147706;
    _meanBS[1][1][1]=0.0238305184;
    _meanBS[1][1][2]=402.215233;
    _meanBS[1][1][3]=455.848092;
    _meanBM[1][1][0]=0.000434549102;
    _meanBM[1][1][1]=0.0443539812;
    _meanBM[1][1][2]=-39970930.5;
    _meanBM[1][1][3]=-635.815445;
    _meanR9[1][1][0]=-0.411370898;
    _meanR9[1][1][1]=1.30133082;
    _meanR9[1][1][2]=-0.890618718;
    _meanR9[1][1][3]=0;

    _sigmaScale[1][1][0]=1.49352299;
    _sigmaScale[1][1][1]=1.38778e-17;
    _sigmaScale[1][1][2]=0.0248352105;
    _sigmaScale[1][1][3]=0;
    _sigmaAT[1][1][0]=-1.18239629;
    _sigmaAT[1][1][1]=0;
    _sigmaAT[1][1][2]=0;
    _sigmaAT[1][1][3]=0;
    _sigmaAC[1][1][0]=0.00155030534;
    _sigmaAC[1][1][1]=-0.673931391;
    _sigmaAC[1][1][2]=134075.829;
    _sigmaAC[1][1][3]=-7e+09;
    _sigmaAS[1][1][0]=6.95848091e-05;
    _sigmaAS[1][1][1]=0.522471203;
    _sigmaAS[1][1][2]=463.305497;
    _sigmaAS[1][1][3]=1159.49992;
    _sigmaAM[1][1][0]=-0.00509006951;
    _sigmaAM[1][1][1]=0.945276887;
    _sigmaAM[1][1][2]=46.4072512;
    _sigmaAM[1][1][3]=7.11474e+12;
    _sigmaBT[1][1][0]=-1.59480683;
    _sigmaBT[1][1][1]=0;
    _sigmaBT[1][1][2]=0;
    _sigmaBT[1][1][3]=0;
    _sigmaBC[1][1][0]=-0.00202302997;
    _sigmaBC[1][1][1]=15.4301057;
    _sigmaBC[1][1][2]=-33315545.5;
    _sigmaBC[1][1][3]=-6e+09;
    _sigmaBS[1][1][0]=0.00271126099;
    _sigmaBS[1][1][1]=0.325669289;
    _sigmaBS[1][1][2]=2322.66097;
    _sigmaBS[1][1][3]=298.692034;
    _sigmaBM[1][1][0]=-0.0454765849;
    _sigmaBM[1][1][1]=6.81541098;
    _sigmaBM[1][1][2]=1.9e+09;
    _sigmaBM[1][1][3]=-26353.4449;
    _sigmaR9[1][1][0]=41.1074567;
    _sigmaR9[1][1][1]=-86.9595346;
    _sigmaR9[1][1][2]=45.7818889;
    _sigmaR9[1][1][3]=0;

    _initialised=true;
  }

  if(s=="4_2e") {
    _meanScale[0][0][0]=1.03294629;
    _meanScale[0][0][1]=-0.000210626517;
    _meanScale[0][0][2]=0.000268568795;
    _meanScale[0][0][3]=0.338053561;
    _meanAT[0][0][0]=0.0200811135;
    _meanAT[0][0][1]=0;
    _meanAT[0][0][2]=0;
    _meanAT[0][0][3]=0;
    _meanAC[0][0][0]=-0.00326696352;
    _meanAC[0][0][1]=0.010765809;
    _meanAC[0][0][2]=513.763513;
    _meanAC[0][0][3]=546.438243;
    _meanAS[0][0][0]=0;
    _meanAS[0][0][1]=0;
    _meanAS[0][0][2]=0;
    _meanAS[0][0][3]=0;
    _meanAM[0][0][0]=-0.00135522301;
    _meanAM[0][0][1]=0.166490439;
    _meanAM[0][0][2]=278.324187;
    _meanAM[0][0][3]=245.998361;
    _meanBT[0][0][0]=0;
    _meanBT[0][0][1]=0;
    _meanBT[0][0][2]=0;
    _meanBT[0][0][3]=0;
    _meanBC[0][0][0]=-0.00332906015;
    _meanBC[0][0][1]=0.00792585358;
    _meanBC[0][0][2]=514.766605;
    _meanBC[0][0][3]=488.870257;
    _meanBS[0][0][0]=-0.00199241828;
    _meanBS[0][0][1]=0.0037942702;
    _meanBS[0][0][2]=29.9438726;
    _meanBS[0][0][3]=1077.1644;
    _meanBM[0][0][0]=-0.00159080193;
    _meanBM[0][0][1]=0.107998922;
    _meanBM[0][0][2]=229.934523;
    _meanBM[0][0][3]=231.786153;
    _meanR9[0][0][0]=0.857844414;
    _meanR9[0][0][1]=-16.8494499;
    _meanR9[0][0][2]=125.493331;
    _meanR9[0][0][3]=0;

    _sigmaScale[0][0][0]=0.392737806;
    _sigmaScale[0][0][1]=0.0353140568;
    _sigmaScale[0][0][2]=-0.00613223131;
    _sigmaScale[0][0][3]=0;
    _sigmaAT[0][0][0]=1.02977565;
    _sigmaAT[0][0][1]=0;
    _sigmaAT[0][0][2]=0;
    _sigmaAT[0][0][3]=0;
    _sigmaAC[0][0][0]=-0.00350109526;
    _sigmaAC[0][0][1]=-0.951103069;
    _sigmaAC[0][0][2]=-54434.4267;
    _sigmaAC[0][0][3]=-3e+17;
    _sigmaAS[0][0][0]=0;
    _sigmaAS[0][0][1]=0;
    _sigmaAS[0][0][2]=0;
    _sigmaAS[0][0][3]=0;
    _sigmaAM[0][0][0]=0.00127749544;
    _sigmaAM[0][0][1]=5.03867192;
    _sigmaAM[0][0][2]=563.047721;
    _sigmaAM[0][0][3]=272.293234;
    _sigmaBT[0][0][0]=0.00480679465;
    _sigmaBT[0][0][1]=7.56230742;
    _sigmaBT[0][0][2]=-33600000;
    _sigmaBT[0][0][3]=-257.677353;
    _sigmaBC[0][0][0]=-0.00169935002;
    _sigmaBC[0][0][1]=2790083.26;
    _sigmaBC[0][0][2]=-97275416.4;
    _sigmaBC[0][0][3]=23710676.7;
    _sigmaBS[0][0][0]=0;
    _sigmaBS[0][0][1]=0;
    _sigmaBS[0][0][2]=0;
    _sigmaBS[0][0][3]=0;
    _sigmaBM[0][0][0]=-0.00194553738;
    _sigmaBM[0][0][1]=7.77713222;
    _sigmaBM[0][0][2]=264.960159;
    _sigmaBM[0][0][3]=363.487107;
    _sigmaR9[0][0][0]=0.952571;
    _sigmaR9[0][0][1]=0;
    _sigmaR9[0][0][2]=0;
    _sigmaR9[0][0][3]=0;

    _meanScale[0][1][0]=0.86164193;
    _meanScale[0][1][1]=-0.0001184458;
    _meanScale[0][1][2]=0.000232979403;
    _meanScale[0][1][3]=0.310305987;
    _meanAT[0][1][0]=0.0103409006;
    _meanAT[0][1][1]=0;
    _meanAT[0][1][2]=0;
    _meanAT[0][1][3]=0;
    _meanAC[0][1][0]=-0.00325081301;
    _meanAC[0][1][1]=0.0208748426;
    _meanAC[0][1][2]=165.245698;
    _meanAC[0][1][3]=292.03632;
    _meanAS[0][1][0]=0.0330004;
    _meanAS[0][1][1]=-148569.764;
    _meanAS[0][1][2]=87999432.1;
    _meanAS[0][1][3]=7787218.96;
    _meanAM[0][1][0]=-0.000867413605;
    _meanAM[0][1][1]=0.10580464;
    _meanAM[0][1][2]=396.92529;
    _meanAM[0][1][3]=263.112883;
    _meanBT[0][1][0]=0;
    _meanBT[0][1][1]=0.216283067;
    _meanBT[0][1][2]=312.543466;
    _meanBT[0][1][3]=463.601293;
    _meanBC[0][1][0]=-0.00505883024;
    _meanBC[0][1][1]=0.00182528255;
    _meanBC[0][1][2]=507.478054;
    _meanBC[0][1][3]=-6837.26736;
    _meanBS[0][1][0]=-166707004;
    _meanBS[0][1][1]=0.0928055999;
    _meanBS[0][1][2]=-5.30004162e-11;
    _meanBS[0][1][3]=11442.2;
    _meanBM[0][1][0]=-5.93998135e-05;
    _meanBM[0][1][1]=0.0096852184;
    _meanBM[0][1][2]=59.8040186;
    _meanBM[0][1][3]=-440000000;
    _meanR9[0][1][0]=0.0716647946;
    _meanR9[0][1][1]=-0.204241803;
    _meanR9[0][1][2]=0.154962477;
    _meanR9[0][1][3]=0;

    _sigmaScale[0][1][0]=0.469123815;
    _sigmaScale[0][1][1]=-0.090283052;
    _sigmaScale[0][1][2]=0.000469934719;
    _sigmaScale[0][1][3]=0;
    _sigmaAT[0][1][0]=1.77629522;
    _sigmaAT[0][1][1]=0;
    _sigmaAT[0][1][2]=0;
    _sigmaAT[0][1][3]=0;
    _sigmaAC[0][1][0]=-0.00636220086;
    _sigmaAC[0][1][1]=-0.781271127;
    _sigmaAC[0][1][2]=4.90734224;
    _sigmaAC[0][1][3]=65.6835127;
    _sigmaAS[0][1][0]=0;
    _sigmaAS[0][1][1]=0;
    _sigmaAS[0][1][2]=0;
    _sigmaAS[0][1][3]=0;
    _sigmaAM[0][1][0]=0.000179292631;
    _sigmaAM[0][1][1]=7.62815501;
    _sigmaAM[0][1][2]=743.55507;
    _sigmaAM[0][1][3]=354.656661;
    _sigmaBT[0][1][0]=-0.0507778073;
    _sigmaBT[0][1][1]=3.00903133;
    _sigmaBT[0][1][2]=-0.526032834;
    _sigmaBT[0][1][3]=-0.630748789;
    _sigmaBC[0][1][0]=0.00490009575;
    _sigmaBC[0][1][1]=-1.53772346;
    _sigmaBC[0][1][2]=553415.545;
    _sigmaBC[0][1][3]=2.36808e+19;
    _sigmaBS[0][1][0]=0;
    _sigmaBS[0][1][1]=0;
    _sigmaBS[0][1][2]=0;
    _sigmaBS[0][1][3]=0;
    _sigmaBM[0][1][0]=-0.00113947453;
    _sigmaBM[0][1][1]=3.74348887;
    _sigmaBM[0][1][2]=91.9478901;
    _sigmaBM[0][1][3]=101.304882;
    _sigmaR9[0][1][0]=-0.261512815;
    _sigmaR9[0][1][1]=-1.69974425;
    _sigmaR9[0][1][2]=0;
    _sigmaR9[0][1][3]=0;

    _meanScale[1][0][0]=0.961072344;
    _meanScale[1][0][1]=8.81367775e-05;
    _meanScale[1][0][2]=-0.000270690177;
    _meanScale[1][0][3]=0.745461418;
    _meanAT[1][0][0]=0.532495533;
    _meanAT[1][0][1]=0;
    _meanAT[1][0][2]=0;
    _meanAT[1][0][3]=0;
    _meanAC[1][0][0]=-0.000539999855;
    _meanAC[1][0][1]=0.0100918811;
    _meanAC[1][0][2]=953.905309;
    _meanAC[1][0][3]=808.944612;
    _meanAS[1][0][0]=-0.000597157153;
    _meanAS[1][0][1]=0.0571921693;
    _meanAS[1][0][2]=700.692431;
    _meanAS[1][0][3]=924.653733;
    _meanAM[1][0][0]=0.000230736156;
    _meanAM[1][0][1]=1.77368196;
    _meanAM[1][0][2]=4461.03178;
    _meanAM[1][0][3]=3300.73792;
    _meanBT[1][0][0]=0.483274186;
    _meanBT[1][0][1]=0;
    _meanBT[1][0][2]=0;
    _meanBT[1][0][3]=0;
    _meanBC[1][0][0]=-0.000651403853;
    _meanBC[1][0][1]=0.0111101805;
    _meanBC[1][0][2]=1276.07724;
    _meanBC[1][0][3]=1489.51887;
    _meanBS[1][0][0]=-0.000251246189;
    _meanBS[1][0][1]=0.0530409004;
    _meanBS[1][0][2]=767.699586;
    _meanBS[1][0][3]=835.195311;
    _meanBM[1][0][0]=-0.187856578;
    _meanBM[1][0][1]=-0.00821848896;
    _meanBM[1][0][2]=0.891813494;
    _meanBM[1][0][3]=-580000000;
    _meanR9[1][0][0]=0.96358076;
    _meanR9[1][0][1]=28.7116938;
    _meanR9[1][0][2]=697.709731;
    _meanR9[1][0][3]=0;

    _sigmaScale[1][0][0]=0.46256953;
    _sigmaScale[1][0][1]=-2.50963561e-08;
    _sigmaScale[1][0][2]=0.0139636379;
    _sigmaScale[1][0][3]=0;
    _sigmaAT[1][0][0]=6.47165025;
    _sigmaAT[1][0][1]=0;
    _sigmaAT[1][0][2]=0;
    _sigmaAT[1][0][3]=0;
    _sigmaAC[1][0][0]=48.1275;
    _sigmaAC[1][0][1]=150005000;
    _sigmaAC[1][0][2]=21231.6;
    _sigmaAC[1][0][3]=2.6e+11;
    _sigmaAS[1][0][0]=0.000209127817;
    _sigmaAS[1][0][1]=2.19868731;
    _sigmaAS[1][0][2]=1695.98579;
    _sigmaAS[1][0][3]=967.250228;
    _sigmaAM[1][0][0]=0.0217972665;
    _sigmaAM[1][0][1]=1.26317651;
    _sigmaAM[1][0][2]=34.0924905;
    _sigmaAM[1][0][3]=55.1895282;
    _sigmaBT[1][0][0]=5.21983754;
    _sigmaBT[1][0][1]=0;
    _sigmaBT[1][0][2]=0;
    _sigmaBT[1][0][3]=0;
    _sigmaBC[1][0][0]=-0.004;
    _sigmaBC[1][0][1]=-120000;
    _sigmaBC[1][0][2]=7.49509e+12;
    _sigmaBC[1][0][3]=36643600;
    _sigmaBS[1][0][0]=0.000250338051;
    _sigmaBS[1][0][1]=1.98819262;
    _sigmaBS[1][0][2]=1967.55308;
    _sigmaBS[1][0][3]=1098.23855;
    _sigmaBM[1][0][0]=0.00101799874;
    _sigmaBM[1][0][1]=88.0546723;
    _sigmaBM[1][0][2]=8.47552e+10;
    _sigmaBM[1][0][3]=-132255.757;
    _sigmaR9[1][0][0]=144.031062;
    _sigmaR9[1][0][1]=-6.11507616e-07;
    _sigmaR9[1][0][2]=1.18181734e-08;
    _sigmaR9[1][0][3]=0;

    _meanScale[1][1][0]=0.288888347;
    _meanScale[1][1][1]=6.52038486e-06;
    _meanScale[1][1][2]=0.000173654897;
    _meanScale[1][1][3]=0.422671325;
    _meanAT[1][1][0]=0.0614964598;
    _meanAT[1][1][1]=0;
    _meanAT[1][1][2]=0;
    _meanAT[1][1][3]=0;
    _meanAC[1][1][0]=-0.00123181641;
    _meanAC[1][1][1]=0.0133568947;
    _meanAC[1][1][2]=165.847556;
    _meanAC[1][1][3]=332.705784;
    _meanAS[1][1][0]=-0.00088161986;
    _meanAS[1][1][1]=0.0304986746;
    _meanAS[1][1][2]=382.755876;
    _meanAS[1][1][3]=616.470187;
    _meanAM[1][1][0]=0.000980695422;
    _meanAM[1][1][1]=0.63575757;
    _meanAM[1][1][2]=0.0336097848;
    _meanAM[1][1][3]=0.043315868;
    _meanBT[1][1][0]=0.11623414;
    _meanBT[1][1][1]=0;
    _meanBT[1][1][2]=0;
    _meanBT[1][1][3]=0;
    _meanBC[1][1][0]=-0.00716072255;
    _meanBC[1][1][1]=-0.440696266;
    _meanBC[1][1][2]=1887.74154;
    _meanBC[1][1][3]=118612;
    _meanBS[1][1][0]=-0.000492035977;
    _meanBS[1][1][1]=0.0292167014;
    _meanBS[1][1][2]=433.232787;
    _meanBS[1][1][3]=484.310448;
    _meanBM[1][1][0]=0.00299476541;
    _meanBM[1][1][1]=0.0149328977;
    _meanBM[1][1][2]=-48728700;
    _meanBM[1][1][3]=37.0041547;
    _meanR9[1][1][0]=0.19617696;
    _meanR9[1][1][1]=-0.350976375;
    _meanR9[1][1][2]=0.181094838;
    _meanR9[1][1][3]=0;

    _sigmaScale[1][1][0]=1.26164895;
    _sigmaScale[1][1][1]=-6.61150347e-07;
    _sigmaScale[1][1][2]=0.0280532297;
    _sigmaScale[1][1][3]=0;
    _sigmaAT[1][1][0]=-0.232612761;
    _sigmaAT[1][1][1]=0;
    _sigmaAT[1][1][2]=0;
    _sigmaAT[1][1][3]=0;
    _sigmaAC[1][1][0]=0.00137406444;
    _sigmaAC[1][1][1]=-0.377659364;
    _sigmaAC[1][1][2]=27171.5802;
    _sigmaAC[1][1][3]=-560000000;
    _sigmaAS[1][1][0]=0.00022943714;
    _sigmaAS[1][1][1]=0.335082568;
    _sigmaAS[1][1][2]=590.511812;
    _sigmaAS[1][1][3]=387.352521;
    _sigmaAM[1][1][0]=-0.000780390674;
    _sigmaAM[1][1][1]=1.05127796;
    _sigmaAM[1][1][2]=33.7378914;
    _sigmaAM[1][1][3]=61.3730807;
    _sigmaBT[1][1][0]=0.529507693;
    _sigmaBT[1][1][1]=0;
    _sigmaBT[1][1][2]=0;
    _sigmaBT[1][1][3]=0;
    _sigmaBC[1][1][0]=-0.00203996;
    _sigmaBC[1][1][1]=93000;
    _sigmaBC[1][1][2]=61225800;
    _sigmaBC[1][1][3]=-4.43323e+17;
    _sigmaBS[1][1][0]=0.00125939613;
    _sigmaBS[1][1][1]=0.31048111;
    _sigmaBS[1][1][2]=295.258764;
    _sigmaBS[1][1][3]=263.974257;
    _sigmaBM[1][1][0]=-0.046100748;
    _sigmaBM[1][1][1]=1.22348596;
    _sigmaBM[1][1][2]=1.9e+09;
    _sigmaBM[1][1][3]=1254.99;
    _sigmaR9[1][1][0]=9.09347838;
    _sigmaR9[1][1][1]=-10.0390435;
    _sigmaR9[1][1][2]=0;
    _sigmaR9[1][1][3]=0;

    _initialised=true;
  }

  assert(_initialised);
  return true;
}

// Get the geometry of cracks and gaps from file
bool PhotonFix::initialiseGeometry(const std::string &s) {

  std::ifstream fin("../test/PhotonFix.dat");
  assert(fin);

  std::cout << "Reading in here" << std::endl;
  for(unsigned i(0);i<169;i++) {
    for(unsigned j(0);j<360;j++) {
      for(unsigned k(0);k<2;k++) {
	fin >> _barrelCGap[i][j][k];
      }
    }
  }
 
  for(unsigned i(0);i<33;i++) {
    for(unsigned j(0);j<180;j++) {
      for(unsigned k(0);k<2;k++) {
	fin >> _barrelSGap[i][j][k];
      }
    }
  }
 
  for(unsigned i(0);i<7;i++) {
    for(unsigned j(0);j<18;j++) {
      for(unsigned k(0);k<2;k++) {
	fin >> _barrelMGap[i][j][k];
      }
    }
  }
  for(unsigned i(0);i<100;i++) {
    for(unsigned j(0);j<100;j++) {
      unsigned k;
      fin >> k;
      _endcapCrystal[i][j]=(k==0);
    }
  }
 
  for(unsigned i(0);i<2;i++) {
    for(unsigned j(0);j<7080;j++) {
      for(unsigned k(0);k<2;k++) {
	fin >> _endcapCGap[i][j][k];
      }
    }
  }
 
  for(unsigned i(0);i<2;i++) {
    for(unsigned j(0);j<264;j++) {
      for(unsigned k(0);k<2;k++) {
	fin >> _endcapSGap[i][j][k];
      }
    }
  }
 
  for(unsigned i(0);i<2;i++) {
    for(unsigned j(0);j<1;j++) {
      for(unsigned k(0);k<2;k++) {
	fin >> _endcapMGap[i][j][k];
      }
    }
  }
 
  assert(fin);
 
  return true;
}

const double PhotonFix::_onePi(acos(-1.0));
const double PhotonFix::_twoPi(2.0*acos(-1.0));

bool   PhotonFix::_initialised=false;

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
