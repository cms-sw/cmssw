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
    f+=_sigmaR9[_be][_hl][0]*_r9+_sigmaR9[_be][_hl][1]*_r9*_r9;
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
  for(unsigned be(0);be<2;be++) {
    for(unsigned hl(0);hl<2;hl++) {
      for(unsigned i(0);i<4;i++) {
	o << " _meanScale[" << be << "][" << hl << "][" << i << "]=" << _meanScale[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << " _meanAT[" << be << "][" << hl << "][" << i << "]=" << _meanAT[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << " _meanAC[" << be << "][" << hl << "][" << i << "]=" << _meanAC[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << " _meanAS[" << be << "][" << hl << "][" << i << "]=" << _meanAS[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << " _meanAM[" << be << "][" << hl << "][" << i << "]=" << _meanAM[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << " _meanBT[" << be << "][" << hl << "][" << i << "]=" << _meanBT[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << " _meanBC[" << be << "][" << hl << "][" << i << "]=" << _meanBC[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << " _meanBS[" << be << "][" << hl << "][" << i << "]=" << _meanBS[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << " _meanBM[" << be << "][" << hl << "][" << i << "]=" << _meanBM[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << " _meanR9[" << be << "][" << hl << "][" << i << "]=" << _meanR9[be][hl][i] << ";" << std::endl;
      }
      o << std::endl;
      
      for(unsigned i(0);i<4;i++) {
	o << " _sigmaScale[" << be << "][" << hl << "][" << i << "]=" << _sigmaScale[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << " _sigmaAT[" << be << "][" << hl << "][" << i << "]=" << _sigmaAT[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << " _sigmaAC[" << be << "][" << hl << "][" << i << "]=" << _sigmaAC[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << " _sigmaAS[" << be << "][" << hl << "][" << i << "]=" << _sigmaAS[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << " _sigmaAM[" << be << "][" << hl << "][" << i << "]=" << _sigmaAM[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << " _sigmaBT[" << be << "][" << hl << "][" << i << "]=" << _sigmaBT[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << " _sigmaBC[" << be << "][" << hl << "][" << i << "]=" << _sigmaBC[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << " _sigmaBS[" << be << "][" << hl << "][" << i << "]=" << _sigmaBS[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << " _sigmaBM[" << be << "][" << hl << "][" << i << "]=" << _sigmaBM[be][hl][i] << ";" << std::endl;
      }
      for(unsigned i(0);i<4;i++) {
	o << " _sigmaR9[" << be << "][" << hl << "][" << i << "]=" << _sigmaR9[be][hl][i] << ";" << std::endl;
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
    _meanScale[0][0][0]=0.996757;
    _meanScale[0][0][1]=-1.02729e-05;
    _meanScale[0][0][2]=2.92397e-05;
    _meanScale[0][0][3]=-0.0892806;
    _meanAT[0][0][0]=0.000616189;
    _meanAT[0][0][1]=0;
    _meanAT[0][0][2]=0;
    _meanAT[0][0][3]=0;
    _meanAC[0][0][0]=-0.00343502;
    _meanAC[0][0][1]=0.0102123;
    _meanAC[0][0][2]=460.294;
    _meanAC[0][0][3]=510.925;
    _meanAS[0][0][0]=0;
    _meanAS[0][0][1]=0;
    _meanAS[0][0][2]=0;
    _meanAS[0][0][3]=0;
    _meanAM[0][0][0]=-0.0010363;
    _meanAM[0][0][1]=0.147364;
    _meanAM[0][0][2]=280.742;
    _meanAM[0][0][3]=200.903;
    _meanBT[0][0][0]=0;
    _meanBT[0][0][1]=0.0290516;
    _meanBT[0][0][2]=-113.185;
    _meanBT[0][0][3]=-176099;
    _meanBC[0][0][0]=-0.00262961;
    _meanBC[0][0][1]=0.0134481;
    _meanBC[0][0][2]=699.825;
    _meanBC[0][0][3]=401.452;
    _meanBS[0][0][0]=0;
    _meanBS[0][0][1]=0;
    _meanBS[0][0][2]=0;
    _meanBS[0][0][3]=0;
    _meanBM[0][0][0]=-0.00170238;
    _meanBM[0][0][1]=0.0751873;
    _meanBM[0][0][2]=232.9;
    _meanBM[0][0][3]=225.712;
    _meanR9[0][0][0]=0.946441;
    _meanR9[0][0][1]=19.5162;
    _meanR9[0][0][2]=173.503;
    _meanR9[0][0][3]=0;

    _sigmaScale[0][0][0]=0.196245;
    _sigmaScale[0][0][1]=0.0191932;
    _sigmaScale[0][0][2]=0.00660053;
    _sigmaScale[0][0][3]=0;
    _sigmaAT[0][0][0]=0.179417;
    _sigmaAT[0][0][1]=0;
    _sigmaAT[0][0][2]=0;
    _sigmaAT[0][0][3]=0;
    _sigmaAC[0][0][0]=-0.00383752;
    _sigmaAC[0][0][1]=0.0881981;
    _sigmaAC[0][0][2]=88.6384;
    _sigmaAC[0][0][3]=-3e+17;
    _sigmaAS[0][0][0]=0;
    _sigmaAS[0][0][1]=0;
    _sigmaAS[0][0][2]=0;
    _sigmaAS[0][0][3]=0;
    _sigmaAM[0][0][0]=0.000932441;
    _sigmaAM[0][0][1]=10.1637;
    _sigmaAM[0][0][2]=610.691;
    _sigmaAM[0][0][3]=290.271;
    _sigmaBT[0][0][0]=-0.00599323;
    _sigmaBT[0][0][1]=1.71943;
    _sigmaBT[0][0][2]=90.7352;
    _sigmaBT[0][0][3]=164.449;
    _sigmaBC[0][0][0]=-0.00167802;
    _sigmaBC[0][0][1]=0.303306;
    _sigmaBC[0][0][2]=1326.56;
    _sigmaBC[0][0][3]=765.552;
    _sigmaBS[0][0][0]=0;
    _sigmaBS[0][0][1]=0;
    _sigmaBS[0][0][2]=0;
    _sigmaBS[0][0][3]=0;
    _sigmaBM[0][0][0]=-0.00252598;
    _sigmaBM[0][0][1]=52.0947;
    _sigmaBM[0][0][2]=660.313;
    _sigmaBM[0][0][3]=1052.64;
    _sigmaR9[0][0][0]=0.952982;
    _sigmaR9[0][0][1]=1961.39;
    _sigmaR9[0][0][2]=22382.1;
    _sigmaR9[0][0][3]=0;

    _meanScale[0][1][0]=0.88802;
    _meanScale[0][1][1]=1.61e-05;
    _meanScale[0][1][2]=-1.08e-05;
    _meanScale[0][1][3]=-0.489;
    _meanAT[0][1][0]=-0.00165;
    _meanAT[0][1][1]=0;
    _meanAT[0][1][2]=0;
    _meanAT[0][1][3]=0;
    _meanAC[0][1][0]=-0.003195;
    _meanAC[0][1][1]=0.02012;
    _meanAC[0][1][2]=151.6;
    _meanAC[0][1][3]=255;
    _meanAS[0][1][0]=0;
    _meanAS[0][1][1]=0;
    _meanAS[0][1][2]=0;
    _meanAS[0][1][3]=0;
    _meanAM[0][1][0]=-0.00055;
    _meanAM[0][1][1]=0.1515;
    _meanAM[0][1][2]=534;
    _meanAM[0][1][3]=330;
    _meanBT[0][1][0]=0;
    _meanBT[0][1][1]=0.187;
    _meanBT[0][1][2]=299.298;
    _meanBT[0][1][3]=428.221;
    _meanBC[0][1][0]=-0.00366;
    _meanBC[0][1][1]=0.0038;
    _meanBC[0][1][2]=760;
    _meanBC[0][1][3]=760;
    _meanBS[0][1][0]=0;
    _meanBS[0][1][1]=0;
    _meanBS[0][1][2]=0;
    _meanBS[0][1][3]=0;
    _meanBM[0][1][0]=-0.00316;
    _meanBM[0][1][1]=0.0404;
    _meanBM[0][1][2]=137;
    _meanBM[0][1][3]=146;
    _meanR9[0][1][0]=0.02573;
    _meanR9[0][1][1]=-0.03722;
    _meanR9[0][1][2]=0.0144;
    _meanR9[0][1][3]=0;

    _sigmaScale[0][1][0]=0.66;
    _sigmaScale[0][1][1]=0.07465;
    _sigmaScale[0][1][2]=-0.00373;
    _sigmaScale[0][1][3]=0;
    _sigmaAT[0][1][0]=1.363;
    _sigmaAT[0][1][1]=0;
    _sigmaAT[0][1][2]=0;
    _sigmaAT[0][1][3]=0;
    _sigmaAC[0][1][0]=-0.00342;
    _sigmaAC[0][1][1]=-1.667;
    _sigmaAC[0][1][2]=6.6;
    _sigmaAC[0][1][3]=14;
    _sigmaAS[0][1][0]=0;
    _sigmaAS[0][1][1]=0;
    _sigmaAS[0][1][2]=0;
    _sigmaAS[0][1][3]=0;
    _sigmaAM[0][1][0]=-0.00094;
    _sigmaAM[0][1][1]=12.0228;
    _sigmaAM[0][1][2]=536;
    _sigmaAM[0][1][3]=464;
    _sigmaBT[0][1][0]=0.022;
    _sigmaBT[0][1][1]=3.223;
    _sigmaBT[0][1][2]=-0.361;
    _sigmaBT[0][1][3]=-0.318;
    _sigmaBC[0][1][0]=0.00567264;
    _sigmaBC[0][1][1]=-1.184;
    _sigmaBC[0][1][2]=76000;
    _sigmaBC[0][1][3]=2.14748e+08;
    _sigmaBS[0][1][0]=0;
    _sigmaBS[0][1][1]=0;
    _sigmaBS[0][1][2]=0;
    _sigmaBS[0][1][3]=0;
    _sigmaBM[0][1][0]=-0.00236;
    _sigmaBM[0][1][1]=5.78;
    _sigmaBM[0][1][2]=166;
    _sigmaBM[0][1][3]=203.5;
    _sigmaR9[0][1][0]=-2.94;
    _sigmaR9[0][1][1]=1.53445;
    _sigmaR9[0][1][2]=0;
    _sigmaR9[0][1][3]=0;

    _meanScale[1][0][0]=0.993921;
    _meanScale[1][0][1]=-8.11417e-06;
    _meanScale[1][0][2]=3.23086e-05;
    _meanScale[1][0][3]=-0.0976369;
    _meanAT[1][0][0]=0.0691979;
    _meanAT[1][0][1]=0;
    _meanAT[1][0][2]=0;
    _meanAT[1][0][3]=0;
    _meanAC[1][0][0]=-0.000310485;
    _meanAC[1][0][1]=0.0169478;
    _meanAC[1][0][2]=1301.97;
    _meanAC[1][0][3]=796.53;
    _meanAS[1][0][0]=-0.0005425;
    _meanAS[1][0][1]=0.0676613;
    _meanAS[1][0][2]=790.284;
    _meanAS[1][0][3]=970;
    _meanAM[1][0][0]=0.000134362;
    _meanAM[1][0][1]=0.364822;
    _meanAM[1][0][2]=3396.04;
    _meanAM[1][0][3]=2407.2;
    _meanBT[1][0][0]=0.0456649;
    _meanBT[1][0][1]=0;
    _meanBT[1][0][2]=0;
    _meanBT[1][0][3]=0;
    _meanBC[1][0][0]=-0.000356413;
    _meanBC[1][0][1]=0.0167018;
    _meanBC[1][0][2]=1394.52;
    _meanBC[1][0][3]=823.035;
    _meanBS[1][0][0]=-0.000484;
    _meanBS[1][0][1]=0.0542618;
    _meanBS[1][0][2]=798.07;
    _meanBS[1][0][3]=1019.88;
    _meanBM[1][0][0]=-0.175333;
    _meanBM[1][0][1]=-0.00324513;
    _meanBM[1][0][2]=2.87972;
    _meanBM[1][0][3]=7.58712e+12;
    _meanR9[1][0][0]=0.96393;
    _meanR9[1][0][1]=29.6072;
    _meanR9[1][0][2]=334.89;
    _meanR9[1][0][3]=0;

    _sigmaScale[1][0][0]=0.337911;
    _sigmaScale[1][0][1]=1.03419e-11;
    _sigmaScale[1][0][2]=0.0153846;
    _sigmaScale[1][0][3]=0;
    _sigmaAT[1][0][0]=1.12;
    _sigmaAT[1][0][1]=0;
    _sigmaAT[1][0][2]=0;
    _sigmaAT[1][0][3]=0;
    _sigmaAC[1][0][0]=48.1275;
    _sigmaAC[1][0][1]=1.50005e+08;
    _sigmaAC[1][0][2]=21231.6;
    _sigmaAC[1][0][3]=2.6e+11;
    _sigmaAS[1][0][0]=-0.00036952;
    _sigmaAS[1][0][1]=2.75194;
    _sigmaAS[1][0][2]=1045.06;
    _sigmaAS[1][0][3]=1420.87;
    _sigmaAM[1][0][0]=0.05567;
    _sigmaAM[1][0][1]=0.32;
    _sigmaAM[1][0][2]=420;
    _sigmaAM[1][0][3]=12.8233;
    _sigmaBT[1][0][0]=-0.22312;
    _sigmaBT[1][0][1]=0;
    _sigmaBT[1][0][2]=0;
    _sigmaBT[1][0][3]=0;
    _sigmaBC[1][0][0]=-0.00320087;
    _sigmaBC[1][0][1]=59.2836;
    _sigmaBC[1][0][2]=7.49509e+12;
    _sigmaBC[1][0][3]=1.32714e+07;
    _sigmaBS[1][0][0]=0.000189384;
    _sigmaBS[1][0][1]=2.89111;
    _sigmaBS[1][0][2]=2468.02;
    _sigmaBS[1][0][3]=1294.69;
    _sigmaBM[1][0][0]=0.00154629;
    _sigmaBM[1][0][1]=16.9762;
    _sigmaBM[1][0][2]=1.05e+07;
    _sigmaBM[1][0][3]=-9e+06;
    _sigmaR9[1][0][0]=99.68;
    _sigmaR9[1][0][1]=-9.37265e-08;
    _sigmaR9[1][0][2]=6.91613e-09;
    _sigmaR9[1][0][3]=0;

    _meanScale[1][1][0]=0.340125;
    _meanScale[1][1][1]=-5.02007e-05;
    _meanScale[1][1][2]=0.000208216;
    _meanScale[1][1][3]=-1.32063;
    _meanAT[1][1][0]=-0.092494;
    _meanAT[1][1][1]=0;
    _meanAT[1][1][2]=0;
    _meanAT[1][1][3]=0;
    _meanAC[1][1][0]=-0.00131984;
    _meanAC[1][1][1]=0.00950668;
    _meanAC[1][1][2]=259.804;
    _meanAC[1][1][3]=705.003;
    _meanAS[1][1][0]=-0.00705787;
    _meanAS[1][1][1]=-0.00581517;
    _meanAS[1][1][2]=7.10951e+10;
    _meanAS[1][1][3]=-68.7322;
    _meanAM[1][1][0]=-0.00496966;
    _meanAM[1][1][1]=0.655834;
    _meanAM[1][1][2]=0.0491064;
    _meanAM[1][1][3]=0.0632471;
    _meanBT[1][1][0]=-0.158344;
    _meanBT[1][1][1]=0;
    _meanBT[1][1][2]=0;
    _meanBT[1][1][3]=0;
    _meanBC[1][1][0]=-0.00498431;
    _meanBC[1][1][1]=-18315.4;
    _meanBC[1][1][2]=50760;
    _meanBC[1][1][3]=118612;
    _meanBS[1][1][0]=-0.000620577;
    _meanBS[1][1][1]=0.0247672;
    _meanBS[1][1][2]=437.005;
    _meanBS[1][1][3]=374.823;
    _meanBM[1][1][0]=0.000431046;
    _meanBM[1][1][1]=0.0224695;
    _meanBM[1][1][2]=-1.06e+07;
    _meanBM[1][1][3]=-135.039;
    _meanR9[1][1][0]=0.151826;
    _meanR9[1][1][1]=0.00348896;
    _meanR9[1][1][2]=-0.161781;
    _meanR9[1][1][3]=0;

    _sigmaScale[1][1][0]=1.19993;
    _sigmaScale[1][1][1]=1.38778e-17;
    _sigmaScale[1][1][2]=0.0251314;
    _sigmaScale[1][1][3]=0;
    _sigmaAT[1][1][0]=-0.791929;
    _sigmaAT[1][1][1]=0;
    _sigmaAT[1][1][2]=0;
    _sigmaAT[1][1][3]=0;
    _sigmaAC[1][1][0]=0.0015505;
    _sigmaAC[1][1][1]=-0.530085;
    _sigmaAC[1][1][2]=33506.7;
    _sigmaAC[1][1][3]=-7e+09;
    _sigmaAS[1][1][0]=7.98682e-05;
    _sigmaAS[1][1][1]=0.875124;
    _sigmaAS[1][1][2]=483.102;
    _sigmaAS[1][1][3]=1005.71;
    _sigmaAM[1][1][0]=-0.00510653;
    _sigmaAM[1][1][1]=1.05312;
    _sigmaAM[1][1][2]=49.6466;
    _sigmaAM[1][1][3]=7.11474e+12;
    _sigmaBT[1][1][0]=-1.30079;
    _sigmaBT[1][1][1]=0;
    _sigmaBT[1][1][2]=0;
    _sigmaBT[1][1][3]=0;
    _sigmaBC[1][1][0]=-0.00202301;
    _sigmaBC[1][1][1]=4.19368;
    _sigmaBC[1][1][2]=-1.08161e+07;
    _sigmaBC[1][1][3]=-6e+09;
    _sigmaBS[1][1][0]=0.00182147;
    _sigmaBS[1][1][1]=0.25995;
    _sigmaBS[1][1][2]=881.576;
    _sigmaBS[1][1][3]=304.891;
    _sigmaBM[1][1][0]=-0.0454995;
    _sigmaBM[1][1][1]=7.63845;
    _sigmaBM[1][1][2]=1.9e+09;
    _sigmaBM[1][1][3]=150001;
    _sigmaR9[1][1][0]=5.69741;
    _sigmaR9[1][1][1]=-6.23853;
    _sigmaR9[1][1][2]=0;
    _sigmaR9[1][1][3]=0;

    _initialised=true;
  }

  if(s=="4_2e") {
    _meanScale[0][0][0]=1.02133;
    _meanScale[0][0][1]=-0.000183073;
    _meanScale[0][0][2]=0.000252052;
    _meanScale[0][0][3]=0.300892;
    _meanAT[0][0][0]=0.016876;
    _meanAT[0][0][1]=0;
    _meanAT[0][0][2]=0;
    _meanAT[0][0][3]=0;
    _meanAC[0][0][0]=-0.00327969;
    _meanAC[0][0][1]=0.00932112;
    _meanAC[0][0][2]=430.176;
    _meanAC[0][0][3]=440.94;
    _meanAS[0][0][0]=0;
    _meanAS[0][0][1]=0;
    _meanAS[0][0][2]=0;
    _meanAS[0][0][3]=0;
    _meanAM[0][0][0]=-0.000876726;
    _meanAM[0][0][1]=0.128953;
    _meanAM[0][0][2]=262.609;
    _meanAM[0][0][3]=204.869;
    _meanBT[0][0][0]=0;
    _meanBT[0][0][1]=0;
    _meanBT[0][0][2]=0;
    _meanBT[0][0][3]=0;
    _meanBC[0][0][0]=-0.00316342;
    _meanBC[0][0][1]=0.00801235;
    _meanBC[0][0][2]=552.944;
    _meanBC[0][0][3]=507.239;
    _meanBS[0][0][0]=0;
    _meanBS[0][0][1]=0;
    _meanBS[0][0][2]=0;
    _meanBS[0][0][3]=0;
    _meanBM[0][0][0]=-0.00171652;
    _meanBM[0][0][1]=0.0746995;
    _meanBM[0][0][2]=190.606;
    _meanBM[0][0][3]=210.147;
    _meanR9[0][0][0]=0.847311;
    _meanR9[0][0][1]=-10.0049;
    _meanR9[0][0][2]=67.613;
    _meanR9[0][0][3]=0;

    _sigmaScale[0][0][0]=0.345777;
    _sigmaScale[0][0][1]=0.0375747;
    _sigmaScale[0][0][2]=-0.00516299;
    _sigmaScale[0][0][3]=0;
    _sigmaAT[0][0][0]=0.994444;
    _sigmaAT[0][0][1]=0;
    _sigmaAT[0][0][2]=0;
    _sigmaAT[0][0][3]=0;
    _sigmaAC[0][0][0]=-0.00380373;
    _sigmaAC[0][0][1]=-1.00415;
    _sigmaAC[0][0][2]=110274;
    _sigmaAC[0][0][3]=-3e+17;
    _sigmaAS[0][0][0]=0;
    _sigmaAS[0][0][1]=0;
    _sigmaAS[0][0][2]=0;
    _sigmaAS[0][0][3]=0;
    _sigmaAM[0][0][0]=-0.000881147;
    _sigmaAM[0][0][1]=8.76414;
    _sigmaAM[0][0][2]=455.387;
    _sigmaAM[0][0][3]=412.382;
    _sigmaBT[0][0][0]=0.00578517;
    _sigmaBT[0][0][1]=2.28008;
    _sigmaBT[0][0][2]=-3.36e+07;
    _sigmaBT[0][0][3]=-152.696;
    _sigmaBC[0][0][0]=-0.00161997;
    _sigmaBC[0][0][1]=50513.8;
    _sigmaBC[0][0][2]=-1.26888e+07;
    _sigmaBC[0][0][3]=3.75561e+07;
    _sigmaBS[0][0][0]=0;
    _sigmaBS[0][0][1]=0;
    _sigmaBS[0][0][2]=0;
    _sigmaBS[0][0][3]=0;
    _sigmaBM[0][0][0]=-0.00168415;
    _sigmaBM[0][0][1]=6.89324;
    _sigmaBM[0][0][2]=272.169;
    _sigmaBM[0][0][3]=315.424;
    _sigmaR9[0][0][0]=0.952571;
    _sigmaR9[0][0][1]=0;
    _sigmaR9[0][0][2]=0;
    _sigmaR9[0][0][3]=0;

    _meanScale[0][1][0]=0.870259;
    _meanScale[0][1][1]=0.000234598;
    _meanScale[0][1][2]=-0.000262422;
    _meanScale[0][1][3]=0.277914;
    _meanAT[0][1][0]=1e-06;
    _meanAT[0][1][1]=0;
    _meanAT[0][1][2]=0;
    _meanAT[0][1][3]=0;
    _meanAC[0][1][0]=-0.00324258;
    _meanAC[0][1][1]=0.0209948;
    _meanAC[0][1][2]=167.983;
    _meanAC[0][1][3]=305.464;
    _meanAS[0][1][0]=0.0333341;
    _meanAS[0][1][1]=-591.066;
    _meanAS[0][1][2]=1.48079e+11;
    _meanAS[0][1][3]=1.61987e+07;
    _meanAM[0][1][0]=-0.000991417;
    _meanAM[0][1][1]=0.121994;
    _meanAM[0][1][2]=367.004;
    _meanAM[0][1][3]=289.22;
    _meanBT[0][1][0]=0;
    _meanBT[0][1][1]=0.187;
    _meanBT[0][1][2]=299.298;
    _meanBT[0][1][3]=428.221;
    _meanBC[0][1][0]=-0.00437016;
    _meanBC[0][1][1]=0.00235015;
    _meanBC[0][1][2]=759.85;
    _meanBC[0][1][3]=-5018.79;
    _meanBS[0][1][0]=-1.2e+08;
    _meanBS[0][1][1]=0.0934286;
    _meanBS[0][1][2]=-1.226e-16;
    _meanBS[0][1][3]=11442.2;
    _meanBM[0][1][0]=-0.000153536;
    _meanBM[0][1][1]=0.0101487;
    _meanBM[0][1][2]=60.0003;
    _meanBM[0][1][3]=-4.4e+08;
    _meanR9[0][1][0]=0.0853634;
    _meanR9[0][1][1]=-0.263819;
    _meanR9[0][1][2]=0.200892;
    _meanR9[0][1][3]=0;

    _sigmaScale[0][1][0]=0.705452;
    _sigmaScale[0][1][1]=-0.0781086;
    _sigmaScale[0][1][2]=0.0039561;
    _sigmaScale[0][1][3]=0;
    _sigmaAT[0][1][0]=1.46888;
    _sigmaAT[0][1][1]=0;
    _sigmaAT[0][1][2]=0;
    _sigmaAT[0][1][3]=0;
    _sigmaAC[0][1][0]=-0.00537924;
    _sigmaAC[0][1][1]=-0.740685;
    _sigmaAC[0][1][2]=1.59714;
    _sigmaAC[0][1][3]=31.9922;
    _sigmaAS[0][1][0]=0;
    _sigmaAS[0][1][1]=0;
    _sigmaAS[0][1][2]=0;
    _sigmaAS[0][1][3]=0;
    _sigmaAM[0][1][0]=0.000163134;
    _sigmaAM[0][1][1]=7.30822;
    _sigmaAM[0][1][2]=632.013;
    _sigmaAM[0][1][3]=373.003;
    _sigmaBT[0][1][0]=0.00253725;
    _sigmaBT[0][1][1]=2.97332;
    _sigmaBT[0][1][2]=-0.516548;
    _sigmaBT[0][1][3]=-0.50994;
    _sigmaBC[0][1][0]=0.00578715;
    _sigmaBC[0][1][1]=-0.473225;
    _sigmaBC[0][1][2]=8363.08;
    _sigmaBC[0][1][3]=2.36808e+19;
    _sigmaBS[0][1][0]=0;
    _sigmaBS[0][1][1]=0;
    _sigmaBS[0][1][2]=0;
    _sigmaBS[0][1][3]=0;
    _sigmaBM[0][1][0]=0.000184199;
    _sigmaBM[0][1][1]=3.82105;
    _sigmaBM[0][1][2]=93.4062;
    _sigmaBM[0][1][3]=95.4072;
    _sigmaR9[0][1][0]=1.20664;
    _sigmaR9[0][1][1]=-3.43548;
    _sigmaR9[0][1][2]=0;
    _sigmaR9[0][1][3]=0;

    _meanScale[1][0][0]=0.966787;
    _meanScale[1][0][1]=4.77112e-05;
    _meanScale[1][0][2]=-4.47768e-05;
    _meanScale[1][0][3]=0.0854689;
    _meanAT[1][0][0]=0.0691979;
    _meanAT[1][0][1]=0;
    _meanAT[1][0][2]=0;
    _meanAT[1][0][3]=0;
    _meanAC[1][0][0]=-0.000985244;
    _meanAC[1][0][1]=0.0132327;
    _meanAC[1][0][2]=530.13;
    _meanAC[1][0][3]=1008.64;
    _meanAS[1][0][0]=-0.000571197;
    _meanAS[1][0][1]=0.0579624;
    _meanAS[1][0][2]=687.722;
    _meanAS[1][0][3]=836.21;
    _meanAM[1][0][0]=0.00052402;
    _meanAM[1][0][1]=0.402941;
    _meanAM[1][0][2]=3615.29;
    _meanAM[1][0][3]=1495.1;
    _meanBT[1][0][0]=0.0456649;
    _meanBT[1][0][1]=0;
    _meanBT[1][0][2]=0;
    _meanBT[1][0][3]=0;
    _meanBC[1][0][0]=-0.000675969;
    _meanBC[1][0][1]=0.0118283;
    _meanBC[1][0][2]=1050.1;
    _meanBC[1][0][3]=1329.8;
    _meanBS[1][0][0]=-0.000254569;
    _meanBS[1][0][1]=0.0576783;
    _meanBS[1][0][2]=853.571;
    _meanBS[1][0][3]=938.698;
    _meanBM[1][0][0]=-0.186506;
    _meanBM[1][0][1]=-0.00654349;
    _meanBM[1][0][2]=1.09286;
    _meanBM[1][0][3]=-5.8e+08;
    _meanR9[1][0][0]=0.951164;
    _meanR9[1][0][1]=-0.323083;
    _meanR9[1][0][2]=506.054;
    _meanR9[1][0][3]=0;

    _sigmaScale[1][0][0]=0.443288;
    _sigmaScale[1][0][1]=1.73472e-18;
    _sigmaScale[1][0][2]=0.0134633;
    _sigmaScale[1][0][3]=0;
    _sigmaAT[1][0][0]=6.60294;
    _sigmaAT[1][0][1]=0;
    _sigmaAT[1][0][2]=0;
    _sigmaAT[1][0][3]=0;
    _sigmaAC[1][0][0]=48.1275;
    _sigmaAC[1][0][1]=1.50005e+08;
    _sigmaAC[1][0][2]=21231.6;
    _sigmaAC[1][0][3]=2.6e+11;
    _sigmaAS[1][0][0]=0.000112234;
    _sigmaAS[1][0][1]=2.38201;
    _sigmaAS[1][0][2]=1368.36;
    _sigmaAS[1][0][3]=1014.58;
    _sigmaAM[1][0][0]=0.0227945;
    _sigmaAM[1][0][1]=1.35035;
    _sigmaAM[1][0][2]=34.8078;
    _sigmaAM[1][0][3]=41.2653;
    _sigmaBT[1][0][0]=4.6512;
    _sigmaBT[1][0][1]=0;
    _sigmaBT[1][0][2]=0;
    _sigmaBT[1][0][3]=0;
    _sigmaBC[1][0][0]=-0.00385034;
    _sigmaBC[1][0][1]=9860.11;
    _sigmaBC[1][0][2]=7.49509e+12;
    _sigmaBC[1][0][3]=5.55794e+07;
    _sigmaBS[1][0][0]=0.000287389;
    _sigmaBS[1][0][1]=2.22754;
    _sigmaBS[1][0][2]=2066.43;
    _sigmaBS[1][0][3]=926.247;
    _sigmaBM[1][0][0]=0.00110983;
    _sigmaBM[1][0][1]=4.70582;
    _sigmaBM[1][0][2]=8.47552e+10;
    _sigmaBM[1][0][3]=-4572.21;
    _sigmaR9[1][0][0]=141.283;
    _sigmaR9[1][0][1]=-6.07621e-07;
    _sigmaR9[1][0][2]=1.40907e-08;
    _sigmaR9[1][0][3]=0;

    _meanScale[1][1][0]=0.290862;
    _meanScale[1][1][1]=-5.93303e-05;
    _meanScale[1][1][2]=0.000431058;
    _meanScale[1][1][3]=0.350738;
    _meanAT[1][1][0]=-0.092494;
    _meanAT[1][1][1]=0;
    _meanAT[1][1][2]=0;
    _meanAT[1][1][3]=0;
    _meanAC[1][1][0]=-0.00105099;
    _meanAC[1][1][1]=0.0195863;
    _meanAC[1][1][2]=162.87;
    _meanAC[1][1][3]=259.479;
    _meanAS[1][1][0]=-0.00820192;
    _meanAS[1][1][1]=-0.00404515;
    _meanAS[1][1][2]=-9e+12;
    _meanAS[1][1][3]=-12.1537;
    _meanAM[1][1][0]=0.00430129;
    _meanAM[1][1][1]=0.647006;
    _meanAM[1][1][2]=0.0524654;
    _meanAM[1][1][3]=0.057336;
    _meanBT[1][1][0]=-0.158344;
    _meanBT[1][1][1]=0;
    _meanBT[1][1][2]=0;
    _meanBT[1][1][3]=0;
    _meanBC[1][1][0]=-0.00661496;
    _meanBC[1][1][1]=-0.0551321;
    _meanBC[1][1][2]=1286.49;
    _meanBC[1][1][3]=118612;
    _meanBS[1][1][0]=-0.000517633;
    _meanBS[1][1][1]=0.0294134;
    _meanBS[1][1][2]=414.964;
    _meanBS[1][1][3]=487.294;
    _meanBM[1][1][0]=0.00298075;
    _meanBM[1][1][1]=0.0146853;
    _meanBM[1][1][2]=-7.17525e+06;
    _meanBM[1][1][3]=41.9253;
    _meanR9[1][1][0]=0.213804;
    _meanR9[1][1][1]=-0.395466;
    _meanR9[1][1][2]=0.202973;
    _meanR9[1][1][3]=0;

    _sigmaScale[1][1][0]=1.29656;
    _sigmaScale[1][1][1]=9.61084e-11;
    _sigmaScale[1][1][2]=0.0270649;
    _sigmaScale[1][1][3]=0;
    _sigmaAT[1][1][0]=-0.0115339;
    _sigmaAT[1][1][1]=0;
    _sigmaAT[1][1][2]=0;
    _sigmaAT[1][1][3]=0;
    _sigmaAC[1][1][0]=0.0013797;
    _sigmaAC[1][1][1]=-0.445789;
    _sigmaAC[1][1][2]=29539.8;
    _sigmaAC[1][1][3]=-3.6e+08;
    _sigmaAS[1][1][0]=0.000213606;
    _sigmaAS[1][1][1]=0.547654;
    _sigmaAS[1][1][2]=892.47;
    _sigmaAS[1][1][3]=408.826;
    _sigmaAM[1][1][0]=-0.00121353;
    _sigmaAM[1][1][1]=1.10052;
    _sigmaAM[1][1][2]=31.303;
    _sigmaAM[1][1][3]=80.0479;
    _sigmaBT[1][1][0]=0.470361;
    _sigmaBT[1][1][1]=0;
    _sigmaBT[1][1][2]=0;
    _sigmaBT[1][1][3]=0;
    _sigmaBC[1][1][0]=-0.00222463;
    _sigmaBC[1][1][1]=18440.6;
    _sigmaBC[1][1][2]=5.80805e+07;
    _sigmaBC[1][1][3]=-4.43323e+17;
    _sigmaBS[1][1][0]=0.00156455;
    _sigmaBS[1][1][1]=0.337173;
    _sigmaBS[1][1][2]=281.318;
    _sigmaBS[1][1][3]=252.457;
    _sigmaBM[1][1][0]=-0.0455886;
    _sigmaBM[1][1][1]=1.47875;
    _sigmaBM[1][1][2]=1.9e+09;
    _sigmaBM[1][1][3]=1193.93;
    _sigmaR9[1][1][0]=8.89993;
    _sigmaR9[1][1][1]=-9.8386;
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
