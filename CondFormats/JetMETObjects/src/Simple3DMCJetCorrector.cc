//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: Simple3DMCJetCorrector.cc,v 1.4 2007/08/15 12:50:38 ratnik Exp $
//
// MC Jet Corrector
//
#include "CondFormats/JetMETObjects/interface/Simple3DMCJetCorrector.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

#include "Math/PtEtaPhiE4D.h"
#include "Math/LorentzVector.h"
typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > PtEtaPhiELorentzVectorD;
typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > XYZTLorentzVectorD;

class Simple3DMCJetCorrectorParameters {
public:
  Simple3DMCJetCorrectorParameters (const std::string& fDataFile);
  ~Simple3DMCJetCorrectorParameters () {}
  double correction (double fPt, double fEta, double fEmFraction) const;
private:
  std::vector<double> mPar;
};

Simple3DMCJetCorrectorParameters::Simple3DMCJetCorrectorParameters (const std::string& fDataFile) 
{
  std::ifstream in( fDataFile.c_str() );
  std::string line;
  while( std::getline( in, line)){
    if(!line.size() || line[0]=='#') continue;
    std::istringstream linestream(line);
    double value;
    while (linestream >> value) mPar.push_back (value);
    break;  // only one parameters line is expected
  }
  if (mPar.size() < 20) {
    std::cerr << "Simple3DMCJetCorrectorParameters-> Initialization error: less than 20 parameters provided" << std::endl;
  }
}


double Simple3DMCJetCorrectorParameters::correction (double fPt, double fEta, double fEmFraction) const {
  double aeta = fabs(fEta);
  double aeta2 = aeta*aeta;
  double aeta3 = aeta2*aeta;
  double aeta4 = aeta3*aeta;
  double emf = fEmFraction;
  double emf2 = emf*emf;
  double emf3 = emf2*emf;
  double emf4 = emf3*emf;

  double p0 = mPar[0]*emf + 
    mPar[1]*aeta2 + mPar[2]*emf*aeta + mPar[3]*emf2 + 
    mPar[4]*emf3;
  double p1 = mPar[5] + 
    mPar[6]*aeta + mPar[7]*emf + 
    mPar[8]*aeta2 + mPar[9]*aeta*emf + mPar[10]*emf2 + 
    mPar[11]*aeta3 + mPar[12]*aeta2*emf + mPar[13]*aeta*emf2 + mPar[14]*emf3 + 
    mPar[15]*aeta4 + mPar[16]*aeta3*emf + mPar[17]*aeta2*emf2 + mPar[18]*aeta*emf3 + mPar[19]*emf4;
  double offset = p0 * (pow (fPt, p1) - 1.);
  double result = 1;
  // corrections valid in limited area only;
  if (fPt > 0 && fPt < 300 && aeta < 2.5) result = 1. + offset/fPt;
  return result;
}

Simple3DMCJetCorrector::Simple3DMCJetCorrector () 
  : mParameters (0) 
{}

Simple3DMCJetCorrector::Simple3DMCJetCorrector (const std::string& fDataFile) 
  : mParameters (new Simple3DMCJetCorrectorParameters (fDataFile)) 
{}

Simple3DMCJetCorrector::~Simple3DMCJetCorrector () {
  delete mParameters;
}

double Simple3DMCJetCorrector::correctionXYZTEmfraction (double fPx, double fPy, double fPz, double fE, double fEmFraction) const {
  XYZTLorentzVectorD p4 (fPx, fPy, fPz, fE);
  return correctionPtEtaPhiEEmfraction (p4.Pt(), p4.Eta(), p4.Phi(), p4.E(), fEmFraction);
}

double Simple3DMCJetCorrector::correctionPtEtaPhiEEmfraction (double fPt, double fEta, double fPhi, double fE, double fEmFraction) const {
  return mParameters->correction (fPt, fEta, fEmFraction);
}

double Simple3DMCJetCorrector::correctionEtEtaPhiPEmfraction (double fEt, double fEta, double fPhi, double fP, double fEmFraction) const {
  double costhetainv = cosh (fEta);
  return correctionPtEtaPhiEEmfraction (fP/costhetainv, fEta, fPhi, fEt*costhetainv, fEmFraction);
}
