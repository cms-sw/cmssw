//
// Original Author: Attilio Santocchia Feb. 28, 2008
//
// Jet Parton dependent corrections 
//
#include "CondFormats/JetMETObjects/interface/SimpleL7PartonCorrector.h"

#include "CondFormats/JetMETObjects/interface/SimpleJetCorrectorParameters.h"

#include <vector>

#include "FWCore/Utilities/interface/Exception.h"

#include "Math/PtEtaPhiE4D.h"
#include "Math/LorentzVector.h"
typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > PtEtaPhiELorentzVectorD;
typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > XYZTLorentzVectorD;

namespace {
  const unsigned nParameters = 5;
}

SimpleL7PartonCorrector::SimpleL7PartonCorrector () 
  : mParameters (0) 
{}

SimpleL7PartonCorrector::SimpleL7PartonCorrector (const std::string& fDataFile, const std::string& fSection) 
  : mParameters (new SimpleJetCorrectorParameters (fDataFile, fSection)) 
{}

SimpleL7PartonCorrector::~SimpleL7PartonCorrector () {
  delete mParameters;
}

double SimpleL7PartonCorrector::correctionPtEta (double fPt, double fEta) const {
  const std::vector<float>& p = mParameters->record (0).parameters ();
/*
  std::cout << p[0] << " " 
       << p[1] << " " 
       << p[2] << " "
       << p[3] << " "
       << p[4] << " "
       << p[5] << " "
       << p[6] << " "
       << p[7] << " "
       << p[8] << " "
       << fEta << " " << fPt << std::endl;
*/
  if( fPt < p[0] ) return 1.;
  float aPar = 1/(p[2]*fPt+p[3]) + p[4];
  float bPar = p[5]+p[6]*log(fPt);
  float cPar = p[7]+p[8]*fPt;
  float correction = aPar+bPar*fabs(fEta)+cPar*fEta*fEta;
  return 1./correction;
}

double SimpleL7PartonCorrector::correctionXYZT (double fPx, double fPy, double fPz, double fE) const {
  XYZTLorentzVectorD p4 (fPx, fPy, fPz, fE);
  return correctionPtEta (p4.Pt(), p4.Eta());
}

double SimpleL7PartonCorrector::correctionEtEtaPhiP (double fEt, double fEta, double fPhi, double fP) const {
  double costhetainv = cosh (fEta);
  return correctionPtEta (fP/costhetainv, fEta);
}


