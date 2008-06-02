#include "CondFormats/JetMETObjects/interface/SimpleL3AbsoluteCorrector.h"
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

SimpleL3AbsoluteCorrector::SimpleL3AbsoluteCorrector () 
  : mParameters (0) 
{}

SimpleL3AbsoluteCorrector::SimpleL3AbsoluteCorrector (const std::string& fDataFile) 
  : mParameters (new SimpleJetCorrectorParameters (fDataFile)) 
{}

SimpleL3AbsoluteCorrector::~SimpleL3AbsoluteCorrector () {
  delete mParameters;
}

double SimpleL3AbsoluteCorrector::correctionPtEta (double fPt, double fEta) const {
  unsigned band = mParameters->bandIndex (fEta);
  double result = correctionBandPtEta (band, fPt, fEta);  
  return result;
}

double SimpleL3AbsoluteCorrector::correctionXYZT (double fPx, double fPy, double fPz, double fE) const {
  XYZTLorentzVectorD p4 (fPx, fPy, fPz, fE);
  return correctionPtEta (p4.Pt(), p4.Eta());
}

double SimpleL3AbsoluteCorrector::correctionEtEtaPhiP (double fEt, double fEta, double fPhi, double fP) const {
  double costhetainv = cosh (fEta);
  return correctionPtEta (fP/costhetainv, fEta);
}

double SimpleL3AbsoluteCorrector::correctionBandPtEta (unsigned fBand, double fPt, double fEta) const {
  if (fBand >= mParameters->size()) {
    throw cms::Exception ("SimpleL3AbsoluteCorrector") 
      << "wrong band: " << fBand << ": only " <<  mParameters->size() << " is available";
  }
  const std::vector<float>& p = mParameters->record (fBand).parameters ();
  if (p.size() != nParameters) {
    throw cms::Exception ("SimpleL3AbsoluteCorrector") 
      << "wrong # of parameters: " << nParameters << " expected, " << p.size() << " got";
  }
  double pt = (fPt < p[0]) ? p[0] : (fPt > p[1]) ? p[1] : fPt;
  double result = p[2]+p[3]/(pow(pt,p[4])+p[5]);
  return result;
}
