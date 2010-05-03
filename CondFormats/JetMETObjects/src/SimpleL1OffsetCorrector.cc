#include "CondFormats/JetMETObjects/interface/SimpleL1OffsetCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleJetCorrectorParameters.h"
#include <vector>
#include "FWCore/Utilities/interface/Exception.h"
#include "Math/PtEtaPhiE4D.h"
#include "Math/LorentzVector.h"
typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > PtEtaPhiELorentzVectorD;
typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > XYZTLorentzVectorD;

namespace {
  const unsigned nParameters = 3;
}

SimpleL1OffsetCorrector::SimpleL1OffsetCorrector () 
  : mParameters (0) 
{}

SimpleL1OffsetCorrector::SimpleL1OffsetCorrector (const std::string& fDataFile) 
  : mParameters (new SimpleJetCorrectorParameters (fDataFile)) 
{}

SimpleL1OffsetCorrector::~SimpleL1OffsetCorrector () {
  delete mParameters;
}

double SimpleL1OffsetCorrector::correctionEnEta (double fE, double fEta) const {
  int band = mParameters->bandIndex(fEta);
  if (band<0) band = fEta<0 ? 0 : mParameters->size()-1;
  double result = correctionBandEnEta (band, fE, fEta);  
  return result;
}

double SimpleL1OffsetCorrector::correctionXYZT (double fPx, double fPy, double fPz, double fE) const {
  XYZTLorentzVectorD p4 (fPx, fPy, fPz, fE);
  return correctionEnEta (p4.E(), p4.Eta());
}

double SimpleL1OffsetCorrector::correctionEtEtaPhiP (double fEt, double fEta, double fPhi, double fP) const {
  double costhetainv = cosh (fEta);
  return correctionEnEta (fEt*costhetainv, fEta);
}

double SimpleL1OffsetCorrector::correctionBandEnEta (unsigned fBand, double fE, double fEta) const {
  if (fBand >= mParameters->size()) {
    throw cms::Exception ("SimpleL1OffsetCorrector") 
      << "wrong band: " << fBand << ": only " <<  mParameters->size() << " is available";
  }
  const std::vector<float>& p = mParameters->record (fBand).parameters ();
  if (p.size() != nParameters) {
    throw cms::Exception ("SimpleL1OffsetCorrector") 
      << "wrong # of parameters: " << nParameters << " expected, " << p.size() << " got";
  }
  double E = (fE < p[0]) ? p[0] : (fE > p[1]) ? p[1] : fE;
  double result = (E-p[2])/E;
  return result;
}
