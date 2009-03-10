#include "CondFormats/JetMETObjects/interface/SimpleL3PFAbsoluteCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleJetCorrectorParameters.h"
#include <vector>
#include "FWCore/Utilities/interface/Exception.h"
#include "Math/PtEtaPhiE4D.h"
#include "Math/LorentzVector.h"
typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > PtEtaPhiELorentzVectorD;
typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > XYZTLorentzVectorD;

namespace {
  const unsigned nParameters = 8;
}

SimpleL3PFAbsoluteCorrector::SimpleL3PFAbsoluteCorrector () 
  : mParameters (0) 
{}

SimpleL3PFAbsoluteCorrector::SimpleL3PFAbsoluteCorrector (const std::string& fDataFile) 
  : mParameters (new SimpleJetCorrectorParameters (fDataFile)) 
{}

SimpleL3PFAbsoluteCorrector::~SimpleL3PFAbsoluteCorrector () {
  delete mParameters;
}

double SimpleL3PFAbsoluteCorrector::correctionPtEta (double fPt, double fEta) const {
  int band = mParameters->bandIndex(fEta);
  if (band<0) band = fEta<0 ? 0 : mParameters->size()-1;
  double result = correctionBandPtEta (band, fPt, fEta);  
  return result;
}

double SimpleL3PFAbsoluteCorrector::correctionXYZT (double fPx, double fPy, double fPz, double fE) const {
  XYZTLorentzVectorD p4 (fPx, fPy, fPz, fE);
  return correctionPtEta (p4.Pt(), p4.Eta());
}

double SimpleL3PFAbsoluteCorrector::correctionEtEtaPhiP (double fEt, double fEta, double fPhi, double fP) const {
  double costhetainv = cosh (fEta);
  return correctionPtEta (fP/costhetainv, fEta);
}

double SimpleL3PFAbsoluteCorrector::correctionBandPtEta (unsigned fBand, double fPt, double fEta) const {
  if (fBand >= mParameters->size()) {
    throw cms::Exception ("SimpleL3PFAbsoluteCorrector") 
      << "wrong band: " << fBand << ": only " <<  mParameters->size() << " is available";
  }
  const std::vector<float>& p = mParameters->record (fBand).parameters ();
  if (p.size() != nParameters) {
    throw cms::Exception ("SimpleL3AbsoluteCorrector") 
      << "wrong # of parameters: " << nParameters << " expected, " << p.size() << " got";
  }
  double pt = (fPt < p[0]) ? p[0] : (fPt > p[1]) ? p[1] : fPt;
  double z = log10(pt);
  double result = p[2]+p[3]/(z*z+p[4])+p[5]*exp(-p[6]*(z-p[7])*(z-p[7]));
  return result;
}
