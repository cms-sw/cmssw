//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: SimpleL5FlavorCorrector.cc,v 1.1 2007/11/16 00:09:58 fedor Exp $
//
// MC Jet Corrector
//
#include "CondFormats/JetMETObjects/interface/SimpleL5FlavorCorrector.h"

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

SimpleL5FlavorCorrector::SimpleL5FlavorCorrector () 
  : mParameters (0) 
{}

SimpleL5FlavorCorrector::SimpleL5FlavorCorrector (const std::string& fDataFile) 
  : mParameters (new SimpleJetCorrectorParameters (fDataFile)) 
{}

SimpleL5FlavorCorrector::~SimpleL5FlavorCorrector () {
  delete mParameters;
}

double SimpleL5FlavorCorrector::correctionPtEta (double fPt, double fEta) const {
  unsigned band = mParameters->bandIndex (fEta);
  unsigned band1 = band;
  double etaMiddle1 = mParameters->record (band).etaMiddle();
  double etaMiddle2 = etaMiddle1;
  unsigned band2 = band1;
  if (fEta < etaMiddle1) { // interpolate to the left
    if (band > 0) {
      --band1;
      etaMiddle1 = mParameters->record (band1).etaMiddle();
    }
  }
  else { // interpolate to the right
    if (band < mParameters->size()-1) {
      ++band2;
      etaMiddle2 = mParameters->record (band2).etaMiddle();
    }
  }

  double correction1 = correctionBandPtEta (band1, fPt, fEta);
  double result = correction1;
  if (band2 != band1) {
    double correction2 = correctionBandPtEta (band2, fPt, fEta);
    // linear interpolation
    result = (correction1*(etaMiddle2-fEta)+correction2*(fEta-etaMiddle1))/(etaMiddle2-etaMiddle1);
  }
  return result;
}

double SimpleL5FlavorCorrector::correctionXYZT (double fPx, double fPy, double fPz, double fE) const {
  XYZTLorentzVectorD p4 (fPx, fPy, fPz, fE);
  return correctionPtEta (p4.Pt(), p4.Eta());
}

double SimpleL5FlavorCorrector::correctionEtEtaPhiP (double fEt, double fEta, double fPhi, double fP) const {
  double costhetainv = cosh (fEta);
  return correctionPtEta (fP/costhetainv, fEta);
}

double SimpleL5FlavorCorrector::correctionBandPtEta (unsigned fBand, double fPt, double fEta) const {
  if (fBand >= mParameters->size()) {
    throw cms::Exception ("SimpleL5FlavorCorrector") 
      << "wrong band: " << fBand << ": only " <<  mParameters->size() << " is available";
  }
  const std::vector<float>& p = mParameters->record (fBand).parameters ();
  if (p.size() != nParameters) {
    throw cms::Exception ("SimpleL5FlavorCorrector") 
      << "wrong # of parameters: " << nParameters << " expected, " << p.size() << " got";
  }
  double pt = (fPt < p[0]) ? p[0] : (fPt > p[1]) ? p[1] : fPt;
  double logPt = log10(pt);
  double result = p[2]+logPt*(p[3]+logPt*p[4]);
  return result;
}

