#include "CondFormats/JetMETObjects/interface/SimpleL2RelativeCorrector.h"
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

SimpleL2RelativeCorrector::SimpleL2RelativeCorrector () 
  : mParameters (0) 
{}

SimpleL2RelativeCorrector::SimpleL2RelativeCorrector (const std::string& fDataFile) 
  : mParameters (new SimpleJetCorrectorParameters (fDataFile)) 
{}

SimpleL2RelativeCorrector::~SimpleL2RelativeCorrector () {
  delete mParameters;
}

double SimpleL2RelativeCorrector::correctionPtEta (double fPt, double fEta) const {
  double result = 1.;
  unsigned band = mParameters->bandIndex (fEta);
  if (band==0 || band==mParameters->size()-1)
    result = correctionBandPtEta (band, fPt, fEta);
  else
    { 
      double etaMiddle[3];
      double etaValue[3];
      for(int i=0; i<3; i++)
        {  
          etaMiddle[i] = mParameters->record (band+i-1).etaMiddle();
          etaValue[i]  = correctionBandPtEta (band+i-1, fPt, fEta);
        }
      result = quadraticInterpolation(fEta,etaMiddle,etaValue);  
    }
  return result;
}

double SimpleL2RelativeCorrector::correctionXYZT (double fPx, double fPy, double fPz, double fE) const {
  XYZTLorentzVectorD p4 (fPx, fPy, fPz, fE);
  return correctionPtEta (p4.Pt(), p4.Eta());
}

double SimpleL2RelativeCorrector::correctionEtEtaPhiP (double fEt, double fEta, double fPhi, double fP) const {
  double costhetainv = cosh (fEta);
  return correctionPtEta (fP/costhetainv, fEta);
}

double SimpleL2RelativeCorrector::correctionBandPtEta (unsigned fBand, double fPt, double fEta) const {
  if (fBand >= mParameters->size()) {
    throw cms::Exception ("SimpleL2RelativeCorrector") 
      << "wrong band: " << fBand << ": only " <<  mParameters->size() << " is available";
  }
  const std::vector<float>& p = mParameters->record (fBand).parameters ();
  if (p.size() != nParameters) {
    throw cms::Exception ("SimpleL2RelativeCorrector") 
      << "wrong # of parameters: " << nParameters << " expected, " << p.size() << " got";
  }
  double pt = (fPt < p[0]) ? p[0] : (fPt > p[1]) ? p[1] : fPt;
  double logpt = log10(pt);
  double result = p[2]+logpt*(p[3]+logpt*(p[4]+logpt*(p[5]+logpt*(p[6]+logpt*p[7]))));
  return result;
}

double SimpleL2RelativeCorrector::quadraticInterpolation(double z, const double x[3], const double y[3]) const
{
  // Quadratic interpolation through the points (x[i],y[i]). First find the parabola that
  // is defined by the points and then calculate the y(z).
  double D[4],a[3];
  D[0] = x[0]*x[1]*(x[0]-x[1])+x[1]*x[2]*(x[1]-x[2])+x[2]*x[0]*(x[2]-x[0]);
  D[3] = y[0]*(x[1]-x[2])+y[1]*(x[2]-x[0])+y[2]*(x[0]-x[1]);
  D[2] = y[0]*(pow(x[2],2)-pow(x[1],2))+y[1]*(pow(x[0],2)-pow(x[2],2))+y[2]*(pow(x[1],2)-pow(x[0],2));
  D[1] = y[0]*x[1]*x[2]*(x[1]-x[2])+y[1]*x[0]*x[2]*(x[2]-x[0])+y[2]*x[0]*x[1]*(x[0]-x[1]);
  if (D[0] != 0)
    {
      a[0] = D[1]/D[0];
      a[1] = D[2]/D[0];
      a[2] = D[3]/D[0];
    }
  else
    {
      a[0] = 0;
      a[1] = 0;
      a[2] = 0;
    }
  double r = a[0]+z*(a[1]+z*a[2]);
  return r;
}
