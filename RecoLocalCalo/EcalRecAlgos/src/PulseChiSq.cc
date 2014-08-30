#include "RecoLocalCalo/EcalRecAlgos/interface/PulseChiSq.h"
#include <math.h>
#include "vdt/vdtMath.h"

PulseChiSq::PulseChiSq(const std::vector<double> &samples, const TMatrixDSym &samplecov, const std::set<int> &bxs, const TVectorD &fullpulse, const TMatrixDSym &fullpulsecov, ROOT::Math::Minimizer &minim) :
  _sampvec(samples.size(),samples.data()),
  _pulsemat(samples.size(),bxs.size()),
  _invcov(samplecov),
  _ampvec(bxs.size()),
  _workvec(samples.size())
{
 
  _invcov.Invert();
  
  const unsigned int nsample = _sampvec.GetNrows();
  
  for (std::set<int>::const_iterator bxit = bxs.begin(); bxit!=bxs.end(); ++bxit) {
    int ipulse = std::distance(bxs.begin(),bxit);
    minim.SetLowerLimitedVariable(ipulse,TString::Format("amp_%i",ipulse).Data(),0.,0.001,0.);
  }
  
  for (std::set<int>::const_iterator bxit = bxs.begin(); bxit!=bxs.end(); ++bxit) {
    int ipulse = std::distance(bxs.begin(),bxit);
    int bx = *bxit;
    int firstsamplet = std::max(0,bx + 3);
    int offset = -3-bx;
        
    for (unsigned int isample = firstsamplet; isample<nsample; ++isample) {
      _pulsemat(isample,ipulse) = fullpulse(isample+offset);
    }
  }
  
  minim.SetFunction(*this);
  
}  


void PulseChiSq::updateCov(const double *invals, const TMatrixDSym &samplecov, const std::set<int> &bxs, const TMatrixDSym &fullpulsecov) {
 
  const unsigned int nsample = _sampvec.GetNrows();
  
  _invcov = samplecov;
  
  for (std::set<int>::const_iterator bxit = bxs.begin(); bxit!=bxs.end(); ++bxit) {
    int ipulse = std::distance(bxs.begin(),bxit);
    int bx = *bxit;
    int firstsamplet = std::max(0,bx + 3);
    int offset = -3-bx;
        
    double ampsq = invals[ipulse]*invals[ipulse];
    for (unsigned int isample = firstsamplet; isample<nsample; ++isample) {
      for (unsigned int jsample = firstsamplet; jsample<nsample; ++jsample) {
        _invcov(isample,jsample) += ampsq*fullpulsecov(isample+offset,jsample+offset);
      }
    }
  }
  _invcov.Invert();
    
}


double PulseChiSq::DoEval(const double *invals) const {
  
  _ampvec.SetElements(invals);
  _workvec = _sampvec - _pulsemat*_ampvec;
  return _invcov.Similarity(_workvec);    
  
}
