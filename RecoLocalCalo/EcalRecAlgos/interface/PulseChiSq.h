#ifndef PulseChiSq_h
#define PulseChiSq_h

#include "TMatrixD.h"
#include "TVectorD.h"
#include "TMatrixDSym.h"
#include "Math/Minimizer.h"
#include "Math/IFunction.h"

#include <set>

//class PulseChiSq : public ROOT::Math::IGradientFunctionMultiDim {
class PulseChiSq : public ROOT::Math::IBaseFunctionMultiDim {
  public:
    PulseChiSq(const std::vector<double> &samples, const TMatrixDSym &samplecov, const std::set<int> &bxs, const TVectorD &fullpulse, const TMatrixDSym &fullpulsecov, ROOT::Math::Minimizer &minim);
    unsigned int NDim() const { return _pulsemat.GetNcols(); }
    IBaseFunctionMultiDim *Clone() const { return new PulseChiSq(*this); }
    
    void updateCov(const double *invals, const TMatrixDSym &samplecov, const std::set<int> &bxs, const TMatrixDSym &fullpulsecov);
    
    const TMatrixD &pulsemat() const { return _pulsemat; }
    const TMatrixDSym &invcov() const { return _invcov; }
    
  protected:
    TVectorD _sampvec;
    TMatrixD _pulsemat;
    TMatrixDSym _invcov;
    mutable TVectorD _ampvec;
    mutable TVectorD _workvec;

    
  private:
    double DoEval(const double *invals) const;
};

#endif
