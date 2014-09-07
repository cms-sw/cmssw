#ifndef PulseChiSqSNNLS_h
#define PulseChiSqSNNLS_h

#include "TMatrixD.h"
#include "TVectorD.h"
#include "TMatrixDSym.h"
#include "Math/Minimizer.h"
#include "Math/IFunction.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/TDecompCholFast.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EigenMatrixTypes.h"

#include <set>
#include <array>

class PulseChiSqSNNLS {
  public:
    PulseChiSqSNNLS();
    ~PulseChiSqSNNLS();
    
    
    bool DoFit(const std::vector<double> &samples, const SampleMatrix &samplecor, double pederr, const std::set<int> &bxs, const FullSampleVector &fullpulse, const FullSampleMatrix &fullpulsecov);
    
    const SamplePulseMatrix &pulsemat() const { return _pulsemat; }
    const SampleMatrix &invcov() const { return _invcov; }
    
    const PulseVector &X() const { return _ampvecmin; }
    const PulseVector &Errors() const { return _errvec; }
    
    double ChiSq() const { return _chisq; }
    void disableErrorCalculation() { _computeErrors = false; }

  protected:
    
    bool Minimize(const SampleMatrix &samplecor, double pederr, const std::set<int> &bxs, const FullSampleMatrix &fullpulsecov);
    bool NNLS();
    bool updateCov(const SampleMatrix &samplecor, double pederr, const std::set<int> &bxs, const FullSampleMatrix &fullpulsecov);
    double ComputeChiSq();
    double ComputeApproxUncertainty(unsigned int ipulse);
    
    
    SampleVector _sampvec;
    SampleMatrix _invcov;
    SamplePulseMatrix _pulsemat;
    PulseVector _ampvec;
    PulseVector _errvec;
    PulseVector _ampvecmin;
    
    SampleDecompLDLT _covdecomp;

    std::set<unsigned int> _idxsP;
    std::set<unsigned int> _idxsFixed;
    
    double _chisq;
    bool _computeErrors;
};

#endif
