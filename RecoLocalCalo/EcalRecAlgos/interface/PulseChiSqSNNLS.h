#ifndef PulseChiSqSNNLS_h
#define PulseChiSqSNNLS_h

#define EIGEN_NO_DEBUG // kill throws in eigen code
#include "RecoLocalCalo/EcalRecAlgos/interface/EigenMatrixTypes.h"

#include <set>
#include <array>

class PulseChiSqSNNLS {
  public:
    
    typedef BXVector::Index Index;
    
    PulseChiSqSNNLS();
    ~PulseChiSqSNNLS();
    
    
    bool DoFit(const SampleVector &samples, const SampleMatrix &samplecor, double pederr, const BXVector &bxs, const FullSampleVector &fullpulse, const FullSampleMatrix &fullpulsecov);
    
    const SamplePulseMatrix &pulsemat() const { return _pulsemat; }
    const SampleMatrix &invcov() const { return _invcov; }
    
    const PulseVector &X() const { return _ampvecmin; }
    const PulseVector &Errors() const { return _errvec; }
    const BXVector &BXs() const { return _bxsmin; }
    
    double ChiSq() const { return _chisq; }
    void disableErrorCalculation() { _computeErrors = false; }
    void setMaxIters(int n) { _maxiters = n;}
    void setMaxIterWarnings(bool b) { _maxiterwarnings = b;}

  protected:
    
    bool Minimize(const SampleMatrix &samplecor, double pederr, const FullSampleMatrix &fullpulsecov);
    bool NNLS();
    bool OnePulseMinimize();
    bool updateCov(const SampleMatrix &samplecor, double pederr, const FullSampleMatrix &fullpulsecov);
    double ComputeChiSq();
    double ComputeApproxUncertainty(unsigned int ipulse);
    
    
    SampleVector _sampvec;
    SampleMatrix _invcov;
    SamplePulseMatrix _pulsemat;
    PulseVector _ampvec;
    PulseVector _errvec;
    PulseVector _ampvecmin;
    
    SampleDecompLLT _covdecomp;
    SampleMatrix _covdecompLinv;
    PulseMatrix _topleft_work;
    PulseDecompLDLT _pulsedecomp;

    BXVector _bxs;
    BXVector _bxsmin;
    unsigned int _npulsetot;
    unsigned int _nP;
    
    SamplePulseMatrix invcovp;
    PulseMatrix aTamat;
    PulseVector aTbvec;
    PulseVector wvec;
    PulseVector updatework;
    
    PulseVector ampvecpermtest;
    
    double _chisq;
    bool _computeErrors;
    int _maxiters;
    bool _maxiterwarnings;
};

#endif
