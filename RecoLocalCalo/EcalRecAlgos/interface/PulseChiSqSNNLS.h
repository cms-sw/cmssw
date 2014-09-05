#ifndef PulseChiSqSNNLS_h
#define PulseChiSqSNNLS_h

#include "TMatrixD.h"
#include "TVectorD.h"
#include "TMatrixDSym.h"
#include "Math/Minimizer.h"
#include "Math/IFunction.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/TDecompCholFast.h"

#include <set>
#include <array>

class PulseChiSqSNNLS {
  public:
    PulseChiSqSNNLS();
    ~PulseChiSqSNNLS();
    
    
    bool DoFit(const std::vector<double> &samples, const TMatrixDSym &samplecor, double pederr, const std::set<int> &bxs, const TVectorD &fullpulse, const TMatrixDSym &fullpulsecov);
    
    const TMatrixD &pulsemat() const { return _pulsemat; }
    const TMatrixDSym &invcov() const { return _invcov; }
    
    const TVectorD &X() const { return _ampvecmin; }
    const TVectorD &Errors() const { return _errvec; }
    
    double ChiSq() const { return _chisq; }
    void disableErrorCalculation() { _computeErrors = false; }

  protected:
    
    bool Minimize(const TMatrixDSym &samplecor, double pederr, const std::set<int> &bxs, const TMatrixDSym &fullpulsecov);
    bool NNLS();
    bool updateCov(const TMatrixDSym &samplecor, double pederr, const std::set<int> &bxs, const TMatrixDSym &fullpulsecov);
    double ComputeChiSq();
    
    TVectorD _sampvec;
    TMatrixDSym _invcov;
    TVectorD _workvec;
    TMatrixD _pulsemat;
    TVectorD _ampvec;
    TVectorD _ampvecmin;
    TVectorD _errvec;
    TMatrixD _workmat;
    TMatrixD _aTamat;
    TVectorD _wvec;
    TVectorD _aTbvec;
    TVectorD _aTbcorvec;
    TMatrixDSym _aPmat;
    TVectorD _sPvec;
    TDecompCholFast _decompP;
    std::array<double,10*10> _pulsematstorage;
    std::array<double,10> _ampvecstorage;
    std::array<double,10> _ampvecminstorage;
    std::array<double,10> _errvecstorage;
    std::array<double,10*10> _workmatstorage;
    std::array<double,10*10> _aTamatstorage;
    std::array<double,10> _wvecstorage;
    std::array<double,10> _aTbvecstorage;
    std::array<double,10> _aTbcorvecstorage;
    std::array<double,10*10> _aPstorage;
    std::array<double,10> _sPstorage;
    std::array<double,10*10> _decompPstorage;
    std::set<unsigned int> _idxsP;
    std::set<unsigned int> _idxsFixed;
    
    double _chisq;
    bool _computeErrors;
};

#endif
