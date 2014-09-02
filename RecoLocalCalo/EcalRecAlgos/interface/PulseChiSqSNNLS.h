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
    
    bool Minimize(const std::vector<double> &samples, const TMatrixDSym &samplecor, double pederr, const std::set<int> &bxs, const TVectorD &fullpulse, const TMatrixDSym &fullpulsecov);
    
    const TMatrixD &pulsemat() const { return _pulsemat; }
    const TMatrixDSym &invcov() const { return _invcov; }
    
    const double *X() const { return _ampvec.GetMatrixArray(); }
    
    double ChiSq() const { return _chisq; }
        
  protected:
    
    bool NNLS();
    bool updateCov(const TMatrixDSym &samplecor, double pederr, const std::set<int> &bxs, const TMatrixDSym &fullpulsecov);
    double ComputeChiSq();
    
    TVectorD _sampvec;
    TMatrixDSym _invcov;
    TVectorD _workvec;
    TMatrixD _pulsemat;
    TVectorD _ampvec;
    TMatrixD _workmat;
    TMatrixD _aTamat;
    TVectorD _wvec;
    TVectorD _aTbvec;
    TMatrixDSym _aPmat;
    TVectorD _sPvec;
    TDecompCholFast _decompP;
    std::array<double,10*10> _pulsematstorage;
    std::array<double,10> _ampvecstorage;
    std::array<double,10*10> _workmatstorage;
    std::array<double,10*10> _aTamatstorage;
    std::array<double,10> _wvecstorage;
    std::array<double,10> _aTbvecstorage;
    std::array<double,10*10> _aPstorage;
    std::array<double,10> _sPstorage;
    std::array<double,10*10> _decompPstorage;
    std::set<unsigned int> _idxsP;
    
    double _chisq;
};

#endif
