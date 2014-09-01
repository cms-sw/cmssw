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
    PulseChiSqSNNLS(const std::vector<double> &samples, const TMatrixDSym &samplecov, const std::set<int> &bxs, const TVectorD &fullpulse, const TMatrixDSym &fullpulsecov);
    ~PulseChiSqSNNLS();
    unsigned int NDim() const { return _pulsemat.GetNcols(); }
    
    bool updateCov(const double *invals, const TMatrixDSym &samplecov, const std::set<int> &bxs, const TMatrixDSym &fullpulsecov);
    
    const TMatrixD &pulsemat() const { return _pulsemat; }
    const TMatrixDSym &invcov() const { return _invcov; }
    
    const double *X() const { return _ampvec.GetMatrixArray(); }
    
    bool Minimize();
    
    double ChiSq();
        
  protected:
    TVectorD _sampvec;
    TMatrixD _pulsemat;
    TMatrixDSym _invcov;
    TVectorD _ampvec;
    TVectorD _workvec;
    TMatrixD _workmat;
    TMatrixD _aTamat;
    TVectorD _wvec;
    TVectorD _aTbvec;
    TMatrixDSym _aPmat;
    TVectorD _sPvec;
    TDecompCholFast _decompP;
    std::array<double,10*10> _aPstorage;
    std::array<double,10> _sPstorage;
    std::array<double,10*10> _decompPstorage;
    std::set<unsigned int> _idxsP;
};

#endif
