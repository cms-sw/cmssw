//
// This is the type returned by jet correctors
//
// Igor Volobouev
// Aug 3, 2012
//

#ifndef JetMETCorrections_FFTJetObjects_FFTJetCorrectorResult_h
#define JetMETCorrections_FFTJetObjects_FFTJetCorrectorResult_h

#include "DataFormats/Math/interface/LorentzVector.h"

class FFTJetCorrectorResult
{
public:
    typedef math::XYZTLorentzVector LorentzVector;

    inline FFTJetCorrectorResult(const LorentzVector& v,
                                 const double correctionScale,
                                 const double systematicUncertainty)
        : vec_(v), scale_(correctionScale), sigma_(systematicUncertainty) {}

    inline const LorentzVector& vec() const {return vec_;}
    inline double scale() const {return scale_;}
    inline double sigma() const {return sigma_;}

    inline void setVec(const LorentzVector& v) {vec_ = v;}
    inline void setScale(const double s) {scale_ = s;}
    inline void setSigma(const double s) {sigma_ = s;}

private:
    FFTJetCorrectorResult();

    LorentzVector vec_;
    double scale_;
    double sigma_;
};

#endif // JetMETCorrections_FFTJetObjects_FFTJetCorrectorResult_h
