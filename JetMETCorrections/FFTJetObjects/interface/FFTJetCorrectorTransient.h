//
// This is the type which gets updated by each level
// of jet corrections
//
// Igor Volobouev
// Aug 3, 2012
//

#ifndef JetMETCorrections_FFTJetObjects_FFTJetCorrectorTransient_h
#define JetMETCorrections_FFTJetObjects_FFTJetCorrectorTransient_h

#include <cmath>
#include "DataFormats/Math/interface/LorentzVector.h"

class FFTJetCorrectorTransient
{
public:
    typedef math::XYZTLorentzVector LorentzVector;

    inline FFTJetCorrectorTransient(const LorentzVector& v,
                                    const double initialScale=1.0,
                                    const double initialSigma=0.0)
        : vec_(v), scale_(initialScale),
          variance_(initialSigma*initialSigma) {}

    inline const LorentzVector& vec() const {return vec_;}
    inline double scale() const {return scale_;}
    inline double sigma() const {return sqrt(variance_);}
    inline double variance() const {return variance_;}

    inline void setVec(const LorentzVector& v) {vec_ = v;}
    inline void setScale(const double s) {scale_ = s;}
    inline void setSigma(const double s) {variance_ = s*s;}
    inline void setVariance(const double v) {variance_ = fabs(v);}

    inline FFTJetCorrectorTransient& operator*=(const double& d)
    {
        // Do not change the sigma -- assume that it is relative to jet Pt
        vec_ *= d;
        scale_ *= d;
        return *this;
    }

private:
    FFTJetCorrectorTransient();

    LorentzVector vec_;
    double scale_;
    double variance_;
};

#endif // JetMETCorrections_FFTJetObjects_FFTJetCorrectorTransient_h
