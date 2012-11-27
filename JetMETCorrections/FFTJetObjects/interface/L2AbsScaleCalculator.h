#ifndef JetMETCorrections_FFTJetObjects_L2AbsScaleCalculator_h
#define JetMETCorrections_FFTJetObjects_L2AbsScaleCalculator_h

#include <cmath>
#include <cassert>

#include "JetMETCorrections/FFTJetObjects/interface/AbsFFTSpecificScaleCalculator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

class L2AbsScaleCalculator : public AbsFFTSpecificScaleCalculator
{
public:
    inline explicit L2AbsScaleCalculator(const edm::ParameterSet& ps) 
        : m_radiusFactor(ps.getParameter<double>("radiusFactor")),
          m_zeroPtLog(ps.getParameter<double>("zeroPtLog")),
          m_takePtLog(ps.getParameter<bool>("takePtLog")) {}

    inline virtual ~L2AbsScaleCalculator() {}

    inline virtual void mapFFTJet(const reco::Jet& /* jet */,
                                  const reco::FFTJet<float>& fftJet,
                                  const math::XYZTLorentzVector& current,
                                  double* buf, const unsigned dim) const
    {
        if (dim != 2)
            throw cms::Exception("FFTJetBadConfig")
                << "In L2AbsScaleCalculator::mapFFTJet: "
                << "invalid table dimensionality: "
                << dim << std::endl;
        assert(buf);
        const double radius = fftJet.f_recoScale();
        const double pt = current.pt();
        buf[0] = radius*m_radiusFactor;
        if (m_takePtLog)
        {
            if (pt > 0.0)
                buf[1] = log(pt);
            else
                buf[1] = m_zeroPtLog;
        }
        else
            buf[1] = pt;
    }

private:
    double m_radiusFactor;
    double m_zeroPtLog;
    bool m_takePtLog;
};

#endif // JetMETCorrections_FFTJetObjects_L2AbsScaleCalculator_h
