#ifndef JetMETCorrections_FFTJetObjects_FFTJetCorrector_h
#define JetMETCorrections_FFTJetObjects_FFTJetCorrector_h

#include <vector>
#include <boost/shared_ptr.hpp>

#include "JetMETCorrections/FFTJetObjects/interface/AbsFFTJetAdjuster.h"
#include "JetMETCorrections/FFTJetObjects/interface/AbsFFTJetScaleCalculator.h"
#include "JetMETCorrections/FFTJetObjects/interface/FFTJetCorrectorApp.h"

template<class Jet, class Adjustable>
class FFTJetCorrector
{
public:
    typedef Jet jet_type;
    typedef Adjustable adjustable_type;
    typedef AbsFFTJetScaleCalculator<jet_type, adjustable_type> AbsScaler;
    typedef AbsFFTJetAdjuster<jet_type, adjustable_type> AbsAdjuster;

    inline FFTJetCorrector(boost::shared_ptr<const AbsAdjuster> adjuster,
                           const std::vector<boost::shared_ptr<const AbsScaler> >& scalers,
                           const unsigned i_level, const FFTJetCorrectorApp a)
        : adjuster_(adjuster), scalers_(scalers),
          level_(i_level), app_(a) {}

    inline void correct(const Jet& jet, const bool isMC,
                        const Adjustable& in, Adjustable* out) const
    {
        if ((isMC && app_ == FFTJetCorrectorApp::DATA_ONLY) ||
            (!isMC && app_ == FFTJetCorrectorApp::MC_ONLY))
            // Do not need to apply this corrector. Simply copy
            // the transient data from the input to the output.
            *out = in;
        else
        {
            const unsigned nAdj = scalers_.size();
            //If we only need a small buffer, use one on the stack
            // else we will use a std::vector
            constexpr size_t kStaticBufferSize = 10;
            double staticBuffer[kStaticBufferSize];
            std::vector<double> dynamicBuffer;
            double* buf = nullptr;
            if(nAdj<=kStaticBufferSize) {
                buf = &(staticBuffer[0]);
            } else {
                dynamicBuffer.resize(nAdj);
                buf = &(dynamicBuffer[0]);
            }
            for (unsigned i=0; i<nAdj; ++i)
                buf[i] = scalers_[i]->scale(jet, in);
            adjuster_->adjust(jet, in, buf, nAdj, out);
        }
    }

    inline unsigned level() const {return level_;}
    inline FFTJetCorrectorApp app() const {return app_;}

private:
    boost::shared_ptr<const AbsAdjuster> adjuster_;
    std::vector<boost::shared_ptr<const AbsScaler> > scalers_;
    unsigned level_;
    FFTJetCorrectorApp app_;
};

#endif // JetMETCorrections_FFTJetObjects_FFTJetCorrector_h
