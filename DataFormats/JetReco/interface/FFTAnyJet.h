 /** \class reco::FFTAnyJet
 *
 * \short Implements inheritance relationships for FFTJet jets
 *
 * \author Igor Volobouev, TTU
 *
 * \version   $Id: FFTAnyJet.h,v 1.2 2011/11/14 20:13:21 igv Exp $
 ************************************************************/

#ifndef DataFormats_JetReco_FFTAnyJet_h
#define DataFormats_JetReco_FFTAnyJet_h

#include "DataFormats/JetReco/interface/FFTJet.h"

namespace reco {
    template<class AnyJet>
    class FFTAnyJet : public AnyJet
    {
    public:
        typedef AnyJet Base;

        inline FFTAnyJet() : AnyJet(), fftJetSpecific_() {}
        inline virtual ~FFTAnyJet() {}

        inline FFTAnyJet(const AnyJet& jet, const FFTJet<float>& fftjet)
            : AnyJet(jet), fftJetSpecific_(fftjet) {}

        inline virtual FFTAnyJet* clone () const 
            {return new FFTAnyJet(*this);}
            
        inline const FFTJet<float>& getFFTSpecific() const
            {return fftJetSpecific_;}

    private:
        FFTJet<float> fftJetSpecific_;
    };
}

#endif // DataFormats_JetReco_FFTAnyJet_h
