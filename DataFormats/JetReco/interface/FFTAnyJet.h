 /** \class reco::FFTAnyJet
 *
 * \short Implements inheritance relationships for FFTJet jets
 *
 * \author Igor Volobouev, TTU
 *
 * \version   $Id: FFTJet.h,v 1.0 2010/11/11 23:09:47 igv Exp $
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
            
        inline const FFTJet<float>& getFFTSpecific() {return fftJetSpecific_;}

    private:
        FFTJet<float> fftJetSpecific_;
    };
}

#endif // DataFormats_JetReco_FFTAnyJet_h
