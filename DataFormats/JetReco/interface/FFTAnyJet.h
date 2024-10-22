/** \class reco::FFTAnyJet
 *
 * \short Implements inheritance relationships for FFTJet jets
 *
 * \author Igor Volobouev, TTU
 *
 ************************************************************/

#ifndef DataFormats_JetReco_FFTAnyJet_h
#define DataFormats_JetReco_FFTAnyJet_h

#include "DataFormats/JetReco/interface/FFTJet.h"

namespace reco {
  template <class AnyJet>
  class FFTAnyJet : public AnyJet {
  public:
    typedef AnyJet Base;

    inline FFTAnyJet() : AnyJet(), fftJetSpecific_() {}
    inline ~FFTAnyJet() override {}

    inline FFTAnyJet(const AnyJet& jet, const FFTJet<float>& fftjet) : AnyJet(jet), fftJetSpecific_(fftjet) {}

    inline FFTAnyJet* clone() const override { return new FFTAnyJet(*this); }

    inline const FFTJet<float>& getFFTSpecific() const { return fftJetSpecific_; }

  private:
    FFTJet<float> fftJetSpecific_;
  };
}  // namespace reco

#endif  // DataFormats_JetReco_FFTAnyJet_h
