/** \class reco::FFTJetPileupSummary
*
* \short Summary info for pile-up determined by Gaussian filtering
*
* \author Igor Volobouev, TTU
*
* \version  $Id: FFTJetPileupSummary.h,v 1.1 2011/04/26 22:57:31 igv Exp $
************************************************************/

#ifndef DataFormats_JetReco_FFTJetPileupSummary_h
#define DataFormats_JetReco_FFTJetPileupSummary_h

namespace reco {
    class FFTJetPileupSummary
    {
    public:
        inline FFTJetPileupSummary()
            : uncalibratedQuantile_(-100000.f), pileupRho_(-1000000.f),
              pileupRhoUncert_(-1.0), uncertaintyCode_(-1) {}

        inline FFTJetPileupSummary(const float uncalibrated,
                                   const float pileup,
                                   const float uncert = -1.f,
                                   const int code = -1)
            : uncalibratedQuantile_(uncalibrated), pileupRho_(pileup),
              pileupRhoUncert_(uncert), uncertaintyCode_(code)  {}

        // The original point at which the pile-up estimate was found.
        // This does not use any calibration curve or Neyman construction,
        // and can serve as an input to an improved user-defined calibration.
        inline float uncalibratedQuantile() const 
            {return uncalibratedQuantile_;}

        // The estimate of pile-up transverse energy (or momentum) density
        inline float pileupRho() const {return pileupRho_;}

        // Uncertainty of the pile-up density estimate
        inline float pileupRhoUncertainty() const {return pileupRhoUncert_;}

        // The "zone" of the uncertainty in the Neyman belt construction.
        // Suggested values are as follows:
        //
        // -1 -- uncertainty is unknown
        //
        //  0 -- estimated uncertainty does not come from the Neyman belt
        //
        //  1 -- the estimate does not intersect the belt at all (typically,
        //       the uncertainty in this case will be set to 0). This just
        //       means that your value of rho is unlikely, and there is no
        //       way to come up with a resonable frequentist uncertainty
        //       estimate using the adopted ordering principle.
        //
        //  2 -- the estimate intersects one error band only. The height
        //       of that band is used as the uncertainty.
        //
        //  3 -- the estimate intersects the center of the belt (the
        //       calibration curve) and one error band. The distance
        //       between the center and the band (one-sided uncertainty)
        //       is used as the uncertainty in this summary.
        //
        //  4 -- the estimate intersects the complete belt. Only in this
        //       case one gets a completely meanigful frequentist uncertainty
        //       which is typicaly calculated as the belt half-width along
        //       the line of intersect.
        //
        inline int uncertaintyCode() const {return uncertaintyCode_;}

    private:
        float uncalibratedQuantile_;
        float pileupRho_;
        float pileupRhoUncert_;
        int uncertaintyCode_;
    };
}

#endif // DataFormats_JetReco_FFTJetPileupSummary_h
