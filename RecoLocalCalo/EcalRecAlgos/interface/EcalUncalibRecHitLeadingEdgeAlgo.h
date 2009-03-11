#ifndef RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitLeadingEdgeAlgo_HH
#define RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitLeadingEdgeAlgo_HH

/** \class EcalUncalibRecHitLeadingEdgeAlgo
  *  Template used to compute amplitude using the leading edge sample
  *
  *  $Id: EcalUncalibRecHitLeadingEdgeAlgo.h
  *  $Date:
  *  $Revision:
  *  \author F. Ferri, M. Malberti
  */

#include "Math/SVector.h"
#include "Math/SMatrix.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecAbsAlgo.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"

// not good to depend on simulation... FIXME
#include "SimCalorimetry/EcalSimAlgos/interface/EcalSimParameterMap.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalShape.h"


template < class C > class EcalUncalibRecHitLeadingEdgeAlgo : public EcalUncalibRecHitRecAbsAlgo < C > {
      public:
	// destructor
	EcalUncalibRecHitLeadingEdgeAlgo < C > () : leadingSample_(0) { };
	virtual ~ EcalUncalibRecHitLeadingEdgeAlgo < C > () { };

        void setLeadingEdgeSample( int isample ) { leadingSample_ = isample; }
        int getLeadingEdgeSample() { return leadingSample_; }

	/// Compute parameters
	virtual EcalUncalibratedRecHit makeRecHit(const C & dataFrame, 
                        const double *pedestals,
                        const double *gainRatios,
                        const EcalWeightSet::EcalWeightMatrix** weights, 
                        const EcalWeightSet::EcalChi2WeightMatrix** chi2Matrix
                        )
        {
		double amplitude_(-1.), pedestal_(-1.), jitter_(-1.), chi2_(-1.);

		// Get time samples
		ROOT::Math::SVector < double, C::MAXSAMPLES > frame;
		int gainId0 = 1;
		int iGainSwitch = 0;
		for (int iSample = 0; iSample < C::MAXSAMPLES; iSample++) {
			int gainId = dataFrame.sample(iSample).gainId();

                        // useless?? FIXME!
			//if (gainId != gainId0) iGainSwitch = 1;
			//if (!iGainSwitch) {
			//	frame(iSample) = double (dataFrame.sample(iSample).adc());
                        //} else {
                        //        frame(iSample) = double (((double) (dataFrame.sample(iSample).adc()) - pedestals[gainId - 1]) * gainRatios[gainId - 1]);
                        //}

			if ( iSample == leadingSample_ ) {
				frame(iSample) =
				    double (((double)
					     (dataFrame.sample( leadingSample_ ).adc() -
					      pedestals[dataFrame.sample( leadingSample_ ).gainId()]) * 
                                              saturationCorrection( iSample, leadingSample_ ) *
					      gainRatios[ dataFrame.sample( leadingSample_ ).gainId() - 1] ));
                                amplitude_ = frame(iSample);
                                pedestal_ = 0;
			}
		}

		return EcalUncalibratedRecHit(dataFrame.id(), amplitude_, pedestal_, jitter_, chi2_);
	}

	// saturation correction  
	double saturationCorrection(int thisSample, int unsaturatedSample)
        {
		// get the EcalShape
		EcalSimParameterMap parameterMap;
		EBDetId barrel(1, 1);
		double thisPhase = parameterMap.simParameters(barrel).timePhase();	// (theShape)(thisPhase) = 1
		EcalShape theShape(thisPhase);
		double tzero = thisPhase - (parameterMap.simParameters(barrel).binOfMaximum() - 1.) * 25.;
		double correction = (theShape) (tzero + thisSample * 25.0) / (theShape) (tzero + unsaturatedSample * 25.0);
		return correction;
	}

        private:
                int leadingSample_;
};
#endif
