#ifndef RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitRatioMethodAlgo_HH
#define RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitRatioMethodAlgo_HH

/** \class EcalUncalibRecHitRatioMethodAlgo
 *  Template used to compute amplitude, pedestal, time jitter, chi2 of a pulse
 *  using a ratio method
 *
 *  $Id: EcalUncalibRecHitRatioMethodAlgo.h,v 1.3 2009/02/20 00:30:38 franzoni Exp $
 *  $Date: 2009/02/20 00:30:38 $
 *  $Revision: 1.3 $
 *  \author A. Ledovskoy (Design) - M. Balazs (Implementation)
 */

#include "Math/SVector.h"
#include "Math/SMatrix.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecAbsAlgo.h"
#include "CondFormats/EcalObjects/interface/EcalWeightSet.h"
#include <vector>

template < class C > class EcalUncalibRecHitRatioMethodAlgo {
      public:
	struct Ratio {
		int index;
		double value;
		double error;
	};
	struct Tmax {
		int index;
		double value;
		double error;
	};
	struct CalculatedRecHit {
		double amplitudeMax;
		double timeMax;
		double timeError;
	};

	virtual ~ EcalUncalibRecHitRatioMethodAlgo < C > () { };
	virtual EcalUncalibratedRecHit makeRecHit(const C & dataFrame,
						  const double *pedestals,
						  const double *gainRatios,
						  std::vector < double >&timeFitParameters,
						  std::vector < double >&amplitudeFitParameters,
						  std::pair < double, double >&timeFitLimits);

        // more function to be able to compute 
        // amplitude and time separately
        void init( const C &dataFrame, const double * pedestals, const double * gainRatios );
        void computeTime(std::vector < double >&timeFitParameters, std::pair < double, double >&timeFitLimits);
        void computeAmplitude( std::vector< double > &amplitudeFitParameters );
        CalculatedRecHit getCalculatedRecHit() { return calculatedRechit_; };

      protected:
	std::vector < double > amplitudes_;
	std::vector < Ratio > ratios_;
	std::vector < Tmax > times_;

	double pedestal_;

	CalculatedRecHit calculatedRechit_;
};

template <class C>
void EcalUncalibRecHitRatioMethodAlgo<C>::init( const C &dataFrame, const double * pedestals, const double * gainRatios )
{
	calculatedRechit_.timeMax = 0;
	calculatedRechit_.amplitudeMax = 0;
	calculatedRechit_.timeError = 0;
	amplitudes_.clear();
	amplitudes_.reserve(C::MAXSAMPLES);
	ratios_.clear();
	ratios_.reserve(C::MAXSAMPLES);
	times_.clear();
	times_.reserve(C::MAXSAMPLES);

	// pedestal obtained from presamples
	pedestal_ = 0;
	short num = 0;
	if (dataFrame.sample(0).gainId() == 1) {
		pedestal_ += double (dataFrame.sample(0).adc());
		num++;
	}
	if (dataFrame.sample(1).gainId() == 1) {
		pedestal_ += double (dataFrame.sample(0).adc());
		num++;
	}
	if (dataFrame.sample(2).gainId() == 1) {
		pedestal_ += double (dataFrame.sample(0).adc());
		num++;
	}

	if (num != 0)
		pedestal_ /= num;
	else
		pedestal_ = 200;

	// ped-subtracted and gain-renormalized samples
	double sample;
	int GainId;
	for (int iSample = 0; iSample < C::MAXSAMPLES; iSample++) {
		GainId = dataFrame.sample(iSample).gainId();

		if (GainId == 1) {
			sample = double (dataFrame.sample(iSample).adc() - pedestal_);
		} else {
			sample = (double (dataFrame.sample(iSample).adc() - pedestals[GainId - 1])) *gainRatios[GainId - 1];
		}
		amplitudes_.push_back(sample);
	}

	// Initial guess for Tmax using "Sum Of Three" method
	int tMax = -1;
	double SumOfThreeSamples = -100;
	for (unsigned int i = amplitudes_.size() - 2; i >= 1; i--) {
		double tmp = amplitudes_[i - 1] + amplitudes_[i] + amplitudes_[i + 1];
		if (tmp >= SumOfThreeSamples) {
			SumOfThreeSamples = tmp;
			tMax = i;
		}
	}

	calculatedRechit_.timeMax = double (tMax);
	double fraction = 0;
	if (SumOfThreeSamples > 0)
		fraction = (amplitudes_[tMax + 1] - amplitudes_[tMax - 1]) / SumOfThreeSamples;
	if (fraction > -1.0 && fraction < 1.0)
		calculatedRechit_.timeMax = calculatedRechit_.timeMax + fraction;

	calculatedRechit_.amplitudeMax = SumOfThreeSamples / 2.72;
}


template<class C>
void EcalUncalibRecHitRatioMethodAlgo<C>::computeTime(std::vector < double >&timeFitParameters,
	    std::pair < double, double >&timeFitLimits)
{
	//////////////////////////////////////////////////////////////
	//                                                          //
	//              RATIO METHOD FOR TIME STARTS HERE           //
	//                                                          //
	//////////////////////////////////////////////////////////////


	// make a vector of Ratios and it's uncertainties
	//
	//       Ratio[i] = Amp[i]/Amp[i+1]
	//       where Amp[i] is pedestal subtracted ADC value in a time sample [i]
	//
	for (unsigned int i = 0; i < amplitudes_.size() - 2; i++) {

		if (amplitudes_[i] > 0.1 && amplitudes_[i + 1] > 0.1) {

			// effective amplitude for these two samples
			double Aeff =
			    1.0 / sqrt(1.0 / (amplitudes_[i] * amplitudes_[i]) +
				       1.0 / (amplitudes_[i + 1] *
					      amplitudes_[i + 1]));

			// ratio 
			double Rtmp = amplitudes_[i] / amplitudes_[i + 1];

			// error due to stat fluctuations of time samples
			double err1 = Rtmp / Aeff;

			// error due to fluctuations of pedestal
			double err2 = (1.0 - Rtmp) / amplitudes_[i + 1];

			if (Aeff > 5.0
			    && (i - calculatedRechit_.timeMax) > -3.1
			    && (i - calculatedRechit_.timeMax) < 1.5
			    && Rtmp >= timeFitLimits.first
			    && Rtmp <= timeFitLimits.second) {
				Ratio currentRatio = { i, Rtmp,
					sqrt(err1 * err1 + err2 * err2)
				};
				ratios_.push_back(currentRatio);
			}
		}
	}


	// make a vector of Tmax measurements that correspond to each
	// ratio. Each measurement have it's value and the error

	double time_max = 0;
	double time_wgt = 0;


	for (unsigned int i = 0; i < ratios_.size(); i++) {

		double time_max_i = ratios_[i].index;

		// calculate polynomial for Tmax

		double u = timeFitParameters[timeFitParameters.size() - 1];
		for (int k = timeFitParameters.size() - 2; k >= 0; k--) {
			u = u * ratios_[i].value + timeFitParameters[k];
		}

		// calculate derivative for Tmax error
		double du =
		    (timeFitParameters.size() -
		     1) * timeFitParameters[timeFitParameters.size() - 1];
		for (int k = timeFitParameters.size() - 2; k >= 1; k--) {
			du = du * ratios_[i].value + k * timeFitParameters[k];
		}


		// running sums for weighted average
		double errorsquared =
		    ratios_[i].error * ratios_[i].error * du * du;
		if (errorsquared > 0) {

			time_max += (time_max_i - u) / errorsquared;
			time_wgt += 1.0 / errorsquared;
			Tmax currentTmax =
			    { ratios_[i].index, (time_max_i - u),
		     sqrt(errorsquared) };
			times_.push_back(currentTmax);

		}

	}

	// calculate weighted average of all Tmax measurements
	if (time_wgt > 0) {
		calculatedRechit_.timeMax = time_max / time_wgt;
		calculatedRechit_.timeError = 1.0 / sqrt(time_wgt);
	}
	// calculate chisquared of all Tmax measurements. These are internal
	// values and they don't get returned

	double chi2one = 0;
	double chi2two = 0;
	for (unsigned int i = 0; i < times_.size(); i++) {
		double dummy =
		    (times_[i].value -
		     calculatedRechit_.timeMax) / times_[i].error;
		chi2one += dummy * dummy;
		chi2two +=
		    (times_[i].value -
		     calculatedRechit_.timeMax) * (times_[i].value -
						   calculatedRechit_.timeMax);
	}
	if (times_.size() > 1) {
		chi2one = chi2one / (times_.size() - 1);
		chi2two = chi2two / (times_.size() - 1);
	}
}

template<class C>
void EcalUncalibRecHitRatioMethodAlgo<C>::computeAmplitude( std::vector< double > &amplitudeFitParameters )
{
	////////////////////////////////////////////////////////////////
	//                                                            //
	//             CALCULATE AMPLITUDE                            //
	//                                                            //
	////////////////////////////////////////////////////////////////


	double alpha = amplitudeFitParameters[0];
	double beta = amplitudeFitParameters[1];

	double sum1 = 0;
	double sum2 = 0;
	for (unsigned int i = 0; i < amplitudes_.size() - 1; i++) {

		double f = 0;
		double termOne = 1 + (i - calculatedRechit_.timeMax) / (alpha * beta);
		if (termOne > 1.e-5) f = exp(alpha * log(termOne)) * exp(-(i - calculatedRechit_.timeMax) / beta);
		if ((f > 0.6 && i <= calculatedRechit_.timeMax)
		    || (f > 0.4 && i >= calculatedRechit_.timeMax)) {
			sum1 += amplitudes_[i] * f;
			sum2 += f * f;
		}

	}
	if (sum2 > 0) {
		calculatedRechit_.amplitudeMax = sum1 / sum2;
        }
}



template < class C > EcalUncalibratedRecHit
    EcalUncalibRecHitRatioMethodAlgo < C >::makeRecHit(const C & dataFrame,
						       const double *pedestals,
						       const double *gainRatios,
						       std::vector < double >&timeFitParameters,
						       std::vector < double >&amplitudeFitParameters,
						       std::pair < double, double >&timeFitLimits)
{

        init( dataFrame, pedestals, gainRatios );
        computeTime( timeFitParameters, timeFitLimits );
        computeAmplitude( amplitudeFitParameters );

	// 1st parameters is id
	//
	// 2nd parameters is amplitude. It is calculated by this method.
	//
	// 3rd parameter is pedestal. It is not calculated. This method
	// relies on input parameters for pedestals and gain ratio. Return
	// zero.
	//
	// 4th parameter is jitter which is a bad choice to call Tmax. It is
	// calculated by this method (in 25 nsec clock units)
	//
	// GF subtract 5 so that jitter==0 corresponds to synchronous hit
	//
	//
	// 5th parameter is chi2. It is possible to calculate chi2 for
	// Tmax. It is possible to calculate chi2 for Amax. However, these
	// values are not very useful without TmaxErr and AmaxErr. This
	// method can return one value for chi2 but there are 4 different
	// parameters that have useful information about the quality of Amax
	// ans Tmax. For now we can return TmaxErr. The quality of Tmax and
	// Amax can be judged from the magnitude of TmaxErr

	return EcalUncalibratedRecHit(dataFrame.id(),
				      calculatedRechit_.amplitudeMax,
                                      pedestal_,
				      calculatedRechit_.timeMax - 5,
				      calculatedRechit_.timeError);
}
#endif
