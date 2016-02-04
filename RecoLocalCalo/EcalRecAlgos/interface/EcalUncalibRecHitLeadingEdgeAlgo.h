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

#include "FWCore/MessageLogger/interface/MessageLogger.h"


template < class C > class EcalUncalibRecHitLeadingEdgeAlgo : public EcalUncalibRecHitRecAbsAlgo < C > {
      public:
	// destructor
	EcalUncalibRecHitLeadingEdgeAlgo < C > () : leadingSample_(0), shape_(0) { };
	virtual ~ EcalUncalibRecHitLeadingEdgeAlgo < C > () { };

        void setLeadingEdgeSample( int isample ) { leadingSample_ = isample; }
        int getLeadingEdgeSample() { return leadingSample_; }

        void setPulseShape( std::vector<double> & shape ) { shape_ = shape; }
        std::vector<double> & getPulseShape() { return shape_; }

	/// Compute parameters
	virtual EcalUncalibratedRecHit makeRecHit(const C & dataFrame, 
                        const double *pedestals,
                        const double *gainRatios,
                        const EcalWeightSet::EcalWeightMatrix** weights, 
                        const EcalWeightSet::EcalChi2WeightMatrix** chi2Matrix
                        )
        {
		double amplitude_(-1.), pedestal_(-1.), jitter_(-1.), chi2_(-1.);

		// compute amplitude
		amplitude_ = double (((double) (dataFrame.sample( leadingSample_ ).adc() -
					      pedestals[ dataFrame.sample( leadingSample_ ).gainId() - 1]) * 
					      saturationCorrection( leadingSample_ ) *
					      gainRatios[ dataFrame.sample( leadingSample_ ).gainId() - 1] )); 



		return EcalUncalibratedRecHit(dataFrame.id(), amplitude_, pedestal_, jitter_, chi2_);
	}

	// saturation correction  
	double saturationCorrection(int unsaturatedSample)
        {
                if ( unsaturatedSample > 0 && unsaturatedSample < (int)shape_.size() ) {
                        return 1./ shape_[ unsaturatedSample ];
                } else {
                        edm::LogError("EcalUncalibRecHitLeadingEdgeAlgo") << "Invalid sample " << unsaturatedSample 
                                << " for a shape vector of size " << shape_.size();
                        return 0;
                }
	}

        private:
                int leadingSample_;
                std::vector< double > shape_;
};
#endif
