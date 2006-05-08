#ifndef _TrackDistanceWeightEstimator_H_
#define _TrackDistanceWeightEstimator_H_

#include "CommonTools/Clustering1D/interface/WeightEstimator.h"

/**
 *  weight estimator that uses the distance as the weights.
 */

template <class T>
class TrackDistanceWeightEstimator : public WeightEstimator<T>
{
public:
    double operator() ( const T * track ) const
    {
        return 1.;
    };

    TrackDistanceWeightEstimator * clone() const
    {
        return new TrackDistanceWeightEstimator ( * this );
    };
};

#endif
