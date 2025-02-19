#ifndef _TrivialWeightEstimator_H_
#define _TrivialWeightEstimator_H_

#include "CommonTools/Clustering1D/interface/WeightEstimator.h"

#include <vector>

/**
 * \class TrivialWeightEstimator
 *  trivial WeightEstimator that returns 1.
 */
template <class T>
class TrivialWeightEstimator : public WeightEstimator<T>
{
public:
    double weight ( const std::vector < const T * > & ) const
    {
        return 1.0;
    }

    TrivialWeightEstimator * clone () const
    {
        return new TrivialWeightEstimator<T> ( *this );
    };
};

#endif
