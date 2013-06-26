#ifndef _WeightEstimator_H_
#define _WeightEstimator_H_

#include <vector>

/**
 *  Estimator that returns the weight (="quality") of a cluster.
 *  Abstract base class.
 */

template <class T>
class WeightEstimator
{
public:
    virtual double weight( const std::vector < const T * > & ) const = 0;
    virtual WeightEstimator * clone() const = 0;

    virtual ~WeightEstimator()
    {}
    ;
};

#endif
