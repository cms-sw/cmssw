#ifndef _MtvClusterizer1D_H_
#define _MtvClusterizer1D_H_

#include "CommonTools/Clustering1D/interface/Clusterizer1D.h"
#include "CommonTools/Clustering1D/interface/Clusterizer1DCommons.h"
#include "CommonTools/Clustering1D/interface/TrivialWeightEstimator.h"
#include "CommonTools/Clustering1D/interface/Clustering1DException.h"

#include <vector>
#include <cmath>
#include <algorithm>

/** Hsm clusterizer in one dimension, originally designed for ApexPoint Finding
 */

template <class T>
class MtvClusterizer1D : public Clusterizer1D<T>
{
public:
    MtvClusterizer1D ( const WeightEstimator<T> & est = TrivialWeightEstimator<T>() );
    MtvClusterizer1D ( const MtvClusterizer1D & );
    ~MtvClusterizer1D();

    std::pair < std::vector < Cluster1D<T> >, std::vector < const T * > >
    operator() ( const std::vector< Cluster1D<T> > & ) const;

    virtual MtvClusterizer1D * clone() const;

private:
    WeightEstimator<T> * theEstimator;
    float theErrorStretchFactor;
};

/*
 *                              --- implementation ---
 *
 */

template <class T>
MtvClusterizer1D<T>::MtvClusterizer1D(
    const MtvClusterizer1D<T> & o ) : theEstimator ( o.theEstimator->clone() )
{}


template <class T>
MtvClusterizer1D<T>::MtvClusterizer1D(
    const WeightEstimator<T> & est ) : theEstimator ( est.clone() )
{}


template <class T>
MtvClusterizer1D<T>::~MtvClusterizer1D()
{
    delete theEstimator;
}

template <class T>
MtvClusterizer1D<T> * MtvClusterizer1D<T>::clone() const
{
    return new MtvClusterizer1D<T> ( * this );
}

template <class T>
std::pair < std::vector < Cluster1D<T> >, std::vector < const T * > >
MtvClusterizer1D<T>::operator() ( const std::vector < Cluster1D<T> > & ov ) const
{
    typedef Cluster1D<T> Cluster1D;
    using namespace Clusterizer1DCommons;
    std::vector < const T * > unusedtracks;
    switch ( ov.size() )
    {
    case 0:
        throw Clustering1DException("[MtvClusterizer1D] no values given" );
    case 1:
        std::pair < std::vector < Cluster1D >, std::vector < const T * > > ret ( ov, unusedtracks );
        return ret;
    };
    std::vector < Cluster1D > v = ov;
    sort ( v.begin(), v.end(), ComparePairs<T>() );
    std::vector < Cluster1D > sols;
    std::vector < const T * > trks;

    typename std::vector< Cluster1D >::iterator cur = v.begin();
    typename std::vector< Cluster1D >::iterator   end = (v.end() - 1 );
    double cur_min = cur->weight() + ( cur+1 )->weight();

    for ( typename std::vector< Cluster1D >::iterator i=v.begin();
            i!=end ; ++i )
    {
        double cur_val = i->weight() + ( i+1 )->weight();
        if ( cur_val > cur_min )
        {
            cur_min = cur_val;
            cur = i;
        };
    };

    double weight = ( cur->weight() + (cur+1)->weight() );
    double est = ( cur->weight() * cur->position().value() +
                   (cur+1)->weight() * (cur+1)->position().value()) / weight;
    double sigma = sqrt ( square ( cur->position().value() - est ) +
                          square ( (cur+1)->position().value() - est ) );
    double err=0.;
    int inliers=0;

    for ( typename std::vector< Cluster1D >::iterator i=v.begin();
            i!=v.end() ; ++i )
    {
        if ( fabs ( i->position().value() - est ) < 3 * sigma )
        {
            // all within 3 sigma are 'in'
            add
                ( i->tracks(), trks );
            err+= square ( i->position().value() - est );
            inliers++;
        }
        else
        {
            add
                ( i->tracks(), unusedtracks );
        };
    };
    err /= ( inliers - 1 ); // the algo definitely produces 2 or more inliers
    err = sqrt ( err );

    sols.push_back ( Cluster1D ( Measurement1D ( est,err ), trks, weight ) );
    std::pair < std::vector < Cluster1D >, std::vector < const T * > > ret ( sols, unusedtracks );
    return ret;
}

#endif
