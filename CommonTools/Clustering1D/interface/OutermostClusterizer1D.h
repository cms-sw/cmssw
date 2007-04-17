#ifndef _OutermostClusterizer1D_H_
#define _OutermostClusterizer1D_H_

#include "CommonTools/Clustering1D/interface/Clusterizer1D.h"
#include "CommonTools/Clustering1D/interface/Clusterizer1DCommons.h"
#include "CommonTools/Clustering1D/interface/TrivialWeightEstimator.h"
#include "CommonTools/Clustering1D/interface/Clustering1DException.h"
#include "CommonTools/Clustering1D/interface/Cluster1DMerger.h"

#include <vector>
#include <cmath>
#include <algorithm>

/** 
 *   Produces two clusters for each end of the 1d data points.
 *   It then puts 50 % of the points in each cluster.
 */

template <class T>
class OutermostClusterizer1D : public Clusterizer1D<T>
{
public:
    /** \param fraction fraction of values that will be considered to be 'in'.
     */
    OutermostClusterizer1D (
                      const WeightEstimator<T> & est = TrivialWeightEstimator<T>() );
    OutermostClusterizer1D ( const OutermostClusterizer1D & );
    ~OutermostClusterizer1D();

    std::pair < std::vector < Cluster1D<T> >, std::vector < const T * > >
    operator() ( const std::vector< Cluster1D<T> > & ) const;

    virtual OutermostClusterizer1D * clone() const;

private:
    WeightEstimator<T> * theEstimator;
};

/*
 *                              --- implementation ---
 *
 */

template <class T>
OutermostClusterizer1D<T>::OutermostClusterizer1D( const OutermostClusterizer1D<T> & o ) 
    : theEstimator( o.theEstimator->clone() )
{}

template <class T>
OutermostClusterizer1D<T>::OutermostClusterizer1D(
    const WeightEstimator<T> & est ) : theEstimator ( est.clone() )
{}

template <class T>
OutermostClusterizer1D<T>::~OutermostClusterizer1D()
{
    delete theEstimator;
}

template <class T>
OutermostClusterizer1D<T> * OutermostClusterizer1D<T>::clone() const
{
    return new OutermostClusterizer1D<T>( *this );
}

template <class T>
std::pair < std::vector< Cluster1D<T> >, std::vector< const T * > >
OutermostClusterizer1D<T>::operator() ( const std::vector < Cluster1D<T> > & ov ) const
{
    using namespace Clusterizer1DCommons;
    typedef Cluster1D<T> Cluster1D;
    std::vector < const T * > unusedtracks;

    switch ( ov.size() )
    {
    case 0:
        throw Clustering1DException("[OutermostClusterizer1D] no values given" );
    case 1:
      {
        std::pair < std::vector < Cluster1D >, std::vector < const T * > > ret ( ov, unusedtracks );
        return ret;
      };
    case 2:
      {
        std::pair < std::vector < Cluster1D >, std::vector < const T * > > ret ( ov, unusedtracks );
        return ret;
      };
    };

    std::vector < Cluster1D > v = ov;
    sort ( v.begin(), v.end(), ComparePairs<T>() );
    std::vector < Cluster1D > sols;
    int sze=v.size()/2;
    Cluster1D tmp = v[0];
    Cluster1DMerger< T > merger ( *theEstimator );
    // merge the inner half to the primary cluster
    for ( typename std::vector< Cluster1D >::const_iterator i=v.begin()+1; i!=v.begin()+sze ; ++i )
    {
      tmp = merger ( tmp, *i );
    }
    sols.push_back ( tmp );
    tmp=v[sze];
    for ( typename std::vector< Cluster1D >::const_iterator i=v.begin()+sze+1; i!=v.end() ; ++i )
    {
      tmp = merger ( tmp, *i );
    }
    sols.push_back ( tmp );

    std::pair < std::vector < Cluster1D >, std::vector < const T * > > ret ( sols, unusedtracks );
    return ret;
}

#endif
