#ifndef _MultiClusterizer1D_H_
#define _MultiClusterizer1D_H_

#include "CommonTools/Clustering1D/interface/Clusterizer1D.h"
#include "CommonTools/Clustering1D/interface/Clusterizer1DCommons.h"
#include "CommonTools/Clustering1D/interface/TrivialWeightEstimator.h"
#include "CommonTools/Clustering1D/interface/Clustering1DException.h"
#include "CommonTools/Clustering1D/interface/FsmwClusterizer1D.h"

#include <vector>
#include <map>
#include <cmath>
#include <algorithm>

/** 
 *  A clusterizer that uses a "single" clusterizer iteratively ...
 */

template <class T>
class MultiClusterizer1D : public Clusterizer1D<T>
{
public:
    //    typedef Cluster1D<T> Cluster1D;
    MultiClusterizer1D ( const Clusterizer1D<T> & single,
                      const WeightEstimator<T> & est = TrivialWeightEstimator<T>() );
    MultiClusterizer1D ( const MultiClusterizer1D & );
    ~MultiClusterizer1D();

    std::pair < std::vector < Cluster1D<T> >, std::vector < const T * > >
    operator() ( const std::vector< Cluster1D<T> > & ) const;

    virtual MultiClusterizer1D * clone() const;

private:
    WeightEstimator<T> * theEstimator;
    Clusterizer1D<T> * theSingle;
};

/*
 *                              --- implementation ---
 *
 */

/*
namespace MultiClusterizer1DNameSpace
{
}*/

template <class T>
MultiClusterizer1D<T>::MultiClusterizer1D( const MultiClusterizer1D<T> & o ) 
    : theEstimator( o.theEstimator->clone() ), theSingle ( o.theSingle->clone() )
{}

template <class T>
MultiClusterizer1D<T>::MultiClusterizer1D( const Clusterizer1D<T> & single
    , const WeightEstimator<T> & est ) 
    : theEstimator ( est.clone() ), theSingle ( single.clone() )
{}

template <class T>
MultiClusterizer1D<T>::~MultiClusterizer1D()
{
    delete theEstimator;
    delete theSingle;
}

template <class T>
MultiClusterizer1D<T> * MultiClusterizer1D<T>::clone() const
{
    return new MultiClusterizer1D<T>( *this );
}

template <class T>
std::pair < std::vector< Cluster1D<T> >, std::vector< const T * > >
MultiClusterizer1D<T>::operator() ( const std::vector < Cluster1D<T> > & ov ) const
{
    using namespace Clusterizer1DCommons;
    // using namespace MultiClusterizer1DNameSpace;
    typedef Cluster1D<T> Cluster1D;
    std::vector < const T * > unusedtracks;
    switch ( ov.size() )
    {
    case 0:
       throw Clustering1DException("[MultiClusterizer1D] no values given" );
    case 1:
       std::pair < std::vector < Cluster1D >, std::vector < const T * > > ret ( ov, unusedtracks );
       return ret;
    };

    std::pair < std::vector< Cluster1D >, std::vector< const T * > > res;

    // works only with one track per cluster!!!
    std::map < const T *, Cluster1D > ass;
    std::vector < Cluster1D > cur;

    for ( typename std::vector< Cluster1D >::const_iterator i=ov.begin(); 
          i!=ov.end() ; ++i )
    {
      if ( i->tracks().size()==1 )
      {
        ass[ i->tracks()[0] ]=*i;
      }
      cur.push_back ( *i );
    }

    int ctr=0;
    try {
      while ( true )
      {
        std::pair < std::vector< Cluster1D >, std::vector< const T * > > tmp = (*theSingle)( cur );

        for ( typename std::vector< Cluster1D >::const_iterator i=tmp.first.begin(); 
              i!=tmp.first.end() ; ++i )
        {
          res.first.push_back ( *i );
        }
        res.second=tmp.second;

        cur.clear();

        for ( typename std::vector< const T * >::const_iterator 
              i=res.second.begin(); i!=res.second.end() ; ++i )
        {
          cur.push_back ( ass[*i] );
        }
        if ( ctr++ > 5 ) break;
        if ( cur.size() < 2 ) break;
      }
    } catch ( ... ) {};

    return res;
}

#endif
