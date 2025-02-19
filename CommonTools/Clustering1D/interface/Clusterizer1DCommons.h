#ifndef _Clusterizer1DCommons_H_
#define _Clusterizer1DCommons_H_

#include "CommonTools/Clustering1D/interface/Cluster1D.h"

namespace Clusterizer1DCommons
{
// typedef Clusterizer1DCommons::Cluster1D Cluster1D;
inline double square ( const double a )
{
    return a*a;
}

template <class T>
struct ComparePairs
{
    bool operator() ( const Cluster1D<T> & c1,
                      const Cluster1D<T> & c2 )
    {
        return ( c1.position().value() < c2.position().value() );
    };
};

template <class T>
void add
    ( const std::vector < const T * > & source,
            std::vector < const T * > & dest )
{
    for ( typename std::vector< const T * >::const_iterator i=source.begin();
            i!=source.end() ; ++i )
    {
        dest.push_back ( *i );
    };
}

} // namespace Clusterizer1DCommons

#endif
