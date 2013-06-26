#ifndef _Clusterizer1D_H_
#define _Clusterizer1D_H_

#include "CommonTools/Clustering1D/interface/Cluster1D.h"

#include <vector>
#include <utility>

/**
 * \class Clusterizer1D
 *  purely abstract interface to clustering algorithms that operate on
 *  Cluster1D<T>.
 */

template < class T >
class Clusterizer1D
{
public:
    virtual ~Clusterizer1D()
    {}
    ;
    virtual std::pair< std::vector< Cluster1D<T> >, std::vector< const T * > > operator ()
        ( const std::vector< Cluster1D<T> > & ) const = 0;

    virtual Clusterizer1D * clone() const = 0;
};

#endif
