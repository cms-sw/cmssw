#ifndef _Cluster1DMerger_H_
#define _Cluster1DMerger_H_

#include "CommonTools/Clustering1D/interface/Cluster1D.h"
#include "CommonTools/Clustering1D/interface/WeightEstimator.h"
#include <cmath>

/**
 *  The class that should always be used to merge
 *  two Cluster1D into a single Cluster1D.
 */

template < class T >
class Cluster1DMerger
{
public:
    Cluster1DMerger ( const WeightEstimator<T> & );
    ~Cluster1DMerger();
    Cluster1DMerger ( const Cluster1DMerger & );
    Cluster1D<T> operator() ( const Cluster1D<T> & first,
                            const Cluster1D<T> & second ) const;
private:
    WeightEstimator<T> * theEstimator;
};

/*
 *                                implementation
 */

template <class T>
Cluster1DMerger<T>::Cluster1DMerger
( const WeightEstimator<T> & est ) : theEstimator ( est.clone() )
{}

template <class T>
Cluster1DMerger<T>::~Cluster1DMerger()
{
    delete theEstimator;
}

template <class T>
Cluster1DMerger<T>::Cluster1DMerger ( const Cluster1DMerger & other ) :
        theEstimator ( other.theEstimator->clone() )
{}

template <class T>
Cluster1D<T> Cluster1DMerger<T>::operator() ( const Cluster1D<T> & first,
        const Cluster1D<T> & second ) const
{
    std::vector < const T * > tracks = first.tracks();
    std::vector < const T * > sectracks = second.tracks();
    for ( typename std::vector< const T * >::const_iterator i=sectracks.begin(); 
          i!=sectracks.end() ; ++i )
    {
        tracks.push_back ( *i );
    };
    double V1=first.position().error() * first.position().error();
    double V2=second.position().error() * second.position().error();
    double C1=first.weight() / V1;
    double C2=second.weight() / V2;

    double newpos = ( first.position().value() * C1 +
                     second.position().value() * C2 ) / ( C1 + C2 );

    double newerr = sqrt ( C1 * C1 * V1 + C2 * C2 * V2 ) / ( C1 + C2 );
    double newWeight = theEstimator->weight ( tracks );

    Measurement1D newmeas ( newpos, newerr );
    return Cluster1D<T> ( newmeas, tracks, newWeight );
}

#endif
