#ifndef _Cluster1DMerger_H_
#define _Cluster1DMerger_H_

#include "CommonTools/Clustering1D/interface/Cluster1D.h"
#include "CommonTools/Clustering1D/interface/WeightEstimator.h"

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
    for ( typename std::vector< const T * >::const_iterator i=sectracks.begin(); i!=sectracks.end() ; ++i )
    {
        tracks.push_back ( *i );
    };
    float newpos = ( first.position().value() * first.weight() / first.position().error() / first.position().error() +
                     second.position().value() * second.weight() / second.position().error() / second.position().error() ) /
                   ( first.weight() / first.position().error()/ first.position().error() +
                     second.weight() / second.position().error()/ second.position().error() );

    float newerr = sqrt ( first.position().error() * first.position().error() +
                          second.position().error() * second.position().error() );
    float newWeight = theEstimator->weight ( tracks );
    Measurement1D newmeas ( newpos, newerr );
    return Cluster1D<T> ( newmeas, tracks, newWeight );
}

#endif
