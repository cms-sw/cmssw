#ifndef _Cluster1D_H_
#define _Cluster1D_H_

#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"

#include <vector>

/**
 *  A generic templated cluster that lives in 1d.
 */

template < class T >
class Cluster1D
{
public:
    Cluster1D (); // needed :-(
    Cluster1D ( const Measurement1D & meas,
              const std::vector < const T * > & tracks, double weight = 1.0 );

    Measurement1D position() const;
    std::vector < const T * > tracks() const;
    double weight() const;
    // bool operator== ( const Cluster1D<T> & other ) const;

private:
    Measurement1D theMeasurement1D;
    std::vector < const T *> theTracks;
    double theWeight;
};

/*
 *                                 implementation
 */

template <class T>
Cluster1D<T>::Cluster1D( const Measurement1D & meas,
                     const std::vector < const T * > & t,
                     double weight ) :
        theMeasurement1D(meas), theTracks(t), theWeight(weight)
{}

template <class T>
Cluster1D<T>::Cluster1D() :
        theMeasurement1D(), theTracks(), theWeight(0.)
{}


template <class T>
std::vector < const T * > Cluster1D<T>::tracks() const
{
    return theTracks;
}

template <class T>
Measurement1D Cluster1D<T>::position() const
{
    return theMeasurement1D;
}

template <class T>
double Cluster1D<T>::weight() const
{
    return theWeight;
}

#endif
