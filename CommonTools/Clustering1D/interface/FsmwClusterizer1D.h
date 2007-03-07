#ifndef _FsmwClusterizer1D_H_
#define _FsmwClusterizer1D_H_

#include "CommonTools/Clustering1D/interface/Clusterizer1D.h"
#include "CommonTools/Clustering1D/interface/Clusterizer1DCommons.h"
#include "CommonTools/Clustering1D/interface/TrivialWeightEstimator.h"
#include "CommonTools/Clustering1D/interface/Clustering1DException.h"

#include <vector>
#include <cmath>
#include <algorithm>

/** Fraction-of sample mode with weights clustering
 */

template <class T>
class FsmwClusterizer1D : public Clusterizer1D<T>
{
public:
    /** \param fraction fraction of values that will be considered to be 'in'.
     */
    FsmwClusterizer1D ( double fraction = .05, double n_sigma_in = 3.,
                      const WeightEstimator<T> & est = TrivialWeightEstimator<T>() );
    FsmwClusterizer1D ( const FsmwClusterizer1D & );
    ~FsmwClusterizer1D();

    std::pair < std::vector < Cluster1D<T> >, std::vector < const T * > >
    operator() ( const std::vector< Cluster1D<T> > & ) const;

    virtual FsmwClusterizer1D * clone() const;

private:
    WeightEstimator<T> * theEstimator;
    double theFraction;
    double theNSigmaIn;
};

/*
 *                              --- implementation ---
 *
 */

namespace FsmwClusterizer1DNameSpace
{
/*
 *  Function that computes the 'fraction-of sample mode with weights'.
 *  The modefinder, that is.
 *  Warning, values have to be sorted in this method!!
 */
template <class T>
std::pair < typename std::vector< Cluster1D<T> >::const_iterator,
typename std::vector< Cluster1D<T> >::const_iterator >
fsmw ( const std::vector< Cluster1D<T> > & values, double fraction )
{
    typedef Cluster1D<T> Cluster1D;
    typename std::vector< Cluster1D >::const_iterator begin = values.begin();
    typename std::vector< Cluster1D >::const_iterator end = values.end()-1;

    while (1)
    {
#ifdef FsmwClusterizer1DDebug
        cout << "Begin at " << begin->position().value() << endl;
#endif

        const int size = (int) (end-begin);
#ifdef FsmwClusterizer1DDebug

        cout << "Size " << size << endl;
#endif

        int stepsize = (int) floor ( ( 1+ size ) * fraction );
        if ( stepsize == 0 )
            stepsize=1;
#ifdef FsmwClusterizer1DDebug

        cout << "Old end at " << end->position().value() << endl;
#endif

        end=begin+stepsize;
        typename std::vector< Cluster1D >::const_iterator new_begin = begin;
        typename std::vector< Cluster1D >::const_iterator new_end = end;

#ifdef FsmwClusterizer1DDebug

        cout << "New end at " << end->position().value() << endl;
        cout << "stepsize " << stepsize << endl;
#endif

        // Old version: used the weights of just the end points
        // double totalweight = begin->weight() + end->weight();

        // new version: sums up the weights of all points involved
        // _including_ the "end" point
        double totalweight = end->weight();
        for ( typename std::vector< Cluster1D >::const_iterator w=begin; w!=end ; ++w )
        {
            totalweight+=w->weight();
        };

        double div=fabs ( end->position().value() - begin->position().value() ) /
                   totalweight;
#ifdef FsmwClusterizer1DDebug

        cout << "Div at " << begin->position().value() << ":" << (end)->position().value()
        << " = " << div << endl;
#endif

        for ( typename std::vector< Cluster1D >::const_iterator i = (begin + 1);
                i!=(begin + size - stepsize + 1); ++i )
        {
            // FIXME wrong
            // double tmpweight = i->weight() + (i+stepsize)->weight();
            //
            // new version: sums up the weights of all points in the interval
            // _including_ the end point (i+stepsize)
            double tmpweight = 0.;
            for ( typename std::vector< Cluster1D >::const_iterator wt=i; wt!=(i+stepsize+1); ++wt )
            {
                tmpweight+=wt->weight();
            };

            double tmpdiv = fabs( i->position().value() - (i+stepsize)->position().value() )
                            / tmpweight;
#ifdef FsmwClusterizer1DDebug

            cout << "Div at " << i->position().value() << ":" << (i+stepsize)->position().value()
            << " = " << tmpdiv << endl;
#endif

            if ( tmpdiv < div)
            {
                new_begin= i;
                new_end = i+stepsize;
                div= tmpdiv;
            };
        };
#ifdef FsmwClusterizer1DDebug

        cout << "---- new interval: " << new_begin->position().value()
        << ":" << new_end->position().value() << endl;
#endif

        begin = new_begin;
        end = new_end;
        if ( size < 4 )
            break;
    };

    std::pair < typename std::vector< Cluster1D >::const_iterator,
    typename std::vector< Cluster1D >::const_iterator > ret ( begin, end );
    return ret;
}
}

template <class T>
FsmwClusterizer1D<T>::FsmwClusterizer1D( const FsmwClusterizer1D<T> & o ) 
    : theEstimator( o.theEstimator->clone() ), theFraction ( o.theFraction ),
    theNSigmaIn ( o.theNSigmaIn )
{}


template <class T>
FsmwClusterizer1D<T>::FsmwClusterizer1D( double fraction, double nsig, const WeightEstimator<T> & est ) 
    : theEstimator ( est.clone() ), theFraction ( fraction ), theNSigmaIn ( nsig )
{}


template <class T>
FsmwClusterizer1D<T>::~FsmwClusterizer1D()
{
    delete theEstimator;
}

template <class T>
FsmwClusterizer1D<T> * FsmwClusterizer1D<T>::clone() const
{
    return new FsmwClusterizer1D<T>( *this );
}

template <class T>
std::pair < std::vector< Cluster1D<T> >, std::vector< const T * > >
FsmwClusterizer1D<T>::operator() ( const std::vector < Cluster1D<T> > & ov ) const
{
    using namespace Clusterizer1DCommons;
    using namespace FsmwClusterizer1DNameSpace;
    typedef Cluster1D<T> Cluster1D;
    std::vector < const T * > unusedtracks;

    switch ( ov.size() )
    {
    case 0:
        throw Clustering1DException("[FsmwClusterizer1D] no values given" );
    case 1:
	std::pair < std::vector < Cluster1D >, std::vector < const T * > > ret ( ov, unusedtracks );
        return ret;
    };

    std::vector < Cluster1D > v = ov;
    sort ( v.begin(), v.end(), ComparePairs<T>() );
    std::vector < Cluster1D > sols;

    std::pair < typename std::vector< Cluster1D >::const_iterator,
    typename std::vector< Cluster1D >::const_iterator > estors
    = fsmw ( v, theFraction );

    double weight = estors.first->weight() + estors.second->weight();
    double est = ( estors.first->weight() * estors.first->position().value() +
                   estors.second->weight() * estors.second->position().value() ) /
                 weight;
    double err=0.;
    double sigma = sqrt ( square ( estors.first->position().value() - est ) +
                          square ( estors.second->position().value() - est ));
    /*
    std::cout << "[FsmwClusterizer1D] first=" << estors.first->position().value()
              << " second=" << estors.second->position().value()
              << " est=" << est << std::endl;
    double sigma = sqrt ( square ( estors.first->position().error() ) +
                          square ( estors.second->position().error() ) );
    double sigma = estors.first->position().error();
                          */
    std::vector < const T * > trks;
    int inliers=0;

    for ( typename std::vector< Cluster1D >::iterator i=v.begin();
            i!=v.end() ; ++i )
    {
        /*
        std::cout << "[FsmwClusterizer1D] see if they're in: delta="
                  << 10000 * fabs ( i->position().value() - est )
                  << " sigma=" << 10000 * sigma << std::endl;
         */
        if ( fabs ( i->position().value() - est ) < theNSigmaIn * sigma )
        {
            // all within theNSigmaIn sigma are 'in'
            add ( i->tracks(), trks );
            err+= square ( i->position().value() - est );
            inliers++;
        } else {
            add ( i->tracks(), unusedtracks );
        };
    };
    err /= ( inliers - 1 ); // the algo definitely produces 2 or more inliers
    err = sqrt ( err );


    err=sqrt(err);
    sols.push_back ( Cluster1D ( Measurement1D ( est, err ), trks, weight ) );

    std::pair < std::vector < Cluster1D >, std::vector < const T * > > ret ( sols, unusedtracks );
    return ret;
}

#endif
