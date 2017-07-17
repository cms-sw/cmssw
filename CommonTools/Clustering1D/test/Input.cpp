#include "CommonTools/Clustering1D/interface/Clusterizer1DCommons.h"

#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <sstream>

using namespace std;

namespace
{
inline Cluster1D<string> createCluster ( float pos, float err, float weight, string name )
{
    vector < const string * > names;
    names.push_back ( new string(name) );
    Cluster1D<string> ret ( Measurement1D ( pos, err ), names, weight );
    return ret;
}

char nameCtr='a';

void resetClusterName()
{
    nameCtr='a';
}

Cluster1D<string> createCluster ( float pos, float err, float weight )
{
    vector < const string * > names;
    char fld[2];
    fld[0]=nameCtr;
    fld[1]='\0';
    names.push_back ( new string( fld ) );
    nameCtr++;
    Cluster1D<string> ret ( Measurement1D ( pos, err ), names, weight );
    return ret;
}

inline void deleteCluster ( vector < Cluster1D<string> > & clus )
{
    cout << "[Deleting Sequence] ..." << flush;
    for ( vector< Cluster1D<string> >::const_iterator i=clus.begin();
            i!=clus.end() ; ++i )
    {
        vector < const string * > names = i->tracks();
        for ( vector< const string * >::const_iterator nm=names.begin();
                nm!=names.end() ; ++i )
        {
            delete *nm;
        };
    };
    cout << " done." << endl;
}

// makes sure that the leftmost cluster is called 'a'
// and so on
vector < Cluster1D < string > > sortCluster (
    const vector < Cluster1D < string > > & in )
{
    vector < Cluster1D < string > > ret;
    vector < Cluster1D < string > > tmp = in;
    partial_sort_copy ( in.begin(), in.end(), tmp.begin(), tmp.end(),
                        Clusterizer1DCommons::ComparePairs<string> () );
    resetClusterName();
    for ( vector< Cluster1D<string> >::const_iterator i=tmp.begin(); i!=tmp.end() ; ++i )
    {
        ret.push_back ( createCluster ( i->position().value() , i->position().error(),
                                        i->weight() ) );
    };
    return ret;
}

vector < Cluster1D<string> > trivialInput()
{
    vector < Cluster1D<string> > ret;
    ret.push_back ( createCluster ( 0.1, 0.05, .5 ) );
    ret.push_back ( createCluster ( 5.0, 0.32, .4 ) );
    ret.push_back ( createCluster ( 10.0, 0.22, .7 ) );
    return ret;
}

vector < Cluster1D<string> > threeItems()
{
    vector < Cluster1D<string> > ret;
    ret.push_back ( createCluster ( 0.1, 0.05, .8 ) );
    ret.push_back ( createCluster ( 5.0, 0.32, .9 ) );
    ret.push_back ( createCluster ( 10.0, 0.22, .6 ) );
    return ret;
}

vector < Cluster1D<string> > fourItems()
{
    vector < Cluster1D<string> > ret;
    ret.push_back ( createCluster ( 0.1, 0.05, .6 ) );
    ret.push_back ( createCluster ( 5.0, 0.32, .7 ) );
    ret.push_back ( createCluster ( 10.0, 0.22, .5 ) );
    ret.push_back ( createCluster ( 12.0, 0.22, .8 ) );
    return ret;
}

/**
 *  User calls this, or the next.
 *  In this function the user gives the name of
 *  the "secnario"
 */
inline vector < Cluster1D<string> > createInput( string name )
{
    // that's a map that maps the function to ordinary names.
    map < string, vector < Cluster1D<string> > (*)() > inputs;
    inputs["Trivial"]=trivialInput;
    inputs["Three"]=threeItems;
    inputs["Four"]=fourItems;

    // give me the pointer to the function that implements the
    // "scenario"
    vector < Cluster1D<string> > (*addr)() = inputs[name];
    if (addr)
    {
        return ((*addr))();
        // vector < Cluster1D < string > > tmp = ((*addr))();
        // now make sure that the leftmost cluster is called 'a'
        // and so on
        // return sortCluster ( tmp );
    };

    cout << "[Input.cc] input " << name << " unknown" << endl;
    exit(-1);
}

/**
 *  The user gives the number of clusters
 */
inline vector < Cluster1D<string> > createInput ( int n )
{
    vector < Cluster1D<string> > ret;
    for ( int i=0; i< n ; ++i )
    {
//        ret.push_back ( createCluster (  RandGauss::shoot( 0., 5. ),
//                                         RandFlat::shoot(0., .05),
//                                         pow ( RandFlat::shoot (0., .1 ), -2 ) ) );
        ret.push_back ( createCluster (  5.00*drand48(),
                                         0.05*drand48(),
                                         pow ( 0.1*drand48(), -2 ) ) );
    };
    return sortCluster ( ret );
    //    sort ( ret.begin(), ret.end(), ClusterizerCommons::ComparePairs<string>() );
    //    return ret;
}

}
