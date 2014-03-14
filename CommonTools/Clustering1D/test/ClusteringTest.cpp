#include "CommonTools/Clustering1D/test/Input.cpp"
#include "CommonTools/Clustering1D/interface/Cluster1DMerger.h"
#include "CommonTools/Clustering1D/interface/TrivialWeightEstimator.h"

#define HaveMtv
#define HaveFsmw
#define HaveDivisive
#ifdef HaveMtv
#include "CommonTools/Clustering1D/interface/MtvClusterizer1D.h"
#endif
#ifdef HaveFsmw
#include "CommonTools/Clustering1D/interface/FsmwClusterizer1D.h"
#endif
#ifdef HaveDivisive
#include "CommonTools/Clustering1D/interface/DivisiveClusterizer1D.h"
#endif

#include <string>
#include <iostream>

using namespace std;

namespace
{
  void print ( const Cluster1D<string> & obj )
  {
      cout << "   Cluster1D ";
      vector < const string * > names = obj.tracks();
      for ( vector< const string * >::iterator nm=names.begin();
              nm!=names.end() ; ++nm )
      {
          cout << **nm;
      };
      cout << " at " << obj.position().value() << " +/- "
           << obj.position().error() << " weight " << obj.weight();
      cout << endl;
  }
    
  void print ( const vector < Cluster1D<string> > & obj )
  {
      for ( vector< Cluster1D<string> >::const_iterator i=obj.begin();
              i!=obj.end() ; ++i )
      {
        print ( *i );
      }
  }

  void print ( const pair < vector < Cluster1D<string> >,
               vector < const string * > > & obj )
  {
      vector < Cluster1D<string > > sol = obj.first;
      for ( vector< Cluster1D<string> >::const_iterator i=sol.begin();
              i!=sol.end() ; ++i )
      {
          cout << "   Cluster1D: ";
          vector < const string * > names = i->tracks();
          for ( vector< const string * >::iterator nm=names.begin();
                  nm!=names.end() ; ++nm )
          {
              cout << **nm;
          };
          cout << " at " << i->position().value() << " +/- "
          << i->position().error() << " weight " << i->weight() << endl;
      }

      cout << " Discarded: ";
      vector <const string * > disc = obj.second;
      for ( vector< const string * >::const_iterator i=disc.begin();
              i!=disc.end() ; ++i )
      {
          cout << **i;
      }
      cout << endl;
  }

  inline void run ( const vector < Cluster1D<string> > & input )
  {
    cout << endl << "Input: " << endl;
    print ( input );
    cout << endl;


    /* Cluster1Dize */
    #ifdef HaveFsmw
    FsmwClusterizer1D<string> fsmw;
    pair < vector < Cluster1D<string> >, vector < const string * > > ret=
        fsmw( input );
    cout << endl << "Fsmw finds: " << endl;
    print ( ret );
    #endif

    #ifdef HaveMtv
    MtvClusterizer1D<string> mtv;
    ret= mtv( input );
    cout << endl << "Mtv finds: " << endl;
    print ( ret );
    #endif

    #ifdef HaveDivisive
    DivisiveClusterizer1D<string> div(50., 1,true, 10.,true);
    ret = div( input );
    cout << endl << "Divisive finds: " << endl;
    print ( ret );
    #endif

    cout << endl;
  }

  void mergingResult ( const Cluster1D<string> & one,
                       const Cluster1D<string> & two )
  {
    cout << "Merger test:" << endl;
    print ( one );
    print ( two );
    
    Cluster1D<string> result = Cluster1DMerger<string>( TrivialWeightEstimator < string > () )
      ( one, two );
    cout << "Merge result: " << endl;
    print ( result );
  }
  
  void mergerTest()
  {
    string one_s="a";
    vector < const string * > one_names;
    one_names.push_back ( &one_s );
    Cluster1D<string> one ( Measurement1D ( 1.0, 0.1 ), one_names, 1.0 );

    
    vector < const string * > two_names;
    string two_s="b";
    two_names.push_back ( &two_s );
    Cluster1D<string> two ( Measurement1D ( 2.0, 0.2 ), two_names, 1.0 );

    mergingResult ( one, two );
  }
}



int main( int argc, char ** argv )
{
    mergerTest();
    /*

    cout << "Three Items:" << endl
              << "============" << endl;
    vector < Cluster1D<string> > input = threeItems();
    run ( input );
    cout << endl
              << "Four Items:" << endl
              << "============" << endl;
    input = fourItems();
    run ( input );
    */

    /*
    cout << endl
              << "Seven:" << endl
              << "============" << endl;
    input = createInput( 7 );
    run ( input );*/
}
