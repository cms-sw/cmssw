#include "CommonTools/Clustering1D/test/Input.cpp"

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

namespace
{
  void print ( const std::vector < Cluster1D<string> > & obj )
  {
      for ( std::vector< Cluster1D<string> >::const_iterator i=obj.begin();
              i!=obj.end() ; ++i )
      {
          std::cout << "   Cluster1D ";
          std::vector < const string * > names = i->tracks();
          for ( std::vector< const string * >::iterator nm=names.begin();
                  nm!=names.end() ; ++nm )
          {
              std::cout << **nm;
          };
          std::cout << " at " << i->position().value() << " +/- "
          << i->position().error() << " weight " << i->weight();
          std::cout << std::endl;
      }
  }

  void print ( const pair < std::vector < Cluster1D<string> >,
               std::vector < const string * > > & obj )
  {
      std::vector < Cluster1D<string > > sol = obj.first;
      for ( std::vector< Cluster1D<string> >::const_iterator i=sol.begin();
              i!=sol.end() ; ++i )
      {
          std::cout << "   Cluster1D: ";
          std::vector < const string * > names = i->tracks();
          for ( std::vector< const string * >::iterator nm=names.begin();
                  nm!=names.end() ; ++nm )
          {
              std::cout << **nm;
          };
          std::cout << " at " << i->position().value() << " +/- "
          << i->position().error() << " weight " << i->weight() << std::endl;
      }

      std::cout << " Discarded: ";
      std::vector <const string * > disc = obj.second;
      for ( std::vector< const string * >::const_iterator i=disc.begin();
              i!=disc.end() ; ++i )
      {
          std::cout << **i;
      }
      std::cout << std::endl;
  }

  void run ( const std::vector < Cluster1D<string> > & input )
  {
    std::cout << std::endl << "Input: " << std::endl;
    print ( input );
    std::cout << std::endl;


    /* Cluster1Dize */
    #ifdef HaveFsmw
    FsmwClusterizer1D<string> fsmw;
    pair < std::vector < Cluster1D<string> >, std::vector < const string * > > ret=
        fsmw( input );
    std::cout << std::endl << "Fsmw finds: " << std::endl;
    print ( ret );
    #endif

    #ifdef HaveMtv
    MtvClusterizer1D<string> mtv;
    ret= mtv( input );
    std::cout << std::endl << "Mtv finds: " << std::endl;
    print ( ret );
    #endif

    #ifdef HaveDivisive
    DivisiveClusterizer1D<string> div(50., 1,true, 10.,true);
    ret = div( input );
    std::cout << std::endl << "Divisive finds: " << std::endl;
    print ( ret );
    #endif

    std::cout << std::endl;
  }
}



int main( int argc, char ** argv )
{

    std::cout << "Three Items:" << std::endl
              << "============" << std::endl;
    std::vector < Cluster1D<string> > input = threeItems();
    run ( input );
    std::cout << std::endl
              << "Four Items:" << std::endl
              << "============" << std::endl;
    input = fourItems();
    run ( input );
    std::cout << std::endl
              << "Seven:" << std::endl
              << "============" << std::endl;
    input = createInput( 7 );
    run ( input );
}
