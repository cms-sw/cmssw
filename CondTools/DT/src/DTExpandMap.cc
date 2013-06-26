/** \class DTExpandMap
 *
 *  See header file for a description of this class.
 *
 *  $Date: 2012/01/29 11:23:50 $
 *  $Revision: 1.2 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// This Class' Header --
//----------------------
#include "CondTools/DT/interface/DTExpandMap.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

 
//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <fstream>
#include <vector>

class DTMapEntry {

 public:

  DTMapEntry();
  DTMapEntry( int xk1, int xk2, int xk3, int xk4, int xk5,
              int xv1, int xv2, int xv3, int xv4, int xv5, int xv6 );
  ~DTMapEntry();
  int k1;
  int k2;
  int k3;
  int k4;
  int k5;
  int v1;
  int v2;
  int v3;
  int v4;
  int v5;
  int v6;

};

DTMapEntry::DTMapEntry():
 k1( 0 ),
 k2( 0 ),
 k3( 0 ),
 k4( 0 ),
 k5( 0 ),
 v1( 0 ),
 v2( 0 ),
 v3( 0 ),
 v4( 0 ),
 v5( 0 ),
 v6( 0 ) {
}

DTMapEntry::DTMapEntry( int xk1, int xk2, int xk3, int xk4, int xk5,
                        int xv1, int xv2, int xv3, int xv4, int xv5,
                        int xv6 ):
 k1( xk1 ),
 k2( xk2 ),
 k3( xk3 ),
 k4( xk4 ),
 k5( xk5 ),
 v1( xv1 ),
 v2( xv2 ),
 v3( xv3 ),
 v4( xv4 ),
 v5( xv5 ),
 v6( xv6 ) {
}

DTMapEntry::~DTMapEntry() {
}

void DTExpandMap::expandSteering( std::ifstream& file ) {

  std::vector<int> key;
  int k1;
  int k2;
  int k3;
  int k4;
  int k5;
  int v1;
  int v2;
  int v3;
  int v4;
  int v5;
  int v6;

  std::vector<DTMapEntry> entryList;
  while ( file >> k1 >> k2 >> k3 >> k4 >> k5
               >> v1 >> v2 >> v3 >> v4 >> v5 >> v6 ) {
    std::vector<int> key;
    key.push_back( k1 );
    key.push_back( k2 );
    key.push_back( k3 );
    key.push_back( k4 );
    key.push_back( k5 );
    DTMapEntry currentEntry( k1, k2, k3, k4, k5,
                             v1, v2, v3, v4, v5, v6 );
    entryList.push_back( currentEntry );
  }

  int ddu;
  int ros;
  int rch;
  int tdc;
  int tch;
  int whe;
  int sta;
  int sec;
  int rob;
  int qua;
  int lay;
  int cel;
  int mt1;
  int mi1;
  int mt2;
  int mi2;
  int def;
  int wha;
  int sea;
  std::vector<DTMapEntry>::const_iterator iter = entryList.begin();
  std::vector<DTMapEntry>::const_iterator iend = entryList.end();
  std::vector<DTMapEntry>::const_iterator iros = entryList.end();
  std::vector<DTMapEntry>::const_iterator irob = entryList.end();
  while ( iter != iend ) {
    const DTMapEntry& rosEntry( *iter++ );
    if ( rosEntry.k1 > 0x3fffffff ) continue;
    ddu = rosEntry.k1;
    ros = rosEntry.k2;
    whe = rosEntry.v1;
    def = rosEntry.v2;
    sec = rosEntry.v3;
    rob = rosEntry.v4;
    mt1 = rosEntry.v5;
    mi1 = rosEntry.v6;
    iros = entryList.begin();
    while ( iros != iend ) {
      wha = whe;
      sea = sec;
      const DTMapEntry& rchEntry( *iros++ );
      if ( ( rchEntry.k1 != mt1 ) ||
           ( rchEntry.k2 != mi1 ) ) continue;
      rch =  rchEntry.k3;
      if (   rchEntry.v1 != def   ) wha = rchEntry.v1;
      sta =  rchEntry.v2;
      if (   rchEntry.v3 != def   ) sea = rchEntry.v3;
      rob =  rchEntry.v4;
      mt2 =  rchEntry.v5;
      mi2 =  rchEntry.v6;
      irob = entryList.begin();
      while ( irob != iend ) {
        const DTMapEntry& robEntry( *irob++ );
        if ( ( robEntry.k1 != mt2 ) ||
             ( robEntry.k2 != mi2 ) ) continue;
        if (   robEntry.k3 != rob   ) {
          std::cout << "ROB mismatch " << rob << " "
                                       << robEntry.k3 << std::endl;
        }
        tdc =  robEntry.k4;
        tch =  robEntry.k5;
        qua =  robEntry.v4;
        lay =  robEntry.v5;
        cel =  robEntry.v6;
        std::cout << ddu << " "
                  << ros << " "
                  << rch << " "
                  << tdc << " "
                  << tch << " "
                  << wha << " "
                  << sta << " "
                  << sea << " "
                  << qua << " "
                  << lay << " "
                  << cel << std::endl;
      }
    }
  }

  return;

}


