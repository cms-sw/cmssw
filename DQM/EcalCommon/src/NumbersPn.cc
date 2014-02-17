// $Id: NumbersPn.cc,v 1.9 2010/08/08 08:46:05 dellaric Exp $

/*!
  \file NumbersPn.cc
  \brief Some "id" conversions
  \version $Revision: 1.9 $
  \date $Date: 2010/08/08 08:46:05 $
*/

#include <sstream>
#include <iomanip>

#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "DQM/EcalCommon/interface/NumbersPn.h"

//-------------------------------------------------------------------------

// return the PN index [0-79] from EcalPnDiodeDetId.id().iPnId() [0-9];
int NumbersPn::ipnEE( const int ism, const int ipnid ) throw( std::runtime_error ) {
  
  if( ism >=1 && ism <= 18 ) {

    int myFED = -1;

    if( ism ==  1 ) myFED = 0;
    if( ism ==  2 ) myFED = 1;
    if( ism ==  5 ) myFED = 2;
    if( ism ==  6 ) myFED = 3;
    if( ism == 10 ) myFED = 4;
    if( ism == 11 ) myFED = 5;
    if( ism == 14 ) myFED = 6;
    if( ism == 15 ) myFED = 7;

    return 10*myFED + (ipnid-1);

  } else {

    std::ostringstream s;
    s << "Wrong SM id determination: iSM = " << ism;
    throw( std::runtime_error( s.str() ) );

  }

}

//-------------------------------------------------------------------------

// return the list of PNs for a given crystal
void NumbersPn::getPNs( const int ism, const int ix, const int iy, std::vector<int>& PNsInLM ) throw( std::runtime_error ) {

  int ilm = NumbersPn::iLM(ism, ix, iy );

  PNsInLM.clear();

  if( ilm == 0 ) {
    PNsInLM.push_back(25);
    PNsInLM.push_back(26);
    PNsInLM.push_back(27);
    PNsInLM.push_back(28);
    
    PNsInLM.push_back(30);
    PNsInLM.push_back(31);
    PNsInLM.push_back(32);
    PNsInLM.push_back(33);

    return;
  }
  if( ilm == 1 ) {
    PNsInLM.push_back(25);
    PNsInLM.push_back(26);
    PNsInLM.push_back(27);
    PNsInLM.push_back(28);

    PNsInLM.push_back(30);
    PNsInLM.push_back(31);
    PNsInLM.push_back(32);
    PNsInLM.push_back(33);

    return;
  }
  if( ilm == 2 ) {
    PNsInLM.push_back(20);
    PNsInLM.push_back(21);
    // PNsInLM.push_back(22);
    PNsInLM.push_back(23);
    PNsInLM.push_back(24);

    PNsInLM.push_back(35);
    PNsInLM.push_back(36);
    PNsInLM.push_back(37);
    PNsInLM.push_back(38);
    PNsInLM.push_back(39);

    return;
  } 
  if( ilm == 3 ) {
    PNsInLM.push_back(20);
    PNsInLM.push_back(21);
    // PNsInLM.push_back(22);
    PNsInLM.push_back(23);
    PNsInLM.push_back(24);

    PNsInLM.push_back(35);
    PNsInLM.push_back(36);
    PNsInLM.push_back(37);
    PNsInLM.push_back(38);
    PNsInLM.push_back(39);

    return;
  }
  if( ilm == 4 ) {
    PNsInLM.push_back(20);
    PNsInLM.push_back(21);
    // PNsInLM.push_back(22);
    PNsInLM.push_back(23);
    PNsInLM.push_back(24);

    PNsInLM.push_back(35);
    PNsInLM.push_back(36);
    PNsInLM.push_back(37);
    PNsInLM.push_back(38);
    PNsInLM.push_back(39);

    return;
  }
  if( ilm == 5 ) {
    PNsInLM.push_back(0);
    PNsInLM.push_back(1);
    PNsInLM.push_back(2);
    PNsInLM.push_back(4);
    // PNsInLM.push_back(9);

    PNsInLM.push_back(14);
    PNsInLM.push_back(15);
    PNsInLM.push_back(16);
    PNsInLM.push_back(17);
    PNsInLM.push_back(18);

    return;
  }
  if( ilm == 6 ) {
    PNsInLM.push_back(0);
    PNsInLM.push_back(1);
    PNsInLM.push_back(2);
    PNsInLM.push_back(4);
    // PNsInLM.push_back(9);

    PNsInLM.push_back(14);
    PNsInLM.push_back(15);
    PNsInLM.push_back(16);
    PNsInLM.push_back(17);
    PNsInLM.push_back(18);

    return;
  }
  if( ilm == 7 ) {
    PNsInLM.push_back(0);
    PNsInLM.push_back(1);
    PNsInLM.push_back(2);
    PNsInLM.push_back(4);

    PNsInLM.push_back(15);
    PNsInLM.push_back(16);
    PNsInLM.push_back(17);
    PNsInLM.push_back(18);

    return;
  }
  if( ilm == 8 ) {
    PNsInLM.push_back(5);
    PNsInLM.push_back(6);
    PNsInLM.push_back(7);
    PNsInLM.push_back(8);

    PNsInLM.push_back(11);
    PNsInLM.push_back(12);

    return;
  }
  if( ilm == 9 ) {
    PNsInLM.push_back(5);
    PNsInLM.push_back(6);
    PNsInLM.push_back(7);
    PNsInLM.push_back(8);

    PNsInLM.push_back(11);
    PNsInLM.push_back(12);

    return;
  }
  if( ilm == 10 ) {
    PNsInLM.push_back(65);
    PNsInLM.push_back(66);
    PNsInLM.push_back(67);
    PNsInLM.push_back(68);

    PNsInLM.push_back(70);
    PNsInLM.push_back(71);
    PNsInLM.push_back(72);
    PNsInLM.push_back(73);

    return;
  }
  if( ilm == 11 ) {
    PNsInLM.push_back(65);
    PNsInLM.push_back(66);
    PNsInLM.push_back(67);
    PNsInLM.push_back(68);

    PNsInLM.push_back(70);
    PNsInLM.push_back(71);
    PNsInLM.push_back(72);
    PNsInLM.push_back(73);

    return;
  }
  if( ilm == 12 ) {
    PNsInLM.push_back(60);
    PNsInLM.push_back(61);
    PNsInLM.push_back(62);
    PNsInLM.push_back(63);
    PNsInLM.push_back(64);

    PNsInLM.push_back(75);
    PNsInLM.push_back(76);
    PNsInLM.push_back(78);
    PNsInLM.push_back(79);

    return;
  }
  if( ilm == 13 ) {
    PNsInLM.push_back(60);
    PNsInLM.push_back(61);
    PNsInLM.push_back(62);
    PNsInLM.push_back(63);
    PNsInLM.push_back(64);
    PNsInLM.push_back(69);

    PNsInLM.push_back(74);
    PNsInLM.push_back(75);
    PNsInLM.push_back(76);
    PNsInLM.push_back(77);
    PNsInLM.push_back(78);
    PNsInLM.push_back(79);

    return;
  }
  if( ilm == 14 ) {
    PNsInLM.push_back(60);
    PNsInLM.push_back(61);
    PNsInLM.push_back(62);
    PNsInLM.push_back(63);
    PNsInLM.push_back(64);
    PNsInLM.push_back(69);

    PNsInLM.push_back(74);
    PNsInLM.push_back(75);
    PNsInLM.push_back(76);
    PNsInLM.push_back(77);
    PNsInLM.push_back(78);
    PNsInLM.push_back(79);

    return;
  }
  if( ilm == 15 ) {
    PNsInLM.push_back(40);
    PNsInLM.push_back(41);
    PNsInLM.push_back(42);
    PNsInLM.push_back(43);
    PNsInLM.push_back(44);
    PNsInLM.push_back(49);

    PNsInLM.push_back(54);
    PNsInLM.push_back(55);
    PNsInLM.push_back(56);
    PNsInLM.push_back(57);
    PNsInLM.push_back(58);
    PNsInLM.push_back(59);

    return;
  }
  if( ilm == 16 ) {
    PNsInLM.push_back(40);
    PNsInLM.push_back(41);
    PNsInLM.push_back(42);
    PNsInLM.push_back(43);
    PNsInLM.push_back(44);
    PNsInLM.push_back(49);

    PNsInLM.push_back(54);
    PNsInLM.push_back(55);
    PNsInLM.push_back(56);
    PNsInLM.push_back(57);
    PNsInLM.push_back(58);
    PNsInLM.push_back(59);

    return;
  }
  if( ilm == 17 ) {
    PNsInLM.push_back(40);
    PNsInLM.push_back(41);
    PNsInLM.push_back(42);
    PNsInLM.push_back(43);
    PNsInLM.push_back(44);

    PNsInLM.push_back(55);
    PNsInLM.push_back(56);
    PNsInLM.push_back(58);
    PNsInLM.push_back(59);

    return;
  }
  if( ilm == 18 ) {
    PNsInLM.push_back(45);
    PNsInLM.push_back(46);
    PNsInLM.push_back(47);
    PNsInLM.push_back(48);

    PNsInLM.push_back(50);
    PNsInLM.push_back(51);
    PNsInLM.push_back(52);
    PNsInLM.push_back(53);

    return;
  }
  if( ilm == 19 ) {
    PNsInLM.push_back(45);
    PNsInLM.push_back(46);
    PNsInLM.push_back(47);
    PNsInLM.push_back(48);

    PNsInLM.push_back(50);
    PNsInLM.push_back(51);
    PNsInLM.push_back(52);
    PNsInLM.push_back(53);

    return;
  }

  std::ostringstream s;
  s << "Wrong LM id determination: iLM = " << ilm;
  throw( std::runtime_error( s.str() ) );
  
}

//-------------------------------------------------------------------------

// return the LM for a given crystal
int NumbersPn::iLM( const int ism, const int ix, const int iy ) throw( std::runtime_error ) {

  int iz = 0;

  if( ism >=  1 && ism <=  9 ) iz = -1;
  if( ism >= 10 && ism <= 18 ) iz = +1;

  if( EEDetId::validDetId(ix, iy, iz) ) {

    // EE-
    if( ism == 1 ) return 7;
    if( ism == 2 ) return 8;
    if( ism == 3 ) return 9;
    if( ism == 4 ) return 0;
    if( ism == 5 ) return 1;
    if( ism == 6 ) return 2;
    if( ism == 7 ) return 3;
    if( ism == 8 ) {
      if(ix<=50) return 4;
      else return 5;
    }
    if( ism == 9 ) return 6;
  
    // EE+
    if( ism == 10 ) return 17;
    if( ism == 11 ) return 18;
    if( ism == 12 ) return 19;
    if( ism == 13 ) return 10;
    if( ism == 14 ) return 11;
    if( ism == 15 ) return 12;
    if( ism == 16 ) return 13;
    if( ism == 17 ) {
      if(ix<=50) return 14;
      else return 15;
    }
    if( ism == 18 ) return 16;

  }

  std::ostringstream s;
  s << "Wrong LM id determination: iSM = " << ism << " ix = " << ix << " iy = " << iy;
  throw( std::runtime_error( s.str() ) );

}

//-------------------------------------------------------------------------

