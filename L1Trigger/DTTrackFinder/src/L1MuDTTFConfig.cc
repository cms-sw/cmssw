//-------------------------------------------------
//
//   Class: L1MuDTTFConfig
//
//   Description: DTTrackFinder parameters for L1MuDTTrackFinder
//
//
//   $Date: 2006/06/01 00:00:00 $
//   $Revision: 1.1 $
//
//   Author :
//   N. Neumeister            CERN EP
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------
using namespace std;

//-----------------------
// This Class's Header --
//-----------------------

#include "L1Trigger/DTTrackFinder/src/L1MuDTTFConfig.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <string>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------


// --------------------------------
//       class L1MuDTTFConfig
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuDTTFConfig::L1MuDTTFConfig() { 
    
    setDefaults(); 

}
  

//--------------
// Destructor --
//--------------
L1MuDTTFConfig::~L1MuDTTFConfig() {}


//--------------
// Operations --
//--------------

void L1MuDTTFConfig::setDefaults() {
  
  m_debug = true;
  m_dbgLevel = -1;
  if ( m_dbgLevel == -1 ) {
    m_dbgLevel = 0;
  }

  m_overlap = true;

  // set min and max bunch crossing
  //  m_BxMin = -9;
  //  m_BxMax =  7;
  m_BxMin =  0;
  m_BxMax = 40;

  // set Filter for Extrapolator
  m_extTSFilter = 1;

  // set switch for EX21 usage
  m_useEX21 = false;

  // set switch for eta track finder usage
  m_etaTF = true;

  // set Filter for Out-of-time Track Segments
  m_TSOutOfTimeFilter = false;
  m_TSOutOfTimeWindow = 1;

  // set precision for extrapolation
  m_NbitsExtPhi  = 8;
  m_NbitsExtPhib = 8;

  // set precision for pt-assignment
  m_NbitsPtaPhi  = 12;
  m_NbitsPtaPhib = 10;

  // set precision for phi-assignment look-up tables
  m_NbitsPhiPhi  = 12;
  m_NbitsPhiPhib = 10;

  if ( Debug(1) ) cout << endl;
  if ( Debug(1) ) cout << "*******************************************" << endl;  
  if ( Debug(1) ) cout << "**** L1 barrel Track Finder settings : ****" << endl;
  if ( Debug(1) ) cout << "*******************************************" << endl;
  if ( Debug(1) ) cout << endl;
  
  if ( Debug(1) ) cout << "L1 barrel Track Finder : debug level: " << m_dbgLevel << endl;

  if ( Debug(1) && m_overlap ) {
    cout << "L1 barrel Track Finder : barrel-endcap overlap region : on" << endl;
  }
  if ( Debug(1) && !m_overlap ) { 
    cout << "L1 barrel Track Finder : barrel-endcap overlap region : off" << endl;
  }

  if ( Debug(1) ) cout << "L1 barrel Track Finder : minimal bunch-crossing : " << m_BxMin << endl;
  if ( Debug(1) ) cout << "L1 barrel Track Finder : maximal bunch-crossing : " << m_BxMax << endl;

  if ( Debug(1) ) cout << "L1 barrel Track Finder : Extrapolation Filter : " << m_extTSFilter << endl;

  if ( Debug(1) && m_useEX21 ) {
    cout << "L1 barrel Track Finder : use EX21 extrapolations : on" << endl;
  }
  if ( Debug(1) && !m_useEX21 ) {
    cout << "L1 barrel Track Finder : use EX21 extrapolations : off" << endl;
  }

  if ( Debug(1) && m_etaTF ) {
    cout << "L1 barrel Track Finder : Eta Track Finder : on" << endl;
  }
  if ( Debug(1) && !m_etaTF ) {
    cout << "L1 barrel Track Finder : Eta Track Finder : off" << endl;
  }

  if ( Debug(1) && m_TSOutOfTimeFilter ) {
    cout << "L1 barrel Track Finder : out-of-time TS filter : on" << endl;
    cout << "L1 barrel Track Finder : out-of-time TS filter window : " << m_TSOutOfTimeWindow << endl;
  }
  if ( Debug(1) && !m_TSOutOfTimeFilter ) {
    cout << "L1 barrel Track Finder : out-of-time TS filter : off" << endl;
  }

  if ( Debug(1) ) cout << "L1 barrel Track Finder : # of bits used for phi  (extrapolation)  : " << m_NbitsExtPhi << endl;
  if ( Debug(1) ) cout << "L1 barrel Track Finder : # of bits used for phib (extrapolation)  : " << m_NbitsExtPhib << endl;
  if ( Debug(1) ) cout << "L1 barrel Track Finder : # of bits used for phi  (pt-assignment)  : " << m_NbitsPtaPhi << endl;
  if ( Debug(1) ) cout << "L1 barrel Track Finder : # of bits used for phib (pt-assignment)  : " << m_NbitsPtaPhib << endl;
  if ( Debug(1) ) cout << "L1 barrel Track Finder : # of bits used for phi  (phi-assignment) : " << m_NbitsPhiPhi << endl;
  if ( Debug(1) ) cout << "L1 barrel Track Finder : # of bits used for phib (phi-assignment) : " << m_NbitsPhiPhib << endl;

}


// static data members

bool L1MuDTTFConfig::m_debug = false;
int L1MuDTTFConfig::m_dbgLevel = -1;
bool L1MuDTTFConfig::m_overlap = true;
int L1MuDTTFConfig::m_BxMin = 0;
int L1MuDTTFConfig::m_BxMax = 40;
int L1MuDTTFConfig::m_extTSFilter  = 1;
bool L1MuDTTFConfig::m_useEX21 = false;
bool L1MuDTTFConfig::m_etaTF = true;
bool L1MuDTTFConfig::m_TSOutOfTimeFilter = false;
int L1MuDTTFConfig::m_TSOutOfTimeWindow = 1;
int L1MuDTTFConfig::m_NbitsExtPhi  = 8; 
int L1MuDTTFConfig::m_NbitsExtPhib = 8;
int L1MuDTTFConfig::m_NbitsPtaPhi  = 12; 
int L1MuDTTFConfig::m_NbitsPtaPhib = 10;
int L1MuDTTFConfig::m_NbitsPhiPhi  = 12; 
int L1MuDTTFConfig::m_NbitsPhiPhib = 10;
