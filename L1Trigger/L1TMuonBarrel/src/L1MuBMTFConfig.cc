//-------------------------------------------------
//
//   Class: L1MuBMTFConfig
//
//   Description: L1TMuonBarrelTrackProducer parameters for L1MuBMTrackFinder
//
//
//
//   Author :
//   N. Neumeister            CERN EP
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "L1Trigger/L1TMuonBarrel/src/L1MuBMTFConfig.h"

#include "CondFormats/L1TObjects/interface/L1TMuonBarrelParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonBarrelParamsRcd.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <string>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

using namespace std;

// --------------------------------
//       class L1MuBMTFConfig
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuBMTFConfig::L1MuBMTFConfig(const edm::ParameterSet & ps ) {

    m_ps = &ps;
    //m_es = &c;
    setDefaults();

}


//--------------
// Destructor --
//--------------
L1MuBMTFConfig::~L1MuBMTFConfig() {}


//--------------
// Operations --
//--------------

void L1MuBMTFConfig::setDefaults() {

  m_BMDigiInputTag = m_ps->getParameter<edm::InputTag>("DTDigi_Source");
  m_BMThetaDigiInputTag = m_ps->getParameter<edm::InputTag>("DTDigi_Theta_Source");

  m_debug = true;
  m_dbgLevel = m_ps->getUntrackedParameter<int>("Debug",0);

  if ( Debug(1) ) cout << endl;
  if ( Debug(1) ) cout << "*******************************************" << endl;
  if ( Debug(1) ) cout << "**** L1 barrel Track Finder settings : ****" << endl;
  if ( Debug(1) ) cout << "*******************************************" << endl;
  if ( Debug(1) ) cout << endl;

  if ( Debug(1) ) cout << "L1 barrel Track Finder : BM Digi Source:  " <<  m_BMDigiInputTag << endl;
  if ( Debug(1) ) cout << "L1 barrel Track Finder : BM Digi Source:  " <<  m_BMThetaDigiInputTag << endl;
  if ( Debug(1) ) cout << endl;

  if ( Debug(1) ) cout << "L1 barrel Track Finder : debug level: " << m_dbgLevel << endl;

}


void L1MuBMTFConfig::setDefaultsES(const edm::EventSetup& c) {
    m_es = &c;


  const L1TMuonBarrelParamsRcd& bmtfParamsRcd = m_es->get<L1TMuonBarrelParamsRcd>();
  bmtfParamsRcd.get(bmtfParamsHandle);
  L1TMuonBarrelParams *bmtfParams = new L1TMuonBarrelParams();
  bmtfParams = new L1TMuonBarrelParams(*bmtfParamsHandle.product());


  // set min and max bunch crossing
  m_BxMin = bmtfParams->get_BX_min();
  m_BxMax = bmtfParams->get_BX_max();

  // set Filter for Extrapolator
  m_extTSFilter = bmtfParams->get_Extrapolation_Filter();

  // set switch for open LUTs usage
  m_openLUTs = bmtfParams->get_Open_LUTs();

  // set switch for EX21 usage
  m_useEX21 = bmtfParams->get_Extrapolation_21();

  // set switch for eta track finder usage
  m_etaTF = bmtfParams->get_EtaTrackFinder();

  // set Filter for Out-of-time Track Segments
  m_TSOutOfTimeFilter = bmtfParams->get_OutOfTime_Filter();
  m_TSOutOfTimeWindow = bmtfParams->get_OutOfTime_Filter_Window();

  // set precision for extrapolation
  m_NbitsExtPhi = bmtfParams->get_Extrapolation_nbits_Phi();
  m_NbitsExtPhib = bmtfParams->get_Extrapolation_nbits_PhiB();

  // set precision for pt-assignment
  m_NbitsPtaPhi = bmtfParams->get_PT_Assignment_nbits_Phi();
  m_NbitsPtaPhib = bmtfParams->get_PT_Assignment_nbits_PhiB();

  // set precision for phi-assignment look-up tables
  m_NbitsPhiPhi = bmtfParams->get_PHI_Assignment_nbits_Phi();
  m_NbitsExtPhib = bmtfParams->get_PHI_Assignment_nbits_PhiB();

  if ( Debug(1) ) cout << "L1 barrel Track Finder : minimal bunch-crossing : " << m_BxMin << endl;
  if ( Debug(1) ) cout << "L1 barrel Track Finder : maximal bunch-crossing : " << m_BxMax << endl;

  if ( Debug(1) ) cout << "L1 barrel Track Finder : Extrapolation Filter : " << m_extTSFilter << endl;

  if ( Debug(1) && m_openLUTs) {
    cout << "L1 barrel Track Finder : use open LUTs : on" << endl;
  }
  if ( Debug(1) && !m_openLUTs) {
    cout << "L1 barrel Track Finder : use open LUTs : off" << endl;
  }

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

  delete bmtfParams;

}


// static data members

edm::InputTag L1MuBMTFConfig::m_BMDigiInputTag = edm::InputTag();
edm::InputTag L1MuBMTFConfig::m_BMThetaDigiInputTag = edm::InputTag();

bool L1MuBMTFConfig::m_debug = false;
int  L1MuBMTFConfig::m_dbgLevel = -1;
int  L1MuBMTFConfig::m_BxMin = -9;
int  L1MuBMTFConfig::m_BxMax =  7;
int  L1MuBMTFConfig::m_extTSFilter  = 1;
bool L1MuBMTFConfig::m_openLUTs  = false;
bool L1MuBMTFConfig::m_useEX21 = false;
bool L1MuBMTFConfig::m_etaTF = true;
bool L1MuBMTFConfig::m_TSOutOfTimeFilter = false;
int  L1MuBMTFConfig::m_TSOutOfTimeWindow = 1;
int  L1MuBMTFConfig::m_NbitsExtPhi  = 8;
int  L1MuBMTFConfig::m_NbitsExtPhib = 8;
int  L1MuBMTFConfig::m_NbitsPtaPhi  = 12;
int  L1MuBMTFConfig::m_NbitsPtaPhib = 10;
int  L1MuBMTFConfig::m_NbitsPhiPhi  = 10;
int  L1MuBMTFConfig::m_NbitsPhiPhib = 10;
