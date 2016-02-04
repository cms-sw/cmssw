/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/09/14 13:54:17 $
 *  $Revision: 1.2 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/interface/DTHVAbstractCheck.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------


//---------------
// C++ Headers --
//---------------


//-------------------
// Initializations --
//-------------------
DTHVAbstractCheck* DTHVAbstractCheck::instance = 0;

//----------------
// Constructors --
//----------------
DTHVAbstractCheck::DTHVAbstractCheck() {
}

//--------------
// Destructor --
//--------------
DTHVAbstractCheck::~DTHVAbstractCheck() {
}

//--------------
// Operations --
//--------------
DTHVAbstractCheck* DTHVAbstractCheck::getInstance() {
  return instance;
}


bool DTHVAbstractCheck::chkFlag( const DTHVAbstractCheck::flag& f ) {
  return ( f.a || f.c || f.s );
}


bool DTHVAbstractCheck::compare( const DTHVAbstractCheck::flag& fl,
                                 const DTHVAbstractCheck::flag& fr ) {
  return ( ( fl.a == fr.a ) &&
           ( fl.c == fr.c ) &&
           ( fl.s == fr.s ) );
}


void DTHVAbstractCheck::setValue(
                        int rawId, int type,
                        float valueA, float valueC, float valueS,
                        const std::map<int,timedMeasurement>& snapshotValues,
                        const std::map<int,int>& aliasMap,
                        const std::map<int,int>& layerMap ) {
  return;
}


void DTHVAbstractCheck::setStatus(
                        int rawId,
                        int flagA, int flagC, int flagS,
                        const std::map<int,timedMeasurement>& snapshotValues,
                        const std::map<int,int>& aliasMap,
                        const std::map<int,int>& layerMap ) {
  return;
}

