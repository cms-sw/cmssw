/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/05/14 11:42:56 $
 *  $Revision: 1.2 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondFormats/DTObjects/interface/DTConfigAbstractHandler.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
//#include "CondCore/DBCommon/interface/TypedRef.h"
//#include "CondFormats/DTObjects/interface/DTDBAbstractSession.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>

//-------------------
// Initializations --
//-------------------
DTConfigAbstractHandler* DTConfigAbstractHandler::instance = 0;

//----------------
// Constructors --
//----------------
DTConfigAbstractHandler::DTConfigAbstractHandler() {
}


DTConfigAbstractHandler::~DTConfigAbstractHandler() {
}


//--------------
// Operations --
//--------------
/// get static object
DTConfigAbstractHandler* DTConfigAbstractHandler::getInstance() {
  return instance;
}


int DTConfigAbstractHandler::get( const edm::EventSetup& context,
                                  int cfgId, const DTKeyedConfig*& obj ) {
  obj = 0;
  return 999;
}


int DTConfigAbstractHandler::get( const DTKeyedConfigListRcd& keyRecord,
                                  int cfgId, const DTKeyedConfig*& obj ) {
  obj = 0;
  return 999;
}


void DTConfigAbstractHandler::getData( const edm::EventSetup& context,
                                       int cfgId,
                                       std::vector<std::string>& list) {
  return;
}


void DTConfigAbstractHandler::getData( const DTKeyedConfigListRcd& keyRecord,
                                       int cfgId,
                                       std::vector<std::string>& list) {
  return;
}


void DTConfigAbstractHandler::purge() {
  return;
}

