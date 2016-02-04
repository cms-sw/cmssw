/*
 *  See header file for a description of this class.
 *
 *  $Date: 2011/02/08 15:49:03 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondFormats/DTObjects/interface/DTCompactMapAbstractHandler.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------


//---------------
// C++ Headers --
//---------------
#include <iostream>

//-------------------
// Initializations --
//-------------------
DTCompactMapAbstractHandler* DTCompactMapAbstractHandler::instance = 0;

//----------------
// Constructors --
//----------------
DTCompactMapAbstractHandler::DTCompactMapAbstractHandler() {
}


DTCompactMapAbstractHandler::~DTCompactMapAbstractHandler() {
}


//--------------
// Operations --
//--------------
/// get static object
DTCompactMapAbstractHandler* DTCompactMapAbstractHandler::getInstance() {
  return instance;
}


DTReadOutMapping* DTCompactMapAbstractHandler::expandMap( const DTReadOutMapping& compMap ) {
  return 0;
}

