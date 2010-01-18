/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009-12-10 17:57:08 $
 *  $Revision: 1.1 $
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


