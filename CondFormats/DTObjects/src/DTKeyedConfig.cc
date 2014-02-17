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
#include "CondFormats/DTObjects/interface/DTKeyedConfig.h"

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


//----------------
// Constructors --
//----------------
DTKeyedConfig::DTKeyedConfig() {
}


DTKeyedConfig::DTKeyedConfig( const DTKeyedConfig& obj ) {
  cfgId = obj.cfgId;
  data_iterator d_iter = obj.dataList.begin();
  data_iterator d_iend = obj.dataList.end();
  while ( d_iter != d_iend ) dataList.push_back( *d_iter++ );
  link_iterator l_iter = obj.linkList.begin();
  link_iterator l_iend = obj.linkList.end();
  while ( l_iter != l_iend ) linkList.push_back( *l_iter++ );
}


//--------------
// Destructor --
//--------------
DTKeyedConfig::~DTKeyedConfig() {
}

//--------------
// Operations --
//--------------
int DTKeyedConfig::getId() const {
  return cfgId;
}


void DTKeyedConfig::setId( int id ) {
  cfgId = id;
}


void DTKeyedConfig::add( const std::string& data ) {
  dataList.push_back( data );
}


void DTKeyedConfig::add( int id ) {
  linkList.push_back( id );
}


DTKeyedConfig::data_iterator DTKeyedConfig::dataBegin() const {
  return dataList.begin();
}


DTKeyedConfig::data_iterator DTKeyedConfig::dataEnd() const {
  return dataList.end();
}


DTKeyedConfig::link_iterator DTKeyedConfig::linkBegin() const {
  return linkList.begin();
}


DTKeyedConfig::link_iterator DTKeyedConfig::linkEnd() const {
  return linkList.end();
}






