/*
 *  See header file for a description of this class.
 *
 *  $Date: 2012/02/15 16:04:15 $
 *  $Revision: 1.4 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/plugins/DTKeyedConfigDBInit.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondFormats/DTObjects/interface/DTKeyedConfig.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/DBOutputService/interface/KeyedElement.h"

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
DTKeyedConfigDBInit::DTKeyedConfigDBInit( const edm::ParameterSet& ps ):
 container(  ps.getParameter<std::string> ( "container" ) ),
 iov(        ps.getParameter<std::string> ( "iov"       ) ) {
}

//--------------
// Destructor --
//--------------
DTKeyedConfigDBInit::~DTKeyedConfigDBInit() {
}

//--------------
// Operations --
//--------------
void DTKeyedConfigDBInit::beginJob() {
  return;
}


void DTKeyedConfigDBInit::analyze( const edm::Event& e,
                                   const edm::EventSetup& c ) {
  return;
}


void DTKeyedConfigDBInit::endJob() {

  edm::Service<cond::service::PoolDBOutputService> outdb;
  DTKeyedConfig* bk = new DTKeyedConfig();
  bk->setId( 999999999 );
  bk->add( "dummy" );
  cond::KeyedElement k( bk, 999999999 );
  outdb->writeOne( k.m_obj, k.m_key, container );

  std::vector<cond::Time_t> * kl = new std::vector<cond::Time_t>;
  kl->push_back( 999999999 );
  outdb->writeOne(kl,1,iov);

  return;

}


DEFINE_FWK_MODULE(DTKeyedConfigDBInit);

