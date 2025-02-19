/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/10/25 10:54:52 $
 *  $Revision: 1.3 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/plugins/DTKeyedConfigDBDump.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondFormats/DTObjects/interface/DTKeyedConfig.h"
#include "CondFormats/DataRecord/interface/DTKeyedConfigListRcd.h"
#include "CondCore/IOVService/interface/KeyList.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

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
DTKeyedConfigDBDump::DTKeyedConfigDBDump( const edm::ParameterSet& ps ) {
}

//--------------
// Destructor --
//--------------
DTKeyedConfigDBDump::~DTKeyedConfigDBDump() {
}

//--------------
// Operations --
//--------------
void DTKeyedConfigDBDump::beginJob() {

  return;

}


void DTKeyedConfigDBDump::analyze( const edm::Event& e,
                                   const edm::EventSetup& c ) {
  edm::eventsetup::EventSetupRecordKey
    recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("DTKeyedConfigListRcd"));
  if( recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
    //record not found
    std::cout <<"Record \"DTKeyedConfigListRcd "<<"\" does not exist "<<std::endl;
  }
  edm::ESHandle<cond::KeyList> klh;
  std::cout<<"got eshandle"<<std::endl;
  c.get<DTKeyedConfigListRcd>().get(klh);
  std::cout<<"got context"<<std::endl;
  cond::KeyList const &  kl= *klh.product();
  cond::KeyList* kp = const_cast<cond::KeyList*>( &kl );
  std::vector<unsigned long long> nkeys;
  nkeys.push_back( 999999999 );
  std::cout << "now load" << std::endl;
  kp->load( nkeys );
  std::cout << "now get" << std::endl;
  const DTKeyedConfig* pkc = kp->get<DTKeyedConfig>(0);
  std::cout << "now check" << std::endl;
  if ( pkc != 0 ) std::cout << pkc->getId() << " "
                            << *( pkc->dataBegin() ) << std::endl;
  else            std::cout << "not found" << std::endl;
  std::cout << std::endl;
  std::vector<unsigned long long> nvoid;
  kp->load( nvoid );
  return;
}

DEFINE_FWK_MODULE(DTKeyedConfigDBDump);

