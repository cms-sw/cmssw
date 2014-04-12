/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/05/14 11:43:08 $
 *  $Revision: 1.2 $
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
#include "CondCore/CondDB/interface/KeyList.h"
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
  edm::ESHandle<cond::persistency::KeyList> klh;
  std::cout<<"got eshandle"<<std::endl;
  c.get<DTKeyedConfigListRcd>().get(klh);
  std::cout<<"got context"<<std::endl;
  cond::persistency::KeyList const &  kl= *klh.product();
  cond::persistency::KeyList* kp = const_cast<cond::persistency::KeyList*>( &kl );
  std::vector<unsigned long long> nkeys;
  nkeys.push_back( 999999999 );
  std::cout << "now load" << std::endl;
  kp->load( nkeys );
  std::cout << "now get" << std::endl;
  boost::shared_ptr<DTKeyedConfig> pkc = kp->get<DTKeyedConfig>(0);
  std::cout << "now check" << std::endl;
  if ( pkc.get() ) std::cout << pkc->getId() << " "
                            << *( pkc->dataBegin() ) << std::endl;
  else            std::cout << "not found" << std::endl;
  std::cout << std::endl;
  std::vector<unsigned long long> nvoid;
  kp->load( nvoid );
  return;
}

DEFINE_FWK_MODULE(DTKeyedConfigDBDump);

