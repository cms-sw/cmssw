// -*- C++ -*-
//
// Package:    L1ObjectKeysOnlineProdBase
// Class:      L1ObjectKeysOnlineProdBase
// 
/**\class L1ObjectKeysOnlineProdBase L1ObjectKeysOnlineProdBase.h CondTools/L1ObjectKeysOnlineProdBase/src/L1ObjectKeysOnlineProdBase.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Werner Man-Li Sun
//         Created:  Fri Aug 22 19:51:36 CEST 2008
// $Id: L1ObjectKeysOnlineProdBase.cc,v 1.2 2010/02/01 22:00:05 wsun Exp $
//
//


// system include files

// user include files
#include "CondTools/L1Trigger/interface/L1ObjectKeysOnlineProdBase.h"

#include "CondTools/L1Trigger/interface/Exception.h"

#include "CondFormats/L1TObjects/interface/L1TriggerKeyList.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"

#include "FWCore/Framework/interface/EventSetup.h"

//
// class declaration
//

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1ObjectKeysOnlineProdBase::L1ObjectKeysOnlineProdBase(const edm::ParameterSet& iConfig)
   : m_omdsReader(
	iConfig.getParameter< std::string >( "onlineDB" ),
	iConfig.getParameter< std::string >( "onlineAuthentication" ) )
{
   //the following line is needed to tell the framework what
   // data is being produced

  // The subsystemLabel is used by L1TriggerKeyOnlineProd to identify the
  // L1TriggerKeys to concatenate.
  setWhatProduced(this,
		  iConfig.getParameter< std::string >( "subsystemLabel" )
		  );

   //now do what ever other initialization is needed
}


L1ObjectKeysOnlineProdBase::~L1ObjectKeysOnlineProdBase()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
L1ObjectKeysOnlineProdBase::ReturnType
L1ObjectKeysOnlineProdBase::produce(const L1TriggerKeyRcd& iRecord)
{
   using namespace edm::es;

  // Get L1TriggerKey with label "SubsystemKeysOnly".  Re-throw exception if
  // not present.
  edm::ESHandle< L1TriggerKey > subsystemKeys ;
  try
    {
      iRecord.get( "SubsystemKeysOnly", subsystemKeys ) ;
    }
  catch( l1t::DataAlreadyPresentException& ex )
    {
      throw ex ;
    }

  // Copy L1TriggerKey to new object.
  boost::shared_ptr<L1TriggerKey> pL1TriggerKey ;
  pL1TriggerKey = boost::shared_ptr< L1TriggerKey >(
    new L1TriggerKey( *subsystemKeys ) ) ;

  // Get object keys.
  fillObjectKeys( pL1TriggerKey ) ;

  return pL1TriggerKey ;
}

//define this as a plug-in
//DEFINE_FWK_EVENTSETUP_MODULE(L1ObjectKeysOnlineProdBase);
