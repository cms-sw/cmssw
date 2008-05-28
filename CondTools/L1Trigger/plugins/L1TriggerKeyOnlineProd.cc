// -*- C++ -*-
//
// Package:    L1TriggerKeyOnlineProd
// Class:      L1TriggerKeyOnlineProd
// 
/**\class L1TriggerKeyOnlineProd L1TriggerKeyOnlineProd.h CondTools/L1TriggerKeyOnlineProd/src/L1TriggerKeyOnlineProd.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Werner Man-Li Sun
//         Created:  Sun Mar  2 03:03:32 CET 2008
// $Id: L1TriggerKeyOnlineProd.cc,v 1.2 2008/04/16 23:49:30 wsun Exp $
//
//


// system include files

// user include files
#include "CondTools/L1Trigger/plugins/L1TriggerKeyOnlineProd.h"

#include "CondTools/L1Trigger/interface/Exception.h"

#include "CondFormats/L1TObjects/interface/L1TriggerKeyList.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"

#include "FWCore/Framework/interface/HCTypeTagTemplate.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RelationalAccess/ICursor.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Attribute.h"

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
L1TriggerKeyOnlineProd::L1TriggerKeyOnlineProd(const edm::ParameterSet& iConfig)
   : m_tscKey( iConfig.getParameter< std::string >( "tscKey" ) ),
     m_omdsReader(
	iConfig.getParameter< std::string >( "onlineDB" ),
	iConfig.getParameter< std::string >( "onlineAuthentication" ) ),
     m_forceGeneration( iConfig.getParameter< bool >( "forceGeneration" ) )
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);

   //now do what ever other initialization is needed
}


L1TriggerKeyOnlineProd::~L1TriggerKeyOnlineProd()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
L1TriggerKeyOnlineProd::ReturnType
L1TriggerKeyOnlineProd::produce(const L1TriggerKeyRcd& iRecord)
{
   using namespace edm::es;
   boost::shared_ptr<L1TriggerKey> pL1TriggerKey ;

   // Get L1TriggerKeyList
   const L1TriggerKeyListRcd& keyListRcd =
      iRecord.getRecord< L1TriggerKeyListRcd >() ;
   edm::ESHandle< L1TriggerKeyList > keyList ;
   keyListRcd.get( keyList ) ;

   // If L1TriggerKeyList does not contain TSC key, token is empty
   if( keyList->token( m_tscKey ) == std::string() ||
       m_forceGeneration )
   {
      // Instantiate new L1TriggerKey
      pL1TriggerKey = boost::shared_ptr< L1TriggerKey >(
         new L1TriggerKey() ) ;
      pL1TriggerKey->setTSCKey( m_tscKey ) ;

      // Get subsystem keys from OMDS
      std::string tableString = "GCT_CONFIG" ;

      std::vector< std::string > queryStrings ;
      queryStrings.push_back( "CONFIG_KEY" ) ;

      std::string conditionString = "" ;
      coral::AttributeList attributes ;

      boost::shared_ptr< coral::IQuery > query
	( m_omdsReader.newQuery( tableString, queryStrings,
				 conditionString, attributes ) ) ;
      coral::ICursor& cursor = query->execute() ;
      while( cursor.next() )
	{
	  const coral::AttributeList& row = cursor.currentRow() ;
	  std::string key = row[ "CONFIG_KEY" ].data< std::string >() ;
	  std::cout << "CONFIG_KEY " << key << std::endl ;
	}
   }
   else
   {
     throw l1t::DataAlreadyPresentException(
        "L1TriggerKey for TSC key " + m_tscKey + " already in CondDB." ) ;
   }

   return pL1TriggerKey ;
}

//define this as a plug-in
//DEFINE_FWK_EVENTSETUP_MODULE(L1TriggerKeyOnlineProd);
