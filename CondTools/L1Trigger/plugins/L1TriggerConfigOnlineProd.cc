// -*- C++ -*-
//
// Package:    L1TriggerConfigOnlineProd
// Class:      L1TriggerConfigOnlineProd
// 
/**\class L1TriggerConfigOnlineProd L1TriggerConfigOnlineProd.h CondTools/L1TriggerConfigOnlineProd/src/L1TriggerConfigOnlineProd.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Werner Man-Li Sun
//         Created:  Sat Mar  1 05:02:13 CET 2008
// $Id: L1TriggerConfigOnlineProd.cc,v 1.1 2008/03/03 21:52:18 wsun Exp $
//
//


// system include files

// user include files
#include "CondTools/L1Trigger/plugins/L1TriggerConfigOnlineProd.h"

#include "CondTools/L1Trigger/interface/Exception.h"

#include "CondFormats/L1TObjects/interface/L1TriggerKey.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"
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
L1TriggerConfigOnlineProd::L1TriggerConfigOnlineProd(const edm::ParameterSet& iConfig)
  : m_omdsReader( iConfig.getParameter< std::string >( "onlineDB" ),
		  iConfig.getParameter< std::string >( "onlineAuthentication" )
		  ),
    m_forceGeneration( iConfig.getParameter< bool >( "forceGeneration") )
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced( this, &L1TriggerConfigOnlineProd::produceL1JetEtScaleRcd ) ;

   //now do what ever other initialization is needed
}


L1TriggerConfigOnlineProd::~L1TriggerConfigOnlineProd()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// Called from produce methods.
// bool is true if the subsystem data should be made.
// If bool is false, produce method should throw DataAlreadyPresentException.
template< class TRcd, class TData >
bool L1TriggerConfigOnlineProd::getSubsystemKey( const TRcd& record,
					boost::shared_ptr< TData > data,
					std::string& subsystemKey )
{
   // Get L1TriggerKey
   const L1TriggerKeyRcd& keyRcd =
      record.template getRecord< L1TriggerKeyRcd >() ;

   // Explanation of funny syntax: since record is dependent, we are not
   // expecting getRecord to be a template so the compiler parses it
   // as a non-template. http://gcc.gnu.org/ml/gcc-bugs/2005-11/msg03685.html

   // If L1TriggerKey is invalid, then all configuration objects are
   // already in ORCON.
   edm::ESHandle< L1TriggerKey > key ;
   try
   {
      keyRcd.get( key ) ;
   }
   catch( l1t::DataAlreadyPresentException& ex )
   {
      subsystemKey = std::string() ;
      return false ;      
   }

//    if( !key.isValid() )
//    {
//       subsystemKey = std::string() ;
//       return false ;
//    }

   // Get subsystem key from L1TriggerKey
   std::string recordName =
      edm::eventsetup::heterocontainer::HCTypeTagTemplate< TRcd,
      edm::eventsetup::EventSetupRecordKey >::className() ;
   std::string dataType =
      edm::eventsetup::heterocontainer::HCTypeTagTemplate< TData,
      edm::eventsetup::DataKey >::className() ;

   subsystemKey = key->get( recordName, dataType ) ;

   std::cout << "L1TriggerConfigOnlineProd record " << recordName
	     << " type " << dataType
	     << " sub key " << subsystemKey
	     << std::endl ;

   // Get L1TriggerKeyList
   const L1TriggerKeyListRcd& keyListRcd =
      record.template getRecord< L1TriggerKeyListRcd >() ;
   edm::ESHandle< L1TriggerKeyList > keyList ;
   keyListRcd.get( keyList ) ;

   // If L1TriggerKeyList does not contain subsystem key, token is empty
   return
      keyList->token( recordName, dataType, subsystemKey ) == std::string() ;
}

// ------------ method called to produce the data  ------------
boost::shared_ptr<L1CaloEtScale>
L1TriggerConfigOnlineProd::produceL1JetEtScaleRcd( const L1JetEtScaleRcd& iRecord )
{
   using namespace edm::es;
   boost::shared_ptr<L1CaloEtScale> pL1CaloEtScale ;

   // Get subsystem key and check if already in ORCON
   std::string key ;
   if( getSubsystemKey( iRecord, pL1CaloEtScale, key ) ||
       m_forceGeneration )
   {
      // Key not in ORCON -- get data from OMDS and make C++ object
     std::string tableString = "GCT" ;

     std::vector< std::string > queryStrings ;
     queryStrings.push_back( "RUN" ) ;

     boost::shared_ptr< coral::IQuery > query
       ( m_omdsReader.newQuery( tableString, queryStrings ) ) ;
     coral::ICursor& cursor = query->execute() ;
     while( cursor.next() )
       {
	 const coral::AttributeList& row = cursor.currentRow() ;
	 int run = row[ "RUN" ].data< int >() ;
       }

      pL1CaloEtScale = boost::shared_ptr< L1CaloEtScale >(
	 new L1CaloEtScale() ) ;
   }
   else
   {
     throw l1t::DataAlreadyPresentException(
        "L1JetEtScale for key " + key + " already in CondDB." ) ;
   }

   return pL1CaloEtScale ;
}

//define this as a plug-in
//DEFINE_FWK_EVENTSETUP_MODULE(L1TriggerConfigOnlineProd);
