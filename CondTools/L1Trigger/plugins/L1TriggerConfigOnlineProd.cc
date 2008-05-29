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
// $Id: L1TriggerConfigOnlineProd.cc,v 1.3 2008/05/28 17:54:06 wsun Exp $
//
//


// system include files

// user include files
#include "CondTools/L1Trigger/plugins/L1TriggerConfigOnlineProd.h"

// #include "FWCore/Framework/interface/HCTypeTagTemplate.h"
// #include "FWCore/Framework/interface/EventSetup.h"

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
  setWhatProduced( this, &L1TriggerConfigOnlineProd::produceL1RCTParameters ) ;

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

// ------------ method called to produce the data  ------------

// // Skeleton for produce functions
// boost::shared_ptr< TData >
// L1TriggerConfigOnlineProd::produce( const TRcd& iRecord )
// {
//    using namespace edm::es;
//    boost::shared_ptr< TData > pData ;

//    // Get subsystem key and check if already in ORCON
//    std::string key ;
//    if( getSubsystemKey( iRecord, pData, key ) ||
//        m_forceGeneration )
//    {
//      // Key not in ORCON -- get data from OMDS and make C++ object
//    }
//    else
//    {
//      throw l1t::DataAlreadyPresentException(
//         "TData for key " + key + " already in CondDB." ) ;
//    }

//    return pData ;
// }

//define this as a plug-in
//DEFINE_FWK_EVENTSETUP_MODULE(L1TriggerConfigOnlineProd);
