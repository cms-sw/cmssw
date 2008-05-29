#ifndef CondTools_L1Trigger_L1TriggerConfigOnlineProd_h
#define CondTools_L1Trigger_L1TriggerConfigOnlineProd_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TriggerConfigOnlineProd
// 
/**\class L1TriggerConfigOnlineProd L1TriggerConfigOnlineProd.h CondTools/L1Trigger/interface/L1TriggerConfigOnlineProd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Sat Mar  1 05:06:43 CET 2008
// $Id: L1TriggerConfigOnlineProd.h,v 1.2 2008/05/28 17:54:06 wsun Exp $
//

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondTools/L1Trigger/interface/OMDSReader.h"

#include "CondTools/L1Trigger/interface/Exception.h"

#include "CondFormats/L1TObjects/interface/L1TriggerKey.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"
#include "CondFormats/L1TObjects/interface/L1TriggerKeyList.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"

#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"
#include "CondFormats/DataRecord/interface/L1RCTParametersRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"

// forward declarations

class L1TriggerConfigOnlineProd : public edm::ESProducer {
   public:
      L1TriggerConfigOnlineProd(const edm::ParameterSet&);
      ~L1TriggerConfigOnlineProd();

      boost::shared_ptr<L1RCTParameters> produceL1RCTParameters(
	 const L1RCTParametersRcd&);
   private:
      // ----------member data ---------------------------
      l1t::OMDSReader m_omdsReader ;
      bool m_forceGeneration ;

      // Called from produce methods.
      // bool is true if the subsystem data should be made.
      // If bool is false, produce method should return null pointer.
      template< class TRcd, class TData >
      bool getSubsystemKey( const TRcd& record,
			    boost::shared_ptr< TData > data,
			    std::string& subsystemKey ) ;
};

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

#endif
