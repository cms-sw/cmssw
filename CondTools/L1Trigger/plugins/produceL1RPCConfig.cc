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
//         Created:  Mon Jul  7 23:51:18 CEST 2008
// $Id$
//
//


// system include files

// user include files
#include "CondTools/L1Trigger/plugins/L1TriggerConfigOnlineProd.h"

#include "CondTools/L1Trigger/interface/Exception.h"

#include "FWCore/Framework/interface/HCTypeTagTemplate.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RelationalAccess/ICursor.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Attribute.h"

// ------------ method called to produce the data  ------------

boost::shared_ptr<L1RPCConfig>
L1TriggerConfigOnlineProd::produceL1RPCConfig( const L1RPCConfigRcd& iRecord )
{
   using namespace edm::es;
   boost::shared_ptr<L1RPCConfig> pL1RPCConfig ;

   // Always throw exception because all RPCConfig objects are already
   // supposed to be in ORCON.

   // Get subsystem key and check if already in ORCON
   std::string key ;
   if( getSubsystemKey( iRecord, pL1RPCConfig, key ) ||
       m_forceGeneration )
   {
     std::cout << "L1RPCConfig with key " << key << " missing from ORCON"
	       << std::endl ;
   }

   throw l1t::DataAlreadyPresentException(
      "L1RPCConfig for key " + key + " already in CondDB." ) ;

   return pL1RPCConfig ;
}
