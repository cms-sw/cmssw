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
// $Id: L1TriggerConfigOnlineProd.h,v 1.1 2008/03/03 21:52:18 wsun Exp $
//

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondTools/L1Trigger/interface/OMDSReader.h"

// forward declarations
#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"
#include "CondFormats/DataRecord/interface/L1RCTParametersRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"

class L1TriggerConfigOnlineProd : public edm::ESProducer {
   public:
      L1TriggerConfigOnlineProd(const edm::ParameterSet&);
      ~L1TriggerConfigOnlineProd();

      boost::shared_ptr<L1RCTParameters> produceL1RCTParameters(
	 const L1RCTParametersRcd&);
      boost::shared_ptr<L1CaloEtScale> produceL1JetEtScale(
	 const L1JetEtScaleRcd&);
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

#endif
