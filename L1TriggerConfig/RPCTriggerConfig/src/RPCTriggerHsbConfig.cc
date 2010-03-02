// -*- C++ -*-
//
// Package:    RPCTriggerHsbConfig
// Class:      RPCTriggerHsbConfig
// 
/**\class RPCTriggerHsbConfig RPCTriggerHsbConfig.h L1TriggerConfig/RPCTriggerHsbConfig/src/RPCTriggerHsbConfig.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tomasz Maciej Frueboes
//         Created:  Wed Apr  9 13:57:29 CEST 2008
// $Id: RPCTriggerHsbConfig.cc,v 1.1 2010/02/26 15:50:58 fruboes Exp $
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1RPCHsbConfigRcd.h"
#include "CondFormats/L1TObjects/interface/L1RPCHsbConfig.h"



//
// class decleration
//

class RPCTriggerHsbConfig : public edm::ESProducer {
   public:
      RPCTriggerHsbConfig(const edm::ParameterSet&);
      ~RPCTriggerHsbConfig();

      typedef std::auto_ptr<L1RPCHsbConfig> ReturnType;

      ReturnType produce(const L1RPCHsbConfigRcd&);
   private:
      int m_hsb0[8];
      int m_hsb1[8];

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
RPCTriggerHsbConfig::RPCTriggerHsbConfig(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);

   std::vector<int> hsbconf0 = iConfig.getParameter< std::vector<int > >("hsb0Mask");
   std::vector<int> hsbconf1 = iConfig.getParameter< std::vector<int > >("hsb1Mask");


   if (hsbconf0.size() !=8 || hsbconf1.size() != 8 )
        throw cms::Exception("BadConfig") << " hsbMask needs to be 8 digits long \n";

   for (int i = 0; i < 8; ++i ) { m_hsb0[i] = hsbconf0.at(i); }
   for (int i = 0; i < 8; ++i ) { m_hsb1[i] = hsbconf1.at(i); }
  
   // contents of the vector wont be checked here - there are also different sources of this cfg

}


RPCTriggerHsbConfig::~RPCTriggerHsbConfig()
{
 

}


//
// member functions
//

// ------------ method called to produce the data  ------------
RPCTriggerHsbConfig::ReturnType
RPCTriggerHsbConfig::produce(const L1RPCHsbConfigRcd& iRecord)
{

   using namespace edm::es;
   std::auto_ptr<L1RPCHsbConfig> pRPCTriggerHsbConfig = std::auto_ptr<L1RPCHsbConfig>( new L1RPCHsbConfig() );

   pRPCTriggerHsbConfig->setHsbMask(0, m_hsb0);
   pRPCTriggerHsbConfig->setHsbMask(1, m_hsb1);

   return pRPCTriggerHsbConfig ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(RPCTriggerHsbConfig);
