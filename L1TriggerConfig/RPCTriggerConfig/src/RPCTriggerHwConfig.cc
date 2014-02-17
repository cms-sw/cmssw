// -*- C++ -*-
//
// Package:    RPCTriggerHwConfig
// Class:      RPCTriggerHwConfig
// 
/**\class RPCTriggerHwConfig RPCTriggerHwConfig.h L1TriggerConfig/RPCTriggerHwConfig/src/RPCTriggerHwConfig.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tomasz Maciej Frueboes
//         Created:  Wed Apr  9 13:57:29 CEST 2008
// $Id: RPCTriggerHwConfig.cc,v 1.5 2010/02/26 15:50:59 fruboes Exp $
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1RPCHwConfigRcd.h"
#include "CondFormats/RPCObjects/interface/L1RPCHwConfig.h"



//
// class decleration
//

class RPCTriggerHwConfig : public edm::ESProducer {
   public:
      RPCTriggerHwConfig(const edm::ParameterSet&);
      ~RPCTriggerHwConfig();

      typedef std::auto_ptr<L1RPCHwConfig> ReturnType;

      ReturnType produce(const L1RPCHwConfigRcd&);
   private:
      // ----------member data ---------------------------
    std::vector<int> m_disableTowers;
    std::vector<int>   m_disableCrates;
    std::vector<int> m_disableTowersInCrates;

    std::vector<int> m_enableTowers;
    std::vector<int> m_enableCrates;
    std::vector<int> m_enableTowersInCrates;

    bool m_disableAll;

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
RPCTriggerHwConfig::RPCTriggerHwConfig(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);

   //now do what ever other initialization is needed
    m_disableTowers =  iConfig.getParameter<std::vector<int> >("disableTowers");
    m_disableCrates =  iConfig.getParameter<std::vector<int> >("disableCrates");
    m_disableTowersInCrates =  iConfig.getParameter<std::vector<int> >("disableTowersInCrates");

    m_disableAll = iConfig.getParameter<bool>("disableAll");

    m_enableTowers =  iConfig.getParameter<std::vector<int> >("enableTowers");
    m_enableCrates =  iConfig.getParameter<std::vector<int> >("enableCrates");
    m_enableTowersInCrates =  iConfig.getParameter<std::vector<int> >("enableTowersInCrates");

    if (m_disableAll) {
      m_disableTowers.clear();
      m_disableCrates.clear();
      m_disableTowersInCrates.clear();
      // check if m_enableTowers  & m_enableCrates are not empty?
    }



}


RPCTriggerHwConfig::~RPCTriggerHwConfig()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
RPCTriggerHwConfig::ReturnType
RPCTriggerHwConfig::produce(const L1RPCHwConfigRcd& iRecord)
{
   using namespace edm::es;
   std::auto_ptr<L1RPCHwConfig> pL1RPCHwConfig = std::auto_ptr<L1RPCHwConfig>( new L1RPCHwConfig() );

   if (m_disableAll) {
     pL1RPCHwConfig->enableAll(false);
     std::vector<int>::iterator crIt = m_enableCrates.begin();
     std::vector<int>::iterator twIt = m_enableTowers.begin();
     for (; crIt!=m_enableCrates.end(); ++crIt){
       pL1RPCHwConfig->enableCrate(*crIt,true);
     }
     for (; twIt!=m_enableTowers.end(); ++twIt){
       pL1RPCHwConfig->enableTower(*twIt,true);
     }
     for (unsigned int It=0; It<m_enableTowersInCrates.size(); It++) {
       if (It%2 == 0)
       pL1RPCHwConfig->enableTowerInCrate(m_enableTowersInCrates[It+1], m_enableTowersInCrates[It], true);
     }
   } else {
     std::vector<int>::iterator crIt = m_disableCrates.begin();
     std::vector<int>::iterator twIt = m_disableTowers.begin();
     for (; crIt!=m_disableCrates.end(); ++crIt){
       pL1RPCHwConfig->enableCrate(*crIt,false);
     }
     for (; twIt!=m_disableTowers.end(); ++twIt){
       pL1RPCHwConfig->enableTower(*twIt,false);
     }
     for (unsigned int It=0; It<m_disableTowersInCrates.size(); It++) {
       if (It%2 == 0)
       pL1RPCHwConfig->enableTowerInCrate(m_disableTowersInCrates[It+1], m_disableTowersInCrates[It], false);
     }

   }

   return pL1RPCHwConfig ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(RPCTriggerHwConfig);
