// -*- C++ -*-
//
// Package:    DumpConeDefinition
// Class:      DumpConeDefinition
// 
/**\class DumpConeDefinition DumpConeDefinition.cc L1TriggerConfig/DumpConeDefinition/src/DumpConeDefinition.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tomasz Maciej Frueboes
//         Created:  Wed Apr  9 14:03:40 CEST 2008
// $Id: DumpConeDefinition.cc,v 1.4 2010/07/06 16:14:01 ghete Exp $
//
//


// system include files
#include <memory>
#include "CondFormats/L1TObjects/interface/L1RPCConeDefinition.h"
#include "CondFormats/DataRecord/interface/L1RPCConeDefinitionRcd.h"
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"



#include "CondFormats/DataRecord/interface/L1RPCHwConfigRcd.h"
#include "CondFormats/RPCObjects/interface/L1RPCHwConfig.h"

#include <fstream>

//
// class decleration
//

class DumpConeDefinition : public edm::EDAnalyzer {
   public:
      explicit DumpConeDefinition(const edm::ParameterSet&);
      ~DumpConeDefinition();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      
      // ----------member data ---------------------------
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
DumpConeDefinition::DumpConeDefinition(const edm::ParameterSet& iConfig)


{
   //now do what ever initialization is needed


}


DumpConeDefinition::~DumpConeDefinition()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void DumpConeDefinition::analyze(const edm::Event& iEvent,
        const edm::EventSetup& iSetup) {
    using namespace edm;

    edm::ESHandle<L1RPCConeDefinition> l1RPCConeDefinition;
    iSetup.get<L1RPCConeDefinitionRcd> ().get(l1RPCConeDefinition);

    L1RPCConeDefinition::TLPSizeVec::const_iterator it =
            l1RPCConeDefinition->getLPSizeVec().begin();
    L1RPCConeDefinition::TLPSizeVec::const_iterator itEnd =
            l1RPCConeDefinition->getLPSizeVec().end();

    LogTrace("DumpConeDefinition") << std::endl;

    LogDebug("DumpConeDefinition")
            << "\n Printing L1RPCConeDefinitionRcd record\n" << std::endl;
    LogTrace("DumpConeDefinition") << "\nlog plane sizes dump: \n" << std::endl;

    for (; it != itEnd; ++it) {
        LogTrace("DumpConeDefinition")
                << "Tw " << (int) it->m_tower
                << " lp " << (int) it->m_LP
                << " size " << (int) it->m_size
                << std::endl;

    }


    LogTrace("DumpConeDefinition") << "\nRing to tower connections dump: \n"
            << std::endl;
    L1RPCConeDefinition::TRingToTowerVec::const_iterator itR =
            l1RPCConeDefinition->getRingToTowerVec().begin();

    const L1RPCConeDefinition::TRingToTowerVec::const_iterator itREnd =
            l1RPCConeDefinition->getRingToTowerVec().end();

    for (; itR != itREnd; ++itR) {
        LogTrace("DumpConeDefinition")
                << "EP " << (int) itR->m_etaPart
                << " hwPL " << (int) itR->m_hwPlane
                << " tw " << (int) itR->m_tower
                << " index " << (int) itR->m_index
                << std::endl;
    }

}


// ------------ method called once each job just before starting event loop  ------------
void 
DumpConeDefinition::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
DumpConeDefinition::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(DumpConeDefinition);
