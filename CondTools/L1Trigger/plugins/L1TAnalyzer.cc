//
// Original Author:  Giedrius Bacevicius
//         Created:  Wed Jul 11 13:52:35 CEST 2007
// $Id$


// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "CondFormats/L1TObjects/interface/L1TriggerKey.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"
#include "CondFormats/L1TObjects/interface/L1CSCTPParameters.h"
#include "CondFormats/DataRecord/interface/L1CSCTPParametersRcd.h"

#include "CondCore/DBCommon/interface/Exception.h"

#include "CondTools/L1Trigger/plugins/L1TAnalyzer.h"

//
// constructors and destructor
//
L1TAnalyzer::L1TAnalyzer(const edm::ParameterSet& iConfig)
{
}


L1TAnalyzer::~L1TAnalyzer()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}

// ------------ method called to for each event  ------------
void L1TAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    // Try to get key
    edm::ESHandle<L1TriggerKey> handle;
    iSetup.get<L1TriggerKeyRcd> ().get (handle);
    const L1TriggerKey * key = handle.product ();
    assert (!key->get ("L1CSCTPParametersRcd", "L1CSCTPParameters").empty ());

    //std::cerr << "Got key: " << key->getKey () << std::endl;

    // and CSCTPParameters
    edm::ESHandle<L1CSCTPParameters> cscHandle;
    iSetup.get<L1CSCTPParametersRcd> ().get (cscHandle);
    const L1CSCTPParameters * csc = cscHandle.product ();

    std::cerr << "CSCTP got: " << csc->alctFifoTbins () << std::endl;
}

// ------------ method called once each job just before starting event loop  ------------
void L1TAnalyzer::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void L1TAnalyzer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TAnalyzer);

