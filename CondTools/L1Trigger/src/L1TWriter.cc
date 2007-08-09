// -*- C++ -*-
//
// Package:    CondTools/L1Trigger
// Class:      L1TWriter
// 
/**\class L1TWriter.cc CondTools/L1Trigger/src/L1TWriter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Giedrius Bacevicius
//         Created:  Wed Jul 11 13:52:35 CEST 2007
// $Id$
//
//


// system include files
#include <memory>
#include <iostream>
#include <sstream>

// user include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ValidityInterval.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/Ref.h"

#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVEditor.h"

#include "CondFormats/L1TObjects/interface/L1TriggerKey.h"
#include "CondFormats/L1TObjects/interface/L1CSCTPParameters.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"
#include "CondFormats/DataRecord/interface/L1CSCTPParametersRcd.h"

#include "CondTools/L1Trigger/src/L1TWriter.h"

namespace l1t
{
    
//
// constructors and destructor
//
L1TWriter::L1TWriter(const edm::ParameterSet& iConfig)
    : keyTagName(iConfig.getParameter<std::string> ("keyTag"))
{
    // Initialize session used with this object
    session = new cond::DBSession ();
    session->sessionConfiguration ().setAuthenticationMethod (cond::Env);
//    session->sessionConfiguration ().setMessageLevel (cond::Info);
    session->connectionConfiguration ().enableConnectionSharing ();
    session->connectionConfiguration ().enableReadOnlySessionOnUpdateConnections ();

    // create Coral connection and pool. This ones should be connected on required basis
    coral = new cond::RelationalStorageManager (iConfig.getParameter<std::string> ("connect"), session);
    pool = new cond::PoolStorageManager (iConfig.getParameter<std::string> ("connect"),
            iConfig.getParameter<std::string> ("catalog") , session);

    // and data object
    metadata = new cond::MetaData (*coral);

    // load key IOV token to use futher
    keyIOVToken = getIOVToken (keyTagName);
}


L1TWriter::~L1TWriter()
{
    delete metadata;
    delete coral;
    delete pool;

    delete session;
}

std::string L1TWriter::generateDataTagName (const unsigned long long sinceTime) const
{
    std::stringstream ss;
    ss << "L1Key:" << keyTagName << "_since:" << sinceTime;
    return ss.str ();
}

std::string L1TWriter::getIOVToken (const std::string & tag) const
{
    // I need to add ReadWriteCreate because DB may not be created at the moment
    coral->connect (cond::ReadWriteCreate);
    coral->startTransaction (false);

    // if available - load token
    // if not return empty string
    std::string iovToken;
    if (metadata->hasTag (tag))
         iovToken = metadata->getToken (tag);

    coral->commit ();
    coral->disconnect ();

    return iovToken;
}

// ------------ method called to for each event  ------------
void L1TWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    // save key to DB
    // here we need to insert current IOV into a list of previous ones
    // This can be easaly done via PoolDBService, but due to limited interface i have to do it myself...
    L1TriggerKey * newKey = new L1TriggerKey (generateDataTagName (iEvent.id ().run ()));

    // put data to DB
    pool->connect ();
    pool->startTransaction (false);

    cond::IOVService * manager = new cond::IOVService (*pool);
    cond::IOVEditor * editor;

    // addMapping should be called only if this is a new record
    // thervise I suppose one should provide key to IOVEditor
    bool addMappingRequired = false;
    if (keyIOVToken.empty ())
    {
        // first key with this tag
        addMappingRequired = true;
        editor = manager->newIOVEditor ();
    }
    else
        editor = manager->newIOVEditor (keyIOVToken);

    cond::Ref<L1TriggerKey> keyRef (*pool, newKey);
    keyRef.markWrite ("L1TriggerKeyRcd");
    editor->insert (iEvent.id ().run (), keyRef.token ());
    keyIOVToken = editor->token ();

    delete editor;
    editor = manager->newIOVEditor ();

    // now save real configuration data
    // this part is easy - we just add the data with infinitive IOV and some custom key
    edm::ESHandle<L1CSCTPParameters> handle;
    iSetup.get<L1CSCTPParametersRcd> ().get (handle);

    cond::Ref<L1CSCTPParameters> cscRef (*pool, new L1CSCTPParameters (*(handle.product ())));
    cscRef.markWrite ("L1CSCTPParametersRcd");
    std::string cscRefToken = cscRef.token ();
    editor->insert (edm::IOVSyncValue::endOfTime ().eventID ().run (), cscRefToken);
    std::string dataIOVToken = editor->token ();

    delete editor;
    delete manager;

    // update all metadata - add mappings
    pool->commit ();
    pool->disconnect ();

    coral->connect (cond::ReadWrite);
    coral->startTransaction (true);

    // see if this the first key, so that we need to add mapping
    if (addMappingRequired)
        metadata->addMapping (keyTagName, keyIOVToken);
    metadata->addMapping (newKey->getKey (), dataIOVToken);

    coral->commit ();
    coral->disconnect ();
}

}   // ns

