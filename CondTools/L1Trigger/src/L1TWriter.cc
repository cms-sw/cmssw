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
// $Id: L1TWriter.cc,v 1.2 2007/08/12 15:46:09 giedrius Exp $
//
//


// system include files
#include <memory>
#include <iostream>
#include <sstream>
#include <algorithm>

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
    : sinceRun (-1), executionNumber (0),
    writer (iConfig.getParameter<std::string> ("connect"), iConfig.getParameter<std::string> ("catalog"))
{
    std::vector<std::string> params = iConfig.getParameterNames ();
    if (std::find (params.begin (), params.end (), std::string ("keyTag")) != params.end ())
        keyTag = std::string (iConfig.getParameter<std::string> ("keyTag"));

    if (std::find (params.begin (), params.end (), std::string ("keyValue")) != params.end ())
        keyValue = std::string (iConfig.getParameter<std::string> ("keyValue"));

    if (std::find (params.begin (), params.end (), std::string ("sinceRun")) != params.end ())
        sinceRun = iConfig.getParameter<int> ("sinceRun");

    // Load records that we have to save
    if (std::find (params.begin (), params.end (), std::string ("toSave")) != params.end ())
    {
        typedef std::vector<edm::ParameterSet> ToSave;
        ToSave toSave = iConfig.getParameter<ToSave> ("toSave");
        for (ToSave::const_iterator it = toSave.begin (); it != toSave.end (); it++)
        {
            std::string record = it->getParameter<std::string> ("record");
            std::vector<std::string> recordItems = it->getParameter<std::vector<std::string> > ("data");

            // Copy items to the list items list
            Items::iterator rec = items.insert (std::make_pair (record, std::set<std::string> ())).first;
            for (std::vector<std::string>::const_iterator it = recordItems.begin (); it != recordItems.end (); it ++)
                rec->second.insert (*it);
        }
    }
}


L1TWriter::~L1TWriter()
{
}

L1TriggerKey L1TWriter::createKey (const std::string & tag, const unsigned long long run) const
{
    if (!keyValue.empty ())
        return L1TriggerKey (keyValue);
    else
        return L1TriggerKey::fromRun (tag, run);
}

// ------------ method called to for each event  ------------
void L1TWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    // sanity check, to make sure we are not abusing system
    if (executionNumber != 0)
    {
        // we can't have sinceRun set when running this for second time
        if (sinceRun > 0)
            throw cond::Exception ("L1TWriter") << "We are executing this analyzer for the second time. "
                << "However, sinceRun parameter is set. This means that I have to save twice the same "
                << "L1 key for the same run.";

        // Check if we are not forcing to use keyValues
        // TODO: should we use not infitive IOVs here?
        if (!keyValue.empty ())
            throw cond::Exception ("L1TWriter") << "We are executing this analyzer for the second time. "
                << "However, keyValue parameter is set. This means that I have to save twice the same "
                << "payload with infitivine IOV.";

    }

    executionNumber ++;
    // if we are using sinceRun, do not try to run this method more then once
    unsigned long long run = iEvent.id ().run ();
    if (sinceRun > 0)
        run = sinceRun;

    // save key to DB
    // here we need to insert current IOV into a list of previous ones
    // This can be easaly done via PoolDBService, but due to limited interface i have to do it myself...
    L1TriggerKey newKey = createKey (keyTag, run);
    if (!keyTag.empty ())
        writer.writeKey (new L1TriggerKey (newKey), keyTag, run);

    // Write payload if required
    // TODO: dummy check, this should be moved to more elegant logic
    if (items ["L1CSCTPParametersRcd"].find ("L1CSCTPParameters") != items ["L1CSCTPParametersRcd"].end ())
        writePayload<L1CSCTPParametersRcd, L1CSCTPParameters> (newKey, "L1CSCTPParametersRcd", iSetup);
}

template<typename Record, typename Value>
void L1TWriter::writePayload (const L1TriggerKey & key, const std::string & recordName, const edm::EventSetup & iSetup)
{
    edm::ESHandle<Value> handle;
    iSetup.get<Record> ().get (handle);

    writer.writePayload (key, new Value (*(handle.product ())), recordName);
}

}   // ns

