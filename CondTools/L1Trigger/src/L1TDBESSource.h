#ifndef CONDTOOLS_L1Trigger_L1DBESSOURCE_H
#define CONDTOOLS_L1Trigger_L1DBESSOURCE_H

#include <memory>
#include <map>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/DataProxyProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"

#include "CondFormats/DataRecord/interface/L1CSCTPParametersRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"
#include "CondFormats/L1TObjects/interface/L1CSCTPParameters.h"
#include "CondFormats/L1TObjects/interface/L1TriggerKey.h"

#include "CondTools/L1Trigger/src/Interval.h"
#include "CondTools/L1Trigger/src/DataReader.h"


// -*- C++ -*-
//
// Package:    L1TDBESSource
// Class:      L1TDBESSource
// 
/**\class L1TDBESSource L1TDBESSource.h CondTools/L1TDBESSource/src/L1TDBESSource.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Giedrius Bacevicius
//         Created:  Thu Jul 19 13:14:44 CEST 2007
// $Id: L1TDBESSource.h,v 1.2 2007/08/14 12:10:24 giedrius Exp $
namespace l1t
{
/* Class that will load data from PoolDB and stores it into EventSetup.
 *
 * Loading data from DB consists from several steps:
 *   1. Load L1TriggerKey from db with given tag and IOV value from EventSetup
 *   2. Load other objects (at the moment L1CSCTPParameters) with tag stored in loaded
 *      L1TriggerKey. IOV does not matter
 *   3. Save loaded objects into EventSetup.
 */
class L1TDBESSource : public edm::eventsetup::DataProxyProvider, public edm::EventSetupRecordIntervalFinder
{
public:
    /* Constructors and destructors */
    L1TDBESSource(const edm::ParameterSet&);
    ~L1TDBESSource();

protected:
    /* Returns IOV interval which includes provided IOVSyncValue. This interval
     * is equal to IOV of L1TriggerKey that is valid at the given time. There should
     * be only one such interval.
     */
    virtual void setIntervalFor (const edm::eventsetup::EventSetupRecordKey&,
                         const edm::IOVSyncValue &,
                         edm::ValidityInterval&);

    /* Methods that are required to register and provide proxies for the key record and all others */
    virtual void registerProxies(const edm::eventsetup::EventSetupRecordKey& recordKey, KeyedProxies& proxyList);
    void registerKey (const edm::eventsetup::EventSetupRecordKey& recordKey, KeyedProxies& proxyList);
    void registerPayload (const edm::eventsetup::EventSetupRecordKey& recordKey, KeyedProxies& proxyList);

    /* Does nothing very interesting, just resets all proxies */
    virtual void newInterval(const edm::eventsetup::EventSetupRecordKey& recordKey, const edm::ValidityInterval& interval);

    /* Informs framework, that we can produce record with provided name */
    void registerRecord (const std::string & record);

    /* Loads all IOV stored in DB for provided tag */
    std::string keyTag;
    DataReader reader;

    /* List of items that we are suposed to load. This is a list of record->types (may be several) */
    typedef std::map<std::string, std::set<std::string> > Items;
    Items items;
};

}   // ns

#endif

