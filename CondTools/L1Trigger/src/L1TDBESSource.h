#ifndef CONDTOOLS_L1Trigger_L1DBESSOURCE_H
#define CONDTOOLS_L1Trigger_L1DBESSOURCE_H

#include <memory>
#include <map>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProducts.h"

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
// $Id: L1TDBESSource.h,v 1.1 2007/08/09 14:53:59 giedrius Exp $
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
class L1TDBESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder
{
public:
    /* Constructors and destructors */
    L1TDBESSource(const edm::ParameterSet&);
    ~L1TDBESSource();

    // I can't put all results in a single method, becaus that would mean that
    // this data should go into same Record. We whant to have a seperate record
    // for each product
    /* These two methods will do the actual data loading from DB */
    std::auto_ptr<L1TriggerKey> produceKey(const L1TriggerKeyRcd&);
    std::auto_ptr<L1CSCTPParameters> produceCSC(const L1CSCTPParametersRcd&);
protected:
    /* Returns IOV interval which includes provided IOVSyncValue. This interval
     * is equal to IOV of L1TriggerKey that is valid at the given time. There should
     * be only one such interval.
     */
    virtual void setIntervalFor (const edm::eventsetup::EventSetupRecordKey&,
                         const edm::IOVSyncValue &,
                         edm::ValidityInterval&);

    /* Loads all IOV stored in DB for provided tag */
    std::string keyTag;
    DataReader reader;

    typedef std::map<std::string, std::set<std::string> > Items;
    Items items;
};

}   // ns

#endif

