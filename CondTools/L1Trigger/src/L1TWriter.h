#ifndef CondTools_L1Trigger_L1TWriter
#define CondTools_L1Trigger_L1TWriter

#include <string>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/RelationalStorageManager.h"
#include "CondCore/DBCommon/interface/PoolStorageManager.h"
#include "CondCore/MetaDataService/interface/MetaData.h"

namespace l1t
{
//
// class decleration
//

/* This class will store L1TriggerKey and associated data to the DB.
 *
 * This class use case is as follows:
 *  1. Generate new L1TriggerKey with unique value.
 *  2. Save this L1TriggerKey to database with provided tag (from config file) and IOV
 *      that is extracted from EventSetup.
 *  3. Save all other L1Trigger configuration data to DB using infinitive IOV and tag that is
 *      taken from L1TriggerKey value generated in first step.
 */
class L1TWriter : public edm::EDAnalyzer {
   public:
      explicit L1TWriter(const edm::ParameterSet&);
      ~L1TWriter();

      /* Performs saving of the data
       */
       void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);

   private:
      /* Function that generates name for the tag that should be used to save real data
       * (e.g. CSCTP parameters). Currently it is generated as a string that includes:
       *  * Name of L1TriggerKey tag used
       *  * Validity time start
       */
      std::string generateDataTagName (const unsigned long long sinceTime) const;

      /* Loads IOV coresponding to provided tag from database. If this tag is not found,
       * i.e. it is new tag, then empty string is returned
       */
      std::string getIOVToken (const std::string & tag) const;

      // L1 Key Configuration data
      std::string keyTagName;
      std::string keyIOVToken;

      // sesion configuration data
      cond::DBSession * session;
      cond::RelationalStorageManager * coral;
      cond::PoolStorageManager * pool;
      cond::MetaData * metadata;
};

}
#endif
