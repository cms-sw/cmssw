#ifndef CondTools_L1Trigger_L1TWriter
#define CondTools_L1Trigger_L1TWriter

#include <string>
#include <map>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "CondTools/L1Trigger/src/DataWriter.h"

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
      /* Helper function that will write payload to database extracted from EventSetup
       */
      template<typename Record, typename Value>
      void writePayload (L1TriggerKey & key, const std::string & recordName, const edm::EventSetup & iSetup);

      std::string keyTag;
      int sinceRun;

      // Used to make sura that sinceRun is not set when we are running
      // this class for several runs
      int executionNumber;

      DataWriter writer;

      // store list of records and associated items to extract from the EventSetup
      // formats is as follows:
      // RecordName -> list of items to extract from it
      typedef std::map<std::string, std::set<std::string> > Items;
      Items items;
};

}
#endif
