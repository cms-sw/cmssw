#include "CondTools/L1Trigger/interface/L1ObjectKeysOnlineProdBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class L1GctTSCObjectKeysOnlineProd : public L1ObjectKeysOnlineProdBase {
   public:
      L1GctTSCObjectKeysOnlineProd(const edm::ParameterSet& iConfig)
         : L1ObjectKeysOnlineProdBase( iConfig ) {}
      ~L1GctTSCObjectKeysOnlineProd() {}

      virtual void fillObjectKeys( ReturnType pL1TriggerKey ) ;
   private:
};

void
L1GctTSCObjectKeysOnlineProd::fillObjectKeys( ReturnType pL1TriggerKey )
{
      // kMYSUBSYSTEM = kCSCTF, kDTTF, kRPC, kGMT, kRCT, kGCT, kGT, or kTSP0
      // subsystemKey = TRIGGERSUP_CONF.{CSCTF_KEY, DTTF_KEY, RPC_KEY, GMT_KEY, RCT_KEY, GCT_KEY, GT_KEY}
      std::string subsystemKey = pL1TriggerKey->subsystemKey( L1TriggerKey::kGCT ) ;

      if( !subsystemKey.empty() )
      {
         // Execute SQL queries to get data from OMDS (using key) and make C++ object.
         // Example: SELECT A_PARAMETER FROM CMS_XXX.XXX_CONF WHERE XXX_CONF.XXX_KEY = subsystemKey
         l1t::OMDSReader::QueryResults objectKeyResults =
         m_omdsReader.basicQuery(
            "CONFIG_KEY",
            "CMS_GCT",
            "XXX_CONF",
            "XXX_CONF.CONFIG_KEY",
            m_omdsReader.singleAttribute( subsystemKey  ) );

         std::string objectKey ;

         // check if query was successful
         if( objectKeyResults.queryFailed() )
         {
             edm::LogError("L1-O2O")
                     << "Problem with key for record L1GctJetFinderParamsRcd: query failed ";
         }
         else if( objectKeyResults.numberRows() != 1 )
         {
             edm::LogError("L1-O2O")
                     << "Problem with key for record L1GctJetFinderParamsRcd: "
                     << (objectKeyResults.numberRows()) << " rows were returned";
         }
         else
         {
            objectKeyResults.fillVariable( objectKey ) ;
         }

         pL1TriggerKey->add( "L1GctJetFinderParamsRcd", "L1GctJetFinderParams", objectKey ) ;

         // Repeat for additional object keys for this subsystem.
      }
}

DEFINE_FWK_EVENTSETUP_MODULE(L1GctTSCObjectKeysOnlineProd);
