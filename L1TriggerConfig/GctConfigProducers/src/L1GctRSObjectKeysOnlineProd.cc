#include "CondTools/L1Trigger/interface/L1ObjectKeysOnlineProdBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class L1GctRSObjectKeysOnlineProd : public L1ObjectKeysOnlineProdBase {
   public:
      L1GctRSObjectKeysOnlineProd(const edm::ParameterSet& iConfig) ;
      ~L1GctRSObjectKeysOnlineProd() {}

      virtual void fillObjectKeys( ReturnType pL1TriggerKey ) ;
   private:
      bool m_enableL1GctChannelMask ;
};

L1GctRSObjectKeysOnlineProd::L1GctRSObjectKeysOnlineProd(const edm::ParameterSet& iConfig)
   : L1ObjectKeysOnlineProdBase( iConfig ),
     m_enableL1GctChannelMask( iConfig.getParameter< bool >( "enableL1GctChannelMask" ) )
{
}

void
L1GctRSObjectKeysOnlineProd::fillObjectKeys( ReturnType pL1TriggerKey )
{
   if( m_enableL1GctChannelMask )
   {
      // Execute SQL queries to get data from OMDS (using key) and make C++ object.
      // Example: SELECT A_KEY FROM CMS_XXX.XXX_RUN_SETTINGS_KEYS_CURRENT

      l1t::OMDSReader::QueryResults objectKeyResults =
         m_omdsReader.basicQuery(
            "GCT_MASKS_KEY",
            "CMS_GCT",
            "GCT_MASKS_CURRENT" );

      std::string maskKey ;

      // check if query was successful
      if( objectKeyResults.queryFailed() )
      {
          edm::LogError("L1-O2O")
                  << "Problem with key for record L1GctChannelMaskRcd: query failed ";
      }
      else if( objectKeyResults.numberRows() != 1 )
      {
          edm::LogError("L1-O2O")
                  << "Problem with key for record L1GctChannelMaskRcd: "
                  << (objectKeyResults.numberRows()) << " rows were returned";
      }
      else
      {
         objectKeyResults.fillVariable( maskKey ) ;
      }

      pL1TriggerKey->add( "L1GctChannelMaskRcd", "L1GctChannelMask", maskKey ) ;
   }
}

DEFINE_FWK_EVENTSETUP_MODULE(L1GctRSObjectKeysOnlineProd);
