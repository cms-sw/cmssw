/**
 * \class L1MuGMTRSKeysOnlineProd
 *
 *
 * Description: online producer for GMT RUN SETTINGS.
 *
 * Implementation:
 *    
 *
 * \author: Ivan Mikulec
 *
 * $Date: 2009/03/25 20:51:08 $
 * $Revision: 1.1 $
 *
 */


#include "CondTools/L1Trigger/interface/L1ObjectKeysOnlineProdBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class L1MuGMTRSKeysOnlineProd : public L1ObjectKeysOnlineProdBase {
   public:
      L1MuGMTRSKeysOnlineProd(const edm::ParameterSet& iConfig) ;
      ~L1MuGMTRSKeysOnlineProd() {}

      virtual void fillObjectKeys( ReturnType pL1TriggerKey ) ;
   private:
      bool m_enableL1MuGMTChannelMask ;
};

L1MuGMTRSKeysOnlineProd::L1MuGMTRSKeysOnlineProd(const edm::ParameterSet& iConfig)
   : L1ObjectKeysOnlineProdBase( iConfig ),
     m_enableL1MuGMTChannelMask( iConfig.getParameter< bool >( "enableL1MuGMTChannelMask" ) )
{
}

void
L1MuGMTRSKeysOnlineProd::fillObjectKeys( ReturnType pL1TriggerKey )
{
   if( m_enableL1MuGMTChannelMask )
   {
      // Execute SQL queries to get data from OMDS (using key) and make C++ object.
      // Example: SELECT A_KEY FROM CMS_XXX.XXX_RUN_SETTINGS_KEYS_CURRENT
      l1t::OMDSReader::QueryResults objectKeyResults =
         m_omdsReader.basicQuery(
            "GMT_RUN_SETTINGS_FK",
            "CMS_GMT",
            "GMT_RUN_SETTINGS_KEY_CURRENT" );

      std::string objectKey ;

      // check if query was successful
      if( objectKeyResults.queryFailed() )
      {
          edm::LogError("L1-O2O")
                  << "Problem with key for record L1MuGMTChannelMaskRcd: query failed ";
      }
      else if( objectKeyResults.numberRows() != 1 )
      {
          edm::LogError("L1-O2O")
                  << "Problem with key for record L1MuGMTChannelMaskRcd: "
                  << (objectKeyResults.numberRows()) << " rows were returned";
      }
      else
      {
         objectKeyResults.fillVariable( objectKey ) ;
      }

      pL1TriggerKey->add( "L1MuGMTChannelMaskRcd", "L1MuGMTChannelMask", objectKey ) ;
   }
}

DEFINE_FWK_EVENTSETUP_MODULE(L1MuGMTRSKeysOnlineProd);
