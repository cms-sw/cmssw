#include "CondTools/L1Trigger/interface/L1ObjectKeysOnlineProdBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


class L1GctConfigKeyOnlineProducer : public L1ObjectKeysOnlineProdBase {
   public:
      L1GctConfigKeyOnlineProducer(const edm::ParameterSet& iConfig)
         : L1ObjectKeysOnlineProdBase( iConfig ) 
      {
	LogDebug( "L1-O2O" ) << "L1GctConfigKeyOnlineProducer created"  << std::endl;
      }
      ~L1GctConfigKeyOnlineProducer() {}

      virtual void fillObjectKeys( ReturnType pL1TriggerKey ) ;
   private:
};



void
L1GctConfigKeyOnlineProducer::fillObjectKeys( ReturnType pL1TriggerKey )
{
    
      std::string subsystemKey = pL1TriggerKey->subsystemKey( L1TriggerKey::kGCT ) ;

      l1t::OMDSReader::QueryResults keyResults =
      m_omdsReader.basicQuery(
          // SELECTed column
	  "LUT_KEY",
	  // schema name
	  "CMS_GCT",
	  // table name
          "GCT_CONFIG",
	  // WHERE lhs
	  "GCT_CONFIG.CONFIG_KEY",
	  // WHERE rhs
	  m_omdsReader.singleAttribute( subsystemKey  ) );

      if( lutKeyResults.queryFailed() || keyResults.numberRows() != 1 ) // check if query was successful
      {
         edm::LogError( "L1-O2O" ) << "Problem extracting GCT key from GCT config key." ;
         return ;
      }

      std::string objectKey ;
      softwareConfigKeyResults.fillVariable(objectKey) ;

      //      pL1TriggerKey->add( "L1MuGMTParametersRcd", "L1MuGMTParameters", objectKey ) ;
}

DEFINE_FWK_EVENTSETUP_MODULE(L1GctConfigKeyOnlineProducer);
