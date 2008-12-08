#include "L1TriggerConfig/CSCTFConfigProducers/interface/CSCTFObjectKeysOnlineProd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

void
CSCTFObjectKeysOnlineProd::fillObjectKeys( ReturnType pL1TriggerKey )
{
      // kXXX = kCSCTF, kDTTF, kRPC, kGMT, kRCT, mkGCT, kGT, or kTSP0
      // subsystemKey = TRIGGERSUP_CONF.{CSCTF_KEY, DTTF_KEY, RPC_KEY, GMT_KEY, RCT_KEY, GCT_KEY, GT_KEY}
      std::string csctfKey = pL1TriggerKey->subsystemKey( L1TriggerKey::kCSCTF ) ;

      // Execute SQL queries to get data from OMDS (using key) and make C++ object.
      // Example: SELECT A_PARAMETER FROM CMS_XXX.XXX_CONF WHERE XXX_CONF.XXX_KEY = csctfKey
      l1t::OMDSReader::QueryResults objectKeyResults =
//       m_omdsReader.basicQuery(
//          "A_PARAMETER",
//          "CMS_XXX",
//          "XXX_CONF",
//          "XXX_CONF.XXX_KEY",
//          m_omdsReader.singleAttribute( csctfKey  ) );

      m_omdsReader.basicQuery(
         "SP1_KEY",
         "CMS_CSC_TF",
         "CSCTF_CONF",
         "CSCTF_CONF.CSCTF_KEY",
         m_omdsReader.singleAttribute( csctfKey  ) );

      if( objectKeyResults.queryFailed() || objectKeyResults.numberRows() != 1 ) // check if query was successful
      {
         edm::LogError( "L1-O2O" ) << "Problem with CSCTF  key." ;
         return ;
      }

      long long int objectKey ;
      objectKeyResults.fillVariable( objectKey ) ;
       std::stringstream ss;
       ss<<objectKey;
       std::string strKey=ss.str();


      pL1TriggerKey->add( "L1MuCSCTFConfigurationRcd", "L1MuCSCTFConfiguration", strKey ) ;
}


