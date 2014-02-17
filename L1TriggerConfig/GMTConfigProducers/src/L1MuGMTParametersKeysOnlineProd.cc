//-------------------------------------------------
//
//   \class L1MuGMTParametersOnlineProd
//
//   Description:      A key producer to deduce the GMT LUT keys from the master 
//                     GMT configuration  key, closely following the example of 
//
//   https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideL1ConfigOnlineProd r11
//
//   $Date: 2009/03/13 17:55:39 $
//   $Revision: 1.2 $
//
//   Author :
//   Thomas Themel
//
//--------------------------------------------------

#include "CondTools/L1Trigger/interface/L1ObjectKeysOnlineProdBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


class L1MuGMTParametersKeysOnlineProd : public L1ObjectKeysOnlineProdBase {
   public:
      L1MuGMTParametersKeysOnlineProd(const edm::ParameterSet& iConfig)
         : L1ObjectKeysOnlineProdBase( iConfig ) 
      {
	LogDebug( "L1-O2O" ) << "L1MuGMTParametersKeysOnlineProd created"  << std::endl;
      }
      ~L1MuGMTParametersKeysOnlineProd() {}

      virtual void fillObjectKeys( ReturnType pL1TriggerKey ) ;
   private:
};



void
L1MuGMTParametersKeysOnlineProd::fillObjectKeys( ReturnType pL1TriggerKey )
{
    
      std::string subsystemKey = pL1TriggerKey->subsystemKey( L1TriggerKey::kGMT ) ;

      if( !subsystemKey.empty() )
	{

      l1t::OMDSReader::QueryResults lutKeyResults =
      m_omdsReader.basicQuery(
          // SELECTed column
	  "LUT_KEY",
	  // schema name
	  "CMS_GMT",
	  // table name
          "GMT_CONFIG",
	  // WHERE lhs
	  "GMT_CONFIG.KEY",
	  // WHERE rhs
	  m_omdsReader.singleAttribute( subsystemKey  ) );

      if( lutKeyResults.queryFailed() || lutKeyResults.numberRows() != 1 ) // check if query was successful
      {
         edm::LogError( "L1-O2O" ) << "Problem extracting GMT LUT key from GMT config key." ;
         return ;
      }

      l1t::OMDSReader::QueryResults softwareConfigKeyResults =
      m_omdsReader.basicQuery(
	  // SELECTed column
         "GMT_SOFTWARE_CONFIG",
          // schema name
         "CMS_GMT",
	  // table name
         "GMT_LUTS",
	  // WHERE lhs
         "GMT_LUTS.KEY",
	  // WHERE rhs
         lutKeyResults);

      if( softwareConfigKeyResults.queryFailed() || softwareConfigKeyResults.numberRows() != 1 ) // check if query was successful
      {
         edm::LogError( "L1-O2O" ) << "Problem extracting GMT software config key from GMT config key." ;
         return ;
      }

      std::string objectKey ;
      softwareConfigKeyResults.fillVariable(objectKey) ;

      pL1TriggerKey->add( "L1MuGMTParametersRcd", "L1MuGMTParameters", objectKey ) ;
	}
}

DEFINE_FWK_EVENTSETUP_MODULE(L1MuGMTParametersKeysOnlineProd);
