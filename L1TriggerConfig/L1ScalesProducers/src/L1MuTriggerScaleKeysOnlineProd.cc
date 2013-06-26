//-------------------------------------------------
//
//   \class L1MuGMTParametersOnlineProd
//
//   Description:      A key producer to deduce the GMT LUT keys from the master 
//                     GMT configuration  key, closely following the example of 
//
//   https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideL1ConfigOnlineProd r11
//
//   $Date: 2009/03/18 14:13:10 $
//   $Revision: 1.3 $
//
//   Author :
//   Thomas Themel
//
//--------------------------------------------------

#include "CondTools/L1Trigger/interface/L1ObjectKeysOnlineProdBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondCore/DBCommon/interface/Exception.h"


class L1MuTriggerScaleKeysOnlineProd : public L1ObjectKeysOnlineProdBase {
   public:
      L1MuTriggerScaleKeysOnlineProd(const edm::ParameterSet& iConfig)
          : L1ObjectKeysOnlineProdBase( iConfig ),
            m_objectTypes(iConfig.getParameter<std::vector<std::string> >("objectTypes")),
            m_recordTypes(iConfig.getParameter<std::vector<std::string> >("recordTypes"))
      {
        if(m_objectTypes.size() != m_recordTypes.size()) { 
            throw cond::Exception("mismatch: need equal number objectType and recordType entries!");
        }        
      }

      ~L1MuTriggerScaleKeysOnlineProd() {}

      virtual void fillObjectKeys( ReturnType pL1TriggerKey ) ;

   private:    
    std::vector<std::string> m_objectTypes;
    std::vector<std::string> m_recordTypes;

};

void
L1MuTriggerScaleKeysOnlineProd::fillObjectKeys( ReturnType pL1TriggerKey )
{
    
      std::string subsystemKey = pL1TriggerKey->subsystemKey( L1TriggerKey::kGMT ) ;

      if( !subsystemKey.empty() )
	{
      // Lookup GT scales key from GMT main config key
      l1t::OMDSReader::QueryResults scalesKeyResults =
      m_omdsReader.basicQuery(
          // SELECTed column
	  "SCALES_KEY",
	  // schema name
	  "CMS_GMT",
	  // table name
          "GMT_CONFIG",
	  // WHERE lhs
	  "GMT_CONFIG.KEY",
	  // WHERE rhs
	  m_omdsReader.singleAttribute( subsystemKey  ) );

      if( scalesKeyResults.numberRows() != 1 ) // check if query was successful
      {
         edm::LogError( "L1-O2O" ) << "Problem extracting GMT scales key from GMT config key." ;
         return ;
      }

      std::string objectKey ;
      scalesKeyResults.fillVariable(objectKey) ;

      edm::LogError( "L1-O2O" ) << "Registering " << m_recordTypes.size() << " keys ";
      // register the scales key for all the scales types we need to produce
      for(unsigned i = 0; i < m_recordTypes.size() ; ++i) { 
	edm::LogError( "L1-O2O" ) << "Registering scales key " << objectKey << " for " << m_recordTypes[i];
          pL1TriggerKey->add(m_recordTypes[i], m_objectTypes[i], objectKey ) ;
      }
	}
}

DEFINE_FWK_EVENTSETUP_MODULE(L1MuTriggerScaleKeysOnlineProd);
