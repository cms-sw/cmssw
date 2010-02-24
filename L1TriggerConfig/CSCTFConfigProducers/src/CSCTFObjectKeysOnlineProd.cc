#include "L1TriggerConfig/CSCTFConfigProducers/interface/CSCTFObjectKeysOnlineProd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

void CSCTFObjectKeysOnlineProd::fillObjectKeys( ReturnType pL1TriggerKey )
{
  std::string csctfKey = pL1TriggerKey->subsystemKey( L1TriggerKey::kCSCTF ) ;



  if( !csctfKey.empty() )
    {
      //----------------------------------------------------------------------------
      // register the main CSCTF key
      pL1TriggerKey->add( "L1MuCSCTFConfigurationRcd", 
			  "L1MuCSCTFConfiguration", 
			  csctfKey ) ;
      //----------------------------------------------------------------------------
      
      //----------------------------------------------------------------------------
      // PT LUT
      //
      // while the sp configuration can change from sector to sector, the pt LUT file 
      //loaded in the CSCTF boards is the same for each SP => the same PTLUT_VERSION
      // RETRIEVE THE VERSION FROM THE SP1 CONFIGURATION 
      //
      // e.g., SELECT PTLUT_VERSION FROM CMS_CSC_TF.CSCTF_SP_CONF WHERE CSCTF_SP_CONF.SP_KEY = '1702100001'
      // e.g., CSCTF key of the type day/month/year, e.g. 170210 = 17th February 2010
      std::string sp1key = csctfKey + "0001";

      // query
      l1t::OMDSReader::QueryResults objectKeyResults =
	m_omdsReader.basicQuery(
				"PTLUT_VERSION",
				"CMS_CSC_TF",
				"CSCTF_SP_CONF",
				"CSCTF_SP_CONF.SP_KEY",
				m_omdsReader.singleAttribute( sp1key  ) );

      // check if query was successful
      if( objectKeyResults.queryFailed() || objectKeyResults.numberRows() != 1 ) 
	{
	  edm::LogError( "L1-O2O" ) << "Problem with CSCTF key while retrieving "
				    << "the PTLUT_VERSION" ;
	  return ;
	}
	 
      // register the pt LUT key
      std::string ptLutKey;
      objectKeyResults.fillVariable( ptLutKey ) ;
      pL1TriggerKey->add( "L1MuCSCPtLutRcd", "L1MuCSCPtLut", ptLutKey ) ;
      //----------------------------------------------------------------------------
    }
}

