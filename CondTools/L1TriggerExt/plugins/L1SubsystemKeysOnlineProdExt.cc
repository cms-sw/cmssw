#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondTools/L1TriggerExt/plugins/L1SubsystemKeysOnlineProdExt.h"

#include "CondTools/L1Trigger/interface/Exception.h"
#include "CondTools/L1TriggerExt/interface/DataWriterExt.h"

#include "CondFormats/L1TObjects/interface/L1TriggerKeyListExt.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListExtRcd.h"

#include "FWCore/Framework/interface/EventSetup.h"

L1SubsystemKeysOnlineProdExt::L1SubsystemKeysOnlineProdExt(const edm::ParameterSet& iConfig)
   : m_tscKey( iConfig.getParameter< std::string >( "tscKey" ) ),
     m_omdsReader(
	iConfig.getParameter< std::string >( "onlineDB" ),
	iConfig.getParameter< std::string >( "onlineAuthentication" ) ),
     m_forceGeneration( iConfig.getParameter< bool >( "forceGeneration" ) )
{
   //the following line is needed to tell the framework what
   // data is being produced
  setWhatProduced(this, "SubsystemKeysOnly");

   //now do what ever other initialization is needed
}


L1SubsystemKeysOnlineProdExt::~L1SubsystemKeysOnlineProdExt()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
L1SubsystemKeysOnlineProdExt::ReturnType
L1SubsystemKeysOnlineProdExt::produce(const L1TriggerKeyExtRcd& iRecord)
{
   using namespace edm::es;
   boost::shared_ptr<L1TriggerKeyExt> pL1TriggerKey ;

   // Get L1TriggerKeyListExt
   L1TriggerKeyListExt keyList ;
   l1t::DataWriterExt dataWriter ;
   if( !dataWriter.fillLastTriggerKeyList( keyList ) )
     {
       edm::LogError( "L1-O2O" )
	 << "Problem getting last L1TriggerKeyListExt" ;
     }

   // If L1TriggerKeyListExt does not contain TSC key, token is empty
   if( keyList.token( m_tscKey ) == std::string() ||
       m_forceGeneration )
     {
       // Instantiate new L1TriggerKey
       pL1TriggerKey = boost::shared_ptr< L1TriggerKeyExt >(
	 new L1TriggerKeyExt() ) ;
       pL1TriggerKey->setTSCKey( m_tscKey ) ;

       edm::LogVerbatim( "L1-O2O" ) << "TSC KEY " << m_tscKey ;

       // Get subsystem keys from OMDS

       // SELECT uGT_KEY FROM TRIGGERSUP_CONF WHERE TRIGGERSUP_CONF.TS_KEY = m_tscKey
       std::vector< std::string > queryStrings ;
       queryStrings.push_back( "UGT_KEY" ) ;

       l1t::OMDSReader::QueryResults subkeyResults =
	 m_omdsReader.basicQuery( queryStrings,
				  "CMS_TRG_L1_CONF",
				  "L1_TRG_CONF_KEYS",
				  "L1_TRG_CONF_KEYS.ID",
				  m_omdsReader.singleAttribute( m_tscKey ) ) ;

       if( subkeyResults.queryFailed() ||
	   subkeyResults.numberRows() != 1 ) // check query successful
	 {
	   edm::LogError( "L1-O2O" ) << "Problem with subsystem keys." ;
	   return pL1TriggerKey ;
	 }

       std::string uGTKey;

       subkeyResults.fillVariable( "UGT_KEY", uGTKey ) ;
       pL1TriggerKey->setSubsystemKey( L1TriggerKeyExt::kuGT, uGTKey ) ;
       edm::LogVerbatim( "L1-O2O" ) << "UGT_KEY " << uGTKey ;

   }
   else
   {
     throw l1t::DataAlreadyPresentException(
        "L1TriggerKeyExt for TSC key " + m_tscKey + " already in CondDB." ) ;
   }

   return pL1TriggerKey ;
}

//define this as a plug-in
//DEFINE_FWK_EVENTSETUP_MODULE(L1SubsystemKeysOnlineProdExt);
