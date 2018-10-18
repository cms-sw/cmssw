#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondTools/L1TriggerExt/plugins/L1SubsystemKeysOnlineProdExt.h"

#include "CondTools/L1Trigger/interface/Exception.h"
#include "CondTools/L1TriggerExt/interface/DataWriterExt.h"

#include "CondFormats/L1TObjects/interface/L1TriggerKeyListExt.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListExtRcd.h"

#include "FWCore/Framework/interface/EventSetup.h"

L1SubsystemKeysOnlineProdExt::L1SubsystemKeysOnlineProdExt(const edm::ParameterSet& iConfig)
   : m_tscKey( iConfig.getParameter< std::string >( "tscKey" ) ),
     m_rsKey ( iConfig.getParameter< std::string >( "rsKey"  ) ),
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
   std::unique_ptr<L1TriggerKeyExt> pL1TriggerKey ;

   // Get L1TriggerKeyListExt
   L1TriggerKeyListExt keyList ;
if( !m_forceGeneration ){
   l1t::DataWriterExt dataWriter ;
   if( !dataWriter.fillLastTriggerKeyList( keyList ) )
     {
       edm::LogError( "L1-O2O" )
	 << "Problem getting last L1TriggerKeyListExt" ;
     }
}
   // For RUN II the subsystem CondFormats for ALGO and RS are merged together -> we join ALGO and RS keys


   // combine the ALGO and RS keys:
   std::string delimeter = ":";
   std::string m_Key = m_tscKey + delimeter + m_rsKey;

   // If L1TriggerKeyListExt does not contain TSC key, token is empty
   if( keyList.token( m_Key ).empty() ||
       m_forceGeneration )
     {
       // Instantiate new L1TriggerKey
       pL1TriggerKey = std::make_unique< L1TriggerKeyExt >();

       pL1TriggerKey->setTSCKey( m_Key ) ;

       edm::LogVerbatim( "L1-O2O" ) << "TSC KEY:	" << m_tscKey ;
       edm::LogVerbatim( "L1-O2O" ) << "RS_KEY: 	" << m_rsKey ;

       // Get subsystem keys from OMDS

       // select * from CMS_TRG_L1_CONF.L1_TRG_CONF_KEYS where ID = m_tscKey
       std::vector< std::string > queryStrings ;
       queryStrings.push_back( "UGT_KEY"    ) ;
       queryStrings.push_back( "UGMT_KEY"   ) ;
//       queryStrings.push_back( "CALO_KEY"   ) ;
       queryStrings.push_back( "BMTF_KEY"   ) ;
       queryStrings.push_back( "OMTF_KEY"   ) ;
       queryStrings.push_back( "EMTF_KEY"   ) ;
       queryStrings.push_back( "TWINMUX_KEY") ;

       l1t::OMDSReader::QueryResults subkeyResults =
	 m_omdsReader.basicQuery( queryStrings,
				  "CMS_TRG_L1_CONF",
				  "L1_TRG_CONF_KEYS",
				  "L1_TRG_CONF_KEYS.ID",
				  m_omdsReader.singleAttribute( m_tscKey ) ) ;

       if( subkeyResults.queryFailed() ||
	   subkeyResults.numberRows() != 1 ) // check query successful
	 {
	   edm::LogError( "L1-O2O" ) << "Problem with subsystem TSC key: " << m_tscKey ;
	   return pL1TriggerKey ;
	 }

       std::string uGTKey, uGMTKey, CALOKey, BMTFKey, OMTFKey, EMTFKey, TWINMUXKey;

       subkeyResults.fillVariable( "UGT_KEY",     uGTKey    ) ;
       subkeyResults.fillVariable( "UGMT_KEY",    uGMTKey   ) ;
//       subkeyResults.fillVariable( "CALO_KEY",    CALOKey   ) ;
       subkeyResults.fillVariable( "BMTF_KEY",    BMTFKey   ) ;
       subkeyResults.fillVariable( "OMTF_KEY",    OMTFKey   ) ;
       subkeyResults.fillVariable( "EMTF_KEY",    EMTFKey   ) ;
       subkeyResults.fillVariable( "TWINMUX_KEY", TWINMUXKey) ;

       // For RUN II the subsystem CondFormats for ALGO and RS are merged together -> we join ALGO and RS keys

       queryStrings.clear();
       queryStrings.push_back( "UGT_RS_KEY"    );
       queryStrings.push_back( "UGMT_RS_KEY"   );
//       queryStrings.push_back( "CALO_RS_KEY" );
       queryStrings.push_back( "BMTF_RS_KEY"   );
       queryStrings.push_back( "EMTF_RS_KEY"   );
       queryStrings.push_back( "OMTF_RS_KEY"   );
       queryStrings.push_back( "TWINMUX_RS_KEY");

       subkeyResults =
	 m_omdsReader.basicQuery( queryStrings,
				  "CMS_TRG_L1_CONF",
				  "L1_TRG_RS_KEYS",
				  "L1_TRG_RS_KEYS.ID",
				  m_omdsReader.singleAttribute( m_rsKey ) ) ;

       if( subkeyResults.queryFailed() ||
	   subkeyResults.numberRows() != 1 ) // check query successful
	 {
	   edm::LogError( "L1-O2O" ) << "Problem with subsystem RS key: " << m_rsKey ;
	   return pL1TriggerKey ;
	 }

       std::string uGTrsKey, uGMTrsKey, CALOrsKey, BMTFrsKey, OMTFrsKey, EMTFrsKey, TWINMUXrsKey;

       subkeyResults.fillVariable( "UGT_RS_KEY",     uGTrsKey    ) ;
       subkeyResults.fillVariable( "UGMT_RS_KEY",    uGMTrsKey   ) ;
//       subkeyResults.fillVariable( "CALO_RS_KEY",    CALOrsKey ) ;
       subkeyResults.fillVariable( "BMTF_RS_KEY",    BMTFrsKey   ) ;
       subkeyResults.fillVariable( "OMTF_RS_KEY",    OMTFrsKey   ) ;
       subkeyResults.fillVariable( "EMTF_RS_KEY",    EMTFrsKey   ) ;
       subkeyResults.fillVariable( "TWINMUX_RS_KEY", TWINMUXrsKey) ;

// The offline CALO folks want to have CALOL1 and CALOL2 together -> provide the top level TSC key for the kCALO payload and let them handle the rest
       CALOKey   = m_tscKey;
       CALOrsKey = m_rsKey;

       pL1TriggerKey->setSubsystemKey( L1TriggerKeyExt::kuGT,    uGTKey     + delimeter + uGTrsKey    ) ;
       pL1TriggerKey->setSubsystemKey( L1TriggerKeyExt::kuGMT,   uGMTKey    + delimeter + uGMTrsKey   ) ;
       pL1TriggerKey->setSubsystemKey( L1TriggerKeyExt::kCALO,   CALOKey    + delimeter + CALOrsKey ) ;
       pL1TriggerKey->setSubsystemKey( L1TriggerKeyExt::kBMTF,   BMTFKey    + delimeter + BMTFrsKey   ) ;
       pL1TriggerKey->setSubsystemKey( L1TriggerKeyExt::kOMTF,   OMTFKey    + delimeter + OMTFrsKey   ) ;
       pL1TriggerKey->setSubsystemKey( L1TriggerKeyExt::kEMTF,   EMTFKey    + delimeter + EMTFrsKey   ) ;
       pL1TriggerKey->setSubsystemKey( L1TriggerKeyExt::kTWINMUX,TWINMUXKey + delimeter + TWINMUXrsKey) ;

       edm::LogVerbatim( "L1-O2O" ) << "UGT_KEY:	" << uGTKey   ;
       edm::LogVerbatim( "L1-O2O" ) << "UGT_RS_KEY:	" << uGTrsKey ;
       edm::LogVerbatim( "L1-O2O" ) << "UGMT_KEY:	" << uGMTKey  ;
       edm::LogVerbatim( "L1-O2O" ) << "UGMT_RS_KEY:	" << uGMTrsKey ;
       edm::LogVerbatim( "L1-O2O" ) << "CALO_KEY:	" << CALOKey  ;
       edm::LogVerbatim( "L1-O2O" ) << "CALO_RS_KEY:	" << CALOrsKey ;
       edm::LogVerbatim( "L1-O2O" ) << "BMTF_KEY:	" << BMTFKey  ;
       edm::LogVerbatim( "L1-O2O" ) << "BMTF_RS_KEY:	" << BMTFrsKey ;
       edm::LogVerbatim( "L1-O2O" ) << "OMTF_KEY:	" << OMTFKey  ;
       edm::LogVerbatim( "L1-O2O" ) << "OMTF_RS_KEY:	" << OMTFrsKey ;
       edm::LogVerbatim( "L1-O2O" ) << "EMTF_KEY:	" << EMTFKey  ;
       edm::LogVerbatim( "L1-O2O" ) << "EMTF_RS_KEY:	" << EMTFrsKey ;
       edm::LogVerbatim( "L1-O2O" ) << "TWINMUX_KEY:	" << TWINMUXKey;
       edm::LogVerbatim( "L1-O2O" ) << "TWINMUX_RS_KEY:	" << TWINMUXrsKey ;

   }
   else
   {
     throw l1t::DataAlreadyPresentException(
        "L1TriggerKeyExt for TSC key " + m_tscKey + " and RS key " + m_rsKey + " already in CondDB." ) ;
   }

   return pL1TriggerKey ;
}

//define this as a plug-in
//DEFINE_FWK_EVENTSETUP_MODULE(L1SubsystemKeysOnlineProdExt);
