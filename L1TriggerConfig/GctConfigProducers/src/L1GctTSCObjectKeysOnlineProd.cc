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

	// get main crate key
         l1t::OMDSReader::QueryResults mainCrateKeyResults =
	   m_omdsReader.basicQuery(
				   "GCT_MAIN_CRATE_KEY",
				   "CMS_GCT",
				   "GCT_CONFIG",
				   "GCT_CONFIG.CONFIG_KEY",
				   m_omdsReader.singleAttribute( subsystemKey  ) );


	 // get phys params key
         l1t::OMDSReader::QueryResults physParamsKeyResults =
	   m_omdsReader.basicQuery(
				   "GCT_PHYS_PARAMS_KEY",
				   "CMS_GCT",
				   "GCT_MAIN_CRATE",
				   "GCT_MAIN_CRATE.CONFIG_KEY",
				   mainCrateKeyResults );
	 
         std::string physParamsKey ;
	 
         if( physParamsKeyResults.queryFailed() ) {
	   edm::LogError("L1-O2O")
	     << "Problem with key for record L1GctJetFinderParamsRcd: query failed ";
	 }
         else if( physParamsKeyResults.numberRows() != 1 ) {
	   edm::LogError("L1-O2O")
	     << "Problem with key for record L1GctJetFinderParamsRcd: "
	     << (physParamsKeyResults.numberRows()) << " rows were returned";
	 }
         else {
	   physParamsKeyResults.fillVariable( physParamsKey ) ;
	 }
	 
         pL1TriggerKey->add( "L1GctJetFinderParamsRcd", "L1GctJetFinderParams", physParamsKey ) ;
         pL1TriggerKey->add( "L1JetEtScaleRcd", "L1CaloEtScale", physParamsKey ) ;
         pL1TriggerKey->add( "L1HtMissScaleRcd", "L1CaloEtScale", physParamsKey ) ;
         pL1TriggerKey->add( "L1HfRingEtScaleRcd", "L1CaloEtScale", physParamsKey ) ;
	 
      }
}

DEFINE_FWK_EVENTSETUP_MODULE(L1GctTSCObjectKeysOnlineProd);
