#include "CondTools/L1Trigger/interface/L1ConfigOnlineProdBase.h"
#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"
#include "CondFormats/DataRecord/interface/L1GctJetFinderParamsRcd.h"

class L1GctJetFinderParamsOnlineProd : public L1ConfigOnlineProdBase< L1GctJetFinderParamsRcd, L1GctJetFinderParams > {
   public:
      L1GctJetFinderParamsOnlineProd(const edm::ParameterSet& iConfig)
         : L1ConfigOnlineProdBase< L1GctJetFinderParamsRcd, L1GctJetFinderParams >( iConfig ) {}
      ~L1GctJetFinderParamsOnlineProd() {}

      virtual boost::shared_ptr< L1GctJetFinderParams > newObject( const std::string& objectKey ) ;
   private:
};

boost::shared_ptr< L1GctJetFinderParams >
L1GctJetFinderParamsOnlineProd::newObject( const std::string& objectKey )
{
   // Execute SQL queries to get data from OMDS (using key) and make C++ object
   // Example: SELECT A_PARAMETER FROM CMS_XXX.XXX_CONF WHERE XXX_CONF.XXX_KEY = objectKey

  // get parameters
  std::vector< std::string > columns;
  columns.push_back( "GCT_RGN_ET_LSB" );
  columns.push_back( "GCT_HT_LSB" );
  columns.push_back( "GCT_CJET_SEED_ET_THRESHOLD" );
  columns.push_back( "GCT_TJET_SEED_ET_THRESHOLD" );
  columns.push_back( "GCT_FJET_SEED_ET_THRESHOLD" );
  columns.push_back( "GCT_HT_JET_ET_THRESHOLD" );
  columns.push_back( "GCT_MHT_JET_ET_THRESHOLD" );
  columns.push_back( "GCT_TAU_ISO_ET_THRESHOLD" );
  columns.push_back( "GCT_CEN_JET_ETA_MAX" );
  columns.push_back( "GCT_JET_CORR_KEY" );
  
  l1t::OMDSReader::QueryResults results =
    m_omdsReader.basicQuery(
			    columns,
			    "CMS_GCT",
			    "GCT_PHYS_PARAMS",
			    "GCT_PHYS_PARAMS.CONFIG_KEY",
			    m_omdsReader.singleAttribute( objectKey ) ) ;

   if( results.queryFailed() ) // check if query was successful
   {
      edm::LogError( "L1-O2O" ) << "Problem with L1GctJetFinderParams key." ;
      return boost::shared_ptr< L1GctJetFinderParams >() ;
   }

   // fill values
   double rgnEtLsb=0.;
   double htLsb=0.;
   double cJetSeed=0.;
   double tJetSeed=0.;
   double fJetSeed=0.;
   double tauIsoEtThresh=0.;
   double htJetEtThresh=0.;
   double mhtJetEtThresh=0.;
   short int etaBoundary=7;
   int corrType=0;
   std::vector< std::vector<double> > jetCorrCoeffs;
   std::vector< std::vector<double> > tauCorrCoeffs;
   bool convertToEnergy=false;            // Not in OMDS
   std::vector<double> energyConvCoeffs(11);  // Not in OMDS
   std::string jetCorrKey;

   results.fillVariable( "GCT_RGN_ET_LSB", rgnEtLsb );
   results.fillVariable( "GCT_HT_LSB", htLsb );
   results.fillVariable( "GCT_CJET_SEED_ET_THRESHOLD", cJetSeed );
   results.fillVariable( "GCT_TJET_SEED_ET_THRESHOLD", tJetSeed );
   results.fillVariable( "GCT_FJET_SEED_ET_THRESHOLD", fJetSeed );
   results.fillVariable( "GCT_TAU_ISO_ET_THRESHOLD", tauIsoEtThresh );
   results.fillVariable( "GCT_HT_JET_ET_THRESHOLD", htJetEtThresh );
   results.fillVariable( "GCT_MHT_JET_ET_THRESHOLD", mhtJetEtThresh );
   results.fillVariable( "GCT_CEN_JET_ETA_MAX", etaBoundary );
   results.fillVariable( "GCT_JET_CORR_KEY", jetCorrKey );

   edm::LogInfo("L1-O2O") << "L1 jet corrections key : " << jetCorrKey << std::endl;

   // get jet corr coefficients
   std::vector< std::string > jetCorrColumns;
   jetCorrColumns.push_back( "GCT_JETCORR_TYPE" );
   jetCorrColumns.push_back( "GCT_JETCORR_NETA_10" );
   jetCorrColumns.push_back( "GCT_JETCORR_NETA_9" );
   jetCorrColumns.push_back( "GCT_JETCORR_NETA_8" );
   jetCorrColumns.push_back( "GCT_JETCORR_NETA_7" );
   jetCorrColumns.push_back( "GCT_JETCORR_NETA_6" );
   jetCorrColumns.push_back( "GCT_JETCORR_NETA_5" );
   jetCorrColumns.push_back( "GCT_JETCORR_NETA_4" );
   jetCorrColumns.push_back( "GCT_JETCORR_NETA_3" );
   jetCorrColumns.push_back( "GCT_JETCORR_NETA_2" );
   jetCorrColumns.push_back( "GCT_JETCORR_NETA_1" );
   jetCorrColumns.push_back( "GCT_JETCORR_NETA_0" );
   jetCorrColumns.push_back( "GCT_JETCORR_PETA_0" );
   jetCorrColumns.push_back( "GCT_JETCORR_PETA_1" );
   jetCorrColumns.push_back( "GCT_JETCORR_PETA_2" );
   jetCorrColumns.push_back( "GCT_JETCORR_PETA_3" );
   jetCorrColumns.push_back( "GCT_JETCORR_PETA_4" );
   jetCorrColumns.push_back( "GCT_JETCORR_PETA_5" );
   jetCorrColumns.push_back( "GCT_JETCORR_PETA_6" );
   jetCorrColumns.push_back( "GCT_JETCORR_PETA_7" );
   jetCorrColumns.push_back( "GCT_JETCORR_PETA_8" );
   jetCorrColumns.push_back( "GCT_JETCORR_PETA_9" );
   jetCorrColumns.push_back( "GCT_JETCORR_PETA_10" );
   
   l1t::OMDSReader::QueryResults jetCorrResults =
    m_omdsReader.basicQuery(
			    jetCorrColumns,
			    "CMS_GCT",
			    "GCT_JET_CORRECTIONS",
			    "GCT_JET_CORRECTIONS.CONFIG_KEY",
			    m_omdsReader.singleAttribute( jetCorrKey ) ) ;

   if( jetCorrResults.queryFailed() ) // check if query was successful
   {
      edm::LogError( "L1-O2O" ) << "Problem getting L1 jet corrections" ;
      return boost::shared_ptr< L1GctJetFinderParams >() ;
   }

   // fill jet corr type
   jetCorrResults.fillVariable( "GCT_JETCORR_TYPE", corrType );

   edm::LogInfo("L1-O2O") << "L1 jet corrections type : " << corrType << std::endl;

   // get coefficients
   for (unsigned i=0; i < L1GctJetFinderParams::NUMBER_ETA_VALUES; ++i) {

     // get corr key for eta value
     std::stringstream etaCol;
     etaCol << "GCT_JETCORR_NETA_" << std::dec << i;
     std::string etaKey;
     jetCorrResults.fillVariable( etaCol.str(), etaKey );

     std::vector< std::string > coeffColumns;
     coeffColumns.push_back( "GCT_JETCORR_C0" );
     coeffColumns.push_back( "GCT_JETCORR_C1" );
     coeffColumns.push_back( "GCT_JETCORR_C2" );
     coeffColumns.push_back( "GCT_JETCORR_C3" );
     coeffColumns.push_back( "GCT_JETCORR_C4" );
     coeffColumns.push_back( "GCT_JETCORR_C5" );
     coeffColumns.push_back( "GCT_JETCORR_C6" );
     coeffColumns.push_back( "GCT_JETCORR_C7" );
     coeffColumns.push_back( "GCT_JETCORR_C8" );
     coeffColumns.push_back( "GCT_JETCORR_C9" );
     coeffColumns.push_back( "GCT_JETCORR_C10" );
     coeffColumns.push_back( "GCT_JETCORR_C11" );
     coeffColumns.push_back( "GCT_JETCORR_C12" );
     coeffColumns.push_back( "GCT_JETCORR_C13" );
     coeffColumns.push_back( "GCT_JETCORR_C14" );
     coeffColumns.push_back( "GCT_JETCORR_C15" );
     coeffColumns.push_back( "GCT_JETCORR_C16" );
     coeffColumns.push_back( "GCT_JETCORR_C17" );
     coeffColumns.push_back( "GCT_JETCORR_C18" );
     coeffColumns.push_back( "GCT_JETCORR_C19" );
     
     l1t::OMDSReader::QueryResults jetCorrResults =
       m_omdsReader.basicQuery(
			       coeffColumns,
			       "CMS_GCT",
			       "GCT_JET_COEFFS",
			       "GCT_JET_COEFFS.CONFIG_KEY",
			       m_omdsReader.singleAttribute( etaKey ) ) ;
     
     if( results.queryFailed() ) // check if query was successful
       {
	 edm::LogError( "L1-O2O" ) << "Problem getting L1 jet correction coefficients" ;
	 return boost::shared_ptr< L1GctJetFinderParams >() ;
       }

     // fill coeffs - TODO
     std::vector<double> coeffs;

     unsigned nCoeffs = 0;
     if (corrType == 0) nCoeffs = 0;
     else {
       edm::LogError( "L1-O2O" ) << "Unsupported jet correction type : " << corrType ;
       return boost::shared_ptr< L1GctJetFinderParams >() ;
     }

     for (unsigned j=0; j< nCoeffs; ++j) {
       std::stringstream coeffCol;
       coeffCol << "GCT_JETCORR_C" << std::dec << j;
       int coeff;
       jetCorrResults.fillVariable( coeffCol.str(), coeff );

       coeffs.push_back(coeff);
     }

     jetCorrCoeffs.push_back(coeffs);
    
     // copy to tau coeffs
     if (i < L1GctJetFinderParams::N_CENTRAL_ETA_VALUES) tauCorrCoeffs.push_back(coeffs);
     
   }
   
   

   return boost::shared_ptr< L1GctJetFinderParams >( 
						    new L1GctJetFinderParams( rgnEtLsb,
									      htLsb,
									      cJetSeed,
									      fJetSeed,
									      tJetSeed,
									      tauIsoEtThresh,
									      htJetEtThresh,
									      mhtJetEtThresh,
									      etaBoundary,
									      corrType,
									      jetCorrCoeffs,
									      tauCorrCoeffs,
									      convertToEnergy,
									      energyConvCoeffs

 )
						    );

}


DEFINE_FWK_EVENTSETUP_MODULE(L1GctJetFinderParamsOnlineProd);
