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

   // TODO - get calibration coefficients
   

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
   unsigned corrType=0;
   std::vector< std::vector<double> > jetCorrCoeffs(11);
   std::vector< std::vector<double> > tauCorrCoeffs(7);
   bool convertToEnergy=false;            // Not in OMDS
   std::vector<double> energyConvCoeffs(11);  // Not in OMDS

   results.fillVariable( "GCT_RGN_ET_LSB", rgnEtLsb );
   results.fillVariable( "GCT_HT_LSB", htLsb );
   results.fillVariable( "GCT_CJET_SEED_ET_THRESHOLD", cJetSeed );
   results.fillVariable( "GCT_TJET_SEED_ET_THRESHOLD", tJetSeed );
   results.fillVariable( "GCT_FJET_SEED_ET_THRESHOLD", fJetSeed );
   results.fillVariable( "GCT_TAU_ISO_ET_THRESHOLD", tauIsoEtThresh );
   results.fillVariable( "GCT_HT_JET_ET_THRESHOLD", htJetEtThresh );
   results.fillVariable( "GCT_MHT_JET_ET_THRESHOLD", mhtJetEtThresh );
   results.fillVariable( "GCT_CEN_JET_ETA_MAX", etaBoundary );

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
