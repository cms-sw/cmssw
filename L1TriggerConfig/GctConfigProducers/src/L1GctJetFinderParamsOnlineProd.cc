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
   l1t::OMDSReader::QueryResults results =
       m_omdsReader.basicQuery(
         "A_DATUM",
         "CMS_XXX",
         "XXX_CONF",
         "XXX_CONF.XXX_KEY",
         m_omdsReader.singleAttribute( objectKey ) ) ;

   if( results.queryFailed() ) // check if query was successful
   {
      edm::LogError( "L1-O2O" ) << "Problem with L1RCTParameters key." ;
      return boost::shared_ptr< L1GctJetFinderParams >() ;
   }

   double rgnEtLsb=0.;
   double htLsb=0.;
   double cJetSeed=0.;
   double tJetSeed=0.;
   double fJetSeed=0.;
   double tauIsoEtThresh=0.;
   double htJetEtThresh=0.;
   double mhtJetEtThresh=0.;
   unsigned etaBoundary=7;

   // how to get data out of results???

   return boost::shared_ptr< L1GctJetFinderParams >( 
						    new L1GctJetFinderParams( rgnEtLsb,
									      htLsb,
									      cJetSeed,
									      fJetSeed,
									      tJetSeed,
									      tauIsoEtThresh,
									      htJetEtThresh,
									      mhtJetEtThresh,
									      etaBoundary )
						    );

}


DEFINE_FWK_EVENTSETUP_MODULE(L1GctJetFinderParamsOnlineProd);
