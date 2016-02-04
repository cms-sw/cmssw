#include "CondTools/L1Trigger/interface/L1ConfigOnlineProdBase.h"
#include "CondFormats/L1TObjects/interface/L1MuGMTChannelMask.h"
#include "CondFormats/DataRecord/interface/L1MuGMTChannelMaskRcd.h"

class L1MuGMTChannelMaskOnlineProducer : public L1ConfigOnlineProdBase< L1MuGMTChannelMaskRcd, L1MuGMTChannelMask > {
   public:
      L1MuGMTChannelMaskOnlineProducer(const edm::ParameterSet& iConfig)
         : L1ConfigOnlineProdBase< L1MuGMTChannelMaskRcd, L1MuGMTChannelMask >( iConfig ) {}
      ~L1MuGMTChannelMaskOnlineProducer() {}

      virtual boost::shared_ptr< L1MuGMTChannelMask > newObject( const std::string& objectKey ) ;
   private:
};

boost::shared_ptr< L1MuGMTChannelMask >
L1MuGMTChannelMaskOnlineProducer::newObject( const std::string& objectKey )
{

   std::vector<std::string> columns;
   columns.push_back("ENABLE_RPCB");
   columns.push_back("ENABLE_CSC");
   columns.push_back("ENABLE_DT");
   columns.push_back("ENABLE_RPCF");

   // Execute SQL queries to get data from OMDS (using key) and make C++ object
   // Example: SELECT A_PARAMETER FROM CMS_XXX.XXX_CONF WHERE XXX_CONF.XXX_KEY = objectKey
   l1t::OMDSReader::QueryResults results =
       m_omdsReader.basicQuery(
         columns,
         "CMS_GMT",
         "GMT_RUN_SETTINGS",
         "GMT_RUN_SETTINGS.ID",
         m_omdsReader.singleAttribute( objectKey ) ) ;

   if( results.queryFailed() ) // check if query was successful
   {
      edm::LogError( "L1-O2O" ) << "L1MuGMTChannelMaskOnlineProducer: Problem getting " << objectKey << " key from GMT_RUN_SETTING." ;
      return boost::shared_ptr< L1MuGMTChannelMask >() ;
   }

   unsigned mask = 0;
   bool maskaux;
   results.fillVariable( "ENABLE_RPCB", maskaux ) ;
   if(!maskaux) mask|=2;
   results.fillVariable( "ENABLE_CSC", maskaux ) ;
   if(!maskaux) mask|=4;
   results.fillVariable( "ENABLE_DT", maskaux ) ;
   if(!maskaux) mask|=1;
   results.fillVariable( "ENABLE_RPCF", maskaux ) ;
   if(!maskaux) mask|=8;

   boost::shared_ptr< L1MuGMTChannelMask > gmtchanmask = boost::shared_ptr< L1MuGMTChannelMask >( new L1MuGMTChannelMask() );

   gmtchanmask->setSubsystemMask(mask);
   
   return gmtchanmask;
}

DEFINE_FWK_EVENTSETUP_MODULE(L1MuGMTChannelMaskOnlineProducer);
