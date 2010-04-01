#include "CondTools/L1Trigger/interface/L1ConfigOnlineProdBase.h"
#include "CondFormats/L1TObjects/interface/L1GctChannelMask.h"
#include "CondFormats/DataRecord/interface/L1GctChannelMaskRcd.h"

class L1GctChannelMaskOnlineProd : public L1ConfigOnlineProdBase< L1GctChannelMaskRcd, L1GctChannelMask > {
   public:
      L1GctChannelMaskOnlineProd(const edm::ParameterSet& iConfig)
         : L1ConfigOnlineProdBase< L1GctChannelMaskRcd, L1GctChannelMask >( iConfig ) {}
      ~L1GctChannelMaskOnlineProd() {}

      virtual boost::shared_ptr< L1GctChannelMask > newObject( const std::string& objectKey ) ;
   private:
};

boost::shared_ptr< L1GctChannelMask >
L1GctChannelMaskOnlineProd::newObject( const std::string& objectKey )
{
  // get EM mask data
  l1t::OMDSReader::QueryResults emMaskResults =
    m_omdsReader.basicQuery(
			    "GCT_EM_MASK",
			    "CMS_GCT",
			    "GCT_MASKS",
			    "GCT_MASKS.CONFIG_KEY",
			    m_omdsReader.singleAttribute( objectKey ) ) ;
  
  if( emMaskResults.queryFailed() ) // check if query was successful
    {
      edm::LogError( "L1-O2O" ) << "Problem with L1GctChannelMask EM mask key." ;
      return boost::shared_ptr< L1GctChannelMask >() ;
    }
  
  int emMask ;
  emMaskResults.fillVariable( emMask ) ;
  
  // create masks object
  boost::shared_ptr< L1GctChannelMask > masks( new L1GctChannelMask() );
  
  // set EM masks
  for (int i=0; i<18; i++) {
    if (emMask & (1<<i) != 0) masks->maskEmCrate(i);  // mask crate if emMask bit i is set
  }
  
  // set region masks
  
  return masks;
  
}

DEFINE_FWK_EVENTSETUP_MODULE(L1GctChannelMaskOnlineProd);
