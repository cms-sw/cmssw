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
  
  if( emMaskResults.queryFailed() ) { // check if query was successful
    edm::LogError( "L1-O2O" ) << "Problem with L1GctChannelMask EM mask for key "  << objectKey;
    return boost::shared_ptr< L1GctChannelMask >() ;
  }
  
  int emMask = -1;
  emMaskResults.fillVariable( emMask ) ;
  
  // get region masks
  l1t::OMDSReader::QueryResults rgnMaskKeyResults =
    m_omdsReader.basicQuery(
			    "GCT_RGN_MASK_KEY",
			    "CMS_GCT",
			    "GCT_MASKS",
			    "GCT_MASKS.CONFIG_KEY",
			    m_omdsReader.singleAttribute( objectKey ) ) ;
  
  if( rgnMaskKeyResults.queryFailed() ) { // check if query was successful
    edm::LogError( "L1-O2O" ) << "Problem with L1GctChannelMask region mask for key "  << objectKey;
    return boost::shared_ptr< L1GctChannelMask >() ;
  }

  std::string rgnKey;
  rgnMaskKeyResults.fillVariable( rgnKey );

  std::vector< std::string > rgnMaskCols;
  rgnMaskCols.push_back("RCT_CRATE_0_RGN_MASK");
  rgnMaskCols.push_back("RCT_CRATE_1_RGN_MASK");
  rgnMaskCols.push_back("RCT_CRATE_2_RGN_MASK");
  rgnMaskCols.push_back("RCT_CRATE_3_RGN_MASK");
  rgnMaskCols.push_back("RCT_CRATE_4_RGN_MASK");
  rgnMaskCols.push_back("RCT_CRATE_5_RGN_MASK");
  rgnMaskCols.push_back("RCT_CRATE_6_RGN_MASK");
  rgnMaskCols.push_back("RCT_CRATE_7_RGN_MASK");
  rgnMaskCols.push_back("RCT_CRATE_8_RGN_MASK");
  rgnMaskCols.push_back("RCT_CRATE_9_RGN_MASK");
  rgnMaskCols.push_back("RCT_CRATE_10_RGN_MASK");
  rgnMaskCols.push_back("RCT_CRATE_11_RGN_MASK");
  rgnMaskCols.push_back("RCT_CRATE_12_RGN_MASK");
  rgnMaskCols.push_back("RCT_CRATE_13_RGN_MASK");
  rgnMaskCols.push_back("RCT_CRATE_14_RGN_MASK");
  rgnMaskCols.push_back("RCT_CRATE_15_RGN_MASK");
  rgnMaskCols.push_back("RCT_CRATE_16_RGN_MASK");
  rgnMaskCols.push_back("RCT_CRATE_17_RGN_MASK");

  l1t::OMDSReader::QueryResults rgnMaskResults =
    m_omdsReader.basicQuery(
			    rgnMaskCols,
			    "CMS_GCT",
			    "GCT_RGN_MASKS",
			    "GCT_RGN_MASKS.CONFIG_KEY",
			    m_omdsReader.singleAttribute( rgnKey ) ) ;
  

  // get energy sum masks
  l1t::OMDSReader::QueryResults esumMaskKeyResults =
    m_omdsReader.basicQuery(
			    "GCT_ESUM_MASK_KEY",
			    "CMS_GCT",
			    "GCT_MASKS",
			    "GCT_MASKS.CONFIG_KEY",
			    m_omdsReader.singleAttribute( objectKey ) ) ;
  
  if( esumMaskKeyResults.queryFailed() ) { // check if query was successful
    edm::LogError( "L1-O2O" ) << "Problem with L1GctChannelMask energy sum mask for key "  << objectKey;
    return boost::shared_ptr< L1GctChannelMask >() ;
  }

  std::string esumKey;
  esumMaskKeyResults.fillVariable( esumKey );

  std::vector< std::string > esumMaskCols;
  esumMaskCols.push_back("GCT_TET_MASK");
  esumMaskCols.push_back("GCT_MET_MASK");
  esumMaskCols.push_back("GCT_HT_MASK");
  esumMaskCols.push_back("GCT_MHT_MASK");

  l1t::OMDSReader::QueryResults esumMaskResults =
    m_omdsReader.basicQuery(
			    esumMaskCols,
			    "CMS_GCT",
			    "GCT_ESUM_MASKS",
			    "GCT_ESUM_MASKS.CONFIG_KEY",
			    m_omdsReader.singleAttribute( esumKey ) ) ;


  // create masks object
  boost::shared_ptr< L1GctChannelMask > masks( new L1GctChannelMask() );
  
  // set EM masks
  for (int i=0; i<18; i++) {
    if ((emMask & (1<<i)) != 0) masks->maskEmCrate(i);  // mask crate if emMask bit i is set
  }
 
  // set region masks
  for (unsigned irct=0; irct<18; irct++) {
    std::stringstream rgnCol;
    rgnCol << "RCT_CRATE_" << std::dec << irct << "_RGN_MASK";
    int mask;
    rgnMaskResults.fillVariable( rgnCol.str(), mask );
    if (mask != 0) {
      for (unsigned irgn=0; irgn<22; ++irgn) {
	if ((mask & (1<<irgn)) != 0) {
	  edm::LogError( "L1-O2O" ) << "Masked region, but no O2O code!";
	}
      }
    }
  }

  // set esum masks
  int tetMask, metMask, htMask, mhtMask ;
  esumMaskResults.fillVariable( "GCT_TET_MASK", tetMask ) ;
  esumMaskResults.fillVariable( "GCT_MET_MASK", metMask ) ;
  esumMaskResults.fillVariable( "GCT_HT_MASK", htMask ) ;
  esumMaskResults.fillVariable( "GCT_MHT_MASK", mhtMask ) ;

  for (int ieta=0; ieta<22; ieta++) {
    if ((tetMask & (1<<ieta)) != 0) masks->maskTotalEt(ieta);  
    if ((metMask & (1<<ieta)) != 0) masks->maskMissingEt(ieta);  
    if ((htMask & (1<<ieta)) != 0) masks->maskTotalHt(ieta);  
    if ((mhtMask & (1<<ieta)) != 0) masks->maskMissingHt(ieta);  
  }
    
  return masks;
  
}

DEFINE_FWK_EVENTSETUP_MODULE(L1GctChannelMaskOnlineProd);
