#include <iostream>
#include <fstream>
#include <stdexcept>

#include "CondTools/L1TriggerExt/interface/L1ConfigOnlineProdBaseExt.h"
#include "CondFormats/L1TObjects/interface/L1TGlobalPrescalesVetos.h"
#include "CondFormats/DataRecord/interface/L1TGlobalPrescalesVetosRcd.h"
#include "CondFormats/DataRecord/interface/L1TGlobalPrescalesVetosO2ORcd.h"
#include "L1Trigger/L1TGlobal/interface/PrescalesVetosHelper.h"
#include "L1Trigger/L1TCommon/interface/TriggerSystem.h"
#include "L1Trigger/L1TCommon/interface/XmlConfigParser.h"
#include "OnlineDBqueryHelper.h"

class L1TGlobalPrescalesVetosOnlineProd : public L1ConfigOnlineProdBaseExt<L1TGlobalPrescalesVetosO2ORcd,L1TGlobalPrescalesVetos> {
private:
public:
    virtual std::shared_ptr<L1TGlobalPrescalesVetos> newObject(const std::string& objectKey, const L1TGlobalPrescalesVetosO2ORcd& record) override ;

    L1TGlobalPrescalesVetosOnlineProd(const edm::ParameterSet&);
    ~L1TGlobalPrescalesVetosOnlineProd(void){}
};

L1TGlobalPrescalesVetosOnlineProd::L1TGlobalPrescalesVetosOnlineProd(const edm::ParameterSet& iConfig) : L1ConfigOnlineProdBaseExt<L1TGlobalPrescalesVetosO2ORcd,L1TGlobalPrescalesVetos>(iConfig) {}

std::shared_ptr<L1TGlobalPrescalesVetos> L1TGlobalPrescalesVetosOnlineProd::newObject(const std::string& objectKey, const L1TGlobalPrescalesVetosO2ORcd& record) {
    using namespace edm::es;

    edm::LogInfo( "L1-O2O: L1TGlobalPrescalesVetosOnlineProd" ) << "Producing L1TGlobalPrescalesVetos with RS key = " << objectKey ;

    if( objectKey.empty() ){
        edm::LogError( "L1-O2O: L1TGlobalPrescalesVetosOnlineProd" ) << "Key is empty";
        throw std::runtime_error("Empty objecKey");
    }

    std::vector< std::string > queryColumns ;
    queryColumns.push_back( "ALGOBX_MASK"     ) ;
    queryColumns.push_back( "ALGO_FINOR_MASK" ) ;
    queryColumns.push_back( "ALGO_FINOR_VETO" ) ;
    queryColumns.push_back( "ALGO_PRESCALE"   ) ;

    std::string prescale_key, bxmask_key, mask_key, vetomask_key;
    std::string xmlPayload_prescale, xmlPayload_mask_algobx, xmlPayload_mask_finor, xmlPayload_mask_veto;
    try {
        std::map<std::string,std::string> subKeys = l1t::OnlineDBqueryHelper::fetch( queryColumns, "UGT_RS_KEYS", objectKey, m_omdsReader );
        prescale_key = subKeys["ALGO_PRESCALE"];
        bxmask_key   = subKeys["ALGOBX_MASK"];
        mask_key     = subKeys["ALGO_FINOR_MASK"];
        vetomask_key = subKeys["ALGO_FINOR_VETO"];
        xmlPayload_prescale    = l1t::OnlineDBqueryHelper::fetch( {"CONF"}, "UGT_RS_CLOBS", prescale_key, m_omdsReader )["CONF"];
        xmlPayload_mask_algobx = l1t::OnlineDBqueryHelper::fetch( {"CONF"}, "UGT_RS_CLOBS", bxmask_key,   m_omdsReader )["CONF"];
        xmlPayload_mask_finor  = l1t::OnlineDBqueryHelper::fetch( {"CONF"}, "UGT_RS_CLOBS", mask_key,     m_omdsReader )["CONF"];
        xmlPayload_mask_veto   = l1t::OnlineDBqueryHelper::fetch( {"CONF"}, "UGT_RS_CLOBS", vetomask_key, m_omdsReader )["CONF"];
    } catch ( std::runtime_error &e ) {
        edm::LogError( "L1-O2O: L1TGlobalPrescalesVetosOnlineProd" ) << e.what();
        throw std::runtime_error("Broken key");
    }


    // for debugging purposes dump the payloads to /tmp
    std::ofstream output1(std::string("/tmp/").append(prescale_key.substr(0,prescale_key.find("/"))).append(".xml"));
    output1<<xmlPayload_prescale;
    output1.close();
    std::ofstream output2(std::string("/tmp/").append(mask_key.substr(0,mask_key.find("/"))).append(".xml"));
    output2<<xmlPayload_mask_finor;
    output2.close();
    std::ofstream output3(std::string("/tmp/").append(bxmask_key.substr(0,bxmask_key.find("/"))).append(".xml"));
    output3<<xmlPayload_mask_algobx;
    output3.close();
    std::ofstream output4(std::string("/tmp/").append(vetomask_key.substr(0,vetomask_key.find("/"))).append(".xml"));
    output4<<xmlPayload_mask_veto;
    output4.close();

/// the code below is adopted from https://github.com/cms-l1t-offline/cmssw/blob/thomreis_o2o-dev_CMSSW_8_0_7/L1Trigger/L1TGlobal/plugins/L1TGlobalPrescalesVetosESProducer.cc#L124-L365

  unsigned int m_numberPhysTriggers = 512;
  unsigned int m_bx_mask_default = 1;


  std::vector<std::vector<int> > prescales;
  std::vector<unsigned int> triggerMasks;
  std::vector<int> triggerVetoMasks;
  std::map<int, std::vector<int> > triggerAlgoBxMaskAlgoTrig;

  // Prescales
    l1t::XmlConfigParser xmlReader_prescale;
    l1t::TriggerSystem ts_prescale;
    ts_prescale.addProcessor("uGtProcessor", "uGtProcessor","-1","-1");

    // run the parser 
    xmlReader_prescale.readDOMFromString( xmlPayload_prescale ); // initialize it
    xmlReader_prescale.readRootElement( ts_prescale, "uGT" ); // extract all of the relevant context
    ts_prescale.setConfigured();

    const std::map<std::string, l1t::Parameter> &settings_prescale = ts_prescale.getParameters("uGtProcessor");
    std::map<std::string,unsigned int> prescaleColumns = settings_prescale.at("prescales").getColumnIndices();

    unsigned int numColumns_prescale = prescaleColumns.size();
    
    int NumPrescaleSets = numColumns_prescale - 1;
///  there may be "missing" rows/bits in the xml description meaning that triggers are not unused, so go for max
    std::vector<unsigned int> algoBits = settings_prescale.at("prescales").getTableColumn<unsigned int>("algo/prescale-index");
    int NumAlgos_prescale = *std::max_element(algoBits.begin(), algoBits.end()) + 1;

    if( NumPrescaleSets > 0 ){
      // Fill default prescale set
      for( int iSet=0; iSet<NumPrescaleSets; iSet++ ){
	prescales.push_back(std::vector<int>());
	for( int iBit = 0; iBit < NumAlgos_prescale; ++iBit ){
	  int inputDefaultPrescale = 0; // only prescales that are set in the block below are used
	  prescales[iSet].push_back(inputDefaultPrescale);
	}
      }

      for(auto &col : prescaleColumns){
        if( col.second<1 ) continue; // we don't care for the algorithms' indicies in 0th column
        int iSet = col.second - 1;
        std::vector<unsigned int> prescalesForSet = settings_prescale.at("prescales").getTableColumn<unsigned int>(col.first.c_str());
        for(unsigned int row=0; row<prescalesForSet.size(); row++){
          unsigned int prescale = prescalesForSet[row];
          unsigned int algoBit  = algoBits[row];
          prescales[iSet][algoBit] = prescale;
        }
      }
    }


  ///////////////////////////////////////////////////////////////////////////////////////////////////////////

  // finor mask
  // Setting mask to default 1 (unmask)
  for( unsigned int iAlg=0; iAlg < m_numberPhysTriggers; iAlg++ )
    triggerMasks.push_back(1);

    l1t::XmlConfigParser xmlReader_mask_finor;
    l1t::TriggerSystem ts_mask_finor;
    ts_mask_finor.addProcessor("uGtProcessor", "uGtProcessor","-1","-1");

    // run the parser 
    xmlReader_mask_finor.readDOMFromString( xmlPayload_mask_finor ); // initialize it
    xmlReader_mask_finor.readRootElement( ts_mask_finor, "uGT" ); // extract all of the relevant context
    ts_mask_finor.setConfigured();

    const std::map<std::string, l1t::Parameter>& settings_mask_finor = ts_mask_finor.getParameters("uGtProcessor");

    std::vector<unsigned int> algo_mask_finor = settings_mask_finor.at("finorMask").getTableColumn<unsigned int>("algo");
    std::vector<unsigned int> mask_mask_finor = settings_mask_finor.at("finorMask").getTableColumn<unsigned int>("mask");

    for(unsigned int row=0; row<algo_mask_finor.size(); row++){
      unsigned int algoBit = algo_mask_finor[row];
      unsigned int mask    = mask_mask_finor[row];
      if( algoBit < m_numberPhysTriggers ) triggerMasks[algoBit] = mask;
    }

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////

  // veto mask
  // Setting veto mask to default 0 (no veto)
  for( unsigned int iAlg=0; iAlg < m_numberPhysTriggers; iAlg++ )
    triggerVetoMasks.push_back(0);
  
    l1t::XmlConfigParser xmlReader_mask_veto;
    l1t::TriggerSystem ts_mask_veto;
    ts_mask_veto.addProcessor("uGtProcessor", "uGtProcessor","-1","-1");

    // run the parser 
    xmlReader_mask_veto.readDOMFromString( xmlPayload_mask_veto ); // initialize it
    xmlReader_mask_veto.readRootElement( ts_mask_veto, "uGT" ); // extract all of the relevant context
    ts_mask_veto.setConfigured();

    const std::map<std::string, l1t::Parameter>& settings_mask_veto = ts_mask_veto.getParameters("uGtProcessor");
    std::vector<unsigned int> algo_mask_veto = settings_mask_veto.at("vetoMask").getTableColumn<unsigned int>("algo");
    std::vector<unsigned int> veto_mask_veto = settings_mask_veto.at("vetoMask").getTableColumn<unsigned int>("veto");

    for(unsigned int row=0; row<algo_mask_veto.size(); row++){
      unsigned int algoBit = algo_mask_veto[row];
      unsigned int veto    = veto_mask_veto[row];
      if( algoBit < m_numberPhysTriggers ) triggerVetoMasks[algoBit] = int(veto);
    }

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Algo bx mask
    l1t::XmlConfigParser xmlReader_mask_algobx;
    l1t::TriggerSystem ts_mask_algobx;
    ts_mask_algobx.addProcessor("uGtProcessor", "uGtProcessor","-1","-1");

    // run the parser 
    xmlReader_mask_algobx.readDOMFromString( xmlPayload_mask_algobx ); // initialize it
    xmlReader_mask_algobx.readRootElement( ts_mask_algobx, "uGT" ); // extract all of the relevant context
    ts_mask_algobx.setConfigured();

    const std::map<std::string, l1t::Parameter>& settings_mask_algobx = ts_mask_algobx.getParameters("uGtProcessor");
    std::map<std::string,unsigned int> mask_algobx_columns = settings_mask_algobx.at("algorithmBxMask").getColumnIndices();
    std::vector<unsigned int> bunches = settings_mask_algobx.at("algorithmBxMask").getTableColumn<unsigned int>("bx/algo");

    unsigned int numCol_mask_algobx = mask_algobx_columns.size();

    int NumAlgoBitsInMask = numCol_mask_algobx - 1;
    for( int iBit=0; iBit<NumAlgoBitsInMask; iBit++ ){
      std::vector<unsigned int> algo = settings_mask_algobx.at("algorithmBxMask").getTableColumn<unsigned int>(std::to_string(iBit).c_str());
      for(unsigned int bx=0; bx<bunches.size(); bx++){
          if( algo[bx]!=m_bx_mask_default ) triggerAlgoBxMaskAlgoTrig[ bunches[bx] ].push_back(iBit);
      }
    }


  // Set prescales to zero if masked
  for( unsigned int iSet=0; iSet < prescales.size(); iSet++ ){
    for( unsigned int iBit=0; iBit < prescales[iSet].size(); iBit++ ){
      // Add protection in case prescale table larger than trigger mask size
      if( iBit >= triggerMasks.size() ){
            edm::LogError( "L1-O2O: L1TGlobalPrescalesVetosOnlineProd" )
	      << "\nWarning: algoBit in prescale table >= triggerMasks.size() "
	      << "\nWarning: no information on masking bit or not, setting as unmasked "
	      << std::endl;
      }
      else {
	prescales[iSet][iBit] *= triggerMasks[iBit];
      }
    }
  }

//////////////////

  l1t::PrescalesVetosHelper data_( new L1TGlobalPrescalesVetos() );

  data_.setBxMaskDefault       ( m_bx_mask_default         );
  data_.setPrescaleFactorTable ( prescales                 );
  data_.setTriggerMaskVeto     ( triggerVetoMasks          );
  data_.setTriggerAlgoBxMask   ( triggerAlgoBxMaskAlgoTrig );

  using namespace edm::es;
  std::shared_ptr<L1TGlobalPrescalesVetos> pMenu(data_.getWriteInstance());
  return pMenu;

}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TGlobalPrescalesVetosOnlineProd);
