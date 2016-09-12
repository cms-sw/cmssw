#include <iostream>
#include <fstream>
#include <stdexcept>

#include "CondTools/L1TriggerExt/interface/L1ConfigOnlineProdBaseExt.h"
#include "CondFormats/L1TObjects/interface/L1TGlobalPrescalesVetos.h"
#include "CondFormats/DataRecord/interface/L1TGlobalPrescalesVetosRcd.h"
#include "CondFormats/DataRecord/interface/L1TGlobalPrescalesVetosO2ORcd.h"
#include "L1Trigger/L1TGlobal/interface/PrescalesVetosHelper.h"
#include "L1Trigger/L1TCommon/interface/TrigSystem.h"



class L1TGlobalPrescalesVetosOnlineProd : public L1ConfigOnlineProdBaseExt<L1TGlobalPrescalesVetosO2ORcd,L1TGlobalPrescalesVetos> {
private:
public:
    virtual boost::shared_ptr<L1TGlobalPrescalesVetos> newObject(const std::string& objectKey, const L1TGlobalPrescalesVetosO2ORcd& record) override ;

    L1TGlobalPrescalesVetosOnlineProd(const edm::ParameterSet&);
    ~L1TGlobalPrescalesVetosOnlineProd(void){}
};

L1TGlobalPrescalesVetosOnlineProd::L1TGlobalPrescalesVetosOnlineProd(const edm::ParameterSet& iConfig) : L1ConfigOnlineProdBaseExt<L1TGlobalPrescalesVetosO2ORcd,L1TGlobalPrescalesVetos>(iConfig) {}

boost::shared_ptr<L1TGlobalPrescalesVetos> L1TGlobalPrescalesVetosOnlineProd::newObject(const std::string& objectKey, const L1TGlobalPrescalesVetosO2ORcd& record) {
    using namespace edm::es;

    edm::LogInfo( "L1-O2O: L1TGlobalPrescalesVetosOnlineProd" ) << "Producing L1TGlobalPrescalesVetos with RS key =" << objectKey ;

    if( objectKey.empty() ){
        edm::LogInfo( "L1-O2O: L1TGlobalPrescalesVetosOnlineProd" ) << "Key is empty";
        throw std::runtime_error("Empty objecKey");
///        return boost::shared_ptr< L1TGlobalPrescalesVetos > ( new L1TGlobalPrescalesVetos() );
    }

    std::string stage2Schema = "CMS_TRG_L1_CONF" ;

    std::vector< std::string > queryStrings ;
    queryStrings.push_back( "MP7"             ) ;
    queryStrings.push_back( "AMC502_EXTCOND"  ) ;
    queryStrings.push_back( "AMC502_FINOR"    ) ;
    queryStrings.push_back( "ALGOBX_MASK"     ) ;
    queryStrings.push_back( "ALGO_FINOR_MASK" ) ;
    queryStrings.push_back( "ALGO_FINOR_VETO" ) ;
    queryStrings.push_back( "ALGO_PRESCALE"   ) ;

    std::string prescale_key, bxmask_key, mask_key, vetomask_key;

    // select MP7, AMC502_EXTCOND, AMC502_FINOR, ALGOBX_MASK, ALGO_FINOR_MASK, ALGO_FINOR_VETO, ALGO_PRESCALE from CMS_TRG_L1_CONF.UGT_RS_KEYS where ID = objectKey ;
    l1t::OMDSReader::QueryResults queryResult =
            m_omdsReader.basicQuery( queryStrings,
                                     stage2Schema,
                                     "UGT_RS_KEYS",
                                     "UGT_RS_KEYS.ID",
                                     m_omdsReader.singleAttribute(objectKey)
                                   ) ;

    if( queryResult.queryFailed() || queryResult.numberRows() != 1 ){
        edm::LogError( "L1-O2O" ) << "Cannot get UGT_RS_KEYS.{MP7,AMC502_EXTCOND,AMC502_FINOR,ALGOBX_MASK,ALGO_FINOR_MASK,ALGO_FINOR_VETO,ALGO_PRESCALE} for ID = " << objectKey;
        throw std::runtime_error("Broken key");
///        return boost::shared_ptr< L1TGlobalPrescalesVetos > ( new L1TGlobalPrescalesVetos() );
    }

    if( !queryResult.fillVariable( "ALGO_PRESCALE",   prescale_key) ) prescale_key = "";
    if( !queryResult.fillVariable( "ALGOBX_MASK",       bxmask_key) )   bxmask_key = "";
    if( !queryResult.fillVariable( "ALGO_FINOR_MASK",     mask_key) )     mask_key = "";
    if( !queryResult.fillVariable( "ALGO_FINOR_VETO", vetomask_key) ) vetomask_key = "";

    queryStrings.clear();
    queryStrings.push_back( "CONF" ) ;


    std::string xmlPayload_prescale, xmlPayload_mask_algobx, xmlPayload_mask_finor, xmlPayload_mask_veto;

    queryResult =
            m_omdsReader.basicQuery( queryStrings,
                                     stage2Schema,
                                     "UGT_RS",
                                     "UGT_RS.ID",
                                     m_omdsReader.singleAttribute(prescale_key)
                                   ) ;

    if( queryResult.queryFailed() || queryResult.numberRows() != 1 ){
        edm::LogError( "L1-O2O: L1TGlobalPrescalesVetosOnlineProd" ) << "Cannot get UGT_RS.CONF for prescale_key = "<<prescale_key ;
        throw std::runtime_error("Broken key");
///        return boost::shared_ptr< L1TGlobalPrescalesVetos >() ;
    }

    if( !queryResult.fillVariable( "CONF", xmlPayload_prescale ) ) xmlPayload_prescale = "";


    queryResult =
            m_omdsReader.basicQuery( queryStrings,
                                     stage2Schema,
                                     "UGT_RS",
                                     "UGT_RS.ID",
                                     m_omdsReader.singleAttribute(mask_key)
                                   ) ;

    if( queryResult.queryFailed() || queryResult.numberRows() != 1 ){
        edm::LogError( "L1-O2O: L1TGlobalPrescalesVetosOnlineProd" ) << "Cannot get UGT_RS.CONF for mask_key = "<<mask_key ;
        throw std::runtime_error("Broken key");
///        return boost::shared_ptr< L1TGlobalPrescalesVetos >() ;
    }

    if( !queryResult.fillVariable( "CONF", xmlPayload_mask_finor ) ) xmlPayload_mask_finor = "";


    queryResult =
            m_omdsReader.basicQuery( queryStrings,
                                     stage2Schema,
                                     "UGT_RS",
                                     "UGT_RS.ID",
                                     m_omdsReader.singleAttribute(bxmask_key)
                                   ) ;

    if( queryResult.queryFailed() || queryResult.numberRows() != 1 ){
        edm::LogError( "L1-O2O: L1TGlobalPrescalesVetosOnlineProd" ) << "Cannot get UGT_RS.CONF for bxmask_key = "<<bxmask_key ;
        throw std::runtime_error("Broken key");
///        return boost::shared_ptr< L1TGlobalPrescalesVetos >() ;
    }

    if( !queryResult.fillVariable( "CONF", xmlPayload_mask_algobx ) ) xmlPayload_mask_algobx = "";


    queryResult =
            m_omdsReader.basicQuery( queryStrings,
                                     stage2Schema,
                                     "UGT_RS",
                                     "UGT_RS.ID",
                                     m_omdsReader.singleAttribute(vetomask_key)
                                   ) ;

    if( queryResult.queryFailed() || queryResult.numberRows() != 1 ){
        edm::LogError( "L1-O2O: L1TGlobalPrescalesVetosOnlineProd" ) << "Cannot get UGT_RS.CONF for vetomask_key = "<<vetomask_key ;
        throw std::runtime_error("Broken key");
///        return boost::shared_ptr< L1TGlobalPrescalesVetos >() ;
    }

    if( !queryResult.fillVariable( "CONF", xmlPayload_mask_veto ) ) xmlPayload_mask_veto = "";

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
    l1t::XmlConfigReader xmlReader_prescale;
    l1t::TrigSystem ts_prescale;
    ts_prescale.addProcRole("uGtProcessor", "uGtProcessor");

    // run the parser 
    xmlReader_prescale.readDOMFromString( xmlPayload_prescale ); // initialize it
    xmlReader_prescale.readRootElement( ts_prescale, "uGT" ); // extract all of the relevant context
    ts_prescale.setConfigured();

    std::map<std::string, l1t::Setting> settings_prescale = ts_prescale.getSettings("uGtProcessor");
    std::vector<l1t::TableRow> tRow_prescale = settings_prescale["prescales"].getTableRows();

    unsigned int numColumns_prescale = 0;
    if( tRow_prescale.size()>0 ){
      std::vector<std::string> firstRow_prescale = tRow_prescale[0].getRow();
      numColumns_prescale = firstRow_prescale.size();
    }
    
    int NumPrescaleSets = numColumns_prescale - 1;
///  there may be "missing" rows/bits in the xml description meaning that triggers are not unused
///    int NumAlgos_prescale = tRow_prescale.size();
    unsigned int NumAlgos_prescale = 0;
    for( auto it=tRow_prescale.begin(); it!=tRow_prescale.end(); it++ ){
        unsigned int algoBit = it->getRowValue<unsigned int>("algo/prescale-index");
        if( NumAlgos_prescale < algoBit+1 ) NumAlgos_prescale = algoBit+1;
    }

    if( NumPrescaleSets > 0 ){
      // Fill default prescale set
      for( int iSet=0; iSet<NumPrescaleSets; iSet++ ){
	prescales.push_back(std::vector<int>());
	for( unsigned int iBit = 0; iBit < NumAlgos_prescale; ++iBit ){
	  int inputDefaultPrescale = 0; // only prescales that are set in the block below are used
	  prescales[iSet].push_back(inputDefaultPrescale);
	}
      }

      for( auto it=tRow_prescale.begin(); it!=tRow_prescale.end(); it++ ){
	unsigned int algoBit = it->getRowValue<unsigned int>("algo/prescale-index");
	for( int iSet=0; iSet<NumPrescaleSets; iSet++ ){
            int prescale = 0;
            try{
                prescale = it->getRowValue<unsigned int>(std::to_string(iSet));
            } catch (std::runtime_error &e){
                edm::LogError( "L1-O2O: L1TGlobalPrescalesVetosOnlineProd" )
                    << "\nWarning: missing value for algoBit " << algoBit << " (row)"
                    << " in prescale set " << iSet << " (column) of " << prescale_key
                    << " also stored in /tmp/" << prescale_key.substr(0,prescale_key.find("/")).append(".xml")
                    << "\nWarning: no information on algoBit, setting to 0 "
                    << std::endl;
            }
	  prescales[iSet][algoBit] = prescale;
	}
      }
    }


  ///////////////////////////////////////////////////////////////////////////////////////////////////////////

  // finor mask
  // Setting mask to default 1 (unmask)
  for( unsigned int iAlg=0; iAlg < m_numberPhysTriggers; iAlg++ )
    triggerMasks.push_back(1);

    l1t::XmlConfigReader xmlReader_mask_finor;
    l1t::TrigSystem ts_mask_finor;
    ts_mask_finor.addProcRole("uGtProcessor", "uGtProcessor");

    // run the parser 
    xmlReader_mask_finor.readDOMFromString( xmlPayload_mask_finor ); // initialize it
    xmlReader_mask_finor.readRootElement( ts_mask_finor, "uGT" ); // extract all of the relevant context
    ts_mask_finor.setConfigured();

    std::map<std::string, l1t::Setting> settings_mask_finor = ts_mask_finor.getSettings("uGtProcessor");
    std::vector<l1t::TableRow> tRow_mask_finor = settings_mask_finor["finorMask"].getTableRows();

    for( auto it=tRow_mask_finor.begin(); it!=tRow_mask_finor.end(); it++ ){
      unsigned int algoBit = it->getRowValue<unsigned int>("algo");
      unsigned int mask = it->getRowValue<unsigned int>("mask");
      if( algoBit < m_numberPhysTriggers ) triggerMasks[algoBit] = mask;
    }

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////

  // veto mask
  // Setting veto mask to default 0 (no veto)
  for( unsigned int iAlg=0; iAlg < m_numberPhysTriggers; iAlg++ )
    triggerVetoMasks.push_back(0);
  
    l1t::XmlConfigReader xmlReader_mask_veto;
    l1t::TrigSystem ts_mask_veto;
    ts_mask_veto.addProcRole("uGtProcessor", "uGtProcessor");

    // run the parser 
    xmlReader_mask_veto.readDOMFromString( xmlPayload_mask_veto ); // initialize it
    xmlReader_mask_veto.readRootElement( ts_mask_veto, "uGT" ); // extract all of the relevant context
    ts_mask_veto.setConfigured();

    std::map<std::string, l1t::Setting> settings_mask_veto = ts_mask_veto.getSettings("uGtProcessor");
    std::vector<l1t::TableRow> tRow_mask_veto = settings_mask_veto["vetoMask"].getTableRows();

    for( auto it=tRow_mask_veto.begin(); it!=tRow_mask_veto.end(); it++ ){
      unsigned int algoBit = it->getRowValue<unsigned int>("algo");
      unsigned int veto = it->getRowValue<unsigned int>("veto");
      if( algoBit < m_numberPhysTriggers ) triggerVetoMasks[algoBit] = int(veto);
    }

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Algo bx mask
    l1t::XmlConfigReader xmlReader_mask_algobx;
    l1t::TrigSystem ts_mask_algobx;
    ts_mask_algobx.addProcRole("uGtProcessor", "uGtProcessor");

    // run the parser 
    xmlReader_mask_algobx.readDOMFromString( xmlPayload_mask_algobx ); // initialize it
    xmlReader_mask_algobx.readRootElement( ts_mask_algobx, "uGT" ); // extract all of the relevant context
    ts_mask_algobx.setConfigured();

    std::map<std::string, l1t::Setting> settings_mask_algobx = ts_mask_algobx.getSettings("uGtProcessor");
    std::vector<l1t::TableRow> tRow_mask_algobx = settings_mask_algobx["algorithmBxMask"].getTableRows();

    unsigned int numCol_mask_algobx = 0;
    if( tRow_mask_algobx.size()>0 ){
      std::vector<std::string> firstRow_mask_algobx = tRow_mask_algobx[0].getRow();
      numCol_mask_algobx = firstRow_mask_algobx.size();
    }
    
    int NumAlgoBitsInMask = numCol_mask_algobx - 1;
    if( NumAlgoBitsInMask > 0 ){
      for( auto it=tRow_mask_algobx.begin(); it!=tRow_mask_algobx.end(); it++ ){
	int bx = it->getRowValue<unsigned int>("bx/algo");
	std::vector<int> maskedBits;
	for( int iBit=0; iBit<NumAlgoBitsInMask; iBit++ ){
	  unsigned int maskBit = it->getRowValue<unsigned int>(std::to_string(iBit));
	  if( maskBit!=m_bx_mask_default ) maskedBits.push_back(iBit);
	}
	if( maskedBits.size()>0 ) triggerAlgoBxMaskAlgoTrig[bx] = maskedBits;
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
  boost::shared_ptr<L1TGlobalPrescalesVetos> pMenu = boost::shared_ptr< L1TGlobalPrescalesVetos >(data_.getWriteInstance());
  return pMenu;

}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TGlobalPrescalesVetosOnlineProd);
