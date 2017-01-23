#include <iostream>
#include <fstream>
#include <stdexcept>
#include <sstream>

#include "CondTools/L1TriggerExt/interface/L1ConfigOnlineProdBaseExt.h"
#include "CondFormats/L1TObjects/interface/CaloParams.h"
#include "CondFormats/DataRecord/interface/L1TCaloStage2ParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1TCaloParamsO2ORcd.h"
#include "L1Trigger/L1TCommon/interface/TriggerSystem.h"
#include "L1Trigger/L1TCommon/interface/XmlConfigParser.h"
#include "L1Trigger/L1TCommon/interface/ConvertToLUT.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"

#include "xercesc/util/PlatformUtils.hpp"
using namespace XERCES_CPP_NAMESPACE;

class L1TCaloParamsOnlineProd : public L1ConfigOnlineProdBaseExt<L1TCaloParamsO2ORcd,l1t::CaloParams> {
private:
public:
    virtual std::shared_ptr<l1t::CaloParams> newObject(const std::string& objectKey, const L1TCaloParamsO2ORcd& record) override ;

    L1TCaloParamsOnlineProd(const edm::ParameterSet&);
    ~L1TCaloParamsOnlineProd(void){}
};

bool
readCaloLayer1OnlineSettings(l1t::CaloParamsHelper& paramsHelper, std::map<std::string, l1t::Parameter>& conf, std::map<std::string, l1t::Mask>& ) {
  const char * expectedParams[] = {
    "layer1ECalScaleFactors",
    "layer1HCalScaleFactors",
    "layer1HFScaleFactors",
    "layer1ECalScaleETBins",
    "layer1HCalScaleETBins",
    "layer1HFScaleETBins"
  };
  for (const auto param : expectedParams) {
    if ( conf.find(param) == conf.end() ) {
      std::cerr << "Unable to locate expected CaloLayer1 parameter: " << param << " in L1 settings payload!";
      return false;
    }
  }
  // Layer 1 LUT specification
  paramsHelper.setLayer1ECalScaleFactors((conf["layer1ECalScaleFactors"].getVector<double>()));
  paramsHelper.setLayer1HCalScaleFactors((conf["layer1HCalScaleFactors"].getVector<double>()));
  paramsHelper.setLayer1HFScaleFactors  ((conf["layer1HFScaleFactors"]  .getVector<double>()));
  paramsHelper.setLayer1ECalScaleETBins(conf["layer1ECalScaleETBins"].getVector<int>());
  paramsHelper.setLayer1HCalScaleETBins(conf["layer1HCalScaleETBins"].getVector<int>());
  paramsHelper.setLayer1HFScaleETBins  (conf["layer1HFScaleETBins"]  .getVector<int>());

  return true;
}

bool
readCaloLayer2OnlineSettings(l1t::CaloParamsHelper& paramsHelper, std::map<std::string, l1t::Parameter>& conf, std::map<std::string, l1t::Mask>& ) {
  const char * expectedParams[] = {
    "leptonSeedThreshold",
    "leptonTowerThreshold",
    "pileUpTowerThreshold",
    "jetSeedThreshold",
    "jetMaxEta",
    "HTMHT_maxJetEta",
    "HT_jetThreshold",
    "MHT_jetThreshold",
    "jetEnergyCalibLUT",
    "ETMET_maxTowerEta",
    "ET_energyCalibLUT",
    "ecalET_energyCalibLUT",
    "METX_energyCalibLUT",
    "METY_energyCalibLUT",
    "egammaRelaxationThreshold",
    "egammaMaxEta",
    "egammaEnergyCalibLUT",
    "egammaIsoLUT",
    "tauMaxEta",
    "tauEnergyCalibLUT",
    "tauIsoLUT1",
    "tauIsoLUT2",
    "towerCountThreshold",
    "towerCountMaxEta",
    "ET_towerThreshold",
    "MET_towerThreshold",
    "jetBypassPileUpSub",
    "egammaBypassCuts",
    "egammaHOverECut_iEtaLT15",
    "egammaHOverECut_iEtaGTEq15"
  };
  for (const auto param : expectedParams) {
    if ( conf.find(param) == conf.end() ) {
      std::cerr << "Unable to locate expected CaloLayer2 parameter: " << param << " in L1 settings payload!";
      return false;
    }
  }
  // Layer 2 params specification
  paramsHelper.setEgSeedThreshold((conf["leptonSeedThreshold"].getValue<int>())/2);
  paramsHelper.setTauSeedThreshold((conf["leptonSeedThreshold"].getValue<int>())/2);
  paramsHelper.setEgNeighbourThreshold((conf["leptonTowerThreshold"].getValue<int>())/2);
  paramsHelper.setTauNeighbourThreshold((conf["leptonTowerThreshold"].getValue<int>())/2);
  paramsHelper.setJetSeedThreshold((conf["jetSeedThreshold"].getValue<int>())/2);
  paramsHelper.setJetBypassPUS(conf["jetBypassPileUpSub"].getValue<bool>());
  paramsHelper.setEgBypassEGVetos(conf["egammaBypassCuts"].getValue<bool>());
  paramsHelper.setEgHOverEcutBarrel(conf["egammaHOverECut_iEtaLT15"].getValue<int>());
  paramsHelper.setEgHOverEcutEndcap(conf["egammaHOverECut_iEtaGTEq15"].getValue<int>());


  // Currently not used // paramsHelper.setEgPileupTowerThresh((conf["pileUpTowerThreshold"].getValue<int>())); 
  // Currently not used // paramsHelper.setTauPileupTowerThresh((conf["pileUpTowerThreshold"].getValue<int>())); 
  // Currently not used // paramsHelper.setJetMaxEta((conf["jetMaxEta"].getValue<int>()));
  
  std::vector<int> etSumEtaMax;
  std::vector<int> etSumEtThresh;
  
  etSumEtaMax.push_back(conf["ETMET_maxTowerEta"].getValue<int>());
  etSumEtaMax.push_back(conf["HTMHT_maxJetEta"].getValue<int>());
  etSumEtaMax.push_back(conf["ETMET_maxTowerEta"].getValue<int>());
  etSumEtaMax.push_back(conf["HTMHT_maxJetEta"].getValue<int>());
  etSumEtaMax.push_back(conf["towerCountMaxEta"].getValue<int>());
  
  etSumEtThresh.push_back(conf["ET_towerThreshold"].getValue<int>()/2); // ETT tower threshold
  etSumEtThresh.push_back(conf["HT_jetThreshold"].getValue<int>()/2);
  etSumEtThresh.push_back(conf["MET_towerThreshold"].getValue<int>()/2); // ETM tower threshold
  etSumEtThresh.push_back(conf["MHT_jetThreshold"].getValue<int>()/2);
  etSumEtThresh.push_back(conf["ET_towerThreshold"].getValue<int>()/2);

  for (uint i=0; i<5; ++i) {
    paramsHelper.setEtSumEtaMax(i, etSumEtaMax.at(i));
    paramsHelper.setEtSumEtThreshold(i, etSumEtThresh.at(i));
  }

  paramsHelper.setJetCalibrationLUT ( l1t::convertToLUT( conf["jetEnergyCalibLUT"].getVector<uint32_t>() ) );
  paramsHelper.setEtSumEttPUSLUT    ( l1t::convertToLUT( conf["ET_energyCalibLUT"].getVector<int>() ) );
  paramsHelper.setEtSumEcalSumPUSLUT( l1t::convertToLUT( conf["ecalET_energyCalibLUT"].getVector<int>() ) );
  paramsHelper.setEtSumXPUSLUT      ( l1t::convertToLUT( conf["METX_energyCalibLUT"].getVector<int>() ) );
  paramsHelper.setEgMaxPtHOverE((conf["egammaRelaxationThreshold"].getValue<int>())/2.);
  paramsHelper.setEgEtaCut((conf["egammaMaxEta"].getValue<int>()));
  paramsHelper.setEgCalibrationLUT  ( l1t::convertToLUT( conf["egammaEnergyCalibLUT"].getVector<int>() ) );
  paramsHelper.setEgIsolationLUT    ( l1t::convertToLUT( conf["egammaIsoLUT"].getVector<int>() ) );

  paramsHelper.setIsoTauEtaMax((conf["tauMaxEta"].getValue<int>()));

  paramsHelper.setTauCalibrationLUT( l1t::convertToLUT( conf["tauEnergyCalibLUT"].getVector<int>() ) );
  paramsHelper.setTauIsolationLUT  ( l1t::convertToLUT( conf["tauIsoLUT1"].getVector<int>() ) );
  paramsHelper.setTauIsolationLUT2 ( l1t::convertToLUT( conf["tauIsoLUT2"].getVector<int>() ) );

  return true;
}

L1TCaloParamsOnlineProd::L1TCaloParamsOnlineProd(const edm::ParameterSet& iConfig) : L1ConfigOnlineProdBaseExt<L1TCaloParamsO2ORcd,l1t::CaloParams>(iConfig) {}

std::shared_ptr<l1t::CaloParams> L1TCaloParamsOnlineProd::newObject(const std::string& objectKey, const L1TCaloParamsO2ORcd& record) {
    using namespace edm::es;

    const L1TCaloStage2ParamsRcd& baseRcd = record.template getRecord< L1TCaloStage2ParamsRcd >() ;
    edm::ESHandle< l1t::CaloParams > baseSettings ;
    baseRcd.get( baseSettings ) ;


    if( objectKey.empty() ){
        edm::LogError( "L1-O2O: L1TCaloParamsOnlineProd" ) << "Key is empty, returning empty l1t::CaloParams";
        throw std::runtime_error("Empty objectKey");
    }

    std::string tscKey = objectKey.substr(0, objectKey.find(":") );
    std::string  rsKey = objectKey.substr(   objectKey.find(":")+1, std::string::npos );


    std::string stage2Schema = "CMS_TRG_L1_CONF" ;
    edm::LogInfo( "L1-O2O: L1TCaloParamsOnlineProd" ) << "Producing L1TCaloParamsOnlineProd with TSC key =" << tscKey << " and RS key = " << rsKey ;

    // first, find keys for the algo and RS tables

    // ALGO
    std::vector< std::string > queryStrings ;
    queryStrings.push_back( "CALOL1_KEY" ) ;
    queryStrings.push_back( "CALOL2_KEY" ) ;

    std::string calol1_conf_key, calol2_conf_key;

    // select CALOL1_KEY,CALOL2_KEY from CMS_TRG_L1_CONF.L1_TRG_CONF_KEYS where ID = tscKey ;
    l1t::OMDSReader::QueryResults queryResult =
            m_omdsReader.basicQuery( queryStrings,
                                     stage2Schema,
                                     "L1_TRG_CONF_KEYS",
                                     "L1_TRG_CONF_KEYS.ID",
                                     m_omdsReader.singleAttribute(tscKey)
                                   ) ;

    if( queryResult.queryFailed() || queryResult.numberRows() != 1 ){
        edm::LogError( "L1-O2O" ) << "Cannot get L1_TRG_CONF_KEYS.{CALOL1_KEY,CALOL2_KEY} for ID = " << tscKey ;
        throw std::runtime_error("Broken tscKey");
    }

    if( !queryResult.fillVariable( "CALOL1_KEY", calol1_conf_key) ) calol1_conf_key = "";
    if( !queryResult.fillVariable( "CALOL2_KEY", calol2_conf_key) ) calol2_conf_key = "";

    // At this point we have two config keys; now query the payloads' keys for these config keys
    std::string calol1_algo_key, calol2_algo_key;
    queryStrings.clear();
    queryStrings.push_back( "ALGO" );

    // query ALGO configurations
    queryResult =
            m_omdsReader.basicQuery( queryStrings,
                                     stage2Schema,
                                     "CALOL1_KEYS",
                                     "CALOL1_KEYS.ID",
                                     m_omdsReader.singleAttribute(calol1_conf_key)
                                   ) ;

    if( queryResult.queryFailed() || queryResult.numberRows() != 1 ){
        edm::LogError( "L1-O2O: L1TCaloParamsOnlineProd" ) << "Cannot get CALOL1_KEYS.ALGO for ID="<<calol1_conf_key;
        throw std::runtime_error("Broken key");
    }

    if( !queryResult.fillVariable( "ALGO", calol1_algo_key ) ) calol1_algo_key = "";

    queryStrings.push_back( "HW" );
//  No Layer2 for the moment
    queryResult =
            m_omdsReader.basicQuery( queryStrings,
                                     stage2Schema,
                                     "CALOL2_KEYS",
                                     "CALOL2_KEYS.ID",
                                     m_omdsReader.singleAttribute(calol2_conf_key)
                                   ) ;

    if( queryResult.queryFailed() || queryResult.numberRows() != 1 ){
        edm::LogError( "L1-O2O: L1TCaloParamsOnlineProd" ) << "Cannot get CALOL2_KEYS.ALGO for ID="<<calol2_conf_key;
        throw std::runtime_error("Broken key");
    }

    std::string calol2_hw_key;
    if( !queryResult.fillVariable( "ALGO", calol2_algo_key ) ) calol2_algo_key = "";
    if( !queryResult.fillVariable( "HW",   calol2_hw_key   ) ) calol2_hw_key   = "";


    // Now querry the actual payloads
    enum {kCONF=0, kRS, kHW, NUM_TYPES};
    std::map<std::string,std::string> payloads[NUM_TYPES];  // associates key -> XML payload for a given type of payloads
    std::string xmlPayload;

    queryStrings.clear();
    queryStrings.push_back( "CONF" );

    // query CALOL1 ALGO configuration
    queryResult =
            m_omdsReader.basicQuery( queryStrings,
                                     stage2Schema,
                                     "CALOL1_ALGO",
                                     "CALOL1_ALGO.ID",
                                     m_omdsReader.singleAttribute(calol1_algo_key)
                                   ) ;

    if( queryResult.queryFailed() || queryResult.numberRows() != 1 ){
        edm::LogError( "L1-O2O: L1TCaloParamsOnlineProd" ) << "Cannot get CALOL1_ALGO.CONF for ID="<<calol1_algo_key;
        throw std::runtime_error("Broken key");
    }

    if( !queryResult.fillVariable( "CONF", xmlPayload ) ) xmlPayload = "";
    // remember ALGO configuration
    payloads[kCONF][calol1_algo_key.append("_L1")] = xmlPayload;

    std::string calol2_demux_key, calol2_mps_common_key, calol2_mps_jet_key, calol2_mp_egamma_key, calol2_mp_sum_key, calol2_mp_tau_key;

    queryStrings.clear();
    queryStrings.push_back( "DEMUX"      );
    queryStrings.push_back( "MPS_COMMON" );
    queryStrings.push_back( "MPS_JET"    );
    queryStrings.push_back( "MP_EGAMMA"  );
    queryStrings.push_back( "MP_SUM"     );
    queryStrings.push_back( "MP_TAU"     );

    // No CALOL2 ALGO configuration for the moment
    queryResult =
            m_omdsReader.basicQuery( queryStrings,
                                     stage2Schema,
                                     "CALOL2_ALGO_KEYS",
                                     "CALOL2_ALGO_KEYS.ID",
                                     m_omdsReader.singleAttribute(calol2_algo_key)
                                   ) ;

    if( queryResult.queryFailed() || queryResult.numberRows() != 1 ){
        edm::LogError( "L1-O2O: L1TCaloParamsOnlineProd" ) << "Cannot get CALOL2_ALGO_KEYS.{DEMUX,MPS_COMMON,MPS_JET,MP_EGAMMA,MP_SUM,MP_TAU} for ID="<<calol2_algo_key;
        throw std::runtime_error("Broken key");
    }

    if( !queryResult.fillVariable( "DEMUX",      calol2_demux_key      ) ) calol2_demux_key      = "";
    if( !queryResult.fillVariable( "MPS_COMMON", calol2_mps_common_key ) ) calol2_mps_common_key = "";
    if( !queryResult.fillVariable( "MPS_JET",    calol2_mps_jet_key    ) ) calol2_mps_jet_key    = "";
    if( !queryResult.fillVariable( "MP_EGAMMA",  calol2_mp_egamma_key  ) ) calol2_mp_egamma_key  = "";
    if( !queryResult.fillVariable( "MP_SUM",     calol2_mp_sum_key     ) ) calol2_mp_sum_key     = "";
    if( !queryResult.fillVariable( "MP_TAU",     calol2_mp_tau_key     ) ) calol2_mp_tau_key     = "";

    queryStrings.clear();
    queryStrings.push_back( "CONF" );

    // query CALOL2 ALGO configuration
    queryResult =
            m_omdsReader.basicQuery( queryStrings,
                                     stage2Schema,
                                     "CALOL2_ALGO",
                                     "CALOL2_ALGO.ID",
                                     m_omdsReader.singleAttribute(calol2_demux_key)
                                   ) ;

    if( queryResult.queryFailed() || queryResult.numberRows() != 1 ){
        edm::LogError( "L1-O2O: L1TCaloParamsOnlineProd" ) << "Cannot get CALOL2_ALGO.CONF for ID="<<calol2_demux_key;
        throw std::runtime_error("Broken key");
    }

    if( !queryResult.fillVariable( "CONF", xmlPayload ) ) xmlPayload = "";
    // remember ALGO configuration
    payloads[kCONF][calol2_demux_key.append("_L2")] = xmlPayload;

    // query CALOL2 ALGO configuration
    queryResult =
            m_omdsReader.basicQuery( queryStrings,
                                     stage2Schema,
                                     "CALOL2_ALGO",
                                     "CALOL2_ALGO.ID",
                                     m_omdsReader.singleAttribute(calol2_mps_common_key)
                                   ) ;

    if( queryResult.queryFailed() || queryResult.numberRows() != 1 ){
        edm::LogError( "L1-O2O: L1TCaloParamsOnlineProd" ) << "Cannot get CALOL2_ALGO.CONF for ID="<<calol2_mps_common_key;
        throw std::runtime_error("Broken key");
    }

    if( !queryResult.fillVariable( "CONF", xmlPayload ) ) xmlPayload = "";
    // remember ALGO configuration
    payloads[kCONF][calol2_mps_common_key.append("_L2")] = xmlPayload;

    // query CALOL2 ALGO configuration
    queryResult =
            m_omdsReader.basicQuery( queryStrings,
                                     stage2Schema,
                                     "CALOL2_ALGO",
                                     "CALOL2_ALGO.ID",
                                     m_omdsReader.singleAttribute(calol2_mps_jet_key)
                                   ) ;

    if( queryResult.queryFailed() || queryResult.numberRows() != 1 ){
        edm::LogError( "L1-O2O: L1TCaloParamsOnlineProd" ) << "Cannot get CALOL2_ALGO.CONF for ID="<<calol2_mps_jet_key;
        throw std::runtime_error("Broken key");
    }

    if( !queryResult.fillVariable( "CONF", xmlPayload ) ) xmlPayload = "";
    // remember ALGO configuration
    payloads[kCONF][calol2_mps_jet_key.append("_L2")] = xmlPayload;

    // query CALOL2 ALGO configuration
    queryResult =
            m_omdsReader.basicQuery( queryStrings,
                                     stage2Schema,
                                     "CALOL2_ALGO",
                                     "CALOL2_ALGO.ID",
                                     m_omdsReader.singleAttribute(calol2_mp_egamma_key)
                                   ) ;

    if( queryResult.queryFailed() || queryResult.numberRows() != 1 ){
        edm::LogError( "L1-O2O: L1TCaloParamsOnlineProd" ) << "Cannot get CALOL2_ALGO.CONF for ID="<<calol2_mp_egamma_key;
        throw std::runtime_error("Broken key");
    }

    if( !queryResult.fillVariable( "CONF", xmlPayload ) ) xmlPayload = "";
    // remember ALGO configuration
    payloads[kCONF][calol2_mp_egamma_key.append("_L2")] = xmlPayload;

    // query CALOL2 ALGO configuration
    queryResult =
            m_omdsReader.basicQuery( queryStrings,
                                     stage2Schema,
                                     "CALOL2_ALGO",
                                     "CALOL2_ALGO.ID",
                                     m_omdsReader.singleAttribute(calol2_mp_sum_key)
                                   ) ;

    if( queryResult.queryFailed() || queryResult.numberRows() != 1 ){
        edm::LogError( "L1-O2O: L1TCaloParamsOnlineProd" ) << "Cannot get CALOL2_ALGO.CONF for ID="<<calol2_mp_sum_key;
        throw std::runtime_error("Broken key");
    }

    if( !queryResult.fillVariable( "CONF", xmlPayload ) ) xmlPayload = "";
    // remember ALGO configuration
    payloads[kCONF][calol2_mp_sum_key.append("_L2")] = xmlPayload;

    // query CALOL2 ALGO configuration
    queryResult =
            m_omdsReader.basicQuery( queryStrings,
                                     stage2Schema,
                                     "CALOL2_ALGO",
                                     "CALOL2_ALGO.ID",
                                     m_omdsReader.singleAttribute(calol2_mp_tau_key)
                                   ) ;

    if( queryResult.queryFailed() || queryResult.numberRows() != 1 ){
        edm::LogError( "L1-O2O: L1TCaloParamsOnlineProd" ) << "Cannot get CALOL2_ALGO.CONF for ID="<<calol2_mp_tau_key;
        throw std::runtime_error("Broken key");
    }

    if( !queryResult.fillVariable( "CONF", xmlPayload ) ) xmlPayload = "";
    // remember ALGO configuration
    payloads[kCONF][calol2_mp_tau_key.append("_L2")] = xmlPayload;

    queryResult =
            m_omdsReader.basicQuery( queryStrings,
                                     stage2Schema,
                                     "CALOL2_HW",
                                     "CALOL2_HW.ID",
                                     m_omdsReader.singleAttribute(calol2_hw_key)
                                   ) ;

    if( queryResult.queryFailed() || queryResult.numberRows() != 1 ){
        edm::LogError( "L1-O2O: L1TCaloParamsOnlineProd" ) << "Cannot get CALOL2_HW.CONF for ID="<<calol2_hw_key;
        throw std::runtime_error("Broken key");
    }

    if( !queryResult.fillVariable( "CONF", xmlPayload ) ) xmlPayload = "";
    // remember HW configuration
    payloads[kHW][calol2_hw_key] = xmlPayload;

    // for debugging purposes dump the configs to local files
    for(auto &conf : payloads[kCONF]){ 
        std::ofstream output(std::string("/tmp/").append(conf.first.substr(0,conf.first.find("/"))).append(".xml"));
        output<<conf.second;
        output.close();
    }
    for(auto &conf : payloads[kHW]){ 
        std::ofstream output(std::string("/tmp/").append(conf.first.substr(0,conf.first.find("/"))).append(".xml"));
        output<<conf.second;
        output.close();
    }


    l1t::XmlConfigParser xmlReader1;
    xmlReader1.readDOMFromString( payloads[kCONF][calol1_algo_key] );

    l1t::TriggerSystem calol1;
    calol1.addProcessor("processors", "processors","-1","-1");
    xmlReader1.readRootElement( calol1, "calol1" );
    calol1.setConfigured();

    std::map<std::string, l1t::Parameter> calol1_conf = calol1.getParameters("processors");
    std::map<std::string, l1t::Mask>      calol1_rs   ;//= calol1.getMasks   ("processors");

    l1t::TriggerSystem calol2;
//    calol2.addProcRole("processors", "MainProcessor");

    l1t::XmlConfigParser xmlReader2;
    xmlReader2.readDOMFromString( payloads[kHW].begin()->second );
    xmlReader2.readRootElement( calol2, "calol2" );

    for(auto &conf : payloads[kCONF]){ 
        if( conf.first.find("_L2") != std::string::npos ){
            xmlReader2.readDOMFromString( conf.second );
            xmlReader2.readRootElement( calol2, "calol2" );
        }
    }

//    calol2.setSystemId("calol2");
    calol2.setConfigured();

    // Perhaps layer 2 has to look at settings for demux and mp separately? // => No demux settings required
    std::map<std::string, l1t::Parameter> calol2_conf = calol2.getParameters("MP1");
    std::map<std::string, l1t::Mask>      calol2_rs   ;//= calol2.getMasks   ("processors");
    
    l1t::CaloParamsHelper m_params_helper(*(baseSettings.product()));

    readCaloLayer1OnlineSettings(m_params_helper, calol1_conf, calol1_rs);
    readCaloLayer2OnlineSettings(m_params_helper, calol2_conf, calol2_rs);

    std::shared_ptr< l1t::CaloParams > retval = std::make_shared< l1t::CaloParams >( m_params_helper ) ;
    return retval;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TCaloParamsOnlineProd);

