#include <iostream>
#include <fstream>
#include <stdexcept>
#include <sstream>

#include "CondTools/L1TriggerExt/interface/L1ConfigOnlineProdBaseExt.h"
#include "CondFormats/L1TObjects/interface/CaloParams.h"
#include "CondFormats/DataRecord/interface/L1TCaloStage2ParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1TCaloParamsO2ORcd.h"
#include "L1Trigger/L1TCommon/interface/TrigSystem.h"
#include "L1Trigger/L1TCommon/interface/XmlConfigReader.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"

#include "xercesc/util/PlatformUtils.hpp"
using namespace XERCES_CPP_NAMESPACE;

class L1TCaloParamsOnlineProd : public L1ConfigOnlineProdBaseExt<L1TCaloParamsO2ORcd,l1t::CaloParams> {
private:
public:
    virtual boost::shared_ptr<l1t::CaloParams> newObject(const std::string& objectKey, const L1TCaloParamsO2ORcd& record) override ;

    L1TCaloParamsOnlineProd(const edm::ParameterSet&);
    ~L1TCaloParamsOnlineProd(void){}
};

bool
readCaloLayer1OnlineSettings(l1t::CaloParamsHelper& paramsHelper, std::map<std::string, l1t::Setting>& conf, std::map<std::string, l1t::Mask>& ) {
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
readCaloLayer2OnlineSettings(l1t::CaloParamsHelper& paramsHelper, std::map<std::string, l1t::Setting>& conf, std::map<std::string, l1t::Mask>& ) {
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
  paramsHelper.setJetBypassPUS(conf["jetBypassPileUpSub"].getValue<unsigned>()); //these are bools in onlineDB
  paramsHelper.setEgBypassEGVetos(conf["egammaBypassCuts"].getValue<unsigned>()); //these are bools in onlineDB
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
  
  etSumEtThresh.push_back(conf["ET_towerThreshold"].getValue<int>()); // ETT tower threshold
  etSumEtThresh.push_back(conf["HT_jetThreshold"].getValue<int>());
  etSumEtThresh.push_back(conf["MET_towerThreshold"].getValue<int>()); // ETM tower threshold
  etSumEtThresh.push_back(conf["MHT_jetThreshold"].getValue<int>());
  etSumEtThresh.push_back(conf["ET_towerThreshold"].getValue<int>());

  for (uint i=0; i<5; ++i) {
    paramsHelper.setEtSumEtaMax(i, etSumEtaMax.at(i));
    paramsHelper.setEtSumEtThreshold(i, etSumEtThresh.at(i));
  }

  std::stringstream oss;

  std::vector<uint32_t> jetEnergyCalibLUT = conf["jetEnergyCalibLUT"].getVector<uint32_t>();
  oss <<"#<header> V1 "<< ( 32 - __builtin_clz( uint32_t(jetEnergyCalibLUT.size()-1) ) ) <<" 63 </header> "<<std::endl; // hardcode max bits for data
  for(unsigned int i=0; i<jetEnergyCalibLUT.size(); i++) oss << i << " " << jetEnergyCalibLUT[i] << std::endl;

  std::istringstream iss1( oss.str() );
  paramsHelper.setJetCalibrationLUT( l1t::LUT( (std::istream&)iss1 ) );
  oss.str("");

  std::vector<int> etSumEttPUSLUT = conf["ET_energyCalibLUT"].getVector<int>();
  oss <<"#<header> V1 "<< ( 32 - __builtin_clz( uint32_t(etSumEttPUSLUT.size()-1) ) ) <<" 63 </header> "<<std::endl; // hardcode max bits for data
  for(unsigned int i=0; i<etSumEttPUSLUT.size(); i++) oss << i << " " << etSumEttPUSLUT[i] << std::endl;

  std::istringstream iss2( oss.str() );
  paramsHelper.setEtSumEttPUSLUT( l1t::LUT( (std::istream&)iss2 ) );
  oss.str("");

  std::vector<int> etSumEcalSumPUTLUT = conf["ecalET_energyCalibLUT"].getVector<int>();
  oss <<"#<header> V1 "<< ( 32 - __builtin_clz( uint32_t(etSumEcalSumPUTLUT.size()-1) ) ) <<" 63 </header> "<<std::endl; // hardcode max bits for data
  for(unsigned int i=0; i<etSumEcalSumPUTLUT.size(); i++) oss << i << " " << etSumEcalSumPUTLUT[i] << std::endl;

  std::istringstream iss3( oss.str() );
  paramsHelper.setEtSumEcalSumPUSLUT( l1t::LUT( (std::istream&)iss3 ) );
  oss.str("");

  std::vector<int> etSumXPUSLUT = conf["METX_energyCalibLUT"].getVector<int>();
  oss <<"#<header> V1 "<< ( 32 - __builtin_clz( uint32_t(etSumXPUSLUT.size()-1) ) ) <<" 63 </header> "<<std::endl; // hardcode max bits for data
  for(unsigned int i=0; i<etSumXPUSLUT.size(); i++) oss << i << " " << etSumXPUSLUT[i] << std::endl;

  std::istringstream iss4( oss.str() );
  paramsHelper.setEtSumXPUSLUT( l1t::LUT( (std::istream&)iss4 ) );
  oss.str("");

  paramsHelper.setEgMaxPtHOverE((conf["egammaRelaxationThreshold"].getValue<int>()));
  paramsHelper.setEgEtaCut((conf["egammaMaxEta"].getValue<int>()));


  std::vector<int> egCalibrationLUT = conf["egammaEnergyCalibLUT"].getVector<int>();
  oss <<"#<header> V1 "<< ( 32 - __builtin_clz( uint32_t(egCalibrationLUT.size()-1) ) ) <<" 63 </header> "<<std::endl; // hardcode max bits for data
  for(unsigned int i=0; i<egCalibrationLUT.size(); i++) oss << i << " " << egCalibrationLUT[i] << std::endl;

  std::istringstream iss5( oss.str() );
  paramsHelper.setEgCalibrationLUT( l1t::LUT( (std::istream&)iss5 ) );
  oss.str("");

  std::vector<int> egIsolationLUT = conf["egammaIsoLUT"].getVector<int>();
  oss <<"#<header> V1 "<< ( 32 - __builtin_clz( uint32_t(egIsolationLUT.size()-1) ) ) <<" 63 </header> "<<std::endl; // hardcode max bits for data
  for(unsigned int i=0; i<egIsolationLUT.size(); i++) oss << i << " " << egIsolationLUT[i] << std::endl;

  std::istringstream iss6( oss.str() );
  paramsHelper.setEgIsolationLUT( l1t::LUT( (std::istream&)iss6 ) );
  oss.str("");

  //std::cout<<"egammaIsoLUT: "<<std::endl;
  //for(unsigned int i=0; i<paramsHelper.egIsolationLUT()->maxSize(); i++) std::cout << std::setprecision(14) << paramsHelper.egIsolationLUT()->data(i) <<", ";
  //std::cout << std::endl;

  paramsHelper.setIsoTauEtaMax((conf["tauMaxEta"].getValue<int>()));

  std::vector<int> tauCalibrationLUT = conf["tauEnergyCalibLUT"].getVector<int>();
  oss <<"#<header> V1 "<< ( 32 - __builtin_clz( uint32_t(tauCalibrationLUT.size()-1) ) ) <<" 63 </header> "<<std::endl; // hardcode max bits for data
  for(unsigned int i=0; i<tauCalibrationLUT.size(); i++) oss << i << " " << tauCalibrationLUT[i] << std::endl;

  std::istringstream iss7( oss.str() );
  paramsHelper.setTauCalibrationLUT(l1t::LUT( (std::istream&)iss7 ));
  oss.str("");

  std::vector<int> tauIsolationLUT = conf["tauIsoLUT1"].getVector<int>();
  oss <<"#<header> V1 "<< ( 32 - __builtin_clz( uint32_t(tauIsolationLUT.size()-1) ) ) <<" 63 </header> "<<std::endl; // hardcode max bits for data
  for(unsigned int i=0; i<tauIsolationLUT.size(); i++) oss << i << " " << tauIsolationLUT[i] << std::endl;

  std::istringstream iss8( oss.str() );
  paramsHelper.setTauIsolationLUT( l1t::LUT((std::istream&)iss8 ) );
  oss.str("");

  std::vector<int> tauIsolationLUT2 = conf["tauIsoLUT2"].getVector<int>();
  oss <<"#<header> V1 "<< ( 32 - __builtin_clz( uint32_t(tauIsolationLUT2.size()-1) ) ) <<" 63 </header> "<<std::endl; // hardcode max bits for data
  for(unsigned int i=0; i<tauIsolationLUT2.size(); i++) oss << i << " " << tauIsolationLUT2[i] << std::endl;

  std::istringstream iss9( oss.str() );
  paramsHelper.setTauIsolationLUT2( l1t::LUT( (std::istream&)iss9 ) );
  oss.str("");

  return true;
}

L1TCaloParamsOnlineProd::L1TCaloParamsOnlineProd(const edm::ParameterSet& iConfig) : L1ConfigOnlineProdBaseExt<L1TCaloParamsO2ORcd,l1t::CaloParams>(iConfig) {}

boost::shared_ptr<l1t::CaloParams> L1TCaloParamsOnlineProd::newObject(const std::string& objectKey, const L1TCaloParamsO2ORcd& record) {
    using namespace edm::es;

    const L1TCaloStage2ParamsRcd& baseRcd = record.template getRecord< L1TCaloStage2ParamsRcd >() ;
    edm::ESHandle< l1t::CaloParams > baseSettings ;
    baseRcd.get( baseSettings ) ;


    if( objectKey.empty() ){
        edm::LogError( "L1-O2O: L1TCaloParamsOnlineProd" ) << "Key is empty, returning empty l1t::CaloParams";
        throw std::runtime_error("Empty objectKey");
///        return boost::shared_ptr< l1t::CaloParams > ( new l1t::CaloParams( *(baseSettings.product()) ) );
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
///        return boost::shared_ptr< l1t::CaloParams > ( new l1t::CaloParams( *(baseSettings.product()) ) );
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
///        return boost::shared_ptr< l1t::CaloParams >( new l1t::CaloParams( *(baseSettings.product()) ) ) ;
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
///        return boost::shared_ptr< l1t::CaloParams >( new l1t::CaloParams( *(baseSettings.product()) ) ) ;
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
///        return boost::shared_ptr< l1t::CaloParams >( new l1t::CaloParams( *(baseSettings.product()) ) ) ;
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
///        return boost::shared_ptr< l1t::CaloParams >( new l1t::CaloParams( *(baseSettings.product()) ) ) ;
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
///        return boost::shared_ptr< l1t::CaloParams >( new l1t::CaloParams( *(baseSettings.product()) ) ) ;
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
///        return boost::shared_ptr< l1t::CaloParams >( new l1t::CaloParams( *(baseSettings.product()) ) ) ;
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
///        return boost::shared_ptr< l1t::CaloParams >( new l1t::CaloParams( *(baseSettings.product()) ) ) ;
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
///        return boost::shared_ptr< l1t::CaloParams >( new l1t::CaloParams( *(baseSettings.product()) ) ) ;
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
///        return boost::shared_ptr< l1t::CaloParams >( new l1t::CaloParams( *(baseSettings.product()) ) ) ;
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
///        return boost::shared_ptr< l1t::CaloParams >( new l1t::CaloParams( *(baseSettings.product()) ) ) ;
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


    l1t::XmlConfigReader xmlReader1;
    xmlReader1.readDOMFromString( payloads[kCONF][calol1_algo_key] );

    l1t::TrigSystem calol1;
    calol1.addProcRole("processors", "processors");
    xmlReader1.readRootElement( calol1, "calol1" );
    calol1.setConfigured();

    std::map<std::string, l1t::Setting> calol1_conf = calol1.getSettings("processors");
    std::map<std::string, l1t::Mask>    calol1_rs   ;//= calol1.getMasks   ("processors");

    l1t::TrigSystem calol2;
//    calol2.addProcRole("processors", "MainProcessor");

    l1t::XmlConfigReader xmlReader2;
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
    std::map<std::string, l1t::Setting> calol2_conf = calol2.getSettings("MP1");
    std::map<std::string, l1t::Mask>    calol2_rs   ;//= calol2.getMasks   ("processors");
    
    l1t::CaloParamsHelper m_params_helper(*(baseSettings.product()));

    readCaloLayer1OnlineSettings(m_params_helper, calol1_conf, calol1_rs);
    readCaloLayer2OnlineSettings(m_params_helper, calol2_conf, calol2_rs);

    return boost::shared_ptr< l1t::CaloParams >( new l1t::CaloParams ( m_params_helper ) ) ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TCaloParamsOnlineProd);

