#include <iostream>
#include <fstream>
#include <stdexcept>
#include <sstream>

#include "CondTools/L1TriggerExt/interface/L1ConfigOnlineProdBaseExt.h"
#include "CondFormats/L1TObjects/interface/CaloParams.h"
#include "CondFormats/DataRecord/interface/L1TCaloParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1TCaloParamsO2ORcd.h"
#include "L1Trigger/L1TCommon/interface/TriggerSystem.h"
#include "L1Trigger/L1TCommon/interface/XmlConfigParser.h"
#include "L1Trigger/L1TCommon/interface/ConvertToLUT.h"
//#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"
#include "CaloParamsHelperO2O.h"
#include "OnlineDBqueryHelper.h"

#include "xercesc/util/PlatformUtils.hpp"
using namespace XERCES_CPP_NAMESPACE;

class L1TCaloParamsOnlineProd : public L1ConfigOnlineProdBaseExt<L1TCaloParamsO2ORcd,l1t::CaloParams> {
private:
    unsigned int exclusiveLayer; // 0 - process calol1 and calol2, 1 - only calol1, 2 - only calol2
    bool transactionSafe;

    bool readCaloLayer1OnlineSettings(l1t::CaloParamsHelperO2O& paramsHelper, std::map<std::string, l1t::Parameter>& conf, std::map<std::string, 
l1t::Mask>& );
    bool readCaloLayer2OnlineSettings(l1t::CaloParamsHelperO2O& paramsHelper, std::map<std::string, l1t::Parameter>& conf, std::map<std::string, 
l1t::Mask>& );
public:
    std::shared_ptr<l1t::CaloParams> newObject(const std::string& objectKey, const L1TCaloParamsO2ORcd& record) override ;

    L1TCaloParamsOnlineProd(const edm::ParameterSet&);
    ~L1TCaloParamsOnlineProd(void) override{}
};

bool
L1TCaloParamsOnlineProd::readCaloLayer1OnlineSettings(l1t::CaloParamsHelperO2O& paramsHelper, std::map<std::string, l1t::Parameter>& conf, 
std::map<std::string, l1t::Mask>& ) {
  const char * expectedParams[] = {
    "layer1ECalScaleFactors",
    "layer1HCalScaleFactors",
    "layer1HFScaleFactors",
    "layer1ECalScaleETBins",
    "layer1HCalScaleETBins",
    "layer1HFScaleETBins"
    // Optional params
    //"layer1ECalScalePhiBins",
    //"layer1HCalScalePhiBins",
    //"layer1HFScalePhiBins",
    //"layer1SecondStageLUT"
  };
  for (const auto param : expectedParams) {
    if ( conf.find(param) == conf.end() ) {
      edm::LogError("L1-O2O: L1TCaloParamsOnlineProd") << "Unable to locate expected CaloLayer1 parameter: " << param << " in L1 settings payload!";
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

  if( conf.find("layer1ECalScalePhiBins") != conf.end() )
      paramsHelper.setLayer1ECalScalePhiBins(conf["layer1ECalScalePhiBins"].getVector<unsigned int>()); // std::vector<unsigned>(36,0)
  if( conf.find("layer1HCalScalePhiBins") != conf.end() )
      paramsHelper.setLayer1HCalScalePhiBins(conf["layer1HCalScalePhiBins"].getVector<unsigned int>());
  if( conf.find("layer1HFScalePhiBins") != conf.end() )
      paramsHelper.setLayer1HFScalePhiBins  (conf["layer1HFScalePhiBins"]  .getVector<unsigned int>());
  if( conf.find("layer1SecondStageLUT") != conf.end() )
      paramsHelper.setLayer1SecondStageLUT(conf["layer1SecondStageLUT"].getVector<unsigned int>() );

  return true;
}

bool
L1TCaloParamsOnlineProd::readCaloLayer2OnlineSettings(l1t::CaloParamsHelperO2O& paramsHelper, std::map<std::string, l1t::Parameter>& conf, 
std::map<std::string, l1t::Mask>& ) {
  const char * expectedParams[] = {
    "leptonSeedThreshold",
    "leptonTowerThreshold",
    "pileUpTowerThreshold",
    "egammaRelaxationThreshold",
    "egammaMaxEta",
    "egammaBypassCuts",
    "egammaBypassShape",
    "egammaBypassEcalFG",
    "egammaBypassExtendedHOverE",
    "egammaHOverECut_iEtaLT15",
    "egammaHOverECut_iEtaGTEq15",
    "egammaEnergyCalibLUT",
    "egammaIsoLUT1",
    "egammaIsoLUT2",
    "tauMaxEta",
    "tauEnergyCalibLUT",
    "tauIsoLUT",
    "tauTrimmingLUT",
    "jetSeedThreshold",
    "HTMHT_maxJetEta",
    "HT_jetThreshold",
    "MHT_jetThreshold",
    "jetBypassPileUpSub",
    "jetEnergyCalibLUT",
    "jetPUSUsePhiRing",
    "towerCountThreshold",
    "towerCountMaxEta",
    "ETMET_maxTowerEta",
    "ecalET_towerThresholdLUT",
    "ET_towerThresholdLUT",
    "MET_towerThresholdLUT",
    "ET_centralityLowerThresholds",
    "ET_centralityUpperThresholds",
    "ET_energyCalibLUT",
    "ecalET_energyCalibLUT",
    "MET_energyCalibLUT",
    "METHF_energyCalibLUT",
    "MET_phiCalibLUT",
    "METHF_phiCalibLUT",
  };

  for (const auto param : expectedParams) {
    if ( conf.find(param) == conf.end() ) {
      edm::LogError("L1-O2O: L1TCaloParamsOnlineProd") << "Unable to locate expected CaloLayer2 parameter: " << param << " in L1 settings payload!";
      return false;
    }
  }
  // Layer 2 params specification
  paramsHelper.setEgSeedThreshold((conf["leptonSeedThreshold"].getValue<int>())/2);
  paramsHelper.setTauSeedThreshold((conf["leptonSeedThreshold"].getValue<int>())/2);
  paramsHelper.setEgNeighbourThreshold((conf["leptonTowerThreshold"].getValue<int>())/2);
  paramsHelper.setTauNeighbourThreshold((conf["leptonTowerThreshold"].getValue<int>())/2);
  paramsHelper.setPileUpTowerThreshold((conf["pileUpTowerThreshold"].getValue<int>())/2);

  paramsHelper.setEgMaxPtHOverE((conf["egammaRelaxationThreshold"].getValue<int>())/2.);
  paramsHelper.setEgEtaCut((conf["egammaMaxEta"].getValue<int>()));
  paramsHelper.setEgBypassEGVetos(conf["egammaBypassCuts"].getValue<bool>());
  paramsHelper.setEgBypassShape( conf["egammaBypassShape"].getValue<bool>() );
  paramsHelper.setEgBypassECALFG( conf["egammaBypassEcalFG"].getValue<bool>() );
  paramsHelper.setEgBypassExtHOverE( conf["egammaBypassExtendedHOverE"].getValue<bool>() );
  paramsHelper.setEgHOverEcutBarrel(conf["egammaHOverECut_iEtaLT15"].getValue<int>());
  paramsHelper.setEgHOverEcutEndcap(conf["egammaHOverECut_iEtaGTEq15"].getValue<int>());
  paramsHelper.setEgCalibrationLUT  ( l1t::convertToLUT( conf["egammaEnergyCalibLUT"].getVector<int>() ) );
  paramsHelper.setEgIsolationLUT    ( l1t::convertToLUT( conf["egammaIsoLUT1"].getVector<int>() ) );
  paramsHelper.setEgIsolationLUT2   ( l1t::convertToLUT( conf["egammaIsoLUT2"].getVector<int>() ) );

  paramsHelper.setIsoTauEtaMax((conf["tauMaxEta"].getValue<int>()));
  paramsHelper.setTauCalibrationLUT( l1t::convertToLUT( conf["tauEnergyCalibLUT"].getVector<int>() ) );
  paramsHelper.setTauIsolationLUT  ( l1t::convertToLUT( conf["tauIsoLUT"].getVector<int>() ) );
  paramsHelper.setTauTrimmingShapeVetoLUT( l1t::convertToLUT( conf["tauTrimmingLUT"].getVector<int>() ) );


  paramsHelper.setJetSeedThreshold((conf["jetSeedThreshold"].getValue<int>())/2);
  paramsHelper.setJetBypassPUS(conf["jetBypassPileUpSub"].getValue<bool>());
  paramsHelper.setJetPUSUsePhiRing(conf["jetPUSUsePhiRing"].getValue<bool>());
  paramsHelper.setJetCalibrationLUT ( l1t::convertToLUT( conf["jetEnergyCalibLUT"].getVector<uint32_t>() ) );

  std::vector<int> etSumEtaMax;
  std::vector<int> etSumEtThresh;
  
  etSumEtaMax.push_back(conf["ETMET_maxTowerEta"].getValue<int>());
  etSumEtaMax.push_back(conf["HTMHT_maxJetEta"].getValue<int>());
  etSumEtaMax.push_back(conf["ETMET_maxTowerEta"].getValue<int>());
  etSumEtaMax.push_back(conf["HTMHT_maxJetEta"].getValue<int>());
  etSumEtaMax.push_back(conf["towerCountMaxEta"].getValue<int>());

  etSumEtThresh.push_back(0); //deprecated by EttPUSLUT
  etSumEtThresh.push_back(conf["HT_jetThreshold"].getValue<int>()/2);
  etSumEtThresh.push_back(0); //deprecated by MetPUSLUT
  etSumEtThresh.push_back(conf["MHT_jetThreshold"].getValue<int>()/2);
  etSumEtThresh.push_back(conf["towerCountThreshold"].getValue<int>()/2);

  for (uint i=0; i<5; ++i) {
    paramsHelper.setEtSumEtaMax(i, etSumEtaMax.at(i));
    paramsHelper.setEtSumEtThreshold(i, etSumEtThresh.at(i));
  }

  paramsHelper.setEtSumMetPUSLUT    ( l1t::convertToLUT( conf["MET_towerThresholdLUT"].getVector<int>() ) );
  paramsHelper.setEtSumEttPUSLUT    ( l1t::convertToLUT( conf["ET_towerThresholdLUT"].getVector<int>() ) );
  paramsHelper.setEtSumEcalSumPUSLUT( l1t::convertToLUT( conf["ecalET_towerThresholdLUT"].getVector<int>() ) );

  std::vector<double> etSumCentLowerValues;
  std::vector<double> etSumCentUpperValues;

  etSumCentLowerValues = conf["ET_centralityLowerThresholds"].getVector<double>();
  etSumCentUpperValues = conf["ET_centralityUpperThresholds"].getVector<double>();

  for(uint i=0; i<8; ++i){
    paramsHelper.setEtSumCentLower(i, etSumCentLowerValues[i]/2);
    paramsHelper.setEtSumCentUpper(i, etSumCentUpperValues[i]/2);
  }

  // demux tower sum calib LUTs
  paramsHelper.setEtSumEttCalibrationLUT    ( l1t::convertToLUT( conf["ET_energyCalibLUT"].getVector<int>() ) );
  paramsHelper.setEtSumEcalSumCalibrationLUT( l1t::convertToLUT( conf["ecalET_energyCalibLUT"].getVector<int>() ) );
  paramsHelper.setMetCalibrationLUT      ( l1t::convertToLUT( conf["MET_energyCalibLUT"].getVector<int>() ) );
  paramsHelper.setMetHFCalibrationLUT      ( l1t::convertToLUT( conf["METHF_energyCalibLUT"].getVector<int>() ) );
  paramsHelper.setMetPhiCalibrationLUT      ( l1t::convertToLUT( conf["MET_phiCalibLUT"].getVector<int>() ) );
  paramsHelper.setMetHFPhiCalibrationLUT      ( l1t::convertToLUT( conf["METHF_phiCalibLUT"].getVector<int>() ) );

  return true;
}

L1TCaloParamsOnlineProd::L1TCaloParamsOnlineProd(const edm::ParameterSet& iConfig) : 
    L1ConfigOnlineProdBaseExt<L1TCaloParamsO2ORcd,l1t::CaloParams>(iConfig)
{
    exclusiveLayer  = iConfig.getParameter<uint32_t>("exclusiveLayer");
    transactionSafe = iConfig.getParameter<bool>("transactionSafe");
}

std::shared_ptr<l1t::CaloParams> L1TCaloParamsOnlineProd::newObject(const std::string& objectKey, const L1TCaloParamsO2ORcd& record) {
    using namespace edm::es;

    const L1TCaloParamsRcd& baseRcd = record.template getRecord< L1TCaloParamsRcd >() ;
    edm::ESHandle< l1t::CaloParams > baseSettings ;
    baseRcd.get( baseSettings ) ;


    if( objectKey.empty() ){
        edm::LogError( "L1-O2O: L1TCaloParamsOnlineProd" ) << "Key is empty";
        if( transactionSafe )
            throw std::runtime_error("SummaryForFunctionManager: Calo  | Faulty  | Empty objectKey");
        else {
            edm::LogError( "L1-O2O: L1TCaloParamsOnlineProd" ) << "returning unmodified prototype of l1t::CaloParams";
            return std::make_shared< l1t::CaloParams >( *(baseSettings.product()) ) ;
        }
    }

    std::string tscKey = objectKey.substr(0, objectKey.find(":") );
    std::string  rsKey = objectKey.substr(   objectKey.find(":")+1, std::string::npos );

    edm::LogInfo( "L1-O2O: L1TCaloParamsOnlineProd" ) << "Producing L1TCaloParamsOnlineProd with TSC key = " << tscKey << " and RS key = " << rsKey;

    std::string calol1_top_key, calol1_algo_key;
    std::string calol1_algo_payload;
    std::string calol2_top_key, calol2_algo_key, calol2_hw_key;
    std::string calol2_hw_payload;
    std::map<std::string,std::string> calol2_algo_payloads;  // key -> XML payload
    try {

        std::map<std::string,std::string> topKeys =
            l1t::OnlineDBqueryHelper::fetch( {"CALOL1_KEY","CALOL2_KEY"},
                                             "L1_TRG_CONF_KEYS",
                                             tscKey,
                                             m_omdsReader
                                           );

      if( exclusiveLayer == 0 || exclusiveLayer == 1 ){

        calol1_top_key = topKeys["CALOL1_KEY"];

        calol1_algo_key = l1t::OnlineDBqueryHelper::fetch( {"ALGO"},
                                                           "CALOL1_KEYS",
                                                           calol1_top_key,
                                                           m_omdsReader
                                                         ) ["ALGO"];

        calol1_algo_payload = l1t::OnlineDBqueryHelper::fetch( {"CONF"},
                                                               "CALOL1_CLOBS",
                                                                calol1_algo_key,
                                                                m_omdsReader
                                                             ) ["CONF"];
      }

      if( exclusiveLayer == 0 || exclusiveLayer == 2 ){

        calol2_top_key = topKeys["CALOL2_KEY"];

        std::map<std::string,std::string> calol2_keys =
            l1t::OnlineDBqueryHelper::fetch( {"ALGO","HW"},
                                             "CALOL2_KEYS",
                                             calol2_top_key,
                                             m_omdsReader
                                           );

        calol2_hw_key = calol2_keys["HW"];
        calol2_hw_payload = l1t::OnlineDBqueryHelper::fetch( {"CONF"},
                                                             "CALOL2_CLOBS",
                                                              calol2_hw_key,
                                                              m_omdsReader
                                                           ) ["CONF"];

        calol2_algo_key = calol2_keys["ALGO"];

        std::map<std::string,std::string> calol2_algo_keys =
            l1t::OnlineDBqueryHelper::fetch( {"DEMUX","MPS_COMMON","MPS_JET","MP_EGAMMA","MP_SUM","MP_TAU"},
                                             "CALOL2_ALGO_KEYS",
                                             calol2_algo_key,
                                             m_omdsReader
                                           );

        for(auto &key : calol2_algo_keys)
            calol2_algo_payloads[ key.second ] = 
                l1t::OnlineDBqueryHelper::fetch( {"CONF"},
                                                 "CALOL2_CLOBS",
                                                 key.second,
                                                 m_omdsReader
                                               ) ["CONF"];
      }

    } catch ( std::runtime_error &e ) {
        edm::LogError( "L1-O2O: L1TCaloParamsOnlineProd" ) << e.what();
        if( transactionSafe )
            throw std::runtime_error(std::string("SummaryForFunctionManager: Calo  | Faulty  | ") + e.what());
        else {
            edm::LogError( "L1-O2O: L1TCaloParamsOnlineProd" ) << "returning unmodified prototype of l1t::CaloParams";
            return std::make_shared< l1t::CaloParams >( *(baseSettings.product()) ) ;
        }
    }

    if( exclusiveLayer == 0 || exclusiveLayer == 2 ){
        // for debugging purposes dump the configs to local files
        for(auto &conf : calol2_algo_payloads){ 
            std::ofstream output(std::string("/tmp/").append(conf.first.substr(0,conf.first.find("/"))).append(".xml"));
            output<<conf.second;
            output.close();
        }
        std::ofstream output(std::string("/tmp/").append(calol2_hw_key.substr(0,calol2_hw_key.find("/"))).append(".xml"));
        output << calol2_hw_payload;
        output.close();
    }
    if( exclusiveLayer == 0 || exclusiveLayer == 1 )
    { 
        std::ofstream output(std::string("/tmp/").append(calol1_algo_key.substr(0,calol1_algo_key.find("/"))).append(".xml"));
        output << calol1_algo_payload;
        output.close();
    }

    l1t::CaloParamsHelperO2O m_params_helper( *(baseSettings.product()) );


    if( exclusiveLayer == 0 || exclusiveLayer == 1 ){
        try {
            l1t::XmlConfigParser xmlReader1;
            xmlReader1.readDOMFromString( calol1_algo_payload );

            l1t::TriggerSystem calol1;
            calol1.addProcessor("processors", "processors","-1","-1");
            xmlReader1.readRootElement( calol1, "calol1" );
            calol1.setConfigured();

            std::map<std::string, l1t::Parameter> calol1_conf = calol1.getParameters("processors");
            std::map<std::string, l1t::Mask>      calol1_rs   ;//= calol1.getMasks   ("processors");

            if( !readCaloLayer1OnlineSettings(m_params_helper, calol1_conf, calol1_rs) )
                throw std::runtime_error("Parsing error for CaloLayer1");

        } catch ( std::runtime_error &e ){
            edm::LogError( "L1-O2O: L1TCaloParamsOnlineProd" ) << e.what();
            if( transactionSafe )
                throw std::runtime_error(std::string("SummaryForFunctionManager: Calo  | Faulty  | ") + e.what());
            else {
                edm::LogError( "L1-O2O: L1TCaloParamsOnlineProd" ) << "returning unmodified prototype of l1t::CaloParams";
                return std::make_shared< l1t::CaloParams >( *(baseSettings.product()) ) ;
            }
        }
    }


    if( exclusiveLayer == 0 || exclusiveLayer == 2 ){
        try {
            l1t::TriggerSystem calol2;
            l1t::XmlConfigParser xmlReader2;
            xmlReader2.readDOMFromString( calol2_hw_payload );
            xmlReader2.readRootElement( calol2, "calol2" );

            for(auto &conf : calol2_algo_payloads){ 
                xmlReader2.readDOMFromString( conf.second );
                xmlReader2.readRootElement( calol2, "calol2" );
            }

//            calol2.setSystemId("calol2");
            calol2.setConfigured();

            std::map<std::string, l1t::Parameter> calol2_conf = calol2.getParameters("MP1");
	    std::map<std::string, l1t::Parameter> calol2_conf_demux = calol2.getParameters("DEMUX");
	    calol2_conf.insert( calol2_conf_demux.begin(), calol2_conf_demux.end() ) ;
            std::map<std::string, l1t::Mask>      calol2_rs   ;//= calol2.getMasks   ("processors");

            if( !readCaloLayer2OnlineSettings(m_params_helper, calol2_conf, calol2_rs) )
                throw std::runtime_error("Parsing error for CaloLayer2");

        } catch ( std::runtime_error &e ){
            edm::LogError( "L1-O2O: L1TCaloParamsOnlineProd" ) << e.what();
            if( transactionSafe )
                throw std::runtime_error(std::string("SummaryForFunctionManager: Calo  | Faulty  | ") + e.what());
            else {
                edm::LogError( "L1-O2O: L1TCaloParamsOnlineProd" ) << "returning unmodified prototype of l1t::CaloParams";
                return std::make_shared< l1t::CaloParams >( *(baseSettings.product()) ) ;
            }
        }
    }
    
    std::shared_ptr< l1t::CaloParams > retval = std::make_shared< l1t::CaloParams >( m_params_helper ) ;
    
    edm::LogInfo( "L1-O2O: L1TCaloParamsOnlineProd" ) << "SummaryForFunctionManager: Calo  | OK      | All looks good";
    return retval;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TCaloParamsOnlineProd);


