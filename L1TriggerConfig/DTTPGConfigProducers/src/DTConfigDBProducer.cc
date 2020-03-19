// -*- C++ -*-
//
// Package:     DTTPGConfigProducers
// Class:       DTConfigDBProducer
//
/**\class  DTConfigDBProducer  DTConfigDBProducer.h
 L1TriggerConfig/DTTPGConfigProducers/interface/DTConfigDBProducer.h

 Description: A Producer for the DT config, data retrieved from DB

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sara Vanini
//         Created:  September 2008
//
//
// system include files
#include <memory>
#include <vector>
#include <iomanip>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CondTools/DT/interface/DTKeyedConfigCache.h"

#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManager.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManagerRcd.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"

#include "CondFormats/DTObjects/interface/DTCCBConfig.h"
#include "CondFormats/DTObjects/interface/DTKeyedConfig.h"
#include "CondFormats/DataRecord/interface/DTCCBConfigRcd.h"
#include "CondFormats/DataRecord/interface/DTKeyedConfigListRcd.h"
#include "CondFormats/DataRecord/interface/DTT0Rcd.h"
#include "CondFormats/DataRecord/interface/DTTPGParametersRcd.h"

#include "L1TriggerConfig/DTTPGConfigProducers/src/DTPosNegType.h"

using std::cout;
using std::endl;
using std::unique_ptr;
using std::vector;

//
// class declaration
//

class DTConfigDBProducer : public edm::ESProducer {
public:
  //! Constructor
  DTConfigDBProducer(const edm::ParameterSet &);

  //! Destructor
  ~DTConfigDBProducer() override;

  //! ES produce method
  std::unique_ptr<DTConfigManager> produce(const DTConfigManagerRcd &);

private:
  //! Read DTTPG pedestal configuration
  void readDBPedestalsConfig(const DTConfigManagerRcd &iRecord, DTConfigManager &dttpgConfig);

  //! Read CCB string configuration
  int readDTCCBConfig(const DTConfigManagerRcd &iRecord, DTConfigManager &dttpgConfig);

  //! SV for debugging purpose ONLY
  void configFromCfg(DTConfigManager &dttpgConfig);

  //! SV for debugging purpose ONLY
  DTConfigPedestals buildTrivialPedestals();

  //! 110629 SV function for CCB configuration check
  int checkDTCCBConfig(DTConfigManager &dttpgConfig);

  std::string mapEntryName(const DTChamberId &chambid) const;

  // ----------member data ---------------------------
  edm::ParameterSet m_ps;

  edm::ESGetToken<DTTPGParameters, DTTPGParametersRcd> m_dttpgParamsToken;
  edm::ESGetToken<DTT0, DTT0Rcd> m_t0iToken;
  edm::ESGetToken<DTCCBConfig, DTCCBConfigRcd> m_ccb_confToken;
  edm::ESGetToken<cond::persistency::KeyList, DTKeyedConfigListRcd> m_keyListToken;

  // debug flags
  bool m_debugDB;
  int m_debugBti;
  int m_debugTraco;
  bool m_debugTSP;
  bool m_debugTST;
  bool m_debugTU;
  bool m_debugSC;
  bool m_debugLUTs;
  bool m_debugPed;

  // general DB requests
  bool m_UseT0;

  bool cfgConfig;

  bool flagDBBti, flagDBTraco, flagDBTSS, flagDBTSM, flagDBLUTS;

  DTKeyedConfigCache cfgCache;
};

//
// constructors and destructor
//

DTConfigDBProducer::DTConfigDBProducer(const edm::ParameterSet &p) {
  // tell the framework what record is being produced
  auto cc = setWhatProduced(this, &DTConfigDBProducer::produce);

  cfgConfig = p.getParameter<bool>("cfgConfig");

  // get and store parameter set and config manager pointer
  m_ps = p;

  // debug flags
  m_debugDB = p.getParameter<bool>("debugDB");
  m_debugBti = p.getParameter<int>("debugBti");
  m_debugTraco = p.getParameter<int>("debugTraco");
  m_debugTSP = p.getParameter<bool>("debugTSP");
  m_debugTST = p.getParameter<bool>("debugTST");
  m_debugTU = p.getParameter<bool>("debugTU");
  m_debugSC = p.getParameter<bool>("debugSC");
  m_debugLUTs = p.getParameter<bool>("debugLUTs");
  m_debugPed = p.getParameter<bool>("debugPed");

  m_UseT0 = p.getParameter<bool>("UseT0");  // CB check for a better way to do it

  if (not cfgConfig) {
    cc.setConsumes(m_dttpgParamsToken).setConsumes(m_ccb_confToken).setConsumes(m_keyListToken);
    if (m_UseT0) {
      cc.setConsumes(m_t0iToken);
    }
  }
}

DTConfigDBProducer::~DTConfigDBProducer() {}

//
// member functions
//

std::unique_ptr<DTConfigManager> DTConfigDBProducer::produce(const DTConfigManagerRcd &iRecord) {
  using namespace edm;

  std::unique_ptr<DTConfigManager> dtConfig = std::unique_ptr<DTConfigManager>(new DTConfigManager());
  DTConfigManager &dttpgConfig = *(dtConfig.get());

  // DB specific requests
  bool tracoLutsFromDB = m_ps.getParameter<bool>("TracoLutsFromDB");
  bool useBtiAcceptParam = m_ps.getParameter<bool>("UseBtiAcceptParam");

  // set specific DB requests
  dttpgConfig.setLutFromDB(tracoLutsFromDB);
  dttpgConfig.setUseAcceptParam(useBtiAcceptParam);

  // set debug
  edm::ParameterSet conf_ps = m_ps.getParameter<edm::ParameterSet>("DTTPGParameters");
  bool dttpgdebug = conf_ps.getUntrackedParameter<bool>("Debug");
  dttpgConfig.setDTTPGDebug(dttpgdebug);

  int code;
  if (cfgConfig) {
    dttpgConfig.setLutFromDB(false);
    configFromCfg(dttpgConfig);
    code = 2;
  } else {
    code = readDTCCBConfig(iRecord, dttpgConfig);
    readDBPedestalsConfig(iRecord, dttpgConfig);
    // 110628 SV add config check
    if (code != -1 && checkDTCCBConfig(dttpgConfig) > 0)
      code = -1;
  }
  // cout << "DTConfigDBProducer::produce CODE " << code << endl;
  if (code == -1) {
    dttpgConfig.setCCBConfigValidity(false);
  } else if (code == 2) {
    LogVerbatim("DTTPG") << "DTConfigDBProducer::produce : Trivial : " << endl
                         << "configurations has been read from cfg" << endl;
  } else if (code == 0) {
    LogVerbatim("DTTPG") << "DTConfigDBProducer::produce : " << endl
                         << "Configurations successfully read from OMDS" << endl;
  } else {
    LogProblem("DTTPG") << "DTConfigDBProducer::produce : " << endl << "Wrong configuration return CODE" << endl;
  }

  return dtConfig;
}

void DTConfigDBProducer::readDBPedestalsConfig(const DTConfigManagerRcd &iRecord, DTConfigManager &dttpgConfig) {
  const auto &dttpgParams = iRecord.get(m_dttpgParamsToken);

  DTConfigPedestals pedestals;
  pedestals.setDebug(m_debugPed);

  if (m_UseT0) {
    pedestals.setUseT0(true);
    pedestals.setES(&dttpgParams, &iRecord.get(m_t0iToken));
    // cout << "checkDTCCBConfig CODE is " << checkDTCCBConfig() << endl;

  } else {
    pedestals.setUseT0(false);
    pedestals.setES(&dttpgParams);
  }

  dttpgConfig.setDTConfigPedestals(pedestals);
}

int DTConfigDBProducer::checkDTCCBConfig(DTConfigManager &dttpgConfig) {
  // 110627 SV test if configuration from CCB has correct number of chips,
  // return error code:
  // check_cfg_code = 1 : NO correct BTI number
  // check_cfg_code = 2 : NO correct TRACO number
  // check_cfg_code = 3 : NO correct valid TSS number
  // check_cfg_code = 4 : NO correct valid TSM number

  int check_cfg_code = 0;

  // do not take chambers from MuonGeometryRecord to avoid geometry dependency
  for (int iwh = -2; iwh <= 2; iwh++) {
    for (int ise = 1; ise <= 12; ise++) {
      for (int ist = 1; ist <= 4; ist++) {
        check_cfg_code = 0;
        DTChamberId chid(iwh, ist, ise);

        // retrive number of configurated chip
        int nbti = dttpgConfig.getDTConfigBtiMap(chid).size();
        int ntraco = dttpgConfig.getDTConfigTracoMap(chid).size();
        int ntss = dttpgConfig.getDTConfigTSPhi(chid)->nValidTSS();
        int ntsm = dttpgConfig.getDTConfigTSPhi(chid)->nValidTSM();

        // check BTIs
        if ((ist == 1 && nbti != 168) || (ist == 2 && nbti != 192) || (ist == 3 && nbti != 224) ||
            (ist == 4 &&
             (ise == 1 || ise == 2 || ise == 3 || ise == 5 || ise == 6 || ise == 7 || ise == 8 || ise == 12) &&
             nbti != 192) ||
            (ist == 4 && (ise == 9 || ise == 11) && nbti != 96) || (ist == 4 && ise == 10 && nbti != 128) ||
            (ist == 4 && ise == 4 && nbti != 160)) {
          check_cfg_code = 1;
          return check_cfg_code;
        }

        // check TRACOs
        if ((ist == 1 && ntraco != 13) || (ist == 2 && ntraco != 16) || (ist == 3 && ntraco != 20) ||
            (ist == 4 &&
             (ise == 1 || ise == 2 || ise == 3 || ise == 5 || ise == 6 || ise == 7 || ise == 8 || ise == 12) &&
             ntraco != 24) ||
            (ist == 4 && (ise == 9 || ise == 11) && ntraco != 12) || (ist == 4 && ise == 10 && ntraco != 16) ||
            (ist == 4 && ise == 4 && ntraco != 20)) {
          check_cfg_code = 2;
          return check_cfg_code;
        }

        // check TSS
        if ((ist == 1 && ntss != 4) || (ist == 2 && ntss != 4) || (ist == 3 && ntss != 5) ||
            (ist == 4 &&
             (ise == 1 || ise == 2 || ise == 3 || ise == 5 || ise == 6 || ise == 7 || ise == 8 || ise == 12) &&
             ntss != 6) ||
            (ist == 4 && (ise == 9 || ise == 11) && ntss != 3) || (ist == 4 && ise == 10 && ntss != 4) ||
            (ist == 4 && ise == 4 && ntss != 5)) {
          check_cfg_code = 3;
          return check_cfg_code;
        }

        // check TSM
        if (ntsm != 1) {
          check_cfg_code = 4;
          return check_cfg_code;
        }

        // if(check_cfg_code){
        // cout << "nbti " << nbti << " ntraco " << ntraco << " ntss " << ntss
        // << " ntsm " << ntsm << endl; cout << "Check: ch " << ist << " sec "
        // << ise << " wh " << iwh << " == >check_cfg_code " << check_cfg_code
        // << endl;
        //}
      }  // end st loop
    }    // end sec loop

    // SV MB4 has two more chambers
    for (int ise = 13; ise <= 14; ise++) {
      DTChamberId chid(iwh, 4, ise);

      int nbti = dttpgConfig.getDTConfigBtiMap(chid).size();
      int ntraco = dttpgConfig.getDTConfigTracoMap(chid).size();
      int ntss = dttpgConfig.getDTConfigTSPhi(chid)->nValidTSS();
      int ntsm = dttpgConfig.getDTConfigTSPhi(chid)->nValidTSM();

      if ((ise == 13 && nbti != 160) || (ise == 14 && nbti != 128)) {
        check_cfg_code = 1;
        return check_cfg_code;
      }
      if ((ise == 13 && ntraco != 20) || (ise == 14 && ntraco != 16)) {
        check_cfg_code = 2;
        return check_cfg_code;
      }
      if ((ise == 13 && ntss != 5) || (ise == 14 && ntss != 4)) {
        check_cfg_code = 3;
        return check_cfg_code;
      }
      if (ntsm != 1) {
        check_cfg_code = 4;
        return check_cfg_code;
      }
      // if(check_cfg_code){
      // cout << "nbti " << nbti << " ntraco " << ntraco << " ntss " << ntss <<
      // " ntsm " << ntsm << endl; cout << "Check: ch " << 4 << " sec " << ise
      // << " wh " << iwh << " == >check_cfg_code " << check_cfg_code << endl;
      //}
    }  // end sec 13 14

  }  // end wh loop

  // cout << "CheckDTCCB: config OK! check_cfg_code = " << check_cfg_code <<
  // endl;
  return check_cfg_code;
}

int DTConfigDBProducer::readDTCCBConfig(const DTConfigManagerRcd &iRecord, DTConfigManager &dttpgConfig) {
  using namespace edm::eventsetup;

  // initialize CCB validity flag
  dttpgConfig.setCCBConfigValidity(true);

  // get DTCCBConfigRcd from DTConfigManagerRcd (they are dependent records)
  const auto &ccb_conf = iRecord.get(m_ccb_confToken);
  int ndata = std::distance(ccb_conf.begin(), ccb_conf.end());

  const DTKeyedConfigListRcd &keyRecord = iRecord.getRecord<DTKeyedConfigListRcd>();

  if (m_debugDB) {
    cout << ccb_conf.version() << endl;
    cout << ndata << " data in the container" << endl;
  }

  edm::ValidityInterval iov(iRecord.getRecord<DTCCBConfigRcd>().validityInterval());
  unsigned int currValidityStart = iov.first().eventID().run();
  unsigned int currValidityEnd = iov.last().eventID().run();

  if (m_debugDB)
    cout << "valid since run " << currValidityStart << " to run " << currValidityEnd << endl;

  // if there are no data in the container, configuration from cfg files...
  if (ndata == 0) {
    return -1;
  }

  // get DTTPGMap for retrieving bti number and traco number
  edm::ParameterSet conf_map = m_ps.getUntrackedParameter<edm::ParameterSet>("DTTPGMap");

  // loop over chambers
  DTCCBConfig::ccb_config_map configKeys(ccb_conf.configKeyMap());
  DTCCBConfig::ccb_config_iterator iter = configKeys.begin();
  DTCCBConfig::ccb_config_iterator iend = configKeys.end();

  // 110628 SV check that number of CCB is equal to total number of chambers
  if (ccb_conf.configKeyMap().size() != 250)  // check the number of chambers!!!
    return -1;

  auto const &keyList = keyRecord.get(m_keyListToken);

  // read data from CCBConfig
  while (iter != iend) {
    // 110628 SV moved here from constructor, to check config consistency for
    // EVERY chamber initialize flags to check if data are present in OMDS
    flagDBBti = false;
    flagDBTraco = false;
    flagDBTSS = false;
    flagDBTSM = false;
    flagDBLUTS = false;

    // get chamber id
    const DTCCBId &ccbId = iter->first;
    if (m_debugDB)
      cout << " Filling configuration for chamber : wh " << ccbId.wheelId << " st " << ccbId.stationId << " se "
           << ccbId.sectorId << " -> " << endl;

    // get chamber type and id from ccbId
    int mbtype = DTPosNegType::getCT(ccbId.wheelId, ccbId.sectorId, ccbId.stationId);
    int posneg = DTPosNegType::getPN(ccbId.wheelId, ccbId.sectorId, ccbId.stationId);
    if (m_debugDB)
      cout << "Chamber type : " << mbtype << " posneg : " << posneg << endl;
    DTChamberId chambid(ccbId.wheelId, ccbId.stationId, ccbId.sectorId);

    // get brick identifiers list
    const std::vector<int> &ccbConf = iter->second;
    std::vector<int>::const_iterator cfgIter = ccbConf.begin();
    std::vector<int>::const_iterator cfgIend = ccbConf.end();

    // TSS-TSM buffers
    unsigned short int tss_buffer[7][31];
    unsigned short int tsm_buffer[9];
    int ntss = 0;

    // loop over configuration bricks
    while (cfgIter != cfgIend) {
      // get brick identifier
      int id = *cfgIter++;
      if (m_debugDB)
        cout << " BRICK " << id << endl;

      // create strings list
      std::vector<std::string> list;

      cfgCache.getData(keyList, id, list);

      // loop over strings
      std::vector<std::string>::const_iterator s_iter = list.begin();
      std::vector<std::string>::const_iterator s_iend = list.end();
      while (s_iter != s_iend) {
        if (m_debugDB)
          cout << "        ----> " << *s_iter << endl;

        // copy string in unsigned int buffer
        std::string str = *s_iter++;
        unsigned short int buffer[100];  // 2 bytes
        int c = 0;
        const char *cstr = str.c_str();
        const char *ptr = cstr + 2;
        const char *end = cstr + str.length();
        while (ptr < end) {
          char c1 = *ptr++;
          int i1 = 0;
          if ((c1 >= '0') && (c1 <= '9'))
            i1 = c1 - '0';
          if ((c1 >= 'a') && (c1 <= 'f'))
            i1 = 10 + c1 - 'a';
          if ((c1 >= 'A') && (c1 <= 'F'))
            i1 = 10 + c1 - 'A';
          char c2 = *ptr++;
          int i2 = 0;
          if ((c2 >= '0') && (c2 <= '9'))
            i2 = c2 - '0';
          if ((c2 >= 'a') && (c2 <= 'f'))
            i2 = 10 + c2 - 'a';
          if ((c2 >= 'A') && (c2 <= 'F'))
            i2 = 10 + c2 - 'A';
          buffer[c] = (i1 * 16) + i2;
          c++;
        }  // end loop over string

        // BTI configuration string
        if (buffer[2] == 0x54) {
          if (m_debugDB)
            cout << "BTI STRING found in DB" << endl;

          // BTI configuration read for BTI
          flagDBBti = true;

          // compute sl and bti number from board and chip
          int brd = buffer[3];   // Board Nr.
          int chip = buffer[4];  // Chip Nr.

          if (brd > 7) {
            cout << "Not existing board ... " << brd << endl;
            return -1;  // Non-existing board
          }
          if (chip > 31) {
            cout << "Not existing chip... " << chip << endl;
            return -1;  // Non existing chip
          }

          // Is it Phi or Theta board?
          bool ThetaSL, PhiSL;
          PhiSL = false;
          ThetaSL = false;
          switch (mbtype) {
            case 1:  // mb1
              if (brd == 6 || brd == 7) {
                ThetaSL = true;
                brd -= 6;
              } else if ((brd < 3 && chip < 32) || (brd == 3 && chip < 8))
                PhiSL = true;
              break;
            case 2:  // mb2
              if (brd == 6 || brd == 7) {
                ThetaSL = true;
                brd -= 6;
              } else if (brd < 4 && chip < 32)
                PhiSL = true;
              break;
            case 3:  // mb3
              if (brd == 6 || brd == 7) {
                ThetaSL = true;
                brd -= 6;
              } else if (brd < 5 && chip < 32)
                PhiSL = true;
              break;
            case 4:  // mb4-s, mb4_8
              if (brd < 6 && chip < 32)
                PhiSL = true;
              break;
            case 5:  // mb4-9
              if (brd < 3 && chip < 32)
                PhiSL = true;
              break;
            case 6:  // mb4-4
              if (brd < 5 && chip < 32)
                PhiSL = true;
              break;
            case 7:  // mb4-10
              if (brd < 4 && chip < 32)
                PhiSL = true;
              break;
          }
          if (!PhiSL && !ThetaSL) {
            cout << "MB type " << mbtype << endl;
            cout << "Board " << brd << " chip " << chip << endl;
            cout << "Not phi SL nor Theta SL" << endl;
            return -1;  // Not PhiSL nor ThetaSL
          }

          // compute SL number and bti number
          int isl = 0;
          int ibti = 0;
          if (PhiSL) {
            if ((chip % 8) < 4)
              isl = 1;  // Phi1
            else
              isl = 3;  // Phi2
            ibti = brd * 16 + (int)(chip / 8) * 4 + (chip % 4);
          } else if (ThetaSL) {
            isl = 2;  // Theta
            if ((chip % 8) < 4)
              ibti = brd * 32 + chip - 4 * (int)(chip / 8);
            else
              ibti = brd * 32 + chip + 12 - 4 * (int)(chip / 8);
          }

          // BTI config constructor from strings
          DTConfigBti bticonf(m_debugBti, buffer);

          dttpgConfig.setDTConfigBti(DTBtiId(chambid, isl, ibti + 1), bticonf);

          if (m_debugDB)
            cout << "Filling BTI config for chamber : wh " << chambid.wheel() << ", st " << chambid.station() << ", se "
                 << chambid.sector() << "... sl " << isl << ", bti " << ibti + 1 << endl;
        }

        // TRACO configuration string
        if (buffer[2] == 0x15) {
          if (m_debugDB)
            cout << "TRACO STRING found in DB" << endl;
          // TRACO configuration read from OMDS
          flagDBTraco = true;

          // TRACO config constructor from strings
          int traco_brd = buffer[3];   // Board Nr.;
          int traco_chip = buffer[4];  // Chip Nr.;
          int itraco = traco_brd * 4 + traco_chip + 1;
          DTConfigTraco tracoconf(m_debugTraco, buffer);
          dttpgConfig.setDTConfigTraco(DTTracoId(chambid, itraco), tracoconf);

          if (m_debugDB)
            cout << "Filling TRACO config for chamber : wh " << chambid.wheel() << ", st " << chambid.station()
                 << ", se " << chambid.sector() << ", board " << traco_brd << ", chip " << traco_chip << ", traco "
                 << itraco << endl;
        }

        // TSS configuration string
        if (buffer[2] == 0x16) {
          if (m_debugDB)
            cout << "TSS STRING found in DB" << endl;
          // TSS configuration read from OMDS
          flagDBTSS = true;

          unsigned short int itss = buffer[3];
          for (int i = 0; i < 31; i++)
            tss_buffer[itss][i] = buffer[i];
          ntss++;
        }

        // TSM configuration string
        if (buffer[2] == 0x17) {
          if (m_debugDB)
            cout << "TSM STRING found in DB" << endl;

          // TSM configuration read from OMDS
          flagDBTSM = true;

          for (int i = 0; i < 9; i++)
            tsm_buffer[i] = buffer[i];
        }

        // LUT configuration string
        if (buffer[2] == 0xA8) {
          if (m_debugDB)
            cout << "LUT STRING found in DB" << endl;

          // LUT parameters read from OMDS
          flagDBLUTS = true;
          DTConfigLUTs lutconf(m_debugLUTs, buffer);
          // lutconf.setDebug(m_debugLUTs);
          dttpgConfig.setDTConfigLUTs(chambid, lutconf);
        }

      }  // end string iteration
    }    // end brick iteration

    // TSS + TSM configurations are set in DTConfigTSPhi constructor
    if (flagDBTSM && flagDBTSS) {
      DTConfigTSPhi tsphiconf(m_debugTSP, tss_buffer, ntss, tsm_buffer);
      dttpgConfig.setDTConfigTSPhi(chambid, tsphiconf);
    }

    // get configuration for TSTheta, SC and TU from .cfg
    edm::ParameterSet conf_ps = m_ps.getParameter<edm::ParameterSet>("DTTPGParameters");
    edm::ParameterSet tups = conf_ps.getParameter<edm::ParameterSet>("TUParameters");

    // TSTheta configuration from .cfg
    DTConfigTSTheta tsthetaconf(tups.getParameter<edm::ParameterSet>("TSThetaParameters"));
    tsthetaconf.setDebug(m_debugTST);
    dttpgConfig.setDTConfigTSTheta(chambid, tsthetaconf);

    // SC configuration from .cfg
    DTConfigSectColl sectcollconf(conf_ps.getParameter<edm::ParameterSet>("SectCollParameters"));
    sectcollconf.setDebug(m_debugSC);
    dttpgConfig.setDTConfigSectColl(DTSectCollId(chambid.wheel(), chambid.sector()), sectcollconf);

    // TU configuration from .cfg
    DTConfigTrigUnit trigunitconf(tups);
    trigunitconf.setDebug(m_debugTU);
    dttpgConfig.setDTConfigTrigUnit(chambid, trigunitconf);

    ++iter;

    // 110628 SV moved inside CCB loop to check for every chamber
    // moved to exception handling no attempt to configure from cfg is DB is
    // missing SV comment exception handling and activate flag in
    // DTConfigManager
    if (!flagDBBti || !flagDBTraco || !flagDBTSS || !flagDBTSM) {
      return -1;
    }
    if (!flagDBLUTS && dttpgConfig.lutFromDB() == true) {
      return -1;
    }
  }  // end loop over CCB

  return 0;
}

std::string DTConfigDBProducer::mapEntryName(const DTChamberId &chambid) const {
  int iwh = chambid.wheel();
  std::ostringstream os;
  os << "wh";
  if (iwh < 0) {
    os << 'm' << -iwh;
  } else {
    os << iwh;
  }
  os << "st" << chambid.station() << "se" << chambid.sector();
  return os.str();
}

void DTConfigDBProducer::configFromCfg(DTConfigManager &dttpgConfig) {
  // ... but still set CCB validity flag to let the emulator run
  dttpgConfig.setCCBConfigValidity(true);

  // create config classes&C.
  edm::ParameterSet conf_ps = m_ps.getParameter<edm::ParameterSet>("DTTPGParameters");
  edm::ParameterSet conf_map = m_ps.getUntrackedParameter<edm::ParameterSet>("DTTPGMap");
  bool dttpgdebug = conf_ps.getUntrackedParameter<bool>("Debug");
  DTConfigSectColl sectcollconf(conf_ps.getParameter<edm::ParameterSet>("SectCollParameters"));
  edm::ParameterSet tups = conf_ps.getParameter<edm::ParameterSet>("TUParameters");
  DTConfigBti bticonf(tups.getParameter<edm::ParameterSet>("BtiParameters"));
  DTConfigTraco tracoconf(tups.getParameter<edm::ParameterSet>("TracoParameters"));
  DTConfigTSTheta tsthetaconf(tups.getParameter<edm::ParameterSet>("TSThetaParameters"));
  DTConfigTSPhi tsphiconf(tups.getParameter<edm::ParameterSet>("TSPhiParameters"));
  DTConfigTrigUnit trigunitconf(tups);
  DTConfigLUTs lutconf(tups.getParameter<edm::ParameterSet>("LutParameters"));

  for (int iwh = -2; iwh <= 2; ++iwh) {
    for (int ist = 1; ist <= 4; ++ist) {
      for (int ise = 1; ise <= 12; ++ise) {
        DTChamberId chambid(iwh, ist, ise);
        vector<int> nmap = conf_map.getUntrackedParameter<vector<int>>(mapEntryName(chambid).c_str());

        if (dttpgdebug) {
          cout << " Filling configuration for chamber : wh " << chambid.wheel() << ", st " << chambid.station()
               << ", se " << chambid.sector() << endl;
        }

        // fill the bti map
        if (!flagDBBti) {
          for (int isl = 1; isl <= 3; isl++) {
            int ncell = nmap[isl - 1];
            //	  cout << ncell <<" , ";
            for (int ibti = 0; ibti < ncell; ibti++) {
              dttpgConfig.setDTConfigBti(DTBtiId(chambid, isl, ibti + 1), bticonf);
              if (dttpgdebug)
                cout << "Filling BTI config for chamber : wh " << chambid.wheel() << ", st " << chambid.station()
                     << ", se " << chambid.sector() << "... sl " << isl << ", bti " << ibti + 1 << endl;
            }
          }
        }

        // fill the traco map
        if (!flagDBTraco) {
          int ntraco = nmap[3];
          // cout << ntraco << " }" << endl;
          for (int itraco = 0; itraco < ntraco; itraco++) {
            dttpgConfig.setDTConfigTraco(DTTracoId(chambid, itraco + 1), tracoconf);
            if (dttpgdebug)
              cout << "Filling TRACO config for chamber : wh " << chambid.wheel() << ", st " << chambid.station()
                   << ", se " << chambid.sector() << ", traco " << itraco + 1 << endl;
          }
        }

        // fill TS & TrigUnit
        if (!flagDBTSS || !flagDBTSM) {
          dttpgConfig.setDTConfigTSTheta(chambid, tsthetaconf);
          dttpgConfig.setDTConfigTSPhi(chambid, tsphiconf);
          dttpgConfig.setDTConfigTrigUnit(chambid, trigunitconf);
        }
      }
    }
  }

  for (int iwh = -2; iwh <= 2; ++iwh) {
    for (int ise = 13; ise <= 14; ++ise) {
      int ist = 4;
      DTChamberId chambid(iwh, ist, ise);
      vector<int> nmap = conf_map.getUntrackedParameter<vector<int>>(mapEntryName(chambid).c_str());

      if (dttpgdebug) {
        cout << " Filling configuration for chamber : wh " << chambid.wheel() << ", st " << chambid.station() << ", se "
             << chambid.sector() << endl;
      }

      // fill the bti map
      if (!flagDBBti) {
        for (int isl = 1; isl <= 3; isl++) {
          int ncell = nmap[isl - 1];
          // 	cout << ncell <<" , ";
          for (int ibti = 0; ibti < ncell; ibti++) {
            dttpgConfig.setDTConfigBti(DTBtiId(chambid, isl, ibti + 1), bticonf);
            if (dttpgdebug)
              cout << "Filling BTI config for chamber : wh " << chambid.wheel() << ", st " << chambid.station()
                   << ", se " << chambid.sector() << "... sl " << isl << ", bti " << ibti + 1 << endl;
          }
        }
      }

      // fill the traco map
      if (!flagDBTraco) {
        int ntraco = nmap[3];
        //       cout << ntraco << " }" << endl;
        for (int itraco = 0; itraco < ntraco; itraco++) {
          dttpgConfig.setDTConfigTraco(DTTracoId(chambid, itraco + 1), tracoconf);
          if (dttpgdebug)
            cout << "Filling TRACO config for chamber : wh " << chambid.wheel() << ", st " << chambid.station()
                 << ", se " << chambid.sector() << ", traco " << itraco + 1 << endl;
        }
      }

      // fill TS & TrigUnit
      if (!flagDBTSS || !flagDBTSM) {
        dttpgConfig.setDTConfigTSTheta(chambid, tsthetaconf);
        dttpgConfig.setDTConfigTSPhi(chambid, tsphiconf);
        dttpgConfig.setDTConfigTrigUnit(chambid, trigunitconf);
      }
    }
  }

  // loop on Sector Collectors
  for (int wh = -2; wh <= 2; wh++)
    for (int se = 1; se <= 12; se++)
      dttpgConfig.setDTConfigSectColl(DTSectCollId(wh, se), sectcollconf);

  // fake collection of pedestals
  dttpgConfig.setDTConfigPedestals(buildTrivialPedestals());

  return;
}

DTConfigPedestals DTConfigDBProducer::buildTrivialPedestals() {
  DTTPGParameters *m_tpgParams = new DTTPGParameters();

  int counts = m_ps.getParameter<int>("bxOffset");
  float fine = m_ps.getParameter<double>("finePhase");

  if (m_debugPed)
    cout << "DTConfigTrivialProducer::buildPedestals()" << endl;

  // DTTPGParameters tpgParams;
  for (int iwh = -2; iwh <= 2; ++iwh) {
    for (int ist = 1; ist <= 4; ++ist) {
      for (int ise = 1; ise <= 14; ++ise) {
        if (ise > 12 && ist != 4)
          continue;

        DTChamberId chId(iwh, ist, ise);
        m_tpgParams->set(chId, counts, fine, DTTimeUnits::ns);
      }
    }
  }

  DTConfigPedestals tpgPedestals;
  tpgPedestals.setUseT0(false);
  tpgPedestals.setES(m_tpgParams);

  return tpgPedestals;
}

DEFINE_FWK_EVENTSETUP_MODULE(DTConfigDBProducer);
