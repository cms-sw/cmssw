#include "CondTools/Ecal/interface/EcalTPGBadXTHandler.h"

#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/RunTPGConfigDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigMainInfo.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigBadXTInfo.h"
#include "OnlineDB/EcalCondDB/interface/RunList.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatus.h"

#include <iostream>
#include <fstream>

#include <ctime>
#include <unistd.h>

#include <string>
#include <cstdio>
#include <typeinfo>
#include <sstream>

popcon::EcalTPGBadXTHandler::EcalTPGBadXTHandler(const edm::ParameterSet& ps)
    : m_name(ps.getUntrackedParameter<std::string>("name", "EcalTPGBadXTHandler")) {
  edm::LogInfo("EcalTPGBadXTHandler") << "EcalTPGBadXT Source handler constructor.";
  m_firstRun = static_cast<unsigned int>(atoi(ps.getParameter<std::string>("firstRun").c_str()));
  m_lastRun = static_cast<unsigned int>(atoi(ps.getParameter<std::string>("lastRun").c_str()));
  m_sid = ps.getParameter<std::string>("OnlineDBSID");
  m_user = ps.getParameter<std::string>("OnlineDBUser");
  m_pass = ps.getParameter<std::string>("OnlineDBPassword");
  m_locationsource = ps.getParameter<std::string>("LocationSource");
  m_location = ps.getParameter<std::string>("Location");
  m_gentag = ps.getParameter<std::string>("GenTag");
  m_runtype = ps.getParameter<std::string>("RunType");

  edm::LogInfo("EcalTPGBadXTHandler") << m_sid << "/" << m_user << "/" << m_location << "/" << m_gentag;
}

popcon::EcalTPGBadXTHandler::~EcalTPGBadXTHandler() {}

void popcon::EcalTPGBadXTHandler::getNewObjects() {
  edm::LogInfo("EcalTPGBadXTHandler") << "Started GetNewObjects!!!";

  unsigned int max_since = 0;
  max_since = static_cast<unsigned int>(tagInfo().lastInterval.since);
  edm::LogInfo("EcalTPGBadXTHandler") << "max_since : " << max_since;
  edm::LogInfo("EcalTPGBadXTHandler") << "retrieved last payload ";

  // here we retrieve all the runs after the last from online DB
  edm::LogInfo("EcalTPGBadXTHandler") << "Retrieving run list from ONLINE DB ... ";

  edm::LogInfo("EcalTPGBadXTHandler") << "Making connection...";
  econn = new EcalCondDBInterface(m_sid, m_user, m_pass);
  edm::LogInfo("EcalTPGBadXTHandler") << "Done.";

  if (!econn) {
    std::cout << " connection parameters " << m_sid << "/" << m_user << std::endl;
    //      cerr << e.what() << std::endl;
    throw cms::Exception("OMDS not available");
  }

  LocationDef my_locdef;
  my_locdef.setLocation(m_location);

  RunTypeDef my_rundef;
  my_rundef.setRunType(m_runtype);

  RunTag my_runtag;
  my_runtag.setLocationDef(my_locdef);
  my_runtag.setRunTypeDef(my_rundef);
  my_runtag.setGeneralTag(m_gentag);

  readFromFile("last_tpg_badXT_settings.txt");

  unsigned int min_run;

  if (m_firstRun < m_i_run_number) {
    min_run = m_i_run_number + 1;
  } else {
    min_run = m_firstRun;
  }
  if (min_run < max_since) {
    min_run = max_since + 1;  // we have to add 1 to the last transferred one
  }

  std::cout << "m_i_run_number" << m_i_run_number << "m_firstRun " << m_firstRun << "max_since " << max_since
            << std::endl;

  unsigned int max_run = m_lastRun;
  edm::LogInfo("EcalTPGBadXTHandler") << "min_run= " << min_run << "max_run= " << max_run;

  RunList my_list;
  my_list = econn->fetchGlobalRunListByLocation(my_runtag, min_run, max_run, my_locdef);
  //	my_list=econn->fetchRunListByLocation(my_runtag, min_run, max_run, my_locdef);

  std::vector<RunIOV> run_vec = my_list.getRuns();
  size_t num_runs = run_vec.size();

  std::cout << "number of runs is : " << num_runs << std::endl;

  std::vector<EcalLogicID> my_EcalLogicId;
  std::vector<EcalLogicID> my_EcalLogicId_EE;

  unsigned int irun = 0;
  if (num_runs > 0) {
    my_EcalLogicId = econn->getEcalLogicIDSetOrdered(
        "ECAL_crystal_number_fedccuxt", 610, 650, 1, 100, 0, 100, "EB_crystal_number", 123);

    my_EcalLogicId_EE = econn->getEcalLogicIDSetOrdered(
        "ECAL_crystal_number_fedccuxt", 600, 700, 1, 100, 0, 100, "EE_crystal_number", 123);

    for (size_t kr = 0; kr < run_vec.size(); kr++) {
      std::cout << "here we are in run " << kr << std::endl;
      irun = static_cast<unsigned int>(run_vec[kr].getRunNumber());

      std::cout << " **************** " << std::endl;
      std::cout << " **************** " << std::endl;
      std::cout << " run= " << irun << std::endl;

      // retrieve the data :
      std::map<EcalLogicID, RunTPGConfigDat> dataset;
      econn->fetchDataSet(&dataset, &run_vec[kr]);

      std::string the_config_tag = "";
      int the_config_version = 0;

      std::map<EcalLogicID, RunTPGConfigDat>::const_iterator it;

      int nr = 0;
      for (it = dataset.begin(); it != dataset.end(); it++) {
        ++nr;
        //EcalLogicID ecalid  = it->first;

        RunTPGConfigDat dat = it->second;
        the_config_tag = dat.getConfigTag();
        the_config_version = dat.getVersion();
      }

      // it is all the same for all SM... get the last one

      std::cout << " run= " << irun << " tag " << the_config_tag << " version=" << the_config_version << std::endl;

      // here we should check if it is the same as previous run.

      if ((the_config_tag != m_i_tag || the_config_version != m_i_version) && nr > 0) {
        std::cout << "the tag is different from last transferred run ... retrieving last config set from DB"
                  << std::endl;

        FEConfigMainInfo fe_main_info;
        fe_main_info.setConfigTag(the_config_tag);
        fe_main_info.setVersion(the_config_version);
        std::cout << " version=" << fe_main_info.getVersion() << std::endl;

        try {
          std::cout << " before fetch config set" << std::endl;
          econn->fetchConfigSet(&fe_main_info);
          std::cout << " after fetch config set" << std::endl;

          // now get TPGBadXT
          int badxtId = fe_main_info.getBxtId();

          if (badxtId != m_i_badXT && badxtId != 0) {
            FEConfigBadXTInfo fe_badXt_info;
            fe_badXt_info.setId(badxtId);
            econn->fetchConfigSet(&fe_badXt_info);
            std::vector<FEConfigBadXTDat> dataset_TpgBadXT;
            econn->fetchConfigDataSet(&dataset_TpgBadXT, &fe_badXt_info);

            // NB new

            EcalTPGCrystalStatus* badXt;
            badXt = produceEcalTrgChannelStatus();

            typedef std::vector<FEConfigBadXTDat>::const_iterator CIfeped;
            EcalLogicID ecid_xt;
            FEConfigBadXTDat rd_badXt;

            for (CIfeped p = dataset_TpgBadXT.begin(); p != dataset_TpgBadXT.end(); p++) {
              rd_badXt = *p;

              int fed_id = rd_badXt.getFedId();
              //int tcc_id=rd_badXt.getTCCId();
              int tt_id = rd_badXt.getTTId();
              int xt_id = rd_badXt.getXTId();

              // EB data
              if (fed_id >= 610 && fed_id <= 645) {
                // logic id is 1011ssxxxx
                // get SM id
                int sm_num = 0;
                if (fed_id <= 627)
                  sm_num = fed_id - 609 + 18;
                if (fed_id > 627)
                  sm_num = fed_id - 627;

                // get crystal id
                int xt_num = 0;

                for (size_t ixt = 0; ixt < my_EcalLogicId.size(); ixt++) {
                  if (my_EcalLogicId[ixt].getID1() == fed_id && my_EcalLogicId[ixt].getID2() == tt_id &&
                      my_EcalLogicId[ixt].getID3() == xt_id) {
                    //1011060504
                    int ecid = my_EcalLogicId[ixt].getLogicID();
                    xt_num = (ecid) - (101100 + sm_num) * 10000;
                  }
                }

                std::cout << " masking crystal " << sm_num << "/" << xt_num << " from fed/tt/xt" << fed_id << "/"
                          << tt_id << "/" << xt_id << std::endl;
                if (sm_num == 0 && xt_num == 0) {
                  std::cout << " ERROR FOR crystal from fed/tt/xt" << fed_id << "/" << tt_id << "/" << xt_id
                            << std::endl;
                }
                EBDetId ebdetid(sm_num, xt_num, EBDetId::SMCRYSTALMODE);

                badXt->setValue(ebdetid.rawId(), rd_badXt.getStatus());
              } else {
                // EE data

                long x = 0;
                long y = 0;
                long z = 0;

                for (size_t ixt = 0; ixt < my_EcalLogicId_EE.size(); ixt++) {
                  if (my_EcalLogicId_EE[ixt].getID1() == fed_id && my_EcalLogicId_EE[ixt].getID2() == tt_id &&
                      my_EcalLogicId_EE[ixt].getID3() == xt_id) {
                    long ecid = (long)my_EcalLogicId_EE[ixt].getLogicID();
                    // logic_id 201Zxxxyyy Z=0 / 2 -> z= -1 / 1 , x -> 1 100,  y -> 1 100
                    y = ecid - ((long)(ecid / 1000)) * 1000;
                    x = (ecid - y) / 1000;
                    x = x - ((long)(x / 1000)) * 1000;
                    z = (ecid - y - x * 1000) / 1000000 - 2010;
                    if (z == 0)
                      z = -1;
                    if (z == 2)
                      z = 1;
                  }
                }

                if (x == 0 && y == 0 && z == 0) {
                  std::cout << " ERROR FOR crystal from fed/tt/xt" << fed_id << "/" << tt_id << "/" << xt_id
                            << std::endl;
                }
                EEDetId eedetid(x, y, z);
                badXt->setValue(eedetid.rawId(), rd_badXt.getStatus());
              }
            }  //end for over data

            edm::LogInfo("EcalTPGBadXTHandler") << "Finished badXT reading";

            Time_t snc = (Time_t)irun;
            m_to_transfer.push_back(std::make_pair((EcalTPGCrystalStatus*)badXt, snc));

            m_i_run_number = irun;
            m_i_tag = the_config_tag;
            m_i_version = the_config_version;
            m_i_badXT = badxtId;

            writeFile("last_tpg_badXT_settings.txt");

          } else {
            m_i_run_number = irun;
            m_i_tag = the_config_tag;
            m_i_version = the_config_version;

            writeFile("last_tpg_badXT_settings.txt");

            std::cout << " even if the tag/version is not the same, the badXT id is the same -> no transfer needed "
                      << std::endl;
          }

        }

        catch (std::exception& e) {
          std::cout << "ERROR: THIS CONFIG DOES NOT EXIST: tag=" << the_config_tag << " version=" << the_config_version
                    << std::endl;
          std::cout << e.what() << std::endl;
          m_i_run_number = irun;
        }
        std::cout << " **************** " << std::endl;

      } else if (nr == 0) {
        m_i_run_number = irun;
        std::cout << " no tag saved to RUN_TPGCONFIG_DAT by EcalSupervisor -> no transfer needed " << std::endl;
        std::cout << " **************** " << std::endl;
      } else {
        m_i_run_number = irun;
        m_i_tag = the_config_tag;
        m_i_version = the_config_version;
        std::cout << " the tag/version is the same -> no transfer needed " << std::endl;
        std::cout << " **************** " << std::endl;
        writeFile("last_tpg_badXT_settings.txt");
      }

    }  //end for over kr (nr of runs)
  }    //end if

  delete econn;

  edm::LogInfo("EcalTPGBadXTHandler") << "Ecal - > end of getNewObjects -----------";
}

void popcon::EcalTPGBadXTHandler::readFromFile(const char* inputFile) {
  //-------------------------------------------------------------

  m_i_tag = "";
  m_i_version = 0;
  m_i_run_number = 0;
  m_i_badXT = 0;

  FILE* inpFile;  // input file
  inpFile = fopen(inputFile, "r");
  if (!inpFile) {
    edm::LogError("EcalTPGBadXTHandler") << "*** Can not open file: " << inputFile;
    return;
  }

  char line[256];

  std::ostringstream str;

  fgets(line, 255, inpFile);
  m_i_tag = to_string(line);
  str << "gen tag " << m_i_tag << std::endl;  // should I use this?

  fgets(line, 255, inpFile);
  m_i_version = atoi(line);
  str << "version= " << m_i_version << std::endl;

  fgets(line, 255, inpFile);
  m_i_run_number = atoi(line);
  str << "run_number= " << m_i_run_number << std::endl;

  fgets(line, 255, inpFile);
  m_i_badXT = atoi(line);
  str << "badXT_config= " << m_i_badXT << std::endl;

  fclose(inpFile);  // close inp. file
}

void popcon::EcalTPGBadXTHandler::writeFile(const char* inputFile) {
  //-------------------------------------------------------------

  std::ofstream myfile;
  myfile.open(inputFile);
  myfile << m_i_tag << std::endl;
  myfile << m_i_version << std::endl;
  myfile << m_i_run_number << std::endl;
  myfile << m_i_badXT << std::endl;

  myfile.close();
}

EcalTPGCrystalStatus* popcon::EcalTPGBadXTHandler::produceEcalTrgChannelStatus() {
  EcalTPGCrystalStatus* ical = new EcalTPGCrystalStatus();
  // barrel
  for (int ieta = -EBDetId::MAX_IETA; ieta <= EBDetId::MAX_IETA; ++ieta) {
    if (ieta == 0)
      continue;
    for (int iphi = EBDetId::MIN_IPHI; iphi <= EBDetId::MAX_IPHI; ++iphi) {
      if (EBDetId::validDetId(ieta, iphi)) {
        EBDetId ebid(ieta, iphi);
        ical->setValue(ebid, 0);
      }
    }
  }
  // endcap
  for (int iX = EEDetId::IX_MIN; iX <= EEDetId::IX_MAX; ++iX) {
    for (int iY = EEDetId::IY_MIN; iY <= EEDetId::IY_MAX; ++iY) {
      // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
      if (EEDetId::validDetId(iX, iY, 1)) {
        EEDetId eedetidpos(iX, iY, 1);
        ical->setValue(eedetidpos, 0);
      }
      if (EEDetId::validDetId(iX, iY, -1)) {
        EEDetId eedetidneg(iX, iY, -1);
        ical->setValue(eedetidneg, 0);
      }
    }
  }
  return ical;
}
