#include "CondTools/Ecal/interface/EcalLaserHandler.h"

#include "CondTools/Ecal/interface/EcalTPGOddWeightGroupHandler.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/RunTPGConfigDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigMainInfo.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigSlidingInfo.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigSlidingDat.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"

#include <iostream>
#include <fstream>

#include <ctime>
#include <unistd.h>

#include <string>
#include <cstdio>
#include <typeinfo>
#include <sstream>

const Int_t kEBStrips = 12240, kEEStrips = 2936;

popcon::EcalTPGOddWeightGroupHandler::EcalTPGOddWeightGroupHandler(const edm::ParameterSet& ps)
    : m_name(ps.getUntrackedParameter<std::string>("name", "EcalTPGOddWeightGroupHandler")) {
  edm::LogInfo("EcalTPGOddWeightGroupHandler") << "EcalTPGOddWeightGroup Source handler constructor.";
  m_firstRun = static_cast<unsigned int>(atoi(ps.getParameter<std::string>("firstRun").c_str()));
  m_lastRun = static_cast<unsigned int>(atoi(ps.getParameter<std::string>("lastRun").c_str()));
  m_sid = ps.getParameter<std::string>("OnlineDBSID");
  m_user = ps.getParameter<std::string>("OnlineDBUser");
  m_pass = ps.getParameter<std::string>("OnlineDBPassword");
  m_locationsource = ps.getParameter<std::string>("LocationSource");
  m_location = ps.getParameter<std::string>("Location");
  m_gentag = ps.getParameter<std::string>("GenTag");
  m_runtype = ps.getParameter<std::string>("RunType");
  m_file_type = ps.getParameter<std::string>("fileType");  // xml/txt
  m_file_name = ps.getParameter<std::string>("fileName");

  edm::LogInfo("EcalTPGOddWeightGroupHandler") << m_sid << "/" << m_user << "/" << m_location << "/" << m_gentag;
}

popcon::EcalTPGOddWeightGroupHandler::~EcalTPGOddWeightGroupHandler() {}

void popcon::EcalTPGOddWeightGroupHandler::getNewObjects() {
  if (m_file_type == "txt") {
    readtxtFile();
  } else if (m_file_type == "xml") {
    readxmlFile();
  } else {
    edm::LogInfo("EcalTPGOddWeightGroupHandler") << "Started GetNewObjects!!!";

    //check whats already inside of database
    if (tagInfo().size) {
      //check whats already inside of database
      std::cout << "got offlineInfo = " << std::endl;
      std::cout << "tag name = " << tagInfo().name << std::endl;
      std::cout << "size = " << tagInfo().size << std::endl;
    } else {
      std::cout << " First object for this tag " << std::endl;
    }

    unsigned int max_since = 0;
    max_since = static_cast<unsigned int>(tagInfo().lastInterval.since);
    edm::LogInfo("EcalTPGOddWeightGroupHandler") << "max_since : " << max_since;

    edm::LogInfo("EcalTPGOddWeightGroupHandler") << "retrieved last payload ";

    // here we retrieve all the runs after the last from online DB

    edm::LogInfo("EcalTPGOddWeightGroupHandler") << "Retrieving run list from ONLINE DB ... ";

    edm::LogInfo("EcalTPGOddWeightGroupHandler") << "Making connection...";
    econn = new EcalCondDBInterface(m_sid, m_user, m_pass);
    edm::LogInfo("EcalTPGOddWeightGroupHandler") << "Done.";

    if (!econn) {
      std::cout << " connection parameters " << m_sid << "/" << m_user << std::endl;
      //	    cerr << e.what() << std::endl;
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

    readFromFile("last_tpg_OddweightGroup_settings.txt");

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
    edm::LogInfo("EcalTPGOddWeightGroupHandler") << "min_run= " << min_run << " max_run= " << max_run;

    RunList my_list;
    my_list = econn->fetchGlobalRunListByLocation(my_runtag, min_run, max_run, my_locdef);
    //        my_list=econn->fetchRunListByLocation(my_runtag,min_run,max_run,my_locdef);

    std::vector<RunIOV> run_vec = my_list.getRuns();
    size_t num_runs = run_vec.size();

    std::cout << "number of runs is : " << num_runs << std::endl;

    unsigned int irun = 0;
    if (num_runs > 0) {
      // going to query the ecal logic id
      std::vector<EcalLogicID> my_StripEcalLogicId_EE;
      my_StripEcalLogicId_EE =
          econn->getEcalLogicIDSetOrdered("ECAL_readout_strip", 1, 2000, 1, 70, 0, 5, "EE_offline_stripid", 123);

      std::cout << " GOT the logic ID for the EE trigger strips " << std::endl;

      for (size_t kr = 0; kr < run_vec.size(); kr++) {
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

          try {
            std::cout << " before fetch config set" << std::endl;
            econn->fetchConfigSet(&fe_main_info);
            std::cout << " after fetch config set" << std::endl;

            // now get TPGOddWeightGroup
            int wId = fe_main_info.getWei2Id();

            if (wId != m_i_oddweightGroup) {
              FEConfigOddWeightInfo fe_oddw_info;
              fe_oddw_info.setId(wId);
              econn->fetchConfigSet(&fe_oddw_info);
              std::map<EcalLogicID, FEConfigOddWeightDat> dataset_TpgW;
              econn->fetchDataSet(&dataset_TpgW, &fe_oddw_info);

              EcalTPGOddWeightGroup* weightG = new EcalTPGOddWeightGroup;
              typedef std::map<EcalLogicID, FEConfigOddWeightDat>::const_iterator CIfesli;
              EcalLogicID ecid_xt;
              int weightGroup;

              std::map<std::string, int> map;
              std::string str;

              for (CIfesli p = dataset_TpgW.begin(); p != dataset_TpgW.end(); p++) {
                ecid_xt = p->first;
                weightGroup = p->second.getWeightGroupId();

                std::string ecid_name = ecid_xt.getName();

                // EB data
                if (ecid_name == "EB_VFE") {
                  int sm = ecid_xt.getID1();
                  int tt = ecid_xt.getID2();
                  int strip = ecid_xt.getID3();
                  int tcc = sm + 54;
                  if (sm > 18)
                    tcc = sm + 18;

                  // simple formula to calculate the Srip EB identifier

                  unsigned int stripEBId = 303176 + (tt - 1) * 64 + (strip - 1) * 8 + (tcc - 37) * 8192;

                  weightG->setValue(stripEBId, weightGroup);
                } else if (ecid_name == "ECAL_readout_strip") {
                  // EE data to add
                  int id1 = ecid_xt.getID1();
                  int id2 = ecid_xt.getID2();
                  int id3 = ecid_xt.getID3();

                  bool set_the_strip = false;
                  int stripEEId;
                  for (size_t istrip = 0; istrip < my_StripEcalLogicId_EE.size(); istrip++) {
                    if (!set_the_strip) {
                      if (my_StripEcalLogicId_EE[istrip].getID1() == id1 &&
                          my_StripEcalLogicId_EE[istrip].getID2() == id2 &&
                          my_StripEcalLogicId_EE[istrip].getID3() == id3) {
                        stripEEId = my_StripEcalLogicId_EE[istrip].getLogicID();
                        set_the_strip = true;
                        break;
                      }
                    }
                  }

                  if (set_the_strip) {
                    weightG->setValue(stripEEId, weightGroup);

                  } else {
                    std::cout << " these may be the additional towers TCC/TT " << id1 << "/" << id2 << std::endl;
                  }
                }
              }

              Time_t snc = (Time_t)irun;

              m_to_transfer.push_back(std::make_pair((EcalTPGOddWeightGroup*)weightG, snc));

              m_i_run_number = irun;
              m_i_tag = the_config_tag;
              m_i_version = the_config_version;
              m_i_oddweightGroup = wId;

              writeFile("last_tpg_OddweightGroup_settings.txt");

            } else {
              m_i_run_number = irun;
              m_i_tag = the_config_tag;
              m_i_version = the_config_version;

              writeFile("last_tpg_OddweightGroup_settings.txt");

              std::cout
                  << " even if the tag/version is not the same, the weight group id is the same -> no transfer needed "
                  << std::endl;
            }

          }

          catch (std::exception& e) {
            std::cout << "ERROR: THIS CONFIG DOES NOT EXIST: tag=" << the_config_tag
                      << " version=" << the_config_version << std::endl;
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
          writeFile("last_tpg_OddweightGroup_settings.txt");
        }
      }
    }

    delete econn;
  }  // usual way
  edm::LogInfo("EcalTPGOddWeightGroupHandler") << "Ecal - > end of getNewObjects -----------";
}

void popcon::EcalTPGOddWeightGroupHandler::readtxtFile() {
  std::cout << " reading the input file " << m_file_name << std::endl;
  std::ifstream fInput;
  fInput.open(m_file_name);
  if (!fInput.is_open()) {
    std::cout << "ERROR : cannot open file " << m_file_name << std::endl;
    exit(1);
  }
  int weightGroup, stripEBId, stripEEId;
  EcalTPGOddWeightGroup* weightG = new EcalTPGOddWeightGroup;
  for (int strip = 0; strip < kEBStrips; strip++) {
    fInput >> stripEBId >> weightGroup;
    weightG->setValue(stripEBId, weightGroup);
  }
  for (int strip = 0; strip < kEEStrips; strip++) {
    fInput >> stripEEId >> weightGroup;
    weightG->setValue(stripEEId, weightGroup);
  }
  try {
    Time_t snc = (Time_t)m_firstRun;
    m_to_transfer.push_back(std::make_pair((EcalTPGOddWeightGroup*)weightG, snc));
  } catch (std::exception& e) {
    std::cout << "EcalTPGOddWeightGroupHandler::readtxtFile error : " << e.what() << std::endl;
  }
  std::cout << " **************** " << std::endl;
}

void popcon::EcalTPGOddWeightGroupHandler::readxmlFile() {
  std::cout << " reading the input file " << m_file_name << std::endl;
  std::ifstream fxml;
  fxml.open(m_file_name);
  if (!fxml.is_open()) {
    std::cout << "ERROR : cannot open file " << m_file_name << std::endl;
    exit(1);
  }
  std::string dummyLine, bid;
  int weightGroup, stripEBId, stripEEId;
  EcalTPGOddWeightGroup* weightG = new EcalTPGOddWeightGroup;
  for (int i = 0; i < 6; i++)
    std::getline(fxml, dummyLine);  // skip first lines
  fxml >> bid;
  //  std::cout << bid << std::endl;
  std::size_t found = bid.find("</");
  std::string stt = bid.substr(7, found - 7);
  for (int i = 0; i < 2; i++)
    std::getline(fxml, dummyLine);  //    <item_version>0</item_version>
  for (int strip = 0; strip < kEBStrips; strip++) {
    std::getline(fxml, dummyLine);  //    <item
    fxml >> bid;                    //    <first
    found = bid.find("</");
    stt = bid.substr(7, found - 7);
    std::istringstream sg1(stt);
    sg1 >> stripEBId;
    std::getline(fxml, dummyLine);
    fxml >> bid;  //    <second
    found = bid.find("</");
    stt = bid.substr(8, found - 8);
    std::istringstream sg2(stt);
    sg2 >> weightGroup;
    weightG->setValue(stripEBId, weightGroup);
    for (int i = 0; i < 2; i++)
      std::getline(fxml, dummyLine);  //    </item>
  }
  for (int strip = 0; strip < kEEStrips; strip++) {
    std::getline(fxml, dummyLine);  //    <item
    fxml >> bid;                    //    <first
    found = bid.find("</");
    stt = bid.substr(7, found - 7);
    std::istringstream sg1(stt);
    sg1 >> stripEEId;
    std::getline(fxml, dummyLine);
    fxml >> bid;  //    <second
    found = bid.find("</");
    stt = bid.substr(8, found - 8);
    std::istringstream sg2(stt);
    sg2 >> weightGroup;
    weightG->setValue(stripEEId, weightGroup);
    for (int i = 0; i < 2; i++)
      std::getline(fxml, dummyLine);  //    </item>
  }
  try {
    Time_t snc = (Time_t)m_firstRun;
    m_to_transfer.push_back(std::make_pair((EcalTPGOddWeightGroup*)weightG, snc));
  } catch (std::exception& e) {
    std::cout << "EcalTPGOddWeightGroupHandler::readtxtFile error : " << e.what() << std::endl;
  }
  std::cout << " **************** " << std::endl;
}

void popcon::EcalTPGOddWeightGroupHandler::readFromFile(const char* inputFile) {
  //-------------------------------------------------------------

  m_i_tag = "";
  m_i_version = 0;
  m_i_run_number = 0;
  m_i_oddweightGroup = 0;

  FILE* inpFile;  // input file
  inpFile = fopen(inputFile, "r");
  if (!inpFile) {
    edm::LogError("EcalTPGOddWeightGroupHandler") << "*** Can not open file: " << inputFile;
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
  m_i_oddweightGroup = atoi(line);
  str << "weightGroup_config= " << m_i_oddweightGroup << std::endl;

  fclose(inpFile);  // close inp. file
}

void popcon::EcalTPGOddWeightGroupHandler::writeFile(const char* inputFile) {
  //-------------------------------------------------------------

  std::ofstream myfile;
  myfile.open(inputFile);
  myfile << m_i_tag << std::endl;
  myfile << m_i_version << std::endl;
  myfile << m_i_run_number << std::endl;
  myfile << m_i_oddweightGroup << std::endl;

  myfile.close();
}
