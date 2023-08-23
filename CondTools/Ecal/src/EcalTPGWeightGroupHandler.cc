#include "CondTools/Ecal/interface/EcalLaserHandler.h"

#include "CondTools/Ecal/interface/EcalTPGWeightGroupHandler.h"
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

popcon::EcalTPGWeightGroupHandler::EcalTPGWeightGroupHandler(const edm::ParameterSet& ps)
    : m_name(ps.getUntrackedParameter<std::string>("name", "EcalTPGWeightGroupHandler")) {
  edm::LogInfo("EcalTPGWeightGroupHandler") << "EcalTPGWeightGroup Source handler constructor.";
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

  edm::LogInfo("EcalTPGWeightGroupHandler") << m_sid << "/" << m_user << "/" << m_location << "/" << m_gentag;
}

popcon::EcalTPGWeightGroupHandler::~EcalTPGWeightGroupHandler() {}

void popcon::EcalTPGWeightGroupHandler::getNewObjects() {
  if (m_file_type == "txt") {
    readtxtFile();
  } else if (m_file_type == "xml") {
    readxmlFile();
  } else {
    edm::LogInfo("EcalTPGWeightGroupHandler") << "Started GetNewObjects!!!";

    //check whats already inside of database
    if (tagInfo().size) {
      //check whats already inside of database
      edm::LogInfo(" got offlineInfo = ");
      edm::LogInfo(" tag name = ") << tagInfo().name;
      edm::LogInfo(" size = ") << tagInfo().size;
    } else {
      edm::LogInfo(" First object for this tag ");
    }

    unsigned int max_since = 0;
    max_since = static_cast<unsigned int>(tagInfo().lastInterval.since);
    edm::LogInfo("EcalTPGWeightGroupHandler") << "max_since : " << max_since;

    edm::LogInfo("EcalTPGWeightGroupHandler") << "retrieved last payload ";

    // here we retrieve all the runs after the last from online DB

    edm::LogInfo("EcalTPGWeightGroupHandler") << "Retrieving run list from ONLINE DB ... ";

    edm::LogInfo("EcalTPGWeightGroupHandler") << "Making connection...";
    econn = new EcalCondDBInterface(m_sid, m_user, m_pass);
    edm::LogInfo("EcalTPGWeightGroupHandler") << "Done.";

    if (!econn) {
      edm::LogInfo(" connection parameters ") << m_sid << "/" << m_user;
      //	    cerr << e.what();
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

    readFromFile("last_tpg_weightGroup_settings.txt");

    unsigned int min_run;

    if (m_firstRun < m_i_run_number) {
      min_run = m_i_run_number + 1;
    } else {
      min_run = m_firstRun;
    }

    if (min_run < max_since) {
      min_run = max_since + 1;  // we have to add 1 to the last transferred one
    }

    edm::LogInfo("m_i_run_number") << m_i_run_number << "m_firstRun " << m_firstRun << "max_since " << max_since;

    unsigned int max_run = m_lastRun;
    edm::LogInfo("EcalTPGWeightGroupHandler") << "min_run= " << min_run << " max_run= " << max_run;

    RunList my_list;
    my_list = econn->fetchGlobalRunListByLocation(my_runtag, min_run, max_run, my_locdef);
    //        my_list=econn->fetchRunListByLocation(my_runtag,min_run,max_run,my_locdef);

    std::vector<RunIOV> run_vec = my_list.getRuns();
    size_t num_runs = run_vec.size();

    edm::LogInfo("number of runs is : ") << num_runs;

    unsigned int irun = 0;
    if (num_runs > 0) {
      // going to query the ecal logic id
      std::vector<EcalLogicID> my_StripEcalLogicId_EE;
      my_StripEcalLogicId_EE =
          econn->getEcalLogicIDSetOrdered("ECAL_readout_strip", 1, 2000, 1, 70, 0, 5, "EE_offline_stripid", 123);

      edm::LogInfo(" GOT the logic ID for the EE trigger strips ");

      for (size_t kr = 0; kr < run_vec.size(); kr++) {
        irun = static_cast<unsigned int>(run_vec[kr].getRunNumber());

        edm::LogInfo(" **************** ");
        edm::LogInfo(" **************** ");
        edm::LogInfo(" run= ") << irun;

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

        edm::LogInfo(" run= ") << irun << " tag " << the_config_tag << " version=" << the_config_version;

        // here we should check if it is the same as previous run.

        if ((the_config_tag != m_i_tag || the_config_version != m_i_version) && nr > 0) {
          edm::LogInfo("the tag is different from last transferred run ... retrieving last config set from DB");

          FEConfigMainInfo fe_main_info;
          fe_main_info.setConfigTag(the_config_tag);
          fe_main_info.setVersion(the_config_version);

          try {
            edm::LogInfo(" before fetch config set");
            econn->fetchConfigSet(&fe_main_info);
            edm::LogInfo(" after fetch config set");

            // now get TPGWeightGroup
            int wId = fe_main_info.getWeiId();

            if (wId != m_i_weightGroup) {
              FEConfigWeightInfo fe_w_info;
              fe_w_info.setId(wId);
              econn->fetchConfigSet(&fe_w_info);
              std::map<EcalLogicID, FEConfigWeightDat> dataset_TpgW;
              econn->fetchDataSet(&dataset_TpgW, &fe_w_info);

              EcalTPGWeightGroup* weightG = new EcalTPGWeightGroup;
              typedef std::map<EcalLogicID, FEConfigWeightDat>::const_iterator CIfesli;
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
                    edm::LogInfo(" these may be the additional towers TCC/TT ") << id1 << "/" << id2;
                  }
                }
              }

              Time_t snc = (Time_t)irun;

              m_to_transfer.push_back(std::make_pair((EcalTPGWeightGroup*)weightG, snc));

              m_i_run_number = irun;
              m_i_tag = the_config_tag;
              m_i_version = the_config_version;
              m_i_weightGroup = wId;

              writeFile("last_tpg_weightGroup_settings.txt");

            } else {
              m_i_run_number = irun;
              m_i_tag = the_config_tag;
              m_i_version = the_config_version;

              writeFile("last_tpg_weightGroup_settings.txt");

              edm::LogInfo(
                  " even if the tag/version is not the same, the weight group id is the same -> no transfer needed ");
            }

          }

          catch (std::exception& e) {
            edm::LogInfo("ERROR: THIS CONFIG DOES NOT EXIST: tag=")
                << the_config_tag << " version=" << the_config_version;
            edm::LogInfo("Exception") << e.what();
            m_i_run_number = irun;
          }
          edm::LogInfo(" **************** ");

        } else if (nr == 0) {
          m_i_run_number = irun;
          edm::LogInfo(" no tag saved to RUN_TPGCONFIG_DAT by EcalSupervisor -> no transfer needed ");
          edm::LogInfo(" **************** ");
        } else {
          m_i_run_number = irun;
          m_i_tag = the_config_tag;
          m_i_version = the_config_version;
          edm::LogInfo(" the tag/version is the same -> no transfer needed ");
          edm::LogInfo(" **************** ");
          writeFile("last_tpg_weightGroup_settings.txt");
        }
      }
    }

    delete econn;
  }  // usual way
  edm::LogInfo("EcalTPGWeightGroupHandler") << "Ecal - > end of getNewObjects -----------";
}

void popcon::EcalTPGWeightGroupHandler::readtxtFile() {
  edm::LogInfo(" reading the input file ") << m_file_name;
  std::ifstream fInput;
  fInput.open(m_file_name);
  if (!fInput.is_open()) {
    edm::LogInfo("ERROR : cannot open file ") << m_file_name;
    exit(1);
  }
  int weightGroup, stripEBId, stripEEId;
  EcalTPGWeightGroup* weightG = new EcalTPGWeightGroup;
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
    m_to_transfer.push_back(std::make_pair((EcalTPGWeightGroup*)weightG, snc));
  } catch (std::exception& e) {
    edm::LogInfo("EcalTPGWeightGroupHandler::readtxtFile error : ") << e.what();
  }
  edm::LogInfo(" **************** ");
}

void popcon::EcalTPGWeightGroupHandler::readxmlFile() {
  edm::LogInfo(" reading the input file ") << m_file_name;
  std::ifstream fxml;
  fxml.open(m_file_name);
  if (!fxml.is_open()) {
    edm::LogInfo("ERROR : cannot open file ") << m_file_name;
    exit(1);
  }
  std::string dummyLine, bid;
  int weightGroup, stripEBId, stripEEId;
  EcalTPGWeightGroup* weightG = new EcalTPGWeightGroup;
  for (int i = 0; i < 6; i++)
    std::getline(fxml, dummyLine);  // skip first lines
  fxml >> bid;
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
    m_to_transfer.push_back(std::make_pair((EcalTPGWeightGroup*)weightG, snc));
  } catch (std::exception& e) {
    edm::LogInfo("EcalTPGWeightGroupHandler::readtxtFile error : ") << e.what();
  }
  edm::LogInfo(" **************** ");
}

void popcon::EcalTPGWeightGroupHandler::readFromFile(const char* inputFile) {
  //-------------------------------------------------------------

  m_i_tag = "";
  m_i_version = 0;
  m_i_run_number = 0;
  m_i_weightGroup = 0;

  FILE* inpFile;  // input file
  inpFile = fopen(inputFile, "r");
  if (!inpFile) {
    edm::LogError("EcalTPGWeightGroupHandler") << "*** Can not open file: " << inputFile;
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
  m_i_weightGroup = atoi(line);
  str << "weightGroup_config= " << m_i_weightGroup << std::endl;

  fclose(inpFile);  // close inp. file
}

void popcon::EcalTPGWeightGroupHandler::writeFile(const char* inputFile) {
  //-------------------------------------------------------------

  std::ofstream myfile;
  myfile.open(inputFile);
  myfile << m_i_tag << std::endl;
  myfile << m_i_version << std::endl;
  myfile << m_i_run_number << std::endl;
  myfile << m_i_weightGroup << std::endl;

  myfile.close();
}
