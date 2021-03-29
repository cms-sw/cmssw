#include "CondTools/Ecal/interface/EcalLaserHandler.h"

#include "CondTools/Ecal/interface/EcalTPGOddWeightIdMapHandler.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/RunTPGConfigDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigMainInfo.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigWeightInfo.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <fstream>

#include <ctime>
#include <unistd.h>

#include <string>
#include <cstdio>
#include <typeinfo>
#include <sstream>

popcon::EcalTPGOddWeightIdMapHandler::EcalTPGOddWeightIdMapHandler(const edm::ParameterSet& ps)
    : m_name(ps.getUntrackedParameter<std::string>("name", "EcalTPGOddWeightIdMapHandler")) {
  edm::LogInfo("EcalTPGOddWeightIdMapHandler") << "EcalTPGOddWeightIdMap Source handler constructor";
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

  edm::LogInfo("EcalTPGOddWeightIdMapHandler") << m_sid << "/" << m_user << "/" << m_location << "/" << m_gentag;
}

popcon::EcalTPGOddWeightIdMapHandler::~EcalTPGOddWeightIdMapHandler() {}

void popcon::EcalTPGOddWeightIdMapHandler::getNewObjects() {
  if (m_file_type == "txt") {
    readtxtFile();
  } else if (m_file_type == "xml") {
    readxmlFile();
  } else {
    edm::LogInfo("EcalTPGOddWeightIdMapHandler") << "Started GetNewObjects!!!";

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
    edm::LogInfo("EcalTPGOddWeightIdMapHandler") << "max_since : " << max_since;
    //Ref weightIdMap_db = lastPayload();

    edm::LogInfo("EcalTPGOddWeightIdMapHandler") << "retrieved last payload ";

    // here we retrieve all the runs after the last from online DB
    edm::LogInfo("EcalTPGOddWeightIdMapHandler") << "Retrieving run list from ONLINE DB ... ";

    edm::LogInfo("EcalTPGOddWeightIdMapHandler") << "Making connection...";
    econn = new EcalCondDBInterface(m_sid, m_user, m_pass);
    edm::LogInfo("EcalTPGOddWeightIdMapHandler") << "Done.";

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

    readFromFile("last_tpg_OddweightIdMap_settings.txt");

    unsigned int min_run = m_i_run_number + 1;

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
    edm::LogInfo("EcalTPGOddWeightIdMapHandler") << "min_run= " << min_run << "max_run= " << max_run;

    RunList my_list;
    my_list = econn->fetchGlobalRunListByLocation(my_runtag, min_run, max_run, my_locdef);
    //        my_list=econn->fetchRunListByLocation(my_runtag,min_run,max_run,my_locdef);

    std::vector<RunIOV> run_vec = my_list.getRuns();
    size_t num_runs = run_vec.size();
    edm::LogInfo("EcalTPGOddWeightIdMapHandler") << "number of Mon runs is : " << num_runs;

    unsigned int irun;
    if (num_runs > 0) {
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

            // now get TPGOddWeightIdMap
            int weightId = fe_main_info.getWei2Id();

            if (weightId != m_i_oddweightIdMap) {
              FEConfigOddWeightInfo fe_odd_weight_info;
              fe_odd_weight_info.setId(weightId);
              econn->fetchConfigSet(&fe_odd_weight_info);
              std::map<EcalLogicID, FEConfigOddWeightGroupDat> dataset_TpgWeight;
              econn->fetchDataSet(&dataset_TpgWeight, &fe_odd_weight_info);
              edm::LogInfo("EcalTPGOddWeightIdMapHandler") << "Got object!";
              EcalTPGOddWeightIdMap* weightMap = new EcalTPGOddWeightIdMap;
              typedef std::map<EcalLogicID, FEConfigOddWeightGroupDat>::const_iterator CIfeweight;
              EcalLogicID ecid_xt;
              FEConfigOddWeightGroupDat rd_w;

              int igroups = 0;
              for (CIfeweight p = dataset_TpgWeight.begin(); p != dataset_TpgWeight.end(); p++) {
                rd_w = p->second;
                // EB and EE data
                EcalTPGWeights w;
                unsigned int weight0 = static_cast<unsigned int>(rd_w.getWeight4());
                unsigned int weight1 = static_cast<unsigned int>(rd_w.getWeight3());
                unsigned int weight2 = static_cast<unsigned int>(rd_w.getWeight2());
                unsigned int weight3 = static_cast<unsigned int>(rd_w.getWeight1() - 0x80);
                unsigned int weight4 = static_cast<unsigned int>(rd_w.getWeight0());

                w.setValues(weight0, weight1, weight2, weight3, weight4);
                weightMap->setValue(rd_w.getWeightGroupId(), w);

                ++igroups;
              }

              edm::LogInfo("EcalTPGOddWeightIdMapHandler") << "found " << igroups << "Weight groups";

              Time_t snc = (Time_t)irun;
              m_to_transfer.push_back(std::make_pair((EcalTPGOddWeightIdMap*)weightMap, snc));

              m_i_run_number = irun;
              m_i_tag = the_config_tag;
              m_i_version = the_config_version;
              m_i_oddweightIdMap = weightId;

              writeFile("last_tpg_OddweightIdMap_settings.txt");

            } else {
              m_i_run_number = irun;
              m_i_tag = the_config_tag;
              m_i_version = the_config_version;

              writeFile("last_tpg_OddweightIdMap_settings.txt");

              std::cout
                  << " even if the tag/version is not the same, the weightIdMap id is the same -> no transfer needed "
                  << std::endl;
            }

          } catch (std::exception& e) {
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
          writeFile("last_tpg_OddweightIdMap_settings.txt");
        }
      }
    }

    delete econn;
  }  // usual way
  edm::LogInfo("EcalTPGOddWeightIdMapHandler") << "Ecal - > end of getNewObjects -----------";
}
void popcon::EcalTPGOddWeightIdMapHandler::readtxtFile() {
  std::cout << " reading the input file " << m_file_name << std::endl;
  std::ifstream fInput;
  fInput.open(m_file_name);
  if (!fInput.is_open()) {
    std::cout << "ERROR : cannot open file " << m_file_name << std::endl;
    exit(1);
  }
  unsigned int wloc[5];
  EcalTPGWeights w;
  EcalTPGOddWeightIdMap* weightMap = new EcalTPGOddWeightIdMap;
  int igroups = 0;
  std::string line;
  while (!fInput.eof()) {
    getline(fInput, line);
    if (!line.empty()) {
      std::stringstream ss;
      ss << line;
      ss >> wloc[0] >> wloc[1] >> wloc[2] >> wloc[3] >> wloc[4];
      //      std::cout <<  wloc[0] << " " <<  wloc[1] << " " <<  wloc[2] << " " <<  wloc[3] << " " <<  wloc[4] << std::endl;
      w.setValues(wloc[0], wloc[1], wloc[2], wloc[3], wloc[4]);
      weightMap->setValue(igroups, w);
      igroups++;
    }
  }
  edm::LogInfo("EcalTPGOddWeightIdMapHandler") << "found " << igroups << " Weight groups";
  try {
    Time_t snc = (Time_t)m_firstRun;
    m_to_transfer.push_back(std::make_pair((EcalTPGOddWeightIdMap*)weightMap, snc));
  } catch (std::exception& e) {
    std::cout << "EcalTPGOddWeightIdMapHandler::readtxtFile error : " << e.what() << std::endl;
  }
  std::cout << " **************** " << std::endl;
}

void popcon::EcalTPGOddWeightIdMapHandler::readxmlFile() {
  std::cout << " reading the input file " << m_file_name << std::endl;
  std::ifstream fxml;
  fxml.open(m_file_name);
  if (!fxml.is_open()) {
    std::cout << "ERROR : cannot open file " << m_file_name << std::endl;
    exit(1);
  }
  std::string dummyLine, bid;
  unsigned int wloc[5];
  EcalTPGWeights w;
  EcalTPGOddWeightIdMap* weightMap = new EcalTPGOddWeightIdMap;
  int ngroups, igroups;
  for (int i = 0; i < 5; i++)
    std::getline(fxml, dummyLine);  // skip first lines
  // get the Weight group number
  fxml >> bid;
  std::string stt = bid.substr(7, 1);
  std::istringstream sc(stt);
  sc >> ngroups;
  edm::LogInfo("EcalTPGOddWeightIdMapHandler") << "found " << ngroups << " Weight groups";
  for (int i = 0; i < 2; i++)
    std::getline(fxml, dummyLine);  //    <item_version>0</item_version>
  //  std::cout << dummyLine << std::endl;
  for (int i = 0; i < ngroups; i++) {
    std::getline(fxml, dummyLine);  //    <item
    //    std::cout  << " group " << i << " first line " << dummyLine << std::endl;
    fxml >> bid;  //    <first
    std::size_t found = bid.find("</");
    stt = bid.substr(7, found - 7);
    std::istringstream sg1(stt);
    sg1 >> igroups;
    if (igroups != i) {
      std::cout << " group " << i << ": " << bid << " igroups " << igroups << std::endl;
      exit(-1);
    }
    for (int i = 0; i < 2; i++)
      std::getline(fxml, dummyLine);  // < second
    for (int i = 0; i < 5; i++) {
      fxml >> bid;
      found = bid.find("</");
      stt = bid.substr(5, found - 5);
      std::istringstream w(stt);
      w >> wloc[i];
    }
    w.setValues(wloc[0], wloc[1], wloc[2], wloc[3], wloc[4]);
    weightMap->setValue(igroups, w);
    for (int i = 0; i < 3; i++)
      std::getline(fxml, dummyLine);  //    </item>
    //    std::cout << " group " << i << " last line " << dummyLine << std::endl;
  }
  try {
    Time_t snc = (Time_t)m_firstRun;
    m_to_transfer.push_back(std::make_pair((EcalTPGOddWeightIdMap*)weightMap, snc));
  } catch (std::exception& e) {
    std::cout << "EcalTPGOddWeightIdMapHandler::readxmlFile error : " << e.what() << std::endl;
  }
  std::cout << " **************** " << std::endl;
}

void popcon::EcalTPGOddWeightIdMapHandler::readFromFile(const char* inputFile) {
  //-------------------------------------------------------------

  m_i_tag = "";
  m_i_version = 0;
  m_i_run_number = 0;
  m_i_oddweightIdMap = 0;

  FILE* inpFile;  // input file
  inpFile = fopen(inputFile, "r");
  if (!inpFile) {
    edm::LogError("EcalTPGOddWeightIdMapHandler") << "*** Can not open file: " << inputFile;
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
  m_i_oddweightIdMap = atoi(line);
  str << "weightIdMap_config= " << m_i_oddweightIdMap << std::endl;

  fclose(inpFile);  // close inp. file
}

void popcon::EcalTPGOddWeightIdMapHandler::writeFile(const char* inputFile) {
  //-------------------------------------------------------------

  std::ofstream myfile;
  myfile.open(inputFile);
  myfile << m_i_tag << std::endl;
  myfile << m_i_version << std::endl;
  myfile << m_i_run_number << std::endl;
  myfile << m_i_oddweightIdMap << std::endl;

  myfile.close();
}
