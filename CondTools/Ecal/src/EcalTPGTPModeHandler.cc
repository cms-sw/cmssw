#include "CondTools/Ecal/interface/EcalLaserHandler.h"

#include "CondTools/Ecal/interface/EcalTPGTPModeHandler.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/RunTPGConfigDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigMainInfo.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigSlidingInfo.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigSlidingDat.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"

#include <iostream>
#include <fstream>
#include <map>

#include <ctime>
#include <unistd.h>

#include <string>
#include <cstdio>
#include <typeinfo>
#include <sstream>

popcon::EcalTPGTPModeHandler::EcalTPGTPModeHandler(const edm::ParameterSet& ps)
    : m_name(ps.getUntrackedParameter<std::string>("name", "EcalTPGTPModeHandler")) {
  edm::LogInfo("EcalTPGTPModeHandler") << "Ecal TPG TPMode Source handler constructor.";
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

  edm::LogInfo("EcalTPGTPModeHandler") << m_sid << "/" << m_user << "/" << m_location << "/" << m_gentag;
}

popcon::EcalTPGTPModeHandler::~EcalTPGTPModeHandler() {}

void popcon::EcalTPGTPModeHandler::getNewObjects() {
  if (m_file_type == "txt") {
    readtxtFile();
  } else {
    edm::LogVerbatim("EcalTPGTPModeHandler") << "Started GetNewObjects!!!";

    //check whats already inside of database
    if (tagInfo().size) {
      //check whats already inside of database
      edm::LogVerbatim("EcalTPGTPModeHandler") << "got offlineInfo = ";
      edm::LogVerbatim("EcalTPGTPModeHandler") << "tag name = " << tagInfo().name;
      edm::LogVerbatim("EcalTPGTPModeHandler") << "size = " << tagInfo().size;
    } else {
      edm::LogVerbatim("EcalTPGTPModeHandler") << " First object for this tag ";
    }

    unsigned int max_since = 0;
    max_since = static_cast<unsigned int>(tagInfo().lastInterval.since);
    edm::LogVerbatim("EcalTPGTPModeHandler") << "max_since : " << max_since;

    edm::LogVerbatim("EcalTPGTPModeHandler") << "retrieved last payload ";

    // here we retrieve all the runs after the last from online DB

    edm::LogVerbatim("EcalTPGTPModeHandler") << "Retrieving run list from ONLINE DB ... ";

    edm::LogVerbatim("EcalTPGTPModeHandler") << "Making connection...";
    auto econn = std::make_shared<EcalCondDBInterface>(m_sid, m_user, m_pass);
    edm::LogVerbatim("EcalTPGTPModeHandler") << "Done.";

    if (!econn) {
      throw cms::Exception("OMDS not available") << " connection parameters " << m_sid << "/" << m_user;
    }

    LocationDef my_locdef;
    my_locdef.setLocation(m_location);

    RunTypeDef my_rundef;
    my_rundef.setRunType(m_runtype);

    RunTag my_runtag;
    my_runtag.setLocationDef(my_locdef);
    my_runtag.setRunTypeDef(my_rundef);
    my_runtag.setGeneralTag(m_gentag);

    readFromFile("last_tpg_TPMode_settings.txt");

    unsigned int min_run;

    if (m_firstRun < m_i_run_number) {
      min_run = m_i_run_number + 1;
    } else {
      min_run = m_firstRun;
    }

    if (min_run < max_since) {
      min_run = max_since + 1;  // we have to add 1 to the last transferred one
    }

    edm::LogVerbatim("EcalTPGTPModeHandler")
        << "m_i_run_number" << m_i_run_number << "m_firstRun " << m_firstRun << "max_since " << max_since;

    unsigned int max_run = m_lastRun;
    edm::LogVerbatim("EcalTPGTPModeHandler") << "min_run= " << min_run << " max_run= " << max_run;

    RunList my_list;
    my_list = econn->fetchGlobalRunListByLocation(my_runtag, min_run, max_run, my_locdef);

    std::vector<RunIOV> run_vec = my_list.getRuns();
    size_t num_runs = run_vec.size();

    edm::LogVerbatim("EcalTPGTPModeHandler") << "number of runs is : " << num_runs;

    unsigned int irun = 0;
    if (num_runs > 0) {
      for (size_t kr = 0; kr < run_vec.size(); kr++) {
        irun = static_cast<unsigned int>(run_vec[kr].getRunNumber());
        edm::LogVerbatim("EcalTPGTPModeHandler") << " **************** ";
        edm::LogVerbatim("EcalTPGTPModeHandler") << " **************** ";
        edm::LogVerbatim("EcalTPGTPModeHandler") << " run= " << irun;

        // retrieve the data :
        std::map<EcalLogicID, RunTPGConfigDat> dataset;
        econn->fetchDataSet(&dataset, &run_vec[kr]);

        std::string the_config_tag = "";
        int the_config_version = 0;

        std::map<EcalLogicID, RunTPGConfigDat>::const_iterator it;

        int nr = 0;
        for (it = dataset.begin(); it != dataset.end(); it++) {
          ++nr;
          EcalLogicID ecalid = it->first;
          RunTPGConfigDat dat = it->second;
          the_config_tag = dat.getConfigTag();
          the_config_version = dat.getVersion();
        }

        // it is all the same for all SM... get the last one
        edm::LogVerbatim("EcalTPGTPModeHandler")
            << " run= " << irun << " tag " << the_config_tag << " version=" << the_config_version;

        // here we should check if it is the same as previous run.

        if ((the_config_tag != m_i_tag || the_config_version != m_i_version) && nr > 0) {
          edm::LogVerbatim("EcalTPGTPModeHandler")
              << "the tag is different from last transferred run ... retrieving last config set from DB";

          FEConfigMainInfo fe_main_info;
          fe_main_info.setConfigTag(the_config_tag);
          fe_main_info.setVersion(the_config_version);

          try {
            edm::LogVerbatim("EcalTPGTPModeHandler") << " before fetch config set";
            econn->fetchConfigSet(&fe_main_info);
            edm::LogVerbatim("EcalTPGTPModeHandler") << " after fetch config set";

            // now get TPGTPMode
            int wId = fe_main_info.getWei2Id();
            if (wId != m_i_TPMode) {
              FEConfigOddWeightInfo fe_odd_weight_info;
              fe_odd_weight_info.setId(wId);
              econn->fetchConfigSet(&fe_odd_weight_info);
              std::map<EcalLogicID, FEConfigOddWeightModeDat> dataset_mode;
              econn->fetchDataSet(&dataset_mode, &fe_odd_weight_info);

              typedef std::map<EcalLogicID, FEConfigOddWeightModeDat>::const_iterator CIfem;
              FEConfigOddWeightModeDat rd_mode;

              int rd_modev[19] = {0};
              int k = 0;
              for (CIfem p = dataset_mode.begin(); p != dataset_mode.end(); p++) {
                rd_mode = p->second;
                rd_modev[0] = rd_mode.getEnableEBOddFilter();
                rd_modev[1] = rd_mode.getEnableEEOddFilter();
                rd_modev[2] = rd_mode.getEnableEBOddPeakFinder();
                rd_modev[3] = rd_mode.getEnableEEOddPeakFinder();
                rd_modev[4] = rd_mode.getDisableEBEvenPeakFinder();
                rd_modev[5] = rd_mode.getDisableEEEvenPeakFinder();
                rd_modev[6] = rd_mode.getFenixEBStripOutput();
                rd_modev[7] = rd_mode.getFenixEEStripOutput();
                rd_modev[8] = rd_mode.getFenixEBStripInfobit2();
                rd_modev[9] = rd_mode.getFenixEEStripInfobit2();
                rd_modev[10] = rd_mode.getFenixEBTcpOutput();
                rd_modev[11] = rd_mode.getFenixEBTcpInfobit1();
                rd_modev[12] = rd_mode.getFenixEETcpOutput();
                rd_modev[13] = rd_mode.getFenixEETcpInfobit1();

                edm::LogVerbatim("EcalTPGTPModeHandler") << "here is the value for the weight mode: ";
                edm::LogVerbatim("EcalTPGTPModeHandler") << " EnableEBOddFilter:" << rd_modev[0];
                edm::LogVerbatim("EcalTPGTPModeHandler") << " EnableEEOddFilter:" << rd_modev[1];
                edm::LogVerbatim("EcalTPGTPModeHandler") << " EnableEBOddPeakFinder:" << rd_modev[2];
                edm::LogVerbatim("EcalTPGTPModeHandler") << " EnableEEOddPeakFinder:" << rd_modev[3];
                edm::LogVerbatim("EcalTPGTPModeHandler") << " DisableEBEvenPeakFinder:" << rd_modev[4];
                edm::LogVerbatim("EcalTPGTPModeHandler") << " DisableEEEvenPeakFinder:" << rd_modev[5];
                edm::LogVerbatim("EcalTPGTPModeHandler") << " FenixEBStripOutput:" << rd_modev[6];
                edm::LogVerbatim("EcalTPGTPModeHandler") << " FenixEEStripOutput:" << rd_modev[7];
                edm::LogVerbatim("EcalTPGTPModeHandler") << " FenixEBStripInfobit2:" << rd_modev[8];
                edm::LogVerbatim("EcalTPGTPModeHandler") << " FenixEEStripInfobit2:" << rd_modev[9];
                edm::LogVerbatim("EcalTPGTPModeHandler") << " FenixEBTcpOutput:" << rd_modev[10];
                edm::LogVerbatim("EcalTPGTPModeHandler") << " FenixEBTcpinfobit1:" << rd_modev[11];
                edm::LogVerbatim("EcalTPGTPModeHandler") << " FenixEETcpOutput:" << rd_modev[12];
                edm::LogVerbatim("EcalTPGTPModeHandler") << " FenixEETcpinfobit1:" << rd_modev[13];
                k = k + 1;
              }

              edm::LogVerbatim("EcalTPGTPModeHandler") << "*****************************************";
              edm::LogVerbatim("EcalTPGTPModeHandler") << "read done " << wId;
              edm::LogVerbatim("EcalTPGTPModeHandler") << "*****************************************";

              EcalTPGTPMode* tpMode = new EcalTPGTPMode;
              tpMode->EnableEBOddFilter = rd_modev[0];
              tpMode->EnableEEOddFilter = rd_modev[1];
              tpMode->EnableEBOddPeakFinder = rd_modev[2];
              tpMode->EnableEEOddPeakFinder = rd_modev[3];
              tpMode->DisableEBEvenPeakFinder = rd_modev[4];
              tpMode->DisableEEEvenPeakFinder = rd_modev[5];
              tpMode->FenixEBStripOutput = rd_modev[6];
              tpMode->FenixEEStripOutput = rd_modev[7];
              tpMode->FenixEBStripInfobit2 = rd_modev[8];
              tpMode->FenixEEStripInfobit2 = rd_modev[9];
              tpMode->EBFenixTcpOutput = rd_modev[10];
              tpMode->EBFenixTcpInfobit1 = rd_modev[11];
              tpMode->EEFenixTcpOutput = rd_modev[12];
              tpMode->EEFenixTcpInfobit1 = rd_modev[13];
              tpMode->FenixPar15 = 0;
              tpMode->FenixPar16 = 0;
              tpMode->FenixPar17 = 0;
              tpMode->FenixPar18 = 0;
              Time_t snc = (Time_t)irun;
              m_to_transfer.push_back(std::make_pair((EcalTPGTPMode*)tpMode, snc));

              m_i_run_number = irun;
              m_i_tag = the_config_tag;
              m_i_version = the_config_version;
              m_i_TPMode = wId;

              writeFile("last_tpg_TPMode_settings.txt");

            } else {
              m_i_run_number = irun;
              m_i_tag = the_config_tag;
              m_i_version = the_config_version;

              writeFile("last_tpg_TPMode_settings.txt");

              edm::LogVerbatim("EcalTPGTPModeHandler")
                  << " even if the tag/version is not the same, the weight group id is the same -> no transfer needed ";
            }

          }

          catch (std::exception& e) {
            throw cms::Exception("FileReadError") << "ERROR: THIS CONFIG DOES NOT EXIST: tag=" << the_config_tag
                                                  << " version=" << the_config_version << "\n"
                                                  << e.what();
          }
          edm::LogVerbatim("EcalTPGTPModeHandler") << " **************** ";

        } else if (nr == 0) {
          m_i_run_number = irun;
          edm::LogVerbatim("EcalTPGTPModeHandler")
              << " no tag saved to RUN_TPGCONFIG_DAT by EcalSupervisor -> no transfer needed ";
          edm::LogVerbatim("EcalTPGTPModeHandler") << " **************** ";
        } else {
          m_i_run_number = irun;
          m_i_tag = the_config_tag;
          m_i_version = the_config_version;
          edm::LogVerbatim("EcalTPGTPModeHandler") << " the tag/version is the same -> no transfer needed ";
          edm::LogVerbatim("EcalTPGTPModeHandler") << " **************** ";
          writeFile("last_tpg_TPMode_settings.txt");
        }
      }
    }

  }  // usual way
  edm::LogVerbatim("EcalTPGTPModeHandler") << "Ecal - > end of getNewObjects -----------";
}

void popcon::EcalTPGTPModeHandler::readtxtFile() {
  edm::LogVerbatim("EcalTPGTPModeHandler") << " reading the input file " << m_file_name;
  std::ifstream fInput;
  fInput.open(m_file_name);
  if (!fInput.is_open()) {
    throw cms::Exception("FileOpenError") << "ERROR : cannot open file " << m_file_name;
  }
  std::map<std::string, int> values;
  EcalTPGTPMode* tpMode = new EcalTPGTPMode;

  std::string key;
  int value;
  while (fInput.good()) {
    fInput >> key >> value;
    values[key] = value;
  }

  try {
    tpMode->EnableEBOddFilter = values["EnableEBOddFilter"];
    tpMode->EnableEEOddFilter = values["EnableEEOddFilter"];
    tpMode->EnableEBOddPeakFinder = values["EnableEBOddPeakFinder"];
    tpMode->EnableEEOddPeakFinder = values["EnableEEOddPeakFinder"];
    tpMode->DisableEBEvenPeakFinder = values["DisableEBEvenPeakFinder"];
    tpMode->DisableEEEvenPeakFinder = values["DisableEEEvenPeakFinder"];
    tpMode->FenixEBStripOutput = values["FenixEBStripOutput"];
    tpMode->FenixEEStripOutput = values["FenixEEStripOutput"];
    tpMode->FenixEBStripInfobit2 = values["FenixEBStripInfobit2"];
    tpMode->FenixEEStripInfobit2 = values["FenixEEStripInfobit2"];
    tpMode->EBFenixTcpOutput = values["EBFenixTcpOutput"];
    tpMode->EBFenixTcpInfobit1 = values["EBFenixTcpInfobit1"];
    tpMode->EEFenixTcpOutput = values["EEFenixTcpOutput"];
    tpMode->EEFenixTcpInfobit1 = values["EEFenixTcpInfobit1"];

    Time_t snc = (Time_t)m_firstRun;
    m_to_transfer.push_back(std::make_pair((EcalTPGTPMode*)tpMode, snc));

  } catch (std::exception& e) {
    throw cms::Exception("FileReadError") << "EcalTPGTPModeHandler::readtxtFile error : " << e.what();
  }
  edm::LogVerbatim("EcalTPGTPModeHandler") << " **************** ";
}

void popcon::EcalTPGTPModeHandler::readFromFile(const char* inputFile) {
  m_i_tag = "";
  m_i_version = 0;
  m_i_run_number = 0;
  m_i_TPMode = 0;

  FILE* inpFile;  // input file
  inpFile = fopen(inputFile, "r");
  if (!inpFile) {
    edm::LogError("EcalTPGTPModeHandler") << "*** Can not open file: " << inputFile;
    return;
  }

  char line[256];

  std::ostringstream str;

  fgets(line, 255, inpFile);
  m_i_tag = to_string(line);
  str << "gen tag " << m_i_tag << std::endl;

  fgets(line, 255, inpFile);
  m_i_version = atoi(line);
  str << "version= " << m_i_version << std::endl;

  fgets(line, 255, inpFile);
  m_i_run_number = atoi(line);
  str << "run_number= " << m_i_run_number << std::endl;

  fgets(line, 255, inpFile);
  m_i_TPMode = atoi(line);
  str << "TPMode_config= " << m_i_TPMode << std::endl;

  fclose(inpFile);  // close inp. file
}

void popcon::EcalTPGTPModeHandler::writeFile(const char* inputFile) {
  std::ofstream myfile;
  myfile.open(inputFile);
  myfile << m_i_tag << std::endl;
  myfile << m_i_version << std::endl;
  myfile << m_i_run_number << std::endl;
  myfile << m_i_TPMode << std::endl;

  myfile.close();
}
