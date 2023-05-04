#include "CondTools/Ecal/interface/EcalTPGBadStripHandler.h"

#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/RunTPGConfigDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigMainInfo.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigBadStripInfo.h"
#include "OnlineDB/EcalCondDB/interface/RunList.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/EcalObjects/interface/EcalTPGStripStatus.h"

#include <iostream>
#include <fstream>

#include <ctime>
#include <unistd.h>

#include <string>
#include <cstdio>
#include <typeinfo>
#include <sstream>

popcon::EcalTPGBadStripHandler::EcalTPGBadStripHandler(const edm::ParameterSet& ps)
    : m_name(ps.getUntrackedParameter<std::string>("name", "EcalTPGBadStripHandler")) {
  edm::LogInfo("EcalTPGBadStripHandler") << "EcalTPGStripStatus Source handler constructor.";
  m_firstRun = static_cast<unsigned int>(atoi(ps.getParameter<std::string>("firstRun").c_str()));
  m_lastRun = static_cast<unsigned int>(atoi(ps.getParameter<std::string>("lastRun").c_str()));
  m_sid = ps.getParameter<std::string>("OnlineDBSID");
  m_user = ps.getParameter<std::string>("OnlineDBUser");
  m_pass = ps.getParameter<std::string>("OnlineDBPassword");
  m_locationsource = ps.getParameter<std::string>("LocationSource");
  m_location = ps.getParameter<std::string>("Location");
  m_gentag = ps.getParameter<std::string>("GenTag");
  m_runtype = ps.getParameter<std::string>("RunType");

  edm::LogInfo("EcalTPGBadStripHandler") << m_sid << "/" << m_user << "/" << m_location << "/" << m_gentag;
}

popcon::EcalTPGBadStripHandler::~EcalTPGBadStripHandler() {}

void popcon::EcalTPGBadStripHandler::getNewObjects() {
  edm::LogInfo("EcalTPGBadStripHandler") << "Started GetNewObjects!!!";

  unsigned int max_since = 0;
  max_since = static_cast<unsigned int>(tagInfo().lastInterval.since);
  edm::LogInfo("EcalTPGBadStripHandler") << "max_since : " << max_since;
  edm::LogInfo("EcalTPGBadStripHandler") << "retrieved last payload ";

  // here we retrieve all the runs after the last from online DB
  edm::LogInfo("EcalTPGBadStripHandler") << "Retrieving run list from ONLINE DB ... ";

  edm::LogInfo("EcalTPGBadStripHandler") << "Making connection...";
  econn = new EcalCondDBInterface(m_sid, m_user, m_pass);
  edm::LogInfo("EcalTPGBadStripHandler") << "Done.";

  if (!econn) {
    std::cout << " connection parameters " << m_sid << "/" << m_user << std::endl;
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

  readFromFile("last_tpg_badStrip_settings.txt");

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
  edm::LogInfo("EcalTPGBadStripHandler") << "min_run= " << min_run << "max_run= " << max_run;

  RunList my_list;
  my_list = econn->fetchGlobalRunListByLocation(my_runtag, min_run, max_run, my_locdef);
  //my_list=econn->fetchRunListByLocation(my_runtag, min_run, max_run, my_locdef);

  std::vector<RunIOV> run_vec = my_list.getRuns();
  size_t num_runs = run_vec.size();

  std::cout << "number of runs is : " << num_runs << std::endl;

  std::string str = "";

  unsigned int irun = 0;

  if (num_runs > 0) {
    // going to query the ecal logic id
    std::vector<EcalLogicID> my_StripEcalLogicId_EE;
    my_StripEcalLogicId_EE = econn->getEcalLogicIDSetOrdered("EE_trigger_strip",
                                                             1,
                                                             1000,  //"TCC"
                                                             1,
                                                             100,  //tower
                                                             0,
                                                             5,  //strip
                                                             "EE_offline_stripid",
                                                             123);

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
        std::cout << " run= " << irun << " tag " << the_config_tag << " version=" << the_config_version << std::endl;
        std::cout << "the tag is different from last transferred run ... retrieving last config set from DB"
                  << std::endl;

        FEConfigMainInfo fe_main_info;
        fe_main_info.setConfigTag(the_config_tag);
        fe_main_info.setVersion(the_config_version);

        try {
          econn->fetchConfigSet(&fe_main_info);

          // now get TPGStripStatus
          int badstripId = fe_main_info.getBstId();

          if (badstripId != m_i_badStrip) {
            FEConfigBadStripInfo fe_badStrip_info;
            fe_badStrip_info.setId(badstripId);

            econn->fetchConfigSet(&fe_badStrip_info);

            std::vector<FEConfigBadStripDat> dataset_TpgBadStrip;

            econn->fetchConfigDataSet(&dataset_TpgBadStrip, &fe_badStrip_info);

            EcalTPGStripStatus* stripStatus = new EcalTPGStripStatus;
            typedef std::vector<FEConfigBadStripDat>::const_iterator CIfeped;

            FEConfigBadStripDat rd_badStrip;
            //unsigned int  rd_stripStatus;

            // put at 1 the strip that are bad
            for (CIfeped p = dataset_TpgBadStrip.begin(); p != dataset_TpgBadStrip.end(); p++) {
              rd_badStrip = *p;

              //int fed_num=rd_badStrip.getFedId();
              int tcc_num = rd_badStrip.getTCCId();
              int tt_num = rd_badStrip.getTTId();
              int strip_num = rd_badStrip.getStripId();

              //std::cout << fed_num << " " << tcc_num << " " << tt_num << " " << strip_num << std::endl;

              // EE data
              int stripid;

              bool set_the_strip = false;
              for (size_t istrip = 0; istrip < my_StripEcalLogicId_EE.size(); istrip++) {
                if (!set_the_strip) {
                  if (my_StripEcalLogicId_EE[istrip].getID1() == tcc_num &&
                      my_StripEcalLogicId_EE[istrip].getID2() == tt_num &&
                      my_StripEcalLogicId_EE[istrip].getID3() == strip_num) {
                    stripid = my_StripEcalLogicId_EE[istrip].getLogicID();

                    set_the_strip = true;
                    break;
                  }
                }
              }

              if (set_the_strip) {
                stripStatus->setValue(stripid, (unsigned int)rd_badStrip.getStatus());
              }
            }

            edm::LogInfo("EcalTPGBadStripHandler") << "Finished badStrip reading.";

            Time_t snc = (Time_t)irun;

            m_to_transfer.push_back(std::make_pair((EcalTPGStripStatus*)stripStatus, snc));

            m_i_run_number = irun;
            m_i_tag = the_config_tag;
            m_i_version = the_config_version;
            m_i_badStrip = badstripId;

            writeFile("last_tpg_badStrip_settings.txt");

          } else {
            m_i_run_number = irun;
            m_i_tag = the_config_tag;
            m_i_version = the_config_version;

            writeFile("last_tpg_badStrip_settings.txt");

            //  std::cout<< " even if the tag/version is not the same, the badStrip id is the same -> no transfer needed "<< std::endl;
          }

        }

        catch (std::exception& e) {
          std::cout << "ERROR: THIS CONFIG DOES NOT EXIST: tag=" << the_config_tag << " version=" << the_config_version
                    << std::endl;
          std::cout << e.what() << std::endl;
          m_i_run_number = irun;
        }

      } else if (nr == 0) {
        m_i_run_number = irun;
        //	      std::cout<< " no tag saved to RUN_TPGCONFIG_DAT by EcalSupervisor -> no transfer needed "<< std::endl;
      } else {
        m_i_run_number = irun;
        m_i_tag = the_config_tag;
        m_i_version = the_config_version;

        writeFile("last_tpg_badStrip_settings.txt");
      }
    }
  }

  delete econn;

  edm::LogInfo("EcalTPGBadStripHandler") << "Ecal - > end of getNewObjects -----------";
}

void popcon::EcalTPGBadStripHandler::readFromFile(const char* inputFile) {
  //-------------------------------------------------------------

  m_i_tag = "";
  m_i_version = 0;
  m_i_run_number = 0;
  m_i_badStrip = 0;

  FILE* inpFile;  // input file
  inpFile = fopen(inputFile, "r");
  if (!inpFile) {
    edm::LogError("EcalTPGBadStripHandler") << "*** Can not open file: " << inputFile;
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
  m_i_badStrip = atoi(line);
  str << "badTT_config= " << m_i_badStrip << std::endl;

  fclose(inpFile);  // close inp. file
}

void popcon::EcalTPGBadStripHandler::writeFile(const char* inputFile) {
  //-------------------------------------------------------------

  std::ofstream myfile;
  myfile.open(inputFile);
  myfile << m_i_tag << std::endl;
  myfile << m_i_version << std::endl;
  myfile << m_i_run_number << std::endl;
  myfile << m_i_badStrip << std::endl;

  myfile.close();
}
