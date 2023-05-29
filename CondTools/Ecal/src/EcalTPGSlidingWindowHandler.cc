#include "CondTools/Ecal/interface/EcalLaserHandler.h"

#include "CondTools/Ecal/interface/EcalTPGSlidingWindowHandler.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/RunTPGConfigDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigMainInfo.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigSlidingInfo.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigSlidingDat.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/EcalObjects/interface/EcalTPGSlidingWindow.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include <iostream>
#include <fstream>

#include <ctime>
#include <unistd.h>

#include <string>
#include <cstdio>
#include <typeinfo>
#include <sstream>

popcon::EcalTPGSlidingWindowHandler::EcalTPGSlidingWindowHandler(const edm::ParameterSet& ps)
    : m_name(ps.getUntrackedParameter<std::string>("name", "EcalTPGSlidingWindowHandler")) {
  edm::LogInfo("EcalTPGSlidingWindowHandler") << "EcalTPGSlidingWindow Source handler constructor";
  m_firstRun = static_cast<unsigned int>(atoi(ps.getParameter<std::string>("firstRun").c_str()));
  m_lastRun = static_cast<unsigned int>(atoi(ps.getParameter<std::string>("lastRun").c_str()));
  m_sid = ps.getParameter<std::string>("OnlineDBSID");
  m_user = ps.getParameter<std::string>("OnlineDBUser");
  m_pass = ps.getParameter<std::string>("OnlineDBPassword");
  m_locationsource = ps.getParameter<std::string>("LocationSource");
  m_location = ps.getParameter<std::string>("Location");
  m_gentag = ps.getParameter<std::string>("GenTag");
  m_runtype = ps.getParameter<std::string>("RunType");

  edm::LogInfo("EcalTPGSlidingWindowHandler") << m_sid << "/" << m_user << "/" << m_location << "/" << m_gentag;
}

popcon::EcalTPGSlidingWindowHandler::~EcalTPGSlidingWindowHandler() {}

void popcon::EcalTPGSlidingWindowHandler::getNewObjects() {
  edm::LogInfo("EcalTPGSlidingWindowHandler") << "Started GetNewObjects!!!";

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
  edm::LogInfo("EcalTPGSlidingWindowHandler") << "max_since : " << max_since;
  edm::LogInfo("EcalTPGSlidingWindowHandler") << "retrieved last payload ";

  // here we retrieve all the runs after the last from online DB

  edm::LogInfo("EcalTPGSlidingWindowHandler") << "Retrieving run list from ONLINE DB ... ";

  edm::LogInfo("EcalTPGSlidingWindowHandler") << "Making connection...";
  econn = new EcalCondDBInterface(m_sid, m_user, m_pass);
  edm::LogInfo("EcalTPGSlidingWindowHandler") << "Done.";

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

  readFromFile("last_tpg_sliding_settings.txt");

  unsigned int min_run;
  if (m_firstRun < m_i_run_number) {
    min_run = m_i_run_number + 1;  // we have to add 1 to the last transferred one
  } else {
    min_run = m_firstRun;
  }

  if (min_run < max_since) {
    min_run = max_since + 1;  // we have to add 1 to the last transferred one
  }

  std::cout << "m_i_run_number" << m_i_run_number << "m_firstRun " << m_firstRun << "max_since " << max_since
            << std::endl;

  unsigned int max_run = m_lastRun;

  edm::LogInfo("EcalTPGSlidingWindowHandler") << "min_run= " << min_run << "max_run= " << max_run;

  RunList my_list;
  my_list = econn->fetchGlobalRunListByLocation(my_runtag, min_run, max_run, my_locdef);
  //        my_list=econn->fetchRunListByLocation(my_runtag,min_run,max_run, my_locdef);

  std::vector<RunIOV> run_vec = my_list.getRuns();
  size_t num_runs = run_vec.size();

  std::cout << "number of runs is : " << num_runs << std::endl;

  unsigned int irun = 0;
  if (num_runs > 0) {
    // going to query the ecal logic id
    std::vector<EcalLogicID> my_StripEcalLogicId_EE;
    my_StripEcalLogicId_EE =
        econn->getEcalLogicIDSetOrdered("ECAL_readout_strip", 1, 1000, 1, 70, 0, 5, "EE_offline_stripid", 123);
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
      FEConfigMainInfo fe_main_info;

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

          // now get TPGSlidingWindow
          int slidingId = fe_main_info.getSliId();

          if (slidingId != m_i_sliding) {
            FEConfigSlidingInfo fe_sli_info;
            fe_sli_info.setId(slidingId);
            econn->fetchConfigSet(&fe_sli_info);
            std::map<EcalLogicID, FEConfigSlidingDat> dataset_TpgSli;
            econn->fetchDataSet(&dataset_TpgSli, &fe_sli_info);

            EcalTPGSlidingWindow* sliW = new EcalTPGSlidingWindow;
            typedef std::map<EcalLogicID, FEConfigSlidingDat>::const_iterator CIfesli;
            EcalLogicID ecid_xt;
            FEConfigSlidingDat rd_sli;

            for (CIfesli p = dataset_TpgSli.begin(); p != dataset_TpgSli.end(); p++) {
              ecid_xt = p->first;
              rd_sli = p->second;

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

                sliW->setValue(stripEBId, (unsigned int)rd_sli.getSliding());

              } else if (ecid_name == "ECAL_readout_strip") {
                // EE data
                int id1 = ecid_xt.getID1();  // dcc
                int id2 = ecid_xt.getID2();  // ccu
                int id3 = ecid_xt.getID3();  // strip

                bool set_the_strip = false;
                int stripid;
                for (size_t istrip = 0; istrip < my_StripEcalLogicId_EE.size(); istrip++) {
                  if (!set_the_strip) {
                    if (my_StripEcalLogicId_EE[istrip].getID1() == id1 &&
                        my_StripEcalLogicId_EE[istrip].getID2() == id2 &&
                        my_StripEcalLogicId_EE[istrip].getID3() == id3) {
                      stripid = my_StripEcalLogicId_EE[istrip].getLogicID();
                      set_the_strip = true;
                      break;
                    }
                  }
                }

                if (set_the_strip) {
                  sliW->setValue(stripid, (unsigned int)rd_sli.getSliding());

                } else {
                  std::cout << " these may be the additional towers TCC/TT " << id1 << "/" << id2 << std::endl;
                }
              }
            }

            Time_t snc = (Time_t)irun;
            m_to_transfer.push_back(std::make_pair((EcalTPGSlidingWindow*)sliW, snc));

            m_i_run_number = irun;
            m_i_tag = the_config_tag;
            m_i_version = the_config_version;
            m_i_sliding = slidingId;

            writeFile("last_tpg_sliding_settings.txt");

          } else {
            m_i_run_number = irun;
            m_i_tag = the_config_tag;
            m_i_version = the_config_version;

            writeFile("last_tpg_sliding_settings.txt");

            std::cout
                << " even if the tag/version is not the same, the sliding windows id is the same -> no transfer needed "
                << std::endl;
          }
        } catch (std::exception& e) {
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
        writeFile("last_tpg_sliding_settings.txt");
      }
    }
  }

  delete econn;

  edm::LogInfo("EcalTPGSlidingWindowHandler") << "Ecal - > end of getNewObjects -----------";
}

void popcon::EcalTPGSlidingWindowHandler::readFromFile(const char* inputFile) {
  //-------------------------------------------------------------

  m_i_tag = "";
  m_i_version = 0;
  m_i_run_number = 0;
  m_i_sliding = 0;

  FILE* inpFile;  // input file
  inpFile = fopen(inputFile, "r");
  if (!inpFile) {
    edm::LogError("EcalTPGSlidingWindowHandler") << "*** Can not open file: " << inputFile;
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
  m_i_sliding = atoi(line);
  str << "sliding_config= " << m_i_sliding << std::endl;

  fclose(inpFile);  // close inp. file
}

void popcon::EcalTPGSlidingWindowHandler::writeFile(const char* inputFile) {
  //-------------------------------------------------------------

  std::ofstream myfile;
  myfile.open(inputFile);
  myfile << m_i_tag << std::endl;
  myfile << m_i_version << std::endl;
  myfile << m_i_run_number << std::endl;
  myfile << m_i_sliding << std::endl;

  myfile.close();
}
