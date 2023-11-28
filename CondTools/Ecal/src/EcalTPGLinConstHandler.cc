#include "CondTools/Ecal/interface/EcalTPGLinConstHandler.h"

#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/RunTPGConfigDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigMainInfo.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigLUTInfo.h"
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

popcon::EcalTPGLinConstHandler::EcalTPGLinConstHandler(const edm::ParameterSet &ps)
    : m_name(ps.getUntrackedParameter<std::string>("name", "EcalTPGLinConstHandler")) {
  edm::LogInfo("EcalTPGLinConstHandler") << "EcalTPGLinConst Source handler constructor";
  m_firstRun = static_cast<unsigned int>(atoi(ps.getParameter<std::string>("firstRun").c_str()));
  m_lastRun = static_cast<unsigned int>(atoi(ps.getParameter<std::string>("lastRun").c_str()));
  m_sid = ps.getParameter<std::string>("OnlineDBSID");
  m_user = ps.getParameter<std::string>("OnlineDBUser");
  m_pass = ps.getParameter<std::string>("OnlineDBPassword");
  m_locationsource = ps.getParameter<std::string>("LocationSource");
  m_location = ps.getParameter<std::string>("Location");
  m_gentag = ps.getParameter<std::string>("GenTag");
  m_runtype = ps.getParameter<std::string>("RunType");

  edm::LogInfo("EcalTPGLinConstHandler") << m_sid << "/" << m_user << "/" << m_location << "/" << m_gentag;
}

popcon::EcalTPGLinConstHandler::~EcalTPGLinConstHandler() {}

void popcon::EcalTPGLinConstHandler::getNewObjects() {
  edm::LogInfo("EcalTPGLinConstHandler") << "Started getNewObjects";

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
  edm::LogInfo("EcalTPGLinConstHandler") << "max_since = " << max_since;
  edm::LogInfo("EcalTPGLinConstHandler") << "Retrieved last payload ";

  // here we retrieve all the runs after the last from online DB
  edm::LogInfo("EcalTPGLinConstHandler") << "Retrieving run list from ONLINE DB ... " << std::endl;

  edm::LogInfo("EcalTPGLinConstHandler") << "Making connection..." << std::flush;
  econn = new EcalCondDBInterface(m_sid, m_user, m_pass);
  edm::LogInfo("EcalTPGLinConstHandler") << "Done." << std::endl;

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

  readFromFile("last_tpg_lin_settings.txt");

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
  edm::LogInfo("EcalTPGLinConstHandler") << "min_run=  " << min_run << "max_run = " << max_run;

  RunList my_list;
  my_list = econn->fetchGlobalRunListByLocation(my_runtag, min_run, max_run, my_locdef);
  //    	my_list=econn->fetchRunListByLocation(my_runtag, min_run, max_run, my_locdef);

  std::vector<RunIOV> run_vec = my_list.getRuns();
  size_t num_runs = run_vec.size();

  std::cout << "number of runs is : " << num_runs << std::endl;

  unsigned int irun = 0;
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

          // now get TPGLinConst
          int linId = fe_main_info.getLinId();

          if (linId != m_i_lin) {
            FEConfigLinInfo fe_lin_info;
            fe_lin_info.setId(linId);
            econn->fetchConfigSet(&fe_lin_info);
            std::map<EcalLogicID, FEConfigLinDat> dataset_TpgLin;
            econn->fetchDataSet(&dataset_TpgLin, &fe_lin_info);

            EcalTPGLinearizationConst *linC = new EcalTPGLinearizationConst;
            typedef std::map<EcalLogicID, FEConfigLinDat>::const_iterator CIfelin;
            EcalLogicID ecid_xt;
            FEConfigLinDat rd_lin;

            for (CIfelin p = dataset_TpgLin.begin(); p != dataset_TpgLin.end(); p++) {
              ecid_xt = p->first;
              rd_lin = p->second;
              std::string ecid_name = ecid_xt.getName();

              //EB data
              if (ecid_name == "EB_crystal_number") {
                int sm_num = ecid_xt.getID1();
                int xt_num = ecid_xt.getID2();
                EBDetId ebdetid(sm_num, xt_num, EBDetId::SMCRYSTALMODE);

                EcalTPGLinearizationConst::Item item;
                item.mult_x1 = rd_lin.getMultX1();
                item.mult_x6 = rd_lin.getMultX6();
                item.mult_x12 = rd_lin.getMultX12();
                item.shift_x1 = rd_lin.getShift1();
                item.shift_x6 = rd_lin.getShift6();
                item.shift_x12 = rd_lin.getShift12();

                linC->insert(std::make_pair(ebdetid.rawId(), item));
              } else {
                //EE data
                int z = ecid_xt.getID1();
                int x = ecid_xt.getID2();
                int y = ecid_xt.getID3();
                EEDetId eedetid(x, y, z, EEDetId::XYMODE);

                EcalTPGLinearizationConst::Item item;

                item.mult_x1 = rd_lin.getMultX1();
                item.mult_x6 = rd_lin.getMultX6();
                item.mult_x12 = rd_lin.getMultX12();
                item.shift_x1 = rd_lin.getShift1();
                item.shift_x6 = rd_lin.getShift6();
                item.shift_x12 = rd_lin.getShift12();

                linC->insert(std::make_pair(eedetid.rawId(), item));
              }
            }

            Time_t snc = (Time_t)irun;
            m_to_transfer.push_back(std::make_pair((EcalTPGLinearizationConst *)linC, snc));

            m_i_run_number = irun;
            m_i_tag = the_config_tag;
            m_i_version = the_config_version;
            m_i_lin = linId;

            writeFile("last_tpg_lin_settings.txt");

          } else {
            m_i_run_number = irun;
            m_i_tag = the_config_tag;
            m_i_version = the_config_version;

            writeFile("last_tpg_lin_settings.txt");

            std::cout << " even if the tag/version is not the same, the linearization constants id is the same -> no "
                         "transfer needed "
                      << std::endl;
          }
        } catch (std::exception &e) {
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
        writeFile("last_tpg_lin_settings.txt");
      }
    }
  }

  delete econn;

  edm::LogInfo("EcalTPGLinConstHandler") << "Ecal - > end of getNewObjects -----------";
}

void popcon::EcalTPGLinConstHandler::readFromFile(const char *inputFile) {
  //-------------------------------------------------------------

  m_i_tag = "";
  m_i_version = 0;
  m_i_run_number = 0;
  m_i_lin = 0;

  FILE *inpFile;  // input file
  inpFile = fopen(inputFile, "r");
  if (!inpFile) {
    edm::LogError("EcalTPGLinConstHandler") << "*** Can not open file: " << inputFile;
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
  m_i_lin = atoi(line);
  str << "lin_config= " << m_i_lin << std::endl;

  fclose(inpFile);  // close inp. file
}

void popcon::EcalTPGLinConstHandler::writeFile(const char *inputFile) {
  //-------------------------------------------------------------

  std::ofstream myfile;
  myfile.open(inputFile);
  myfile << m_i_tag << std::endl;
  myfile << m_i_version << std::endl;
  myfile << m_i_run_number << std::endl;
  myfile << m_i_lin << std::endl;

  myfile.close();
}
