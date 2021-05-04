#ifndef PixelConfigFile_h
#define PixelConfigFile_h
/*! \file CalibFormats/SiPixelObjects/interface/PixelConfigFile.h
*   \brief This class implements..
*
*   OK, first this is not a DB; this class will try to
*   define an interface to accessing the configuration data.
*/

#include "CalibFormats/SiPixelObjects/interface/PixelConfigBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelConfigAlias.h"
#include "CalibFormats/SiPixelObjects/interface/PixelConfig.h"
#include "CalibFormats/SiPixelObjects/interface/PixelConfigList.h"
#include "CalibFormats/SiPixelObjects/interface/PixelAliasList.h"
#include "CalibFormats/SiPixelObjects/interface/PixelVersionAlias.h"
#include "CalibFormats/SiPixelObjects/interface/PixelCalibBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelConfigKey.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTrimBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTrimAllPixels.h"
#include "CalibFormats/SiPixelObjects/interface/PixelMaskBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelMaskAllPixels.h"
#include "CalibFormats/SiPixelObjects/interface/PixelDACSettings.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTBMSettings.h"
#include "CalibFormats/SiPixelObjects/interface/PixelDetectorConfig.h"
#include "CalibFormats/SiPixelObjects/interface/PixelLowVoltageMap.h"
#include "CalibFormats/SiPixelObjects/interface/PixelMaxVsf.h"
#include "CalibFormats/SiPixelObjects/interface/PixelNameTranslation.h"
#include "CalibFormats/SiPixelObjects/interface/PixelFEDCard.h"
#include "CalibFormats/SiPixelObjects/interface/PixelCalibConfiguration.h"
#include "CalibFormats/SiPixelObjects/interface/PixelPortCardConfig.h"
#include "CalibFormats/SiPixelObjects/interface/PixelPortcardMap.h"
#include "CalibFormats/SiPixelObjects/interface/PixelDelay25Calib.h"
#include "CalibFormats/SiPixelObjects/interface/PixelFECConfig.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTKFECConfig.h"
#include "CalibFormats/SiPixelObjects/interface/PixelFEDConfig.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTTCciConfig.h"
#include "CalibFormats/SiPixelObjects/interface/PixelLTCConfig.h"
#include "CalibFormats/SiPixelObjects/interface/PixelFEDTestDAC.h"
#include "CalibFormats/SiPixelObjects/interface/PixelGlobalDelay25.h"
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <sstream>
#include <typeinfo>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define DEBUG_CF_ 0

namespace pos {
  /*! \class PixelConfigFile PixelConfigFile.h "interface/PixelConfigFile.h"
*
*
*   OK, first this is not a DB; this class will try to
*   define an interface to accessing the configuration data.
*/
  class PixelConfigFile {
  public:
    static std::vector<std::pair<std::string, unsigned int> > getAliases() {
      PixelAliasList& aliases = getAlias();
      std::vector<std::pair<std::string, unsigned int> > tmp;
      for (unsigned int i = 0; i < aliases.nAliases(); i++) {
        std::pair<std::string, unsigned int> apair(aliases.name(i), aliases.key(i));
        tmp.push_back(apair);
      }
      return tmp;
    }

    static std::vector<std::string> getVersionAliases(std::string path) { return getAlias().getVersionAliases(path); }

    static bool getVersionAliases(std::string configAlias,
                                  unsigned int& key,
                                  std::vector<std::pair<std::string, std::string> >& versionAliases) {
      PixelConfigAlias* alias = getAlias().versionAliases(configAlias);
      if (alias == 0) {
        return false;
      }
      key = alias->key();
      versionAliases = alias->versionAliases();
      return true;
    }

    static std::map<std::string, unsigned int> getAliases_map() {
      PixelAliasList& aliases = getAlias();
      std::map<std::string, unsigned int> tmp;
      for (unsigned int i = 0; i < aliases.nAliases(); i++) {
        tmp.insert(make_pair(aliases.name(i), aliases.key(i)));
      }
      return tmp;
    }

    static PixelConfigList& configList() {
      static PixelConfigList theConfigList = getConfig();
      return theConfigList;
    }

    static unsigned int getVersion(std::string path, std::string alias) { return getAlias().getVersion(path, alias); }

    // Added by Dario, May 20th, 2008 =====================================================================
    static pos::pathVersionAliasMmap getVersionData() { return getAlias().getVersionData(); }
    static pos::pathVersionAliasMmap getVersionData(std::string koc) { return getAlias().getVersionData(koc); }

    static std::vector<pathAliasPair> getConfigAliases(std::string path) { return getAlias().getConfigAliases(path); }
    // End of Dario's addition ============================================================================

    static void addAlias(std::string alias, unsigned int key) {
      PixelConfigAlias anAlias(alias, key);
      getAlias().insertAlias(anAlias);
      getAlias().writefile();
    }

    static void addAlias(std::string alias,
                         unsigned int key,
                         std::vector<std::pair<std::string, std::string> > versionaliases) {
      PixelConfigAlias anAlias(alias, key);
      for (unsigned int i = 0; i < versionaliases.size(); i++) {
        anAlias.addVersionAlias(versionaliases[i].first, versionaliases[i].second);
      }
      getAlias().insertAlias(anAlias);
      getAlias().writefile();
    }

    static std::vector<std::pair<std::string, unsigned int> > getVersions(pos::PixelConfigKey key) {
      static PixelConfigList& configs = getConfig();
      PixelConfig& theConfig = configs[key.key()];
      return theConfig.versions();
    }

    static void addVersionAlias(std::string path, unsigned int version, std::string alias) {
      PixelConfigList& configs = getConfig();

      PixelVersionAlias anAlias(path, version, alias);
      getAlias().insertVersionAlias(anAlias);
      getAlias().updateConfigAlias(path, version, alias, configs);
      getAlias().writefile();
      configs.writefile();
    }

    static unsigned int makeKey(std::vector<std::pair<std::string, unsigned int> > versions) {
      PixelConfig config;

      for (unsigned int i = 0; i < versions.size(); i++) {
        config.add(versions[i].first, versions[i].second);
      }

      PixelConfigList& configs = getConfig();

      unsigned int newkey = configs.add(config);

      configs.writefile();

      return newkey;
    }

    static PixelConfigList& getConfig() {
      static PixelConfigList configs;

      //FIXME

      static int counter = 0;

      if (counter != 0) {
        while (counter != 0) {
          std::cout << __LINE__
                    << "]\t[PixelConfigFile::getConfig()]\t\t\t\t    Waiting for other thread to complete reading"
                    << std::endl;
          ::sleep(1);
        }
        return configs;
      }

      counter++;

      static std::string directory;
      static int first = 1;

      directory = std::getenv("PIXELCONFIGURATIONBASE");
      std::string filename = directory + "/configurations.txt";
      /* Don't know who put this snippet of code here: this case is already contemplated in the 'else' of the 'if' statement below
      if(!first)
	{
 	  configs.reload(filename) 
	}
*/
      if (first) {
        first = 0;
        configs.readfile(filename);
        forceConfigReload(false);
      } else {
        //	  if( getForceConfigReload() ) {
        configs.reload(filename);
        forceConfigReload(false);
        //	  }
      }

      counter--;

      return configs;
    }

    static PixelAliasList& getAlias() {
      static std::string directory;
      static int first = 1;

      static PixelAliasList aliases;

      directory = std::getenv("PIXELCONFIGURATIONBASE");
      std::string filename = directory + "/aliases.txt";

      if (first) {
        first = 0;
        aliases.readfile(filename);

        forceAliasesReload(false);
      } else {
        if (getForceAliasesReload()) {
          aliases.readfile(filename);
          forceAliasesReload(false);
        }
      }

      return aliases;
    }

    static void forceAliasesReload(bool m) {
      if (getForceAliasesReload() != m) {
        getForceAliasesReload() = m;
      }
    }

    static void forceConfigReload(bool m) {
      if (getForceConfigReload() != m) {
        getForceConfigReload() = m;
      }
    }

    //Returns the path the the configuration data.
    static std::string getPath(std::string path, PixelConfigKey key) {
      unsigned int theKey = key.key();

      assert(theKey <= getConfig().size());

      unsigned int last = path.find_last_of("/");
      assert(last != (unsigned int)std::string::npos);

      std::string base = path.substr(0, last);
      std::string ext = path.substr(last + 1);

      unsigned int slashpos = base.find_last_of("/");
      if (slashpos == (unsigned int)std::string::npos) {
        std::cout << "[pos::PixelConfigFile::getPath()]\t\t\tOn path:" << path << std::endl;
        std::cout << "[pos::PixelConfigFile::getPath()]\t\t\tRecall that you need a trailing /" << std::endl;
        ::abort();
      }

      std::string dir = base.substr(slashpos + 1);

      //      std::cout << "[pos::PixelConfigFile::get()]\t\t\tExtracted dir:" <<dir <<std::endl;
      //      std::cout << "[pos::PixelConfigFile::get()]\t\t\tExtracted base:"<<base<<std::endl;
      //      std::cout << "[pos::PixelConfigFile::get()]\t\t\tExtracted ext :"<<ext <<std::endl;

      unsigned int version;
      int err = getConfig()[theKey].find(dir, version);
      // assert(err==0);
      if (0 != err) {
        return "";
      }

      std::ostringstream s1;
      s1 << version;
      std::string strversion = s1.str();

      static std::string directory;
      directory = std::getenv("PIXELCONFIGURATIONBASE");

      std::string fullpath = directory + "/" + dir + "/" + strversion + "/";

      return fullpath;
    }

    //Returns a pointer to the data found in the path with configuration key.
    template <class T>
    static void get(T*& data, std::string path, PixelConfigKey key) {
      unsigned int theKey = key.key();

      if (theKey >= configList().size()) {
        configList() = getConfig();
      }

      assert(theKey <= configList().size());

      unsigned int last = path.find_last_of("/");
      assert(last != (unsigned int)std::string::npos);

      std::string base = path.substr(0, last);
      std::string ext = path.substr(last + 1);

      unsigned int slashpos = base.find_last_of("/");
      if (slashpos == (unsigned int)std::string::npos) {
        std::cout << "[pos::PixelConfigFile::get()]\t\t\tAsking for data of type:" << typeid(data).name() << std::endl;
        std::cout << "[pos::PixelConfigFile::get()]\t\t\tOn path:" << path << std::endl;
        std::cout << "[pos::PixelConfigFile::get()]\t\t\tRecall that you need a trailing /" << std::endl;
        ::abort();
      }

      std::string dir = base.substr(slashpos + 1);

      //      std::cout << "[pos::PixelConfigFile::get()]\t\t\tExtracted dir:" <<dir <<std::endl;
      //      std::cout << "[pos::PixelConfigFile::get()]\t\t\tExtracted base:"<<base<<std::endl;
      //      std::cout << "[pos::PixelConfigFile::get()]\t\t\tExtracted ext :"<<ext <<std::endl;

      unsigned int version = 0;
      int err = configList()[theKey].find(dir, version);
      // assert(err==0);
      if (0 != err) {
        std::cout << "[PixelConfigFile.h::get] error loading config list. " << theKey << " " << dir << " " << version
                  << std::endl;
        data = 0;
        return;
      }

      std::ostringstream s1;
      s1 << version;
      std::string strversion = s1.str();

      static std::string directory;
      directory = std::getenv("PIXELCONFIGURATIONBASE");

      std::string fullpath = directory + "/" + dir + "/" + strversion + "/";

      //std::cout << "Directory for configuration data:"<<fullpath<<std::endl;

      try {
        if (typeid(data) == typeid(PixelTrimBase*)) {
          //std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelTrimBase" << std::endl;
          assert(dir == "trim");
          data = (T*)new PixelTrimAllPixels(fullpath + "ROC_Trims_module_" + ext + ".dat");
          return;
        } else if (typeid(data) == typeid(PixelMaskBase*)) {
          //std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelMaskBase" << std::endl;
          assert(dir == "mask");
          data = (T*)new PixelMaskAllPixels(fullpath + "ROC_Masks_module_" + ext + ".dat");
          return;
        } else if (typeid(data) == typeid(PixelDACSettings*)) {
          //std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelDACSettings" << std::endl;
          assert(dir == "dac");
          data = (T*)new PixelDACSettings(fullpath + "ROC_DAC_module_" + ext + ".dat");
          return;
        } else if (typeid(data) == typeid(PixelTBMSettings*)) {
          //std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelTBMSettings" << std::endl;
          assert(dir == "tbm");
          data = (T*)new PixelTBMSettings(fullpath + "TBM_module_" + ext + ".dat");
          return;
        } else if (typeid(data) == typeid(PixelDetectorConfig*)) {
          //std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelDetectorConfig" << std::endl;
          assert(dir == "detconfig");
          data = (T*)new PixelDetectorConfig(fullpath + "detectconfig.dat");
          return;
        } else if (typeid(data) == typeid(PixelLowVoltageMap*)) {
          //std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill fetch PixelLowVoltageMap" << std::endl;
          assert(dir == "lowvoltagemap");
          data = (T*)new PixelLowVoltageMap(fullpath + "lowvoltagemap.dat");
          //std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return the PixelLowVoltageMap" << std::endl;
          return;
        } else if (typeid(data) == typeid(PixelMaxVsf*)) {
          //std::cout << "Will fetch PixelMaxVsf" << std::endl;
          assert(dir == "maxvsf");
          data = (T*)new PixelMaxVsf(fullpath + "maxvsf.dat");
          //std::cout << "Will return the PixelMaxVsf" << std::endl;
          return;
        } else if (typeid(data) == typeid(PixelNameTranslation*)) {
          //std::cout << __LINE__ << "]\t[pos::PixelConfigFile::get()]\t\t\t    Will return PixelNameTranslation*" << std::endl;
          assert(dir == "nametranslation");
          data = (T*)new PixelNameTranslation(fullpath + "translation.dat");
          return;
        } else if (typeid(data) == typeid(PixelFEDCard*)) {
          //std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelFEDCard" << std::endl;
          assert(dir == "fedcard");
          //std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill open:"<<fullpath+"params_fed_"+ext+".dat"<< std::endl;
          data = (T*)new PixelFEDCard(fullpath + "params_fed_" + ext + ".dat");
          return;
        } else if (typeid(data) == typeid(PixelCalibBase*)) {
          //std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelCalibBase" << std::endl;
          assert(dir == "calib");
          std::string calibfile = fullpath + "calib.dat";
          //std::cout << "[pos::PixelConfigFile::get()]\t\t\tLooking for file " << calibfile << std::endl;
          std::ifstream calibin(calibfile.c_str());
          if (calibin.good()) {
            data = (T*)new PixelCalibConfiguration(calibfile);
          } else {
            calibfile = fullpath + "delay25.dat";
            //std::cout << "[pos::PixelConfigFile::get()]\t\t\tNow looking for file " << calibfile << std::endl;
            std::ifstream delayin(calibfile.c_str());
            if (delayin.good()) {
              data = (T*)new PixelDelay25Calib(calibfile);
            } else {
              calibfile = fullpath + "fedtestdac.dat";
              //std::cout << "[pos::PixelConfigFile::get()]\t\t\tNow looking for file " << calibfile << std::endl;
              std::ifstream delayin(calibfile.c_str());
              if (delayin.good()) {
                data = (T*)new PixelFEDTestDAC(calibfile);
              } else {
                throw std::runtime_error(
                    "[pos::PixelConfigFile::get()]\t\t\tCan't find calibration file calib.dat or delay25.dat or "
                    "fedtestdac.dat");
              }
            }
          }
          return;
        } else if (typeid(data) == typeid(PixelTKFECConfig*)) {
          //std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelTKFECConfig" << std::endl;
          assert(dir == "tkfecconfig");
          data = (T*)new PixelTKFECConfig(fullpath + "tkfecconfig.dat");
          return;
        } else if (typeid(data) == typeid(PixelFECConfig*)) {
          //std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelFECConfig" << std::endl;
          assert(dir == "fecconfig");
          data = (T*)new PixelFECConfig(fullpath + "fecconfig.dat");
          return;
        } else if (typeid(data) == typeid(PixelFEDConfig*)) {
          //std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelFEDConfig" << std::endl;
          assert(dir == "fedconfig");
          data = (T*)new PixelFEDConfig(fullpath + "fedconfig.dat");
          return;
        } else if (typeid(data) == typeid(PixelPortCardConfig*)) {
          //std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelPortCardConfig" << std::endl;
          assert(dir == "portcard");
          data = (T*)new PixelPortCardConfig(fullpath + "portcard_" + ext + ".dat");
          return;
        } else if (typeid(data) == typeid(PixelPortcardMap*)) {
          //std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelPortcardMap" << std::endl;
          assert(dir == "portcardmap");
          data = (T*)new PixelPortcardMap(fullpath + "portcardmap.dat");
          return;
        } else if (typeid(data) == typeid(PixelDelay25Calib*)) {
          //cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelDelay25Calib" << std::endl;
          assert(dir == "portcard");
          data = (T*)new PixelDelay25Calib(fullpath + "delay25.dat");
          return;
        } else if (typeid(data) == typeid(PixelTTCciConfig*)) {
          //cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelTTCciConfig" << std::endl;
          assert(dir == "ttcciconfig");
          data = (T*)new PixelTTCciConfig(fullpath + "TTCciConfiguration.txt");
          return;
        } else if (typeid(data) == typeid(PixelLTCConfig*)) {
          //cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelLTCConfig" << std::endl;
          assert(dir == "ltcconfig");
          data = (T*)new PixelLTCConfig(fullpath + "LTCConfiguration.txt");
          return;
        } else if (typeid(data) == typeid(PixelGlobalDelay25*)) {
          //std::cout << "[pos::PixelConfigFile::get()]\t\t\tWill return PixelGlobalDelay25" << std::endl;
          assert(dir == "globaldelay25");
          data = (T*)new PixelGlobalDelay25(fullpath + "globaldelay25.dat");
          return;
        } else {
          std::cout << "[pos::PixelConfigFile::get()]\t\t\tNo match" << std::endl;
          assert(0);
        }
      } catch (std::exception& e) {
        std::cout << "[PixelConfigFile::get] Caught exception while constructing configuration object. Will rethrow."
                  << std::endl;
        throw;
      }
    }

    //----- Method added by Dario (March 10, 2008)
    template <class T>
    static bool configurationDataExists(T*& data, std::string path, PixelConfigKey key) {
      std::string mthn = "]\t[pos::PixelConfigFile::configurationDataExists()]\t    ";
      /*       pos::PixelTimeFormatter * timer = new pos::PixelTimeFormatter("PixelConfigFile::ConfigurationDataExists") ; */
      unsigned int theKey = key.key();

      assert(theKey <= configList().size());

      unsigned int last = path.find_last_of("/");
      assert(last != (unsigned int)std::string::npos);

      std::string base = path.substr(0, last);
      std::string ext = path.substr(last + 1);

      unsigned int slashpos = base.find_last_of("/");
      if (slashpos == (unsigned int)std::string::npos) {
        std::cout << __LINE__ << mthn << "Asking for data of type:" << typeid(data).name() << std::endl;
        std::cout << __LINE__ << mthn << "On path:" << path << std::endl;
        std::cout << __LINE__ << mthn << "Recall that you need a trailing /" << std::endl;
        ::abort();
      }

      std::string dir = base.substr(slashpos + 1);
      /*       timer->stopTimer() ; */
      //      std::cout << __LINE__ << mthn << "Extracted dir:"  << dir  <<std::endl;
      //      std::cout << __LINE__ << mthn << "Extracted base:" << base <<std::endl;
      //      std::cout << __LINE__ << mthn << "Extracted ext :" << ext  <<std::endl;

      unsigned int version = 0;
      int err = configList()[theKey].find(dir, version);
      // assert(err==0);
      if (0 != err) {
        data = 0;
        return false;
      }
      /*       timer->stopTimer() ; */
      /*       delete timer ; */

      std::ostringstream s1;
      s1 << version;
      std::string strversion = s1.str();

      static std::string directory;
      directory = std::getenv("PIXELCONFIGURATIONBASE");

      std::string fullpath = directory + "/" + dir + "/" + strversion + "/";

      //std::cout << __LINE__ << mthn << "Directory for configuration data:"<<fullpath<<std::endl;

      std::string fileName;
      if (typeid(data) == typeid(PixelTrimBase*)) {
        fileName = fullpath + "ROC_Trims_module_" + ext + ".dat";
      } else if (typeid(data) == typeid(PixelMaskBase*)) {
        fileName = fullpath + "ROC_Masks_module_" + ext + ".dat";
      } else if (typeid(data) == typeid(PixelDACSettings*)) {
        fileName = fullpath + "ROC_DAC_module_" + ext + ".dat";
      } else if (typeid(data) == typeid(PixelTBMSettings*)) {
        fileName = fullpath + "TBM_module_" + ext + ".dat";
      } else if (typeid(data) == typeid(PixelDetectorConfig*)) {
        fileName = fullpath + "detectconfig.dat";
      } else if (typeid(data) == typeid(PixelLowVoltageMap*)) {
        fileName = fullpath + "lowvoltagemap.dat";
      } else if (typeid(data) == typeid(PixelMaxVsf*)) {
        fileName = fullpath + "maxvsf.dat";
      } else if (typeid(data) == typeid(PixelNameTranslation*)) {
        fileName = fullpath + "translation.dat";
      } else if (typeid(data) == typeid(PixelFEDCard*)) {
        fileName = fullpath + "params_fed_" + ext + ".dat";
      } else if (typeid(data) == typeid(PixelTKFECConfig*)) {
        fileName = fullpath + "tkfecconfig.dat";
      } else if (typeid(data) == typeid(PixelFECConfig*)) {
        fileName = fullpath + "fecconfig.dat";
      } else if (typeid(data) == typeid(PixelFEDConfig*)) {
        fileName = fullpath + "fedconfig.dat";
      } else if (typeid(data) == typeid(PixelPortCardConfig*)) {
        fileName = fullpath + "portcard_" + ext + ".dat";
      } else if (typeid(data) == typeid(PixelPortcardMap*)) {
        fileName = fullpath + "portcardmap.dat";
      } else if (typeid(data) == typeid(PixelDelay25Calib*)) {
        fileName = fullpath + "delay25.dat";
      } else if (typeid(data) == typeid(PixelTTCciConfig*)) {
        fileName = fullpath + "TTCciConfiguration.txt";
      } else if (typeid(data) == typeid(PixelLTCConfig*)) {
        fileName = fullpath + "LTCConfiguration.txt";
      } else if (typeid(data) == typeid(PixelGlobalDelay25*)) {
        fileName = fullpath + "globaldelay25.dat";
      } else if (typeid(data) == typeid(PixelCalibBase*)) {
        assert(dir == "calib");
        std::string calibfile = fullpath + "calib.dat";
        std::ifstream calibin(calibfile.c_str());
        if (calibin.good()) {
          std::cout << __LINE__ << mthn << "Found " << calibfile << std::endl;
          return true;
        } else {
          calibfile = fullpath + "delay25.dat";
          std::ifstream delayin(calibfile.c_str());
          if (delayin.good()) {
            std::cout << __LINE__ << mthn << "Found " << calibfile << std::endl;
            return true;
          } else {
            calibfile = fullpath + "fedtestdac.dat";
            std::ifstream delayin(calibfile.c_str());
            if (delayin.good()) {
              std::cout << __LINE__ << mthn << "Found " << calibfile << std::endl;
              return true;
            } else {
              std::cout << mthn << "Can't find calibration file calib.dat or delay25.dat or fedtestdac.dat"
                        << std::endl;
              return false;
            }
          }
        }
      } else {
        std::cout << __LINE__ << mthn << "No match of class type" << std::endl;
        return false;
      }
      /*
      struct stat * tmp = NULL ;
      if(stat(fileName.c_str(), tmp)==0)
	{
	  std::cout << mthn << "Found(stat) " << fileName << std::endl ; 
	  return true ;
	}
      else
	{
	  std::cout << mthn << "Not found(stat) " << fileName << std::endl ; 
	  return false ;
	}
      */
      std::ifstream in(fileName.c_str());
      if (!in.good()) {
        std::cout << __LINE__ << mthn << "Not found " << fileName << std::endl;
        return false;
      }
      in.close();
      if (DEBUG_CF_)
        std::cout << __LINE__ << mthn << "Found " << fileName << std::endl;
      return true;
    }
    //----- End of method added by Dario (March 10, 2008)

    //Returns a pointer to the data found in the path with configuration key.
    template <class T>
    static void get(T*& data, std::string path, unsigned int version) {
      std::string mthn = "]\t[pos::PixelConfigFile::get()]\t\t\t\t    ";

      unsigned int last = path.find_last_of("/");
      assert(last != (unsigned int)std::string::npos);

      std::string base = path.substr(0, last);
      std::string ext = path.substr(last + 1);

      unsigned int slashpos = base.find_last_of("/");
      //if (slashpos==std::string::npos) {
      //std::cout << __LINE__ << mthn << "Asking for data of type:"          << typeid(data).name() << std::endl;
      //std::cout << __LINE__ << mthn << "On path:"                          << path                << std::endl;
      //std::cout << __LINE__ << mthn << "Recall that you need a trailing /"                        << std::endl;
      //::abort();
      //}

      std::string dir = base.substr(slashpos + 1);

      //std::cout << __LINE__ << mthn << "Extracted dir :" << dir  <<std::endl;
      //std::cout << __LINE__ << mthn << "Extracted base:" << base <<std::endl;
      //std::cout << __LINE__ << mthn << "Extracted ext :" << ext  <<std::endl;

      std::ostringstream s1;
      s1 << version;
      std::string strversion = s1.str();

      static std::string directory;
      directory = std::getenv("PIXELCONFIGURATIONBASE");

      std::string fullpath = directory + "/" + dir + "/" + strversion + "/";

      //std::cout << __LINE__ << mthn << "Directory for configuration data:"<<fullpath<<std::endl;

      if (typeid(data) == typeid(PixelTrimBase*)) {
        //std::cout << __LINE__ << mthn << "Will return PixelTrimBase" << std::endl;
        assert(dir == "trim");
        data = (T*)new PixelTrimAllPixels(fullpath + "ROC_Trims_module_" + ext + ".dat");
        return;
      } else if (typeid(data) == typeid(PixelMaskBase*)) {
        //std::cout << __LINE__ << mthn << "Will return PixelMaskBase" << std::endl;
        assert(dir == "mask");
        data = (T*)new PixelMaskAllPixels(fullpath + "ROC_Masks_module_" + ext + ".dat");
        return;
      } else if (typeid(data) == typeid(PixelDACSettings*)) {
        //std::cout << __LINE__ << mthn << "Will return PixelDACSettings" << std::endl;
        assert(dir == "dac");
        data = (T*)new PixelDACSettings(fullpath + "ROC_DAC_module_" + ext + ".dat");
        return;
      } else if (typeid(data) == typeid(PixelTBMSettings*)) {
        //std::cout << __LINE__ << mthn << "Will return PixelTBMSettings" << std::endl;
        assert(dir == "tbm");
        data = (T*)new PixelTBMSettings(fullpath + "TBM_module_" + ext + ".dat");
        return;
      } else if (typeid(data) == typeid(PixelDetectorConfig*)) {
        //std::cout << __LINE__ << mthn << "Will return PixelDACSettings" << std::endl;
        assert(dir == "detconfig");
        data = (T*)new PixelDetectorConfig(fullpath + "detectconfig.dat");
        return;
      } else if (typeid(data) == typeid(PixelLowVoltageMap*)) {
        //std::cout << __LINE__ << mthn << "Will fetch1 PixelLowVoltageMap" << std::endl;
        assert(dir == "lowvoltagemap");
        data = (T*)new PixelLowVoltageMap(fullpath + "detectconfig.dat");
        //std::cout << __LINE__ << mthn << "Will return1 the PixelLowVoltageMap" << std::endl;
        return;
      } else if (typeid(data) == typeid(PixelMaxVsf*)) {
        //std::cout << __LINE__ << mthn << "Will fetch1 PixelMaxVsf" << std::endl;
        assert(dir == "maxvsf");
        data = (T*)new PixelMaxVsf(fullpath + "maxvsf.dat");
        //std::cout << __LINE__ << mthn << "Will return1 the PixelMaxVsf" << std::endl;
        return;
      } else if (typeid(data) == typeid(PixelNameTranslation*)) {
        //std::cout << __LINE__ << mthn << "Will return PixelNameTranslation" << std::endl;
        assert(dir == "nametranslation");
        data = (T*)new PixelNameTranslation(fullpath + "translation.dat");
        return;
      } else if (typeid(data) == typeid(PixelFEDCard*)) {
        //std::cout << __LINE__ << mthn << "Will return PixelFEDCard" << std::endl;
        assert(dir == "fedcard");
        //std::cout << __LINE__ << mthn << "Will open:"<<fullpath+"params_fed_"+ext+".dat"<< std::endl;
        data = (T*)new PixelFEDCard(fullpath + "params_fed_" + ext + ".dat");
        return;
      } else if (typeid(data) == typeid(PixelCalibBase*)) {
        //std::cout << __LINE__ << mthn << "Will return PixelCalibBase" << std::endl;
        assert(base == "calib");
        std::string calibfile = fullpath + "calib.dat";
        //std::cout << mthn << "Looking for file " << calibfile << std::endl;
        std::ifstream calibin(calibfile.c_str());
        if (calibin.good()) {
          data = (T*)new PixelCalibConfiguration(calibfile);
        } else {
          calibfile = fullpath + "delay25.dat";
          //std::cout << __LINE__ << mthn << "Now looking for file " << calibfile << std::endl;
          std::ifstream delayin(calibfile.c_str());
          if (delayin.good()) {
            data = (T*)new PixelDelay25Calib(calibfile);
          } else {
            calibfile = fullpath + "fedtestdac.dat";
            //std::cout << __LINE__ << mthn << "Now looking for file " << calibfile << std::endl;
            std::ifstream delayin(calibfile.c_str());
            if (delayin.good()) {
              data = (T*)new PixelFEDTestDAC(calibfile);
            } else {
              std::cout << __LINE__ << mthn << "Can't find calibration file calib.dat or delay25.dat or fedtestdac.dat"
                        << std::endl;
              data = 0;
            }
          }
        }
        return;
      } else if (typeid(data) == typeid(PixelTKFECConfig*)) {
        //std::cout << __LINE__ << mthn << "Will return PixelTKFECConfig" << std::endl;
        assert(dir == "tkfecconfig");
        data = (T*)new PixelTKFECConfig(fullpath + "tkfecconfig.dat");
        return;
      } else if (typeid(data) == typeid(PixelFECConfig*)) {
        //std::cout << __LINE__ << mthn << "Will return PixelFECConfig" << std::endl;
        assert(dir == "fecconfig");
        data = (T*)new PixelFECConfig(fullpath + "fecconfig.dat");
        return;
      } else if (typeid(data) == typeid(PixelFEDConfig*)) {
        //std::cout << __LINE__ << mthn << "Will return PixelFEDConfig" << std::endl;
        assert(dir == "fedconfig");
        data = (T*)new PixelFEDConfig(fullpath + "fedconfig.dat");
        return;
      } else if (typeid(data) == typeid(PixelPortCardConfig*)) {
        //std::cout << __LINE__ << mthn << "Will return PixelPortCardConfig" << std::endl;
        assert(dir == "portcard");
        data = (T*)new PixelPortCardConfig(fullpath + "portcard_" + ext + ".dat");
        return;
      } else if (typeid(data) == typeid(PixelPortcardMap*)) {
        //std::cout << __LINE__ << mthn << "Will return PixelPortcardMap" << std::endl;
        assert(dir == "portcardmap");
        data = (T*)new PixelPortcardMap(fullpath + "portcardmap.dat");
        return;
      } else if (typeid(data) == typeid(PixelDelay25Calib*)) {
        //cout << __LINE__ << mthn << "Will return PixelDelay25Calib" << std::endl;
        assert(dir == "portcard");
        data = (T*)new PixelDelay25Calib(fullpath + "delay25.dat");
        return;
      } else if (typeid(data) == typeid(PixelTTCciConfig*)) {
        //cout << __LINE__ << mthn << "Will return PixelTTCciConfig" << std::endl;
        assert(dir == "ttcciconfig");
        data = (T*)new PixelTTCciConfig(fullpath + "TTCciConfiguration.txt");
        return;
      } else if (typeid(data) == typeid(PixelLTCConfig*)) {
        //cout << __LINE__ << mthn << "Will return PixelLTCConfig" << std::endl;
        assert(dir == "ltcconfig");
        data = (T*)new PixelLTCConfig(fullpath + "LTCConfiguration.txt");
        return;
      } else if (typeid(data) == typeid(PixelGlobalDelay25*)) {
        //std::cout << __LINE__ << mthn << "Will return PixelGlobalDelay25" << std::endl;
        assert(dir == "globaldelay25");
        data = (T*)new PixelGlobalDelay25(fullpath + "globaldelay25.dat");
        return;
      } else {
        std::cout << __LINE__ << mthn << "No class match" << std::endl;
        assert(0);
        data = 0;
        return;
      }
    }

    template <class T>
    static void get(std::map<std::string, T*>& pixelObjects, PixelConfigKey key) {
      typename std::map<std::string, T*>::iterator iObject = pixelObjects.begin();

      for (; iObject != pixelObjects.end(); ++iObject) {
        get(iObject->second, iObject->first, key);
      }
    }

    static int makeNewVersion(std::string path, std::string& dir) {
      //std::cout << "[pos::PixelConfigFile::makeNewVersion()]\t\tInserting data on path:"<<path<<std::endl;
      struct stat stbuf;
      std::string directory = std::getenv("PIXELCONFIGURATIONBASE");
      directory += "/";
      directory += path;
      if (stat(directory.c_str(), &stbuf) != 0) {
        std::cout << "[pos::PixelConfigFile::makeNewVersion()]\t\tThe path:" << path << " does not exist." << std::endl;
        std::cout << "[pos::PixelConfigFile::makeNewVersion()]\t\tFull path:" << directory << std::endl;
        return -1;
      }
      directory += "/";
      int version = -1;
      do {
        version++;
        std::ostringstream s1;
        s1 << version;
        std::string strversion = s1.str();
        dir = directory + strversion;
        //std::cout << "[pos::PixelConfigFile::makeNewVersion()]\t\tWill check for version:"<<dir<<std::endl;
      } while (stat(dir.c_str(), &stbuf) == 0);
      //std::cout << "[pos::PixelConfigFile::makeNewVersion()]\t\tThe new version is:"<<version<<std::endl;
      mkdir(dir.c_str(), 0777);
      return version;
    }

    template <class T>
    static int put(const T* object, std::string path) {
      std::string dir;
      int version = makeNewVersion(path, dir);
      object->writeASCII(dir);
      return version;
    }

    template <class T>
    static int put(std::vector<T*> objects, std::string path) {
      std::cout << "[pos::PixelConfigFile::put()]\t\t# of objects to write: " << objects.size() << std::endl;
      std::string dir;
      int version = makeNewVersion(path, dir);
      for (unsigned int i = 0; i < objects.size(); i++) {
        // std::cout << "[pos::PixelConfigFile::put()]\t\t\t\t\tWill write i="<<i<<" ptr: "<<objects[i]<<std::endl;
        objects[i]->writeASCII(dir);
      }
      return version;
    }

  private:
    static bool& getForceAliasesReload() {
      static bool forceAliasesReload = false;
      return forceAliasesReload;
    }
    static bool& getForceConfigReload() {
      static bool forceConfigReload = false;
      return forceConfigReload;
    }
  };

}  // namespace pos
#endif
