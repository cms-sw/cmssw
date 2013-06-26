#ifndef SiStripUtility_H
#define SiStripUtility_H

/** \class SiStripUtility
 * *
 *  Class that handles the SiStrip Quality Tests
 * 
 *  $Date: 2013/01/03 18:59:35 $
 *  $Revision: 1.17 $
 *  \author Suchandra Dutta
  */

#include <vector>
#include <fstream>
#include <string>
#include <map>
#include <stdint.h>

class MonitorElement;
class DQMStore;
class TrackerTopology;
class SiStripUtility
{
 public:
 
 static int getMEList(std::string name, std::vector<std::string>& values);
 static bool checkME(std::string element, std::string name, std::string& full_path);
 static int getMEList(std::string name, std::string& dir_path, std::vector<std::string>& me_names);

 static void split(const std::string& str, std::vector<std::string>& tokens, 
             const std::string& delimiters=" ");
 static void getMEStatusColor(int status, int& rval, int&gval, int& bval);
 static void getDetectorStatusColor(int status, int& rval, int&gval, int& bval);
 static void getMEStatusColor(int status, int& icol, std::string& tag);
 static int getMEStatus(MonitorElement* me);
 static int getMEStatus(MonitorElement* me, int& bad_channels);
 static void getModuleFolderList(DQMStore* dqm_store, std::vector<std::string>& m_ids);
 static void getMEValue(MonitorElement* me, std::string & val);
 static bool goToDir(DQMStore * dqm_store, std::string name);
 static void getSubDetectorTag(uint32_t det_id, std::string& subdet_tag, const TrackerTopology* tTopo);
 static void setBadModuleFlag(std::string & hname, uint16_t& flg);
 static void getBadModuleStatus(uint16_t flag, std::string& message);
 static void getTopFolderPath(DQMStore* dqm_store, std::string type, std::string& path);   
};

#endif
