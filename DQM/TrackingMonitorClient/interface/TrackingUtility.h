#ifndef TrackingUtility_H
#define TrackingUtility_H

/** \class TrackingUtility
 * *
 *  Class that handles the Tracking Quality Tests
 * *
  */

#include <vector>
#include <fstream>
#include <string>
#include <map>
#include <stdint.h>

#include "DQMServices/Core/interface/DQMStore.h"

class MonitorElement;
class TrackerTopology;
class TrackingUtility
{
 public:
 
 static int getMEList(std::string name, std::vector<std::string>& values);
 static bool checkME(std::string element, std::string name, std::string& full_path);
 static int getMEList(std::string name, std::string& dir_path, std::vector<std::string>& me_names);

 static void split(const std::string& str, std::vector<std::string>& tokens, 
             const std::string& delimiters=" ");
 static void getMEStatusColor(int status, int& rval, int&gval, int& bval);
 static void getMEStatusColor(int status, int& icol, std::string& tag);
 static int getMEStatus(MonitorElement* me);
 static int getMEStatus(MonitorElement* me, int& bad_channels);
 static void getModuleFolderList(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, std::vector<std::string>& m_ids);
 static void getMEValue(MonitorElement* me, std::string & val);
 static bool goToDir(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, std::string name);
 static void setBadModuleFlag(std::string & hname, uint16_t& flg);
 static void getBadModuleStatus(uint16_t flag, std::string& message);
 static void getTopFolderPath(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, std::string top_dir, std::string& path);   
};

#endif
