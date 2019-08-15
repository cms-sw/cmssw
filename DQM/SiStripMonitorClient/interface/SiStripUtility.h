#ifndef SiStripUtility_H
#define SiStripUtility_H

/** \class SiStripUtility
 * *
 *  Class that handles the SiStrip Quality Tests
 *
 *  \author Suchandra Dutta
 */

#include "DQMServices/Core/interface/DQMStore.h"

#include <vector>
#include <fstream>
#include <string>
#include <map>
#include <cstdint>

class TrackerTopology;

class SiStripUtility {
public:
  typedef dqm::harvesting::DQMStore DQMStore;
  typedef dqm::harvesting::MonitorElement MonitorElement;

  static int getMEList(std::string const& name, std::vector<std::string>& values);
  static bool checkME(std::string const& element, std::string const& name, std::string& full_path);
  static int getMEList(std::string const& name, std::string& dir_path, std::vector<std::string>& me_names);

  static void split(std::string const& str, std::vector<std::string>& tokens, std::string const& delimiters = " ");
  static void getMEStatusColor(int status, int& rval, int& gval, int& bval);
  static void getDetectorStatusColor(int status, int& rval, int& gval, int& bval);
  static void getMEStatusColor(int status, int& icol, std::string& tag);
  static int getMEStatus(MonitorElement const* me);
  static int getMEStatus(MonitorElement const* me, int& bad_channels);
  static void getModuleFolderList(DQMStore& dqm_store, std::vector<std::string>& m_ids);
  static void getModuleFolderList(DQMStore::IBooker& ibooker,
                                  DQMStore::IGetter& igetter,
                                  std::vector<std::string>& m_ids);
  static void getMEValue(MonitorElement const* me, std::string& val);
  static bool goToDir(DQMStore& dqm_store, std::string const& name);
  static bool goToDir(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter, std::string const& name);
  static void getSubDetectorTag(uint32_t det_id, std::string& subdet_tag, const TrackerTopology* tTopo);
  static void setBadModuleFlag(std::string& hname, uint16_t& flg);
  static void getBadModuleStatus(uint16_t flag, std::string& message);
  static void getTopFolderPath(DQMStore& dqm_store, std::string const& top_dir, std::string& path);
  static void getTopFolderPath(DQMStore::IBooker& ibooker,
                               DQMStore::IGetter& igetter,
                               std::string const& top_dir,
                               std::string& path);
};

#endif
