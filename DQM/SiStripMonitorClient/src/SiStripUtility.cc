#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

//
// Get a list of MEs in a folder
//
int SiStripUtility::getMEList(std::string const& name, std::vector<std::string>& values) {
  values.clear();
  auto prefix_str = name.substr(0, (name.find(":")));
  prefix_str += "/";
  auto const temp_str = name.substr(name.find(":") + 1);
  split(temp_str, values, ",");
  for (auto& value : values) {
    value.insert(0, prefix_str);
  }
  return values.size();
}
//
// Get a list of MEs in a folder and the path name
//
int SiStripUtility::getMEList(std::string const& name, std::string& dir_path, std::vector<std::string>& values) {
  values.clear();
  dir_path = name.substr(0, (name.find(":")));
  dir_path += "/";
  auto const temp_str = name.substr(name.find(":") + 1);
  split(temp_str, values, ",");
  return values.size();
}

// Check if the requested ME exists in a folder
bool SiStripUtility::checkME(std::string const& name, std::string const& me_name, std::string& full_path) {
  if (name.find(name) == std::string::npos)
    return false;
  auto prefix_str = name.substr(0, (name.find(":")));
  prefix_str += "/";
  auto const temp_str = name.substr(name.find(":") + 1);
  std::vector<std::string> values;
  split(temp_str, values, ",");
  for (auto const& value : values) {
    if (value.find(me_name) != std::string::npos) {
      full_path = prefix_str + value;
      return true;
    }
  }
  return false;
}
//
// -- Split a given string into a number of strings using given
//    delimiters and fill a vector with splitted strings
//
void SiStripUtility::split(const std::string& str, std::vector<std::string>& tokens, const std::string& delimiters) {
  // Skip delimiters at beginning.
  std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);

  // Find first "non-delimiter".
  std::string::size_type pos = str.find_first_of(delimiters, lastPos);

  while (std::string::npos != pos || std::string::npos != lastPos) {
    // Found a token, add it to the std::vector.
    tokens.push_back(str.substr(lastPos, pos - lastPos));

    // Skip delimiters.  Note the "not_of"
    lastPos = str.find_first_not_of(delimiters, pos);

    // Find next "non-delimiter"
    pos = str.find_first_of(delimiters, lastPos);
  }
}
//
// -- Get Color code from Status
//
void SiStripUtility::getMEStatusColor(int status, int& rval, int& gval, int& bval) {
  if (status == dqm::qstatus::STATUS_OK) {
    rval = 0;
    gval = 255;
    bval = 0;
  } else if (status == dqm::qstatus::WARNING) {
    rval = 255;
    gval = 255;
    bval = 0;
  } else if (status == dqm::qstatus::ERROR) {
    rval = 255;
    gval = 0;
    bval = 0;
  } else if (status == dqm::qstatus::OTHER) {
    rval = 255;
    gval = 150;
    bval = 0;
  } else {
    rval = 0;
    gval = 0;
    bval = 255;
  }
}
//
// -- Get Color code from Status
//
void SiStripUtility::getMEStatusColor(int status, int& icol, std::string& tag) {
  if (status == dqm::qstatus::STATUS_OK) {
    tag = "Ok";
    icol = 3;
  } else if (status == dqm::qstatus::WARNING) {
    tag = "Warning";
    icol = 5;
  } else if (status == dqm::qstatus::ERROR) {
    tag = "Error";
    icol = 2;
  } else if (status == dqm::qstatus::OTHER) {
    tag = "Other";
    icol = 1;
  } else {
    tag = " ";
    icol = 1;
  }
}
//
// -- Get Color code from Status
//
void SiStripUtility::getDetectorStatusColor(int status, int& rval, int& gval, int& bval) {
  // No Error
  if (status == 0) {
    rval = 0;
    gval = 255;
    bval = 0;
    return;
  }
  // Error detected in FED Channel
  if (((status >> 0) & 0x1) > 0) {
    rval = 150;
    gval = 0;
    bval = 0;
    return;
  }
  // Excluded FED Channel
  if (((status >> 3) & 0x1) > 0) {
    rval = 100;
    gval = 100;
    bval = 255;
    return;
  }
  // DCS Error
  if (((status >> 4) & 0x1) > 0) {
    rval = 200;
    gval = 20;
    bval = 255;
    return;
  }
  // Digi and Cluster Problem
  if (((status >> 1) & 0x1) > 0) {
    rval = 255;
    bval = 0;
    if (((status >> 2) & 0x1) > 0)
      gval = 0;
    else
      gval = 100;
  } else {
    rval = 251;
    gval = 0;
    bval = 100;
  }
}

//
// -- Get Status of Monitor Element
//
int SiStripUtility::getMEStatus(MonitorElement const* me) {
  int status = 0;
  if (me->getQReports().empty()) {
    return status;
  } else if (me->hasError()) {
    status = dqm::qstatus::ERROR;
  } else if (me->hasWarning()) {
    status = dqm::qstatus::WARNING;
  } else if (me->hasOtherReport()) {
    status = dqm::qstatus::OTHER;
  } else {
    status = dqm::qstatus::STATUS_OK;
  }
  return status;
}
//
// --  Fill Module Names
//
void SiStripUtility::getModuleFolderList(DQMStore& dqm_store, std::vector<std::string>& mfolders) {
  if (auto currDir = dqm_store.pwd(); currDir.find("module_") != std::string::npos) {
    mfolders.push_back(currDir);
  } else {
    auto const subdirs = dqm_store.getSubdirs();
    for (auto const& subdir : subdirs) {
      dqm_store.cd(subdir);
      getModuleFolderList(dqm_store, mfolders);
      dqm_store.goUp();
    }
  }
}

void SiStripUtility::getModuleFolderList(DQMStore::IBooker& ibooker,
                                         DQMStore::IGetter& igetter,
                                         std::vector<std::string>& mfolders) {
  if (auto currDir = ibooker.pwd(); currDir.find("module_") != std::string::npos) {
    mfolders.push_back(currDir);
  } else {
    auto const subdirs = igetter.getSubdirs();
    for (auto const& subdir : subdirs) {
      ibooker.cd(subdir);
      getModuleFolderList(ibooker, igetter, mfolders);
      ibooker.goUp();
    }
  }
}
//
// -- Get Status of Monitor Element
//
int SiStripUtility::getMEStatus(MonitorElement const* me, int& bad_channels) {
  int status = 0;
  if (me->getQReports().empty()) {
    bad_channels = -1;
  } else {
    std::vector<QReport*> qreports = me->getQReports();
    bad_channels = qreports[0]->getBadChannels().size();
    if (me->hasError()) {
      status = dqm::qstatus::ERROR;
    } else if (me->hasWarning()) {
      status = dqm::qstatus::WARNING;
    } else if (me->hasOtherReport()) {
      status = dqm::qstatus::OTHER;
    } else {
      status = dqm::qstatus::STATUS_OK;
    }
  }
  return status;
}
//
// -- Get Status of Monitor Element
//
void SiStripUtility::getMEValue(MonitorElement const* me, std::string& val) {
  val = "";
  if (me) {
    if (me->kind() == MonitorElement::Kind::REAL) {
      val = std::to_string(me->getFloatValue());
    } else if (me->kind() == MonitorElement::Kind::INT) {
      val = std::to_string(me->getIntValue());
    }
  }
}
//
// -- go to a given Directory
//
bool SiStripUtility::goToDir(DQMStore& dqm_store, std::string const& name) {
  std::string currDir = dqm_store.pwd();
  std::string dirName = currDir.substr(currDir.find_last_of("/") + 1);
  if (dirName.find(name) == 0) {
    return true;
  }
  auto const subdirs = dqm_store.getSubdirs();
  for (auto const& fname : subdirs) {
    if ((fname.find("Reference") != std::string::npos) || (fname.find("AlCaReco") != std::string::npos) ||
        (fname.find("HLT") != std::string::npos) || (fname.find("IsolatedBunches") != std::string::npos))
      continue;
    dqm_store.cd(fname);
    if (!goToDir(dqm_store, name)) {
      dqm_store.goUp();
    } else {
      return true;
    }
  }
  return false;
}

bool SiStripUtility::goToDir(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter, std::string const& name) {
  std::string currDir = ibooker.pwd();
  std::string dirName = currDir.substr(currDir.find_last_of("/") + 1);
  if (dirName.find(name) == 0) {
    return true;
  }
  auto const subdirs = igetter.getSubdirs();
  for (auto const& fname : subdirs) {
    if ((fname.find("Reference") != std::string::npos) || (fname.find("AlCaReco") != std::string::npos) ||
        (fname.find("HLT") != std::string::npos) || (fname.find("IsolatedBunches") != std::string::npos))
      continue;
    igetter.cd(fname);
    if (!goToDir(ibooker, igetter, name)) {
      ibooker.goUp();
    } else
      return true;
  }
  return false;
}

//
// -- Get Sub Detector tag from DetId
//
void SiStripUtility::getSubDetectorTag(uint32_t const det_id, std::string& subdet_tag, const TrackerTopology* tTopo) {
  StripSubdetector const subdet(det_id);
  subdet_tag = "";
  switch (subdet.subdetId()) {
    case StripSubdetector::TIB: {
      subdet_tag = "TIB";
      return;
    }
    case StripSubdetector::TID: {
      if (tTopo->tidSide(det_id) == 2) {
        subdet_tag = "TIDF";
      } else if (tTopo->tidSide(det_id) == 1) {
        subdet_tag = "TIDB";
      }
      return;
    }
    case StripSubdetector::TOB: {
      subdet_tag = "TOB";
      return;
    }
    case StripSubdetector::TEC: {
      if (tTopo->tecSide(det_id) == 2) {
        subdet_tag = "TECF";
      } else if (tTopo->tecSide(det_id) == 1) {
        subdet_tag = "TECB";
      }
    }
  }
}
//
// -- Set Bad Channel Flag from hname
//
void SiStripUtility::setBadModuleFlag(std::string& hname, uint16_t& flg) {
  if (hname.find("FractionOfBadChannels") != std::string::npos)
    flg |= (1 << 0);
  else if (hname.find("NumberOfDigi") != std::string::npos)
    flg |= (1 << 1);
  else if (hname.find("NumberOfCluster") != std::string::npos)
    flg |= (1 << 2);
  else if (hname.find("ExcludedFedChannel") != std::string::npos)
    flg |= (1 << 3);
  else if (hname.find("DCSError") != std::string::npos)
    flg |= (1 << 4);
}
//
// -- Get the Status Message from Bad Module Flag
//
void SiStripUtility::getBadModuleStatus(uint16_t flag, std::string& message) {
  if (flag == 0)
    message += " No Error";
  else {
    if (((flag >> 0) & 0x1) > 0)
      message += " Fed BadChannel : ";
    if (((flag >> 1) & 0x1) > 0)
      message += " # of Digi : ";
    if (((flag >> 2) & 0x1) > 0)
      message += " # of Clusters :";
    if (((flag >> 3) & 0x1) > 0)
      message += " Excluded FED Channel ";
    if (((flag >> 4) & 0x1) > 0)
      message += " DCSError ";
  }
}
//
// -- Set Event Info Folder
//
void SiStripUtility::getTopFolderPath(DQMStore& dqm_store, std::string const& top_dir, std::string& path) {
  path = "";
  dqm_store.cd();
  if (dqm_store.dirExists(top_dir)) {
    dqm_store.cd(top_dir);
    path = dqm_store.pwd();
  } else {
    if (SiStripUtility::goToDir(dqm_store, top_dir)) {
      std::string mdir = "MechanicalView";
      if (SiStripUtility::goToDir(dqm_store, mdir)) {
        path = dqm_store.pwd();
        path = path.substr(0, path.find(mdir) - 1);
      }
    }
  }
}

void SiStripUtility::getTopFolderPath(DQMStore::IBooker& ibooker,
                                      DQMStore::IGetter& igetter,
                                      std::string const& top_dir,
                                      std::string& path) {
  path = "";
  ibooker.cd();
  if (igetter.dirExists(top_dir)) {
    ibooker.cd(top_dir);
    path = ibooker.pwd();
  } else {
    if (SiStripUtility::goToDir(ibooker, igetter, top_dir)) {
      std::string tdir = "MechanicalView";
      if (SiStripUtility::goToDir(ibooker, igetter, tdir)) {
        path = ibooker.pwd();
        path = path.substr(0, path.find(tdir) - 1);
      }
    }
  }
}
