#include "DQM/TrackingMonitorClient/interface/TrackingUtility.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
//
// Get a list of MEs in a folder
//
int TrackingUtility::getMEList(std::string name, std::vector<std::string>& values) {
  values.clear();
  std::string prefix_str = name.substr(0, (name.find(':')));
  prefix_str += "/";
  std::string temp_str = name.substr(name.find(':') + 1);
  split(temp_str, values, ",");
  for (std::vector<std::string>::iterator it = values.begin(); it != values.end(); it++)
    (*it).insert(0, prefix_str);
  return values.size();
}
//
// Get a list of MEs in a folder and the path name
//
int TrackingUtility::getMEList(std::string name, std::string& dir_path, std::vector<std::string>& values) {
  values.clear();
  dir_path = name.substr(0, (name.find(':')));
  dir_path += "/";
  std::string temp_str = name.substr(name.find(':') + 1);
  split(temp_str, values, ",");
  return values.size();
}

// Check if the requested ME exists in a folder
bool TrackingUtility::checkME(std::string name, std::string me_name, std::string& full_path) {
  if (name.find(name) == std::string::npos)
    return false;
  std::string prefix_str = name.substr(0, (name.find(':')));
  prefix_str += "/";
  std::string temp_str = name.substr(name.find(':') + 1);
  std::vector<std::string> values;
  split(temp_str, values, ",");
  for (std::vector<std::string>::iterator it = values.begin(); it != values.end(); it++) {
    if ((*it).find(me_name) != std::string::npos) {
      full_path = prefix_str + (*it);
      return true;
    }
  }
  return false;
}
//
// -- Split a given string into a number of strings using given
//    delimiters and fill a vector with splitted strings
//
void TrackingUtility::split(const std::string& str, std::vector<std::string>& tokens, const std::string& delimiters) {
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
void TrackingUtility::getMEStatusColor(int status, int& rval, int& gval, int& bval) {
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
void TrackingUtility::getMEStatusColor(int status, int& icol, std::string& tag) {
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
// -- Get Status of Monitor Element
//
int TrackingUtility::getMEStatus(MonitorElement* me) {
  int status = 0;
  if (me->getQReports().empty()) {
    status = 0;
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
void TrackingUtility::getModuleFolderList(DQMStore::IBooker& ibooker,
                                          DQMStore::IGetter& igetter,
                                          std::vector<std::string>& mfolders) {
  std::string currDir = ibooker.pwd();
  if (currDir.find("module_") != std::string::npos) {
    //    std::string mId = currDir.substr(currDir.find("module_")+7, 9);
    mfolders.push_back(currDir);
  } else {
    std::vector<std::string> subdirs = igetter.getSubdirs();
    for (std::vector<std::string>::const_iterator it = subdirs.begin(); it != subdirs.end(); it++) {
      ibooker.cd(*it);
      getModuleFolderList(ibooker, igetter, mfolders);
      ibooker.goUp();
    }
  }
}
//
// -- Get Status of Monitor Element
//
int TrackingUtility::getMEStatus(MonitorElement* me, int& bad_channels) {
  int status = 0;
  if (me->getQReports().empty()) {
    status = 0;
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
void TrackingUtility::getMEValue(MonitorElement* me, std::string& val) {
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
bool TrackingUtility::goToDir(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter, std::string name) {
  std::string currDir = ibooker.pwd();
  std::string dirName = currDir.substr(currDir.find_last_of('/') + 1);
  if (dirName.find(name) == 0) {
    return true;
  }
  std::vector<std::string> subDirVec = igetter.getSubdirs();
  for (std::vector<std::string>::const_iterator ic = subDirVec.begin(); ic != subDirVec.end(); ic++) {
    const std::string& fname = (*ic);
    if ((fname.find("Reference") != std::string::npos) || (fname.find("AlCaReco") != std::string::npos) ||
        (fname.find("HLT") != std::string::npos))
      continue;
    igetter.cd(fname);
    if (!goToDir(ibooker, igetter, name))
      ibooker.goUp();
    else
      return true;
  }
  return false;
}
//
// -- Set Bad Channel Flag from hname
//
void TrackingUtility::setBadModuleFlag(std::string& hname, uint16_t& flg) {
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
void TrackingUtility::getBadModuleStatus(uint16_t flag, std::string& message) {
  if (flag == 0)
    message += " No Error";
  else {
    //    message += " Error from :: ";
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
void TrackingUtility::getTopFolderPath(DQMStore::IBooker& ibooker,
                                       DQMStore::IGetter& igetter,
                                       std::string top_dir,
                                       std::string& path) {
  path = "";
  ibooker.cd();
  if (igetter.dirExists(top_dir)) {
    ibooker.cd(top_dir);
    path = ibooker.pwd();
  } else {
    if (TrackingUtility::goToDir(ibooker, igetter, top_dir)) {
      std::string tdir = "TrackParameters";
      if (TrackingUtility::goToDir(ibooker, igetter, tdir)) {
        path = ibooker.pwd();
        path = path.substr(0, path.find(tdir) - 1);
      }
    }
  }
}
