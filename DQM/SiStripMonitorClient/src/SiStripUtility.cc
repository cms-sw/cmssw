#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
//
// Get a list of MEs in a folder
//
int SiStripUtility::getMEList(std::string name, std::vector<std::string>& values) {
  values.clear();
  std::string prefix_str = name.substr(0,(name.find(":")));
  prefix_str += "/"; 
  std::string temp_str = name.substr(name.find(":")+1);
  split(temp_str, values, ",");
  for (std::vector<std::string>::iterator it = values.begin();
       it != values.end(); it++) (*it).insert(0,prefix_str);
  return values.size();
}
//
// Get a list of MEs in a folder and the path name
//
int SiStripUtility::getMEList(std::string name, std::string& dir_path, std::vector<std::string>& values) {
  values.clear();
  dir_path = name.substr(0,(name.find(":")));
  dir_path += "/"; 
  std::string temp_str = name.substr(name.find(":")+1);
  split(temp_str, values, ",");
  return values.size();
}

// Check if the requested ME exists in a folder
bool SiStripUtility::checkME(std::string name, std::string me_name, std::string& full_path) {
  if (name.find(name) == std::string::npos) return false;
  std::string prefix_str = name.substr(0,(name.find(":")));
  prefix_str += "/"; 
  std::string temp_str = name.substr(name.find(":")+1);
  std::vector<std::string> values;
  split(temp_str, values, ",");
  for (std::vector<std::string>::iterator it = values.begin();
       it != values.end(); it++) {
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
void SiStripUtility::split(const std::string& str, std::vector<std::string>& tokens, const std::string& delimiters) {
  // Skip delimiters at beginning.
  std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);

  // Find first "non-delimiter".
  std::string::size_type pos = str.find_first_of(delimiters, lastPos);

  while (std::string::npos != pos || std::string::npos != lastPos)  {
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
void SiStripUtility::getMEStatusColor(int status, int& rval, int&gval, int& bval) {
  if (status == dqm::qstatus::STATUS_OK) { 
    rval = 0;   gval = 255;   bval = 0; 
  } else if (status == dqm::qstatus::WARNING) { 
    rval = 255; gval = 255; bval = 0;
  } else if (status == dqm::qstatus::ERROR) { 
    rval = 255; gval = 0;  bval = 0;
  } else if (status == dqm::qstatus::OTHER) { 
    rval = 255; gval = 150;  bval = 0;
  } else {
    rval = 0; gval = 0;  bval = 255;
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
void SiStripUtility::getDetectorStatusColor(int status, int& rval, int&gval, int& bval) {
  // No Error
  if (status == 0) {
    rval = 0; gval = 255;bval = 0;
    return;
  }
  // Error detected in FED Channel
  if (((status >> 0) & 0x1) > 0) { 
    rval = 150; gval = 0; bval = 0; 
    return;
  }
  // Excluded FED Channel 
  if (((status >> 3) & 0x1) > 0) {
    rval = 100; gval = 100; bval = 255; 
    return;
  }
  // DCS Error
  if (((status >> 4) & 0x1) > 0) {
    rval = 200; gval = 20; bval = 255; 
    return;
  } 
  // Digi and Cluster Problem   
  if (((status >> 1) & 0x1) > 0) {
    rval = 255; bval = 0;
    if (((status >> 2) & 0x1) > 0) gval = 0;
    else gval = 100;
  } else {
    rval = 251; gval = 0; bval = 100;   
  }
}

//
// -- Get Status of Monitor Element
//
int SiStripUtility::getMEStatus(MonitorElement* me) {
  int status = 0; 
  if (me->getQReports().size() == 0) {
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
void SiStripUtility::getModuleFolderList(DQMStore * dqm_store, std::vector<std::string>& mfolders){
  std::string currDir = dqm_store->pwd();
  if (currDir.find("module_") != std::string::npos)  {
    //    std::string mId = currDir.substr(currDir.find("module_")+7, 9);
    mfolders.push_back(currDir);
  } else {  
    std::vector<std::string> subdirs = dqm_store->getSubdirs();
    for (std::vector<std::string>::const_iterator it = subdirs.begin();
	 it != subdirs.end(); it++) {
      dqm_store->cd(*it);
      getModuleFolderList(dqm_store, mfolders);
      dqm_store->goUp();
    }
  }
}
//
// -- Get Status of Monitor Element
//
int SiStripUtility::getMEStatus(MonitorElement* me, int& bad_channels) {
  int status = 0; 
  if (me->getQReports().size() == 0) {
    status       = 0;
    bad_channels = -1;
  } else {
    std::vector<QReport *> qreports = me->getQReports();
    bad_channels =qreports[0]->getBadChannels().size();
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
void SiStripUtility::getMEValue(MonitorElement* me, std::string & val){
  val = "";
  if (me &&  me->kind()==MonitorElement::DQM_KIND_REAL) {
    val = me->valueString();
    val = val.substr(val.find("=")+1);
  }
}
//
// -- go to a given Directory
//
bool SiStripUtility::goToDir(DQMStore * dqm_store, std::string name) {
  std::string currDir = dqm_store->pwd();
  std::string dirName = currDir.substr(currDir.find_last_of("/")+1);
  if (dirName.find(name) == 0) {
    return true;
  }
  std::vector<std::string> subDirVec = dqm_store->getSubdirs();
  for (std::vector<std::string>::const_iterator ic = subDirVec.begin();
       ic != subDirVec.end(); ic++) {
    std::string fname = (*ic);
    if ((fname.find("Reference") != std::string::npos) ||
         (fname.find("AlCaReco") != std::string::npos)) continue;
    dqm_store->cd(fname);
    if (!goToDir(dqm_store, name))  dqm_store->goUp();
    else return true;
  }
  return false;  
}
//
// -- Get Sub Detector tag from DetId
//
void SiStripUtility::getSubDetectorTag(uint32_t det_id, std::string& subdet_tag) {
  StripSubdetector subdet(det_id);
  subdet_tag = "";
  switch (subdet.subdetId()) 
    {
    case StripSubdetector::TIB:
      {
	subdet_tag = "TIB";
	break;
      }
    case StripSubdetector::TID:
      {
	TIDDetId tidId(det_id);
	if (tidId.side() == 2) {
	  subdet_tag = "TIDF";
	}  else if (tidId.side() == 1) {
	  subdet_tag = "TIDB";
	}
	break;       
      }
    case StripSubdetector::TOB:
      {
	subdet_tag = "TOB";
	break;
      }
    case StripSubdetector::TEC:
      {
	TECDetId tecId(det_id);
	if (tecId.side() == 2) {
	  subdet_tag = "TECF";
	}  else if (tecId.side() == 1) {
	  subdet_tag = "TECB";	
	}
	break;       
      }
    }
}
//
// -- Set Bad Channel Flag from hname
// 
void SiStripUtility::setBadModuleFlag(std::string & hname, uint16_t& flg){
  
  if (hname.find("FractionOfBadChannels")   != std::string::npos) flg |= (1<<0);
  else if (hname.find("NumberOfDigi")       != std::string::npos) flg |= (1<<1);
  else if (hname.find("NumberOfCluster")    != std::string::npos) flg |= (1<<2);
  else if (hname.find("ExcludedFedChannel") != std::string::npos) flg |= (1<<3);
  else if (hname.find("DCSError")           != std::string::npos) flg |= (1<<4); 
}  
//
// -- Get the Status Message from Bad Module Flag
//
void SiStripUtility::getBadModuleStatus(uint16_t flag, std::string & message){
  if (flag == 0) message += "No Error";
  else {
    message += " Error from :: "; 
    if (((flag >> 0) & 0x1) > 0) message += " Fed BadChannel : ";
    if (((flag >> 1) & 0x1) > 0) message += " # of Digi : ";  
    if (((flag >> 2) & 0x1) > 0) message += " # of Clusters :";
    if (((flag >> 3) & 0x1) > 0) message += " Excluded FED Channel ";
    if (((flag >> 4) & 0x1) > 0) message += " DCSError "; 
  }
}
//
// -- Set Event Info Folder
//
void SiStripUtility::getTopFolderPath(DQMStore * dqm_store, std::string type, std::string& path) {
  if (type != "SiStrip" && type != "Tracking") return;
  path = ""; 
  dqm_store->cd();
  if (type == "SiStrip") {
    if (dqm_store->dirExists(type)) {
      dqm_store->cd(type);
      path = dqm_store->pwd();
    } else {
      if (SiStripUtility::goToDir(dqm_store, type)) {
	std::string mdir = "MechanicalView";
	if (SiStripUtility::goToDir(dqm_store, mdir)) {
	  path = dqm_store->pwd(); 
	  path = path.substr(0, path.find(mdir)-1);
        }
      }
    }
  } else if (type == "Tracking") {
    std::string top_dir = "Tracking";
    if (dqm_store->dirExists(top_dir)) {
      dqm_store->cd(top_dir);
      path = dqm_store->pwd();
    } else {
      if (SiStripUtility::goToDir(dqm_store, top_dir)) {
	std::string tdir = "TrackParameters";
	if (SiStripUtility::goToDir(dqm_store, tdir)) {
	  path = dqm_store->pwd(); 
	  path = path.substr(0, path.find(tdir)-1);
        }
      }	
    }
  }
} 
