#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
using namespace std;
//
// Get a list of MEs in a folder
//
int SiStripUtility::getMEList(string name, vector<string>& values) {
  values.clear();
  string prefix_str = name.substr(0,(name.find(":")));
  prefix_str += "/"; 
  string temp_str = name.substr(name.find(":")+1);
  split(temp_str, values, ",");
  for (vector<string>::iterator it = values.begin();
       it != values.end(); it++) (*it).insert(0,prefix_str);
  return values.size();
}
//
// Get a list of MEs in a folder and the path name
//
int SiStripUtility::getMEList(string name, string& dir_path, vector<string>& values) {
  values.clear();
  dir_path = name.substr(0,(name.find(":")));
  dir_path += "/"; 
  string temp_str = name.substr(name.find(":")+1);
  split(temp_str, values, ",");
  return values.size();
}

// Check if the requested ME exists in a folder
bool SiStripUtility::checkME(string name, string me_name, string& full_path) {
  if (name.find(name) == string::npos) return false;
  string prefix_str = name.substr(0,(name.find(":")));
  prefix_str += "/"; 
  string temp_str = name.substr(name.find(":")+1);
  vector<string> values;
  split(temp_str, values, ",");
  for (vector<string>::iterator it = values.begin();
       it != values.end(); it++) {
    if ((*it).find(me_name) != string::npos) {
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
void SiStripUtility::split(const string& str, vector<string>& tokens, const string& delimiters) {
  // Skip delimiters at beginning.
  string::size_type lastPos = str.find_first_not_of(delimiters, 0);

  // Find first "non-delimiter".
  string::size_type pos = str.find_first_of(delimiters, lastPos);

  while (string::npos != pos || string::npos != lastPos)  {
    // Found a token, add it to the vector.
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
void SiStripUtility::getMEStatusColor(int status, int& icol, string& tag) {
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
void SiStripUtility::getModuleFolderList(DQMStore * dqm_store, vector<string>& mfolders){
  string currDir = dqm_store->pwd();
  if (currDir.find("module_") != string::npos)  {
    string mId = currDir.substr(currDir.find("module_")+7, 9);
    mfolders.push_back(mId);
  } else {  
    vector<string> subdirs = dqm_store->getSubdirs();
    for (vector<string>::const_iterator it = subdirs.begin();
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
void SiStripUtility::getMEValue(MonitorElement* me, string & val){
  val = "";
  if (me) {
    val = me->valueString();
    val = val.substr(val.find("=")+1);
  }
}

