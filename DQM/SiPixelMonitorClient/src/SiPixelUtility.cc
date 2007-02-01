#include "DQM/SiPixelMonitorClient/interface/SiPixelUtility.h"
#include "DQMServices/Core/interface/QTestStatus.h"
using namespace std;
//
// Get a list of MEs in a folder
//
int SiPixelUtility::getMEList(string name, vector<string>& values) {
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
int SiPixelUtility::getMEList(string name, string& dir_path, vector<string>& values) {
  values.clear();
  dir_path = name.substr(0,(name.find(":")));
  dir_path += "/"; 
  string temp_str = name.substr(name.find(":")+1);
  split(temp_str, values, ",");
  return values.size();
}

// Check if the requested ME exists in a folder
bool SiPixelUtility::checkME(string name, string me_name, string& full_path) {
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
void SiPixelUtility::split(const string& str, vector<string>& tokens, const string& delimiters) {
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
void SiPixelUtility::getStatusColor(int status, int& rval, int&gval, int& bval) {
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
void SiPixelUtility::getStatusColor(int status, int& icol, string& tag) {
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
int SiPixelUtility::getStatus(MonitorElement* me) {
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
