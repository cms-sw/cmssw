#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
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
int getMENameList(string name, string& dir_path, string& me_names);
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
