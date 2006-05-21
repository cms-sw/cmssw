#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"

// Get a list of MEs in a folder
int SiStripUtility::getMEList(string element, vector<string>& values) {
  values.clear();
  string prefix_str = element.substr(0,(element.find(":")));
  prefix_str += "/"; 
  string temp_str = element.substr(element.find(":")+1);
  split(temp_str, values, ",");
  for (vector<string>::iterator it = values.begin();
       it != values.end(); it++) (*it).insert(0,prefix_str);
  return values.size();
}
// Check if the requested ME exists in a folder
bool SiStripUtility::getME(string element, string name, string& full_path) {
  if (element.find(name) == std::string::npos) return false;
  string prefix_str = element.substr(0,(element.find(":")));
  prefix_str += "/"; 
  string temp_str = element.substr(element.find(":")+1);
  vector<string> values;
  split(temp_str, values, ",");
  for (vector<string>::iterator it = values.begin();
       it != values.end(); it++) {
    if ((*it).find(name) != std::string::npos) {
      full_path = prefix_str + (*it);
      return true;
    }
  }
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
