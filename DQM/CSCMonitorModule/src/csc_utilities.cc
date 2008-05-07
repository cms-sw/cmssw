/*
 * =====================================================================================
 *
 *       Filename:  csc_utilities.cc
 *
 *    Description:  Various utilities that are being used throughout the code.
 *    Most of them are static functions...
 *
 *        Version:  1.0
 *        Created:  04/21/2008 11:32:19 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), Valdas.Rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#include <DQM/CSCMonitorModule/interface/CSCMonitorModule.h>
#include <vector>

/**
 * Supress "unused function" warnings in compiler output
 */

static std::string getHistoValue(Histo& h, const std::string name, std::string& value, const std::string def_value) __attribute__ ((unused));
static int getHistoValue(Histo& h, const std::string name, int& value, const int def_value) __attribute__ ((unused));
static double getHistoValue(Histo& h, const std::string name, double& value, const int def_value) __attribute__ ((unused));
static int tokenize(const std::string& str, std::vector<std::string>& tokens, const std::string& delimiters) __attribute__ ((unused));
static int ParseAxisLabels(const std::string& s, std::map<int, std::string>& labels) __attribute__ ((unused));
static void getCSCTypeToBinMap(std::map<std::string, int>& tmap) __attribute__ ((unused));
static std::string getCSCTypeLabel(int endcap, int station, int ring ) __attribute__ ((unused));
static const std::string getDDUTag(const unsigned int& dduNumber, std::string& buffer) __attribute__ ((unused));


/**
 * @brief  Format and return DDU Tag (to be folder name)
 * @param  ddNumber DDU number (1 - 36)
 * @param  buffer   Buffer to be filled with data
 * @return 
 */
static const std::string getDDUTag(const unsigned int& dduNumber, std::string& buffer) {

  std::stringstream oss;
  oss << std::setfill('0');
  oss << "DDU_" << std::setw(3) << dduNumber;
  buffer = oss.str();
  return buffer;
}

/**
 * @brief  Find string histogram value in map
 * @param  h Histogram map
 * @param  name parameter name
 * @param  value handler for parameter value
 * @return true if parameter found and filled, false - otherwise
 */
static bool findHistoValue(Histo& h, const std::string name, std::string& value) {
  HistoIter i = h.find(name);
  if(i == h.end()) {
    return false;
  } else {
    value = i->second;
  }
  return true;
}

/**
 * @brief  Find int histogram value in map
 * @param  h Histogram map
 * @param  name parameter name
 * @param  value handler for parameter value
 * @return true if parameter found and filled, false - otherwise
 */
static bool findHistoValue(Histo& h, const std::string name, int& value) {
  HistoIter i = h.find(name);
  if(i == h.end()) {
    return false;
  } else {
    if(EOF == std::sscanf(i->second.c_str(), "%d", &value)) {
      return false;
    }
  }
  return true;
}

/**
 * @brief  Find double histogram value in map
 * @param  h Histogram map
 * @param  name parameter name
 * @param  value handler for parameter value
 * @return true if parameter found and filled, false - otherwise
 */
static bool findHistoValue(Histo& h, const std::string name, double& value) {
  HistoIter i = h.find(name);
  if(i == h.end()) {
    return false;
  } else {
    if(EOF == std::sscanf(i->second.c_str(), "%lf", &value)) {
      return false;
    }
  }
  return true;
}

/**
 * @brief  get Histogram string value out of the map and 
 * @param  h Histogram map
 * @param  name parameter name
 * @param  value handler for parameter value
 * @param  default value if parameter not found 
 * @return pointer to value
 */
static std::string getHistoValue(Histo& h, const std::string name, std::string& value, const std::string def_value = "") {
  if(!findHistoValue(h, name, value)) {
    value = def_value;
  }
  return value;
}

/**
 * @brief  get Histogram int value out of the map and 
 * @param  h Histogram map
 * @param  name parameter name
 * @param  value handler for parameter value
 * @param  default value if parameter not found 
 * @return pointer to value
 */
static int getHistoValue(Histo& h, const std::string name, int& value, const int def_value = 0) {
  if(!findHistoValue(h, name, value)) {
    value = def_value;
  }
  return value;
}

/**
 * @brief  get Histogram double value out of the map and 
 * @param  h Histogram map
 * @param  name parameter name
 * @param  value handler for parameter value
 * @param  default value if parameter not found 
 * @return pointer to value
 */
static double getHistoValue(Histo& h, const std::string name, double& value, const int def_value = 0) {
  if(!findHistoValue(h, name, value)) {
    value = def_value;
  }
  return value;
}

/**
 * @brief  Break string into tokens
 * @param  str source string to break
 * @param  tokens pointer to result vector
 * @param  delimiters delimiter string, default " "
 * @return 
 */
static int tokenize(const std::string& str, std::vector<std::string>& tokens, const std::string& delimiters = " ") {
  std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  std::string::size_type pos = str.find_first_of(delimiters, lastPos);
  while (std::string::npos != pos || std::string::npos != lastPos) {
    tokens.push_back(str.substr(lastPos, pos - lastPos));
    lastPos = str.find_first_not_of(delimiters, pos);
    pos = str.find_first_of(delimiters, lastPos);
  }
  return tokens.size();
}


/**
 * @brief  Parse Axis label string and return values in vector
 * @param  s source string to parse
 * @param  labels pointer to result vector
 * @return number of labels found
 */
static int ParseAxisLabels(const std::string& s, std::map<int, std::string>& labels) {
  std::string tmp = s;
  std::string::size_type pos = tmp.find("|");
  char* stopstring = NULL;

  while (pos != std::string::npos) {
    std::string label_pair = tmp.substr(0, pos);
    tmp.replace(0,pos+1,"");
    if (label_pair.find("=") != std::string::npos) {
      int nbin = strtol(label_pair.substr(0,label_pair.find("=")).c_str(),  &stopstring, 10);
      std::string label = label_pair.substr(label_pair.find("=")+1, label_pair.length());
      while (label.find("\'") != std::string::npos) {
        label.erase(label.find("\'"),1);
      }
      labels[nbin] = label;
    }
    pos = tmp.find("|");
  }
  return labels.size();
}

/**
 * @brief  Construct CSC bin map
 * @param  tmap pointer to result vector
 * @return 
 */
static void getCSCTypeToBinMap(std::map<std::string, int>& tmap) {
  tmap["ME-4/2"] = 0;
  tmap["ME-4/1"] = 1;
  tmap["ME-3/2"] = 2;
  tmap["ME-3/1"] = 3;
  tmap["ME-2/2"] = 4;
  tmap["ME-2/1"] = 5;
  tmap["ME-1/3"] = 6;
  tmap["ME-1/2"] = 7;
  tmap["ME-1/1"] = 8;
  tmap["ME+1/1"] = 9;
  tmap["ME+1/2"] = 10;
  tmap["ME+1/3"] = 11;
  tmap["ME+2/1"] = 12;
  tmap["ME+2/2"] = 13;
  tmap["ME+3/1"] = 14;
  tmap["ME+3/2"] = 15;
  tmap["ME+4/1"] = 16;
  tmap["ME+4/2"] = 17;
}


/**
 * @brief  Get CSC label from CSC parameters
 * @param  endcap Endcap number
 * @param  station Station number
 * @param  ring Ring number
 * @return chamber label
 */
static std::string getCSCTypeLabel(int endcap, int station, int ring ) {
  std::string label = "Unknown";
  std::ostringstream st;
  if ((endcap > 0) && (station > 0) && (ring > 0)) {
    if (endcap == 1) {
      st << "ME+" << station << "/" << ring;
      label = st.str();
    } else if (endcap==2) {
      st << "ME-" << station << "/" << ring;
      label = st.str();
    } else {
      label = "Unknown";
    }
  }
  return label;
}

