/*
 * =====================================================================================
 *
 *       Filename:  CSCUtilities.cc
 *
 *    Description:  Various utilities that are being used throughout the code.
 *    Most of them are functions...
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

#include "DQM/CSCMonitorModule/interface/CSCUtility.h"

/**
 * @brief  Format and return DDU Tag (to be folder name)
 * @param  ddNumber DDU number (1 - 36)
 * @param  buffer   Buffer to be filled with data
 * @return 
 */
std::string CSCUtility::getDDUTag(const unsigned int& dduNumber, std::string& buffer) {
  std::stringstream oss;
  oss << std::setfill('0');
  oss << "DDU_" << std::setw(2) << dduNumber;
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
bool CSCUtility::findHistoValue(Histo& h, const std::string name, std::string& value) {
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
bool CSCUtility::findHistoValue(Histo& h, const std::string name, int& value) {
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
bool CSCUtility::findHistoValue(Histo& h, const std::string name, double& value) {
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
std::string CSCUtility::getHistoValue(Histo& h, const std::string name, std::string& value, const std::string def_value) {
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
int CSCUtility::getHistoValue(Histo& h, const std::string name, int& value, const int def_value) {
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
double CSCUtility::getHistoValue(Histo& h, const std::string name, double& value, const int def_value) {
  if(!findHistoValue(h, name, value)) {
    value = def_value;
  }
  return value;
}

/**
 * @brief  Parse Axis label string and return values in vector
 * @param  s source string to parse
 * @param  labels pointer to result vector
 * @return number of labels found
 */
int CSCUtility::ParseAxisLabels(const std::string& s, std::map<int, std::string>& labels) {
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
  * @brief  Get CSC y-axis position from chamber string
  * @param  cstr Chamber string
  * @return chamber y-axis position
  */
int CSCUtility::getCSCTypeBin(const std::string& cstr) {
  if (cstr.compare("ME-4/2") == 0) return 0;
  if (cstr.compare("ME-4/1") == 0) return 1;
  if (cstr.compare("ME-3/2") == 0) return 2;
  if (cstr.compare("ME-3/1") == 0) return 3;
  if (cstr.compare("ME-2/2") == 0) return 4;
  if (cstr.compare("ME-2/1") == 0) return 5;
  if (cstr.compare("ME-1/3") == 0) return 6;
  if (cstr.compare("ME-1/2") == 0) return 7;
  if (cstr.compare("ME-1/1") == 0) return 8;
  if (cstr.compare("ME+1/1") == 0) return 9;
  if (cstr.compare("ME+1/2") == 0) return 10;
  if (cstr.compare("ME+1/3") == 0) return 11;
  if (cstr.compare("ME+2/1") == 0) return 12;
  if (cstr.compare("ME+2/2") == 0) return 13;
  if (cstr.compare("ME+3/1") == 0) return 14;
  if (cstr.compare("ME+3/2") == 0) return 15;
  if (cstr.compare("ME+4/1") == 0) return 16;
  if (cstr.compare("ME+4/2") == 0) return 17;
  return 0;
}

/**
 * @brief  Get CSC label from CSC parameters
 * @param  endcap Endcap number
 * @param  station Station number
 * @param  ring Ring number
 * @return chamber label
 */
std::string CSCUtility::getCSCTypeLabel(int endcap, int station, int ring ) {
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


/**
 * @brief  Break string into tokens
 * @param  str source string to break
 * @param  tokens pointer to result vector
 * @param  delimiters delimiter string, default " "
 * @return 
 */
int CSCUtility::tokenize(const std::string& str, std::vector<std::string>& tokens, const std::string& delimiters) {
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
 * @brief  Split string according to delimiter
 * @param  str String to split
 * @param  delim Delimiter
 * @param  results Vector to write results to
 * @return 
 */
void CSCUtility::splitString(std::string str, const std::string delim, std::vector<std::string>& results) {
  unsigned int cutAt;
  while ((cutAt = str.find_first_of(delim)) != str.npos) {
    if(cutAt > 0) {
      results.push_back(str.substr(0, cutAt));
    }
    str = str.substr(cutAt + 1);
  }
  if(str.length() > 0) {
    results.push_back(str);
  }
}


/**
 * @brief  Trim string
 * @param  str string to trim
 * @return 
 */
void CSCUtility::trimString(std::string& str) {
  std::string::size_type pos = str.find_last_not_of(' ');
  if(pos != std::string::npos) {
    str.erase(pos + 1);
    pos = str.find_first_not_of(' ');
    if(pos != std::string::npos) 
      str.erase(0, pos);
  } else 
    str.erase(str.begin(), str.end());
}

/**
 * @brief  Calculate super fast hash (from
 * http://www.azillionmonkeys.com/qed/hash.html)
 * @param  data Source Data 
 * @param  length of data
 * @return hash result
 */
uint32_t CSCUtility::fastHash(const char * data, int len) {
  uint32_t hash = len, tmp;
  int rem;

  if (len <= 0 || data == NULL) return 0;
  rem = len & 3;
  len >>= 2;

  /* Main loop */
  for (;len > 0; len--) {
    hash  += get16bits (data);
    tmp    = (get16bits (data+2) << 11) ^ hash;
    hash   = (hash << 16) ^ tmp;
    data  += 2*sizeof (uint16_t);
    hash  += hash >> 11;
  }

  /* Handle end cases */
  switch (rem) {
    case 3: hash += get16bits (data);
            hash ^= hash << 16;
            hash ^= data[sizeof (uint16_t)] << 18;
            hash += hash >> 11;
            break;
    case 2: hash += get16bits (data);
            hash ^= hash << 11;
            hash += hash >> 17;
            break;
    case 1: hash += *data;
            hash ^= hash << 10;
            hash += hash >> 1;
  }

  /* Force "avalanching" of final 127 bits */
  hash ^= hash << 3;
  hash += hash >> 5;
  hash ^= hash << 4;
  hash += hash >> 17;
  hash ^= hash << 25;
  hash += hash >> 6;

  return hash;
}
