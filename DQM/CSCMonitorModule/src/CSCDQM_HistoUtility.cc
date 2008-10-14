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

#include "DQM/CSCMonitorModule/interface/CSCDQM_HistoUtility.h"

namespace cscdqm {

const bool HistoUtility::loadCollection(const char* filename, HistoDefMap& collection) {

  XERCES_CPP_NAMESPACE::XMLPlatformUtils::Initialize();
  XERCES_CPP_NAMESPACE::XercesDOMParser *parser = new XERCES_CPP_NAMESPACE::XercesDOMParser();
  parser->setValidationScheme(XERCES_CPP_NAMESPACE::XercesDOMParser::Val_Always);
  parser->setDoNamespaces(true);
  parser->setDoSchema(true);
  parser->setValidationSchemaFullChecking(false); // this is default
  parser->setCreateEntityReferenceNodes(true);  // this is default
  parser->setIncludeIgnorableWhitespace (false);

  parser->parse(filename);
  XERCES_CPP_NAMESPACE::DOMDocument *doc = parser->getDocument();
  XERCES_CPP_NAMESPACE::DOMNode *docNode = (XERCES_CPP_NAMESPACE::DOMNode*) doc->getDocumentElement();
  
  std::string nodeName = XERCES_CPP_NAMESPACE::XMLString::transcode(docNode->getNodeName());
  if( nodeName != "Booking" ){
    //LOGERROR("loadCollection") << "Wrong booking root node: " << XMLString::transcode(docNode->getNodeName());
    delete parser;
    return false;
  }
  XERCES_CPP_NAMESPACE::DOMNodeList *itemList = docNode->getChildNodes();

  for(uint32_t i=0; i < itemList->getLength(); i++) {

    nodeName = XERCES_CPP_NAMESPACE::XMLString::transcode(itemList->item(i)->getNodeName());
    if(nodeName != "Histogram") {
      continue;
    }

    XERCES_CPP_NAMESPACE::DOMNodeList *props  = itemList->item(i)->getChildNodes();
    Histo h;
    std::string prefix = "", name = "";
    for(uint32_t j = 0; j < props->getLength(); j++) {
      std::string tname  = XERCES_CPP_NAMESPACE::XMLString::transcode(props->item(j)->getNodeName());
      std::string tvalue = XERCES_CPP_NAMESPACE::XMLString::transcode(props->item(j)->getTextContent());
      h.insert(std::make_pair(tname, tvalue));
      if(tname == "Name")   name   = tvalue;
      if(tname == "Prefix") prefix = tvalue;
    }

    if(!name.empty() && !prefix.empty()) {
      HistoDefMapIter it = collection.find(prefix);
      if( it == collection.end()) {
        HistoDef hd;
        hd.insert(make_pair(name, h));
        collection.insert(make_pair(prefix, hd)); 
      } else {
        it->second.insert(make_pair(name, h));
      }
    }

  }

  delete parser;

  return true;

}

/**
 * @brief  Find string histogram value in map
 * @param  h Histogram map
 * @param  name parameter name
 * @param  value handler for parameter value
 * @return true if parameter found and filled, false - otherwise
 */
const bool HistoUtility::getHistoValue(Histo& h, const std::string name, std::string& value, const std::string def_value) {
  HistoIter i = h.find(name);
  if(i == h.end()) {
    value = def_value;
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
const bool HistoUtility::getHistoValue(Histo& h, const std::string name, int& value, const int def_value) {
  HistoIter i = h.find(name);
  if(i == h.end()) {
    value = def_value;
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
const bool HistoUtility::getHistoValue(Histo& h, const std::string name, double& value, const double def_value) {
  HistoIter i = h.find(name);
  if(i == h.end()) {
    value = def_value;
    return false;
  } else {
    if(EOF == std::sscanf(i->second.c_str(), "%lf", &value)) {
      return false;
    }
  }
  return true;
}

/**
 * @brief  Parse Axis label string and return values in vector
 * @param  s source string to parse
 * @param  labels pointer to result vector
 * @return number of labels found
 */
const int HistoUtility::ParseAxisLabels(const std::string& s, std::map<int, std::string>& labels) {
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
void HistoUtility::getCSCTypeToBinMap(std::map<std::string, int>& tmap) {
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
const std::string HistoUtility::getCSCTypeLabel(int endcap, int station, int ring ) {
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
const int HistoUtility::tokenize(const std::string& str, std::vector<std::string>& tokens, const std::string& delimiters) {
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
void HistoUtility::splitString(std::string str, const std::string delim, std::vector<std::string>& results) {
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
void HistoUtility::trimString(std::string& str) {
  std::string::size_type pos = str.find_last_not_of(' ');
  if(pos != std::string::npos) {
    str.erase(pos + 1);
    pos = str.find_first_not_of(' ');
    if(pos != std::string::npos) 
      str.erase(0, pos);
  } else 
    str.erase(str.begin(), str.end());
}

}
