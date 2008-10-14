/*
 * =====================================================================================
 *
 *       Filename:  CSCUtilities.h
 *
 *    Description:  All CSC Utilities
 *
 *        Version:  1.0
 *        Created:  10/06/2008 05:01:01 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius, valdas.rapsevicius@cern.ch 
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */
#ifndef CSCDQM_HistoUtility_H
#define CSCDQM_HistoUtility_H

#include <string>
#include <vector>
#include <sstream>
#include <bitset>
#include <map>
#include <iostream>
#include <fstream>
#include <iomanip>

#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOMNodeList.hpp>

namespace cscdqm {

/**
 * Type Definition Section
 */

typedef std::map<std::string, std::string>     Histo;
typedef Histo::iterator                        HistoIter;
typedef std::map<std::string, Histo>           HistoDef;
typedef HistoDef::iterator                     HistoDefIter;
typedef std::map<std::string, HistoDef>        HistoDefMap;
typedef HistoDefMap::iterator                  HistoDefMapIter;

typedef std::bitset<36>                        BitsetDDU;
typedef std::bitset<32>                        Bitset32;

/**
 * @brief  Converting from string to whatever number (failsafe!) 
 * @param  t result number
 * @param  s source string
 * @param  f base
 * @return true if success, else - false
 */
template <class T>
bool stringToNumber(T& t, const std::string& s, std::ios_base& (*f)(std::ios_base&)) {
  std::istringstream iss(s);
  return !(iss >> f >> t).fail();
}

class HistoUtility {

  public:
  
    static const bool loadCollection(const char* filename, HistoDefMap& collection);    

    static const bool getHistoValue(Histo& h, const std::string name, std::string& value, const std::string def_value = "");
    static const bool getHistoValue(Histo& h, const std::string name, int& value, const int def_value = 0);
    static const bool getHistoValue(Histo& h, const std::string name, double& value, const double def_value = 0.0);

    static const int ParseAxisLabels(const std::string& s, std::map<int, std::string>& labels);
    static void getCSCTypeToBinMap(std::map<std::string, int>& tmap);
    static const std::string getCSCTypeLabel(int endcap, int station, int ring);
    static const int tokenize(const std::string& str, std::vector<std::string>& tokens, const std::string& delimiters = " ");
    static void splitString(std::string str, const std::string delim, std::vector<std::string>& results);
    static void trimString(std::string& str);

  private:


};

}

#endif
