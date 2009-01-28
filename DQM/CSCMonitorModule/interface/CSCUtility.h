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
#ifndef CSCUtility_H
#define CSCUtility_H

#include <string>
#include <vector>
#include <sstream>
#include <bitset>
#include <map>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "stdint.h"

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

#undef get16bits
#if (defined(__GNUC__) && defined(__i386__)) || defined(__WATCOMC__) || defined(_MSC_VER) || defined (__BORLANDC__) || defined (__TURBOC__)
#define get16bits(d) (*((const uint16_t *) (d)))
#endif

#if !defined (get16bits)
#define get16bits(d) ((((uint32_t)(((const uint8_t *)(d))[1])) << 8) + (uint32_t)(((const uint8_t *)(d))[0]) )
#endif

/**
 * @class CSCUtility
 * @brief CSCMonitorModule and CSCHLTMonitorModule utility routines
 */
class CSCUtility {

  public:
  
    static std::string getDDUTag(const unsigned int& dduNumber, std::string& buffer);
    static bool findHistoValue(Histo& h, const std::string name, std::string& value);
    static bool findHistoValue(Histo& h, const std::string name, int& value);
    static bool findHistoValue(Histo& h, const std::string name, double& value);
    static std::string getHistoValue(Histo& h, const std::string name, std::string& value, const std::string def_value = "");
    static int getHistoValue(Histo& h, const std::string name, int& value, const int def_value = 0);
    static double getHistoValue(Histo& h, const std::string name, double& value, const int def_value = 0);
    static int ParseAxisLabels(const std::string& s, std::map<int, std::string>& labels);
    static int getCSCTypeBin(const std::string& cstr);
    static std::string getCSCTypeLabel(int endcap, int station, int ring );
    static int tokenize(const std::string& str, std::vector<std::string>& tokens, const std::string& delimiters = " ");
    static void splitString(std::string str, const std::string delim, std::vector<std::string>& results);
    static void trimString(std::string& str);
    static uint32_t fastHash(const char * data, int len);

};

#endif
