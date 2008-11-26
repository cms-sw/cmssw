/*
 * =====================================================================================
 *
 *       Filename:  CSCDQM_Utility.h
 *
 *    Description:  CSC Utilities class
 *
 *        Version:  1.0
 *        Created:  10/30/2008 04:40:38 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), valdas.rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#ifndef CSCDQM_Utility_H
#define CSCDQM_Utility_H

#include <string>
#include <map>
#include <vector>
#include <sstream>

#include <boost/shared_ptr.hpp>
#include <TString.h>
#include <TPRegexp.h>

namespace cscdqm {

  static const TPRegexp REGEXP_ONDEMAND("^.*%d.*$");

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

  /**
   * @class Utility
   * @brief General and CSCDQM Framework related utility routines
   */
  class Utility {

    public:

      static const std::string getNameById(const std::string& name, const int id);

      static const bool regexMatch(const TPRegexp& re_expression, const std::string& message);
      static const bool regexMatch(const std::string& expression, const std::string& message);

      static const int getCSCTypeBin(const std::string& cstr);
      static const std::string getCSCTypeLabel(int endcap, int station, int ring);
      static const int tokenize(const std::string& str, std::vector<std::string>& tokens, const std::string& delimiters = " ");
      static void splitString(std::string str, const std::string delim, std::vector<std::string>& results);
      static void trimString(std::string& str);

  };

}

#endif
