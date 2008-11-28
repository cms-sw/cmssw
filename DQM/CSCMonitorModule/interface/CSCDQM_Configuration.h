/*
 * =====================================================================================
 *
 *       Filename:  Configuration.h
 *
 *    Description:  CSCDQM Configuration parameter storage
 *
 *        Version:  1.0
 *        Created:  10/03/2008 10:26:04 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius, valdas.rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#ifndef CSCDQM_Configuration_H
#define CSCDQM_Configuration_H

#include <map>
#include <string>

namespace cscdqm {

  typedef enum ConfParameterBool {
    CONF_BINCHECKER_CRC_ALCT,
    CONF_BINCHECKER_CRC_CLCT,
    CONF_BINCHECKER_CRC_CFEB,
    CONF_BINCHECKER_OUTPUT
  };

  typedef enum ConfParameterString {
    CONF_BOOKING_XML_FILE
  };

  typedef enum ConfParameterUInt {
    CONF_DDU_CHECK_MASK,
    CONF_BIN_CHECK_MASK
  };

  typedef enum ConfParameterDouble {
    CONF_EFF_COLD_THRESHOLD,
    CONF_EFF_COLD_SIGFAIL,
    CONF_EFF_HOT_THRESHOLD,
    CONF_EFF_HOT_SIGFAIL,
    CONF_EFF_ERR_THRESHOLD,
    CONF_EFF_ERR_SIGFAIL,
    CONF_EFF_NODATA_THRESHOLD,
    CONF_EFF_NODATA_SIGFAIL
  };

  static const bool         CONF_DEFAULT_BOOL   = false;
  static const std::string  CONF_DEFAULT_STRING = "";
  static const unsigned int CONF_DEFAULT_UINT   = 0;
  static const double       CONF_DEFAULT_DOUBLE = 0.0;

  /**
   * @class Configuration
   * @brief Framework configuration
   */
  class Configuration {

    public:
      
      Configuration() {
        // Setting default values
        add(CONF_BINCHECKER_CRC_ALCT, false);
        add(CONF_BINCHECKER_CRC_ALCT, false);
        add(CONF_BINCHECKER_CRC_ALCT, false);
        add(CONF_BINCHECKER_OUTPUT, false);
      }

      void add(ConfParameterBool param, bool value) { boolMap[param] = value; }
      void add(ConfParameterString param, std::string value) { stringMap[param] = value; }
      void add(ConfParameterUInt param, unsigned int value) { uintMap[param] = value; }
      void add(ConfParameterDouble param, double value) { doubleMap[param] = value; }

      const bool& get(ConfParameterBool param) {  
        std::map<ConfParameterBool, bool>::const_iterator i = boolMap.find(param);
        if (i == boolMap.end()) return CONF_DEFAULT_BOOL;
        return i->second;
      }

      const std::string& get(ConfParameterString param) {  
        std::map<ConfParameterString, std::string>::const_iterator i = stringMap.find(param);
        if (i == stringMap.end()) return CONF_DEFAULT_STRING;
        return i->second;
      }

      const unsigned int& get(ConfParameterUInt param) {  
        std::map<ConfParameterUInt, unsigned int>::const_iterator i = uintMap.find(param);
        if (i == uintMap.end()) return CONF_DEFAULT_UINT;
        return i->second;
      }

      const double& get(ConfParameterDouble param) {  
        std::map<ConfParameterDouble, double>::const_iterator i = doubleMap.find(param);
        if (i == doubleMap.end()) return CONF_DEFAULT_DOUBLE;
        return i->second;
      }

    private:

      std::map<ConfParameterBool, bool> boolMap;
      std::map<ConfParameterString, std::string> stringMap;
      std::map<ConfParameterUInt, unsigned int> uintMap;
      std::map<ConfParameterDouble, double> doubleMap;

  };

}

#endif
