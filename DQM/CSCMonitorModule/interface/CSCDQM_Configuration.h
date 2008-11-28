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

#include <string>
#include "DQM/CSCMonitorModule/interface/CSCDQM_MonitorObjectProvider.h"

namespace cscdqm {

  /**
   * @class Configuration
   * @brief Framework configuration
   */
  class Configuration {

    public:
      
      bool BINCHECKER_CRC_ALCT;
      bool BINCHECKER_CRC_CLCT;
      bool BINCHECKER_CRC_CFEB;
      bool BINCHECKER_OUTPUT;

      std::string BOOKING_XML_FILE;

      unsigned int DDU_CHECK_MASK;
      unsigned int DDU_BINCHECK_MASK;
      unsigned int BINCHECK_MASK;

      double EFF_COLD_THRESHOLD;
      double EFF_COLD_SIGFAIL;
      double EFF_HOT_THRESHOLD;
      double EFF_HOT_SIGFAIL;
      double EFF_ERR_THRESHOLD;
      double EFF_ERR_SIGFAIL;
      double EFF_NODATA_THRESHOLD;
      double EFF_NODATA_SIGFAIL;

      MonitorObjectProvider* provider;

      Configuration() {

        BINCHECKER_CRC_ALCT = false;
        BINCHECKER_CRC_ALCT = false;
        BINCHECKER_CRC_ALCT = false;
        BINCHECKER_OUTPUT   = false;
        DDU_CHECK_MASK    = 0xFFFFFFFF;
        BINCHECK_MASK     = 0xFFFFFFFF;
        DDU_BINCHECK_MASK = 0x02080016;

        provider = NULL;

      }

  };

}

#endif
