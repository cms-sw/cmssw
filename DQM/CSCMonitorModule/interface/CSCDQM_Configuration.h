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

namespace cscdqm {

  typedef enum CONF_PARAMETER_BOOL {
    CONF_BINCHECKER_CRC_ALCT,
    CONF_BINCHECKER_CRC_ALCT,
    CONF_BINCHECKER_CRC_ALCT,
    CONF_BINCHECKER_OUTPUT
  };

  typedef enum CONF_PARAMETER_STRING {
  };

  typedef enum CONF_PARAMETER_UINT {
    CONF_DDU_CHECK_MASK,
    CONF_BIN_CHECK_MASK
  };

  typedef enum CONF_PARAMETER_DOUBLE {
    CONF_EFF_COLD_THRESHOLD,
    CONF_EFF_COLD_SIGFAIL,
    CONF_EFF_HOT_THRESHOLD,
    CONF_EFF_HOT_SIGFAIL,
    CONF_EFF_ERR_THRESHOLD,
    CONF_EFF_ERR_SIGFAIL,
    CONF_EFF_NODATA_THRESHOLD,
    CONF_EFF_NODATA_SIGFAIL
  };

  /**
   * @class Configuration
   * @brief Framework configuration
   */
  class Configuration {

    public:
      
      Configuration();
      ~Configuration();

    private:

  };

}

#endif
