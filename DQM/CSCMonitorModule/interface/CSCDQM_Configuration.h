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

#include <boost/function.hpp>
#include <boost/bind.hpp>

#include "DQM/CSCMonitorModule/interface/CSCDQM_MonitorObjectProvider.h"

namespace cscdqm {

  class Dispatcher;

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
      bool FRAEFF_AUTO_UPDATE;
      bool FRAEFF_SEPARATE_THREAD;

      std::string BOOKING_XML_FILE;
      std::string FOLDER_EMU;
      std::string FOLDER_DDU;
      std::string FOLDER_CSC;
      std::string FOLDER_PAR;

      unsigned int DDU_CHECK_MASK;
      unsigned int DDU_BINCHECK_MASK;
      unsigned int BINCHECK_MASK;
      unsigned int FRAEFF_AUTO_UPDATE_START;
      unsigned int FRAEFF_AUTO_UPDATE_FREQ;

      double EFF_COLD_THRESHOLD;
      double EFF_COLD_SIGFAIL;
      double EFF_HOT_THRESHOLD;
      double EFF_HOT_SIGFAIL;
      double EFF_ERR_THRESHOLD;
      double EFF_ERR_SIGFAIL;
      double EFF_NODATA_THRESHOLD;
      double EFF_NODATA_SIGFAIL;

      boost::function< bool (const HistoDef& histoT, MonitorObject*&) > fnGetHisto;
      boost::function< void (const HistoDef& histoT, MonitorObject*&) > fnPutHisto;
      boost::function< MonitorObject* (const HistoBookRequest&) > fnBook;
      boost::function< CSCDetId (const unsigned int, const unsigned int) > fnGetCSCDetId;
      boost::function< bool (unsigned int&, unsigned int&, unsigned int&) > fnNextBookedCSC;

      Configuration() {

        BINCHECKER_CRC_ALCT = false;
        BINCHECKER_CRC_ALCT = false;
        BINCHECKER_CRC_ALCT = false;
        BINCHECKER_OUTPUT   = false;
        FRAEFF_AUTO_UPDATE  = true;
        FRAEFF_AUTO_UPDATE_START    = 5;
        FRAEFF_AUTO_UPDATE_FREQ     = 1;
        DDU_CHECK_MASK    = 0xFFFFFFFF;
        BINCHECK_MASK     = 0xFFFFFFFF;
        DDU_BINCHECK_MASK = 0x02080016;
        FRAEFF_SEPARATE_THREAD = false;
        FOLDER_EMU = "";
        FOLDER_DDU = "";
        FOLDER_CSC = "";
        FOLDER_PAR = "";

        reset();

      }

      void reset() {
        nEvents = 0;
        nEventsBad = 0;
        nEventsGood = 0;
        nEventsCSC = 0;
        nUnpackedDMB = 0;
      }

      const unsigned long getNEvents() const      { return nEvents; }
      const unsigned long getNEventsBad() const   { return nEventsBad; }
      const unsigned long getNEventsGood() const  { return nEventsGood; }
      const unsigned long getNEventsCSC() const   { return nEventsCSC; }
      const unsigned long getNUnpackedDMB() const { return nUnpackedDMB; }

      void incNEvents()      { nEvents++; }
      void incNEventsBad()   { nEventsBad++; }
      void incNEventsGood()  { nEventsGood++; }
      void incNEventsCSC()   { nEventsCSC++; }
      void incNUnpackedDMB() { nUnpackedDMB++; }

    private:

      unsigned long nEvents;
      unsigned long nEventsBad;
      unsigned long nEventsGood;
      unsigned long nEventsCSC;
      unsigned long nUnpackedDMB; 

  };

}

#endif
