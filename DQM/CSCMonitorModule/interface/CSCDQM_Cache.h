/*
 * =====================================================================================
 *
 *       Filename:  CSCDQM_Cache.h
 *
 *    Description:  Efficiently manages lists of MonitorObject 's 
 *
 *        Version:  1.0
 *        Created:  11/27/2008 10:05:00 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), valdas.rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#ifndef CSCDQM_Cache_H
#define CSCDQM_Cache_H 

#include <map>
#include <set>

#include <boost/shared_ptr.hpp>

#include "DQM/CSCMonitorModule/interface/CSCDQM_Logger.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_HistoType.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_MonitorObject.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_Utility.h"

namespace cscdqm {

  typedef boost::shared_ptr<MonitorObject> MonitorObjectPtr;
  typedef std::map<std::string, MonitorObjectPtr> CacheMap;

  /**
   * @class Cache
   * @brief MonitorObject cache - lists and routines to manage cache
   */
  class Cache {

    private:

      CacheMap cache;

    public:
      
      const bool get(const HistoType& histo, MonitorObject*& mo);
      void put(const HistoType& histo, MonitorObject* mo);

  };

}

#endif
