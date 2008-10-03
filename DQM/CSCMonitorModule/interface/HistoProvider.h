/*
 * =====================================================================================
 *
 *       Filename:  HistoProviderExample.h
 *
 *    Description:  Histo Provider to EventProcessor
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

#ifndef HistoProvider_H
#define HistoProvider_H

#include "DQM/CSCMonitorModule/interface/HistoType.h"
#include "DQM/CSCMonitorModule/interface/EventProcessor.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

typedef cscdqm::EventProcessor<MonitorElement> EventProcessorType;

class HistoProvider {

  public:
    
    HistoProvider() {
      processor = new EventProcessorType(this);
    };

    ~HistoProvider() {
      delete processor;
    };

    const bool getEMUHisto(const cscdqm::HistoType type, MonitorElement* me) { return false; }
    const bool getDDUHisto(const int dduID, const cscdqm::HistoType histo, MonitorElement* me) { return false; }
    const bool getCSCHisto(const int crateID, const int dmbSlot, const cscdqm::HistoType histo, MonitorElement* me) { return false; }

    void getCSCFromMap(const unsigned int crateID, const unsigned int dmbSlot, unsigned int& cscType, unsigned int& cscPosition) { }
    const uint32_t getCSCDetRawId(const int endcap, const int station, const int vmecrate, const int dmb, const int tmb) const { return 0;  }

  private:

    EventProcessorType* processor;

};

#endif
