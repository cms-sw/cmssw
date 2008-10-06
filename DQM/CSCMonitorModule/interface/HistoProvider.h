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

class HistoProvider;

typedef cscdqm::EventProcessor<MonitorElement, HistoProvider> EventProcessorType;

class HistoProvider {

  public:
    
    HistoProvider();
    ~HistoProvider();

    template <typename HistoIdT>
    const bool getHisto(const HistoIdT& histo, MonitorElement* me);
    const bool getEffParamHisto(const std::string& paramName, MonitorElement* me);

    void getCSCFromMap(const unsigned int crateId, const unsigned int dmbId, unsigned int& cscType, unsigned int& cscPosition);
    const uint32_t getCSCDetRawId(const int endcap, const int station, const int vmecrate, const int dmb, const int tmb) const;
    const bool nextCSC(unsigned int& iter, unsigned int& crateId, unsigned int& dmbId) const;

  private:

    EventProcessorType* processor;

};

#endif
