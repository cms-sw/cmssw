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

#ifndef CSCHistoProvider_H
#define CSCHistoProvider_H

#include "DQM/CSCMonitorModule/interface/CSCDQM_HistoType.h"
#include "DQM/CSCMonitorModule/interface/CSCMonitorObject.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

class CSCHistoProvider {

  public:
    
    CSCHistoProvider();
    ~CSCHistoProvider();

    const bool getEMUHisto(const cscdqm::EMUHistoType& histo, CSCMonitorObject* mo);
    const bool getDDUHisto(const cscdqm::DDUHistoType& histo, CSCMonitorObject* mo);
    const bool getCSCHisto(const cscdqm::CSCHistoType& histo, CSCMonitorObject* mo);
    const bool getEffParamHisto(const std::string& paramName, CSCMonitorObject* mo);

    void getCSCFromMap(const unsigned int crateId, const unsigned int dmbId, unsigned int& cscType, unsigned int& cscPosition);
    const uint32_t getCSCDetRawId(const int endcap, const int station, const int vmecrate, const int dmb, const int tmb) const;
    const bool nextCSC(unsigned int& iter, unsigned int& crateId, unsigned int& dmbId) const;

  private:

};

#endif
