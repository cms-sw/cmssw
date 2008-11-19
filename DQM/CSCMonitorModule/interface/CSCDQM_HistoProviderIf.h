/*
 * =====================================================================================
 *
 *       Filename:  CSCDQM_HistoProviderIf.h
 *
 *    Description:  Histo Provider Interface
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

#ifndef CSCDQM_HistoProviderIf_H
#define CSCDQM_HistoProviderIf_H

#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include "DQM/CSCMonitorModule/interface/CSCDQM_HistoType.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_MonitorObjectIf.h"

namespace cscdqm {

  class HistoProvider {

    public:
    
      virtual const bool getEMUHisto(const EMUHistoType& histo, MonitorObject* mo) = 0;
      virtual const bool getDDUHisto(const DDUHistoType& histo, MonitorObject* mo) = 0;
      virtual const bool getCSCHisto(const CSCHistoType& histo, MonitorObject* mo) = 0;
      virtual const bool getEffParamHisto(const std::string& paramName, MonitorObject* mo) = 0;

      virtual void getCSCFromMap(const unsigned int crateId, const unsigned int dmbId, unsigned int& cscType, unsigned int& cscPosition) = 0;
      virtual const uint32_t getCSCDetRawId(const int endcap, const int station, const int vmecrate, const int dmb, const int tmb) const = 0;
      virtual const bool nextCSC(unsigned int& iter, unsigned int& crateId, unsigned int& dmbId) const = 0;

      virtual MonitorObject *bookInt       (const std::string &name) = 0;
      virtual MonitorObject *bookFloat     (const std::string &name) = 0;
      virtual MonitorObject *bookString    (const std::string &name,
                                            const std::string &value) = 0; 
      virtual MonitorObject *book1D        (const std::string &name,
                                            const std::string &title,
                                            int nchX, double lowX, double highX) = 0;
      virtual MonitorObject *book2D        (const std::string &name,
                                            const std::string &title,
                                            int nchX, double lowX, double highX,
                                            int nchY, double lowY, double highY) = 0;
      virtual MonitorObject *book3D        (const std::string &name,
                                            const std::string &title,
                                            int nchX, double lowX, double highX,
                                            int nchY, double lowY, double highY,
                                            int nchZ, double lowZ, double highZ) = 0;
      virtual MonitorObject* bookProfile   (const std::string &name,
                                            const std::string &title,
                                            int nchX, double lowX, double highX,
                                            int nchY, double lowY, double highY,
                                            const char *option = "s") = 0;
      virtual MonitorObject *bookProfile2D (const std::string &name,
                                            const std::string &title,
                                            int nchX, double lowX, double highX,
                                            int nchY, double lowY, double highY,
                                            int nchZ, double lowZ, double highZ,
                                            const char *option = "s") = 0;
  };

}

#endif
