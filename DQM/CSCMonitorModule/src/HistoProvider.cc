/*
 * =====================================================================================
 *
 *       Filename:  HistoProvider.cc
 *
 *    Description:  General Histogram Provider
 *
 *        Version:  1.0
 *        Created:  10/06/2008 10:55:21 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius, valdas.rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#include "DQM/CSCMonitorModule/interface/HistoProvider.h"

namespace cscdqm {

  HistoProvider::HistoProvider() {
    processor = new EventProcessorType(const_cast<HistoProvider*>(this));
  }

  HistoProvider::~HistoProvider() {
    delete processor;
  }

  template <typename HistoIdT>
  const bool HistoProvider::getHisto(const HistoIdT& histo, MonitorElement* me) { 
    if (typeid(histo) == typeid(cscdqm::EMUHistoType)) {
      cscdqm::EMUHistoType _histo = (cscdqm::EMUHistoType) histo;
    } else
    if (typeid(histo) == typeid(cscdqm::DDUHistoType)) {
      cscdqm::DDUHistoType _histo = (cscdqm::DDUHistoType) histo;
    } else
    if (typeid(histo) == typeid(cscdqm::CSCHistoType)) {
      cscdqm::CSCHistoType _histo = (cscdqm::CSCHistoType) histo;
    }  
    return false; 
  }

  void HistoProvider::getCSCFromMap(const unsigned int crateID, const unsigned int dmbSlot, unsigned int& cscType, unsigned int& cscPosition) { 

  }

  const uint32_t HistoProvider::getCSCDetRawId(const int endcap, const int station, const int vmecrate, const int dmb, const int tmb) const {

    return 0;  
  }

  const bool HistoProvider::nextCSC(unsigned int& iter, unsigned int& crateId, unsigned int& dmbId) const {
    return false;
  }

  const bool HistoProvider::getEffParamHisto(const std::string& paramName, MonitorElement* me) {
    return false;
  }

}
