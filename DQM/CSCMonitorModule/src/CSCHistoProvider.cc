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

#include "DQM/CSCMonitorModule/interface/CSCHistoProvider.h"

CSCHistoProvider::CSCHistoProvider() {
}

CSCHistoProvider::~CSCHistoProvider() {
}

const bool CSCHistoProvider::getEMUHisto(const cscdqm::EMUHistoType& histo, CSCMonitorObject* me) { 
  return false; 
}

const bool CSCHistoProvider::getDDUHisto(const cscdqm::DDUHistoType& histo, CSCMonitorObject* me) { 
  return false; 
}

const bool CSCHistoProvider::getCSCHisto(const cscdqm::CSCHistoType& histo, CSCMonitorObject* me) { 
  return false; 
}

void CSCHistoProvider::getCSCFromMap(const unsigned int crateID, const unsigned int dmbSlot, unsigned int& cscType, unsigned int& cscPosition) { 

}

const uint32_t CSCHistoProvider::getCSCDetRawId(const int endcap, const int station, const int vmecrate, const int dmb, const int tmb) const {

  return 0;  
}

const bool CSCHistoProvider::nextCSC(unsigned int& iter, unsigned int& crateId, unsigned int& dmbId) const {
  return false;
}

const bool CSCHistoProvider::getEffParamHisto(const std::string& paramName, CSCMonitorObject* me) {
  return false;
}

