/*
 * =====================================================================================
 *
 *       Filename:  CSCMonitorModuleCmn_HistoProvider.cc
 *
 *    Description:  Histogram Provider methods for CSCMonitorModuleCmn object
 *
 *        Version:  1.0
 *        Created:  11/13/2008 02:35:44 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), valdas.rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#include "DQM/CSCMonitorModule/interface/CSCMonitorModuleCmn.h"

const bool CSCMonitorModuleCmn::getEMUHisto(const cscdqm::EMUHistoType& histo, CSCMonitorObject* mo) {
  return false;
}

const bool CSCMonitorModuleCmn::getDDUHisto(const cscdqm::DDUHistoType& histo, CSCMonitorObject* mo) {
  return false;
}

const bool CSCMonitorModuleCmn::getCSCHisto(const cscdqm::CSCHistoType& histo, CSCMonitorObject* mo) {
  return false;
}

const bool CSCMonitorModuleCmn::getEffParamHisto(const std::string& paramName, CSCMonitorObject* mo) {
  return false;
}

void CSCMonitorModuleCmn::getCSCFromMap(const unsigned int crateId, const unsigned int dmbId, unsigned int& cscType, unsigned int& cscPosition) {
}

const uint32_t CSCMonitorModuleCmn::getCSCDetRawId(const int endcap, const int station, const int vmecrate, const int dmb, const int tmb) const {
  return 0;
}

const bool CSCMonitorModuleCmn::nextCSC(unsigned int& iter, unsigned int& crateId, unsigned int& dmbId) const {
  return false;
}
