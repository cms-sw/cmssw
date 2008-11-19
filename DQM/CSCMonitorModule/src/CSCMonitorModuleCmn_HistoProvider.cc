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

const bool CSCMonitorModuleCmn::getEMUHisto(const cscdqm::EMUHistoType& histo, cscdqm::MonitorObject* mo) {
  return false;
}

const bool CSCMonitorModuleCmn::getDDUHisto(const cscdqm::DDUHistoType& histo, cscdqm::MonitorObject* mo) {
  return false;
}

const bool CSCMonitorModuleCmn::getCSCHisto(const cscdqm::CSCHistoType& histo, cscdqm::MonitorObject* mo) {
  return false;
}

const bool CSCMonitorModuleCmn::getEffParamHisto(const std::string& paramName, cscdqm::MonitorObject* mo) {
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
/*
CSCMonitorObject* CSCMonitorModuleCmn::bookInt (const std::string &name) {
  return (CSCMonitorObject*) dbe->bookInt(name);
}

CSCMonitorObject* CSCMonitorModuleCmn::bookFloat (const std::string &name) {
  return (CSCMonitorObject*) dbe->bookFloat(name);
}

CSCMonitorObject* CSCMonitorModuleCmn::bookString (const std::string &name, const std::string &value) {
  return (CSCMonitorObject*) dbe->bookString(name, value);
}

CSCMonitorObject* CSCMonitorModuleCmn::book1D (const std::string &name, const std::string &title, int nchX, double lowX, double highX) {
  return (CSCMonitorObject*) dbe->book1D(name, title, nchX, lowX, highX);
}

CSCMonitorObject* CSCMonitorModuleCmn::book2D (const std::string &name, const std::string &title, int nchX, double lowX, double highX, int nchY, double lowY, double highY) {
  return (CSCMonitorObject*) dbe->book2D(name, title, nchX, lowX, highX, nchY, lowY, highY);
}

CSCMonitorObject* CSCMonitorModuleCmn::book3D (const std::string &name, const std::string &title, int nchX, double lowX, double highX, int nchY, double lowY, double highY, int nchZ, double lowZ, double highZ) {
  return (CSCMonitorObject*) dbe->book3D(name, title, nchX, lowX, highX, nchY, lowY, highY, nchZ, lowZ, highZ);
}

inline const CSCMonitorObject CSCMonitorModuleCmn::bookProfile (const std::string &name, const std::string &title, int nchX, double lowX, double highX, int nchY, double lowY, double highY, const char *option) {
  return CSCMonitorObject(*(dbe->bookProfile(name, title, nchX, lowX, highX, nchY, lowY, highY, option)));
}

CSCMonitorObject* CSCMonitorModuleCmn::bookProfile2D (const std::string &name, const std::string &title, int nchX, double lowX, double highX, int nchY, double lowY, double highY, int nchZ, double lowZ, double highZ, const char *option) {
  return (CSCMonitorObject*) dbe->bookProfile2D(name, title, nchX, lowX, highX, nchY, lowY, highY, nchZ, lowZ, highZ, option);
}
*/

