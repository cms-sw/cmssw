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

const bool CSCMonitorModuleCmn::getHisto(const cscdqm::HistoType& histo, cscdqm::MonitorObject*& mo) {

  if (typeid(histo)  == typeid(cscdqm::CSCHistoType)) { return false; }
  if (histo.isRef()) return false;

  /// Check if MO is already in the cache. If so - return it and exit
  MOCacheMap::const_iterator i = moCache.find(histo.getUID());
  if (i != moCache.end()) {
    mo = i->second;
    return true;
  }

  /// Take a type and initialize stuff
  const std::type_info& t = typeid(histo);
  std::string path("");

  /// Construct appropriate path (depends on histo type)
  if (t == typeid(cscdqm::EMUHistoType)) {
    path.append(DIR_SUMMARY);
  } else
  if (t == typeid(cscdqm::DDUHistoType)) {
    path.append(DIR_DDU);
    path.append(histo.getTag());
    path.append("/");
  } else
  if (t == typeid(cscdqm::CSCHistoType)) {
    path.append(DIR_CSC);
    path.append(histo.getTag());
    path.append("/");
  } else
    return false;

  std::string id(path);
  id.append(histo.getId());

  /// Get it from DBE
  MonitorElement* me = dbe->get(id);
  if (me == NULL) {
    if (t == typeid(cscdqm::EMUHistoType)) {
      LOG_INFO << "MO [" << t.name() << "] not found: " << histo.getUID() << " in path " << id;
      return false;
    } else
    if (t == typeid(cscdqm::DDUHistoType)) {
      dbe->setCurrentFolder(path);
      LOG_INFO << "Booking DDU histograms in " << path;
      collection->book("DDU");
    } else
    if (t == typeid(cscdqm::CSCHistoType)) {
      dbe->setCurrentFolder(path);
      LOG_INFO << "Booking CSC histograms in " << path;
      collection->book("CSC");
    } else
      return false;
    me = dbe->get(id);
    if (me == NULL) {
      LOG_INFO << "MO [" << t.name() << "] not found: " << histo.getUID() << " in path " << id;
      return false;
    }
  }

  moCache[histo.getUID()] = new CSCMonitorObject(me);

  /// Put to cache for the future
  mo = moCache[histo.getUID()];

  return true;
}

void CSCMonitorModuleCmn::getCSCFromMap(const unsigned int crateId, const unsigned int dmbId, unsigned int& cscType, unsigned int& cscPosition) const {
  CSCDetId cid = pcrate->detId(crateId, dmbId, 0, 0);
  cscPosition  = cid.chamber();
  int iring    = cid.ring();
  int istation = cid.station();
  int iendcap  = cid.endcap();
    
  std::string tlabel = cscdqm::Utility::getCSCTypeLabel(iendcap, istation, iring);
  cscType = cscdqm::Utility::getCSCTypeBin(tlabel);
}

const uint32_t CSCMonitorModuleCmn::getCSCDetRawId(const int endcap, const int station, const int vmecrate, const int dmb, const int tmb) const {
  return 0;
}

const bool CSCMonitorModuleCmn::nextCSC(unsigned int& iter, unsigned int& crateId, unsigned int& dmbId) const {
  return false;
}

