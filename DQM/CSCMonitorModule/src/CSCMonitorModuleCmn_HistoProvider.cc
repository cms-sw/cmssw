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

  if (histo.getId() == cscdqm::h::HISTO_SKIP)         { return false; }
  if (typeid(histo)  == typeid(cscdqm::CSCHistoType)) { return false; }

  // Check if MO is already in the cache. If so - return it and exit
  MOCacheMap::const_iterator i = moCache.find(histo.getUID());
  if (i != moCache.end()) {
    mo = i->second;
    return true;
  }

  // Take a type and initialize stuff
  const std::type_info& t = typeid(histo);
  std::string path("");

  // Construct appropriate path (depends on histo type)
  // EMU Level
  if (t == EMUHistoT) {

    if (histo.getId() == cscdqm::h::EMU_PHYSICS_EMU) {
      path.append(DIR_EVENTINFO);
    } else {
      path.append(DIR_SUMMARY);
    }

  // DDU Level
  } else if (t == DDUHistoT) {

    path.append(DIR_DDU);
    path.append(histo.getTag());
    path.append("/");

  // CSC Level
  } else if (t == CSCHistoT) {

    path.append(DIR_CSC);
    path.append(histo.getTag());
    path.append("/");

  // Parameter Level
  } else if (t == ParHistoT) {

    path.append(DIR_SUMMARY_CONTENTS);

  // Other? Exit
  } else return false;

  std::string id(path);
  id.append(histo.getId());

  // Get MonitorElement from DBE
  MonitorElement* me = dbe->get(id);

  // If MonitorElement was not found
  if (me == NULL) {

    // For EMU Level - report Error and return false
    if (t == EMUHistoT) {

      LOG_INFO << "MO [" << t.name() << "] not found: " << histo.getUID() << " in path " << id;
      return false;

    // For DDU level - book histograms
    } else if (t == DDUHistoT) {

      dbe->setCurrentFolder(path);
      collection->book("DDU");

    // For CSC Level - book histograms
    } else if (t == CSCHistoT) {

      dbe->setCurrentFolder(path);
      collection->book("CSC");

    // For Parameter Level - book histogram
    } else if (t == ParHistoT) {

      dbe->setCurrentFolder(path);
      bookFloat(histo.getId());

    // Other? Exit 
    } else return false;

    // Try getting again, if null again - report and exit
    me = dbe->get(id);
    if (me == NULL) {
      LOG_INFO << "MO [" << t.name() << "] not found: " << histo.getUID() << " in path " << id;
      return false;
    }

  }

  // Put MonitorElement to cache for the future fast retrieval
  moCache[histo.getUID()] = new CSCMonitorObject(me);

  // get it from cache and return
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

