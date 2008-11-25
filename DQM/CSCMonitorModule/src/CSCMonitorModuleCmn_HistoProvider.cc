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

  if (histo.getHistoName() == cscdqm::h::HISTO_SKIP) { return false; }

  // Check if MO is already in the cache. If so - return it and exit
  MOCacheMap::const_iterator i = moCache.find(histo.getFullPath());
  if (i != moCache.end()) {
    mo = i->second;
    return true;
  }

  // Take a type and initialize stuff
  const std::type_info& t = typeid(histo);
  std::string path("");

  // ===================================================
  // Construct appropriate path (depends on histo type)
  // ===================================================
  // 
  // EMU Level
  //
  if (t == EMUHistoT) {

    if (histo.getHistoName() == cscdqm::h::EMU_PHYSICS_EMU) {
      path.append(DIR_EVENTINFO);
    } else {
      path.append(DIR_SUMMARY);
    }

  // 
  // DDU Level
  //
  } else if (t == DDUHistoT) {

    path.append(DIR_DDU);
    path.append(histo.getPath());
    path.append("/");

  // 
  // CSC Level
  //
  } else if (t == CSCHistoT) {

    path.append(DIR_CSC);
    path.append(histo.getPath());
    path.append("/");

  // 
  // Parameter Level
  //
  } else if (t == ParHistoT) {

    path.append(DIR_SUMMARY_CONTENTS);

  // 
  // Other? Exit
  //
  } else return false;

  std::string id(path);
  id.append(histo.getName());

  // Get MonitorElement from DBE
  MonitorElement* me = dbe->get(id);

  // ================================
  // If MonitorElement was not found
  // ================================
  if (me == NULL) {

    // 
    // For EMU Level - report Error and return false
    //

    if (t == EMUHistoT) {

      LOG_INFO << "MO [" << t.name() << "] not found: " << histo.getFullPath() << " in path " << id;
      return false;

    // 
    // For DDU level - book histograms
    //

    } else if (t == DDUHistoT) {

      bookedHistoSet::iterator bhi = bookedHisto.find(histo.getPath());
      if (bhi == bookedHisto.end()) {
        LOG_INFO << "Booking DDU histo set for = " <<  histo.getPath();
        dbe->setCurrentFolder(path);
        collection->book("DDU");
        bookedHisto.insert(histo.getPath());
      }

    // 
    // For CSC Level - book histograms
    //

    } else if (t == CSCHistoT) {

      dbe->setCurrentFolder(path);

      bookedHistoSet::iterator bhi = bookedHisto.find(histo.getPath());
      if (bhi == bookedHisto.end()) {
        LOG_DEBUG << "Booking CSC histo set for = " <<  histo.getPath();
        collection->book("CSC");
        bookedHisto.insert(histo.getPath());
        cscdqm::HistoType *general_histo = const_cast<cscdqm::HistoType*>(&histo);
        cscdqm::CSCHistoType *cschisto   = dynamic_cast<cscdqm::CSCHistoType*>(general_histo);
        bookedCSCs.push_back(*cschisto);
      }

      if (collection->isOnDemand("CSC", histo.getHistoName())) {
        bookedHistoSet::iterator bhi = bookedHisto.find(histo.getFullPath());
        if (bhi == bookedHisto.end()) {
          LOG_DEBUG << "Booking CSC histogram on demand: HistoName = " <<  histo.getHistoName() << " addId = " << histo.getAddId() << " fullPath = " << histo.getFullPath();
          collection->bookOnDemand("CSC", histo.getHistoName(), histo.getAddId());
          bookedHisto.insert(histo.getFullPath());
        }
      }

    // 
    // For Parameter Level - book histogram
    //

    } else if (t == ParHistoT) {

      bookedHistoSet::iterator bhi = bookedHisto.find(histo.getFullPath());
      if (bhi == bookedHisto.end()) {
        dbe->setCurrentFolder(path);
        bookFloat(histo.getName());
        bookedHisto.insert(histo.getFullPath());
      }

    // 
    // Other? Exit 
    //

    } else return false;

    // ==================================================
    // Try getting again, if null again - report and exit
    // ==================================================

    me = dbe->get(id);
    if (me == NULL) {
      LOG_INFO << "MO [" << t.name() << "] not found (after booking): " << histo.getFullPath() << " in path " << id;
      return false;
    }

  }

  // Put MonitorElement to cache for the future fast retrieval
  moCache[histo.getFullPath()] = new CSCMonitorObject(me);

  // get it from cache and return
  mo = moCache[histo.getFullPath()];

  return true;
}

void CSCMonitorModuleCmn::getCSCFromMap(const unsigned int crateId, const unsigned int dmbId, unsigned int& cscType, unsigned int& cscPosition) const {
  CSCDetId cid = getCSCDetId(crateId, dmbId);
  cscPosition  = cid.chamber();
  int iring    = cid.ring();
  int istation = cid.station();
  int iendcap  = cid.endcap();
  std::string tlabel = cscdqm::Utility::getCSCTypeLabel(iendcap, istation, iring);
  cscType = cscdqm::Utility::getCSCTypeBin(tlabel);
}

const bool CSCMonitorModuleCmn::nextCSC(unsigned int& iter, unsigned int& crateId, unsigned int& dmbId) const {
  if (iter < bookedCSCs.size()) {
    LOG_INFO << "Getting " << bookedCSCs.at(iter) << " as #" << iter;
    crateId = bookedCSCs.at(iter).getCrateId();
    dmbId   = bookedCSCs.at(iter).getDMBId();
    iter++;
    return true;
  }
  return false;
}

