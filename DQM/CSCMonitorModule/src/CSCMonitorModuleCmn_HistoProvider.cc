/*
 * =====================================================================================
 *
 *       Filename:  CSCMonitorModuleCmn_MonitorObjectProvider.cc
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
  // EMU Level =========================================
  //
  if (t == EMUHistoT) {

    if (std::string(cscdqm::h::EMU_CSC_STATS_SUMMARY).compare(histo.getHistoName()) == 0) {
      path.append(DIR_EVENTINFO);
    } else {
      path.append(DIR_SUMMARY);
    }

  // 
  // DDU Level =========================================
  //
  } else if (t == DDUHistoT) {

    path.append(DIR_DDU);
    path.append(histo.getPath());
    path.append("/");

  // 
  // CSC Level =========================================
  //
  } else if (t == CSCHistoT) {

    path.append(DIR_CSC);
    path.append(histo.getPath());
    path.append("/");

  // 
  // Parameter Level ===================================
  //
  } else if (t == ParHistoT) {

    if (std::string(cscdqm::h::PAR_REPORT_SUMMARY).compare(histo.getHistoName()) == 0) {
      path.append(DIR_EVENTINFO);
    } else {
      path.append(DIR_SUMMARY_CONTENTS);
    }

  // 
  // Other? Exit =======================================
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

    bool search_again = false;

    // 
    // For EMU Level - do nothing
    //

    if (t == EMUHistoT) {

    // 
    // For DDU level - book histograms
    //

    } else if (t == DDUHistoT) {

      bookedHistoSet::iterator bhi = bookedHisto.find(histo.getPath());
      if (bhi == bookedHisto.end()) {
        //LOG_INFO << "Booking DDU histo set for = " <<  histo.getPath();
        dbe->setCurrentFolder(path);
        dispatcher->getCollection()->book("DDU");
        bookedHisto.insert(histo.getPath());
        search_again = true;
      }

    // 
    // For CSC Level - book histograms
    //

    } else if (t == CSCHistoT) {

      dbe->setCurrentFolder(path);

      bookedHistoSet::iterator bhi = bookedHisto.find(histo.getPath());
      if (bhi == bookedHisto.end()) {
        //LOG_INFO << "Booking CSC histo set for = " <<  histo.getPath();
        dispatcher->getCollection()->book("CSC");
        bookedHisto.insert(histo.getPath());
        cscdqm::HistoType *general_histo = const_cast<cscdqm::HistoType*>(&histo);
        cscdqm::CSCHistoType *cschisto   = dynamic_cast<cscdqm::CSCHistoType*>(general_histo);
        bookedCSCs.push_back(*cschisto);
        search_again = true;
      }

      if (dispatcher->getCollection()->isOnDemand("CSC", histo.getHistoName())) {
        bookedHistoSet::iterator bhi = bookedHisto.find(histo.getFullPath());
        if (bhi == bookedHisto.end()) {
          //LOG_INFO << "Booking CSC histogram on demand: HistoName = " <<  histo.getHistoName() << " addId = " << histo.getAddId() << " fullPath = " << histo.getFullPath();
          dispatcher->getCollection()->bookOnDemand("CSC", histo.getHistoName(), histo.getAddId());
          bookedHisto.insert(histo.getFullPath());
          search_again = true;
        }
      }

    // 
    // For Parameter Level - do nothing
    //

    } else if (t == ParHistoT) {

    }

    // ==================================================
    // Try getting again, if null again - report and exit
    // ==================================================

    if (search_again) me = dbe->get(id);

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
    //LOG_INFO << "Getting " << bookedCSCs.at(iter) << " as #" << iter;
    crateId = bookedCSCs.at(iter).getCrateId();
    dmbId   = bookedCSCs.at(iter).getDMBId();
    iter++;
    return true;
  }
  return false;
}

cscdqm::MonitorObject* CSCMonitorModuleCmn::bookInt(const std::string &name) {
  return new CSCMonitorObject(dbe->bookInt(name));
}

cscdqm::MonitorObject* CSCMonitorModuleCmn::bookInt(const std::string &name, const int default_value) {
  cscdqm::MonitorObject *me = bookInt(name);
  me->Fill(default_value);
  return me;
}

cscdqm::MonitorObject* CSCMonitorModuleCmn::bookFloat(const std::string &name) {
  return new CSCMonitorObject(dbe->bookFloat(name));
}

cscdqm::MonitorObject* CSCMonitorModuleCmn::bookFloat(const std::string &name, const float default_value) {
  cscdqm::MonitorObject *me = bookFloat(name);
  me->Fill(default_value);
  return me;
}

cscdqm::MonitorObject* CSCMonitorModuleCmn::bookString(const std::string &name, const std::string &value) {
  return new CSCMonitorObject(dbe->bookString(name, value));
}

cscdqm::MonitorObject* CSCMonitorModuleCmn::book1D(const std::string &name, const std::string &title, int nchX, double lowX, double highX) {
  return new CSCMonitorObject(dbe->book1D(name, title, nchX, lowX, highX));
}

cscdqm::MonitorObject* CSCMonitorModuleCmn::book2D(const std::string &name, const std::string &title, int nchX, double lowX, double highX, int nchY, double lowY, double highY) {
  if (name.compare(cscdqm::h::EMU_CSC_STATS_SUMMARY) == 0) {
    dbe->setCurrentFolder(DIR_EVENTINFO);
    cscdqm::MonitorObject *me = new CSCMonitorObject(dbe->book2D(cscdqm::h::EMU_CSC_STATS_SUMMARY, title, nchX, lowX, highX, nchY, lowY, highY));
    dbe->setCurrentFolder(DIR_SUMMARY);
    return me;
  }
  return new CSCMonitorObject(dbe->book2D(name, title, nchX, lowX, highX, nchY, lowY, highY));
}

cscdqm::MonitorObject* CSCMonitorModuleCmn::book3D(const std::string &name, const std::string &title, int nchX, double lowX, double highX, int nchY, double lowY, double highY, int nchZ, double lowZ, double highZ) {
  return new CSCMonitorObject(dbe->book3D(name, title, nchX, lowX, highX, nchY, lowY, highY, nchZ, lowZ, highZ));
}

cscdqm::MonitorObject* CSCMonitorModuleCmn::bookProfile(const std::string &name, const std::string &title, int nchX, double lowX, double highX, int nchY, double lowY, double highY, const char *option) {
  return new CSCMonitorObject(dbe->bookProfile(name, title, nchX, lowX, highX, nchY, lowY, highY, option));
}

cscdqm::MonitorObject* CSCMonitorModuleCmn::bookProfile2D(const std::string &name, const std::string &title, int nchX, double lowX, double highX, int nchY, double lowY, double highY, int nchZ, double lowZ, double highZ, const char *option) {
  return new CSCMonitorObject(dbe->bookProfile2D(name, title, nchX, lowX, highX, nchY, lowY, highY, nchZ, lowZ, highZ, option));
}

void CSCMonitorModuleCmn::afterBook(cscdqm::MonitorObject*& me) {
  if (me != NULL) delete me;
}
