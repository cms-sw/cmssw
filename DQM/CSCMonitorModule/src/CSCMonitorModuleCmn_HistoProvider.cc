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

cscdqm::MonitorObject* CSCMonitorModuleCmn::bookMonitorObject(const cscdqm::HistoBookRequest& req) {

  cscdqm::MonitorObject *me = NULL;

  dbe->setCurrentFolder(req.folder);

  switch (req.htype) {
    case cscdqm::INT:
      me = new CSCMonitorObject(dbe->bookInt(req.hdef->getName()));
      me->Fill(req.default_int);
      break;
    case cscdqm::FLOAT:
      if (std::string(req.hdef->getName()).compare(cscdqm::h::names[cscdqm::h::PAR_REPORT_SUMMARY]) == 0) {
        dbe->setCurrentFolder(DIR_EVENTINFO);
      }
      me = new CSCMonitorObject(dbe->bookFloat(req.hdef->getName()));
      me->Fill(req.default_float);
      break;
    case cscdqm::STRING:
      me = new CSCMonitorObject(dbe->bookString(req.hdef->getName(), req.default_string));
      break;
    case cscdqm::H1D: 
      me = new CSCMonitorObject(dbe->book1D(req.hdef->getName(), req.title, req.nchX, req.lowX, req.highX));
      break;
    case cscdqm::H2D:
      if (std::string(req.hdef->getHistoName()).compare(cscdqm::h::names[cscdqm::h::EMU_CSC_STATS_SUMMARY]) == 0) {
        dbe->setCurrentFolder(DIR_EVENTINFO);
      } else {
        me = new CSCMonitorObject(dbe->book2D(req.hdef->getName(), req.title, req.nchX, req.lowX, req.highX, req.nchY, req.lowY, req.highY));
      }
      break;
    case cscdqm::H3D:
      me = new CSCMonitorObject(dbe->book3D(req.hdef->getName(), req.title, req.nchX, req.lowX, req.highX, req.nchY, req.lowY, req.highY, req.nchZ, req.lowZ, req.highZ));
      break;
    case cscdqm::PROFILE:
      me = new CSCMonitorObject(dbe->bookProfile(req.hdef->getName(), req.title, req.nchX, req.lowX, req.highX, req.nchY, req.lowY, req.highY, req.option.c_str()));
      break;
    case cscdqm::PROFILE2D:
      me = new CSCMonitorObject(dbe->bookProfile2D(req.hdef->getName(), req.title, req.nchX, req.lowX, req.highX, req.nchY, req.lowY, req.highY, req.nchZ, req.lowZ, req.highZ, req.option.c_str()));
      break;
  }

  return me;

}

