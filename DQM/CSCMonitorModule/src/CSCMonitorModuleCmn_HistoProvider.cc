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
  std::string name = req.hdef->getName();

  dbe->setCurrentFolder(req.folder);

  if (req.htype == cscdqm::INT) {
    me = new CSCMonitorObject(dbe->bookInt(name));
    me->Fill(req.default_int);
  } else 
  if (req.htype == cscdqm::FLOAT) {
    if (req.hdef->getId() == cscdqm::h::PAR_REPORT_SUMMARY) {
      dbe->setCurrentFolder(DIR_EVENTINFO);
    }
    me = new CSCMonitorObject(dbe->bookFloat(name));
    me->Fill(req.default_float);
  } else 
  if (req.htype == cscdqm::STRING) {
    me = new CSCMonitorObject(dbe->bookString(name, req.default_string));
  } else 
  if (req.htype == cscdqm::H1D) { 
    me = new CSCMonitorObject(dbe->book1D(name, req.title, req.nchX, req.lowX, req.highX));
  } else 
  if (req.htype == cscdqm::H2D) {
    if (req.hdef->getId() == cscdqm::h::EMU_CSC_STATS_SUMMARY) {
      dbe->setCurrentFolder(DIR_EVENTINFO);
      name = "reportSummaryMap";
    }
    me = new CSCMonitorObject(dbe->book2D(name, req.title, req.nchX, req.lowX, req.highX, req.nchY, req.lowY, req.highY));
  } else 
  if (req.htype == cscdqm::H3D) {
    me = new CSCMonitorObject(dbe->book3D(name, req.title, req.nchX, req.lowX, req.highX, req.nchY, req.lowY, req.highY, req.nchZ, req.lowZ, req.highZ));
  } else 
  if (req.htype == cscdqm::PROFILE) {
    me = new CSCMonitorObject(dbe->bookProfile(name, req.title, req.nchX, req.lowX, req.highX, req.nchY, req.lowY, req.highY, req.option.c_str()));
  } else 
  if (req.htype == cscdqm::PROFILE2D) {
    me = new CSCMonitorObject(dbe->bookProfile2D(name, req.title, req.nchX, req.lowX, req.highX, req.nchY, req.lowY, req.highY, req.nchZ, req.lowZ, req.highZ, req.option.c_str()));
  }

  return me;

}

