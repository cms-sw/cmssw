/*
 * =====================================================================================
 *
 *       Filename:  CSCMonitorModule.cc
 *
 *    Description:  CSC Monitor Module
 *
 *        Version:  1.0
 *        Created:  11/13/2008 02:31:12 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), valdas.rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#include "DQM/CSCMonitorModule/interface/CSCMonitorModule.h"

/**
 * @brief  Constructor.
 * @param  ps Parameters.
 */
CSCMonitorModule::CSCMonitorModule(const edm::ParameterSet& ps) : inputTag(INPUT_TAG_LABEL) {

  edm::FileInPath fp;

  edm::ParameterSet params = ps.getUntrackedParameter<edm::ParameterSet>("EventProcessor");
  config.load(params);

  fp = ps.getParameter<edm::FileInPath>("BOOKING_XML_FILE");
  config.setBOOKING_XML_FILE(fp.fullPath());
    
  dbe = edm::Service<DQMStore>().operator->();

  dispatcher = new cscdqm::Dispatcher(&config, const_cast<CSCMonitorModule*>(this));
  dispatcher->init();

}

/**
 * @brief  Destructor.
 */
CSCMonitorModule::~CSCMonitorModule() {
  if (dispatcher) delete dispatcher;
}

/**
 * @brief  Analyze Event.
 * @param  e Event to analyze
 * @param  c Event Setup
 */
void CSCMonitorModule::analyze(const edm::Event& e, const edm::EventSetup& c) {

  // Get crate mapping from database
  edm::ESHandle<CSCCrateMap> hcrate;
  c.get<CSCCrateMapRcd>().get(hcrate);
  pcrate = hcrate.product();
    
  dispatcher->processEvent(e, inputTag);

}

/**
 * @brief  Book Monitor Object on Request.
 * @param  req Request.
 * @return MonitorObject created.
 */
cscdqm::MonitorObject* CSCMonitorModule::bookMonitorObject(const cscdqm::HistoBookRequest& req) {

  cscdqm::MonitorObject *me = NULL;
  std::string name = req.hdef->getName();

  std::string path = req.folder;
  if (req.hdef->getPath().size() > 0) {
    path = path + req.hdef->getPath() + "/";
  }
  
  //std::cout << "Moving to " << path << " for name = " << name << " with fullPath = " << req.hdef->getFullPath() << "\n";

  dbe->setCurrentFolder(path);

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

