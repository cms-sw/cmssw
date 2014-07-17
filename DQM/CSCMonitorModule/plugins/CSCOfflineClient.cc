/*
 * =====================================================================================
 *
 *       Filename:  CSCOfflineClient.cc
 *
 *    Description:  CSC Offline Client
 *
 *        Version:  1.0
 *        Created:  08/20/2009 02:31:12 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), valdas.rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#include "CSCOfflineClient.h"

/**
 * @brief  Constructor.
 * @param  ps Parameters.
 */
CSCOfflineClient::CSCOfflineClient(const edm::ParameterSet& ps) {

  edm::ParameterSet params = ps.getUntrackedParameter<edm::ParameterSet>("EventProcessor");
  config.load(params);

  dispatcher = new cscdqm::Dispatcher(&config, const_cast<CSCOfflineClient*>(this));
  dispatcher->init();

  if (ps.exists("MASKEDHW")) {
    maskedHW = ps.getUntrackedParameter<std::vector<std::string> >("MASKEDHW");
    // dispatcher->maskHWElements(maskedHW);
  }

}

/**
 * @brief  Destructor.
 */
CSCOfflineClient::~CSCOfflineClient() {
  if (dispatcher) delete dispatcher;
}

/*** No longer triggered for DQMEDHarvester ***/
/*
void CSCOfflineClient::endRun(const edm::Run& r, const edm::EventSetup& c) {

  // *
  // *  Putting histograms to internal cache: EMU stuff
  // * /
  dbe->setCurrentFolder(config.getFOLDER_EMU());
  std::vector<std::string> me_names = dbe->getMEs();
  for (std::vector<std::string>::iterator iter = me_names.begin(); iter != me_names.end(); iter++) {
    std::string me_name = *iter;
    MonitorElement* me = dbe->get(config.getFOLDER_EMU() + me_name);
    cscdqm::HistoId id;
    if (me && cscdqm::HistoDef::getHistoIdByName(me_name, id)) {
      const cscdqm::EMUHistoDef def(id);
      cscdqm::MonitorObject* mo = new CSCMonitorObject(me);
      config.fnPutHisto(def, mo);
    }
  }

  // *
  // *  Putting histograms to internal cache: EventInfo
  // * /

  {
    std::string name = DIR_EVENTINFO;
    name += "reportSummaryMap";
    MonitorElement* me = dbe->get(name);
    if (me) {
      const cscdqm::EMUHistoDef def(cscdqm::h::EMU_CSC_STATS_SUMMARY);
      cscdqm::MonitorObject* mo = new CSCMonitorObject(me);
      config.fnPutHisto(def, mo);
    }
  }

  config.incNEvents();
  dispatcher->updateFractionAndEfficiencyHistos();

}
*/

// void CSCOfflineClient::bookHistograms(DQMStore::IBooker & ib, edm::Run const &, edm::EventSetup const &)
void CSCOfflineClient::dqmEndJob(DQMStore::IBooker& ib, DQMStore::IGetter& igetter)
{
  ibooker = &ib;
  dispatcher->book();
  if (maskedHW.size() != 0)
     dispatcher->maskHWElements(maskedHW);


  /*
   *  Putting histograms to internal cache: EMU stuff
   */
  ibooker->setCurrentFolder(config.getFOLDER_EMU());
  std::vector<std::string> me_names = igetter.getMEs();
  for (std::vector<std::string>::iterator iter = me_names.begin(); iter != me_names.end(); iter++) {
    std::string me_name = *iter;
    MonitorElement* me = igetter.get(config.getFOLDER_EMU() + me_name);
    cscdqm::HistoId id;
    if (me && cscdqm::HistoDef::getHistoIdByName(me_name, id)) {
      const cscdqm::EMUHistoDef def(id);
      cscdqm::MonitorObject* mo = new CSCMonitorObject(me);
      config.fnPutHisto(def, mo);
    }
  }

  /*
   *  Putting histograms to internal cache: EventInfo
   */

  {
    std::string name = DIR_EVENTINFO;
    name += "reportSummaryMap";
    MonitorElement* me = igetter.get(name);
    if (me) {
      const cscdqm::EMUHistoDef def(cscdqm::h::EMU_CSC_STATS_SUMMARY);
      cscdqm::MonitorObject* mo = new CSCMonitorObject(me);
      config.fnPutHisto(def, mo);
    }
  }

  config.incNEvents();
  dispatcher->updateFractionAndEfficiencyHistos();

}

/**
 * @brief  Book Monitor Object on Request.
 * @param  req Request.
 * @return MonitorObject created.
 */
cscdqm::MonitorObject* CSCOfflineClient::bookMonitorObject(const cscdqm::HistoBookRequest& req) {

  cscdqm::MonitorObject *me = NULL;
  std::string name = req.hdef->getName();

  std::string path = req.folder;
  if (req.hdef->getPath().size() > 0) {
    path = path + req.hdef->getPath() + "/";
  }
  
  ibooker->cd();
  ibooker->setCurrentFolder(path);

  if (req.htype == cscdqm::INT) {
    me = new CSCMonitorObject(ibooker->bookInt(name));
    me->Fill(req.default_int);
  } else 
  if (req.htype == cscdqm::FLOAT) {
    if (req.hdef->getId() == cscdqm::h::PAR_REPORT_SUMMARY) {
      ibooker->cd();
      ibooker->setCurrentFolder(DIR_EVENTINFO);
    } else if (cscdqm::Utility::regexMatch("^PAR_DCS_", cscdqm::h::keys[req.hdef->getId()])) {
      ibooker->cd();
      ibooker->setCurrentFolder(DIR_DCSINFO);
    } else if (cscdqm::Utility::regexMatch("^PAR_DAQ_", cscdqm::h::keys[req.hdef->getId()])) {
      ibooker->cd();
      ibooker->setCurrentFolder(DIR_DAQINFO);
    } else if (cscdqm::Utility::regexMatch("^PAR_CRT_", cscdqm::h::keys[req.hdef->getId()])) {
      ibooker->cd();
      ibooker->setCurrentFolder(DIR_CRTINFO);
    }
    me = new CSCMonitorObject(ibooker->bookFloat(name));
    me->Fill(req.default_float);
  } else 
  if (req.htype == cscdqm::STRING) {
    me = new CSCMonitorObject(ibooker->bookString(name, req.default_string));
  }

  return me;

}
