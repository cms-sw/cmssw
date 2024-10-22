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

#include "CSCMonitorModule.h"

/**
 * @brief  Constructor.
 * @param  ps Parameters.
 */
CSCMonitorModule::CSCMonitorModule(const edm::ParameterSet& ps)
    : hcrateToken_(esConsumes<CSCCrateMap, CSCCrateMapRcd>()) {
  edm::FileInPath fp;

  inputTag = ps.getUntrackedParameter<edm::InputTag>("InputObjects", (edm::InputTag)INPUT_TAG_LABEL);
  prebookEffParams = ps.getUntrackedParameter<bool>("PREBOOK_EFF_PARAMS", false);
  processDcsScalers = ps.getUntrackedParameter<bool>("PROCESS_DCS_SCALERS", true);
  edm::ParameterSet params = ps.getUntrackedParameter<edm::ParameterSet>("EventProcessor");
  config.load(params);

  fp = ps.getParameter<edm::FileInPath>("BOOKING_XML_FILE");
  config.setBOOKING_XML_FILE(fp.fullPath());

  // dbe = edm::Service<DQMStore>().operator->();

#ifdef DQMLOCAL
  dispatcher = new cscdqm::Dispatcher(&config, const_cast<CSCMonitorModule*>(this));
#endif
#ifdef DQMGLOBAL
  //  edm::ConsumesCollector coco( consumesCollector() );
  dispatcher = new cscdqm::Dispatcher(&config, const_cast<CSCMonitorModule*>(this), inputTag, consumesCollector());
  dcstoken = consumes<DcsStatusCollection>(edm::InputTag("scalersRawToDigi"));

#endif

  dispatcher->init();

  if (ps.exists("MASKEDHW")) {
    maskedHW = ps.getUntrackedParameter<std::vector<std::string> >("MASKEDHW");
    //dispatcher->maskHWElements(maskedHW);
  }
}

/**
 * @brief  Destructor.
 */
CSCMonitorModule::~CSCMonitorModule() {
  if (dispatcher)
    delete dispatcher;
}

/**
 * @brief  Begin the run.
 * @param  r Run object
 * @param  c Event setup
 */

/*
void CSCMonitorModule::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  if (prebookEffParams) {
    dispatcher->updateFractionAndEfficiencyHistos();
  }

}
*/

/**
 * @brief  Analyze Event.
 * @param  e Event to analyze
 * @param  c Event Setup
 */
void CSCMonitorModule::analyze(const edm::Event& e, const edm::EventSetup& c) {
  // Get crate mapping from database
  edm::ESHandle<CSCCrateMap> hcrate;
  /// Use esConsumes to get crate mapping
  hcrate = c.getHandle(hcrateToken_);
  pcrate = hcrate.product();

  cscdqm::HWStandbyType standby;

  // Get DCS status scalers
  if (processDcsScalers) {
    edm::Handle<DcsStatusCollection> dcsStatus;
#ifdef DQMLOCAL
    if (e.getByToken(dcstoken, dcsStatus)) {
#endif
#ifdef DQMGLOBAL
      if (e.getByToken(dcstoken, dcsStatus)) {
#endif
        DcsStatusCollection::const_iterator dcsStatusItr = dcsStatus->begin();
        for (; dcsStatusItr != dcsStatus->end(); ++dcsStatusItr) {
          standby.applyMeP(dcsStatusItr->ready(DcsStatus::CSCp));
          standby.applyMeM(dcsStatusItr->ready(DcsStatus::CSCm));
        }
      }
      standby.process = true;
    }

    dispatcher->processEvent(e, inputTag, standby);
  }

  /**
 * @brief  Book Histograms in the beginRun.
 * @param  ib - DQMStore::IBooker interface access object
 * @param  edm::Run const & - not used
 * @param  edm::EventSetup const & - not used
 */
  void CSCMonitorModule::bookHistograms(DQMStore::IBooker & ib, edm::Run const&, edm::EventSetup const&) {
    /** Store pointer to IBooker to use it in ::bookMonitorObject() callback function used by the Dispatcher::book() method **/
    ibooker = &ib;

    /** 
    * Call Dispatcher histogram booking method to pre-boook all available histograms before processing data. 
    * New PREBOOK_ALL_HISTOS config option should be set to true (default).
    * That should disable original CSC on-the-fly histo booking during data processing. 
    * Changed for multi-threaded framework compatibility.
    * (pre-booking of all histos is not most efficient way for CSCi DQM). 
    *   
    **/
    dispatcher->book();

    if (!maskedHW.empty())
      dispatcher->maskHWElements(maskedHW);
    if (prebookEffParams) {
      dispatcher->updateFractionAndEfficiencyHistos();
    }
  }

  /**
 * @brief  Book Monitor Object on Request.
 * @param  req Request.
 * @return MonitorObject created.
 */
  cscdqm::MonitorObject* CSCMonitorModule::bookMonitorObject(const cscdqm::HistoBookRequest& req) {
    cscdqm::MonitorObject* me = nullptr;
    std::string name = req.hdef->getName();

    std::string path = req.folder;
    if (!req.hdef->getPath().empty()) {
      path = path + req.hdef->getPath() + "/";
    }

    //std::cout << "Moving to " << path << " for name = " << name << " with fullPath = " << req.hdef->getFullPath() << "\n";

    //dbe->setCurrentFolder(path);
    ibooker->cd();
    ibooker->setCurrentFolder(path);

    if (req.htype == cscdqm::INT) {
      me = new CSCMonitorObject(ibooker->bookInt(name));
      me->Fill(req.default_int);
    } else if (req.htype == cscdqm::FLOAT) {
      if (req.hdef->getId() == cscdqm::h::PAR_REPORT_SUMMARY || req.hdef->getId() == cscdqm::h::PAR_CRT_SUMMARY ||
          req.hdef->getId() == cscdqm::h::PAR_DAQ_SUMMARY || req.hdef->getId() == cscdqm::h::PAR_DCS_SUMMARY) {
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
    } else if (req.htype == cscdqm::STRING) {
      me = new CSCMonitorObject(ibooker->bookString(name, req.default_string));
    } else if (req.htype == cscdqm::H1D) {
      me = new CSCMonitorObject(ibooker->book1D(name, req.title, req.nchX, req.lowX, req.highX));
    } else if (req.htype == cscdqm::H2D) {
      if (req.hdef->getId() == cscdqm::h::EMU_CSC_STATS_SUMMARY) {
        ibooker->cd();
        ibooker->setCurrentFolder(DIR_EVENTINFO);
        name = "reportSummaryMap";
      }
      me = new CSCMonitorObject(
          ibooker->book2D(name, req.title, req.nchX, req.lowX, req.highX, req.nchY, req.lowY, req.highY));
    } else if (req.htype == cscdqm::H3D) {
      me = new CSCMonitorObject(ibooker->book3D(
          name, req.title, req.nchX, req.lowX, req.highX, req.nchY, req.lowY, req.highY, req.nchZ, req.lowZ, req.highZ));
    } else if (req.htype == cscdqm::PROFILE) {
      me = new CSCMonitorObject(ibooker->bookProfile(
          name, req.title, req.nchX, req.lowX, req.highX, req.nchY, req.lowY, req.highY, req.option.c_str()));
    } else if (req.htype == cscdqm::PROFILE2D) {
      me = new CSCMonitorObject(ibooker->bookProfile2D(name,
                                                       req.title,
                                                       req.nchX,
                                                       req.lowX,
                                                       req.highX,
                                                       req.nchY,
                                                       req.lowY,
                                                       req.highY,
                                                       req.nchZ,
                                                       req.lowZ,
                                                       req.highZ,
                                                       req.option.c_str()));
    }

    return me;
  }
