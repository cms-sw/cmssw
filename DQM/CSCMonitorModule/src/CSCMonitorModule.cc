/*
 * =====================================================================================
 *
 *       Filename:  CSCMonitorModule.cc
 *
 *    Description:  CSC Monitor module class
 *
 *        Version:  1.0
 *        Created:  04/18/2008 02:26:43 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), Valdas.Rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#include "DQM/CSCMonitorModule/interface/CSCMonitorModule.h"

/**
 * @brief  MonitorModule Constructor
 * @param  ps ParameterSet
 * @return
 */
CSCMonitorModule::CSCMonitorModule(const edm::ParameterSet& ps){
    
    parameters = ps;
    
    edm::FileInPath fp;
    
    hitBookDDU     = parameters.getUntrackedParameter<bool>("hitBookDDU", true);
    examinerMask   = parameters.getUntrackedParameter<unsigned int>("ExaminerMask", 0x7FB7BF6);
    examinerForce  = parameters.getUntrackedParameter<bool>("ExaminerForce", false);
    examinerOutput = parameters.getUntrackedParameter<bool>("ExaminerOutput", false);
    examinerCRCKey = parameters.getUntrackedParameter<unsigned int>("ExaminerCRCKey", 0);
    fractUpdateKey = parameters.getUntrackedParameter<unsigned int>("FractUpdateKey", 1);
    fractUpdateEvF = parameters.getUntrackedParameter<unsigned int>("FractUpdateEventFreq", 1);
    effParameters  = parameters.getUntrackedParameter<edm::ParameterSet>("effParameters");
    
    // Get ant apply dead HW element masks if any
    std::vector<std::string> hwMasks = parameters.getUntrackedParameter<std::vector<std::string> >("AddressMask");
    unsigned int masks_ok = summary.setMaskedHWElements(hwMasks);
    LOGINFO("HW Address Masks") << masks_ok << " out of " << hwMasks.size() << " HW Masks are accepted.";
    
    // Initialize some variables
    inputObjectsTag = parameters.getUntrackedParameter<edm::InputTag>("InputObjects", (edm::InputTag)"source");
    monitorName = parameters.getUntrackedParameter<std::string>("monitorName", "CSC");
    fp = parameters.getParameter<edm::FileInPath>("BookingFile");
    bookingFile = fp.fullPath();
    
    rootDir = monitorName + "/";
    nEvents = 0;
    nCSCEvents = 0;
    L1ANumber = 0;
    
    // Loading histogram collection from XML file
    if(loadCollection()) {
        LOGERROR("initialize") << "Histogram booking failed .. exiting.";
        return;
    }
    
    // Get back-end interface
    dbe = edm::Service<DQMStore>().operator->();
    
    this->init = false;
    
}

/**
 * @brief  MonitorModule Destructor
 * @param
 * @return
 */
CSCMonitorModule::~CSCMonitorModule(){
    
    
}

/**
 * @brief  Function that is being executed prior job. Actuall histogram
 * bookings are done here. All initialization tasks as well.
 * @param  c Event setup object
 * @return
 */
void CSCMonitorModule::beginJob(const edm::EventSetup& c){
    
}

void CSCMonitorModule::setup() {
    
    // Base folder for the contents of this job
    dbe->setCurrentFolder(rootDir + SUMMARY_FOLDER);
    
    // Book EMU level histograms
    book("EMU");
    
    // Book detector summary histograms and stuff
    MonitorElement* me;
    
    // reportSummary stuff booking
    dbe->setCurrentFolder(rootDir + EVENTINFO_FOLDER);
    book("EventInfo");
    me = dbe->bookFloat("reportSummary");
    me->Fill(-1.0);
    
    // reportSummaryContents booking
    dbe->setCurrentFolder(rootDir + SUMCONTENTS_FOLDER);
    cscdqm::Address adr;
    adr.mask.chamber = adr.mask.layer = adr.mask.cfeb = adr.mask.hv = false;
    adr.mask.side = true;
    for (adr.side = 1; adr.side <= N_SIDES; adr.side++) {
        adr.mask.station = adr.mask.ring = false;
        me = dbe->bookFloat(summary.getDetector().AddressName(adr));
        me->Fill(-1.0);
        adr.mask.station = true;
        for (adr.station = 1; adr.station <= N_STATIONS; adr.station++) {
            adr.mask.ring = false;
            dbe->bookFloat(summary.getDetector().AddressName(adr));
            me->Fill(-1.0);
            if (summary.getDetector().NumberOfRings(adr.station) > 1) {
                adr.mask.ring = true;
                for (adr.ring = 1; adr.ring <= summary.getDetector().NumberOfRings(adr.station); adr.ring++) {
                    dbe->bookFloat(summary.getDetector().AddressName(adr));
                    me->Fill(-1.0);
                }
            }
        }
    }
    
    // Write down STATS parameters
    dbe->setCurrentFolder(rootDir + EVENTINFO_FOLDER + "effParameters");
    std::vector<std::string> effParamNames = effParameters.getParameterNamesForType<float>(false);
    for (std::vector<std::string>::iterator iter = effParamNames.begin(); iter != effParamNames.end(); iter++) {
        me = dbe->bookFloat(*iter);
        me->Fill(effParameters.getUntrackedParameter<double>(*iter, -1.0));
    }
    
    LOGINFO("Fraction histograms") << " updateKey = " << fractUpdateKey << ", update on events (freq) = " << fractUpdateEvF;
    
    this->init = true;
    
}

/**
 * @brief  Main Analyzer function that receives Events andi starts
 * the acctual analysis (histogram filling) and so on chain.
 * @param  e Event
 * @param  c EventSetup
 * @return
 */
void CSCMonitorModule::analyze(const edm::Event& e, const edm::EventSetup& c){
    
    // Get crate mapping from database
    edm::ESHandle<CSCCrateMap> hcrate;
    c.get<CSCCrateMapRcd>().get(hcrate);
    pcrate = hcrate.product();
    
    // Lets initialize MEs if it was not done so before
    if (!this->init) {
        this->setup();
    }
    
    // Pass event to monitoring chain
    monitorEvent(e);
    
    // Update fractional histograms if appropriate
    if (nCSCEvents > 0 && fractUpdateKey.test(2) && (nEvents % fractUpdateEvF) == 0) {
        updateFracHistos();
    }
    
}

/**
 * @brief  Function that is being executed at the very end of the job.
 * Histogram savings, calculation of fractional histograms, etc. should be done
 * here.
 * @param
 * @return
 */
void CSCMonitorModule::endJob(void){
    
}

void CSCMonitorModule::beginRun(const edm::Run& r, const edm::EventSetup& context) {
    
}

void CSCMonitorModule::endRun(const edm::Run& r, const edm::EventSetup& context) {
    if (nCSCEvents > 0 && fractUpdateKey.test(0)) {
        updateFracHistos();
    }
}

void CSCMonitorModule::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) {
    if (nCSCEvents > 0 && fractUpdateKey.test(1)) {
        updateFracHistos();
    }
}

const bool CSCMonitorModule::getCSCFromMap(const int crate, const int slot, int& csctype, int& cscposition) const {
    
    if (crate < 1 || crate > 60 || slot < 1 || slot > 10) return false;

    CSCDetId cid = pcrate->detId(crate, slot, 0, 0);
    cscposition  = cid.chamber();
    int iring    = cid.ring();
    int istation = cid.station();
    int iendcap  = cid.endcap();
    
    std::string tlabel = CSCUtility::getCSCTypeLabel(iendcap, istation, iring);
    csctype = CSCUtility::getCSCTypeBin(tlabel);

    return true;
    
}
