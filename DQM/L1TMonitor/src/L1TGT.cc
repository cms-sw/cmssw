/**
 * \class L1TGT
 *
 *
 * Description: DQM for L1 Global Trigger.
 *
 * \author J. Berryhill, I. Mikulec
 * \author Vasile Mihai Ghete - HEPHY Vienna
 *
 * $Date$
 * $Revision$
 *
 */

#include "DQM/L1TMonitor/interface/L1TGT.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"

L1TGT::L1TGT(const edm::ParameterSet& ps) :
            gtSource_(ps.getParameter<edm::InputTag> ("gtSource")),
            gtEvmSource_(ps.getParameter<edm::InputTag> ("gtEvmSource")),
            verbose_(ps.getUntrackedParameter<bool> ("verbose", false)),
            m_dbe(0),
            //
            m_nrEvJob(0), m_nrEvRun(0),
            preGps_(0ULL), preOrb_(0ULL),
            m_previousLS(-1),
            m_previousPfIndex(-1)
{

    m_histFolder = ps.getUntrackedParameter<std::string> ("HistFolder",
            "L1T/L1TGT");

    m_dbe = edm::Service<DQMStore>().operator->();
    if (m_dbe == 0) {
        edm::LogInfo("L1TGT") << "\n Unable to get DQMStore service.";
    } else {

        m_dbe->setVerbose(0);
        m_dbe->setCurrentFolder(m_histFolder);

    }

    // reserve space for 1000 LS
    m_pairLsNumberPfIndex.reserve(1000);
}

L1TGT::~L1TGT() {

    // empty
}

void L1TGT::beginJob() {

    m_nrEvJob = 0;

    preGps_ = 0ULL;
    preOrb_ = 0ULL;

}

void L1TGT::beginRun(const edm::Run& iRun, const edm::EventSetup& evSetup) {

    m_nrEvRun = 0;

    m_dbe = edm::Service<DQMStore>().operator->();

    if (m_dbe == 0) {
        edm::LogInfo("L1TGT") << "\n Unable to get DQMStore service.";
    } else {

        // clean up directory
        m_dbe->setCurrentFolder(m_histFolder);
        if (m_dbe->dirExists(m_histFolder)) {
            m_dbe->rmdir(m_histFolder);
        }

        m_dbe->setCurrentFolder(m_histFolder);

    }

    // book histograms
    bookHistograms();

    // clear bookkeeping for prescale factor change
    m_pairLsNumberPfIndex.clear();

}

void L1TGT::beginLuminosityBlock(const edm::LuminosityBlock& iLumi,
        const edm::EventSetup& evSetup) {

    //

}


//
void L1TGT::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {

    m_nrEvJob++;

    if (verbose_) {
        edm::LogInfo("L1TGT") << "L1TGT: analyze...." << std::endl;
    }

    // initialize Bx, orbit number, luminosity segment number to invalid value
    int tcsBx = -1;
    int gtfeEvmBx = -1;

    long long int orbitTcs = -1;
    int orbitEvmFdl = -1;

    int lsTcs = -1;
    int lsEvmFdl = -1;

    // get once only the LS block number, to be used in many histograms
    const int lsNumber = iEvent.luminosityBlock();

    // open EVM readout record if available
    edm::Handle<L1GlobalTriggerEvmReadoutRecord> gtEvmReadoutRecord;
    iEvent.getByLabel(gtEvmSource_, gtEvmReadoutRecord);

    if (!gtEvmReadoutRecord.isValid()) {
        edm::LogInfo("L1TGT")
                << "can't find L1GlobalTriggerEvmReadoutRecord with label "
                << gtSource_.label();
    } else {

        // get all info from the EVM record if available and fill the histograms

        const L1GtfeWord& gtfeEvmWord = gtEvmReadoutRecord->gtfeWord();
        const L1GtfeExtWord& gtfeEvmExtWord = gtEvmReadoutRecord->gtfeWord();

        gtfeEvmBx = gtfeEvmWord.bxNr();
        int gtfeEvmActiveBoards = gtfeEvmWord.activeBoards();

        if (isActive(gtfeEvmActiveBoards, TCS)) { // if TCS present in the record

            const L1TcsWord& tcsWord = gtEvmReadoutRecord->tcsWord();

            tcsBx = tcsWord.bxNr();
            orbitTcs = tcsWord.orbitNr();
            lsTcs = tcsWord.luminositySegmentNr();

            event_type->Fill(tcsWord.triggerType());
            orbit_lumi->Fill(lsNumber, orbitTcs);

            trigger_number->Fill(tcsWord.partTrigNr());
            event_number->Fill(tcsWord.eventNr());

            trigger_lumi->Fill(lsNumber, tcsWord.partTrigNr());
            event_lumi->Fill(lsNumber, tcsWord.eventNr());
            evnum_trignum_lumi->Fill(lsNumber,
                    double(tcsWord.eventNr()) / double(tcsWord.partTrigNr()));

            boost::uint16_t master = gtfeEvmExtWord.bstMasterStatus();
            boost::uint32_t turnCount = gtfeEvmExtWord.turnCountNumber();
            boost::uint32_t lhcFill = gtfeEvmExtWord.lhcFillNumber();
            boost::uint16_t beam = gtfeEvmExtWord.beamMode();
            boost::uint16_t momentum = gtfeEvmExtWord.beamMomentum();
            boost::uint32_t intensity1 = gtfeEvmExtWord.totalIntensityBeam1();
            boost::uint32_t intensity2 = gtfeEvmExtWord.totalIntensityBeam2();

            BST_MasterStatus->Fill(lsNumber, (float) (master));
            BST_turnCountNumber->Fill(lsNumber, (float) (turnCount));
            BST_lhcFillNumber->Fill((float) (lhcFill % 1000));
            BST_beamMode->Fill(lsNumber, (float) (beam));

            BST_beamMomentum->Fill(lsNumber, (float) (momentum));
            BST_intensityBeam1->Fill(lsNumber, (float) (intensity1));
            BST_intensityBeam2->Fill(lsNumber, (float) (intensity2));

            if (verbose_) {
                edm::LogInfo("L1TGT") << " check mode = " << beam << "    momentum " << momentum
                        << " int2 " << intensity2 << std::endl;
            }

            boost::uint64_t gpsr = gtfeEvmExtWord.gpsTime();
            boost::uint64_t gpshi = (gpsr >> 32) & 0xffffffff;
            boost::uint64_t gpslo = gpsr & 0xffffffff;
            boost::uint64_t gps = gpshi * 1000000 + gpslo;
            //  edm::LogInfo("L1TGT") << "  gpsr = " << std::hex << gpsr << " hi=" << gpshi << " lo=" << gpslo << " gps=" << gps << std::endl;

            Long64_t delorb = orbitTcs - preOrb_;
            Long64_t delgps = gps - preGps_;
            Double_t freq = -1.;

            if (delgps > 0) {
                freq = ((Double_t)(delorb)) * 3564. / ((Double_t)(delgps));
            }

            if (delorb > 0) {
                gpsfreq->Fill(freq);
                gpsfreqwide->Fill(freq);
                gpsfreqlum->Fill(lsNumber, freq);
                if (verbose_) {
                    if (freq > 200.) {
                        edm::LogInfo("L1TGT") << " preOrb_ = " << preOrb_ << " orbitTcs=" << orbitTcs
                                << " delorb=" << delorb << std::hex << " preGps_="
                                << preGps_ << " gps=" << gps << std::dec
                                << " delgps=" << delgps << " freq=" << freq
                                << std::endl;

                    }
                }
            }

            preGps_ = gps;
            preOrb_ = orbitTcs;

        }

        // get info from FDL if active
        if (isActive(gtfeEvmActiveBoards, FDL)) {
            const L1GtFdlWord& fdlWord = gtEvmReadoutRecord->gtFdlWord();

            orbitEvmFdl = fdlWord.orbitNr();
            lsEvmFdl = fdlWord.lumiSegmentNr();
        }

        if ((orbitTcs >= 0) && (orbitEvmFdl >= 0)) {

            int diffOrbit = static_cast<float> (orbitTcs - orbitEvmFdl);
            edm::LogInfo("L1TGT") << "\n orbitTcs = " << orbitTcs << " orbitEvmFdl = "
                    << orbitEvmFdl << " diffOrbit = " << diffOrbit
                    << " orbitEvent = " << iEvent.orbitNumber() << std::endl;

            if (diffOrbit >= MaxOrbitNrDeltaTcsEvmFdl) {
                m_monOrbitNrDeltaTcsEvmFdl->Fill(MaxOrbitNrDeltaTcsEvmFdl);

            } else if (diffOrbit <= -MaxOrbitNrDeltaTcsEvmFdl) {
                m_monOrbitNrDeltaTcsEvmFdl->Fill(-MaxOrbitNrDeltaTcsEvmFdl);

            } else {
                m_monOrbitNrDeltaTcsEvmFdl->Fill(diffOrbit);
                m_monOrbitNrDeltaTcsEvmFdlLs->Fill(lsNumber,
                        diffOrbit);

            }

        } else {

            if (orbitTcs >= 0) {
                // EVM_FDL error
                m_monOrbitNrDeltaTcsEvmFdl->Fill(MaxOrbitNrDeltaTcsEvmFdl);
            } else if (orbitEvmFdl >= 0) {
                // TCS error
                m_monOrbitNrDeltaTcsEvmFdl->Fill(-MaxOrbitNrDeltaTcsEvmFdl);

            } else {
                // TCS and EVM_FDL error
                m_monOrbitNrDeltaTcsEvmFdl->Fill(-MaxOrbitNrDeltaTcsEvmFdl);
                m_monOrbitNrDeltaTcsEvmFdl->Fill(MaxOrbitNrDeltaTcsEvmFdl);
            }
        }

        if ((lsTcs >= 0) && (lsEvmFdl >= 0)) {

            int diffLs = static_cast<float> (lsTcs - lsEvmFdl);
            edm::LogInfo("L1TGT") << "\n lsTcs = " << lsTcs << " lsEvmFdl = " << lsEvmFdl
                    << " diffLs = " << diffLs << " lsEvent = "
                    << lsNumber << std::endl;

            if (diffLs >= MaxLsNrDeltaTcsEvmFdl) {
                m_monLsNrDeltaTcsEvmFdl->Fill(MaxLsNrDeltaTcsEvmFdl);

            } else if (diffLs <= -MaxLsNrDeltaTcsEvmFdl) {
                m_monLsNrDeltaTcsEvmFdl->Fill(-MaxLsNrDeltaTcsEvmFdl);

            } else {
                m_monLsNrDeltaTcsEvmFdl->Fill(diffLs);
                m_monLsNrDeltaTcsEvmFdlLs->Fill(lsNumber, diffLs);

            }

        } else {

            if (lsTcs >= 0) {
                // EVM_FDL error
                m_monLsNrDeltaTcsEvmFdl->Fill(MaxLsNrDeltaTcsEvmFdl);
            } else if (lsEvmFdl >= 0) {
                // TCS error
                m_monLsNrDeltaTcsEvmFdl->Fill(-MaxLsNrDeltaTcsEvmFdl);

            } else {
                // TCS and EVM_FDL error
                m_monLsNrDeltaTcsEvmFdl->Fill(-MaxLsNrDeltaTcsEvmFdl);
                m_monLsNrDeltaTcsEvmFdl->Fill(MaxLsNrDeltaTcsEvmFdl);
            }
        }

    }

    // open GT DAQ readout record - exit if failed
    edm::Handle<L1GlobalTriggerReadoutRecord> gtReadoutRecord;
    iEvent.getByLabel(gtSource_, gtReadoutRecord);

    if (!gtReadoutRecord.isValid()) {
        edm::LogInfo("L1TGT")
                << "can't find L1GlobalTriggerReadoutRecord with label "
                << gtSource_.label();
        return;
    }

    // initialize bx's to invalid value
    int gtfeBx = -1;
    int fdlBx[2] = { -1, -1 };
    int psbBx[2][7] = { { -1, -1, -1, -1, -1, -1, -1 }, { -1, -1, -1, -1, -1,
            -1, -1 } };
    int gmtBx = -1;

    // get info from GTFE DAQ record
    const L1GtfeWord& gtfeWord = gtReadoutRecord->gtfeWord();
    gtfeBx = gtfeWord.bxNr();
    gtfe_bx->Fill(gtfeBx);
    setupversion_lumi->Fill(lsNumber, gtfeWord.setupVersion());
    int gtfeActiveBoards = gtfeWord.activeBoards();

    // look for GMT readout collection from the same source if GMT active
    if (isActive(gtfeActiveBoards, GMT)) {
        edm::Handle<L1MuGMTReadoutCollection> gmtReadoutCollection;
        iEvent.getByLabel(gtSource_, gmtReadoutCollection);

        if (gmtReadoutCollection.isValid()) {
            gmtBx = gmtReadoutCollection->getRecord().getBxNr();
        }
    }

    // get info from FDL if active (including decision word)
    if (isActive(gtfeActiveBoards, FDL)) {
        const L1GtFdlWord& fdlWord = gtReadoutRecord->gtFdlWord();
        fdlBx[0] = fdlWord.bxNr();
        fdlBx[1] = fdlWord.localBxNr();

        /// get Global Trigger algo and technical trigger bit statistics
        const DecisionWord& gtDecisionWord = gtReadoutRecord->decisionWord();
        const TechnicalTriggerWord& gtTTWord =
                gtReadoutRecord->technicalTriggerWord();

        int dbitNumber = 0;
        DecisionWord::const_iterator GTdbitItr;
        algo_bits->Fill(-1.); // fill underflow to normalize
        for (GTdbitItr = gtDecisionWord.begin(); GTdbitItr
                != gtDecisionWord.end(); GTdbitItr++) {
            if (*GTdbitItr) {
                algo_bits->Fill(dbitNumber);
                algo_bits_lumi->Fill(lsNumber, dbitNumber);
                int dbitNumber1 = 0;
                DecisionWord::const_iterator GTdbitItr1;
                for (GTdbitItr1 = gtDecisionWord.begin(); GTdbitItr1
                        != gtDecisionWord.end(); GTdbitItr1++) {
                    if (*GTdbitItr1)
                        algo_bits_corr->Fill(dbitNumber, dbitNumber1);
                    dbitNumber1++;
                }
                int tbitNumber1 = 0;
                TechnicalTriggerWord::const_iterator GTtbitItr1;
                for (GTtbitItr1 = gtTTWord.begin(); GTtbitItr1
                        != gtTTWord.end(); GTtbitItr1++) {
                    if (*GTtbitItr1)
                        tt_bits_corr->Fill(dbitNumber, tbitNumber1);
                    tbitNumber1++;
                }
            }
            dbitNumber++;
        }

        int tbitNumber = 0;
        TechnicalTriggerWord::const_iterator GTtbitItr;
        tt_bits->Fill(-1.); // fill underflow to normalize
        for (GTtbitItr = gtTTWord.begin(); GTtbitItr != gtTTWord.end(); GTtbitItr++) {
            if (*GTtbitItr) {
                tt_bits->Fill(tbitNumber);
                tt_bits_lumi->Fill(lsNumber, tbitNumber);
                int tbitNumber1 = 0;
                TechnicalTriggerWord::const_iterator GTtbitItr1;
                for (GTtbitItr1 = gtTTWord.begin(); GTtbitItr1
                        != gtTTWord.end(); GTtbitItr1++) {
                    if (*GTtbitItr1)
                        tt_bits_corr->Fill(tbitNumber, tbitNumber1);
                    tbitNumber1++;
                }
            }
            tbitNumber++;
        }


        // fill the index of actual prescale factor set
        // the index for technical triggers and algorithm trigger is the same (constraint in L1 GT TS)
        // so we read only pfIndexAlgoTrig (boost::uint16_t)

        const int pfIndexAlgoTrig = fdlWord.gtPrescaleFactorIndexAlgo();
        m_monL1PrescaleFactorSet->Fill(lsNumber,
                static_cast<float> (pfIndexAlgoTrig));

        //

        // check that the combination (pfIndex, lsNumber) is not already included
        // to avoid fake entries due to different event order

        std::pair<int, int> pairLsPfi = std::make_pair(lsNumber,
                pfIndexAlgoTrig);

        CItVecPair cIt = find(m_pairLsNumberPfIndex.begin(),
                m_pairLsNumberPfIndex.end(), pairLsPfi);

        if (cIt == m_pairLsNumberPfIndex.end()) {

            m_pairLsNumberPfIndex.push_back(pairLsPfi);

            m_previousPfIndex = pfIndexAlgoTrig;
            m_previousLS = lsNumber;
        }

    }

    // get info from active PSB's
    int ibit = PSB9; // first psb
    // for now hardcode psb id's - TODO - get them from Vasile's board maps...
    int psbID[7] = { 0xbb09, 0xbb0d, 0xbb0e, 0xbb0f, 0xbb13, 0xbb14, 0xbb15 };
    for (int i = 0; i < 7; i++) {
        if (isActive(gtfeActiveBoards, ibit)) {
            L1GtPsbWord psbWord = gtReadoutRecord->gtPsbWord(psbID[i]);
            psbBx[0][i] = psbWord.bxNr();
            psbBx[1][i] = psbWord.localBxNr();
        }
        ibit++;
    }

    //fill the dbx histo
    if (gtfeEvmBx > -1)
        dbx_module->Fill(0., gtfeEvmBx - gtfeBx);
    if (tcsBx > -1)
        dbx_module->Fill(1., tcsBx - gtfeBx);
    for (int i = 0; i < 2; i++) {
        if (fdlBx[i] > -1)
            dbx_module->Fill(2. + i, fdlBx[i] - gtfeBx);
    }
    for (int j = 0; j < 7; j++) {
        for (int i = 0; i < 2; i++) {
            if (psbBx[i][j] > -1)
                dbx_module->Fill(4. + i + 2 * j, psbBx[i][j] - gtfeBx);
        }
    }
    if (gmtBx > -1)
        dbx_module->Fill(18., gmtBx - gtfeBx);

}


// end section
void L1TGT::endLuminosityBlock(const edm::LuminosityBlock& iLumi,
        const edm::EventSetup& evSetup) {

    edm::LogInfo("L1TGT")
            << "\n  endLuminosityBlock:: Prescale factor indices used in a LS "
            << std::endl;
    for (CItVecPair cIt = m_pairLsNumberPfIndex.begin(); cIt
            != m_pairLsNumberPfIndex.end(); ++cIt) {

        edm::LogVerbatim("L1TGT") << "  lsNumber = " << (*cIt).first
                << " pfIndex = " << (*cIt).second << std::endl;
    }
    edm::LogVerbatim("L1TGT") << std::endl;

    // sort the vector (for pairs: sort after first argument, then after the second argument)
    std::sort(m_pairLsNumberPfIndex.begin(), m_pairLsNumberPfIndex.end());

}

void L1TGT::endRun(const edm::Run& iRrun, const edm::EventSetup& evSetup) {

    edm::LogInfo("L1TGT")
            << "\n  endRun:: Prescale factor indices used in a LS "
            << std::endl;
    for (CItVecPair cIt = m_pairLsNumberPfIndex.begin(); cIt
            != m_pairLsNumberPfIndex.end(); ++cIt) {

        edm::LogVerbatim("L1TGT") << "  lsNumber = " << (*cIt).first
                << " pfIndex = " << (*cIt).second << std::endl;
    }
    edm::LogVerbatim("L1TGT") << std::endl;

    // clear the content of the vector
    m_pairLsNumberPfIndex.clear();

}


void L1TGT::endJob() {

    if (verbose_) {
        edm::LogInfo("L1TGT") << "\n Analyzed " << m_nrEvJob << " events";
    }

    return;
}


// book all histograms for the module
void L1TGT::bookHistograms() {

    if (m_dbe) {
        m_dbe->setCurrentFolder(m_histFolder);

        algo_bits = m_dbe->book1D("algo_bits", "GT algo bits", 128, -0.5, 127.5);
        algo_bits->setAxisTitle("algorithm bits", 1);

        algo_bits_corr = m_dbe->book2D("algo_bits_corr",
                "GT algo bit correlation",
                128, -0.5, 127.5, 128, -0.5, 127.5);
        algo_bits_corr->setAxisTitle("algorithm bits", 1);
        algo_bits_corr->setAxisTitle("algorithm bits", 2);

        tt_bits = m_dbe->book1D("tt_bits",
                "GT technical trigger bits",
                64, -0.5, 63.5);
        tt_bits->setAxisTitle("technical trigger bits", 1);

        tt_bits_corr = m_dbe->book2D("tt_bits_corr",
                "GT tech. trig. bit correlation",
                64, -0.5, 63.5, 64, -0.5, 63.5);
        tt_bits_corr->setAxisTitle("technical trigger bits", 1);
        tt_bits_corr->setAxisTitle("technical trigger bits", 2);

        algo_tt_bits_corr = m_dbe->book2D("algo_tt_bits_corr",
                "GT algo tech. trig. bit correlation",
                128, -0.5, 127.5, 64, -0.5, 63.5);
        algo_tt_bits_corr->setAxisTitle("algorithm bits", 1);
        algo_tt_bits_corr->setAxisTitle("technical trigger bits", 2);

        algo_bits_lumi = m_dbe->book2D("algo_bits_lumi",
                "GT algo bit rate per lumi segment",
                250, 0., 250., 128, -0.5, 127.5);
        algo_bits_lumi->setAxisTitle("Luminosity segment", 1);
        algo_bits_lumi->setAxisTitle("Algorithm bits", 2);

        tt_bits_lumi = m_dbe->book2D("tt_bits_lumi",
                "GT tech. trig. bit rate per lumi segment",
                250, 0., 250., 64, -0.5, 63.5);
        tt_bits_lumi->setAxisTitle("Luminosity segment", 1);
        tt_bits_lumi->setAxisTitle("Technical trigger bits", 2);

        event_type = m_dbe->book1D("event_type", "GT event type", 10, -0.5, 9.5);
        event_type->setAxisTitle("event type", 1);
        event_type->setBinLabel(2, "Physics", 1);
        event_type->setBinLabel(3, "Calibration", 1);
        event_type->setBinLabel(4, "Random", 1);
        event_type->setBinLabel(6, "Traced", 1);
        event_type->setBinLabel(7, "Test", 1);
        event_type->setBinLabel(8, "Error", 1);

        event_number = m_dbe->book1D("event_number",
                "GT Event number (from last resync)",
                100, 0., 50000.);
        event_number->setAxisTitle("event number", 1);

        event_lumi = m_dbe->bookProfile("event_lumi",
                "GT Event number (from last resync) vs lumi section",
                250, 0., 250., 100, -0.1, 1.e15, "s");
        event_lumi->setAxisTitle("Luminosity segment", 1);
        event_lumi->setAxisTitle("event number", 2);

        trigger_number = m_dbe->book1D("trigger_number",
                "GT Trigger number (from start run)",
                100, 0., 50000.);
        trigger_number->setAxisTitle("trigger number", 1);

        trigger_lumi = m_dbe->bookProfile("trigger_lumi",
                "GT Trigger number (from start run) vs lumi section",
                250, 0., 250., 100, -0.1, 1.e15, "s");
        trigger_lumi->setAxisTitle("Luminosity segment", 1);
        trigger_lumi->setAxisTitle("Trigger number", 2);

        evnum_trignum_lumi = m_dbe->bookProfile("evnum_trignum_lumi",
                "GT Event/Trigger number ratio vs lumi section",
                250, 0., 250., 100, -0.1, 2., "s");
        evnum_trignum_lumi->setAxisTitle("Luminosity segment", 1);
        evnum_trignum_lumi->setAxisTitle("Event/trigger number ratio", 2);

        orbit_lumi = m_dbe->bookProfile("orbit_lumi",
                "GT orbit number vs lumi section",
                250, 0., 250., 100, -0.1, 1.e15, "s");
        orbit_lumi->setAxisTitle("Luminosity segment", 1);
        orbit_lumi->setAxisTitle("Orbit number", 2);

        setupversion_lumi = m_dbe->bookProfile("setupversion_lumi",
                "GT setup version vs lumi section",
                250, 0., 250., 100, -0.1, 1.e10, "i");
        setupversion_lumi->setAxisTitle("Luminosity segment", 1);
        setupversion_lumi->setAxisTitle("Setup version", 2);

        gtfe_bx = m_dbe->book1D("gtfe_bx", "GTFE Bx number", 3600, 0., 3600.);
        gtfe_bx->setAxisTitle("GTFE BX number", 1);

        dbx_module = m_dbe->bookProfile("dbx_module",
                "delta Bx of GT modules wrt. GTFE",
                20, 0., 20., 100, -4000., 4000., "i");
        dbx_module->setAxisTitle("GT crate module", 1);
        dbx_module->setAxisTitle("Module Bx - GTFE Bx", 2);
        dbx_module->setBinLabel(1, "GTFEevm", 1);
        dbx_module->setBinLabel(2, "TCS", 1);
        dbx_module->setBinLabel(3, "FDL", 1);
        dbx_module->setBinLabel(4, "FDLloc", 1);
        dbx_module->setBinLabel(5, "PSB9", 1);
        dbx_module->setBinLabel(6, "PSB9loc", 1);
        dbx_module->setBinLabel(7, "PSB13", 1);
        dbx_module->setBinLabel(8, "PSB13loc", 1);
        dbx_module->setBinLabel(9, "PSB14", 1);
        dbx_module->setBinLabel(10, "PSB14loc", 1);
        dbx_module->setBinLabel(11, "PSB15", 1);
        dbx_module->setBinLabel(12, "PSB15loc", 1);
        dbx_module->setBinLabel(13, "PSB19", 1);
        dbx_module->setBinLabel(14, "PSB19loc", 1);
        dbx_module->setBinLabel(15, "PSB20", 1);
        dbx_module->setBinLabel(16, "PSB20loc", 1);
        dbx_module->setBinLabel(17, "PSB21", 1);
        dbx_module->setBinLabel(18, "PSB21loc", 1);
        dbx_module->setBinLabel(19, "GMT", 1);

        BST_MasterStatus = m_dbe->book2D("BST_MasterStatus",
                "BST master status over lumi segment",
                250, 0., 250., 6, -1., 5.);
        BST_MasterStatus->setAxisTitle("Luminosity segment", 1);
        BST_MasterStatus->setAxisTitle("BST master status", 2);
        BST_MasterStatus->setBinLabel(2, "Master Beam 1", 2);
        BST_MasterStatus->setBinLabel(3, "Master Beam 2", 2);

        BST_turnCountNumber = m_dbe->book2D("BST_turnCountNumber",
                "BST turn count over lumi segment",
                250, 0., 250., 250, 0., 4.3e9);
        BST_turnCountNumber->setAxisTitle("Luminosity segment", 1);
        BST_turnCountNumber->setAxisTitle("BST turn count number", 2);

        BST_lhcFillNumber = m_dbe->book1D("BST_lhcFillNumber",
                "BST LHC fill number % 1000", 1000, 0., 1000.);
        BST_lhcFillNumber->setAxisTitle("BST LHC fill number modulo 1000");

        BST_beamMode = m_dbe->book2D("BST_beamMode",
                "BST beam mode over lumi segment",
                250, 0., 250., 25, 1., 26.);
        BST_beamMode->setAxisTitle("Luminosity segment", 1);
        BST_beamMode->setAxisTitle("mode", 2);
        BST_beamMode->setBinLabel(1, "no mode", 2);
        BST_beamMode->setBinLabel(2, "setup", 2);
        BST_beamMode->setBinLabel(3, "inj pilot", 2);
        BST_beamMode->setBinLabel(4, "inj intr", 2);
        BST_beamMode->setBinLabel(5, "inj nomn", 2);
        BST_beamMode->setBinLabel(6, "pre ramp", 2);
        BST_beamMode->setBinLabel(7, "ramp", 2);
        BST_beamMode->setBinLabel(8, "flat top", 2);
        BST_beamMode->setBinLabel(9, "squeeze", 2);
        BST_beamMode->setBinLabel(10, "adjust", 2);
        BST_beamMode->setBinLabel(11, "stable", 2);
        BST_beamMode->setBinLabel(12, "unstable", 2);
        BST_beamMode->setBinLabel(13, "beam dump", 2);
        BST_beamMode->setBinLabel(14, "ramp down", 2);
        BST_beamMode->setBinLabel(15, "recovery", 2);
        BST_beamMode->setBinLabel(16, "inj dump", 2);
        BST_beamMode->setBinLabel(17, "circ dump", 2);
        BST_beamMode->setBinLabel(18, "abort", 2);
        BST_beamMode->setBinLabel(19, "cycling", 2);
        BST_beamMode->setBinLabel(20, "warn beam dump", 2);
        BST_beamMode->setBinLabel(21, "no beam", 2);

        BST_beamMomentum = m_dbe->book2D("BST_beamMomentum",
                "BST beam momentum",
                250, 0., 250., 100, 0., 7200.);
        BST_beamMomentum->setAxisTitle("Luminosity segment", 1);
        BST_beamMomentum->setAxisTitle("Beam momentum", 2);

        gpsfreq = m_dbe->book1D("gpsfreq", "clock frequency measured by GPS",
                1000, 39.95, 40.2);
        gpsfreq->setAxisTitle("CMS clock frequency (MHz)");

        gpsfreqwide = m_dbe->book1D("gpsfreqwide",
                "clock frequency measured by GPS", 1000, -2., 200.);
        gpsfreqwide->setAxisTitle("CMS clock frequency (MHz)");

        gpsfreqlum = m_dbe->book2D("gpsfreqlum",
                "clock frequency measured by GPS",
                250, 0., 250., 100, 39.95, 40.2);
        gpsfreqlum->setAxisTitle("Luminosity segment", 1);
        gpsfreqlum->setAxisTitle("CMS clock frequency (MHz)", 2);

        BST_intensityBeam1 = m_dbe->book2D("BST_intensityBeam1",
                "intensity beam 1",
                250, 0., 250., 100, 0., 100.);
        BST_intensityBeam1->setAxisTitle("Luminosity segment", 1);
        BST_intensityBeam1->setAxisTitle("Beam intensity", 2);

        BST_intensityBeam2 = m_dbe->book2D("BST_intensityBeam2",
                "intensity beam 2",
                250, 0., 250., 100, 0., 100.);
        BST_intensityBeam2->setAxisTitle("Luminosity segment", 1);
        BST_intensityBeam2->setAxisTitle("Beam intensity", 2);

        // prescale factor index monitoring

        m_monL1PrescaleFactorSet = m_dbe->book2D("L1PrescaleFactorSet",
                "Index of L1 prescale factor set",
                500, 0., 500., 25, 0., 25.);
        m_monL1PrescaleFactorSet->setAxisTitle("Luminosity segment", 1);
        m_monL1PrescaleFactorSet->setAxisTitle("L1 PF set index", 2);

        m_monL1PfIndicesPerLs = m_dbe->book1D("L1PfIndicesPerLs",
                "Number of prescale factor indices used per LS", 10, 0., 10.);
        m_monL1PfIndicesPerLs->setAxisTitle("Number of PF indices used per LS", 1);
        m_monL1PfIndicesPerLs->setAxisTitle("Entries/run", 2);


        // TCS vs FDL common quantity monitoring

        m_dbe->setCurrentFolder(m_histFolder + "/TCSvsEvmFDL");

        //    orbit number
        m_monOrbitNrDeltaTcsEvmFdl = m_dbe->book1D("OrbitNrDeltaTcsEvmFdl",
                        "Orbit number difference (TCS - EVM_FDL)",
                        2 * MaxOrbitNrDeltaTcsEvmFdl + 1,
                        static_cast<float>(-(MaxOrbitNrDeltaTcsEvmFdl + 1)),
                        static_cast<float>(MaxOrbitNrDeltaTcsEvmFdl + 1));
        m_monOrbitNrDeltaTcsEvmFdl->setAxisTitle("Orbit number difference", 1);
        m_monOrbitNrDeltaTcsEvmFdl->setAxisTitle("Entries/run", 2);

        m_monLsNrDeltaTcsEvmFdl = m_dbe->book1D("LsNrDeltaTcsEvmFdl",
                        "LS number difference (TCS - EVM_FDL)",
                        2 * MaxLsNrDeltaTcsEvmFdl + 1,
                        static_cast<float>(-(MaxLsNrDeltaTcsEvmFdl + 1)),
                        static_cast<float>(MaxLsNrDeltaTcsEvmFdl + 1));
        m_monLsNrDeltaTcsEvmFdl->setAxisTitle("LS number difference", 1);
        m_monLsNrDeltaTcsEvmFdl->setAxisTitle("Entries/run", 2);

        //    LS number

        m_monOrbitNrDeltaTcsEvmFdlLs = m_dbe->book2D("OrbitNrDeltaTcsEvmFdlLs",
                "Orbit number difference (TCS - EVM_FDL)",
                500, 0., 500.,
                2 * MaxOrbitNrDeltaTcsEvmFdl + 1,
                static_cast<float>(-(MaxOrbitNrDeltaTcsEvmFdl + 1)),
                static_cast<float>(MaxOrbitNrDeltaTcsEvmFdl + 1));
        m_monOrbitNrDeltaTcsEvmFdlLs->setAxisTitle("Luminosity segment", 1);
        m_monOrbitNrDeltaTcsEvmFdlLs->setAxisTitle("Orbit number difference (TCS - EVM_FDL)", 2);

        m_monLsNrDeltaTcsEvmFdlLs = m_dbe->book2D("LsNrDeltaTcsEvmFdlLs",
                "LS number difference (TCS - EVM_FDL)",
                500, 0., 500.,
                2 * MaxLsNrDeltaTcsEvmFdl + 1,
                static_cast<float>(-(MaxLsNrDeltaTcsEvmFdl + 1)),
                static_cast<float>(MaxLsNrDeltaTcsEvmFdl + 1));
        m_monLsNrDeltaTcsEvmFdlLs->setAxisTitle("Luminosity segment", 1);
        m_monLsNrDeltaTcsEvmFdlLs->setAxisTitle("LS number difference (TCS - EVM_FDL)", 2);

        m_dbe->setCurrentFolder(m_histFolder);
   }



}


////////////////////////////////////////////////////////////////////////////////////////
bool L1TGT::isActive(int word, int bit) {
    if (word & (1 << bit))
        return true;
    return false;
}
// static class members
// maximum difference in orbit number
const int L1TGT::MaxOrbitNrDeltaTcsEvmFdl = 24;

// maximum difference in luminosity segment number
const int L1TGT::MaxLsNrDeltaTcsEvmFdl = 24;

