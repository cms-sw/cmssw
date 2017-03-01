//-----------------------------------------------------------------------------
//
//   Class: CSCMotherboard
//
//   Description: 
//    When the Trigger MotherBoard is instantiated it instantiates an ALCT
//    and CLCT board.  The Motherboard takes up to two LCTs from each anode
//    and cathode LCT card and combines them into a single Correlated LCT.
//    The output is up to two Correlated LCTs.
//
//    It can be run in either a test mode, where the arguments are a collection
//    of wire times and arrays of halfstrip and distrip times, or
//    for general use, with with wire digi and comparator digi collections as
//    arguments.  In the latter mode, the wire & strip info is passed on the
//    LCTProcessors, where it is decoded and converted into a convenient form.
//    After running the anode and cathode LCTProcessors, TMB correlates the
//    anode and cathode LCTs.  At present, it simply matches the best CLCT
//    with the best ALCT; perhaps a better algorithm will be determined in
//    the future.  The MotherBoard then determines a few more numbers (such as
//    quality and pattern) from the ALCT and CLCT information, and constructs
//    two correlated LCTs.
//
//    correlateLCTs() may need to be modified to take into account a
//    possibility of ALCTs and CLCTs arriving at different bx times.
//
//   Author List: Benn Tannenbaum 28 August 1999 benn@physics.ucla.edu
//                Based on code by Nick Wisniewski (nw@its.caltech.edu)
//                and a framework by Darin Acosta (acosta@phys.ufl.edu).
//
//
//   Modifications: Numerous later improvements by Jason Mumford and
//                  Slava Valuev (see cvs in ORCA).
//   Porting from ORCA by S. Valuev (Slava.Valuev@cern.ch), May 2006.
//
//-----------------------------------------------------------------------------

#include <L1Trigger/CSCTriggerPrimitives/src/CSCMotherboard.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>

// Default values of configuration parameters.
const unsigned int CSCMotherboard::def_mpc_block_me1a      = 1;
const unsigned int CSCMotherboard::def_alct_trig_enable    = 0;
const unsigned int CSCMotherboard::def_clct_trig_enable    = 0;
const unsigned int CSCMotherboard::def_match_trig_enable   = 1;
const unsigned int CSCMotherboard::def_match_trig_window_size = 7;
const unsigned int CSCMotherboard::def_tmb_l1a_window_size = 7;

CSCMotherboard::CSCMotherboard(unsigned endcap, unsigned station,
                               unsigned sector, unsigned subsector,
                               unsigned chamber,
                               const edm::ParameterSet& conf) :
                   theEndcap(endcap), theStation(station), theSector(sector),
                   theSubsector(subsector), theTrigChamber(chamber) {
  
  theRing = CSCTriggerNumbering::ringFromTriggerLabels(theStation, theTrigChamber);

  // Normal constructor.  -JM
  // Pass ALCT, CLCT, and common parameters on to ALCT and CLCT processors.
  static bool config_dumped = false;

  // Some configuration parameters and some details of the emulator
  // algorithms depend on whether we want to emulate the trigger logic
  // used in TB/MTCC or its idealized version (the latter was used in MC
  // studies since early ORCA days until (and including) CMSSW_2_1_X).
  edm::ParameterSet commonParams =
    conf.getParameter<edm::ParameterSet>("commonParam");
  isMTCC = commonParams.getParameter<bool>("isMTCC");

  // Switch for a new (2007) version of the TMB firmware.
  isTMB07 = commonParams.getParameter<bool>("isTMB07");

  // is it (non-upgrade algorithm) run along with upgrade one?
  isSLHC = commonParams.getParameter<bool>("isSLHC");

  // Choose the appropriate set of configuration parameters depending on
  // isTMB07 and isMTCC flags.
  // Starting with CMSSW_3_1_X, these settings are overwritten by the
  // ones delivered by the EventSetup mechanism.
  edm::ParameterSet alctParams, clctParams;
  if (isTMB07) {
    alctParams = conf.getParameter<edm::ParameterSet>("alctParam07");
    clctParams = conf.getParameter<edm::ParameterSet>("clctParam07");
  }
  else if (isMTCC) {
    alctParams = conf.getParameter<edm::ParameterSet>("alctParamMTCC");
    clctParams = conf.getParameter<edm::ParameterSet>("clctParamMTCC");
  }
  else {
    alctParams = conf.getParameter<edm::ParameterSet>("alctParamOldMC");
    clctParams = conf.getParameter<edm::ParameterSet>("clctParamOldMC");
  }

  // Motherboard parameters:
  edm::ParameterSet tmbParams  =  conf.getParameter<edm::ParameterSet>("tmbParam");
  const edm::ParameterSet me11tmbGemParams(conf.existsAs<edm::ParameterSet>("me11tmbSLHCGEM")?
                                           conf.getParameter<edm::ParameterSet>("me11tmbSLHCGEM"):edm::ParameterSet());
  const edm::ParameterSet me21tmbGemParams(conf.existsAs<edm::ParameterSet>("me21tmbSLHCGEM")?
                                           conf.getParameter<edm::ParameterSet>("me21tmbSLHCGEM"):edm::ParameterSet());
  const edm::ParameterSet me3141tmbRpcParams(conf.existsAs<edm::ParameterSet>("me3141tmbSLHCRPC")?
                                             conf.getParameter<edm::ParameterSet>("me3141tmbSLHCRPC"):edm::ParameterSet());

  const bool runME11ILT(commonParams.existsAs<bool>("runME11ILT")?commonParams.getParameter<bool>("runME11ILT"):false);  
  const bool runME21ILT(commonParams.existsAs<bool>("runME21ILT")?commonParams.getParameter<bool>("runME21ILT"):false);  
  const bool runME3141ILT(commonParams.existsAs<bool>("runME3141ILT")?commonParams.getParameter<bool>("runME3141ILT"):false);

  // run upgrade TMBs for all MEX/1 stations
  if (isSLHC and theRing == 1){    
    if (theStation == 1) {
      tmbParams = conf.getParameter<edm::ParameterSet>("tmbSLHC");
      alctParams = conf.getParameter<edm::ParameterSet>("alctSLHC");
      clctParams = conf.getParameter<edm::ParameterSet>("clctSLHC");
      if (runME11ILT) {
        tmbParams = me11tmbGemParams;
      }
    }
    else if (theStation == 2 and runME21ILT) {
      tmbParams = me21tmbGemParams;
      alctParams = conf.getParameter<edm::ParameterSet>("alctSLHCME21");
      clctParams = conf.getParameter<edm::ParameterSet>("clctSLHCME21");
    }
    else if ((theStation == 3 or theStation == 4) and runME3141ILT) {
      tmbParams = me3141tmbRpcParams;
      alctParams = conf.getParameter<edm::ParameterSet>("alctSLHCME3141");
      clctParams = conf.getParameter<edm::ParameterSet>("clctSLHCME3141");
    }
  }

  mpc_block_me1a    = tmbParams.getParameter<unsigned int>("mpcBlockMe1a");
  alct_trig_enable  = tmbParams.getParameter<unsigned int>("alctTrigEnable");
  clct_trig_enable  = tmbParams.getParameter<unsigned int>("clctTrigEnable");
  match_trig_enable = tmbParams.getParameter<unsigned int>("matchTrigEnable");
  match_trig_window_size =
    tmbParams.getParameter<unsigned int>("matchTrigWindowSize");
  tmb_l1a_window_size = // Common to CLCT and TMB
    tmbParams.getParameter<unsigned int>("tmbL1aWindowSize");

  lct_central_bx = 6;

  // configuration handle for number of early time bins
  early_tbins = tmbParams.getParameter<int>("tmbEarlyTbins");

  // whether to not reuse ALCTs that were used by previous matching CLCTs
  drop_used_alcts = tmbParams.getParameter<bool>("tmbDropUsedAlcts");

  // whether to readout only the earliest two LCTs in readout window
  readout_earliest_2 = tmbParams.getParameter<bool>("tmbReadoutEarliest2");

  infoV = tmbParams.getParameter<int>("verbosity");

  alct.reset( new CSCAnodeLCTProcessor(endcap, station, sector, subsector, chamber, alctParams, commonParams) );
  clct.reset( new CSCCathodeLCTProcessor(endcap, station, sector, subsector, chamber, clctParams, commonParams, tmbParams) );

  //if (theStation==1 && CSCTriggerNumbering::ringFromTriggerLabels(theStation, theTrigChamber)==2) infoV = 3;

  // Check and print configuration parameters.
  checkConfigParameters();
  if (infoV > 0 && !config_dumped) {
    dumpConfigParams();
    config_dumped = true;
  }

  // test to make sure that what goes into a correlated LCT is also what
  // comes back out.
  // testLCT();
}

CSCMotherboard::CSCMotherboard() :
                   theEndcap(1), theStation(1), theSector(1),
                   theSubsector(1), theTrigChamber(1) {
  // Constructor used only for testing.  -JM
  static bool config_dumped = false;

  isMTCC  = false;
  isTMB07 = true;

  early_tbins = 4;

  alct.reset( new CSCAnodeLCTProcessor() );
  clct.reset( new CSCCathodeLCTProcessor() );
  mpc_block_me1a      = def_mpc_block_me1a;
  alct_trig_enable    = def_alct_trig_enable;
  clct_trig_enable    = def_clct_trig_enable;
  match_trig_enable   = def_match_trig_enable;
  match_trig_window_size = def_match_trig_window_size;
  tmb_l1a_window_size = def_tmb_l1a_window_size;

  infoV = 2;

  // Check and print configuration parameters.
  checkConfigParameters();
  if (infoV > 0 && !config_dumped) {
    dumpConfigParams();
    config_dumped = true;
  }
}

CSCMotherboard::~CSCMotherboard() {
}

void CSCMotherboard::clear() {
  if (alct) alct->clear();
  if (clct) clct->clear();
  for (int bx = 0; bx < MAX_LCT_BINS; bx++) {
    firstLCT[bx].clear();
    secondLCT[bx].clear();
  }
}

// Set configuration parameters obtained via EventSetup mechanism.
void CSCMotherboard::setConfigParameters(const CSCDBL1TPParameters* conf) {
  static bool config_dumped = false;

  // Config. parameters for the TMB itself.
  mpc_block_me1a         = conf->tmbMpcBlockMe1a();
  alct_trig_enable       = conf->tmbAlctTrigEnable();
  clct_trig_enable       = conf->tmbClctTrigEnable();
  match_trig_enable      = conf->tmbMatchTrigEnable();
  match_trig_window_size = conf->tmbMatchTrigWindowSize();
  tmb_l1a_window_size    = conf->tmbTmbL1aWindowSize();

  // Config. paramteres for ALCT and CLCT processors.
  alct->setConfigParameters(conf);
  clct->setConfigParameters(conf);

  // Check and print configuration parameters.
  checkConfigParameters();
  if (!config_dumped) {
    dumpConfigParams();
    config_dumped = true;
  }
}

void CSCMotherboard::checkConfigParameters() {
  // Make sure that the parameter values are within the allowed range.

  // Max expected values.
  static const unsigned int max_mpc_block_me1a      = 1 << 1;
  static const unsigned int max_alct_trig_enable    = 1 << 1;
  static const unsigned int max_clct_trig_enable    = 1 << 1;
  static const unsigned int max_match_trig_enable   = 1 << 1;
  static const unsigned int max_match_trig_window_size = 1 << 4;
  static const unsigned int max_tmb_l1a_window_size = 1 << 4;

  // Checks.
  if (mpc_block_me1a >= max_mpc_block_me1a) {
    if (infoV >= 0) edm::LogError("L1CSCTPEmulatorConfigError")
      << "+++ Value of mpc_block_me1a, " << mpc_block_me1a
      << ", exceeds max allowed, " << max_mpc_block_me1a-1 << " +++\n"
      << "+++ Try to proceed with the default value, mpc_block_me1a="
      << def_mpc_block_me1a << " +++\n";
    mpc_block_me1a = def_mpc_block_me1a;
  }
  if (alct_trig_enable >= max_alct_trig_enable) {
    if (infoV >= 0) edm::LogError("L1CSCTPEmulatorConfigError")
      << "+++ Value of alct_trig_enable, " << alct_trig_enable
      << ", exceeds max allowed, " << max_alct_trig_enable-1 << " +++\n"
      << "+++ Try to proceed with the default value, alct_trig_enable="
      << def_alct_trig_enable << " +++\n";
    alct_trig_enable = def_alct_trig_enable;
  }
  if (clct_trig_enable >= max_clct_trig_enable) {
    if (infoV >= 0) edm::LogError("L1CSCTPEmulatorConfigError")
      << "+++ Value of clct_trig_enable, " << clct_trig_enable
      << ", exceeds max allowed, " << max_clct_trig_enable-1 << " +++\n"
      << "+++ Try to proceed with the default value, clct_trig_enable="
      << def_clct_trig_enable << " +++\n";
    clct_trig_enable = def_clct_trig_enable;
  }
  if (match_trig_enable >= max_match_trig_enable) {
    if (infoV >= 0) edm::LogError("L1CSCTPEmulatorConfigError")
      << "+++ Value of match_trig_enable, " << match_trig_enable
      << ", exceeds max allowed, " << max_match_trig_enable-1 << " +++\n"
      << "+++ Try to proceed with the default value, match_trig_enable="
      << def_match_trig_enable << " +++\n";
    match_trig_enable = def_match_trig_enable;
  }
  if (match_trig_window_size >= max_match_trig_window_size) {
    if (infoV >= 0) edm::LogError("L1CSCTPEmulatorConfigError")
      << "+++ Value of match_trig_window_size, " << match_trig_window_size
      << ", exceeds max allowed, " << max_match_trig_window_size-1 << " +++\n"
      << "+++ Try to proceed with the default value, match_trig_window_size="
      << def_match_trig_window_size << " +++\n";
    match_trig_window_size = def_match_trig_window_size;
  }
  if (tmb_l1a_window_size >= max_tmb_l1a_window_size) {
    if (infoV >= 0) edm::LogError("L1CSCTPEmulatorConfigError")
      << "+++ Value of tmb_l1a_window_size, " << tmb_l1a_window_size
      << ", exceeds max allowed, " << max_tmb_l1a_window_size-1 << " +++\n"
      << "+++ Try to proceed with the default value, tmb_l1a_window_size="
      << def_tmb_l1a_window_size << " +++\n";
    tmb_l1a_window_size = def_tmb_l1a_window_size;
  }
}

void CSCMotherboard::run(
 const std::vector<int> w_times[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_WIRES],
 const std::vector<int> hs_times[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS],
 const std::vector<int> ds_times[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS]) {
  // Debug version.  -JM
  clear();
  alct->run(w_times);            // run anode LCT
  clct->run(hs_times, ds_times); // run cathodeLCT

  int bx_alct_matched = 0;
  for (int bx_clct = 0; bx_clct < CSCCathodeLCTProcessor::MAX_CLCT_BINS;
       bx_clct++) {
    if (clct->bestCLCT[bx_clct].isValid()) {
      bool is_matched = false;
      int bx_alct_start = bx_clct - match_trig_window_size/2;
      int bx_alct_stop  = bx_clct + match_trig_window_size/2;
      // Empirical correction to match 2009 collision data (firmware change?)
      if (!isSLHC) bx_alct_stop += match_trig_window_size%2;
      
      for (int bx_alct = bx_alct_start; bx_alct <= bx_alct_stop; bx_alct++) {
        if (bx_alct < 0 || bx_alct >= CSCAnodeLCTProcessor::MAX_ALCT_BINS)
          continue;
        if (alct->bestALCT[bx_alct].isValid()) {
          correlateLCTs(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct],
                        clct->bestCLCT[bx_clct], clct->secondCLCT[bx_clct]);
          is_matched = true;
          bx_alct_matched = bx_alct;
          break;
        }
      }
      // No ALCT within the match time interval found: report CLCT-only LCT
      // (use dummy ALCTs).
      if (!is_matched) {
        correlateLCTs(alct->bestALCT[bx_clct], alct->secondALCT[bx_clct],
                      clct->bestCLCT[bx_clct], clct->secondCLCT[bx_clct]);
      }
    }
    // No valid CLCTs; attempt to make ALCT-only LCT (use dummy CLCTs).
    else {
      int bx_alct = bx_clct - match_trig_window_size/2;
      if (bx_alct >= 0 && bx_alct > bx_alct_matched) {
        if (alct->bestALCT[bx_alct].isValid()) {
          correlateLCTs(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct],
                        clct->bestCLCT[bx_clct], clct->secondCLCT[bx_clct]);
        }
      }
    }
  }
}

void
CSCMotherboard::run(const CSCWireDigiCollection* wiredc,
                    const CSCComparatorDigiCollection* compdc) {
  clear();
  if (alct && clct) {
    {
      std::vector<CSCALCTDigi> alctV = alct->run(wiredc); // run anodeLCT
    }
    {
      std::vector<CSCCLCTDigi> clctV = clct->run(compdc); // run cathodeLCT
    }

    int used_alct_mask[20];
    for (int a=0;a<20;++a) used_alct_mask[a]=0;

    int bx_alct_matched = 0; // bx of last matched ALCT
    for (int bx_clct = 0; bx_clct < CSCCathodeLCTProcessor::MAX_CLCT_BINS;
         bx_clct++) {
      // There should be at least one valid ALCT or CLCT for a
      // correlated LCT to be formed.  Decision on whether to reject
      // non-complete LCTs (and if yes of which type) is made further
      // upstream.
      if (clct->bestCLCT[bx_clct].isValid()) {
        // Look for ALCTs within the match-time window.  The window is
        // centered at the CLCT bx; therefore, we make an assumption
        // that anode and cathode hits are perfectly synchronized.  This
        // is always true for MC, but only an approximation when the
        // data is analyzed (which works fairly good as long as wide
        // windows are used).  To get rid of this assumption, one would
        // need to access "full BX" words, which are not readily
        // available.
        bool is_matched = false;
        int bx_alct_start = bx_clct - match_trig_window_size/2;
        int bx_alct_stop  = bx_clct + match_trig_window_size/2;
        // Empirical correction to match 2009 collision data (firmware change?)
        // (but don't do it for SLHC case, assume it would not be there)
        if (!isSLHC) bx_alct_stop += match_trig_window_size%2;

        for (int bx_alct = bx_alct_start; bx_alct <= bx_alct_stop; bx_alct++) {
          if (bx_alct < 0 || bx_alct >= CSCAnodeLCTProcessor::MAX_ALCT_BINS)
            continue;
          // default: do not reuse ALCTs that were used with previous CLCTs
          if (drop_used_alcts && used_alct_mask[bx_alct]) continue;
          if (alct->bestALCT[bx_alct].isValid()) {
            if (infoV > 1) LogTrace("CSCMotherboard")
              << "Successful ALCT-CLCT match: bx_clct = " << bx_clct
                << "; match window: [" << bx_alct_start << "; " << bx_alct_stop
                << "]; bx_alct = " << bx_alct;
            correlateLCTs(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct],
                          clct->bestCLCT[bx_clct], clct->secondCLCT[bx_clct]);
            used_alct_mask[bx_alct] += 1;
            is_matched = true;
            bx_alct_matched = bx_alct;
            break;
          }
        }
        // No ALCT within the match time interval found: report CLCT-only LCT
        // (use dummy ALCTs).
        if (!is_matched) {
          if (infoV > 1) LogTrace("CSCMotherboard")
            << "Unsuccessful ALCT-CLCT match (CLCT only): bx_clct = "
            << bx_clct << "; match window: [" << bx_alct_start
            << "; " << bx_alct_stop << "]";
          correlateLCTs(alct->bestALCT[bx_clct], alct->secondALCT[bx_clct],
                        clct->bestCLCT[bx_clct], clct->secondCLCT[bx_clct]);
        }
      }
      // No valid CLCTs; attempt to make ALCT-only LCT.  Use only ALCTs
      // which have zeroth chance to be matched at later cathode times.
      // (I am not entirely sure this perfectly matches the firmware logic.)
      // Use dummy CLCTs.
      else {
        int bx_alct = bx_clct - match_trig_window_size/2;
        if (bx_alct >= 0 && bx_alct > bx_alct_matched) {
          if (alct->bestALCT[bx_alct].isValid()) {
            if (infoV > 1) LogTrace("CSCMotherboard")
              << "Unsuccessful ALCT-CLCT match (ALCT only): bx_alct = "
              << bx_alct;
            correlateLCTs(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct],
                          clct->bestCLCT[bx_clct], clct->secondCLCT[bx_clct]);
          }
        }
      }
    }

    if (infoV > 0) {
      for (int bx = 0; bx < MAX_LCT_BINS; bx++) {
        if (firstLCT[bx].isValid())
          LogDebug("CSCMotherboard") << firstLCT[bx];
        if (secondLCT[bx].isValid())
          LogDebug("CSCMotherboard") << secondLCT[bx];
      }
    }
  }
  else {
    if (infoV >= 0) edm::LogError("L1CSCTPEmulatorSetupError")
      << "+++ run() called for non-existing ALCT/CLCT processor! +++ \n";
  }
}

// Returns vector of read-out correlated LCTs, if any.  Starts with
// the vector of all found LCTs and selects the ones in the read-out
// time window.
std::vector<CSCCorrelatedLCTDigi> CSCMotherboard::readoutLCTs() {
  std::vector<CSCCorrelatedLCTDigi> tmpV;

  // The start time of the L1A*LCT coincidence window should be related
  // to the fifo_pretrig parameter, but I am not completely sure how.
  // Just choose it such that the window is centered at bx=7.  This may
  // need further tweaking if the value of tmb_l1a_window_size changes.
  //static int early_tbins = 4;
  
  // Empirical correction to match 2009 collision data (firmware change?)
  static int lct_bins   = tmb_l1a_window_size;
  static int late_tbins = early_tbins + lct_bins;

  static int ifois = 0;
  if (ifois == 0) {
    if (infoV >= 0 && early_tbins < 0) {
      edm::LogWarning("L1CSCTPEmulatorSuspiciousParameters")
        << "+++ early_tbins = " << early_tbins
        << "; in-time LCTs are not getting read-out!!! +++" << "\n";
    }

    if (late_tbins > MAX_LCT_BINS-1) {
      if (infoV >= 0) edm::LogWarning("L1CSCTPEmulatorSuspiciousParameters")
        << "+++ Allowed range of time bins, [0-" << late_tbins
        << "] exceeds max allowed, " << MAX_LCT_BINS-1 << " +++\n"
        << "+++ Set late_tbins to max allowed +++\n";
      late_tbins = MAX_LCT_BINS-1;
    }
    ifois = 1;
  }

  // Start from the vector of all found correlated LCTs and select
  // those within the LCT*L1A coincidence window.
  int bx_readout = -1;
  std::vector<CSCCorrelatedLCTDigi> all_lcts = getLCTs();
  for (std::vector <CSCCorrelatedLCTDigi>::const_iterator plct =
       all_lcts.begin(); plct != all_lcts.end(); plct++) {
    if (!plct->isValid()) continue;

    int bx = (*plct).getBX();
    // Skip LCTs found too early relative to L1Accept.
    if (bx <= early_tbins) {
      if (infoV > 1) LogDebug("CSCMotherboard")
        << " Do not report correlated LCT on key halfstrip "
        << plct->getStrip() << " and key wire " << plct->getKeyWG()
        << ": found at bx " << bx << ", whereas the earliest allowed bx is "
        << early_tbins+1;
      continue;
    }

    // Skip LCTs found too late relative to L1Accept.
    if (bx > late_tbins) {
      if (infoV > 1) LogDebug("CSCMotherboard")
        << " Do not report correlated LCT on key halfstrip "
        << plct->getStrip() << " and key wire " << plct->getKeyWG()
        << ": found at bx " << bx << ", whereas the latest allowed bx is "
        << late_tbins;
      continue;
    }

    // If (readout_earliest_2) take only LCTs in the earliest bx in the read-out window:
    // in digi->raw step, LCTs have to be packed into the TMB header, and
    // currently there is room just for two.
    if (readout_earliest_2) {
      if (bx_readout == -1 || bx == bx_readout) {
        tmpV.push_back(*plct);
        if (bx_readout == -1) bx_readout = bx;
      }
    }
    // if readout_earliest_2 == false, save all LCTs
    else tmpV.push_back(*plct);
  }
  return tmpV;
}

// Returns vector of all found correlated LCTs, if any.
std::vector<CSCCorrelatedLCTDigi> CSCMotherboard::getLCTs() {
  std::vector<CSCCorrelatedLCTDigi> tmpV;

  bool me11 = (theStation == 1 &&
               CSCTriggerNumbering::ringFromTriggerLabels(theStation,
                                                          theTrigChamber)==1);

  // Do not report LCTs found in ME1/A if mpc_block_me1/a is set.
  for (int bx = 0; bx < MAX_LCT_BINS; bx++) {
    if (firstLCT[bx].isValid())
      if (!mpc_block_me1a || (!me11 || firstLCT[bx].getStrip() <= 127))
        tmpV.push_back(firstLCT[bx]);
    if (secondLCT[bx].isValid())
      if (!mpc_block_me1a || (!me11 || secondLCT[bx].getStrip() <= 127))
        tmpV.push_back(secondLCT[bx]);
  }
  return tmpV;
}

void CSCMotherboard::correlateLCTs(CSCALCTDigi bestALCT,
                                   CSCALCTDigi secondALCT,
                                   CSCCLCTDigi bestCLCT,
                                   CSCCLCTDigi secondCLCT) {

  bool anodeBestValid     = bestALCT.isValid();
  bool anodeSecondValid   = secondALCT.isValid();
  bool cathodeBestValid   = bestCLCT.isValid();
  bool cathodeSecondValid = secondCLCT.isValid();

  if (anodeBestValid && !anodeSecondValid)     secondALCT = bestALCT;
  if (!anodeBestValid && anodeSecondValid)     bestALCT   = secondALCT;
  if (cathodeBestValid && !cathodeSecondValid) secondCLCT = bestCLCT;
  if (!cathodeBestValid && cathodeSecondValid) bestCLCT   = secondCLCT;

  // ALCT-CLCT matching conditions are defined by "trig_enable" configuration
  // parameters.
  if ((alct_trig_enable  && bestALCT.isValid()) ||
      (clct_trig_enable  && bestCLCT.isValid()) ||
      (match_trig_enable && bestALCT.isValid() && bestCLCT.isValid())) {
    CSCCorrelatedLCTDigi lct = constructLCTs(bestALCT, bestCLCT);
    int bx = lct.getBX();
    if (bx >= 0 && bx < MAX_LCT_BINS) {
      firstLCT[bx] = lct;
      firstLCT[bx].setTrknmb(1);
    }
    else {
      if (infoV > 0) edm::LogWarning("L1CSCTPEmulatorOutOfTimeLCT")
        << "+++ Bx of first LCT candidate, " << bx
        << ", is not within the allowed range, [0-" << MAX_LCT_BINS-1
        << "); skipping it... +++\n";
    }
  }

  if (((secondALCT != bestALCT) || (secondCLCT != bestCLCT)) &&
      ((alct_trig_enable  && secondALCT.isValid()) ||
       (clct_trig_enable  && secondCLCT.isValid()) ||
       (match_trig_enable && secondALCT.isValid() && secondCLCT.isValid()))) {
    CSCCorrelatedLCTDigi lct = constructLCTs(secondALCT, secondCLCT);
    int bx = lct.getBX();
    if (bx >= 0 && bx < MAX_LCT_BINS) {
      secondLCT[bx] = lct;
      secondLCT[bx].setTrknmb(2);
    }
    else {
      if (infoV > 0) edm::LogWarning("L1CSCTPEmulatorOutOfTimeLCT")
        << "+++ Bx of second LCT candidate, " << bx
        << ", is not within the allowed range, [0-" << MAX_LCT_BINS-1
        << "); skipping it... +++\n";
    }
  }
}

// This method calculates all the TMB words and then passes them to the
// constructor of correlated LCTs.
CSCCorrelatedLCTDigi CSCMotherboard::constructLCTs(const CSCALCTDigi& aLCT,
                                                   const CSCCLCTDigi& cLCT) {
  // CLCT pattern number
  unsigned int pattern = encodePattern(cLCT.getPattern(), cLCT.getStripType());

  // LCT quality number
  unsigned int quality = findQuality(aLCT, cLCT);

  // Bunch crossing: get it from cathode LCT if anode LCT is not there.
  int bx = aLCT.isValid() ? aLCT.getBX() : cLCT.getBX();

  // construct correlated LCT; temporarily assign track number of 0.
  int trknmb = 0;
  CSCCorrelatedLCTDigi thisLCT(trknmb, 1, quality, aLCT.getKeyWG(),
                               cLCT.getKeyStrip(), pattern, cLCT.getBend(),
                               bx, 0, 0, 0, theTrigChamber);
  return thisLCT;
}

// CLCT pattern number: encodes the pattern number itself and
// whether the pattern consists of half-strips or di-strips.
unsigned int CSCMotherboard::encodePattern(const int ptn,
                                           const int stripType) {
  const int kPatternBitWidth = 4;
  unsigned int pattern;

  if (!isTMB07) {
    // Cathode pattern number is a kPatternBitWidth-1 bit word.
    pattern = (abs(ptn) & ((1<<(kPatternBitWidth-1))-1));

    // The pattern has the MSB (4th bit in the default version) set if it
    // consists of half-strips.
    if (stripType) {
      pattern = pattern | (1<<(kPatternBitWidth-1));
    }
  }
  else {
    // In the TMB07 firmware, LCT pattern is just a 4-bit CLCT pattern.
    pattern = (abs(ptn) & ((1<<kPatternBitWidth)-1));
  }

  return pattern;
}

// 4-bit LCT quality number.  Definition can be found in
// http://www.phys.ufl.edu/~acosta/tb/tmb_quality.txt.  Made by TMB lookup
// tables and used for MPC sorting.
unsigned int CSCMotherboard::findQuality(const CSCALCTDigi& aLCT,
                                         const CSCCLCTDigi& cLCT) {
  unsigned int quality = 0;

  if (!isTMB07) {
    bool isDistrip = (cLCT.getStripType() == 0);

    if (aLCT.isValid() && !(cLCT.isValid())) {    // no CLCT
      if (aLCT.getAccelerator()) {quality =  1;}
      else                       {quality =  3;}
    }
    else if (!(aLCT.isValid()) && cLCT.isValid()) { // no ALCT
      if (isDistrip)             {quality =  4;}
      else                       {quality =  5;}
    }
    else if (aLCT.isValid() && cLCT.isValid()) { // both ALCT and CLCT
      if (aLCT.getAccelerator()) {quality =  2;} // accelerator muon
      else {                                     // collision muon
        // CLCT quality is, in fact, the number of layers hit, so subtract 3
        // to get quality analogous to ALCT one.
        int sumQual = aLCT.getQuality() + (cLCT.getQuality()-3);
        if (sumQual < 1 || sumQual > 6) {
          if (infoV >= 0) edm::LogWarning("L1CSCTPEmulatorWrongValues")
            << "+++ findQuality: sumQual = " << sumQual << "+++ \n";
        }
        if (isDistrip) { // distrip pattern
          if (sumQual == 2)      {quality =  6;}
          else if (sumQual == 3) {quality =  7;}
          else if (sumQual == 4) {quality =  8;}
          else if (sumQual == 5) {quality =  9;}
          else if (sumQual == 6) {quality = 10;}
        }
        else {            // halfstrip pattern
          if (sumQual == 2)      {quality = 11;}
          else if (sumQual == 3) {quality = 12;}
          else if (sumQual == 4) {quality = 13;}
          else if (sumQual == 5) {quality = 14;}
          else if (sumQual == 6) {quality = 15;}
        }
      }
    }
  }
#ifdef OLD
  else {
    // Temporary definition, used until July 2008.
    // First if statement is fictitious, just to help the CSC TF emulator
    // handle such cases (one needs to make sure they will be accounted for
    // in the new quality definition.
    if (!(aLCT.isValid()) || !(cLCT.isValid())) {
      if (aLCT.isValid() && !(cLCT.isValid()))      quality = 1; // no CLCT
      else if (!(aLCT.isValid()) && cLCT.isValid()) quality = 2; // no ALCT
      else quality = 0; // both absent; should never happen.
    }
    else {
      // Sum of ALCT and CLCT quality bits.  CLCT quality is, in fact, the
      // number of layers hit, so subtract 3 to put it to the same footing as
      // the ALCT quality.
      int sumQual = aLCT.getQuality() + (cLCT.getQuality()-3);
      if (sumQual < 1 || sumQual > 6) {
        if (infoV >= 0) edm::LogWarning("L1CSCTPEmulatorWrongValues")
          << "+++ findQuality: Unexpected sumQual = " << sumQual << "+++\n";
      }

      // LCT quality is basically the sum of ALCT and CLCT qualities, but split
      // in two groups depending on the CLCT pattern id (higher quality for
      // straighter patterns).
      int offset = 0;
      if (cLCT.getPattern() <= 7) offset = 4;
      else                        offset = 9;
      quality = offset + sumQual;
    }
  }
#endif
  else {
    // 2008 definition.
    if (!(aLCT.isValid()) || !(cLCT.isValid())) {
      if (aLCT.isValid() && !(cLCT.isValid()))      quality = 1; // no CLCT
      else if (!(aLCT.isValid()) && cLCT.isValid()) quality = 2; // no ALCT
      else quality = 0; // both absent; should never happen.
    }
    else {
      int pattern = cLCT.getPattern();
      if (pattern == 1) quality = 3; // layer-trigger in CLCT
      else {
        // CLCT quality is the number of layers hit minus 3.
        // CLCT quality is the number of layers hit.
        bool a4 = (aLCT.getQuality() >= 1);
        bool c4 = (cLCT.getQuality() >= 4);
        //              quality = 4; "reserved for low-quality muons in future"
        if      (!a4 && !c4) quality = 5; // marginal anode and cathode
        else if ( a4 && !c4) quality = 6; // HQ anode, but marginal cathode
        else if (!a4 &&  c4) quality = 7; // HQ cathode, but marginal anode
        else if ( a4 &&  c4) {
          if (aLCT.getAccelerator()) quality = 8; // HQ muon, but accel ALCT
          else {
            // quality =  9; "reserved for HQ muons with future patterns
            // quality = 10; "reserved for HQ muons with future patterns
            if (pattern == 2 || pattern == 3)      quality = 11;
            else if (pattern == 4 || pattern == 5) quality = 12;
            else if (pattern == 6 || pattern == 7) quality = 13;
            else if (pattern == 8 || pattern == 9) quality = 14;
            else if (pattern == 10)                quality = 15;
            else {
              if (infoV >= 0) edm::LogWarning("L1CSCTPEmulatorWrongValues")
                << "+++ findQuality: Unexpected CLCT pattern id = "
                << pattern << "+++\n";
            }
          }
        }
      }
    }
  }
  return quality;
}

void CSCMotherboard::testLCT() {
  unsigned int lctPattern, lctQuality;
  for (int pattern = 0; pattern < 8; pattern++) {
    for (int bend = 0; bend < 2; bend++) {
      for (int cfeb = 0; cfeb < 5; cfeb++) {
        for (int strip = 0; strip < 32; strip++) {
          for (int bx = 0; bx < 7; bx++) {
            for (int stripType = 0; stripType < 2; stripType++) {
              for (int quality = 3; quality < 7; quality++) {
                CSCCLCTDigi cLCT(1, quality, pattern, stripType, bend,
                                 strip, cfeb, bx);
                lctPattern = encodePattern(cLCT.getPattern(),
                                           cLCT.getStripType());
                for (int aQuality = 0; aQuality < 4; aQuality++) {
                  for (int wireGroup = 0; wireGroup < 120; wireGroup++) {
                    for (int abx = 0; abx < 7; abx++) {
                      CSCALCTDigi aLCT(1, aQuality, 0, 1, wireGroup, abx);
                      lctQuality = findQuality(aLCT, cLCT);
                      CSCCorrelatedLCTDigi
                        thisLCT(0, 1, lctQuality, aLCT.getKeyWG(),
                                cLCT.getKeyStrip(), lctPattern, cLCT.getBend(),
                                aLCT.getBX());
                      if (lctPattern != static_cast<unsigned int>(thisLCT.getPattern()) )
                        LogTrace("CSCMotherboard")
                          << "pattern mismatch: " << lctPattern
                          << " " << thisLCT.getPattern();
                      if (bend != thisLCT.getBend()) 
                        LogTrace("CSCMotherboard")
                          << "bend mismatch: " << bend
                          << " " << thisLCT.getBend();
                      int key_strip = 32*cfeb + strip;
                      if (key_strip != thisLCT.getStrip()) 
                        LogTrace("CSCMotherboard")
                          << "strip mismatch: " << key_strip
                          << " " << thisLCT.getStrip();
                      if (wireGroup != thisLCT.getKeyWG()) 
                        LogTrace("CSCMotherboard")
                          << "wire group mismatch: " << wireGroup
                          << " " << thisLCT.getKeyWG();
                      if (abx != thisLCT.getBX()) 
                        LogTrace("CSCMotherboard")
                          << "bx mismatch: " << abx << " " << thisLCT.getBX();
                      if (lctQuality != static_cast<unsigned int>(thisLCT.getQuality())) 
                        LogTrace("CSCMotherboard")
                          << "quality mismatch: " << lctQuality
                          << " " << thisLCT.getQuality();
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

void CSCMotherboard::dumpConfigParams() const {
  std::ostringstream strm;
  strm << "\n";
  strm << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
  strm << "+                   TMB configuration parameters:                  +\n";
  strm << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
  strm << " mpc_block_me1a [block/not block triggers which come from ME1/A] = "
       << mpc_block_me1a << "\n";
  strm << " alct_trig_enable [allow ALCT-only triggers] = "
       << alct_trig_enable << "\n";
  strm << " clct_trig_enable [allow CLCT-only triggers] = "
       << clct_trig_enable << "\n";
  strm << " match_trig_enable [allow matched ALCT-CLCT triggers] = "
       << match_trig_enable << "\n";
  strm << " match_trig_window_size [ALCT-CLCT match window width, in 25 ns] = "
       << match_trig_window_size << "\n";
  strm << " tmb_l1a_window_size [L1Accept window width, in 25 ns bins] = "
       << tmb_l1a_window_size << "\n";
  strm << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
  LogDebug("CSCMotherboard") << strm.str();
}


// compare LCTs by quality
bool CSCMotherboard::sortByQuality(const CSCCorrelatedLCTDigi& lct1, const CSCCorrelatedLCTDigi& lct2) 
{ 
  return lct1.getQuality() > lct2.getQuality();
}

// compare LCTs by GEM bending angle
bool CSCMotherboard::sortByGEMDphi(const CSCCorrelatedLCTDigi& lct1, const CSCCorrelatedLCTDigi& lct2) 
{ 
  //  return lct1.getGEMDPhi() < lct2.getGEMDPhi();
  return true;
}
