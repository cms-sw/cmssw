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
//    of wire times and arrays of halfstrip times, or
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

#include "L1Trigger/CSCTriggerPrimitives/src/CSCMotherboard.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"
#include <iostream>

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
  static std::atomic<bool> config_dumped{false};

  // Parameters common for all boards
  edm::ParameterSet commonParams = conf.getParameter<edm::ParameterSet>("commonParam");

  // is it (non-upgrade algorithm) run along with upgrade one?
  isSLHC = commonParams.getParameter<bool>("isSLHC");

  // ALCT and CLCT configs
  edm::ParameterSet alctParams = conf.getParameter<edm::ParameterSet>("alctParam07");
  edm::ParameterSet clctParams = conf.getParameter<edm::ParameterSet>("clctParam07");

  // Motherboard parameters:
  edm::ParameterSet tmbParams  =  conf.getParameter<edm::ParameterSet>("tmbParam");
  const edm::ParameterSet me11tmbGemParams(conf.existsAs<edm::ParameterSet>("me11tmbSLHCGEM")?
                                           conf.getParameter<edm::ParameterSet>("me11tmbSLHCGEM"):edm::ParameterSet());
  const edm::ParameterSet me21tmbGemParams(conf.existsAs<edm::ParameterSet>("me21tmbSLHCGEM")?
                                           conf.getParameter<edm::ParameterSet>("me21tmbSLHCGEM"):edm::ParameterSet());
  const edm::ParameterSet me3141tmbParams(conf.existsAs<edm::ParameterSet>("me3141tmbSLHC")?
                                             conf.getParameter<edm::ParameterSet>("me3141tmbSLHC"):edm::ParameterSet());

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
      tmbParams = me3141tmbParams;
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

  // configuration handle for number of early time bins
  early_tbins = tmbParams.getParameter<int>("tmbEarlyTbins");

  // whether to not reuse ALCTs that were used by previous matching CLCTs
  drop_used_alcts = tmbParams.getParameter<bool>("tmbDropUsedAlcts");
  drop_used_clcts = tmbParams.getParameter<bool>("tmbDropUsedClcts");

  clct_to_alct = tmbParams.getParameter<bool>("clctToAlct");

  // whether to readout only the earliest two LCTs in readout window
  readout_earliest_2 = tmbParams.getParameter<bool>("tmbReadoutEarliest2");

  infoV = tmbParams.getParameter<int>("verbosity");

  alct.reset( new CSCAnodeLCTProcessor(endcap, station, sector, subsector, chamber, alctParams, commonParams) );
  clct.reset( new CSCCathodeLCTProcessor(endcap, station, sector, subsector, chamber, clctParams, commonParams, tmbParams) );

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
  static std::atomic<bool> config_dumped{false};

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

void CSCMotherboard::clear() {
  if (alct) alct->clear();
  if (clct) clct->clear();
  for (int bx = 0; bx < CSCConstants::MAX_LCT_TBINS; bx++) {
    firstLCT[bx].clear();
    secondLCT[bx].clear();
  }
}

// Set configuration parameters obtained via EventSetup mechanism.
void CSCMotherboard::setConfigParameters(const CSCDBL1TPParameters* conf) {
  static std::atomic<bool> config_dumped{false};

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
			 const std::vector<int> hs_times[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS]) {
  // Debug version.  -JM
  clear();

  // set geometry
  alct->setCSCGeometry(csc_g);
  clct->setCSCGeometry(csc_g);

  alct->run(w_times);            // run anode LCT
  clct->run(hs_times); // run cathodeLCT

  int bx_alct_matched = 0;
  for (int bx_clct = 0; bx_clct < CSCConstants::MAX_CLCT_TBINS;
       bx_clct++) {
    if (clct->bestCLCT[bx_clct].isValid()) {
      bool is_matched = false;
      int bx_alct_start = bx_clct - match_trig_window_size/2;
      int bx_alct_stop  = bx_clct + match_trig_window_size/2;
      // Empirical correction to match 2009 collision data (firmware change?)
      if (!isSLHC) bx_alct_stop += match_trig_window_size%2;

      for (int bx_alct = bx_alct_start; bx_alct <= bx_alct_stop; bx_alct++) {
        if (bx_alct < 0 || bx_alct >= CSCConstants::MAX_ALCT_TBINS)
          continue;
        if (alct->bestALCT[bx_alct].isValid()) {
          correlateLCTs(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct],
                        clct->bestCLCT[bx_clct], clct->secondCLCT[bx_clct],
                        CSCCorrelatedLCTDigi::CLCTALCT);
          is_matched = true;
          bx_alct_matched = bx_alct;
          break;
        }
      }
      // No ALCT within the match time interval found: report CLCT-only LCT
      // (use dummy ALCTs).
      if (!is_matched) {
        correlateLCTs(alct->bestALCT[bx_clct], alct->secondALCT[bx_clct],
                      clct->bestCLCT[bx_clct], clct->secondCLCT[bx_clct],
                      CSCCorrelatedLCTDigi::CLCTONLY);
      }
    }
    // No valid CLCTs; attempt to make ALCT-only LCT (use dummy CLCTs).
    else {
      int bx_alct = bx_clct - match_trig_window_size/2;
      if (bx_alct >= 0 && bx_alct > bx_alct_matched) {
        if (alct->bestALCT[bx_alct].isValid()) {
          correlateLCTs(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct],
                        clct->bestCLCT[bx_clct], clct->secondCLCT[bx_clct],
                        CSCCorrelatedLCTDigi::ALCTONLY);
        }
      }
    }
  }
}

void
CSCMotherboard::run(const CSCWireDigiCollection* wiredc,
                    const CSCComparatorDigiCollection* compdc) {
  clear();

  // set geometry
  alct->setCSCGeometry(csc_g);
  clct->setCSCGeometry(csc_g);

  if (alct && clct) {
    {
      const std::vector<CSCALCTDigi>& alctV = alct->run(wiredc); // run anodeLCT
    }
    {
      const std::vector<CSCCLCTDigi>& clctV = clct->run(compdc); // run cathodeLCT
    }

    // CLCT-centric matching
    if (clct_to_alct){
      int used_alct_mask[20];
      for (int a=0;a<20;++a) used_alct_mask[a]=0;

      int bx_alct_matched = 0; // bx of last matched ALCT
      for (int bx_clct = 0; bx_clct < CSCConstants::MAX_CLCT_TBINS;
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
            if (bx_alct < 0 || bx_alct >= CSCConstants::MAX_ALCT_TBINS)
              continue;
            // default: do not reuse ALCTs that were used with previous CLCTs
            if (drop_used_alcts && used_alct_mask[bx_alct]) continue;
            if (alct->bestALCT[bx_alct].isValid()) {
              if (infoV > 1) LogTrace("CSCMotherboard")
                               << "Successful ALCT-CLCT match: bx_clct = " << bx_clct
                               << "; match window: [" << bx_alct_start << "; " << bx_alct_stop
                               << "]; bx_alct = " << bx_alct;
              correlateLCTs(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct],
                            clct->bestCLCT[bx_clct], clct->secondCLCT[bx_clct],
                            CSCCorrelatedLCTDigi::CLCTALCT);
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
                          clct->bestCLCT[bx_clct], clct->secondCLCT[bx_clct],
                          CSCCorrelatedLCTDigi::CLCTONLY);
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
                            clct->bestCLCT[bx_clct], clct->secondCLCT[bx_clct],
                            CSCCorrelatedLCTDigi::ALCTONLY);
            }
          }
        }
      }
    }
    // ALCT-centric matching
    else {
      int used_clct_mask[20];
      for (int a=0;a<20;++a) used_clct_mask[a]=0;

      int bx_clct_matched = 0; // bx of last matched CLCT
      for (int bx_alct = 0; bx_alct < CSCConstants::MAX_ALCT_TBINS;
           bx_alct++) {
        // There should be at least one valid CLCT or ALCT for a
        // correlated LCT to be formed.  Decision on whether to reject
        // non-complete LCTs (and if yes of which type) is made further
        // upstream.
        if (alct->bestALCT[bx_alct].isValid()) {
          // Look for CLCTs within the match-time window.  The window is
          // centered at the ALCT bx; therefore, we make an assumption
          // that anode and cathode hits are perfectly synchronized.  This
          // is always true for MC, but only an approximation when the
          // data is analyzed (which works fairly good as long as wide
          // windows are used).  To get rid of this assumption, one would
          // need to access "full BX" words, which are not readily
          // available.
          bool is_matched = false;
          int bx_clct_start = bx_alct - match_trig_window_size/2;
          int bx_clct_stop  = bx_alct + match_trig_window_size/2;
          // Empirical correction to match 2009 collision data (firmware change?)
          // (but don't do it for SLHC case, assume it would not be there)
          if (!isSLHC) bx_clct_stop += match_trig_window_size%2;

          for (int bx_clct = bx_clct_start; bx_clct <= bx_clct_stop; bx_clct++) {
            if (bx_clct < 0 || bx_clct >= CSCConstants::MAX_CLCT_TBINS)
              continue;
            // default: do not reuse CLCTs that were used with previous ALCTs
            if (drop_used_clcts && used_clct_mask[bx_clct]) continue;
            if (clct->bestCLCT[bx_clct].isValid()) {
              if (infoV > 1) LogTrace("CSCMotherboard")
                               << "Successful CLCT-ALCT match: bx_alct = " << bx_alct
                               << "; match window: [" << bx_clct_start << "; " << bx_clct_stop
                               << "]; bx_clct = " << bx_clct;
              correlateLCTs(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct],
                            clct->bestCLCT[bx_clct], clct->secondCLCT[bx_clct],
                            CSCCorrelatedLCTDigi::ALCTCLCT);
              used_clct_mask[bx_clct] += 1;
              is_matched = true;
              bx_clct_matched = bx_clct;
              break;
            }
          }
          // No CLCT within the match time interval found: report ALCT-only LCT
          // (use dummy CLCTs).
          if (!is_matched) {
            if (infoV > 1) LogTrace("CSCMotherboard")
                             << "Unsuccessful CLCT-ALCT match (ALCT only): bx_alct = "
                             << bx_alct << "; match window: [" << bx_clct_start
                             << "; " << bx_clct_stop << "]";
            correlateLCTs(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct],
                          clct->bestCLCT[bx_alct], clct->secondCLCT[bx_alct],
                          CSCCorrelatedLCTDigi::ALCTONLY);
          }
        }
        // No valid ALCTs; attempt to make CLCT-only LCT.  Use only CLCTs
        // which have zeroth chance to be matched at later cathode times.
        // (I am not entirely sure this perfectly matches the firmware logic.)
        // Use dummy ALCTs.
        else {
          int bx_clct = bx_alct - match_trig_window_size/2;
          if (bx_clct >= 0 && bx_clct > bx_clct_matched) {
            if (clct->bestCLCT[bx_clct].isValid()) {
              if (infoV > 1) LogTrace("CSCMotherboard")
                               << "Unsuccessful CLCT-ALCT match (CLCT only): bx_clct = "
                               << bx_clct;
              correlateLCTs(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct],
                            clct->bestCLCT[bx_clct], clct->secondCLCT[bx_clct],
                            CSCCorrelatedLCTDigi::CLCTONLY);
            }
          }
        }
      }
    }

    // Debug first and second LCTs
    if (infoV > 0) {
      for (int bx = 0; bx < CSCConstants::MAX_LCT_TBINS; bx++) {
        if (firstLCT[bx].isValid())
          LogDebug("CSCMotherboard") << firstLCT[bx];
        if (secondLCT[bx].isValid())
          LogDebug("CSCMotherboard") << secondLCT[bx];
      }
    }
  }
  // No valid ALCT and/or CLCT processor
  else {
    if (infoV >= 0) edm::LogError("L1CSCTPEmulatorSetupError")
      << "+++ run() called for non-existing ALCT/CLCT processor! +++ \n";
  }
}

// Returns vector of read-out correlated LCTs, if any.  Starts with
// the vector of all found LCTs and selects the ones in the read-out
// time window.
std::vector<CSCCorrelatedLCTDigi> CSCMotherboard::readoutLCTs() const {
  std::vector<CSCCorrelatedLCTDigi> tmpV;

  // The start time of the L1A*LCT coincidence window should be related
  // to the fifo_pretrig parameter, but I am not completely sure how.
  // Just choose it such that the window is centered at bx=7.  This may
  // need further tweaking if the value of tmb_l1a_window_size changes.
  //static int early_tbins = 4;

  // Empirical correction to match 2009 collision data (firmware change?)
  int lct_bins   = tmb_l1a_window_size;
  int late_tbins = early_tbins + lct_bins;

  int ifois = 0;
  if (ifois == 0) {
    if (infoV >= 0 && early_tbins < 0) {
      edm::LogWarning("L1CSCTPEmulatorSuspiciousParameters")
        << "+++ early_tbins = " << early_tbins
        << "; in-time LCTs are not getting read-out!!! +++" << "\n";
    }

    if (late_tbins > CSCConstants::MAX_LCT_TBINS-1) {
      if (infoV >= 0) edm::LogWarning("L1CSCTPEmulatorSuspiciousParameters")
        << "+++ Allowed range of time bins, [0-" << late_tbins
        << "] exceeds max allowed, " << CSCConstants::MAX_LCT_TBINS-1 << " +++\n"
        << "+++ Set late_tbins to max allowed +++\n";
      late_tbins = CSCConstants::MAX_LCT_TBINS-1;
    }
    ifois = 1;
  }

  // Start from the vector of all found correlated LCTs and select
  // those within the LCT*L1A coincidence window.
  int bx_readout = -1;
  const std::vector<CSCCorrelatedLCTDigi>& all_lcts = getLCTs();
  for (auto plct = all_lcts.begin(); plct != all_lcts.end(); plct++) {
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
std::vector<CSCCorrelatedLCTDigi> CSCMotherboard::getLCTs() const {
  std::vector<CSCCorrelatedLCTDigi> tmpV;

  bool me11 = (theStation == 1 &&
               CSCTriggerNumbering::ringFromTriggerLabels(theStation,
                                                          theTrigChamber)==1);

  // Do not report LCTs found in ME1/A if mpc_block_me1/a is set.
  for (int bx = 0; bx < CSCConstants::MAX_LCT_TBINS; bx++) {
    if (firstLCT[bx].isValid())
      if (!mpc_block_me1a || (!me11 || firstLCT[bx].getStrip() <= 127))
        tmpV.push_back(firstLCT[bx]);
    if (secondLCT[bx].isValid())
      if (!mpc_block_me1a || (!me11 || secondLCT[bx].getStrip() <= 127))
        tmpV.push_back(secondLCT[bx]);
  }
  return tmpV;
}

void CSCMotherboard::correlateLCTs(const CSCALCTDigi& bALCT,
                                   const CSCALCTDigi& sALCT,
                                   const CSCCLCTDigi& bCLCT,
                                   const CSCCLCTDigi& sCLCT,
                                   int type)
{
  CSCALCTDigi bestALCT = bALCT;
  CSCALCTDigi secondALCT = sALCT;
  CSCCLCTDigi bestCLCT = bCLCT;
  CSCCLCTDigi secondCLCT = sCLCT;

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
    const CSCCorrelatedLCTDigi& lct = constructLCTs(bestALCT, bestCLCT, type, 1);
    int bx = lct.getBX();
    if (bx >= 0 && bx < CSCConstants::MAX_LCT_TBINS) {
      firstLCT[bx] = lct;
    }
    else {
      if (infoV > 0) edm::LogWarning("L1CSCTPEmulatorOutOfTimeLCT")
        << "+++ Bx of first LCT candidate, " << bx
        << ", is not within the allowed range, [0-" << CSCConstants::MAX_LCT_TBINS-1
        << "); skipping it... +++\n";
    }
  }

  if (((secondALCT != bestALCT) || (secondCLCT != bestCLCT)) &&
      ((alct_trig_enable  && secondALCT.isValid()) ||
       (clct_trig_enable  && secondCLCT.isValid()) ||
       (match_trig_enable && secondALCT.isValid() && secondCLCT.isValid()))) {
    const CSCCorrelatedLCTDigi& lct = constructLCTs(secondALCT, secondCLCT, type, 2);
    int bx = lct.getBX();
    if (bx >= 0 && bx < CSCConstants::MAX_LCT_TBINS) {
      secondLCT[bx] = lct;
    }
    else {
      if (infoV > 0) edm::LogWarning("L1CSCTPEmulatorOutOfTimeLCT")
        << "+++ Bx of second LCT candidate, " << bx
        << ", is not within the allowed range, [0-" << CSCConstants::MAX_LCT_TBINS-1
        << "); skipping it... +++\n";
    }
  }
}

// This method calculates all the TMB words and then passes them to the
// constructor of correlated LCTs.
CSCCorrelatedLCTDigi CSCMotherboard::constructLCTs(const CSCALCTDigi& aLCT,
                                                   const CSCCLCTDigi& cLCT,
                                                   int type,
                                                   int trknmb) const {
  // CLCT pattern number
  unsigned int pattern = encodePattern(cLCT.getPattern(), cLCT.getStripType());

  // LCT quality number
  unsigned int quality = findQuality(aLCT, cLCT);

  // Bunch crossing: get it from cathode LCT if anode LCT is not there.
  int bx = aLCT.isValid() ? aLCT.getBX() : cLCT.getBX();

  // construct correlated LCT
  CSCCorrelatedLCTDigi thisLCT(trknmb, 1, quality, aLCT.getKeyWG(),
                               cLCT.getKeyStrip(), pattern, cLCT.getBend(),
                               bx, 0, 0, 0, theTrigChamber);
  thisLCT.setType(type);
  // make sure to shift the ALCT BX from 8 to 3!
  thisLCT.setALCT(getBXShiftedALCT(aLCT));
  thisLCT.setCLCT(cLCT);
  return thisLCT;
}

// CLCT pattern number: encodes the pattern number itself and
// whether the pattern consists of half-strips or di-strips.
unsigned int CSCMotherboard::encodePattern(const int ptn,
                                           const int stripType) const {
  const int kPatternBitWidth = 4;

  // In the TMB07 firmware, LCT pattern is just a 4-bit CLCT pattern.
  unsigned int pattern = (abs(ptn) & ((1<<kPatternBitWidth)-1));

  return pattern;
}

// 4-bit LCT quality number.
unsigned int CSCMotherboard::findQuality(const CSCALCTDigi& aLCT,
                                         const CSCCLCTDigi& cLCT) const
{
  unsigned int quality = 0;

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

CSCALCTDigi
CSCMotherboard::getBXShiftedALCT(const CSCALCTDigi& aLCT) const
{
  CSCALCTDigi aLCT_shifted = aLCT;
  aLCT_shifted.setBX(aLCT_shifted.getBX() - (CSCConstants::LCT_CENTRAL_BX - tmb_l1a_window_size/2));
  return aLCT_shifted;
}
