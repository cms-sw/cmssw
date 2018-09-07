#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"
#include "L1Trigger/CSCTriggerPrimitives/src/CSCGEMMotherboardME11.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"

CSCGEMMotherboardME11::CSCGEMMotherboardME11(unsigned endcap, unsigned station,
					     unsigned sector, unsigned subsector,
					     unsigned chamber,
					     const edm::ParameterSet& conf)
  : CSCGEMMotherboard(endcap, station, sector, subsector, chamber, conf)
  , allLCTs1b(match_trig_window_size)
  , allLCTs1a(match_trig_window_size)
  // special configuration parameters for ME11 treatment
  , smartME1aME1b(commonParams_.getParameter<bool>("smartME1aME1b"))
  , disableME1a(commonParams_.getParameter<bool>("disableME1a"))
  , gangedME1a(commonParams_.getParameter<bool>("gangedME1a"))
  , dropLowQualityCLCTsNoGEMs_ME1a_(tmbParams_.getParameter<bool>("dropLowQualityCLCTsNoGEMs_ME1a"))
  , dropLowQualityCLCTsNoGEMs_ME1b_(tmbParams_.getParameter<bool>("dropLowQualityCLCTsNoGEMs_ME1b"))
  , dropLowQualityALCTsNoGEMs_ME1a_(tmbParams_.getParameter<bool>("dropLowQualityALCTsNoGEMs_ME1a"))
  , dropLowQualityALCTsNoGEMs_ME1b_(tmbParams_.getParameter<bool>("dropLowQualityALCTsNoGEMs_ME1b"))
  , buildLCTfromALCTandGEM_ME1a_(tmbParams_.getParameter<bool>("buildLCTfromALCTandGEM_ME1a"))
  , buildLCTfromALCTandGEM_ME1b_(tmbParams_.getParameter<bool>("buildLCTfromALCTandGEM_ME1b"))
  , buildLCTfromCLCTandGEM_ME1a_(tmbParams_.getParameter<bool>("buildLCTfromCLCTandGEM_ME1a"))
  , buildLCTfromCLCTandGEM_ME1b_(tmbParams_.getParameter<bool>("buildLCTfromCLCTandGEM_ME1b"))
  , promoteCLCTGEMquality_ME1a_(tmbParams_.getParameter<bool>("promoteCLCTGEMquality_ME1a"))
  , promoteCLCTGEMquality_ME1b_(tmbParams_.getParameter<bool>("promoteCLCTGEMquality_ME1b"))
{
  if (!smartME1aME1b) edm::LogError("L1CSCTPEmulatorConfigError")
    << "+++ Upgrade CSCGEMMotherboardME11 constructed while smartME1aME1b is not set! +++\n";

  const edm::ParameterSet clctParams(conf.getParameter<edm::ParameterSet>("clctSLHC"));
  clct1a.reset( new CSCCathodeLCTProcessor(endcap, station, sector, subsector, chamber, clctParams, commonParams_, tmbParams_) );
  clct1a->setRing(4);

  // set LUTs
  tmbLUT_.reset(new CSCGEMMotherboardLUTME11());
}


CSCGEMMotherboardME11::CSCGEMMotherboardME11() :
  CSCGEMMotherboard()
  , allLCTs1b(match_trig_window_size)
  , allLCTs1a(match_trig_window_size)
{
  // Constructor used only for testing.

  clct1a.reset( new CSCCathodeLCTProcessor() );
  clct1a->setRing(4);
}


CSCGEMMotherboardME11::~CSCGEMMotherboardME11()
{
}


void CSCGEMMotherboardME11::clear()
{
  CSCMotherboard::clear();
  CSCGEMMotherboard::clear();

  if (clct1a) clct1a->clear();
  for (int bx = 0; bx < CSCConstants::MAX_LCT_TBINS; bx++) {
    for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++) {
      for (int i=0;i<CSCConstants::MAX_LCTS_PER_CSC;i++) {
        allLCTs1b(bx,mbx,i).clear();
        allLCTs1a(bx,mbx,i).clear();
      }
    }
  }
}


//===============================================
//use ALCTs, CLCTs, GEMs to build LCTs
//loop over each BX to find valid ALCT in ME1b, try ALCT-CLCT match, if ALCT-CLCT match failed, try ALCT-GEM match
//do the same in ME1a
//sort LCTs according to different algorithm, and send out number of LCTs up to max_lcts
//===============================================

void CSCGEMMotherboardME11::run(const CSCWireDigiCollection* wiredc,
				const CSCComparatorDigiCollection* compdc,
				const GEMPadDigiCollection* gemPads)
{
  clear();
  setupGeometry();
  debugLUTs();

  if (gem_g != nullptr) {
    if (infoV >= 0) edm::LogInfo("L1CSCTPEmulatorSetupInfo")
		      << "+++ run() called for GEM-CSC integrated trigger! +++ \n";
    gemGeometryAvailable = true;
  }

  // check for GEM geometry
  if (not gemGeometryAvailable){
    if (infoV >= 0) edm::LogError("L1CSCTPEmulatorSetupError")
		      << "+++ run() called for GEM-CSC integrated trigger without valid GEM geometry! +++ \n";
    return;
  }
  gemCoPadV = coPadProcessor->run(gemPads); // run copad processor in GE1/1

  if (!( alct and clct and  clct1a and smartME1aME1b))
  {
    if (infoV >= 0) edm::LogError("L1CSCTPEmulatorSetupError")
      << "+++ run() called for non-existing ALCT/CLCT processor! +++ \n";
    return;
  }

  alct->setCSCGeometry(csc_g);
  clct->setCSCGeometry(csc_g);
  clct1a->setCSCGeometry(csc_g);

  alctV = alct->run(wiredc); // run anodeLCT
  clctV1b = clct->run(compdc); // run cathodeLCT in ME1/b
  clctV1a = clct1a->run(compdc); // run cathodeLCT in ME1/a

  // if there are no ALCTs and no CLCTs, it does not make sense to run this TMB
  if (alctV.empty() and clctV1b.empty() and clctV1a.empty()) return;

  int used_clct_mask[20], used_clct_mask_1a[20];
  for (int b=0;b<20;b++)
    used_clct_mask[b] = used_clct_mask_1a[b] = 0;

  // retrieve CSCChamber geometry
  const CSCDetId me1aId(theEndcap, 1, 4, theChamber);
  const CSCChamber* cscChamberME1a(csc_g->chamber(me1aId));

  retrieveGEMPads(gemPads, gemId);
  retrieveGEMCoPads();

  const bool hasPads(!pads_.empty());
  const bool hasCoPads(hasPads and !coPads_.empty());

  // ALCT-centric matching
  for (int bx_alct = 0; bx_alct < CSCConstants::MAX_ALCT_TBINS; bx_alct++)
  {
    if (alct->bestALCT[bx_alct].isValid())
    {
      const int bx_clct_start(bx_alct - match_trig_window_size/2 - alctClctOffset);
      const int bx_clct_stop(bx_alct + match_trig_window_size/2 - alctClctOffset);
      const int bx_copad_start(bx_alct - maxDeltaBXCoPad_);
      const int bx_copad_stop(bx_alct + maxDeltaBXCoPad_);

      if (debug_matching){
        LogTrace("CSCGEMCMotherboardME11") << "========================================================================\n"
                                           << "ALCT-CLCT matching in ME1/1 chamber: " << cscChamber->id() << "\n"
                                           << "------------------------------------------------------------------------\n"
                                           << "+++ Best ALCT Details: " << alct->bestALCT[bx_alct]  << "\n"
                                           << "+++ Second ALCT Details: " << alct->secondALCT[bx_alct] << std::endl;

        printGEMTriggerPads(bx_clct_start, bx_clct_stop, CSCPart::ME11);
        printGEMTriggerCoPads(bx_clct_start, bx_clct_stop, CSCPart::ME11);

        LogTrace("CSCGEMCMotherboardME11") << "------------------------------------------------------------------------ \n"
                                           << "Attempt ALCT-CLCT matching in ME1/b in bx range: [" << bx_clct_start << "," << bx_clct_stop << "]" << std::endl;
      }

      // ALCT-to-CLCT matching in ME1b
      int nSuccesFulMatches = 0;
      for (int bx_clct = bx_clct_start; bx_clct <= bx_clct_stop; bx_clct++)
      {
        if (bx_clct < 0 or bx_clct >= CSCConstants::MAX_CLCT_TBINS) continue;
        if (drop_used_clcts and used_clct_mask[bx_clct]) continue;
        if (clct->bestCLCT[bx_clct].isValid())
        {
          const int quality(clct->bestCLCT[bx_clct].getQuality());
          if (debug_matching) LogTrace("CSCGEMCMotherboardME11") << "++Valid ME1b CLCT: " << clct->bestCLCT[bx_clct] << std::endl;

	  // pick the pad that corresponds
	  matches<GEMPadDigi> mPads;
	  matchingPads<GEMPadDigi>(clct->bestCLCT[bx_clct], clct->secondCLCT[bx_clct],
				   alct->bestALCT[bx_alct], alct->secondALCT[bx_alct], ME1B, mPads);
	  matches<GEMCoPadDigi> mCoPads;
	  matchingPads<GEMCoPadDigi>(clct->bestCLCT[bx_clct], clct->secondCLCT[bx_clct],
				     alct->bestALCT[bx_alct], alct->secondALCT[bx_alct], ME1B, mCoPads);

    // Low quality LCTs have at most 3 layers. Check if there is a matching GEM pad to keep the CLCT
    if (dropLowQualityCLCTsNoGEMs_ME1b_ and quality < 4 and hasPads){
      int nFound(mPads.size());
      // these halfstrips (1,2,3,4) and (125,126,127,128) do not have matching GEM pads because GEM chambers are slightly wider than CSCs
      const bool clctInEdge(clct->bestCLCT[bx_clct].getKeyStrip() < 5 or clct->bestCLCT[bx_clct].getKeyStrip() > 124);
            if (clctInEdge){
              if (debug_matching) LogTrace("CSCGEMCMotherboardME11") << "\tInfo: low quality CLCT in CSC chamber edge, don't care about GEM pads" << std::endl;
            }
            else {
              if (nFound != 0){
                if (debug_matching) LogTrace("CSCGEMCMotherboardME11") << "\tInfo: low quality CLCT with " << nFound << " matching GEM trigger pads" << std::endl;
              }
              else {
                if (debug_matching) LogTrace("CSCGEMCMotherboardME11") << "\tWarning: low quality CLCT without matching GEM trigger pad" << std::endl;
                continue;
              }
            }
          }

          ++nSuccesFulMatches;

          //	    if (infoV > 1) LogTrace("CSCMotherboard")
          int mbx = bx_clct-bx_clct_start;
	  correlateLCTsGEM(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct],
			   clct->bestCLCT[bx_clct], clct->secondCLCT[bx_clct],
			   mPads, mCoPads,
			   allLCTs1b(bx_alct,mbx,0), allLCTs1b(bx_alct,mbx,1), ME1B);
          if (debug_matching ) {
            LogTrace("CSCGEMCMotherboardME11") << "Successful ALCT-CLCT match in ME1b: bx_alct = " << bx_alct
                                               << "; match window: [" << bx_clct_start << "; " << bx_clct_stop
                                               << "]; bx_clct = " << bx_clct << "\n"
                                               << "+++ Best CLCT Details: " << clct->bestCLCT[bx_clct] << "\n"
                                               << "+++ Second CLCT Details: " << clct->secondCLCT[bx_clct] << std::endl;
	    if (allLCTs1b(bx_alct,mbx,0).isValid())
		LogTrace("CSCGEMCMotherboardME11") << "LCT #1 "<< allLCTs1b(bx_alct,mbx,0) << std::endl;
	    else
		LogTrace("CSCGEMCMotherboardME11") << "No valid LCT is built from ALCT-CLCT matching in ME1b"  << std::endl;
	    if (allLCTs1b(bx_alct,mbx,1).isValid())
		LogTrace("CSCGEMCMotherboardME11") << "LCT #2 "<< allLCTs1b(bx_alct,mbx,1) << std::endl;
          }

          if (allLCTs1b(bx_alct,mbx,0).isValid()) {
            used_clct_mask[bx_clct] += 1;
            if (match_earliest_clct_only) break;
          }
        }
      }

      // ALCT-to-GEM matching in ME1b
      int nSuccesFulGEMMatches = 0;
      if (nSuccesFulMatches==0 and buildLCTfromALCTandGEM_ME1b_){
        if (debug_matching) LogTrace("CSCGEMCMotherboardME11") << "++No valid ALCT-CLCT matches in ME1b" << std::endl;
        for (int bx_gem = bx_copad_start; bx_gem <= bx_copad_stop; bx_gem++) {
          if (not hasCoPads) {
            continue;
          }

          // find the best matching copad
	  matches<GEMCoPadDigi> copads;
	  matchingPads<CSCALCTDigi, GEMCoPadDigi>(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct], ME1B, copads);

          if (debug_matching) LogTrace("CSCGEMCMotherboardME11") << "\t++Number of matching GEM CoPads in BX " << bx_alct << " : "<< copads.size() << std::endl;
          if (copads.empty()) {
            continue;
          }

	  CSCGEMMotherboard::correlateLCTsGEM(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct],
					      copads, allLCTs1b(bx_alct,0,0), allLCTs1b(bx_alct,0,1), ME1B);

          if (debug_matching) {
            LogTrace("CSCGEMCMotherboardME11") << "Successful ALCT-GEM CoPad match in ME1b: bx_alct = " << bx_alct << "\n\n"
                                               << "------------------------------------------------------------------------" << std::endl << std::endl;
	    if (allLCTs1b(bx_alct,0,0).isValid())
		LogTrace("CSCGEMCMotherboardME11") << "LCT #1 "<< allLCTs1b(bx_alct,0,0) << std::endl;
	    else
		LogTrace("CSCGEMCMotherboardME11") << "No valid LCT is built from ALCT-GEM matching in ME1b"  << std::endl;
	    if (allLCTs1b(bx_alct,0,1).isValid())
		LogTrace("CSCGEMCMotherboardME11") << "LCT #2 "<< allLCTs1b(bx_alct,0,1) << std::endl;
          }

          if (allLCTs1b(bx_alct,0,0).isValid()) {
            ++nSuccesFulGEMMatches;
            if (match_earliest_clct_only) break;
          }
        }
      }

      if (debug_matching) {
        LogTrace("CSCGEMCMotherboardME11") << "========================================================================" << std::endl
                                           << "Summary: " << std::endl;
        if (nSuccesFulMatches>1)
          LogTrace("CSCGEMCMotherboardME11") << "Too many successful ALCT-CLCT matches in ME1b: " << nSuccesFulMatches
					     << ", CSCDetId " << cscChamber->id()
					     << ", bx_alct = " << bx_alct
					     << "; match window: [" << bx_clct_start << "; " << bx_clct_stop << "]" << std::endl;
        else if (nSuccesFulMatches==1)
          LogTrace("CSCGEMCMotherboardME11") << "1 successful ALCT-CLCT match in ME1b: "
					     << " CSCDetId " << cscChamber->id()
					     << ", bx_alct = " << bx_alct
					     << "; match window: [" << bx_clct_start << "; " << bx_clct_stop << "]" << std::endl;
        else if (nSuccesFulGEMMatches==1)
          LogTrace("CSCGEMCMotherboardME11") << "1 successful ALCT-GEM match in ME1b: "
					     << " CSCDetId " << cscChamber->id()
					     << ", bx_alct = " << bx_alct
					     << "; match window: [" << bx_clct_start << "; " << bx_clct_stop << "]" << std::endl;
        else
          LogTrace("CSCGEMCMotherboardME11") << "Unsuccessful ALCT-CLCT match in ME1b: "
					     << "CSCDetId " << cscChamber->id()
					     << ", bx_alct = " << bx_alct
					     << "; match window: [" << bx_clct_start << "; " << bx_clct_stop << "]" << std::endl;

        LogTrace("CSCGEMCMotherboardME11") << "------------------------------------------------------------------------" << std::endl
                                           << "Attempt ALCT-CLCT matching in ME1/a in bx range: [" << bx_clct_start << "," << bx_clct_stop << "]" << std::endl;
      }

      // ALCT-to-CLCT matching in ME1a
      nSuccesFulMatches = 0;
      for (int bx_clct = bx_clct_start; bx_clct <= bx_clct_stop; bx_clct++)
      {
        if (bx_clct < 0 or bx_clct >= CSCConstants::MAX_CLCT_TBINS) continue;
        if (drop_used_clcts and used_clct_mask_1a[bx_clct]) continue;
        if (clct1a->bestCLCT[bx_clct].isValid())
        {
          const int quality(clct1a->bestCLCT[bx_clct].getQuality());
          if (debug_matching) LogTrace("CSCGEMCMotherboardME11") << "++Valid ME1a CLCT: " << clct1a->bestCLCT[bx_clct] << std::endl;

	  // pick the pad that corresponds
	  matches<GEMPadDigi> mPads;
	  matchingPads<GEMPadDigi>(clct1a->bestCLCT[bx_clct], clct1a->secondCLCT[bx_clct],
				   alct->bestALCT[bx_alct], alct->secondALCT[bx_alct], ME1A, mPads);
	  matches<GEMCoPadDigi> mCoPads;
	  matchingPads<GEMCoPadDigi>(clct1a->bestCLCT[bx_clct], clct1a->secondCLCT[bx_clct],
				     alct->bestALCT[bx_alct], alct->secondALCT[bx_alct], ME1A, mCoPads);

          if (dropLowQualityCLCTsNoGEMs_ME1a_ and quality < 4 and hasPads){
            int nFound(mPads.size());
            const bool clctInEdge(clct1a->bestCLCT[bx_clct].getKeyStrip() < 4 or clct1a->bestCLCT[bx_clct].getKeyStrip() > 93);
            if (clctInEdge){
              if (debug_matching) LogTrace("CSCGEMCMotherboardME11") << "\tInfo: low quality CLCT in CSC chamber edge, don't care about GEM pads" << std::endl;
            }
            else {
              if (nFound != 0){
                if (debug_matching) LogTrace("CSCGEMCMotherboardME11") << "\tInfo: low quality CLCT with " << nFound << " matching GEM trigger pads" << std::endl;
              }
              else {
                if (debug_matching) LogTrace("CSCGEMCMotherboardME11") << "\tWarning: low quality CLCT without matching GEM trigger pad" << std::endl;
                continue;
              }
            }
          }
          ++nSuccesFulMatches;
          int mbx = bx_clct-bx_clct_start;
          correlateLCTsGEM(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct],
			   clct1a->bestCLCT[bx_clct], clct1a->secondCLCT[bx_clct],
			   mPads, mCoPads,
			   allLCTs1a(bx_alct,mbx,0), allLCTs1a(bx_alct,mbx,1), ME1A);
          if (debug_matching) {
            LogTrace("CSCGEMCMotherboardME11") << "Successful ALCT-CLCT match in ME1a: bx_alct = " << bx_alct
                                               << "; match window: [" << bx_clct_start << "; " << bx_clct_stop
                                               << "]; bx_clct = " << bx_clct << "\n"
                                               << "+++ Best CLCT Details: " << clct1a->bestCLCT[bx_clct] << "\n"
                                               << "+++ Second CLCT Details: " << clct1a->secondCLCT[bx_clct] << std::endl;
	    if (allLCTs1a(bx_alct,mbx,0).isValid())
		LogTrace("CSCGEMCMotherboardME11") << "LCT #1 "<< allLCTs1a(bx_alct,mbx,0) << std::endl;
	    else
		LogTrace("CSCGEMCMotherboardME11") << "No valid LCT is built from ALCT-CLCT matching in ME1a"  << std::endl;
	    if (allLCTs1a(bx_alct,mbx,1).isValid())
		LogTrace("CSCGEMCMotherboardME11") << "LCT #2 "<< allLCTs1a(bx_alct,mbx,1) << std::endl;
          }
          if (allLCTs1a(bx_alct,mbx,0).isValid()){
            used_clct_mask_1a[bx_clct] += 1;
            if (match_earliest_clct_only) break;
          }
        }
      }

      // ALCT-to-GEM matching in ME1a
      nSuccesFulGEMMatches = 0;
      if (nSuccesFulMatches==0 and buildLCTfromALCTandGEM_ME1a_){
        if (debug_matching) LogTrace("CSCGEMCMotherboardME11") << "++No valid ALCT-CLCT matches in ME1a" << std::endl;
        for (int bx_gem = bx_copad_start; bx_gem <= bx_copad_stop; bx_gem++) {
          if (not hasCoPads) {
            continue;
          }

          // find the best matching copad
	  matches<GEMCoPadDigi> copads;
	  matchingPads<CSCALCTDigi, GEMCoPadDigi>(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct], ME1A, copads);

          if (debug_matching) LogTrace("CSCGEMCMotherboardME11") << "\t++Number of matching GEM CoPads in BX " << bx_alct << " : "<< copads.size() << std::endl;
          if (copads.empty()) {
            continue;
          }

	  CSCGEMMotherboard::correlateLCTsGEM(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct],
					      copads, allLCTs1a(bx_alct,0,0), allLCTs1a(bx_alct,0,1), ME1A);

          if (debug_matching) {
            LogTrace("CSCGEMCMotherboardME11") << "Successful ALCT-GEM CoPad match in ME1a: bx_alct = " << bx_alct << "\n\n"
                                               << "------------------------------------------------------------------------" << std::endl << std::endl;
	    if (allLCTs1a(bx_alct,0,0).isValid())
		LogTrace("CSCGEMCMotherboardME11") << "LCT #1 "<< allLCTs1a(bx_alct,0,0) << std::endl;
	    else
		LogTrace("CSCGEMCMotherboardME11") << "No valid LCT is built from ALCT-GEM matching in ME1a"  << std::endl;
	    if (allLCTs1a(bx_alct,0,1).isValid())
		LogTrace("CSCGEMCMotherboardME11") << "LCT #2 "<< allLCTs1a(bx_alct,0,1) << std::endl;
          }

          if (allLCTs1a(bx_alct,0,0).isValid()) {
             ++nSuccesFulGEMMatches;
            if (match_earliest_clct_only) break;
          }
        }
      }

      if (debug_matching) {
        LogTrace("CSCGEMCMotherboardME11") << "======================================================================== \n"
                                           << "Summary: " << std::endl;
        if (nSuccesFulMatches>1)
          LogTrace("CSCGEMCMotherboardME11") << "Too many successful ALCT-CLCT matches in ME1a: " << nSuccesFulMatches
					     << ", CSCDetId " << cscChamberME1a->id()
					     << ", bx_alct = " << bx_alct
					     << "; match window: [" << bx_clct_start << "; " << bx_clct_stop << "]" << std::endl;
        else if (nSuccesFulMatches==1)
          LogTrace("CSCGEMCMotherboardME11") << "1 successful ALCT-CLCT match in ME1a: "
					     << " CSCDetId " << cscChamberME1a->id()
					     << ", bx_alct = " << bx_alct
					     << "; match window: [" << bx_clct_start << "; " << bx_clct_stop << "]" << std::endl;
        else if (nSuccesFulGEMMatches==1)
          LogTrace("CSCGEMCMotherboardME11") << "1 successful ALCT-GEM match in ME1a: "
					     << " CSCDetId " << cscChamberME1a->id()
					     << ", bx_alct = " << bx_alct
					     << "; match window: [" << bx_clct_start << "; " << bx_clct_stop << "]" << std::endl;
        else
          LogTrace("CSCGEMCMotherboardME11") << "Unsuccessful ALCT-CLCT match in ME1a: "
					     << "CSCDetId " << cscChamberME1a->id()
					     << ", bx_alct = " << bx_alct
					     << "; match window: [" << bx_clct_start << "; " << bx_clct_stop << "]" << std::endl;
      }

    } // end of ALCT valid block
    else {
      auto coPads(coPads_[bx_alct]);
      if (!coPads.empty()) {
        // keep it simple for the time being, only consider the first copad
        const int bx_clct_start(bx_alct - match_trig_window_size/2);
        const int bx_clct_stop(bx_alct + match_trig_window_size/2);

        // matching in ME1b
        if (buildLCTfromCLCTandGEM_ME1b_) {
          for (int bx_clct = bx_clct_start; bx_clct <= bx_clct_stop; bx_clct++) {
            if (bx_clct < 0 or bx_clct >= CSCConstants::MAX_CLCT_TBINS) continue;
            if (drop_used_clcts and used_clct_mask[bx_clct]) continue;
            if (clct->bestCLCT[bx_clct].isValid()) {
              const int quality(clct->bestCLCT[bx_clct].getQuality());
              // only use high-Q stubs for the time being
              if (quality < 4) continue;
              int mbx = bx_clct-bx_clct_start;
              CSCGEMMotherboard::correlateLCTsGEM(clct->bestCLCT[bx_clct], clct->secondCLCT[bx_clct], coPads,
                                                  allLCTs1b(bx_alct,mbx,0), allLCTs1b(bx_alct,mbx,1), ME1B);
              if (debug_matching) {
                //	    if (infoV > 1) LogTrace("CSCMotherboard")
                LogTrace("CSCGEMCMotherboardME11") << "Successful GEM-CLCT match in ME1b: bx_alct = " << bx_alct
                                                   << "; match window: [" << bx_clct_start << "; " << bx_clct_stop
                                                   << "]; bx_clct = " << bx_clct << "\n"
                                                   << "+++ Best CLCT Details: " << clct->bestCLCT[bx_clct] << "\n"
                                                   << "+++ Second CLCT Details: " << clct->secondCLCT[bx_clct] << std::endl;
              }
              if (allLCTs1b(bx_alct,mbx,0).isValid()) {
                used_clct_mask[bx_clct] += 1;
                if (match_earliest_clct_only) break;
              }
            }
          }
        }

        // matching in ME1a
        if (buildLCTfromCLCTandGEM_ME1a_) {
          for (int bx_clct = bx_clct_start; bx_clct <= bx_clct_stop; bx_clct++) {
            if (bx_clct < 0 || bx_clct >= CSCConstants::MAX_CLCT_TBINS) continue;
            if (drop_used_clcts && used_clct_mask_1a[bx_clct]) continue;
            if (clct1a->bestCLCT[bx_clct].isValid()){
              const int quality(clct1a->bestCLCT[bx_clct].getQuality());
              // only use high-Q stubs for the time being
              if (quality < 4) continue;
              int mbx = bx_clct-bx_clct_start;
	      CSCGEMMotherboard::correlateLCTsGEM(clct1a->bestCLCT[bx_clct], clct1a->secondCLCT[bx_clct], coPads,
						  allLCTs1a(bx_alct,mbx,0), allLCTs1a(bx_alct,mbx,1), ME1A);
              if (debug_matching) {
                //	    if (infoV > 1) LogTrace("CSCMotherboard")
                LogTrace("CSCGEMCMotherboardME11") << "Successful GEM-CLCT match in ME1a: bx_alct = " << bx_alct
                                                   << "; match window: [" << bx_clct_start << "; " << bx_clct_stop
                                                   << "]; bx_clct = " << bx_clct << "\n"
                                                   << "+++ Best CLCT Details: " << clct1a->bestCLCT[bx_clct] << "\n"
                                                   << "+++ Second CLCT Details: " << clct1a->secondCLCT[bx_clct] << std::endl;
              }
              if (allLCTs1a(bx_alct,mbx,0).isValid()){
                used_clct_mask_1a[bx_clct] += 1;
                if (match_earliest_clct_only) break;
              }
            }
          }
        }
      }
    }
  } // end of ALCT-centric matching

  if (debug_matching){
    LogTrace("CSCGEMCMotherboardME11") << "======================================================================== \n"
                                       << "Counting the LCTs \n"
                                       << "========================================================================" << std::endl;
  }

  // reduction of nLCTs per each BX
  for (int bx = 0; bx < CSCConstants::MAX_LCT_TBINS; bx++)
  {
    // counting
    unsigned int n1a=0, n1b=0;
    for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++)
      for (int i=0;i<CSCConstants::MAX_LCTS_PER_CSC;i++)
      {
        int cbx = bx + mbx - match_trig_window_size/2;
        if (allLCTs1b(bx,mbx,i).isValid())
        {
          n1b++;
	    if (infoV > 0) LogDebug("CSCGEMMotherboardME11")
			     << "1b LCT"<<i+1<<" "<<bx<<"/"<<cbx<<": "<<allLCTs1b(bx,mbx,i)<<std::endl;
        }
        if (allLCTs1a(bx,mbx,i).isValid())
        {
          n1a++;
	    if (infoV > 0) LogDebug("CSCGEMMotherboardME11")
			     << "1a LCT"<<i+1<<" "<<bx<<"/"<<cbx<<": "<<allLCTs1a(bx,mbx,i)<<std::endl;
        }
      }
    if (infoV > 0 and n1a+n1b>0) LogDebug("CSCGEMMotherboardME11")
				   <<"bx "<<bx<<" nLCT:"<<n1a<<" "<<n1b<<" "<<n1a+n1b<<std::endl;

    // some simple cross-bx sorting algorithms
    if (tmb_cross_bx_algo == 1 and (n1a>2 or n1b>2) )
    {
      n1a=0, n1b=0;
      for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++)
        for (int i=0;i<CSCConstants::MAX_LCTS_PER_CSC;i++)
        {
          if (allLCTs1b(bx,pref[mbx],i).isValid())
          {
            n1b++;
            if (n1b>2) allLCTs1b(bx,pref[mbx],i).clear();
          }
          if (allLCTs1a(bx,pref[mbx],i).isValid())
          {
            n1a++;
            if (n1a>2) allLCTs1a(bx,pref[mbx],i).clear();
          }
        }

      n1a=0, n1b=0;
      for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++)
        for (int i=0;i<CSCConstants::MAX_LCTS_PER_CSC;i++)
        {
          int cbx = bx + mbx - match_trig_window_size/2;
          if (allLCTs1b(bx,mbx,i).isValid())
          {
            n1b++;
           if (infoV > 0) LogDebug("CSCGEMMotherboardME11")
			    << "1b LCT"<<i+1<<" "<<bx<<"/"<<cbx<<": "<<allLCTs1b(bx,mbx,i)<< std::endl;
          }
          if (allLCTs1a(bx,mbx,i).isValid())
          {
            n1a++;
            if (infoV > 0) LogDebug("CSCGEMMotherboardME11")
			     << "1a LCT"<<i+1<<" "<<bx<<"/"<<cbx<<": "<<allLCTs1a(bx,mbx,i)<< std::endl;
          }
        }
      if (infoV > 0 and n1a+n1b>0) LogDebug("CSCGEMMotherboardME11")
				     <<"bx "<<bx<<" nnLCT:"<<n1a<<" "<<n1b<<" "<<n1a+n1b<<std::endl;
    } // x-bx sorting

    // Maximum 2 per whole ME11 per BX case:
    // (supposedly, now we should have max 2 per bx in each 1a and 1b)
    if (n1a+n1b > max_lcts and tmb_cross_bx_algo == 1)
    {
      // do it simple so far: take all low eta 1/b stubs
      unsigned int nLCT=n1b;
      n1a=0;
      // right now nLCT<=2; cut 1a if necessary
      for (unsigned int mbx=0; mbx<match_trig_window_size; mbx++)
        for (int i=0;i<CSCConstants::MAX_LCTS_PER_CSC;i++)
          if (allLCTs1a(bx,mbx,i).isValid()) {
            nLCT++;
            if (nLCT>max_lcts) allLCTs1a(bx,mbx,i).clear();
            else n1a++;
          }
      if (infoV > 0 and nLCT>0) LogDebug("CSCGEMMotherboardME11")
				  <<"bx "<<bx<<" nnnLCT: "<<n1a<<" "<<n1b<<" "<<n1a+n1b<< std::endl;
    }
  }// reduction per bx

  unsigned int n1b=0, n1a=0;
  LogTrace("CSCGEMCMotherboardME11") << "======================================================================== \n"
                                     << "Counting the final LCTs \n"
                                     << "======================================================================== \n"
                                     << "tmb_cross_bx_algo: " << tmb_cross_bx_algo << std::endl;

  for (const auto& p : readoutLCTs1b()) {
    n1b++;
    LogTrace("CSCGEMCMotherboardME11") << "1b LCT "<<n1b<<"  " << p <<std::endl;
  }

  for (const auto& p : readoutLCTs1a()){
    n1a++;
    LogTrace("CSCGEMCMotherboardME11") << "1a LCT "<<n1a<<"  " << p <<std::endl;

  }
}

std::vector<CSCCorrelatedLCTDigi> CSCGEMMotherboardME11::readoutLCTs1a() const
{
  return readoutLCTs(ME1A);
}


std::vector<CSCCorrelatedLCTDigi> CSCGEMMotherboardME11::readoutLCTs1b() const
{
  return readoutLCTs(ME1B);
}


// Returns vector of read-out correlated LCTs, if any.  Starts with
// the vector of all found LCTs and selects the ones in the read-out
// time window.
std::vector<CSCCorrelatedLCTDigi> CSCGEMMotherboardME11::readoutLCTs(enum CSCPart me1ab) const
{
  std::vector<CSCCorrelatedLCTDigi> tmpV;

  // The start time of the L1A*LCT coincidence window should be related
  // to the fifo_pretrig parameter, but I am not completely sure how.
  // Just choose it such that the window is centered at bx=7.  This may
  // need further tweaking if the value of tmb_l1a_window_size changes.
  //static int early_tbins = 4;
  // The number of LCT bins in the read-out is given by the
  // tmb_l1a_window_size parameter, forced to be odd
  const int lct_bins   =
    (tmb_l1a_window_size % 2 == 0) ? tmb_l1a_window_size + 1 : tmb_l1a_window_size;
  const int late_tbins = early_tbins + lct_bins;

  // Start from the vector of all found correlated LCTs and select
  // those within the LCT*L1A coincidence window.
  int bx_readout = -1;
  std::vector<CSCCorrelatedLCTDigi> all_lcts;
  switch(tmb_cross_bx_algo){
  case 1:
	  if (me1ab == ME1A and not (mpc_block_me1a or disableME1a)) {
	    allLCTs1a.getMatched(all_lcts);
	  }else if (me1ab == ME1B) {
	    allLCTs1b.getMatched(all_lcts);
	  }
	  break;
  case 2: sortLCTs(all_lcts, me1ab, CSCUpgradeMotherboard::sortLCTsByQuality);
    break;
  case 3: sortLCTs(all_lcts, me1ab, CSCUpgradeMotherboard::sortLCTsByGEMDphi);
    break;
  default: LogTrace("CSCGEMCMotherboardME11")<<"tmb_cross_bx_algo error" <<std::endl;
    break;
  }

  for (const auto& lct: all_lcts)
  {
    if (!lct.isValid()) continue;

    int bx = lct.getBX();
    // Skip LCTs found too early relative to L1Accept.
    if (bx <= early_tbins) continue;

    // Skip LCTs found too late relative to L1Accept.
    if (bx > late_tbins) continue;

    // If (readout_earliest_2) take only LCTs in the earliest bx in the read-out window:
    // in digi->raw step, LCTs have to be packed into the TMB header, and
    // currently there is room just for two.
    if (readout_earliest_2 and (bx_readout == -1 or bx == bx_readout) )
    {
      tmpV.push_back(lct);
      if (bx_readout == -1) bx_readout = bx;
    }
    else tmpV.push_back(lct);
  }
  return tmpV;
}

//sort LCTs in each BX
void CSCGEMMotherboardME11::sortLCTs(std::vector<CSCCorrelatedLCTDigi>& LCTs, int bx, enum CSCPart me,
                                     bool (*sorter)(const CSCCorrelatedLCTDigi&, const CSCCorrelatedLCTDigi&)) const
{
  const auto& allLCTs(me==ME1A ? allLCTs1a : allLCTs1b);

  allLCTs.getTimeMatched(bx, LCTs);

  CSCUpgradeMotherboard::sortLCTs(LCTs, *sorter);

  if (LCTs.size() > max_lcts) LCTs.erase(LCTs.begin()+max_lcts, LCTs.end());
}

//sort LCTs in whole LCTs BX window
void CSCGEMMotherboardME11::sortLCTs(std::vector<CSCCorrelatedLCTDigi>& LCTs_final, enum CSCPart me,
                                     bool (*sorter)(const CSCCorrelatedLCTDigi&, const CSCCorrelatedLCTDigi&)) const
{
  for (int bx = 0; bx < CSCConstants::MAX_LCT_TBINS; bx++)
    {
      // get sorted LCTs per subchamber
      std::vector<CSCCorrelatedLCTDigi> LCTs1a;
      std::vector<CSCCorrelatedLCTDigi> LCTs1b;

      CSCGEMMotherboardME11::sortLCTs(LCTs1a, bx, ME1A, *sorter);
      CSCGEMMotherboardME11::sortLCTs(LCTs1b, bx, ME1B, *sorter);

      // temporary collection with all LCTs in the whole chamber
      std::vector<CSCCorrelatedLCTDigi> LCTs_tmp;
      LCTs_tmp.insert(LCTs_tmp.begin(), LCTs1b.begin(), LCTs1b.end());
      LCTs_tmp.insert(LCTs_tmp.end(), LCTs1a.begin(), LCTs1a.end());

      // sort the selected LCTs
      CSCUpgradeMotherboard::sortLCTs(LCTs_tmp, *sorter);

      //LCTs reduction per BX
      if (max_lcts > 0)
        {
	  if (LCTs_tmp.size() > max_lcts) LCTs_tmp.erase(LCTs_tmp.begin()+max_lcts, LCTs_tmp.end());//double check
          // loop on all the selected LCTs
          for (const auto& p: LCTs_tmp){
            // case when you only want to readout ME1A
            if (me==ME1A and std::find(LCTs1a.begin(), LCTs1a.end(), p) != LCTs1a.end()){
              LCTs_final.push_back(p);
            }
            // case when you only want to readout ME1B
            else if(me==ME1B and std::find(LCTs1b.begin(), LCTs1b.end(), p) != LCTs1b.end()){
              LCTs_final.push_back(p);
            }
          }
        }
      else {
        if (!LCTs1a.empty() and !LCTs1b.empty() and me==ME1A)
          LCTs_final.push_back(*LCTs1a.begin());
        else if (!LCTs1a.empty() and !LCTs1b.empty() and me==ME1B)
          LCTs_final.push_back(*LCTs1b.begin());
        else if (!LCTs1a.empty() and LCTs1b.empty() and me==ME1A)
          LCTs_final.insert(LCTs_final.end(), LCTs1a.begin(), LCTs1a.end());
        else if (!LCTs1b.empty() and LCTs1a.empty() and me==ME1B)
          LCTs_final.insert(LCTs_final.end(), LCTs1b.begin(), LCTs1b.end());
      }
    }
}


bool CSCGEMMotherboardME11::doesALCTCrossCLCT(const CSCALCTDigi &a, const CSCCLCTDigi &c, int me) const
{
  if ( !c.isValid() or !a.isValid() ) return false;
  int key_hs = c.getKeyStrip();
  int key_wg = a.getKeyWG();
  if ( me == ME1A )
  {
    if ( !gangedME1a )
    {
      // wrap around ME11 HS number for -z endcap
      if (theEndcap==2) key_hs = CSCConstants::MAX_HALF_STRIP_ME1A_UNGANGED - key_hs;
      if ( key_hs >= (tmbLUT_->get_lut_wg_vs_hs(CSCPart::ME1A))[key_wg][0] and
	   key_hs <= (tmbLUT_->get_lut_wg_vs_hs(CSCPart::ME1A))[key_wg][1]    ) return true;
      return false;
    }
    else
    {
      if (theEndcap==2) key_hs = CSCConstants::MAX_HALF_STRIP_ME1A_GANGED - key_hs;
      if ( key_hs >= (tmbLUT_->get_lut_wg_vs_hs(CSCPart::ME1Ag))[key_wg][0] and
	   key_hs <= (tmbLUT_->get_lut_wg_vs_hs(CSCPart::ME1Ag))[key_wg][1]    ) return true;
      return false;
    }
  }
  if ( me == ME1B)
  {
    if (theEndcap==2) key_hs = CSCConstants::MAX_HALF_STRIP_ME1B - key_hs;
    if ( key_hs >= (tmbLUT_->get_lut_wg_vs_hs(CSCPart::ME1B))[key_wg][0] and
         key_hs <= (tmbLUT_->get_lut_wg_vs_hs(CSCPart::ME1B))[key_wg][1]      ) return true;
  }
  return false;
}


void CSCGEMMotherboardME11::correlateLCTsGEM(const CSCALCTDigi& bALCT,
                                             const CSCALCTDigi& sALCT,
                                             const CSCCLCTDigi& bCLCT,
                                             const CSCCLCTDigi& sCLCT,
                                             const GEMPadDigiIds& pads,
                                             const GEMCoPadDigiIds& copads,
                                             CSCCorrelatedLCTDigi& lct1,
                                             CSCCorrelatedLCTDigi& lct2,
                                             enum CSCPart p) const
{
  CSCALCTDigi bestALCT = bALCT;
  CSCALCTDigi secondALCT = sALCT;
  CSCCLCTDigi bestCLCT = bCLCT;
  CSCCLCTDigi secondCLCT = sCLCT;

  // assume that always anodeBestValid and cathodeBestValid
  if (secondALCT == bestALCT) secondALCT.clear();
  if (secondCLCT == bestCLCT) secondCLCT.clear();

  const bool ok_bb = bestALCT.isValid() and bestCLCT.isValid();
  const bool ok_bs = bestALCT.isValid() and secondCLCT.isValid();
  const bool ok_sb = secondALCT.isValid() and bestCLCT.isValid();
  const bool ok_ss = secondALCT.isValid() and secondCLCT.isValid();

  const int ok11 = doesALCTCrossCLCT( bestALCT, bestCLCT, p);
  const int ok12 = doesALCTCrossCLCT( bestALCT, secondCLCT, p);
  const int ok21 = doesALCTCrossCLCT( secondALCT, bestCLCT, p);
  const int ok22 = doesALCTCrossCLCT( secondALCT, secondCLCT, p);
  const int code = (ok11<<3) | (ok12<<2) | (ok21<<1) | (ok22);

  int dbg=0;
  int ring = p;
  int chamb= CSCTriggerNumbering::chamberFromTriggerLabels(theSector,theSubsector, theStation, theTrigChamber);
  CSCDetId did(theEndcap, theStation, ring, chamb, 0);
  if (dbg) LogTrace("CSCGEMMotherboardME11")<<"debug correlateLCTs in "<<did<< "\n"
	   <<"ALCT1: "<<bestALCT<<"\n"
	   <<"ALCT2: "<<secondALCT<<"\n"
	   <<"CLCT1: "<<bestCLCT<<"\n"
	   <<"CLCT2: "<<secondCLCT<<"\n"
	   <<"ok 11 12 21 22 code = "<<ok11<<" "<<ok12<<" "<<ok21<<" "<<ok22<<" "<<code<<std::endl;

  if ( code==0 ) return;

  // LUT defines correspondence between possible ok## combinations
  // and resulting lct1 and lct2
  int lut[16][2] = {
          //ok: 11 12 21 22
    {0 ,0 }, // 0  0  0  0
    {22,0 }, // 0  0  0  1
    {21,0 }, // 0  0  1  0
    {21,22}, // 0  0  1  1
    {12,0 }, // 0  1  0  0
    {12,22}, // 0  1  0  1
    {12,21}, // 0  1  1  0
    {12,21}, // 0  1  1  1
    {11,0 }, // 1  0  0  0
    {11,22}, // 1  0  0  1
    {11,21}, // 1  0  1  0
    {11,22}, // 1  0  1  1
    {11,12}, // 1  1  0  0
    {11,22}, // 1  1  0  1
    {11,12}, // 1  1  1  0
    {11,22}, // 1  1  1  1
  };

  if (dbg) LogTrace("CSCGEMMotherboardME11")<<"lut 0 1 = "<<lut[code][0]<<" "<<lut[code][1]<<std::endl;

  // check matching copads
  const GEMCoPadDigi& bb_copad = bestMatchingPad<GEMCoPadDigi>(bestALCT,   bestCLCT,   copads, p);
  const GEMCoPadDigi& bs_copad = bestMatchingPad<GEMCoPadDigi>(bestALCT,   secondCLCT, copads, p);
  const GEMCoPadDigi& sb_copad = bestMatchingPad<GEMCoPadDigi>(secondALCT, bestCLCT,   copads, p);
  const GEMCoPadDigi& ss_copad = bestMatchingPad<GEMCoPadDigi>(secondALCT, secondCLCT, copads, p);

  // check matching pads
  const GEMPadDigi& bb_pad = bestMatchingPad<GEMPadDigi>(bestALCT,   bestCLCT,   pads, p);
  const GEMPadDigi& bs_pad = bestMatchingPad<GEMPadDigi>(bestALCT,   secondCLCT, pads, p);
  const GEMPadDigi& sb_pad = bestMatchingPad<GEMPadDigi>(secondALCT, bestCLCT,   pads, p);
  const GEMPadDigi& ss_pad = bestMatchingPad<GEMPadDigi>(secondALCT, secondCLCT, pads, p);

  // evaluate possible combinations
  const bool ok_bb_copad = ok11==1 and ok_bb and bb_copad.isValid();
  const bool ok_bs_copad = ok12==1 and ok_bs and bs_copad.isValid();
  const bool ok_sb_copad = ok21==1 and ok_sb and sb_copad.isValid();
  const bool ok_ss_copad = ok22==1 and ok_ss and ss_copad.isValid();

  const bool ok_bb_pad = (not ok_bb_copad) and ok11==1 and ok_bb and bb_pad.isValid();
  const bool ok_bs_pad = (not ok_bs_copad) and ok12==1 and ok_bs and bs_pad.isValid();
  const bool ok_sb_pad = (not ok_sb_copad) and ok21==1 and ok_sb and sb_pad.isValid();
  const bool ok_ss_pad = (not ok_ss_copad) and ok22==1 and ok_ss and ss_pad.isValid();

  switch (lut[code][0]) {
  case 11:
    if (ok_bb_copad) lct1 = CSCGEMMotherboard::constructLCTsGEM(bestALCT, bestCLCT, bb_copad, p, 1);
    if (ok_bb_pad)   lct1 = CSCGEMMotherboard::constructLCTsGEM(bestALCT, bestCLCT, bb_pad, p, 1);
    break;
  case 12:
    if (ok_bs_copad) lct1 = CSCGEMMotherboard::constructLCTsGEM(bestALCT, secondCLCT, bs_copad, p, 1);
    if (ok_bs_pad)   lct1 = CSCGEMMotherboard::constructLCTsGEM(bestALCT, secondCLCT, bs_pad, p, 1);
    break;
  case 21:
    if (ok_sb_copad) lct1 = CSCGEMMotherboard::constructLCTsGEM(secondALCT, bestCLCT, sb_copad, p, 1);
    if (ok_sb_pad)   lct1 = CSCGEMMotherboard::constructLCTsGEM(secondALCT, bestCLCT, sb_pad, p, 1);
    break;
  case 22:
    if (ok_ss_copad) lct1 = CSCGEMMotherboard::constructLCTsGEM(secondALCT, secondCLCT, ss_copad, p, 1);
    if (ok_ss_pad)   lct1 = CSCGEMMotherboard::constructLCTsGEM(secondALCT, secondCLCT, ss_pad, p, 1);
    break;
  default:
    return;
  }

  if (dbg) LogTrace("CSCGEMMotherboardME11")<<"lct1: "<<lct1<<std::endl;

  switch (lut[code][1]){
  case 12:
    if (ok_bs_copad) lct2 = CSCGEMMotherboard::constructLCTsGEM(bestALCT, secondCLCT, bs_copad, p, 2);
    if (ok_bs_pad)   lct2 = CSCGEMMotherboard::constructLCTsGEM(bestALCT, secondCLCT, bs_pad, p, 2);
    break;
  case 21:
    if (ok_sb_copad) lct2 = CSCGEMMotherboard::constructLCTsGEM(secondALCT, bestCLCT, sb_copad, p, 2);
    if (ok_sb_pad)   lct2 = CSCGEMMotherboard::constructLCTsGEM(secondALCT, bestCLCT, sb_pad, p, 2);
    break;
  case 22:
    if (ok_bb_copad) lct2 = CSCGEMMotherboard::constructLCTsGEM(secondALCT, secondCLCT, bb_copad, p, 2);
    if (ok_bb_pad)   lct2 = CSCGEMMotherboard::constructLCTsGEM(secondALCT, secondCLCT, bb_pad, p, 2);
    break;
  default:
    return;
  }

  if (dbg) LogTrace("CSCGEMMotherboardME11")<<"lct2: "<<lct2<<std::endl;

  if (dbg) LogTrace("CSCGEMMotherboardME11")<<"out of correlateLCTs"<<std::endl;
  return;
}
