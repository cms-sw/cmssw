#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"
#include "L1Trigger/CSCTriggerPrimitives/src/CSCGEMMotherboardME21.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"

CSCGEMMotherboardME21::CSCGEMMotherboardME21(unsigned endcap, unsigned station,
					     unsigned sector, unsigned subsector,
					     unsigned chamber,
					     const edm::ParameterSet& conf)
  : CSCGEMMotherboard(endcap, station, sector, subsector, chamber, conf)
  , allLCTs(match_trig_window_size)
  , dropLowQualityCLCTsNoGEMs_(tmbParams_.getParameter<bool>("dropLowQualityCLCTsNoGEMs"))
  , dropLowQualityALCTsNoGEMs_(tmbParams_.getParameter<bool>("dropLowQualityALCTsNoGEMs"))
  , buildLCTfromALCTandGEM_(tmbParams_.getParameter<bool>("buildLCTfromALCTandGEM"))
  , buildLCTfromCLCTandGEM_(tmbParams_.getParameter<bool>("buildLCTfromCLCTandGEM"))
{
  // set LUTs
  tmbLUT_.reset(new CSCGEMMotherboardLUTME21());
}


CSCGEMMotherboardME21::CSCGEMMotherboardME21()
  : CSCGEMMotherboard()
  , allLCTs(match_trig_window_size)
{
}


CSCGEMMotherboardME21::~CSCGEMMotherboardME21()
{
}


void CSCGEMMotherboardME21::clear()
{
  CSCMotherboard::clear();
  CSCGEMMotherboard::clear();

  for (int bx = 0; bx < CSCConstants::MAX_LCT_TBINS; bx++) {
    for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++) {
      for (int i=0;i<CSCConstants::MAX_LCTS_PER_CSC;i++) {
	allLCTs(bx,mbx,i).clear();
      }
    }
  }
}

void
CSCGEMMotherboardME21::run(const CSCWireDigiCollection* wiredc,
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

  if (!( alct and clct))
  {
    if (infoV >= 0) edm::LogError("L1CSCTPEmulatorSetupError")
      << "+++ run() called for non-existing ALCT/CLCT processor! +++ \n";
    return;
  }

  alct->setCSCGeometry(csc_g);
  clct->setCSCGeometry(csc_g);

  alctV = alct->run(wiredc); // run anodeLCT
  clctV = clct->run(compdc); // run cathodeLCT

  // if there are no ALCTs and no CLCTs, it does not make sense to run this TMB
  if (alctV.empty() and clctV.empty()) return;

  int used_clct_mask[20];
  for (int c=0;c<20;++c) used_clct_mask[c]=0;

  // retrieve pads and copads in a certain BX window for this CSC

  retrieveGEMPads(gemPads, gemId);
  retrieveGEMCoPads();

  const bool hasPads(!pads_.empty());
  const bool hasCoPads(!coPads_.empty());

  // ALCT centric matching
  for (int bx_alct = 0; bx_alct < CSCConstants::MAX_ALCT_TBINS; bx_alct++)
  {
    if (alct->bestALCT[bx_alct].isValid())
    {
      const int bx_clct_start(bx_alct - match_trig_window_size/2);
      const int bx_clct_stop(bx_alct + match_trig_window_size/2);
      const int bx_copad_start(bx_alct - maxDeltaBXCoPad_);
      const int bx_copad_stop(bx_alct + maxDeltaBXCoPad_);

      if (debug_matching){
        LogTrace("CSCGEMCMotherboardME21") << "========================================================================" << std::endl;
        LogTrace("CSCGEMCMotherboardME21") << "ALCT-CLCT matching in ME2/1 chamber: " << cscChamber->id() << std::endl;
        LogTrace("CSCGEMCMotherboardME21") << "------------------------------------------------------------------------" << std::endl;
        LogTrace("CSCGEMCMotherboardME21") << "+++ Best ALCT Details: ";
        alct->bestALCT[bx_alct].print();
        LogTrace("CSCGEMCMotherboardME21") << "+++ Second ALCT Details: ";
        alct->secondALCT[bx_alct].print();

        printGEMTriggerPads(bx_clct_start, bx_clct_stop, CSCPart::ME21);
        printGEMTriggerCoPads(bx_clct_start, bx_clct_stop, CSCPart::ME21);

        LogTrace("CSCGEMCMotherboardME21") << "------------------------------------------------------------------------" << std::endl;
        LogTrace("CSCGEMCMotherboardME21") << "Attempt ALCT-CLCT matching in ME2/1 in bx range: [" << bx_clct_start << "," << bx_clct_stop << "]" << std::endl;
      }

      // ALCT-to-CLCT
      int nSuccessFulMatches = 0;
      for (int bx_clct = bx_clct_start; bx_clct <= bx_clct_stop; bx_clct++)
      {
        if (bx_clct < 0 or bx_clct >= CSCConstants::MAX_CLCT_TBINS) continue;
        if (drop_used_clcts and used_clct_mask[bx_clct]) continue;
        if (clct->bestCLCT[bx_clct].isValid())
        {
          // clct quality
          const int quality(clct->bestCLCT[bx_clct].getQuality());
          // low quality ALCT
          const bool lowQualityALCT(alct->bestALCT[bx_alct].getQuality() == 0);
          // low quality ALCT or CLCT
          const bool lowQuality(quality<4 or lowQualityALCT);
          if (debug_matching) LogTrace("CSCGEMCMotherboardME21") << "++Valid ME21 CLCT: " << clct->bestCLCT[bx_clct] << std::endl;

	  // pick the pad that corresponds
	  matches<GEMPadDigi> mPads;
	  matchingPads<GEMPadDigi>(clct->bestCLCT[bx_clct], clct->secondCLCT[bx_clct],
				   alct->bestALCT[bx_alct], alct->secondALCT[bx_alct], ME21, mPads);
	  matches<GEMCoPadDigi> mCoPads;
	  matchingPads<GEMCoPadDigi>(clct->bestCLCT[bx_clct], clct->secondCLCT[bx_clct],
				     alct->bestALCT[bx_alct], alct->secondALCT[bx_alct], ME21, mCoPads);

          if (dropLowQualityCLCTsNoGEMs_ and lowQuality and hasPads){
            int nFound(mPads.size());
            const bool clctInEdge(clct->bestCLCT[bx_clct].getKeyStrip() < 5 or clct->bestCLCT[bx_clct].getKeyStrip() > 155);
            if (clctInEdge){
              if (debug_matching) LogTrace("CSCGEMCMotherboardME21") << "\tInfo: low quality CLCT in CSC chamber edge, don't care about GEM pads" << std::endl;
            }
            else {
              if (nFound != 0){
                if (debug_matching) LogTrace("CSCGEMCMotherboardME21") << "\tInfo: low quality CLCT with " << nFound << " matching GEM trigger pads" << std::endl;
              }
              else {
                if (debug_matching) LogTrace("CSCGEMCMotherboardME21") << "\tWarning: low quality CLCT without matching GEM trigger pad" << std::endl;
                continue;
              }
            }
          }

          ++nSuccessFulMatches;

          int mbx = bx_clct-bx_clct_start;
	  correlateLCTsGEM(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct],
			   clct->bestCLCT[bx_clct], clct->secondCLCT[bx_clct],
			   mPads, mCoPads,
			   allLCTs(bx_alct,mbx,0), allLCTs(bx_alct,mbx,1), CSCPart::ME21);
          if (debug_matching) {
	    //	    if (infoV > 1) LogTrace("CSCMotherboard")
            LogTrace("CSCGEMCMotherboardME21") << "Successful ALCT-CLCT match in ME21: bx_alct = " << bx_alct
                      << "; match window: [" << bx_clct_start << "; " << bx_clct_stop
                      << "]; bx_clct = " << bx_clct << std::endl;
            LogTrace("CSCGEMCMotherboardME21") << "+++ Best CLCT Details: ";
            clct->bestCLCT[bx_clct].print();
            LogTrace("CSCGEMCMotherboardME21") << "+++ Second CLCT Details: ";
            clct->secondCLCT[bx_clct].print();
          }
          if (allLCTs(bx_alct,mbx,0).isValid()) {
            used_clct_mask[bx_clct] += 1;
            if (match_earliest_clct_only) break;
          }
        }
      }

      // ALCT-to-GEM matching
      int nSuccessFulGEMMatches = 0;
      if (nSuccessFulMatches==0 and buildLCTfromALCTandGEM_){
        if (debug_matching) LogTrace("CSCGEMCMotherboardME21") << "++No valid ALCT-CLCT matches in ME21" << std::endl;
        for (int bx_gem = bx_copad_start; bx_gem <= bx_copad_stop; bx_gem++) {
          if (not hasCoPads) {
            continue;
          }

          // find the best matching copad
	  matches<GEMCoPadDigi> copads;
	  matchingPads<CSCALCTDigi, GEMCoPadDigi>(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct], ME21, copads);

          if (debug_matching) LogTrace("CSCGEMCMotherboardME21") << "\t++Number of matching GEM CoPads in BX " << bx_alct << " : "<< copads.size() << std::endl;
          if (copads.empty()) {
            continue;
          }

	  CSCGEMMotherboard::correlateLCTsGEM(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct],
					      copads, allLCTs(bx_alct,0,0), allLCTs(bx_alct,0,1), CSCPart::ME21);
          if (allLCTs(bx_alct,0,0).isValid()) {
            ++nSuccessFulGEMMatches;
            if (match_earliest_clct_only) break;
          }
          if (debug_matching) {
            LogTrace("CSCGEMCMotherboardME21") << "Successful ALCT-GEM CoPad match in ME21: bx_alct = " << bx_alct << std::endl << std::endl;
            LogTrace("CSCGEMCMotherboardME21") << "------------------------------------------------------------------------" << std::endl << std::endl;
          }
        }
      }

      if (debug_matching) {
        LogTrace("CSCGEMCMotherboardME21") << "========================================================================" << std::endl;
        LogTrace("CSCGEMCMotherboardME21") << "Summary: " << std::endl;
        if (nSuccessFulMatches>1)
          LogTrace("CSCGEMCMotherboardME21") << "Too many successful ALCT-CLCT matches in ME21: " << nSuccessFulMatches
                    << ", CSCDetId " << cscChamber->id()
                    << ", bx_alct = " << bx_alct
                    << "; match window: [" << bx_clct_start << "; " << bx_clct_stop << "]" << std::endl;
        else if (nSuccessFulMatches==1)
          LogTrace("CSCGEMCMotherboardME21") << "1 successful ALCT-CLCT match in ME21: "
                    << " CSCDetId " << cscChamber->id()
                    << ", bx_alct = " << bx_alct
                    << "; match window: [" << bx_clct_start << "; " << bx_clct_stop << "]" << std::endl;
        else if (nSuccessFulGEMMatches==1)
          LogTrace("CSCGEMCMotherboardME21") << "1 successful ALCT-GEM match in ME21: "
                    << " CSCDetId " << cscChamber->id()
                    << ", bx_alct = " << bx_alct
                    << "; match window: [" << bx_clct_start << "; " << bx_clct_stop << "]" << std::endl;
        else
          LogTrace("CSCGEMCMotherboardME21") << "Unsuccessful ALCT-CLCT match in ME21: "
                    << "CSCDetId " << cscChamber->id()
                    << ", bx_alct = " << bx_alct
                    << "; match window: [" << bx_clct_start << "; " << bx_clct_stop << "]" << std::endl;
      }
    }
    // at this point we have invalid ALCTs --> try GEM pad matching
    else{
      auto coPads(coPads_[bx_alct]);
      if (!coPads.empty() and buildLCTfromCLCTandGEM_) {
        const int bx_clct_start(bx_alct - match_trig_window_size/2);
        const int bx_clct_stop(bx_alct + match_trig_window_size/2);

        if (debug_matching){
          LogTrace("CSCGEMCMotherboardME21") << "========================================================================" << std::endl;
          LogTrace("CSCGEMCMotherboardME21") <<"GEM-CLCT matching in ME2/1 chamber: "<< cscChamber->id()<< "in bx:"<<bx_alct<<std::endl;
          LogTrace("CSCGEMCMotherboardME21") << "------------------------------------------------------------------------" << std::endl;
        }
        // GEM-to-CLCT
        int nSuccessFulMatches = 0;
        for (int bx_clct = bx_clct_start; bx_clct <= bx_clct_stop; bx_clct++)
          {
            if (bx_clct < 0 or bx_clct >= CSCConstants::MAX_CLCT_TBINS) continue;
            if (drop_used_clcts and used_clct_mask[bx_clct]) continue;
            if (clct->bestCLCT[bx_clct].isValid())
              {
                const int quality(clct->bestCLCT[bx_clct].getQuality());
                // only use high-Q stubs for the time being
                if (quality < 4) continue;

	     ++nSuccessFulMatches;

	     int mbx = std::abs(clct->bestCLCT[bx_clct].getBX()-bx_alct);
	     int bx_gem = (coPads[0].second).bx(1)+CSCConstants::LCT_CENTRAL_BX;
	     CSCGEMMotherboard::correlateLCTsGEM(clct->bestCLCT[bx_clct], clct->secondCLCT[bx_clct], coPads,
						 allLCTs(bx_gem,mbx,0), allLCTs(bx_gem,mbx,1), CSCPart::ME21);
	     if (debug_matching) {
	       //	    if (infoV > 1) LogTrace("CSCGEMMotherboardME21")
	       LogTrace("CSCGEMCMotherboardME21") << "Successful GEM-CLCT match in ME21: bx_alct = " << bx_alct <<std::endl;
	       //<< "; match window: [" << bx_clct_start << "; " << bx_clct_stop
	       //<< "]; bx_clct = " << bx_clct << std::endl;
	       LogTrace("CSCGEMCMotherboardME21") << "+++ Best CLCT Details: ";
	       clct->bestCLCT[bx_clct].print();
	       LogTrace("CSCGEMCMotherboardME21") << "+++ Second CLCT Details: ";
	       clct->secondCLCT[bx_clct].print();
	     }
	     if (allLCTs(bx_gem,mbx,0).isValid()) {
	       used_clct_mask[bx_gem] += 1;
	       if (match_earliest_clct_only) break;
	     }
	   }
	  }
      }
    }
  }
  // reduction of nLCTs per each BX
  for (int bx = 0; bx < CSCConstants::MAX_LCT_TBINS; bx++)
    {
      // counting
      unsigned int n=0;
      for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++)
	for (int i=0;i<CSCConstants::MAX_LCTS_PER_CSC;i++)
	  {
	    int cbx = bx + mbx - match_trig_window_size/2;
	    if (allLCTs(bx,mbx,i).isValid())
	      {
		++n;
		if (infoV > 0) LogDebug("CSCGEMMotherboardME21")
				 << "LCT"<<i+1<<" "<<bx<<"/"<<cbx<<": "<<allLCTs(bx,mbx,i)<<std::endl;
	      }
	  }

      // some simple cross-bx sorting algorithms
      if (tmb_cross_bx_algo == 1 and (n>2))
	{
	  n=0;
	  for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++)
	    for (int i=0;i<CSCConstants::MAX_LCTS_PER_CSC;i++)
	      {
		if (allLCTs(bx,pref[mbx],i).isValid())
		  {
		    n++;
		    if (n>2) allLCTs(bx,pref[mbx],i).clear();
		  }
	      }

	  n=0;
	  for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++)
	    for (int i=0;i<CSCConstants::MAX_LCTS_PER_CSC;i++)
	      {
		int cbx = bx + mbx - match_trig_window_size/2;
		if (allLCTs(bx,mbx,i).isValid())
		  {
		    n++;
		    if (infoV > 0) LogDebug("CSCGEMMotherboardME21")
				     << "LCT"<<i+1<<" "<<bx<<"/"<<cbx<<": "<<allLCTs(bx,mbx,i)<< std::endl;
		  }
	      }
	  if (infoV > 0 and n>0) LogDebug("CSCGEMMotherboardME21")
				   <<"bx "<<bx<<" nnLCT:"<<n<<" "<<n<<std::endl;
	} // x-bx sorting
    }

  bool first = true;
  unsigned int n=0;
  for (const auto& p : readoutLCTs()) {
    if (debug_matching and first){
      LogTrace("CSCGEMCMotherboardME21") << "========================================================================" << std::endl;
      LogTrace("CSCGEMCMotherboardME21") << "Counting the final LCTs" << std::endl;
      LogTrace("CSCGEMCMotherboardME21") << "========================================================================" << std::endl;
      first = false;
      LogTrace("CSCGEMCMotherboardME21") << "tmb_cross_bx_algo: " << tmb_cross_bx_algo << std::endl;
    }
    n++;
    if (debug_matching)
      LogTrace("CSCGEMCMotherboardME21") << "LCT "<<n<<"  " << p <<std::endl;
  }
}

//readout LCTs
std::vector<CSCCorrelatedLCTDigi> CSCGEMMotherboardME21::readoutLCTs() const
{
  std::vector<CSCCorrelatedLCTDigi> result;
  allLCTs.getMatched(result);
  if (tmb_cross_bx_algo == 2) CSCUpgradeMotherboard::sortLCTs(result, CSCUpgradeMotherboard::sortLCTsByQuality);
  if (tmb_cross_bx_algo == 3) CSCUpgradeMotherboard::sortLCTs(result, CSCUpgradeMotherboard::sortLCTsByGEMDphi);
  return result;
}


void CSCGEMMotherboardME21::correlateLCTsGEM(const CSCALCTDigi& bALCT,
                                             const CSCALCTDigi& sALCT,
                                             const CSCCLCTDigi& bCLCT,
                                             const CSCCLCTDigi& sCLCT,
                                             const GEMPadDigiIds& pads,
                                             const GEMCoPadDigiIds& copads,
                                             CSCCorrelatedLCTDigi& lct1,
                                             CSCCorrelatedLCTDigi& lct2, enum CSCPart p) const
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

  if (!copads.empty() or !pads.empty()){

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
    const bool ok_bb_copad = ok_bb and bb_copad.isValid();
    const bool ok_bs_copad = ok_bs and bs_copad.isValid();
    const bool ok_sb_copad = ok_sb and sb_copad.isValid();
    const bool ok_ss_copad = ok_ss and ss_copad.isValid();

    const bool ok_bb_pad = (not ok_bb_copad) and ok_bb and bb_pad.isValid();
    const bool ok_bs_pad = (not ok_bs_copad) and ok_bs and bs_pad.isValid();
    const bool ok_sb_pad = (not ok_sb_copad) and ok_sb and sb_pad.isValid();
    const bool ok_ss_pad = (not ok_ss_copad) and ok_ss and ss_pad.isValid();

    // possible cases with copad
    if (ok_bb_copad or ok_ss_copad){
      if (ok_bb_copad) lct1 = constructLCTsGEM(bestALCT, bestCLCT, bb_copad, p, 1);
      if (ok_ss_copad) lct2 = constructLCTsGEM(secondALCT, secondCLCT, ss_copad, p, 2);
    }
    else if(ok_bs_copad or ok_sb_copad){
      if (ok_bs_copad) lct1 = constructLCTsGEM(bestALCT, secondCLCT, bs_copad, p, 1);
      if (ok_sb_copad) lct2 = constructLCTsGEM(secondALCT, bestCLCT, sb_copad, p, 2);
    }

    // done processing?
    if (lct1.isValid() and lct2.isValid()) return;

    // possible cases with pad
    if ((ok_bb_pad or ok_ss_pad) and not (ok_bs_copad or ok_sb_copad)){
      if (ok_bb_pad) lct1 = constructLCTsGEM(bestALCT, bestCLCT, bb_pad, p, 1);
      if (ok_ss_pad) lct2 = constructLCTsGEM(secondALCT, secondCLCT, ss_pad, p, 2);
    }
    else if((ok_bs_pad or ok_sb_pad) and not (ok_bb_copad or ok_ss_copad)){
      if (ok_bs_pad) lct1 = constructLCTsGEM(bestALCT, secondCLCT, bs_pad, p, 1);
      if (ok_sb_pad) lct2 = constructLCTsGEM(secondALCT, bestCLCT, sb_pad, p, 2);
    }
  } else {
    // run without gems - happens in less than 0.04% of the time
    if (ok_bb) lct1 = constructLCTs(bestALCT, bestCLCT, CSCCorrelatedLCTDigi::ALCTCLCT, 1);
    if (ok_ss) lct2 = constructLCTs(secondALCT, secondCLCT, CSCCorrelatedLCTDigi::ALCTCLCT, 2);
  }
}
