#include <memory>

#include "L1Trigger/CSCTriggerPrimitives/interface/CSCGEMMotherboard.h"
CSCGEMMotherboard::CSCGEMMotherboard(unsigned endcap,
                                     unsigned station,
                                     unsigned sector,
                                     unsigned subsector,
                                     unsigned chamber,
                                     const edm::ParameterSet& conf)
    : CSCMotherboard(endcap, station, sector, subsector, chamber, conf),
      drop_low_quality_alct_no_gems_(tmbParams_.getParameter<bool>("dropLowQualityALCTsNoGEMs")),
      drop_low_quality_clct_no_gems_(tmbParams_.getParameter<bool>("dropLowQualityCLCTsNoGEMs")),
      build_lct_from_alct_gem_(tmbParams_.getParameter<bool>("buildLCTfromALCTandGEM")),
      build_lct_from_clct_gem_(tmbParams_.getParameter<bool>("buildLCTfromCLCTandGEM")) {
  // case for ME1/1
  if (isME11_) {
    drop_low_quality_clct_no_gems_me1a_ = tmbParams_.getParameter<bool>("dropLowQualityCLCTsNoGEMs_ME1a");
    build_lct_from_clct_gem_me1a_ = tmbParams_.getParameter<bool>("buildLCTfromCLCTandGEM_ME1a");
  }

  max_delta_bx_alct_gem_ = tmbParams_.getParameter<unsigned>("maxDeltaBXALCTGEM");
  max_delta_bx_clct_gem_ = tmbParams_.getParameter<unsigned>("maxDeltaBXCLCTGEM");

  assign_gem_csc_bending_ = tmbParams_.getParameter<bool>("assignGEMCSCBending");
  qualityAssignment_->setGEMCSCBending(assign_gem_csc_bending_);

  drop_used_gems_ = tmbParams_.getParameter<bool>("tmbDropUsedGems");
  match_earliest_gem_only_ = tmbParams_.getParameter<bool>("matchEarliestGemsOnly");

  // These LogErrors are sanity checks and should not be printed
  if (isME11_ and !runME11ILT_) {
    edm::LogError("CSCGEMMotherboard") << "TMB constructed while runME11ILT_ is not set!";
  };

  if (isME21_ and !runME21ILT_) {
    edm::LogError("CSCGEMMotherboard") << "TMB constructed while runME21ILT_ is not set!";
  };

  // super chamber has layer=0!
  gemId = GEMDetId(theRegion, 1, theStation, 0, theChamber, 0).rawId();

  clusterProc_ = std::make_shared<GEMClusterProcessor>(theRegion, theStation, theChamber, conf);

  cscGEMMatcher_ = std::make_unique<CSCGEMMatcher>(theRegion, theStation, theChamber, tmbParams_, conf);
}

CSCGEMMotherboard::~CSCGEMMotherboard() {}

void CSCGEMMotherboard::clear() {
  CSCMotherboard::clear();
  clusterProc_->clear();
}

void CSCGEMMotherboard::run(const CSCWireDigiCollection* wiredc,
                            const CSCComparatorDigiCollection* compdc,
                            const GEMPadDigiClusterCollection* gemClusters) {
  // Step 1: Setup
  clear();

  // check for GEM geometry
  if (gem_g == nullptr) {
    edm::LogError("CSCGEMMotherboard") << "run() called for GEM-CSC integrated trigger without valid GEM geometry! \n";
    return;
  }

  // Check that the processors can deliver data
  if (!(alctProc and clctProc)) {
    edm::LogError("CSCGEMMotherboard") << "run() called for non-existing ALCT/CLCT processor! \n";
    return;
  }

  alctProc->setCSCGeometry(cscGeometry_);
  clctProc->setCSCGeometry(cscGeometry_);

  // Step 2: Run the processors
  const std::vector<CSCALCTDigi>& alctV = alctProc->run(wiredc);  // run anodeLCT
  const std::vector<CSCCLCTDigi>& clctV = clctProc->run(compdc);  // run cathodeLCT

  // Step 2b: encode high multiplicity bits (independent of LCT construction)
  encodeHighMultiplicityBits();

  // if there are no ALCTs and no CLCTs, do not run the ALCT-CLCT correlation
  if (alctV.empty() and clctV.empty())
    return;

  // Step 3: run the GEM cluster processor to get the internal clusters
  clusterProc_->run(gemClusters);
  hasGE21Geometry16Partitions_ = clusterProc_->hasGE21Geometry16Partitions();

  /*
    Mask for bunch crossings were LCTs were previously found
    If LCTs were found in BXi for ALCT-CLCT-2GEM, ALCT-CLCT-1GEM
    or ALCT-CLCT matches, we do not consider BXi in the future. This is
    because we consider  ALCT-CLCT-2GEM, ALCT-CLCT-1GEM, ALCT-CLCT of
    higher quality than CLCT-2GEM and ALCT-2GEM LCTs. The mask is passsed
    from one function to the next.
  */
  bool bunch_crossing_mask[CSCConstants::MAX_ALCT_TBINS] = {false};

  // Step 4: ALCT-centric matching
  matchALCTCLCTGEM(bunch_crossing_mask);

  // Step 5: CLCT-2GEM matching for BX's that were not previously masked
  if (build_lct_from_clct_gem_) {
    matchCLCT2GEM(bunch_crossing_mask);
  }

  // Step 6: ALCT-2GEM matching for BX's that were not previously masked
  if (build_lct_from_alct_gem_) {
    matchALCT2GEM(bunch_crossing_mask);
  }

  // Step 7: Select at most 2 LCTs per BX
  selectLCTs();
}

void CSCGEMMotherboard::matchALCTCLCTGEM(bool bunch_crossing_mask[CSCConstants::MAX_ALCT_TBINS]) {
  // array to mask CLCTs
  bool used_clct_mask[CSCConstants::MAX_CLCT_TBINS] = {false};

  for (int bx_alct = 0; bx_alct < CSCConstants::MAX_ALCT_TBINS; bx_alct++) {
    // do not consider invalid ALCTs
    if (alctProc->getBestALCT(bx_alct).isValid()) {
      for (unsigned mbx = 0; mbx < match_trig_window_size; mbx++) {
        // evaluate the preffered CLCT BX, taking into account that there is an offset in the simulation
        unsigned bx_clct = bx_alct + preferred_bx_match_[mbx] - CSCConstants::ALCT_CLCT_OFFSET;

        // CLCT BX must be in the time window
        if (bx_clct >= CSCConstants::MAX_CLCT_TBINS)
          continue;
        // drop this CLCT if it was previously matched to an ALCT
        if (drop_used_clcts and used_clct_mask[bx_clct])
          continue;
        // do not consider invalid CLCTs
        if (clctProc->getBestCLCT(bx_clct).isValid()) {
          LogTrace("CSCMotherboard") << "Successful ALCT-CLCT match: bx_alct = " << bx_alct << "; bx_clct = " << bx_clct
                                     << "; mbx = " << mbx;

          // now correlate the ALCT and CLCT into LCT.
          // smaller mbx means more preferred!
          correlateLCTsGEM(alctProc->getBestALCT(bx_alct),
                           alctProc->getSecondALCT(bx_alct),
                           clctProc->getBestCLCT(bx_clct),
                           clctProc->getSecondCLCT(bx_clct),
                           clusterProc_->getClusters(bx_alct, max_delta_bx_alct_gem_),
                           allLCTs_(bx_alct, mbx, 0),
                           allLCTs_(bx_alct, mbx, 1));

          if (allLCTs_(bx_alct, mbx, 0).isValid()) {
            // mask this CLCT as used. If a flag is set, the CLCT may or may not be reused
            used_clct_mask[bx_clct] = true;
            // mask this bunch crossing for future considation
            bunch_crossing_mask[bx_alct] = true;
            // if we only consider the first valid CLCT, we move on to the next ALCT immediately
            if (match_earliest_clct_only_)
              break;
          }
        }
      }
    }
  }
}

void CSCGEMMotherboard::matchCLCT2GEM(bool bunch_crossing_mask[CSCConstants::MAX_ALCT_TBINS]) {
  // no matching is done for GE2/1 geometries with 8 eta partitions
  if (isME21_ and !hasGE21Geometry16Partitions_)
    return;

  // array to mask CLCTs
  bool used_clct_mask[CSCConstants::MAX_CLCT_TBINS] = {false};

  for (int bx_gem = 0; bx_gem < CSCConstants::MAX_ALCT_TBINS; bx_gem++) {
    // do not consider LCT building in this BX if the mask was set
    if (bunch_crossing_mask[bx_gem])
      continue;

    // Check that there is at least one valid GEM coincidence cluster in this BX
    if (!clusterProc_->getCoincidenceClusters(bx_gem).empty()) {
      // GEM clusters will have central BX 8. So do ALCTs. But! CLCTs still have central BX 7
      // therefore we need to make a correction. The correction is thus the same as for ALCT-CLCT
      for (unsigned mbx = 0; mbx < 2 * max_delta_bx_clct_gem_ + 1; mbx++) {
        // evaluate the preffered CLCT BX, taking into account that there is an offset in the simulation
        int bx_clct = bx_gem + preferred_bx_match_[mbx] - CSCConstants::ALCT_CLCT_OFFSET;

        // CLCT BX must be in the time window
        if (bx_clct < 0 or bx_clct >= CSCConstants::MAX_CLCT_TBINS)
          continue;
        // drop this CLCT if it was previously matched to a GEM coincidence cluster
        if (drop_used_clcts and used_clct_mask[bx_clct])
          continue;
        // do not consider invalid CLCTs
        if (clctProc->getBestCLCT(bx_clct).isValid()) {
          // if this is an ME1/a CLCT, you could consider not building this LCT
          if (!build_lct_from_clct_gem_me1a_ and isME11_ and
              clctProc->getBestCLCT(bx_clct).getKeyStrip() > CSCConstants::MAX_HALF_STRIP_ME1B)
            continue;
          // mbx is a relative time difference, which is used later in the
          // cross-bunching sorting algorithm to determine the preferred LCTs
          // to be sent to the MPC
          correlateLCTsGEM(clctProc->getBestCLCT(bx_clct),
                           clctProc->getSecondCLCT(bx_clct),
                           clusterProc_->getCoincidenceClusters(bx_gem),
                           allLCTs_(bx_gem, mbx, 0),
                           allLCTs_(bx_gem, mbx, 1));

          if (allLCTs_(bx_gem, mbx, 0).isValid()) {
            // do not consider GEM clusters
            used_clct_mask[bx_clct] = true;
            // mask this bunch crossing for future consideration
            bunch_crossing_mask[bx_gem] = true;
            // if we only consider the first valid GEM coincidence clusters,
            // we move on to the next ALCT immediately
            if (match_earliest_clct_only_)
              break;
          }
        }
      }
    }
  }
}

void CSCGEMMotherboard::matchALCT2GEM(bool bunch_crossing_mask[CSCConstants::MAX_ALCT_TBINS]) {
  // no matching is done for GE2/1 geometries with 8 eta partitions
  if (isME21_ and !hasGE21Geometry16Partitions_)
    return;

  // clear the array to mask GEMs  this window is quite wide.
  // We don't expect GEM coincidence clusters to show up too far
  // from the central BX (8)
  bool used_gem_mask[CSCConstants::MAX_ALCT_TBINS] = {false};

  for (int bx_alct = 0; bx_alct < CSCConstants::MAX_ALCT_TBINS; bx_alct++) {
    // do not consider LCT building in this BX if the mask was set
    if (bunch_crossing_mask[bx_alct])
      continue;

    if (alctProc->getBestALCT(bx_alct).isValid()) {
      for (unsigned mbx = 0; mbx < 2 * max_delta_bx_alct_gem_ + 1; mbx++) {
        // evaluate the preffered GEM BX
        int bx_gem = bx_alct + preferred_bx_match_[mbx];

        if (bx_gem < 0 or bx_gem >= CSCConstants::MAX_ALCT_TBINS)
          continue;
        // drop GEMs in this BX if one of them was previously matched to an ALCT
        if (drop_used_gems_ and used_gem_mask[bx_gem])
          continue;
        // check for at least one valid GEM cluster
        if (!clusterProc_->getCoincidenceClusters(bx_gem).empty()) {
          // now correlate the ALCT and GEM into LCT.
          // smaller mbx means more preferred!
          correlateLCTsGEM(alctProc->getBestALCT(bx_alct),
                           alctProc->getSecondALCT(bx_alct),
                           clusterProc_->getCoincidenceClusters(bx_gem),
                           allLCTs_(bx_alct, mbx, 0),
                           allLCTs_(bx_alct, mbx, 1));

          if (allLCTs_(bx_alct, mbx, 0).isValid()) {
            // do not consider GEM clusters
            used_gem_mask[bx_gem] = true;
            // mask this bunch crossing for future consideration
            bunch_crossing_mask[bx_alct] = true;
            // if we only consider the first valid GEM coincidence clusters,
            // we move on to the next ALCT immediately
            if (match_earliest_gem_only_)
              break;
          }
        }
      }
    }
  }
}

void CSCGEMMotherboard::correlateLCTsGEM(const CSCALCTDigi& bALCT,
                                         const CSCALCTDigi& sALCT,
                                         const CSCCLCTDigi& bCLCT,
                                         const CSCCLCTDigi& sCLCT,
                                         const GEMInternalClusters& clusters,
                                         CSCCorrelatedLCTDigi& lct1,
                                         CSCCorrelatedLCTDigi& lct2) const {
  if (isME21_ and !hasGE21Geometry16Partitions_) {
    // This is an 8-eta partition GE2/1 geometry for which the GE2/1-ME2/1 integrated
    // local trigger is not configured. Matching only ALCTs with CLCTs in ME2/1.

    // do regular ALCT-CLCT correlation
    CSCMotherboard::correlateLCTs(bALCT, sALCT, bCLCT, sCLCT, lct1, lct2, CSCCorrelatedLCTDigi::ALCTCLCT);
  } else {
    // temporary container
    std::vector<CSCCorrelatedLCTDigi> lcts;

    CSCALCTDigi bestALCT = bALCT;
    CSCALCTDigi secondALCT = sALCT;
    CSCCLCTDigi bestCLCT = bCLCT;
    CSCCLCTDigi secondCLCT = sCLCT;

    if (!bestALCT.isValid()) {
      edm::LogError("CSCGEMMotherboard") << "Best ALCT invalid in correlateLCTsGEM!";
      return;
    }

    if (!bestCLCT.isValid()) {
      edm::LogError("CSCGEMMotherboard") << "Best CLCT invalid in correlateLCTsGEM!";
      return;
    }

    // case where there no valid clusters
    if (clusters.empty()) {
      // drop the low-quality ALCTs and CLCTs
      dropLowQualityALCTNoClusters(bestALCT, GEMInternalCluster());
      dropLowQualityALCTNoClusters(secondALCT, GEMInternalCluster());
      dropLowQualityCLCTNoClusters(bestCLCT, GEMInternalCluster());
      dropLowQualityCLCTNoClusters(secondCLCT, GEMInternalCluster());

      // do regular ALCT-CLCT correlation
      CSCMotherboard::correlateLCTs(bALCT, sALCT, bCLCT, sCLCT, lct1, lct2, CSCCorrelatedLCTDigi::ALCTCLCT);
    }
    // case with at least one valid cluster
    else {
      // before matching ALCT-CLCT pairs with clusters, we check if we need
      // to drop particular low quality ALCTs or CLCTs without matching clusters
      // drop low quality CLCTs if no clusters and flags are set
      GEMInternalCluster bestALCTCluster, secondALCTCluster;
      GEMInternalCluster bestCLCTCluster, secondCLCTCluster;
      cscGEMMatcher_->bestClusterBXLoc(bestALCT, clusters, bestALCTCluster);
      cscGEMMatcher_->bestClusterBXLoc(secondALCT, clusters, secondALCTCluster);
      cscGEMMatcher_->bestClusterBXLoc(bestCLCT, clusters, bestCLCTCluster);
      cscGEMMatcher_->bestClusterBXLoc(secondCLCT, clusters, secondCLCTCluster);

      dropLowQualityALCTNoClusters(bestALCT, bestALCTCluster);
      dropLowQualityALCTNoClusters(secondALCT, secondALCTCluster);
      dropLowQualityCLCTNoClusters(bestCLCT, bestCLCTCluster);
      dropLowQualityCLCTNoClusters(secondCLCT, secondCLCTCluster);

      // check which ALCTs and CLCTs are valid after dropping the low-quality ones
      copyValidToInValid(bestALCT, secondALCT, bestCLCT, secondCLCT);

      // We can now check possible triplets and construct all LCTs with
      // valid ALCT, valid CLCTs and coincidence clusters
      GEMInternalCluster bbCluster, bsCluster, sbCluster, ssCluster;
      cscGEMMatcher_->bestClusterBXLoc(bestALCT, bestCLCT, clusters, bbCluster);
      cscGEMMatcher_->bestClusterBXLoc(bestALCT, secondCLCT, clusters, bsCluster);
      cscGEMMatcher_->bestClusterBXLoc(secondALCT, bestCLCT, clusters, sbCluster);
      cscGEMMatcher_->bestClusterBXLoc(secondALCT, secondCLCT, clusters, ssCluster);

      // At this point it is still possible that certain pairs with high-quality
      // ALCTs and CLCTs do not have matching clusters. In that case we construct
      // a regular ALCT-CLCT type LCT. For instance, it could be that two muons went
      // through the chamber, produced 2 ALCTs, 2 CLCTs, but only 1 GEM cluster - because
      // GEM cluster efficiency is not 100% (closer to 95%). So we don't require
      // all clusters to be valid. If they are valid, the LCTs is constructed accordingly.
      // But we do require that the ALCT and CLCT are valid for each pair.
      CSCCorrelatedLCTDigi lctbb, lctbs, lctsb, lctss;
      if (bestALCT.isValid() and bestCLCT.isValid()) {
        constructLCTsGEM(bestALCT, bestCLCT, bbCluster, lctbb);
        lcts.push_back(lctbb);
      }
      if (bestALCT.isValid() and secondCLCT.isValid() and (secondCLCT != bestCLCT)) {
        constructLCTsGEM(bestALCT, secondCLCT, bsCluster, lctbs);
        lcts.push_back(lctbs);
      }
      if (bestCLCT.isValid() and secondALCT.isValid() and (secondALCT != bestALCT)) {
        constructLCTsGEM(secondALCT, bestCLCT, sbCluster, lctsb);
        lcts.push_back(lctsb);
      }
      if (secondALCT.isValid() and secondCLCT.isValid() and (secondALCT != bestALCT) and (secondCLCT != bestCLCT)) {
        constructLCTsGEM(secondALCT, secondCLCT, ssCluster, lctss);
        lcts.push_back(lctss);
      }

      // no LCTs
      if (lcts.empty())
        return;

      // sort by bending angle
      sortLCTsByBending(lcts);

      // retain best two
      lcts.resize(CSCConstants::MAX_LCTS_PER_CSC);

      // assign and set the track number
      if (lcts[0].isValid()) {
        lct1 = lcts[0];
        lct1.setTrknmb(1);
      }

      if (lcts[1].isValid()) {
        lct2 = lcts[1];
        lct2.setTrknmb(2);
      }
    }
  }
}

void CSCGEMMotherboard::correlateLCTsGEM(const CSCCLCTDigi& bCLCT,
                                         const CSCCLCTDigi& sCLCT,
                                         const GEMInternalClusters& clusters,
                                         CSCCorrelatedLCTDigi& lct1,
                                         CSCCorrelatedLCTDigi& lct2) const {
  CSCCLCTDigi bestCLCT = bCLCT;
  CSCCLCTDigi secondCLCT = sCLCT;

  if (!bestCLCT.isValid()) {
    edm::LogError("CSCGEMMotherboard") << "Best CLCT invalid in correlateLCTsGEM!";
    return;
  }

  // if the second best CLCT equals the best CLCT, clear it
  if (secondCLCT == bestCLCT)
    secondCLCT.clear();

  // get the best matching cluster
  GEMInternalCluster bestCluster;
  GEMInternalCluster secondCluster;
  cscGEMMatcher_->bestClusterBXLoc(bestCLCT, clusters, bestCluster);
  cscGEMMatcher_->bestClusterBXLoc(secondCLCT, clusters, secondCluster);

  // drop low quality CLCTs if no clusters and flags are set
  dropLowQualityCLCTNoClusters(bestCLCT, bestCluster);
  dropLowQualityCLCTNoClusters(secondCLCT, secondCluster);

  // construct all LCTs with valid CLCTs and coincidence clusters
  if (bestCLCT.isValid() and bestCluster.isCoincidence()) {
    constructLCTsGEM(bestCLCT, bestCluster, 1, lct1);
  }
  if (secondCLCT.isValid() and secondCluster.isCoincidence()) {
    constructLCTsGEM(secondCLCT, secondCluster, 2, lct2);
  }
}

void CSCGEMMotherboard::correlateLCTsGEM(const CSCALCTDigi& bALCT,
                                         const CSCALCTDigi& sALCT,
                                         const GEMInternalClusters& clusters,
                                         CSCCorrelatedLCTDigi& lct1,
                                         CSCCorrelatedLCTDigi& lct2) const {
  CSCALCTDigi bestALCT = bALCT;
  CSCALCTDigi secondALCT = sALCT;

  if (!bestALCT.isValid()) {
    edm::LogError("CSCGEMMotherboard") << "Best ALCT invalid in correlateLCTsGEM!";
    return;
  }

  // if the second best ALCT equals the best ALCT, clear it
  if (secondALCT == bestALCT)
    secondALCT.clear();

  // get the best matching cluster
  GEMInternalCluster bestCluster;
  GEMInternalCluster secondCluster;
  cscGEMMatcher_->bestClusterBXLoc(bestALCT, clusters, bestCluster);
  cscGEMMatcher_->bestClusterBXLoc(secondALCT, clusters, secondCluster);

  // drop low quality ALCTs if no clusters and flags are set
  dropLowQualityALCTNoClusters(bestALCT, bestCluster);
  dropLowQualityALCTNoClusters(secondALCT, secondCluster);

  // construct all LCTs with valid ALCTs and coincidence clusters
  if (bestALCT.isValid() and bestCluster.isCoincidence()) {
    constructLCTsGEM(bestALCT, bestCluster, 1, lct1);
  }
  if (secondALCT.isValid() and secondCluster.isCoincidence()) {
    constructLCTsGEM(secondALCT, secondCluster, 2, lct2);
  }
}

void CSCGEMMotherboard::constructLCTsGEM(const CSCALCTDigi& alct,
                                         const CSCCLCTDigi& clct,
                                         const GEMInternalCluster& gem,
                                         CSCCorrelatedLCTDigi& thisLCT) const {
  thisLCT.setValid(true);
  if (gem.isCoincidence()) {
    thisLCT.setType(CSCCorrelatedLCTDigi::ALCTCLCT2GEM);
  } else if (gem.isValid()) {
    thisLCT.setType(CSCCorrelatedLCTDigi::ALCTCLCTGEM);
  } else {
    thisLCT.setType(CSCCorrelatedLCTDigi::ALCTCLCT);
  }
  thisLCT.setQuality(qualityAssignment_->findQuality(alct, clct, gem));
  thisLCT.setALCT(getBXShiftedALCT(alct));
  thisLCT.setCLCT(getBXShiftedCLCT(clct));
  // set pads if there are any
  thisLCT.setGEM1(gem.mid1());
  thisLCT.setGEM2(gem.mid2());
  thisLCT.setPattern(encodePattern(clct.getPattern()));
  thisLCT.setMPCLink(0);
  thisLCT.setBX0(0);
  thisLCT.setSyncErr(0);
  thisLCT.setCSCID(theTrigChamber);
  // track number to be set later in final sorting stage
  thisLCT.setTrknmb(0);
  thisLCT.setWireGroup(alct.getKeyWG());
  thisLCT.setStrip(clct.getKeyStrip());
  thisLCT.setBend(clct.getBend());
  thisLCT.setBX(alct.getBX());
  if (runCCLUT_) {
    thisLCT.setRun3(true);
    if (assign_gem_csc_bending_)
      thisLCT.setSlope(cscGEMMatcher_->calculateGEMCSCBending(clct, gem));
    else
      thisLCT.setSlope(clct.getSlope());
    thisLCT.setQuartStripBit(clct.getQuartStripBit());
    thisLCT.setEighthStripBit(clct.getEighthStripBit());
    thisLCT.setRun3Pattern(clct.getRun3Pattern());
  }
}

/* Construct LCT from CSC and GEM information. Option CLCT-2GEM */
void CSCGEMMotherboard::constructLCTsGEM(const CSCCLCTDigi& clct,
                                         const GEMInternalCluster& gem,
                                         int trackNumber,
                                         CSCCorrelatedLCTDigi& thisLCT) const {
  thisLCT.setValid(true);
  thisLCT.setType(CSCCorrelatedLCTDigi::CLCT2GEM);
  thisLCT.setQuality(qualityAssignment_->findQuality(clct, gem));
  thisLCT.setCLCT(getBXShiftedCLCT(clct));
  thisLCT.setGEM1(gem.mid1());
  thisLCT.setGEM2(gem.mid2());
  thisLCT.setPattern(encodePattern(clct.getPattern()));
  thisLCT.setMPCLink(0);
  thisLCT.setBX0(0);
  thisLCT.setSyncErr(0);
  thisLCT.setCSCID(theTrigChamber);
  thisLCT.setTrknmb(trackNumber);
  thisLCT.setWireGroup(gem.getKeyWG());
  thisLCT.setStrip(clct.getKeyStrip());
  thisLCT.setBend(clct.getBend());
  thisLCT.setBX(gem.bx());
  if (runCCLUT_) {
    thisLCT.setRun3(true);
    if (assign_gem_csc_bending_)
      thisLCT.setSlope(cscGEMMatcher_->calculateGEMCSCBending(clct, gem));
    else
      thisLCT.setSlope(clct.getSlope());
    thisLCT.setQuartStripBit(clct.getQuartStripBit());
    thisLCT.setEighthStripBit(clct.getEighthStripBit());
    thisLCT.setRun3Pattern(clct.getRun3Pattern());
  }
}

/* Construct LCT from CSC and GEM information. Option ALCT-2GEM */
void CSCGEMMotherboard::constructLCTsGEM(const CSCALCTDigi& alct,
                                         const GEMInternalCluster& gem,
                                         int trackNumber,
                                         CSCCorrelatedLCTDigi& thisLCT) const {
  thisLCT.setValid(true);
  thisLCT.setType(CSCCorrelatedLCTDigi::ALCT2GEM);
  thisLCT.setQuality(qualityAssignment_->findQuality(alct, gem));
  thisLCT.setALCT(getBXShiftedALCT(alct));
  thisLCT.setGEM1(gem.mid1());
  thisLCT.setGEM2(gem.mid2());
  thisLCT.setPattern(10);
  thisLCT.setMPCLink(0);
  thisLCT.setBX0(0);
  thisLCT.setSyncErr(0);
  thisLCT.setCSCID(theTrigChamber);
  thisLCT.setTrknmb(trackNumber);
  thisLCT.setWireGroup(alct.getKeyWG());
  thisLCT.setStrip(gem.getKeyStrip());
  thisLCT.setBend(0);
  thisLCT.setBX(alct.getBX());
  if (runCCLUT_) {
    thisLCT.setRun3(true);
    thisLCT.setSlope(0);
    thisLCT.setQuartStripBit(false);
    thisLCT.setEighthStripBit(false);
    // ALCT-2GEM type LCTs do not bend in the chamber
    thisLCT.setRun3Pattern(4);
  }
}

void CSCGEMMotherboard::dropLowQualityALCTNoClusters(CSCALCTDigi& alct, const GEMInternalCluster& cluster) const {
  // clear alct if they are of low quality without matching GEM clusters
  if (alct.getQuality() == 0 and !cluster.isValid() and drop_low_quality_alct_no_gems_)
    alct.clear();
}

void CSCGEMMotherboard::dropLowQualityCLCTNoClusters(CSCCLCTDigi& clct, const GEMInternalCluster& cluster) const {
  // Here, we need to check which could be an ME1/a LCT
  bool dropLQCLCTNoGEMs = drop_low_quality_clct_no_gems_;
  if (isME11_ and clct.getKeyStrip() > CSCConstants::MAX_HALF_STRIP_ME1B)
    dropLQCLCTNoGEMs = drop_low_quality_clct_no_gems_me1a_;

  // clear clct if they are of low quality without matching GEM clusters
  if (clct.getQuality() <= 3 and !cluster.isValid() and dropLQCLCTNoGEMs)
    clct.clear();
}

void CSCGEMMotherboard::sortLCTsByBending(std::vector<CSCCorrelatedLCTDigi>& lcts) const {
  /*
    For Run-2 GEM-CSC trigger primitives, which we temporarily have
    to integrate with the Run-2 EMTF during LS2, we sort by quality.
    Larger quality means smaller bending
  */
  if (!runCCLUT_) {
    std::sort(lcts.begin(), lcts.end(), [](const CSCCorrelatedLCTDigi& lct1, const CSCCorrelatedLCTDigi& lct2) -> bool {
      return lct1.getQuality() > lct2.getQuality();
    });
  }

  /*
    For Run-3 GEM-CSC trigger primitives, which we have
    to integrate with the Run-3 EMTF, we sort by slope.
    Smaller slope means smaller bending
  */
  else {
    std::sort(lcts.begin(), lcts.end(), [](const CSCCorrelatedLCTDigi& lct1, const CSCCorrelatedLCTDigi& lct2) -> bool {
      return lct1.getSlope() < lct2.getSlope();
    });
  }
}
