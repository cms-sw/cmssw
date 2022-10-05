#include <memory>

#include "L1Trigger/CSCTriggerPrimitives/interface/CSCGEMMotherboard.h"
CSCGEMMotherboard::CSCGEMMotherboard(unsigned endcap,
                                     unsigned station,
                                     unsigned sector,
                                     unsigned subsector,
                                     unsigned chamber,
                                     const edm::ParameterSet& conf)
    : CSCMotherboard(endcap, station, sector, subsector, chamber, conf),
      drop_low_quality_alct_(tmbParams_.getParameter<bool>("dropLowQualityALCTs")),
      drop_low_quality_clct_(tmbParams_.getParameter<bool>("dropLowQualityCLCTs")),
      build_lct_from_alct_gem_(tmbParams_.getParameter<bool>("buildLCTfromALCTandGEM")),
      build_lct_from_clct_gem_(tmbParams_.getParameter<bool>("buildLCTfromCLCTandGEM")) {
  // case for ME1/1
  if (isME11_) {
    drop_low_quality_clct_me1a_ = tmbParams_.getParameter<bool>("dropLowQualityCLCTs_ME1a");
  }

  alct_gem_bx_window_size_ = tmbParams_.getParameter<unsigned>("windowBXALCTGEM");
  clct_gem_bx_window_size_ = tmbParams_.getParameter<unsigned>("windowBXCLCTGEM");

  assign_gem_csc_bending_ = tmbParams_.getParameter<bool>("assignGEMCSCBending");
  qualityAssignment_->setGEMCSCBending(assign_gem_csc_bending_);

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

//function to convert GEM-CSC amended signed slope into Run2 legacy pattern number
uint16_t CSCGEMMotherboard::Run2PatternConverter(const int slope) const {
  unsigned sign = std::signbit(slope);
  unsigned slope_ = abs(slope);
  uint16_t Run2Pattern = 0;

  if (slope_ < 3)
    Run2Pattern = 10;
  else if (slope_ < 6)
    Run2Pattern = 8 + sign;
  else if (slope_ < 9)
    Run2Pattern = 6 + sign;
  else if (slope_ < 12)
    Run2Pattern = 4 + sign;
  else
    Run2Pattern = 2 + sign;

  return Run2Pattern;
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

  // set CCLUT parameters if necessary
  if (runCCLUT_) {
    clctProc->setESLookupTables(lookupTableCCLUT_);
  }

  // Step 2: Run the processors
  const std::vector<CSCALCTDigi>& alctV = alctProc->run(wiredc);  // run anodeLCT
  const std::vector<CSCCLCTDigi>& clctV = clctProc->run(compdc);  // run cathodeLCT

  // Step 2b: encode high multiplicity bits (independent of LCT construction)
  encodeHighMultiplicityBits();

  // if there are no ALCTs and no CLCTs, do not run the ALCT-CLCT correlation
  if (alctV.empty() and clctV.empty())
    return;

  // set the lookup tables for coordinate conversion and matching
  if (isME11_) {
    clusterProc_->setESLookupTables(lookupTableME11ILT_);
    cscGEMMatcher_->setESLookupTables(lookupTableME11ILT_);
  }
  if (isME21_) {
    clusterProc_->setESLookupTables(lookupTableME21ILT_);
    cscGEMMatcher_->setESLookupTables(lookupTableME21ILT_);
  }

  // Step 3: run the GEM cluster processor to get the internal clusters
  clusterProc_->run(gemClusters);
  hasGE21Geometry16Partitions_ = clusterProc_->hasGE21Geometry16Partitions();

  // Step 4: ALCT-CLCT-GEM matching
  matchALCTCLCTGEM();

  // Step 5: Select at most 2 LCTs per BX
  selectLCTs();
}

void CSCGEMMotherboard::matchALCTCLCTGEM() {
  // no matching is done for GE2/1 geometries with 8 eta partitions
  // this has been superseded by 16-eta partition geometries
  if (isME21_ and !hasGE21Geometry16Partitions_)
    return;

  for (int bx_alct = 0; bx_alct < CSCConstants::MAX_ALCT_TBINS; bx_alct++) {
    // Declaration of all LCTs for this BX:

    // ALCT + CLCT + GEM
    CSCCorrelatedLCTDigi LCTbestAbestCgem, LCTbestAsecondCgem, LCTsecondAbestCgem, LCTsecondAsecondCgem;
    // ALCT + CLCT
    CSCCorrelatedLCTDigi LCTbestAbestC, LCTbestAsecondC, LCTsecondAbestC, LCTsecondAsecondC;
    // CLCT + 2 GEM
    CSCCorrelatedLCTDigi LCTbestCLCTgem, LCTsecondCLCTgem;
    // ALCT + 2 GEM
    CSCCorrelatedLCTDigi LCTbestALCTgem, LCTsecondALCTgem;

    // Construct all the LCTs, selection will come later:

    CSCALCTDigi bestALCT = alctProc->getBestALCT(bx_alct), secondALCT = alctProc->getSecondALCT(bx_alct);
    CSCCLCTDigi bestCLCT, secondCLCT;
    GEMInternalClusters clustersGEM;

    // Find best and second CLCTs by preferred CLCT BX, taking into account that there is an offset in the simulation

    unsigned matchingBX = 0;
    unsigned matching_clctbx = 0;
    unsigned bx_clct = 0;

    std::vector<unsigned> clctBx_qualbend_match;
    sortCLCTByQualBend(bx_alct, clctBx_qualbend_match);

    bool hasLocalShower = false;
    for (unsigned ibx = 1; ibx <= match_trig_window_size / 2; ibx++)
      hasLocalShower = (hasLocalShower or clctProc->getLocalShowerFlag(bx_alct - CSCConstants::ALCT_CLCT_OFFSET - ibx));
    // BestCLCT and secondCLCT
    for (unsigned mbx = 0; mbx < match_trig_window_size; mbx++) {
      //bx_clct_run2 would be overflow when bx_alct is small but it is okay
      unsigned bx_clct_run2 = bx_alct + preferred_bx_match_[mbx] - CSCConstants::ALCT_CLCT_OFFSET;
      unsigned bx_clct_qualbend = clctBx_qualbend_match[mbx];
      bx_clct = (sort_clct_bx_ or not(hasLocalShower)) ? bx_clct_run2 : bx_clct_qualbend;

      if (bx_clct >= CSCConstants::MAX_CLCT_TBINS)
        continue;
      matchingBX = mbx;
      matching_clctbx = mbx;

      if ((clctProc->getBestCLCT(bx_clct)).isValid())
        break;
    }

    bestCLCT = clctProc->getBestCLCT(bx_clct);
    secondCLCT = clctProc->getSecondCLCT(bx_clct);

    if (!bestALCT.isValid() and !secondALCT.isValid() and !bestCLCT.isValid() and !secondCLCT.isValid())
      continue;
    if (!build_lct_from_clct_gem_ and !bestALCT.isValid())
      continue;
    if (!build_lct_from_alct_gem_ and !bestCLCT.isValid())
      continue;

    if (infoV > 1)
      LogTrace("CSCGEMMotherboard") << "GEMCSCOTMB: Successful ALCT-CLCT match: bx_alct = " << bx_alct
                                    << "; bx_clct = " << matching_clctbx << "; mbx = " << matchingBX << " bestALCT "
                                    << bestALCT << " secondALCT " << secondALCT << " bestCLCT " << bestCLCT
                                    << " secondCLCT " << secondCLCT;
    // ALCT + CLCT + GEM

    for (unsigned gmbx = 0; gmbx < alct_gem_bx_window_size_; gmbx++) {
      unsigned bx_gem = bx_alct + preferred_bx_match_[gmbx];
      clustersGEM = clusterProc_->getClusters(bx_gem, GEMClusterProcessor::AllClusters);
      if (!clustersGEM.empty()) {
        correlateLCTsGEM(bestALCT, bestCLCT, clustersGEM, LCTbestAbestCgem);
        correlateLCTsGEM(bestALCT, secondCLCT, clustersGEM, LCTbestAsecondCgem);
        correlateLCTsGEM(secondALCT, bestCLCT, clustersGEM, LCTsecondAbestCgem);
        correlateLCTsGEM(secondALCT, secondCLCT, clustersGEM, LCTsecondAsecondCgem);
        break;
      }
    }

    // ALCT + CLCT

    correlateLCTsGEM(bestALCT, bestCLCT, LCTbestAbestC);
    correlateLCTsGEM(bestALCT, secondCLCT, LCTbestAsecondC);
    correlateLCTsGEM(secondALCT, bestCLCT, LCTsecondAbestC);
    correlateLCTsGEM(secondALCT, secondCLCT, LCTsecondAsecondC);

    // CLCT + 2 GEM

    if (build_lct_from_clct_gem_) {
      unsigned bx_gem = bx_alct;

      clustersGEM = clusterProc_->getClusters(bx_gem, GEMClusterProcessor::CoincidenceClusters);
      correlateLCTsGEM(bestCLCT, clustersGEM, LCTbestCLCTgem);
      clustersGEM = clusterProc_->getClusters(bx_gem, GEMClusterProcessor::CoincidenceClusters);
      correlateLCTsGEM(secondCLCT, clustersGEM, LCTsecondCLCTgem);
    }

    // ALCT + 2 GEM

    if (build_lct_from_alct_gem_) {
      for (unsigned gmbx = 0; gmbx < alct_gem_bx_window_size_; gmbx++) {
        unsigned bx_gem = bx_alct + preferred_bx_match_[gmbx];
        clustersGEM = clusterProc_->getClusters(bx_gem, GEMClusterProcessor::CoincidenceClusters);
        if (!clustersGEM.empty()) {
          correlateLCTsGEM(bestALCT, clustersGEM, LCTbestALCTgem);
          correlateLCTsGEM(secondALCT, clustersGEM, LCTsecondALCTgem);
          break;
        }
      }
    }

    // Select LCTs, following FW logic

    std::vector<CSCCorrelatedLCTDigi> selectedLCTs;

    // CASE => Only bestALCT is valid
    if (bestALCT.isValid() and !secondALCT.isValid() and !bestCLCT.isValid() and !secondCLCT.isValid()) {
      if (LCTbestALCTgem.isValid()) {
        LCTbestALCTgem.setTrknmb(1);
        allLCTs_(bx_alct, matchingBX, 0) = LCTbestALCTgem;
      }
    }

    // CASE => Only bestCLCT is valid
    if (!bestALCT.isValid() and !secondALCT.isValid() and bestCLCT.isValid() and !secondCLCT.isValid()) {
      if (LCTbestCLCTgem.isValid()) {
        LCTbestCLCTgem.setTrknmb(1);
        allLCTs_(bx_alct, matchingBX, 0) = LCTbestCLCTgem;
      }
    }

    // CASE => bestALCT and bestCLCT are valid
    if (bestALCT.isValid() and !secondALCT.isValid() and bestCLCT.isValid() and !secondCLCT.isValid()) {
      if (LCTbestAbestCgem.isValid()) {
        LCTbestAbestCgem.setTrknmb(1);
        allLCTs_(bx_alct, matchingBX, 0) = LCTbestAbestCgem;
      } else if (LCTbestAbestC.isValid()) {
        LCTbestAbestC.setTrknmb(1);
        allLCTs_(bx_alct, matchingBX, 0) = LCTbestAbestC;
      }
    }

    // CASE => bestALCT, secondALCT, bestCLCT are valid
    if (bestALCT.isValid() and secondALCT.isValid() and bestCLCT.isValid() and !secondCLCT.isValid()) {
      CSCCorrelatedLCTDigi lctbb, lctsb;
      if (LCTbestAbestCgem.isValid())
        lctbb = LCTbestAbestCgem;
      else if (LCTbestAbestC.isValid())
        lctbb = LCTbestAbestC;
      if (LCTsecondAbestCgem.isValid())
        lctsb = LCTsecondAbestCgem;
      else if (LCTsecondAbestC.isValid())
        lctsb = LCTsecondAbestC;

      if (lctbb.getQuality() >= lctsb.getQuality() and lctbb.isValid()) {
        selectedLCTs.push_back(lctbb);
        if (LCTsecondALCTgem.isValid() and build_lct_from_alct_gem_)
          selectedLCTs.push_back(LCTsecondALCTgem);
        else if (LCTsecondAbestC.isValid())
          selectedLCTs.push_back(LCTsecondAbestC);
      } else if (lctbb.getQuality() < lctsb.getQuality() and lctsb.isValid()) {
        selectedLCTs.push_back(lctsb);
        if (LCTbestALCTgem.isValid() and build_lct_from_alct_gem_)
          selectedLCTs.push_back(LCTbestALCTgem);
        else if (LCTbestAbestC.isValid())
          selectedLCTs.push_back(LCTbestAbestC);
      }

      sortLCTs(selectedLCTs);

      for (unsigned iLCT = 0; iLCT < std::min(unsigned(selectedLCTs.size()), unsigned(CSCConstants::MAX_LCTS_PER_CSC));
           iLCT++) {
        if (selectedLCTs[iLCT].isValid()) {
          selectedLCTs[iLCT].setTrknmb(iLCT + 1);
          allLCTs_(bx_alct, matchingBX, iLCT) = selectedLCTs[iLCT];
        }
      }
    }

    // CASE => bestALCT, bestCLCT, secondCLCT are valid
    if (bestALCT.isValid() and !secondALCT.isValid() and bestCLCT.isValid() and secondCLCT.isValid()) {
      CSCCorrelatedLCTDigi lctbb, lctbs;
      if (LCTbestAbestCgem.isValid())
        lctbb = LCTbestAbestCgem;
      else if (LCTbestAbestC.isValid())
        lctbb = LCTbestAbestC;
      if (LCTbestAsecondCgem.isValid())
        lctbs = LCTbestAsecondCgem;
      else if (LCTbestAsecondC.isValid())
        lctbs = LCTbestAsecondC;

      if (lctbb.getQuality() >= lctbs.getQuality() and lctbb.isValid()) {
        selectedLCTs.push_back(lctbb);
        if (LCTsecondCLCTgem.isValid() and build_lct_from_clct_gem_)
          selectedLCTs.push_back(LCTsecondCLCTgem);
        else if (LCTbestAsecondC.isValid())
          selectedLCTs.push_back(LCTbestAsecondC);
      } else if (lctbb.getQuality() < lctbs.getQuality() and lctbs.isValid()) {
        selectedLCTs.push_back(lctbs);
        if (LCTbestCLCTgem.isValid() and build_lct_from_alct_gem_)
          selectedLCTs.push_back(LCTbestCLCTgem);
        else if (LCTbestAbestC.isValid())
          selectedLCTs.push_back(LCTbestAbestC);
      }

      sortLCTs(selectedLCTs);

      for (unsigned iLCT = 0; iLCT < std::min(unsigned(selectedLCTs.size()), unsigned(CSCConstants::MAX_LCTS_PER_CSC));
           iLCT++) {
        if (selectedLCTs[iLCT].isValid()) {
          selectedLCTs[iLCT].setTrknmb(iLCT + 1);
          allLCTs_(bx_alct, matchingBX, iLCT) = selectedLCTs[iLCT];
        }
      }
    }

    // CASE => bestALCT, secondALCT, bestCLCT, secondCLCT are valid
    if (bestALCT.isValid() and secondALCT.isValid() and bestCLCT.isValid() and secondCLCT.isValid()) {
      CSCCorrelatedLCTDigi lctbb, lctbs, lctsb, lctss;

      // compute LCT bestA-bestC
      if (LCTbestAbestCgem.isValid())
        lctbb = LCTbestAbestCgem;
      else if (LCTbestAbestC.isValid())
        lctbb = LCTbestAbestC;

      // compute LCT bestA-secondC
      if (LCTbestAsecondCgem.isValid())
        lctbs = LCTbestAsecondCgem;
      else if (LCTbestAsecondC.isValid())
        lctbs = LCTbestAsecondC;

      if (lctbb.getQuality() >= lctbs.getQuality()) {
        // push back LCT bestA-bestC
        selectedLCTs.push_back(lctbb);

        // compute LCT secondA-secondC
        if (LCTsecondAsecondCgem.isValid())
          lctss = LCTsecondAsecondCgem;
        else if (LCTsecondAsecondC.isValid())
          lctss = LCTsecondAsecondC;

        // push back LCT secondA-secondC
        selectedLCTs.push_back(lctss);
      } else {
        // push back LCT bestA-secondC
        selectedLCTs.push_back(lctbs);

        // compute LCT secondA-bestC
        if (LCTsecondAbestCgem.isValid())
          lctsb = LCTsecondAbestCgem;
        else if (LCTsecondAbestC.isValid())
          lctsb = LCTsecondAbestC;

        // push back LCT secondA-bestC
        selectedLCTs.push_back(lctsb);
      }

      sortLCTs(selectedLCTs);

      for (unsigned iLCT = 0; iLCT < std::min(unsigned(selectedLCTs.size()), unsigned(CSCConstants::MAX_LCTS_PER_CSC));
           iLCT++) {
        if (selectedLCTs[iLCT].isValid()) {
          selectedLCTs[iLCT].setTrknmb(iLCT + 1);
          allLCTs_(bx_alct, matchingBX, iLCT) = selectedLCTs[iLCT];
        }
      }
    }
  }
}

// Correlate CSC and GEM information. Option ALCT-CLCT-GEM
void CSCGEMMotherboard::correlateLCTsGEM(const CSCALCTDigi& ALCT,
                                         const CSCCLCTDigi& CLCT,
                                         const GEMInternalClusters& clusters,
                                         CSCCorrelatedLCTDigi& lct) const {
  // Sanity checks on ALCT, CLCT, GEM clusters
  if (!ALCT.isValid()) {
    LogTrace("CSCGEMMotherboard") << "Best ALCT invalid in correlateLCTsGEM";
    return;
  }

  if (!CLCT.isValid()) {
    LogTrace("CSCGEMMotherboard") << "Best CLCT invalid in correlateLCTsGEM";
    return;
  }

  GEMInternalClusters ValidClusters;
  for (const auto& cl : clusters)
    if (cl.isValid())
      ValidClusters.push_back(cl);
  if (ValidClusters.empty())
    return;

  // We can now check possible triplets and construct all LCTs with
  // valid ALCT, valid CLCTs and GEM clusters
  GEMInternalCluster bestCluster;
  cscGEMMatcher_->bestClusterLoc(ALCT, CLCT, ValidClusters, bestCluster);
  if (bestCluster.isValid())
    constructLCTsGEM(ALCT, CLCT, bestCluster, lct);
}

// Correlate CSC information. Option ALCT-CLCT
void CSCGEMMotherboard::correlateLCTsGEM(const CSCALCTDigi& ALCT,
                                         const CSCCLCTDigi& CLCT,
                                         CSCCorrelatedLCTDigi& lct) const {
  // Sanity checks on ALCT, CLCT
  if (!ALCT.isValid() or (ALCT.getQuality() == 0 and drop_low_quality_alct_)) {
    LogTrace("CSCGEMMotherboard") << "Best ALCT invalid in correlateLCTsGEM";
    return;
  }

  bool dropLowQualityCLCT = drop_low_quality_clct_;
  if (isME11_ and CLCT.getKeyStrip() > CSCConstants::MAX_HALF_STRIP_ME1B)
    dropLowQualityCLCT = drop_low_quality_clct_me1a_;

  if (!CLCT.isValid() or (CLCT.getQuality() <= 3 and dropLowQualityCLCT)) {
    LogTrace("CSCGEMMotherboard") << "Best CLCT invalid in correlateLCTsGEM";
    return;
  }

  // construct LCT
  if (match_trig_enable and doesALCTCrossCLCT(ALCT, CLCT)) {
    constructLCTsGEM(ALCT, CLCT, lct);
  }
}

// Correlate CSC and GEM information. Option CLCT-GEM
void CSCGEMMotherboard::correlateLCTsGEM(const CSCCLCTDigi& CLCT,
                                         const GEMInternalClusters& clusters,
                                         CSCCorrelatedLCTDigi& lct) const {
  // Sanity checks on CLCT, GEM clusters
  bool dropLowQualityCLCT = drop_low_quality_clct_;
  if (isME11_ and CLCT.getKeyStrip() > CSCConstants::MAX_HALF_STRIP_ME1B)
    dropLowQualityCLCT = drop_low_quality_clct_me1a_;

  if (!CLCT.isValid() or (CLCT.getQuality() <= 3 and dropLowQualityCLCT)) {
    LogTrace("CSCGEMMotherboard") << "Best CLCT invalid in correlateLCTsGEM";
    return;
  }

  GEMInternalClusters ValidClusters;
  for (const auto& cl : clusters)
    if (cl.isValid())
      ValidClusters.push_back(cl);
  if (ValidClusters.empty())
    return;

  // get the best matching cluster
  GEMInternalCluster bestCluster;
  cscGEMMatcher_->bestClusterLoc(CLCT, ValidClusters, bestCluster);

  // construct all LCTs with valid CLCTs and coincidence clusters
  if (bestCluster.isCoincidence()) {
    constructLCTsGEM(CLCT, bestCluster, lct);
  }
}

// Correlate CSC and GEM information. Option ALCT-GEM
void CSCGEMMotherboard::correlateLCTsGEM(const CSCALCTDigi& ALCT,
                                         const GEMInternalClusters& clusters,
                                         CSCCorrelatedLCTDigi& lct) const {
  // Sanity checks on ALCT, GEM clusters
  if (!ALCT.isValid() or (ALCT.getQuality() == 0 and drop_low_quality_alct_)) {
    LogTrace("CSCGEMMotherboard") << "Best ALCT invalid in correlateLCTsGEM";
    return;
  }

  GEMInternalClusters ValidClusters;
  for (const auto& cl : clusters)
    if (cl.isValid())
      ValidClusters.push_back(cl);
  if (ValidClusters.empty())
    return;

  // get the best matching cluster
  GEMInternalCluster bestCluster;
  cscGEMMatcher_->bestClusterLoc(ALCT, ValidClusters, bestCluster);

  // construct all LCTs with valid ALCTs and coincidence clusters
  if (bestCluster.isCoincidence()) {
    constructLCTsGEM(ALCT, bestCluster, lct);
  }
}

// Construct LCT from CSC and GEM information. Option ALCT-CLCT-GEM
void CSCGEMMotherboard::constructLCTsGEM(const CSCALCTDigi& alct,
                                         const CSCCLCTDigi& clct,
                                         const GEMInternalCluster& gem,
                                         CSCCorrelatedLCTDigi& thisLCT) const {
  thisLCT.setValid(true);
  if (gem.isCoincidence())
    thisLCT.setType(CSCCorrelatedLCTDigi::ALCTCLCT2GEM);
  else if (gem.isValid())
    thisLCT.setType(CSCCorrelatedLCTDigi::ALCTCLCTGEM);
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
  thisLCT.setTrknmb(0);  // will be set later after sorting
  thisLCT.setWireGroup(alct.getKeyWG());
  thisLCT.setStrip(clct.getKeyStrip());
  thisLCT.setBend(clct.getBend());
  thisLCT.setBX(alct.getBX());
  if (runCCLUT_) {
    thisLCT.setRun3(true);
    if (assign_gem_csc_bending_ &&
        gem.isValid()) {  //calculate new slope from strip difference between CLCT and associated GEM
      int slope = cscGEMMatcher_->calculateGEMCSCBending(clct, gem);
      thisLCT.setSlope(abs(slope));
      thisLCT.setBend(std::signbit(slope));
      thisLCT.setPattern(Run2PatternConverter(slope));
    } else
      thisLCT.setSlope(clct.getSlope());
    thisLCT.setQuartStripBit(clct.getQuartStripBit());
    thisLCT.setEighthStripBit(clct.getEighthStripBit());
    thisLCT.setRun3Pattern(clct.getRun3Pattern());
  }
}

// Construct LCT from CSC and GEM information. Option ALCT-CLCT
void CSCGEMMotherboard::constructLCTsGEM(const CSCALCTDigi& aLCT,
                                         const CSCCLCTDigi& cLCT,
                                         CSCCorrelatedLCTDigi& thisLCT) const {
  thisLCT.setValid(true);
  thisLCT.setType(CSCCorrelatedLCTDigi::ALCTCLCT);
  thisLCT.setALCT(getBXShiftedALCT(aLCT));
  thisLCT.setCLCT(getBXShiftedCLCT(cLCT));
  thisLCT.setPattern(encodePattern(cLCT.getPattern()));
  thisLCT.setMPCLink(0);
  thisLCT.setBX0(0);
  thisLCT.setSyncErr(0);
  thisLCT.setCSCID(theTrigChamber);
  thisLCT.setTrknmb(0);  // will be set later after sorting
  thisLCT.setWireGroup(aLCT.getKeyWG());
  thisLCT.setStrip(cLCT.getKeyStrip());
  thisLCT.setBend(cLCT.getBend());
  thisLCT.setBX(aLCT.getBX());
  thisLCT.setQuality(qualityAssignment_->findQuality(aLCT, cLCT));
  if (runCCLUT_) {
    thisLCT.setRun3(true);
    // 4-bit slope value derived with the CCLUT algorithm
    thisLCT.setSlope(cLCT.getSlope());
    thisLCT.setQuartStripBit(cLCT.getQuartStripBit());
    thisLCT.setEighthStripBit(cLCT.getEighthStripBit());
    thisLCT.setRun3Pattern(cLCT.getRun3Pattern());
  }
}

// Construct LCT from CSC and GEM information. Option CLCT-2GEM
void CSCGEMMotherboard::constructLCTsGEM(const CSCCLCTDigi& clct,
                                         const GEMInternalCluster& gem,
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
  thisLCT.setTrknmb(0);  // will be set later after sorting
  thisLCT.setWireGroup(gem.getKeyWG());
  thisLCT.setStrip(clct.getKeyStrip());
  thisLCT.setBend(clct.getBend());
  thisLCT.setBX(gem.bx());
  if (runCCLUT_) {
    thisLCT.setRun3(true);
    if (assign_gem_csc_bending_ &&
        gem.isValid()) {  //calculate new slope from strip difference between CLCT and associated GEM
      int slope = cscGEMMatcher_->calculateGEMCSCBending(clct, gem);
      thisLCT.setSlope(abs(slope));
      thisLCT.setBend(pow(-1, std::signbit(slope)));
      thisLCT.setPattern(Run2PatternConverter(slope));
    } else
      thisLCT.setSlope(clct.getSlope());
    thisLCT.setQuartStripBit(clct.getQuartStripBit());
    thisLCT.setEighthStripBit(clct.getEighthStripBit());
    thisLCT.setRun3Pattern(clct.getRun3Pattern());
  }
}

// Construct LCT from CSC and GEM information. Option ALCT-2GEM
void CSCGEMMotherboard::constructLCTsGEM(const CSCALCTDigi& alct,
                                         const GEMInternalCluster& gem,
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
  thisLCT.setTrknmb(0);  // will be set later after sorting
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

void CSCGEMMotherboard::sortLCTs(std::vector<CSCCorrelatedLCTDigi>& lcts) const {
  // LCTs are sorted by quality. If there are two with the same quality, then the sorting is done by the slope
  std::sort(lcts.begin(), lcts.end(), [](const CSCCorrelatedLCTDigi& lct1, const CSCCorrelatedLCTDigi& lct2) -> bool {
    if (lct1.getQuality() > lct2.getQuality())
      return lct1.getQuality() > lct2.getQuality();
    else if (lct1.getQuality() == lct2.getQuality())
      return lct1.getSlope() < lct2.getSlope();
    else
      return false;
  });
}
