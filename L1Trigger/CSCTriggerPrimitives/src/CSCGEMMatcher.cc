#include "L1Trigger/CSCTriggerPrimitives/interface/CSCGEMMatcher.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/CSCLUTReader.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/GEMInternalCluster.h"
#include "DataFormats/CSCDigi/interface/CSCConstants.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include <cmath>

CSCGEMMatcher::CSCGEMMatcher(
    int endcap, unsigned station, unsigned chamber, const edm::ParameterSet& tmbParams, const edm::ParameterSet& conf)
    : endcap_(endcap), station_(station), chamber_(chamber) {
  isEven_ = (chamber_ % 2 == 0);

  maxDeltaBXALCTGEM_ = tmbParams.getParameter<unsigned>("maxDeltaBXALCTGEM");
  maxDeltaBXCLCTGEM_ = tmbParams.getParameter<unsigned>("maxDeltaBXCLCTGEM");

  matchWithHS_ = tmbParams.getParameter<bool>("matchWithHS");

  maxDeltaHsEven_ = tmbParams.getParameter<unsigned>("maxDeltaHsEven");
  maxDeltaHsOdd_ = tmbParams.getParameter<unsigned>("maxDeltaHsOdd");

  if (station_ == 1) {
    maxDeltaHsEvenME1a_ = tmbParams.getParameter<unsigned>("maxDeltaHsEvenME1a");
    maxDeltaHsOddME1a_ = tmbParams.getParameter<unsigned>("maxDeltaHsOddME1a");
  }

  mitigateSlopeByCosi_ = tmbParams.getParameter<bool>("mitigateSlopeByCosi");
  assign_gem_csc_bending_ = tmbParams.getParameter<bool>("assignGEMCSCBending");

  if (mitigateSlopeByCosi_) {
    gemCscSlopeCosiFiles_ = conf.getParameter<std::vector<std::string>>("gemCscSlopeCosiFiles");

    gem_csc_slope_cosi_2to1_L1_ME11_even_ = std::make_unique<CSCLUTReader>(gemCscSlopeCosiFiles_[0]);
    gem_csc_slope_cosi_2to1_L1_ME11_odd_ = std::make_unique<CSCLUTReader>(gemCscSlopeCosiFiles_[1]);
    gem_csc_slope_cosi_3to1_L1_ME11_even_ = std::make_unique<CSCLUTReader>(gemCscSlopeCosiFiles_[2]);
    gem_csc_slope_cosi_3to1_L1_ME11_odd_ = std::make_unique<CSCLUTReader>(gemCscSlopeCosiFiles_[3]);

    gemCscSlopeCosiCorrectionFiles_ = conf.getParameter<std::vector<std::string>>("gemCscSlopeCosiCorrectionFiles");

    gem_csc_slope_cosi_corr_L1_ME11_even_ = std::make_unique<CSCLUTReader>(gemCscSlopeCosiCorrectionFiles_[0]);
    gem_csc_slope_cosi_corr_L2_ME11_even_ = std::make_unique<CSCLUTReader>(gemCscSlopeCosiCorrectionFiles_[1]);
    gem_csc_slope_cosi_corr_L1_ME11_odd_ = std::make_unique<CSCLUTReader>(gemCscSlopeCosiCorrectionFiles_[2]);
    gem_csc_slope_cosi_corr_L2_ME11_odd_ = std::make_unique<CSCLUTReader>(gemCscSlopeCosiCorrectionFiles_[3]);
  } else {
    gemCscSlopeCorrectionFiles_ = conf.getParameter<std::vector<std::string>>("gemCscSlopeCorrectionFiles");

    gem_csc_slope_corr_L1_ME11_even_ = std::make_unique<CSCLUTReader>(gemCscSlopeCorrectionFiles_[0]);
    gem_csc_slope_corr_L2_ME11_even_ = std::make_unique<CSCLUTReader>(gemCscSlopeCorrectionFiles_[1]);
    gem_csc_slope_corr_L1_ME11_odd_ = std::make_unique<CSCLUTReader>(gemCscSlopeCorrectionFiles_[2]);
    gem_csc_slope_corr_L2_ME11_odd_ = std::make_unique<CSCLUTReader>(gemCscSlopeCorrectionFiles_[3]);
  }

  if (assign_gem_csc_bending_) {
    if (station_ == 1) {
      esDiffToSlopeME1aFiles_ = conf.getParameter<std::vector<std::string>>("esDiffToSlopeME1aFiles");
      esDiffToSlopeME1bFiles_ = conf.getParameter<std::vector<std::string>>("esDiffToSlopeME1bFiles");

      es_diff_slope_L1_ME1a_even_ = std::make_unique<CSCLUTReader>(esDiffToSlopeME1aFiles_[0]);
      es_diff_slope_L1_ME1a_odd_ = std::make_unique<CSCLUTReader>(esDiffToSlopeME1aFiles_[1]);
      es_diff_slope_L2_ME1a_even_ = std::make_unique<CSCLUTReader>(esDiffToSlopeME1aFiles_[2]);
      es_diff_slope_L2_ME1a_odd_ = std::make_unique<CSCLUTReader>(esDiffToSlopeME1aFiles_[3]);

      es_diff_slope_L1_ME1b_even_ = std::make_unique<CSCLUTReader>(esDiffToSlopeME1bFiles_[0]);
      es_diff_slope_L1_ME1b_odd_ = std::make_unique<CSCLUTReader>(esDiffToSlopeME1bFiles_[1]);
      es_diff_slope_L2_ME1b_even_ = std::make_unique<CSCLUTReader>(esDiffToSlopeME1bFiles_[2]);
      es_diff_slope_L2_ME1b_odd_ = std::make_unique<CSCLUTReader>(esDiffToSlopeME1bFiles_[3]);
    }

    if (station_ == 2) {
      esDiffToSlopeME21Files_ = conf.getParameter<std::vector<std::string>>("esDiffToSlopeME21Files");

      es_diff_slope_L1_ME21_even_ = std::make_unique<CSCLUTReader>(esDiffToSlopeME21Files_[0]);
      es_diff_slope_L1_ME21_odd_ = std::make_unique<CSCLUTReader>(esDiffToSlopeME21Files_[1]);
      es_diff_slope_L2_ME21_even_ = std::make_unique<CSCLUTReader>(esDiffToSlopeME21Files_[2]);
      es_diff_slope_L2_ME21_odd_ = std::make_unique<CSCLUTReader>(esDiffToSlopeME21Files_[3]);
    }
  }
}

unsigned CSCGEMMatcher::calculateGEMCSCBending(const CSCCLCTDigi& clct, const GEMInternalCluster& cluster) const {
  // difference in 1/8-strip number
  const unsigned diff = std::abs(int(clct.getKeyStrip(8)) - int(cluster.getKeyStrip(8)));

  unsigned slope = 0;

  // need LUT to convert differences in 1/8-strips between GEM and CSC to slope
  if (station_ == 2) {
    if (isEven_) {
      if (cluster.id().layer() == 1)
        slope = es_diff_slope_L1_ME21_even_->lookup(diff);
      else
        slope = es_diff_slope_L2_ME21_even_->lookup(diff);
    } else {
      if (cluster.id().layer() == 1)
        slope = es_diff_slope_L1_ME21_odd_->lookup(diff);
      else
        slope = es_diff_slope_L2_ME21_odd_->lookup(diff);
    }
  }

  const bool isME1a(station_ == 1 and clct.getKeyStrip() > CSCConstants::MAX_HALF_STRIP_ME1B);

  if (station_ == 1 and isME1a) {
    if (isEven_) {
      if (cluster.id().layer() == 1)
        slope = es_diff_slope_L1_ME1a_even_->lookup(diff);
      else
        slope = es_diff_slope_L2_ME1a_even_->lookup(diff);
    } else {
      if (cluster.id().layer() == 1)
        slope = es_diff_slope_L1_ME1a_odd_->lookup(diff);
      else
        slope = es_diff_slope_L2_ME1a_odd_->lookup(diff);
    }
  } else {
    if (isEven_) {
      if (cluster.id().layer() == 1)
        slope = es_diff_slope_L1_ME1b_even_->lookup(diff);
      else
        slope = es_diff_slope_L2_ME1b_even_->lookup(diff);
    } else {
      if (cluster.id().layer() == 1)
        slope = es_diff_slope_L1_ME1b_odd_->lookup(diff);
      else
        slope = es_diff_slope_L2_ME1b_odd_->lookup(diff);
    }
  }

  return slope;
}

// match an ALCT to GEMInternalCluster by bunch-crossing
void CSCGEMMatcher::matchingClustersBX(const CSCALCTDigi& alct,
                                       const GEMInternalClusters& clusters,
                                       GEMInternalClusters& output) const {
  if (!alct.isValid() or clusters.empty())
    return;

  // select clusters matched in time
  for (const auto& cl : clusters) {
    const unsigned diff = std::abs(int(alct.getBX()) - cl.bx());
    if (diff <= maxDeltaBXALCTGEM_)
      output.push_back(cl);
  }
}

// match a CLCT to GEMInternalCluster by bunch-crossing
void CSCGEMMatcher::matchingClustersBX(const CSCCLCTDigi& clct,
                                       const GEMInternalClusters& clusters,
                                       GEMInternalClusters& output) const {
  if (!clct.isValid() or clusters.empty())
    return;

  // select clusters matched in time
  for (const auto& cl : clusters) {
    const unsigned diff = std::abs(int(clct.getBX()) - cl.bx());
    if (diff <= maxDeltaBXCLCTGEM_)
      output.push_back(cl);
  }
}

// match an ALCT and CLCT to GEMInternalCluster by bunch-crossing
void CSCGEMMatcher::matchingClustersBX(const CSCALCTDigi& alct,
                                       const CSCCLCTDigi& clct,
                                       const GEMInternalClusters& clusters,
                                       GEMInternalClusters& output) const {
  // both need to be valid
  if (!alct.isValid() or !clct.isValid() or clusters.empty())
    return;

  // get the single matches
  GEMInternalClusters alctClusters, clctClusters;
  matchingClustersBX(alct, clusters, alctClusters);
  matchingClustersBX(clct, clusters, clctClusters);

  // get the intersection
  for (const auto& p : alctClusters) {
    for (const auto& q : clctClusters) {
      if (p == q) {
        output.push_back(p);
      }
    }
  }
}

void CSCGEMMatcher::matchingClustersLoc(const CSCALCTDigi& alct,
                                        const GEMInternalClusters& clusters,
                                        GEMInternalClusters& output) const {
  if (!alct.isValid() or clusters.empty())
    return;

  // select clusters matched in wiregroup
  for (const auto& cl : clusters) {
    // for now add 10 wiregroups to make sure the matching can be done
    // this should be quite generous
    unsigned deltaWG(station_ == 1 ? 10 : 20);
    if (cl.min_wg() <= alct.getKeyWG() and alct.getKeyWG() <= cl.max_wg() + deltaWG) {
      output.push_back(cl);
    }
  }
}

void CSCGEMMatcher::matchingClustersLoc(const CSCCLCTDigi& clct,
                                        const GEMInternalClusters& clusters,
                                        GEMInternalClusters& output) const {
  if (!clct.isValid() or clusters.empty())
    return;

  // select clusters matched by 1/2-strip or 1/8-strip
  for (const auto& cl : clusters) {
    const bool isMatched(matchWithHS_ ? matchedClusterLocHS(clct, cl) : matchedClusterLocES(clct, cl));
    if (isMatched) {
      output.push_back(cl);
    }
  }
}

// match by 1/2-strip
bool CSCGEMMatcher::matchedClusterLocHS(const CSCCLCTDigi& clct, const GEMInternalCluster& cluster) const {
  const bool isME1a(station_ == 1 and clct.getKeyStrip() > CSCConstants::MAX_HALF_STRIP_ME1B);

  unsigned halfStripDiff = std::abs(int(clct.getKeyStrip(2)) - int(cluster.getKeyStrip(2)));
  if (isME1a) {
    halfStripDiff = std::abs(int(clct.getKeyStrip(2)) - int(cluster.getKeyStripME1a(2)));
  }

  // 98% acceptance cuts
  unsigned halfStripCut;
  if (isEven_) {
    if (isME1a)
      halfStripCut = maxDeltaHsEvenME1a_;
    else
      halfStripCut = maxDeltaHsEven_;
  } else {
    if (isME1a)
      halfStripCut = maxDeltaHsOddME1a_;
    else
      halfStripCut = maxDeltaHsOdd_;
  }
  // 10 degree chamber is ~0.18 radian wide
  // 98% acceptance for clusters in odd/even chambers for muons with 5 GeV
  // {5, 0.02123785, 0.00928431}
  // This corresponds to 0.12 and 0.052 fractions of the chamber
  // or 20 and 9 half-strips

  // 20 degree chamber is ~0.35 radian wide
  // 98% acceptance for clusters in odd/even chambers for muons with 5 GeV
  // {5, 0.01095490, 0.00631625},
  // This corresponds to 0.031 and 0.018 fractions of the chamber
  // or 5 and 3 half-strips

  return halfStripDiff <= halfStripCut;
}

// match by 1/8-strip
bool CSCGEMMatcher::matchedClusterLocES(const CSCCLCTDigi& clct, const GEMInternalCluster& cl) const {
  // key 1/8-strip
  int key_es = -1;

  //modification of DeltaStrip by CLCT slope
  int SlopeShift = 0;
  uint16_t baseSlope = 0;
  if (mitigateSlopeByCosi_)
    baseSlope = mitigatedSlopeByConsistency(clct);
  else
    baseSlope = clct.getSlope();
  int clctSlope = pow(-1, clct.getBend()) * baseSlope;

  // for coincidences or single clusters in L1
  if (cl.isCoincidence() or cl.id().layer() == 1) {
    key_es = cl.layer1_middle_es();
    if (station_ == 1 and clct.getKeyStrip() > CSCConstants::MAX_HALF_STRIP_ME1B)
      key_es = cl.layer1_middle_es_me1a();

    //set SlopeShift for L1 or Copad case
    SlopeShift =
        CSCGEMSlopeCorrector(true, clctSlope);  // fixed to facing detectors, must be determined at motherboard level
  }

  // for single clusters in L2
  else if (cl.id().layer() == 2) {
    key_es = cl.layer2_middle_es();
    if (station_ == 1 and clct.getKeyStrip() > CSCConstants::MAX_HALF_STRIP_ME1B)
      key_es = cl.layer2_middle_es_me1a();

    //set SlopeShift for L2 case
    SlopeShift =
        CSCGEMSlopeCorrector(false, clctSlope);  // fixed to facing detectors, must be determined at motherboard level

  }

  else
    edm::LogWarning("CSCGEMMatcher") << "cluster.id().layer =" << cl.id().layer() << " out of acceptable range 1-2!";

  // matching by 1/8-strip
  // determine matching window by chamber, assuming facing chambers only are processed
  int window = chamber_ % 2 == 0 ? 20 : 40;

  return std::abs(clct.getKeyStrip(8) - key_es + SlopeShift) < window;
}

void CSCGEMMatcher::matchingClustersLoc(const CSCALCTDigi& alct,
                                        const CSCCLCTDigi& clct,
                                        const GEMInternalClusters& clusters,
                                        GEMInternalClusters& output) const {
  // both need to be valid
  if (!alct.isValid() or !clct.isValid() or clusters.empty())
    return;

  // get the single matches
  GEMInternalClusters alctClusters, clctClusters;
  matchingClustersLoc(alct, clusters, alctClusters);
  matchingClustersLoc(clct, clusters, clctClusters);

  // get the intersection
  for (const auto& p : alctClusters) {
    for (const auto& q : clctClusters) {
      if (p == q) {
        output.push_back(p);
      }
    }
  }
}

void CSCGEMMatcher::matchingClustersBXLoc(const CSCALCTDigi& alct,
                                          const GEMInternalClusters& clusters,
                                          GEMInternalClusters& output) const {
  if (!alct.isValid() or clusters.empty())
    return;

  // match by BX
  GEMInternalClusters clustersBX;
  matchingClustersBX(alct, clusters, clustersBX);

  // match spatially
  matchingClustersLoc(alct, clustersBX, output);
}

void CSCGEMMatcher::matchingClustersBXLoc(const CSCCLCTDigi& clct,
                                          const GEMInternalClusters& clusters,
                                          GEMInternalClusters& output) const {
  if (!clct.isValid() or clusters.empty())
    return;

  // match by BX
  GEMInternalClusters clustersBX;
  matchingClustersBX(clct, clusters, clustersBX);

  // match spatially
  matchingClustersLoc(clct, clustersBX, output);
}

void CSCGEMMatcher::matchingClustersBXLoc(const CSCALCTDigi& alct,
                                          const CSCCLCTDigi& clct,
                                          const GEMInternalClusters& clusters,
                                          GEMInternalClusters& selected) const {
  // both need to be valid
  if (!alct.isValid() or !clct.isValid() or clusters.empty())
    return;

  // match by BX
  GEMInternalClusters clustersBX;
  matchingClustersBX(alct, clct, clusters, clustersBX);

  // match spatially
  matchingClustersLoc(alct, clct, clustersBX, selected);
}

void CSCGEMMatcher::bestClusterBXLoc(const CSCALCTDigi& alct,
                                     const GEMInternalClusters& clusters,
                                     GEMInternalCluster& best) const {
  if (!alct.isValid() or clusters.empty())
    return;

  GEMInternalClusters clustersBXLoc;
  matchingClustersBXLoc(alct, clusters, clustersBXLoc);

  // simply pick the first matching one
  if (!clustersBXLoc.empty())
    best = clustersBXLoc[0];
}

void CSCGEMMatcher::bestClusterBXLoc(const CSCCLCTDigi& clct,
                                     const GEMInternalClusters& clusters,
                                     GEMInternalCluster& best) const {
  if (!clct.isValid() or clusters.empty())
    return;

  // match by BX
  GEMInternalClusters clustersBXLoc;
  matchingClustersBXLoc(clct, clusters, clustersBXLoc);

  // FIXME - for now: pick the first matching one
  if (!clustersBXLoc.empty())
    best = clustersBXLoc[0];
}

void CSCGEMMatcher::bestClusterBXLoc(const CSCALCTDigi& alct,
                                     const CSCCLCTDigi& clct,
                                     const GEMInternalClusters& clusters,
                                     GEMInternalCluster& best) const {
  // match by BX
  GEMInternalClusters clustersBXLoc;
  matchingClustersBXLoc(alct, clct, clusters, clustersBXLoc);

  // FIXME - for now: pick the first matching one
  if (!clustersBXLoc.empty())
    best = clustersBXLoc[0];
}

uint16_t CSCGEMMatcher::mitigatedSlopeByConsistency(const CSCCLCTDigi& clct) const {
  //extract hit values from CLCT hit matrix
  std::vector<std::vector<uint16_t>> CLCTHitMatrix = clct.getHits();
  int CLCTHits[6] = {-1, -1, -1, -1, -1, -1};

  for (unsigned layer = 0; layer < CLCTHitMatrix.size(); ++layer) {
    for (unsigned position = 0; position < CLCTHitMatrix.at(layer).size(); ++position) {
      const uint16_t value = CLCTHitMatrix.at(layer).at(position);
      if (value != 0 && value != 65535) {
        CLCTHits[layer] = (int)value;
        break;
      }
    }
  }

  //calculate slope consistency
  float MinMaxPairDifferences[2] = {999., -999.};
  for (unsigned First = 0; First < 5; ++First) {
    //skip empty layers
    if (CLCTHits[First] == -1)
      continue;
    for (unsigned Second = First + 1; Second < 6; ++Second) {
      //skip empty layers
      if (CLCTHits[Second] == -1)
        continue;
      float PairDifference = (CLCTHits[First] - CLCTHits[Second]) / (float)(Second - First);
      if (PairDifference < MinMaxPairDifferences[0])
        MinMaxPairDifferences[0] = PairDifference;
      if (PairDifference > MinMaxPairDifferences[1])
        MinMaxPairDifferences[1] = PairDifference;
    }
  }

  //calculate consistency of slope indicator: cosi
  uint16_t cosi = std::ceil(std::abs(MinMaxPairDifferences[1] - MinMaxPairDifferences[0]));

  //disambiguate cosi cases

  //extremely inconsistent track, deprecate slope
  if (cosi > 3)
    return 0;
  //consistent slope, do not change
  else if (cosi < 2)
    return clct.getSlope();
  //need to look up in table 2->1
  else if (cosi == 2) {
    if (chamber_ % 2 == 0)
      return gem_csc_slope_cosi_2to1_L1_ME11_even_->lookup(clct.getSlope());
    else
      return gem_csc_slope_cosi_2to1_L1_ME11_odd_->lookup(clct.getSlope());
  }
  //need to look up in table 3->1
  else if (cosi == 3) {
    if (chamber_ % 2 == 0)
      return gem_csc_slope_cosi_3to1_L1_ME11_even_->lookup(clct.getSlope());
    else
      return gem_csc_slope_cosi_3to1_L1_ME11_odd_->lookup(clct.getSlope());
  }
  //just to avoid compiler errors an error code
  else {
    return 999;
  }
}

int CSCGEMMatcher::CSCGEMSlopeCorrector(bool isL1orCoincidence, int cscSlope) const {
  int SlopeShift = 0;
  int SlopeSign = cscSlope / std::abs(cscSlope);
  //account for slope mitigation by cosi, if opted-in
  if (mitigateSlopeByCosi_) {
    //determine cosi-based slope correction
    if (chamber_ % 2 == 0) {
      if (isL1orCoincidence)
        SlopeShift = gem_csc_slope_cosi_corr_L1_ME11_even_->lookup(std::abs(cscSlope));
      else
        SlopeShift = gem_csc_slope_cosi_corr_L2_ME11_even_->lookup(std::abs(cscSlope));
    } else {
      if (isL1orCoincidence)
        SlopeShift = gem_csc_slope_cosi_corr_L1_ME11_odd_->lookup(std::abs(cscSlope));
      else
        SlopeShift = gem_csc_slope_cosi_corr_L2_ME11_odd_->lookup(std::abs(cscSlope));
    }
  } else {
    //determine shift by slope correction
    if (chamber_ % 2 == 0) {
      if (isL1orCoincidence)
        SlopeShift = gem_csc_slope_corr_L1_ME11_even_->lookup(std::abs(cscSlope));
      else
        SlopeShift = gem_csc_slope_corr_L2_ME11_even_->lookup(std::abs(cscSlope));
    } else {
      if (isL1orCoincidence)
        SlopeShift = gem_csc_slope_corr_L1_ME11_odd_->lookup(std::abs(cscSlope));
      else
        SlopeShift = gem_csc_slope_corr_L2_ME11_odd_->lookup(std::abs(cscSlope));
    }
  }
  return std::round(SlopeShift * SlopeSign * endcap_);
}
