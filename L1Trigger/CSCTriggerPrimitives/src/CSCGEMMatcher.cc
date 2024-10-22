#include "L1Trigger/CSCTriggerPrimitives/interface/CSCGEMMatcher.h"
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

  enable_match_gem_me1a_ = tmbParams.getParameter<bool>("enableMatchGEMandME1a");
  enable_match_gem_me1b_ = tmbParams.getParameter<bool>("enableMatchGEMandME1b");

  maxDeltaWG_ = tmbParams.getParameter<unsigned>("maxDeltaWG");
  maxDeltaHsEven_ = tmbParams.getParameter<unsigned>("maxDeltaHsEven");
  maxDeltaHsOdd_ = tmbParams.getParameter<unsigned>("maxDeltaHsOdd");

  matchCLCTpropagation_ = tmbParams.getParameter<bool>("matchCLCTpropagation");

  mitigateSlopeByCosi_ = tmbParams.getParameter<bool>("mitigateSlopeByCosi");
  assign_gem_csc_bending_ = tmbParams.getParameter<bool>("assignGEMCSCBending");
}

//##############################################################
//                Best clusters by location
//##############################################################

void CSCGEMMatcher::bestClusterLoc(const CSCALCTDigi& alct,
                                   const GEMInternalClusters& clusters,
                                   GEMInternalCluster& best) const {
  if (!alct.isValid() or clusters.empty())
    return;

  // match spatially
  GEMInternalClusters clustersLoc;
  matchingClustersLoc(alct, clusters, clustersLoc);

  // simply pick the first matching one
  if (!clustersLoc.empty())
    best = clustersLoc[0];
}

void CSCGEMMatcher::bestClusterLoc(const CSCCLCTDigi& clct,
                                   const GEMInternalClusters& clusters,
                                   const CSCL1TPLookupTableME11ILT* lookupTableME11ILT,
                                   const CSCL1TPLookupTableME21ILT* lookupTableME21ILT,
                                   GEMInternalCluster& best) const {
  if (!clct.isValid() or clusters.empty())
    return;

  // match spatially
  bool ignoreALCTGEMmatch = true;
  GEMInternalClusters clustersLoc;
  matchingClustersLoc(clct, clusters, clustersLoc, ignoreALCTGEMmatch, lookupTableME11ILT, lookupTableME21ILT);

  // the first matching one is also the closest in phi distance (to expected position, if extrapolating), by ordered list in CLCT matching
  if (!clustersLoc.empty())
    best = clustersLoc[0];
}

void CSCGEMMatcher::bestClusterLoc(const CSCALCTDigi& alct,
                                   const CSCCLCTDigi& clct,
                                   const GEMInternalClusters& clusters,
                                   const CSCL1TPLookupTableME11ILT* lookupTableME11ILT,
                                   const CSCL1TPLookupTableME21ILT* lookupTableME21ILT,
                                   GEMInternalCluster& best) const {
  if (!alct.isValid() or !clct.isValid() or clusters.empty())
    return;

  // match spatially
  GEMInternalClusters clustersLoc;
  matchingClustersLoc(alct, clct, clusters, lookupTableME11ILT, lookupTableME21ILT, clustersLoc);

  // the first matching one is also the closest in phi distance (to expected position, if extrapolating), by ordered list in CLCT matching
  if (!clustersLoc.empty()) {
    best = clustersLoc[0];
    if (best.isCoincidence() and !best.isMatchingLayer1() and best.isMatchingLayer2())
      best.set_coincidence(false);
    // std::cout << "\nGEM selected: " << best << "\n" << std::endl;
  }
}

//##############################################################
//                  Matching by locations
//##############################################################

// match an ALCT to GEMInternalCluster by location
void CSCGEMMatcher::matchingClustersLoc(const CSCALCTDigi& alct,
                                        const GEMInternalClusters& clusters,
                                        GEMInternalClusters& output) const {
  if (!alct.isValid() or clusters.empty())
    return;

  int number_of_wg = 0;
  if (station_ == 1)
    number_of_wg = CSCConstants::NUM_WIREGROUPS_ME11;
  else if (station_ == 2)
    number_of_wg = CSCConstants::NUM_WIREGROUPS_ME21;

  // select clusters matched in wiregroup

  for (const auto& cl : clusters) {
    // std::cout << "GEM cluster: " << cl << std::endl;
    bool isMatchedLayer1 = false;
    bool isMatchedLayer2 = false;

    if (cl.id1().layer() == 1) {  // cluster has valid layer 1
      int min_wg = std::max(0, int(cl.layer1_min_wg() - maxDeltaWG_));
      int max_wg = std::min(number_of_wg - 1, int(cl.layer1_max_wg() + maxDeltaWG_));
      if (min_wg <= alct.getKeyWG() and alct.getKeyWG() <= max_wg)
        isMatchedLayer1 = true;
    }
    if (cl.id2().layer() == 2) {  // cluster has valid layer 2
      int min_wg = std::max(0, int(cl.layer2_min_wg() - maxDeltaWG_));
      int max_wg = std::min(number_of_wg - 1, int(cl.layer2_max_wg() + maxDeltaWG_));
      if (min_wg <= alct.getKeyWG() and alct.getKeyWG() <= max_wg)
        isMatchedLayer2 = true;
    }

    // std::cout << "ALCT-GEM matching L1-L2: " << isMatchedLayer1 << " " << isMatchedLayer2 << std::endl;

    if (isMatchedLayer1 or isMatchedLayer2) {
      output.push_back(cl);
      if (isMatchedLayer1)
        output.back().set_matchingLayer1(true);
      if (isMatchedLayer2)
        output.back().set_matchingLayer2(true);
    }
  }
}

// match a CLCT to GEMInternalCluster by location
void CSCGEMMatcher::matchingClustersLoc(const CSCCLCTDigi& clct,
                                        const GEMInternalClusters& clusters,
                                        GEMInternalClusters& output,
                                        bool ignoreALCTGEMmatch,
                                        const CSCL1TPLookupTableME11ILT* lookupTableME11ILT,
                                        const CSCL1TPLookupTableME21ILT* lookupTableME21ILT) const {
  if (!clct.isValid() or clusters.empty())
    return;

  if (station_ == 1 and !enable_match_gem_me1a_ and !enable_match_gem_me1b_)
    return;

  const bool isME1a(station_ == 1 and clct.getKeyStrip() > CSCConstants::MAX_HALF_STRIP_ME1B);

  //determine window size
  unsigned eighthStripCut = isEven_ ? 4 * maxDeltaHsEven_ : 4 * maxDeltaHsOdd_;  // Cut in 1/8 = 4 * cut in 1/2

  for (const auto& cl : clusters) {
    // std::cout << "GEM cluster: " << cl << std::endl;
    // if (!ignoreALCTGEMmatch) std::cout << "IN CLCT-GEM => ALCT-GEM matching L1-L2: " << cl.isMatchingLayer1() << " " << cl.isMatchingLayer2() << std::endl;

    bool isMatchedLayer1 = false;
    bool isMatchedLayer2 = false;

    if (cl.id1().layer() == 1) {  // cluster has valid layer 1
      if ((station_ == 1 and enable_match_gem_me1a_ and
           ((isME1a and cl.roll1() == 8) or (!isME1a and cl.roll1() < 8))) or
          (station_ == 1 and !enable_match_gem_me1a_ and !isME1a) or (station_ == 2)) {
        constexpr bool isLayer2 = false;
        unsigned distanceES =
            abs(matchedClusterDistES(clct, cl, isLayer2, false, lookupTableME11ILT, lookupTableME21ILT));
        if (distanceES <= eighthStripCut)
          isMatchedLayer1 = true;
      }
    }
    if (cl.id2().layer() == 2) {  // cluster has valid layer 2
      if ((station_ == 1 and enable_match_gem_me1a_ and
           ((isME1a and cl.roll2() == 8) or (!isME1a and cl.roll2() < 8))) or
          (station_ == 1 and !enable_match_gem_me1a_ and !isME1a) or (station_ == 2)) {
        constexpr bool isLayer2 = true;
        unsigned distanceES =
            abs(matchedClusterDistES(clct, cl, isLayer2, false, lookupTableME11ILT, lookupTableME21ILT));
        if (distanceES <= eighthStripCut)
          isMatchedLayer2 = true;
      }
    }

    // std::cout << "CLCT-GEM matching L1-L2: " << isMatchedLayer1 << " " << isMatchedLayer2 << std::endl;

    if (((ignoreALCTGEMmatch or cl.isMatchingLayer1()) and isMatchedLayer1) or
        ((ignoreALCTGEMmatch or cl.isMatchingLayer2()) and isMatchedLayer2)) {
      output.push_back(cl);
      output.back().set_matchingLayer1(false);
      output.back().set_matchingLayer2(false);
      if ((ignoreALCTGEMmatch or cl.isMatchingLayer1()) and isMatchedLayer1)
        output.back().set_matchingLayer1(true);
      if ((ignoreALCTGEMmatch or cl.isMatchingLayer2()) and isMatchedLayer2)
        output.back().set_matchingLayer2(true);
    }
  }

  // Sorting of matching cluster prefers copads and ordering by smallest relative distance
  std::sort(
      output.begin(),
      output.end(),
      [clct, lookupTableME11ILT, lookupTableME21ILT, this](const GEMInternalCluster cl1,
                                                           const GEMInternalCluster cl2) -> bool {
        if (cl1.isCoincidence() and !cl2.isCoincidence())
          return cl1.isCoincidence();
        else if ((cl1.isCoincidence() and cl2.isCoincidence()) or (!cl1.isCoincidence() and !cl2.isCoincidence())) {
          bool cl1_isLayer2 = !cl1.isMatchingLayer1() and cl1.isMatchingLayer2();
          bool cl2_isLayer2 = !cl2.isMatchingLayer1() and cl2.isMatchingLayer2();
          unsigned cl1_distanceES =
              abs(matchedClusterDistES(clct, cl1, cl1_isLayer2, false, lookupTableME11ILT, lookupTableME21ILT));
          unsigned cl2_distanceES =
              abs(matchedClusterDistES(clct, cl2, cl2_isLayer2, false, lookupTableME11ILT, lookupTableME21ILT));
          return cl1_distanceES < cl2_distanceES;
        } else
          return false;
      });
}

void CSCGEMMatcher::matchingClustersLoc(const CSCALCTDigi& alct,
                                        const CSCCLCTDigi& clct,
                                        const GEMInternalClusters& clusters,
                                        const CSCL1TPLookupTableME11ILT* lookupTableME11ILT,
                                        const CSCL1TPLookupTableME21ILT* lookupTableME21ILT,
                                        GEMInternalClusters& output) const {
  // both need to be valid
  if (!alct.isValid() or !clct.isValid() or clusters.empty())
    return;

  // get the single matches
  bool ignoreALCTGEMmatch = false;
  GEMInternalClusters alctClusters;
  matchingClustersLoc(alct, clusters, alctClusters);
  matchingClustersLoc(clct, alctClusters, output, ignoreALCTGEMmatch, lookupTableME11ILT, lookupTableME21ILT);
}

//##############################################################
//  Ancillary functions: CLCT to GEM distance in eighth strips
//##############################################################

// calculate distance in eighth-strip units between CLCT and GEM, switch ForceTotal on to calculate total distance without slope extrapolation
int CSCGEMMatcher::matchedClusterDistES(const CSCCLCTDigi& clct,
                                        const GEMInternalCluster& cl,
                                        const bool isLayer2,
                                        const bool ForceTotal,
                                        const CSCL1TPLookupTableME11ILT* lookupTableME11ILT,
                                        const CSCL1TPLookupTableME21ILT* lookupTableME21ILT) const {
  const bool isME1a(station_ == 1 and clct.getKeyStrip() > CSCConstants::MAX_HALF_STRIP_ME1B);

  int cl_es = isME1a ? cl.getKeyStripME1a(8, isLayer2) : cl.getKeyStrip(8, isLayer2);

  int eighthStripDiff = cl_es - clct.getKeyStrip(8);

  if (matchCLCTpropagation_ and !ForceTotal) {  //modification of DeltaStrip by CLCT slope
    int SlopeShift = 0;
    uint16_t baseSlope = -1;
    baseSlope = mitigateSlopeByCosi_ ? mitigatedSlopeByConsistency(clct, lookupTableME11ILT, lookupTableME21ILT)
                                     : clct.getSlope();

    int clctSlope = pow(-1, clct.getBend()) * baseSlope;

    SlopeShift = CSCGEMSlopeCorrector(isME1a, clctSlope, isLayer2, lookupTableME11ILT, lookupTableME21ILT);
    eighthStripDiff -= SlopeShift;
  }

  return eighthStripDiff;
}

//##############################################################
//  Ancillary functions: CLCT COSI
//##############################################################

// function to determine CLCT consistency of slope indicator (COSI) and use it to mitigate slope according to LUT
uint16_t CSCGEMMatcher::mitigatedSlopeByConsistency(const CSCCLCTDigi& clct,
                                                    const CSCL1TPLookupTableME11ILT* lookupTableME11ILT,
                                                    const CSCL1TPLookupTableME21ILT* lookupTableME21ILT) const {
  const bool isME1a(station_ == 1 and clct.getKeyStrip() > CSCConstants::MAX_HALF_STRIP_ME1B);

  // extract hit values from CLCT hit matrix
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

  //Debugging
  //std::cout<<"CLCT Hits = "<<CLCTHits[0]<<", "<<CLCTHits[1]<<", "<<CLCTHits[2]<<", "<<CLCTHits[3]<<", "<<CLCTHits[4]<<", "<<CLCTHits[5]<<std::endl;

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
  //Debugging
  //std::cout<<"COSI = "<<cosi<<std::endl;

  //disambiguate cosi cases

  //extremely inconsistent track, deprecate slope
  if (cosi > 3)
    return 0;
  //consistent slope, do not change
  else if (cosi < 2)
    return clct.getSlope();
  //need to look up in table 2->1
  else if (cosi == 2) {
    if (station_ == 1) {
      if (isME1a) {
        if (chamber_ % 2 == 0)
          return lookupTableME11ILT->CSC_slope_cosi_2to1_L1_ME11a_even(clct.getSlope());
        else
          return lookupTableME11ILT->CSC_slope_cosi_2to1_L1_ME11a_odd(clct.getSlope());
      } else {
        if (chamber_ % 2 == 0)
          return lookupTableME11ILT->CSC_slope_cosi_2to1_L1_ME11b_even(clct.getSlope());
        else
          return lookupTableME11ILT->CSC_slope_cosi_2to1_L1_ME11b_odd(clct.getSlope());
      }
    } else {
      if (chamber_ % 2 == 0)
        return lookupTableME21ILT->CSC_slope_cosi_2to1_L1_ME21_even(clct.getSlope());
      else
        return lookupTableME21ILT->CSC_slope_cosi_2to1_L1_ME21_odd(clct.getSlope());
    }
  }
  //need to look up in table 3->1
  else if (cosi == 3) {
    if (station_ == 1) {
      if (isME1a) {
        if (chamber_ % 2 == 0)
          return lookupTableME11ILT->CSC_slope_cosi_3to1_L1_ME11a_even(clct.getSlope());
        else
          return lookupTableME11ILT->CSC_slope_cosi_3to1_L1_ME11a_odd(clct.getSlope());
      } else {
        if (chamber_ % 2 == 0)
          return lookupTableME11ILT->CSC_slope_cosi_3to1_L1_ME11b_even(clct.getSlope());
        else
          return lookupTableME11ILT->CSC_slope_cosi_3to1_L1_ME11b_odd(clct.getSlope());
      }
    } else {
      if (chamber_ % 2 == 0)
        return lookupTableME21ILT->CSC_slope_cosi_3to1_L1_ME21_even(clct.getSlope());
      else
        return lookupTableME21ILT->CSC_slope_cosi_3to1_L1_ME21_odd(clct.getSlope());
    }
  }
  //just to avoid compiler errors an error code
  else {
    return 999;
  }
}

//##############################################################
//  Ancillary functions: CLCT extrapolation towards GEM
//##############################################################

//function to correct expected GEM position in phi by CSC slope measurement
int CSCGEMMatcher::CSCGEMSlopeCorrector(bool isME1a,
                                        int cscSlope,
                                        bool isLayer2,
                                        const CSCL1TPLookupTableME11ILT* lookupTableME11ILT,
                                        const CSCL1TPLookupTableME21ILT* lookupTableME21ILT) const {
  int SlopeShift = 0;
  int SlopeSign = pow(-1, std::signbit(cscSlope));

  //account for slope mitigation by cosi, if opted-in
  if (mitigateSlopeByCosi_) {
    if (station_ == 1) {
      if (chamber_ % 2 == 0)
        SlopeShift = isME1a ? lookupTableME11ILT->CSC_slope_cosi_corr_L1_ME11a_even(std::abs(cscSlope))
                            : lookupTableME11ILT->CSC_slope_cosi_corr_L1_ME11b_even(std::abs(cscSlope));
      else
        SlopeShift = isME1a ? lookupTableME11ILT->CSC_slope_cosi_corr_L1_ME11a_odd(std::abs(cscSlope))
                            : lookupTableME11ILT->CSC_slope_cosi_corr_L1_ME11b_odd(std::abs(cscSlope));
    } else if (station_ == 2) {
      if (chamber_ % 2 == 0)
        SlopeShift = lookupTableME21ILT->CSC_slope_cosi_corr_L1_ME21_even(std::abs(cscSlope));
      else
        SlopeShift = lookupTableME21ILT->CSC_slope_cosi_corr_L1_ME21_odd(std::abs(cscSlope));
    }
  } else {  //account for slope without mitigation, if opted out
    if (station_ == 1) {
      if (!isLayer2) {
        if (chamber_ % 2 == 0)
          SlopeShift = isME1a ? lookupTableME11ILT->CSC_slope_corr_L1_ME11a_even(std::abs(cscSlope))
                              : lookupTableME11ILT->CSC_slope_corr_L1_ME11b_even(std::abs(cscSlope));
        else
          SlopeShift = isME1a ? lookupTableME11ILT->CSC_slope_corr_L1_ME11a_odd(std::abs(cscSlope))
                              : lookupTableME11ILT->CSC_slope_corr_L1_ME11b_odd(std::abs(cscSlope));
      } else {
        if (chamber_ % 2 == 0)
          SlopeShift = isME1a ? lookupTableME11ILT->CSC_slope_corr_L2_ME11a_even(std::abs(cscSlope))
                              : lookupTableME11ILT->CSC_slope_corr_L2_ME11b_even(std::abs(cscSlope));
        else
          SlopeShift = isME1a ? lookupTableME11ILT->CSC_slope_corr_L2_ME11a_odd(std::abs(cscSlope))
                              : lookupTableME11ILT->CSC_slope_corr_L2_ME11b_odd(std::abs(cscSlope));
      }
    } else if (station_ == 2) {
      if (!isLayer2) {
        if (chamber_ % 2 == 0)
          SlopeShift = lookupTableME21ILT->CSC_slope_corr_L1_ME21_even(std::abs(cscSlope));
        else
          SlopeShift = lookupTableME21ILT->CSC_slope_corr_L1_ME21_odd(std::abs(cscSlope));
      } else {
        if (chamber_ % 2 == 0)
          SlopeShift = lookupTableME21ILT->CSC_slope_corr_L2_ME21_even(std::abs(cscSlope));
        else
          SlopeShift = lookupTableME21ILT->CSC_slope_corr_L2_ME21_odd(std::abs(cscSlope));
      }
    }
  }
  return std::round(SlopeShift * SlopeSign);
}

//##############################################################
//  Ancillary functions: computation of slope corrected by GEM
//##############################################################

//function to replace the CLCT slope by the slope indicated by the strip difference between the CLCT and its matching GEM internal cluster
int CSCGEMMatcher::calculateGEMCSCBending(const CSCCLCTDigi& clct,
                                          const GEMInternalCluster& cluster,
                                          const CSCL1TPLookupTableME11ILT* lookupTableME11ILT,
                                          const CSCL1TPLookupTableME21ILT* lookupTableME21ILT) const {
  const bool isME1a(station_ == 1 and clct.getKeyStrip() > CSCConstants::MAX_HALF_STRIP_ME1B);

  bool isLayer2 = false;
  if (!cluster.isMatchingLayer1() and cluster.isMatchingLayer2())
    isLayer2 = true;

  //ME1a necessitates a different treatment because of a different strip numbering scheme and strip width
  const int SignedEighthStripDiff =
      matchedClusterDistES(clct, cluster, isLayer2, true, lookupTableME11ILT, lookupTableME21ILT);
  const unsigned eighthStripDiff = abs(SignedEighthStripDiff);  //LUTs consider only absolute change

  //use LUTs to determine absolute slope, default 0
  int slopeShift = 0;
  if (station_ == 2) {
    if (!isLayer2) {
      if (isEven_)
        slopeShift = lookupTableME21ILT->es_diff_slope_L1_ME21_even(eighthStripDiff);
      else
        slopeShift = lookupTableME21ILT->es_diff_slope_L1_ME21_odd(eighthStripDiff);
    } else {
      if (isEven_)
        slopeShift = lookupTableME21ILT->es_diff_slope_L2_ME21_even(eighthStripDiff);
      else
        slopeShift = lookupTableME21ILT->es_diff_slope_L2_ME21_odd(eighthStripDiff);
    }
  } else if (station_ == 1) {
    if (isME1a) {  //is in ME1a
      if (!isLayer2) {
        if (isEven_)
          slopeShift = lookupTableME11ILT->es_diff_slope_L1_ME11a_even(eighthStripDiff);
        else
          slopeShift = lookupTableME11ILT->es_diff_slope_L1_ME11a_odd(eighthStripDiff);
      } else {
        if (isEven_)
          slopeShift = lookupTableME11ILT->es_diff_slope_L2_ME11a_even(eighthStripDiff);
        else
          slopeShift = lookupTableME11ILT->es_diff_slope_L2_ME11a_odd(eighthStripDiff);
      }
    } else {
      if (!isLayer2) {
        if (isEven_)
          slopeShift = lookupTableME11ILT->es_diff_slope_L1_ME11b_even(eighthStripDiff);
        else
          slopeShift = lookupTableME11ILT->es_diff_slope_L1_ME11b_odd(eighthStripDiff);
      } else {
        if (isEven_)
          slopeShift = lookupTableME11ILT->es_diff_slope_L2_ME11b_even(eighthStripDiff);
        else
          slopeShift = lookupTableME11ILT->es_diff_slope_L2_ME11b_odd(eighthStripDiff);
      }
    }
  }

  //account for the sign of the difference
  slopeShift *= pow(-1, std::signbit(SignedEighthStripDiff));

  return slopeShift;
}
