#ifndef PHASE_2_L1_GCT_H_INCL
#define PHASE_2_L1_GCT_H_INCL

#include <iostream>
#include <ap_int.h>

// Output collections
#include "DataFormats/L1TCalorimeterPhase2/interface/CaloCrystalCluster.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/CaloTower.h"

#include "L1Trigger/L1CaloTrigger/interface/Phase2L1CaloEGammaUtils.h"

/*
 * Do proximity stitching and brems combination for POSITIVE eta, using GCTcard as input. Write to GCTcombinedClusters.
 * iStartingCard is 0 for even phi boundaries, and 1 for odd phi boundaries.
 * The first argument is for RCTcardEtaPos/Neg, which are arrays of RCTcard_t of size N_RCTCARDS_PHI. We pass by reference for the second argument to modify it.
 */
inline void p2eg::doProximityAndBremsStitching(const p2eg::RCTcard_t (&inputCards)[p2eg::N_RCTCARDS_PHI],
                                               p2eg::RCTcard_t (&outputCards)[p2eg::N_RCTCARDS_PHI],
                                               int iStartingCard,
                                               bool isPositiveEta) {
  for (int i = iStartingCard; i < p2eg::N_RCTCARDS_PHI - 1; i = i + 2) {
    for (int j = 0; j < p2eg::N_RCTGCT_FIBERS; j++) {
      for (int k = 0; k < p2eg::N_RCTCLUSTERS_FIBER; k++) {
        ap_uint<5> towerPhi1 = inputCards[i].RCTtoGCTfiber[j].RCTclusters[k].towPhi;

        ap_uint<15> crystalEta1 = inputCards[i].RCTtoGCTfiber[j].RCTclusters[k].towEta * 5 +
                                  inputCards[i].RCTtoGCTfiber[j].RCTclusters[k].crEta;
        ap_uint<15> crystalPhi1 = inputCards[i].RCTtoGCTfiber[j].RCTclusters[k].crPhi;

        for (int j1 = 0; j1 < p2eg::N_RCTGCT_FIBERS; j1++) {
          for (int k1 = 0; k1 < p2eg::N_RCTCLUSTERS_FIBER; k1++) {
            // For each pair, we check if cluster #1 is in the top card and if cluster #2 is in the bottom card.
            ap_uint<5> towerPhi2 = inputCards[i + 1].RCTtoGCTfiber[j1].RCTclusters[k1].towPhi;
            ap_uint<15> crystalEta2 = inputCards[i + 1].RCTtoGCTfiber[j1].RCTclusters[k1].towEta * 5 +
                                      inputCards[i + 1].RCTtoGCTfiber[j1].RCTclusters[k1].crEta;
            ap_uint<15> crystalPhi2 = inputCards[i + 1].RCTtoGCTfiber[j1].RCTclusters[k1].crPhi;

            // For positive eta, phi1 = 4, phi2 = 0 if cluster 1 is in the top card and cluster 2 is in the bottom card. For negative eta, the reverse is true.
            ap_uint<15> dPhi;
            dPhi = (isPositiveEta) ? ((5 - crystalPhi1) + crystalPhi2) : ((5 - crystalPhi2) + crystalPhi1);
            ap_uint<15> dEta;
            dEta = (crystalEta1 > crystalEta2) ? (crystalEta1 - crystalEta2) : (crystalEta2 - crystalEta1);

            ap_uint<12> one = inputCards[i].RCTtoGCTfiber[j].RCTclusters[k].et;
            ap_uint<12> two = inputCards[i + 1].RCTtoGCTfiber[j1].RCTclusters[k1].et;

            int topTowerPhi = (isPositiveEta) ? 3 : 0;
            int botTowerPhi = (isPositiveEta) ? 0 : 3;

            int topCrystalPhi = (isPositiveEta) ? 4 : 0;
            int botCrystalPhi = (isPositiveEta) ? 0 : 4;

            // First check for proximity stitching: clusters need to be exactly next to each other in crystals (across an RCT card boundary).
            // No requirement on their relative energy.
            if (towerPhi1 == topTowerPhi && crystalPhi1 == topCrystalPhi) {
              if (towerPhi2 == botTowerPhi && crystalPhi2 == botCrystalPhi) {
                if (dEta < 2) {
                  if (one > two) {
                    outputCards[i].RCTtoGCTfiber[j].RCTclusters[k].et = one + two;
                    outputCards[i + 1].RCTtoGCTfiber[j1].RCTclusters[k1].et = 0;
                  } else {
                    outputCards[i].RCTtoGCTfiber[j].RCTclusters[k].et = 0;
                    outputCards[i + 1].RCTtoGCTfiber[j1].RCTclusters[k1].et = one + two;
                  }
                }
              }
            }

            // Next, check for brems correction: clusters need to be next to each other in TOWERS only (not crystals) across an RCT card boundary.
            // And the sub-leading cluster must have a significant (>10%) energy of the larger cluster, in order for them to be combined.
            if (towerPhi1 == topTowerPhi) {
              if (towerPhi2 == botTowerPhi) {
                if ((dPhi <= 5) && (dEta < 2)) {
                  if (one > two) {
                    if (two >
                        (0.10 * one)) {  // Only stitch if the sub-leading cluster has a significant amount of energy
                      outputCards[i].RCTtoGCTfiber[j].RCTclusters[k].et = one + two;
                      outputCards[i + 1].RCTtoGCTfiber[j1].RCTclusters[k1].et = 0;
                    }
                  } else {
                    if (one >
                        (0.10 * two)) {  // Only stitch if the sub-leading cluster has a significant amount of energy
                      outputCards[i].RCTtoGCTfiber[j].RCTclusters[k].et = 0;
                      outputCards[i + 1].RCTtoGCTfiber[j1].RCTclusters[k1].et = one + two;
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

inline p2eg::GCTcard_t p2eg::getClustersCombined(const p2eg::GCTcard_t& GCTcard, unsigned int nGCTCard) {
  p2eg::GCTcard_t GCTcombinedClusters;

  // Initialize the output
  for (int i = 0; i < p2eg::N_RCTCARDS_PHI; i++) {
    for (int j = 0; j < p2eg::N_RCTGCT_FIBERS; j++) {
      for (int k = 0; k < p2eg::N_RCTCLUSTERS_FIBER; k++) {
        GCTcombinedClusters.RCTcardEtaPos[i].RCTtoGCTfiber[j].RCTclusters[k] =
            GCTcard.RCTcardEtaPos[i].RCTtoGCTfiber[j].RCTclusters[k];
        GCTcombinedClusters.RCTcardEtaNeg[i].RCTtoGCTfiber[j].RCTclusters[k] =
            GCTcard.RCTcardEtaNeg[i].RCTtoGCTfiber[j].RCTclusters[k];
      }
    }
  }
  bool isPositiveEta;
  int iStartingCard;

  // we will store new et in the GCTcombinedClusters, 0'ing lower clusters after stitching, dont need to care about other variables they stay the
  // same as input for now at least
  // we combine even phi boundaries positive eta. Start at card 0 (third argument), and tell the function this is positive eta (fourth argument)
  isPositiveEta = true;
  iStartingCard = 0;
  p2eg::doProximityAndBremsStitching(
      GCTcard.RCTcardEtaPos, GCTcombinedClusters.RCTcardEtaPos, iStartingCard, isPositiveEta);

  // now we combine odd phi boundaries positive eta
  isPositiveEta = true;
  iStartingCard = 1;
  p2eg::doProximityAndBremsStitching(
      GCTcard.RCTcardEtaPos, GCTcombinedClusters.RCTcardEtaPos, iStartingCard, isPositiveEta);

  // repeat above steps for NEGATIVE eta, even phi boundaries
  isPositiveEta = false;
  iStartingCard = 0;
  p2eg::doProximityAndBremsStitching(
      GCTcard.RCTcardEtaNeg, GCTcombinedClusters.RCTcardEtaNeg, iStartingCard, isPositiveEta);

  // lastly, NEGATIVE eta, odd phi boundaries
  isPositiveEta = false;
  iStartingCard = 1;
  p2eg::doProximityAndBremsStitching(
      GCTcard.RCTcardEtaNeg, GCTcombinedClusters.RCTcardEtaNeg, iStartingCard, isPositiveEta);

  // we need to store what we did before we start phi stitching
  p2eg::GCTcard_t GCTout;
  for (int i = 0; i < p2eg::N_RCTCARDS_PHI; i++) {
    for (int j = 0; j < p2eg::N_RCTGCT_FIBERS; j++) {
      for (int k = 0; k < p2eg::N_RCTCLUSTERS_FIBER; k++) {
        GCTout.RCTcardEtaPos[i].RCTtoGCTfiber[j].RCTclusters[k] =
            GCTcombinedClusters.RCTcardEtaPos[i].RCTtoGCTfiber[j].RCTclusters[k];
        GCTout.RCTcardEtaNeg[i].RCTtoGCTfiber[j].RCTclusters[k] =
            GCTcombinedClusters.RCTcardEtaNeg[i].RCTtoGCTfiber[j].RCTclusters[k];
      }
    }
  }

  // now we combine eta boundaries, just positive and negative eta
  // Uses RCTcardEtaPos and RCTcardEtaNeg
  for (int i = 0; i < p2eg::N_RCTCARDS_PHI; i++) {
    for (int j = 0; j < p2eg::N_RCTGCT_FIBERS; j++) {
      for (int k = 0; k < p2eg::N_RCTCLUSTERS_FIBER; k++) {
        ap_uint<15> phi1 = (i * 4 + GCTcard.RCTcardEtaPos[i].RCTtoGCTfiber[j].RCTclusters[k].towPhi) * 5 +
                           GCTcard.RCTcardEtaPos[i].RCTtoGCTfiber[j].RCTclusters[k].crPhi;
        ap_uint<15> eta1 = GCTcard.RCTcardEtaPos[i].RCTtoGCTfiber[j].RCTclusters[k].crEta;
        if (GCTcard.RCTcardEtaPos[i].RCTtoGCTfiber[j].RCTclusters[k].towEta == 0 && eta1 == 0) {
          for (int j1 = 0; j1 < p2eg::N_RCTGCT_FIBERS; j1++) {
            for (int k1 = 0; k1 < p2eg::N_RCTCLUSTERS_FIBER; k1++) {
              ap_uint<15> phi2 = (i * 4 + (3 - GCTcard.RCTcardEtaNeg[i].RCTtoGCTfiber[j1].RCTclusters[k1].towPhi)) * 5 +
                                 (4 - GCTcard.RCTcardEtaNeg[i].RCTtoGCTfiber[j1].RCTclusters[k1].crPhi);
              ap_uint<15> eta2 = GCTcard.RCTcardEtaNeg[i].RCTtoGCTfiber[j1].RCTclusters[k1].crEta;
              if (GCTcard.RCTcardEtaNeg[i].RCTtoGCTfiber[j1].RCTclusters[k1].towEta == 0 && eta2 == 0) {
                ap_uint<15> dPhi;
                dPhi = (phi1 > phi2) ? (phi1 - phi2) : (phi2 - phi1);
                if (dPhi < 2) {
                  ap_uint<12> one = GCTcombinedClusters.RCTcardEtaPos[i].RCTtoGCTfiber[j].RCTclusters[k].et;
                  ap_uint<12> two = GCTcombinedClusters.RCTcardEtaNeg[i].RCTtoGCTfiber[j1].RCTclusters[k1].et;
                  if (one > two) {
                    GCTout.RCTcardEtaPos[i].RCTtoGCTfiber[j].RCTclusters[k].et = one + two;
                    GCTout.RCTcardEtaNeg[i].RCTtoGCTfiber[j1].RCTclusters[k1].et = 0;
                  } else {
                    GCTout.RCTcardEtaPos[i].RCTtoGCTfiber[j].RCTclusters[k].et = 0;
                    GCTout.RCTcardEtaNeg[i].RCTtoGCTfiber[j1].RCTclusters[k1].et = one + two;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return GCTout;
}

/*
 * Populate a GCTinternal_t struct (consisting of 64 fibers, each fiber has clusters and towers) by converting RCT clusters and towers to GCT notation.
 */

inline p2eg::GCTinternal_t p2eg::getClustersTowers(const p2eg::GCTcard_t& GCTcard, unsigned int nGCTCard) {
  p2eg::GCTcard_t GCTcombinedClusters;
  p2eg::GCTinternal_t GCTout;

  // here we will stitch the clusters in phi and eta
  GCTcombinedClusters = p2eg::getClustersCombined(GCTcard, nGCTCard);

  // create internal structure of GCT card
  // we start from RCT card 0 - it is overlap with other GCT card and fill structure that we will use to send data to Correlator
  // we only need to care about clusters et in combinrdClusters, since the rest remains unchanged wrt input, the cluster that we set to 0
  // remain in the data at the same place , it will just get 0 et now
  // we need to code Positive and Negative Eta differently !  For negative Eta link 0 for each RCT
  // region becomes 3 in GCT output, the RCT card is rotated around 0:0 point of the card
  // First 16 fibers - positive Eta , second 16 - negative. Eta coded 0...16 and towEtaNeg = 0 or 1 for clusters ;
  // Phi is coded 0...15 , in case if whole card 0...33 and subdevision 1/5 in crPhi and crEta 0...4 for
  // position in tower
  //
  // towers are put in link starting from eta=0, the link number defines Eta negative or positive and Phi position of tower.
  for (int i = 0; i < p2eg::N_RCTCARDS_PHI; i++) {
    for (int j = 0; j < p2eg::N_RCTGCT_FIBERS; j++) {
      for (int k = 0; k < p2eg::N_RCTCLUSTERS_FIBER; k++) {
        bool isPositiveEta;
        // positive eta: initialize from RCT clusters in pos object
        isPositiveEta = true;
        GCTout.GCTCorrfiber[i * 4 + j].GCTclusters[k].initFromRCTCluster(
            i, isPositiveEta, GCTcombinedClusters.RCTcardEtaPos[i].RCTtoGCTfiber[j].RCTclusters[k]);
        // negative eta:  initialize from RCT clusters in neg object
        isPositiveEta = false;
        GCTout.GCTCorrfiber[i * 4 + (3 - j) + p2eg::N_GCTPOSITIVE_FIBERS].GCTclusters[k].initFromRCTCluster(
            i, isPositiveEta, GCTcombinedClusters.RCTcardEtaNeg[i].RCTtoGCTfiber[j].RCTclusters[k]);
      }
      for (int k = 0; k < N_RCTTOWERS_FIBER; k++) {
        GCTout.GCTCorrfiber[i * 4 + j].GCTtowers[k].initFromRCTTower(
            GCTcard.RCTcardEtaPos[i].RCTtoGCTfiber[j].RCTtowers[k]);  // pos eta
        GCTout.GCTCorrfiber[i * 4 + (3 - j) + p2eg::N_GCTPOSITIVE_FIBERS].GCTtowers[k].initFromRCTTower(
            GCTcard.RCTcardEtaNeg[i].RCTtoGCTfiber[j].RCTtowers[k]);  // neg eta
      }
    }
  }
  return GCTout;
}

/*
 * Return full towers with the tower energy (i.e. unclustered energy) and cluster energy added together.
 */
inline p2eg::GCTintTowers_t p2eg::getFullTowers(const p2eg::GCTinternal_t& GCTinternal) {
  p2eg::GCTintTowers_t GCTintTowers;

  // Positive eta
  for (int i = 0; i < p2eg::N_GCTPOSITIVE_FIBERS; i = i + 4) {
    for (int i1 = 0; i1 < 4; i1++) {
      for (int k = 0; k < p2eg::N_GCTTOWERS_FIBER; k++) {
        ap_uint<15> phi = i + i1;
        ap_uint<15> eta = p2eg::N_GCTETA / 2 + k;
        GCTintTowers.GCTtower[eta][phi].et = GCTinternal.GCTCorrfiber[phi].GCTtowers[k].et;
        GCTintTowers.GCTtower[eta][phi].hoe = GCTinternal.GCTCorrfiber[phi].GCTtowers[k].hoe;
        for (int ic1 = 0; ic1 < 4; ic1++) {
          for (int jc = 0; jc < p2eg::N_GCTCLUSTERS_FIBER; jc++) {
            ap_uint<15> eta1 = p2eg::N_GCTETA / 2 + GCTinternal.GCTCorrfiber[i + ic1].GCTclusters[jc].towEta;
            ap_uint<15> phi1 = GCTinternal.GCTCorrfiber[i + ic1].GCTclusters[jc].towPhi;
            if (eta == eta1 && phi == phi1) {
              GCTintTowers.GCTtower[eta][phi].et =
                  (GCTintTowers.GCTtower[eta][phi].et + GCTinternal.GCTCorrfiber[i + ic1].GCTclusters[jc].et);
            }
          }
        }
      }
    }
  }

  // Negative eta
  for (int i = p2eg::N_GCTPOSITIVE_FIBERS; i < p2eg::N_GCTINTERNAL_FIBERS; i = i + 4) {
    for (int i1 = 0; i1 < 4; i1++) {
      for (int k = 0; k < p2eg::N_GCTTOWERS_FIBER; k++) {
        ap_uint<15> eta = p2eg::N_GCTETA / 2 - k - 1;
        ap_uint<15> phi = i + i1 - p2eg::N_GCTPOSITIVE_FIBERS;
        GCTintTowers.GCTtower[eta][phi].et = GCTinternal.GCTCorrfiber[i + i1].GCTtowers[k].et;
        GCTintTowers.GCTtower[eta][phi].hoe = GCTinternal.GCTCorrfiber[i + i1].GCTtowers[k].hoe;
        for (int ic1 = 0; ic1 < 4; ic1++) {
          for (int jc = 0; jc < p2eg::N_GCTCLUSTERS_FIBER; jc++) {
            ap_uint<15> eta1 = p2eg::N_GCTETA / 2 - 1 - GCTinternal.GCTCorrfiber[i + ic1].GCTclusters[jc].towEta;
            ap_uint<15> phi1 = GCTinternal.GCTCorrfiber[i + ic1].GCTclusters[jc].towPhi;
            if (eta == eta1 && phi == phi1) {
              GCTintTowers.GCTtower[eta][phi].et =
                  (GCTintTowers.GCTtower[eta][phi].et + GCTinternal.GCTCorrfiber[i + ic1].GCTclusters[jc].et);
            }
          }
        }
      }
    }
  }

  return GCTintTowers;
}

/*
 * Fill CMSSW collections and correlator outputs, using GCTinternal.
 */
inline void p2eg::writeToCorrelatorAndGTOutputs(
    const p2eg::GCTinternal_t& GCTinternal,
    p2eg::GCTtoCorr_t& GCTtoCorrOutput,
    std::unique_ptr<l1tp2::CaloCrystalClusterCollection> const& gctClustersOutput,
    std::unique_ptr<l1tp2::CaloTowerCollection> const& gctTowersOutput,
    std::unique_ptr<l1t::EGammaBxCollection> const& gctEGammas,
    std::unique_ptr<l1tp2::DigitizedClusterCorrelatorCollection> const& gctDigitizedClustersCorrelator,
    std::unique_ptr<l1tp2::DigitizedTowerCorrelatorCollection> const& gctDigitizedTowersCorrelator,
    std::unique_ptr<l1tp2::DigitizedClusterGTCollection> const& gctDigitizedClustersGT,
    int nGCTCard,
    int fiberStart,
    int fiberEnd,
    int corrFiberIndexOffset,
    int corrTowPhiOffset = 4) {
  for (int i = fiberStart; i < fiberEnd; i++) {
    // In each fiber, first do clusters
    for (int k = 0; k < p2eg::N_GCTCLUSTERS_FIBER; k++) {
      // First do CMSSW cluster outputs
      p2eg::GCTcluster_t thisCluster = GCTinternal.GCTCorrfiber[i].GCTclusters[k];
      if (thisCluster.etFloat() > 0.0) {
        // Make l1tp2::CaloCrystalCluster
        gctClustersOutput->push_back(thisCluster.createCaloCrystalCluster());

        // Make l1t::EGamma
        int bx = 0;
        l1t::EGamma thisEGamma = thisCluster.createL1TEGamma();
        gctEGammas->push_back(bx, thisEGamma);
      }

      // Then the clusters to the correlator: all fields are the same with the exception of towPhi, which
      // needs to be subtracted by 4 because the output to correlator does NOT include the overlap region.
      GCTtoCorrOutput.GCTCorrfiber[i - corrFiberIndexOffset].GCTclusters[k] = thisCluster;
      GCTtoCorrOutput.GCTCorrfiber[i - corrFiberIndexOffset].GCTclusters[k].towPhi =
          (thisCluster.towPhi - corrTowPhiOffset);

      // Make l1tp2::DigitizedClusterCorrelator. The function needs corrTowPhiOffset to know the towPhi in the card excluding the overlap region.
      // The correlator clusters don't need to know the fiber offset.
      if (thisCluster.etFloat() > 0.0) {
        gctDigitizedClustersCorrelator->push_back(thisCluster.createDigitizedClusterCorrelator(corrTowPhiOffset));
      }

      // Make l1tp2::DigitizedClusterGT.
      if (thisCluster.etFloat() > 0.0) {
        bool isValid = true;
        gctDigitizedClustersGT->push_back(thisCluster.createDigitizedClusterGT(isValid));
      }
    }

    // Next do tower outputs
    for (int k = 0; k < p2eg::N_GCTTOWERS_FIBER; k++) {
      // First do CMSSW tower outputs
      p2eg::GCTtower_t thisTower = GCTinternal.GCTCorrfiber[i].GCTtowers[k];
      l1tp2::CaloTower thisL1CaloTower = thisTower.createCaloTowerFromFiberIdx(nGCTCard, i, k);
      gctTowersOutput->push_back(thisL1CaloTower);

      // Then the towers to the correlator. Note the same corrFiberIndexOffset as was done for the clusters
      GCTtoCorrOutput.GCTCorrfiber[i - corrFiberIndexOffset].GCTtowers[k] = thisTower;

      // For the collection, the three arguments are (1) the GCT card, (2) the fiber index in the GCT card (excluding the overlap region), and (3) the tower index in the fiber
      l1tp2::DigitizedTowerCorrelator thisDigitizedTowerCorrelator =
          thisTower.createDigitizedTowerCorrelator(nGCTCard, i - corrFiberIndexOffset, k);
      gctDigitizedTowersCorrelator->push_back(thisDigitizedTowerCorrelator);
    }
  }
}

/* 
 * algo_top: First two arguments are the same as in the original firmware.
 * nGCTCard is 0, 1, or 2 (needed for getting the cluster real eta/phis for CMSSW collections).
 * gctClusters is the CMSSW-style output collection of clusters.
 * gctTowers is the CMSSW-style output collection of towers.
 */

inline void p2eg::algo_top(
    const p2eg::GCTcard_t& GCTcard,
    p2eg::GCTtoCorr_t& GCTtoCorr,
    unsigned int nGCTCard,
    std::unique_ptr<l1tp2::CaloCrystalClusterCollection> const& gctClusters,
    std::unique_ptr<l1tp2::CaloTowerCollection> const& gctTowers,
    std::unique_ptr<l1tp2::CaloTowerCollection> const& gctFullTowers,
    std::unique_ptr<l1t::EGammaBxCollection> const& gctEGammas,
    std::unique_ptr<l1tp2::DigitizedClusterCorrelatorCollection> const& gctDigitizedClustersCorrelator,
    std::unique_ptr<l1tp2::DigitizedTowerCorrelatorCollection> const& gctDigitizedTowersCorrelator,
    std::unique_ptr<l1tp2::DigitizedClusterGTCollection> const& gctDigitizedClustersGT,
    l1tp2::ParametricCalibration calib_) {
  //-------------------------//
  // Initialize the GCT area
  //-------------------------//
  p2eg::GCTinternal_t GCTinternal = p2eg::getClustersTowers(GCTcard, nGCTCard);

  //------------------------------------------------//
  // Combine towers and clusters to get full towers
  //------------------------------------------------//
  p2eg::GCTintTowers_t GCTintTowers = p2eg::getFullTowers(GCTinternal);

  //---------------------------//
  // Compute cluster isolation
  //--------------------------//
  GCTinternal.computeClusterIsolationInPlace(nGCTCard);
  GCTinternal.setIsolationInfo();

  //-----------------------------------------------------------------------------------------------------------------------//
  // Output to correlator and CMSSW collections.
  // For positive eta, skip overlap region, i.e. fibers i = 0, 1, 2, 3, and i = 28, 29, 30, 31.
  // For negative eta, skip overlap region, i.e. fibers 32, 33, 34, 35, and 61, 62, 63, 64.
  //-----------------------------------------------------------------------------------------------------------------------//
  int posEtaFiberStart = p2eg::N_RCTGCT_FIBERS;  // 4, since there are 4 fibers in one RCT card
  int posEtaFiberEnd = (p2eg::N_GCTPOSITIVE_FIBERS - p2eg::N_RCTGCT_FIBERS);
  int negEtaFiberStart = (p2eg::N_GCTPOSITIVE_FIBERS + p2eg::N_RCTGCT_FIBERS);
  int negEtaFiberEnd =
      (p2eg::N_GCTINTERNAL_FIBERS - p2eg::N_RCTGCT_FIBERS);  // first term is number of gct internal fibers

  // When indexing into the correlator output, note that the output to correlator does NOT include the overlap region,
  // so fiber number "i" in GCT is not fiber "i" in the correlator output, it's reduced by 4 in positive eta, and 12 in negative eta.
  // (4 because we are skipping one RCT card in positive eta, and 12 because we are skipping three RCT cards in negative eta)
  int posEtaCorrelatorFiberIndexOffset = 4;
  int negEtaCorrelatorFiberIndexOffset = 12;

  // The offset in the actual towPhi value is going to be the same in pos/neg eta; shifted down by 4 due to no overlap region
  int correlatorTowPhiOffset = 4;

  // Positive eta
  p2eg::writeToCorrelatorAndGTOutputs(GCTinternal,
                                      GCTtoCorr,
                                      gctClusters,
                                      gctTowers,
                                      gctEGammas,
                                      gctDigitizedClustersCorrelator,
                                      gctDigitizedTowersCorrelator,
                                      gctDigitizedClustersGT,
                                      nGCTCard,
                                      posEtaFiberStart,
                                      posEtaFiberEnd,
                                      posEtaCorrelatorFiberIndexOffset,
                                      correlatorTowPhiOffset);
  // Negative eta
  p2eg::writeToCorrelatorAndGTOutputs(GCTinternal,
                                      GCTtoCorr,
                                      gctClusters,
                                      gctTowers,
                                      gctEGammas,
                                      gctDigitizedClustersCorrelator,
                                      gctDigitizedTowersCorrelator,
                                      gctDigitizedClustersGT,
                                      nGCTCard,
                                      negEtaFiberStart,
                                      negEtaFiberEnd,
                                      negEtaCorrelatorFiberIndexOffset,
                                      correlatorTowPhiOffset);

  //-----------------------------------------------------------------------------------------------------------------------//
  // CMSSW outputs for GCT Full Towers (clusters + towers) output for PFClusters.
  //-----------------------------------------------------------------------------------------------------------------------//
  GCTintTowers.writeToPFOutput(nGCTCard, gctFullTowers);
}

#endif
