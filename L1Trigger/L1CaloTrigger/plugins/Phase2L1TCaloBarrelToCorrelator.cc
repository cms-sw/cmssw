// -*- C++ -*-
//
// Package:    L1Trigger/L1CaloTrigger
// Class:      L1TCaloBarrelToCorrelator
//
/*
 Description: Creates digitized EGamma and ParticleFlow clusters to be sent to correlator. 

 Implementation: To be run together with Phase2L1CaloEGammaEmulator.
*/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/L1TCalorimeterPhase2/interface/CaloPFCluster.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/DigitizedClusterCorrelator.h"

#include "DataFormats/L1TCalorimeterPhase2/interface/GCTEmDigiCluster.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/GCTHadDigiCluster.h"

#include <ap_int.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cstdio>
#include "L1Trigger/L1CaloTrigger/interface/Phase2L1CaloBarrelToCorrelator.h"
#include "L1Trigger/L1CaloTrigger/interface/Phase2L1CaloEGammaUtils.h"

//
// class declaration
//

class Phase2GCTBarrelToCorrelatorLayer1 : public edm::stream::EDProducer<> {
public:
  explicit Phase2GCTBarrelToCorrelatorLayer1(const edm::ParameterSet&);
  ~Phase2GCTBarrelToCorrelatorLayer1() override = default;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  const edm::EDGetTokenT<l1tp2::CaloCrystalClusterCollection> gctClusterSrc_;
  const edm::EDGetTokenT<l1tp2::DigitizedClusterCorrelatorCollection> digiInputClusterSrc_;
  const edm::EDGetTokenT<l1tp2::CaloPFClusterCollection> caloPFClustersSrc_;
};

//
// constructors and destructor
//
Phase2GCTBarrelToCorrelatorLayer1::Phase2GCTBarrelToCorrelatorLayer1(const edm::ParameterSet& iConfig)
    : gctClusterSrc_(
          consumes<l1tp2::CaloCrystalClusterCollection>(iConfig.getParameter<edm::InputTag>("gctClustersInput"))),
      digiInputClusterSrc_(consumes<l1tp2::DigitizedClusterCorrelatorCollection>(
          iConfig.getParameter<edm::InputTag>("gctDigiClustersInput"))),
      caloPFClustersSrc_(
          consumes<l1tp2::CaloPFClusterCollection>(iConfig.getParameter<edm::InputTag>("gctPFclusters"))) {
  produces<l1tp2::GCTEmDigiClusterCollection>("GCTEmDigiClusters");
  produces<l1tp2::GCTHadDigiClusterCollection>("GCTHadDigiClusters");
}

void Phase2GCTBarrelToCorrelatorLayer1::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  //***************************************************//
  // Get the GCT digitized clusters and PF clusters
  //***************************************************//
  edm::Handle<l1tp2::CaloCrystalClusterCollection> inputGCTClusters;
  iEvent.getByToken(gctClusterSrc_, inputGCTClusters);

  edm::Handle<l1tp2::DigitizedClusterCorrelatorCollection> inputGCTDigiClusters;
  iEvent.getByToken(digiInputClusterSrc_, inputGCTDigiClusters);

  edm::Handle<l1tp2::CaloPFClusterCollection> inputPFClusters;
  iEvent.getByToken(caloPFClustersSrc_, inputPFClusters);

  //***************************************************//
  // Initialize outputs
  //***************************************************//

  // Em digi cluster output
  auto outputEmClusters = std::make_unique<l1tp2::GCTEmDigiClusterCollection>();
  // Had digi cluster output
  auto outputHadClusters = std::make_unique<l1tp2::GCTHadDigiClusterCollection>();

  // EG Clusters output by GCT SLR (duplicates included)
  l1tp2::GCTEmDigiClusterLink out_eg_GCT1_SLR1_posEta;
  l1tp2::GCTEmDigiClusterLink out_eg_GCT1_SLR1_negEta;
  l1tp2::GCTEmDigiClusterLink out_eg_GCT1_SLR3_posEta;
  l1tp2::GCTEmDigiClusterLink out_eg_GCT1_SLR3_negEta;
  l1tp2::GCTEmDigiClusterLink out_eg_GCT2_SLR1_posEta;
  l1tp2::GCTEmDigiClusterLink out_eg_GCT2_SLR1_negEta;
  l1tp2::GCTEmDigiClusterLink out_eg_GCT2_SLR3_posEta;
  l1tp2::GCTEmDigiClusterLink out_eg_GCT2_SLR3_negEta;
  l1tp2::GCTEmDigiClusterLink out_eg_GCT3_SLR1_posEta;
  l1tp2::GCTEmDigiClusterLink out_eg_GCT3_SLR1_negEta;
  l1tp2::GCTEmDigiClusterLink out_eg_GCT3_SLR3_posEta;
  l1tp2::GCTEmDigiClusterLink out_eg_GCT3_SLR3_negEta;

  // Temporary arrays used to represent the four RCT cards in one SLR one side of eta (positive or eta)
  l1tp2::GCTEmDigiClusterLink buffer_eg_GCT1_SLR1_posEta[4];
  l1tp2::GCTEmDigiClusterLink buffer_eg_GCT1_SLR1_negEta[4];
  l1tp2::GCTEmDigiClusterLink buffer_eg_GCT1_SLR3_posEta[4];
  l1tp2::GCTEmDigiClusterLink buffer_eg_GCT1_SLR3_negEta[4];
  l1tp2::GCTEmDigiClusterLink buffer_eg_GCT2_SLR1_posEta[4];
  l1tp2::GCTEmDigiClusterLink buffer_eg_GCT2_SLR1_negEta[4];
  l1tp2::GCTEmDigiClusterLink buffer_eg_GCT2_SLR3_posEta[4];
  l1tp2::GCTEmDigiClusterLink buffer_eg_GCT2_SLR3_negEta[4];
  l1tp2::GCTEmDigiClusterLink buffer_eg_GCT3_SLR1_posEta[4];
  l1tp2::GCTEmDigiClusterLink buffer_eg_GCT3_SLR1_negEta[4];
  l1tp2::GCTEmDigiClusterLink buffer_eg_GCT3_SLR3_posEta[4];
  l1tp2::GCTEmDigiClusterLink buffer_eg_GCT3_SLR3_negEta[4];

  // PF Clusters output by GCT SLR (duplicates included)
  l1tp2::GCTHadDigiClusterLink out_had_GCT1_SLR1_posEta;
  l1tp2::GCTHadDigiClusterLink out_had_GCT1_SLR1_negEta;
  l1tp2::GCTHadDigiClusterLink out_had_GCT1_SLR3_posEta;
  l1tp2::GCTHadDigiClusterLink out_had_GCT1_SLR3_negEta;
  l1tp2::GCTHadDigiClusterLink out_had_GCT2_SLR1_posEta;
  l1tp2::GCTHadDigiClusterLink out_had_GCT2_SLR1_negEta;
  l1tp2::GCTHadDigiClusterLink out_had_GCT2_SLR3_posEta;
  l1tp2::GCTHadDigiClusterLink out_had_GCT2_SLR3_negEta;
  l1tp2::GCTHadDigiClusterLink out_had_GCT3_SLR1_posEta;
  l1tp2::GCTHadDigiClusterLink out_had_GCT3_SLR1_negEta;
  l1tp2::GCTHadDigiClusterLink out_had_GCT3_SLR3_posEta;
  l1tp2::GCTHadDigiClusterLink out_had_GCT3_SLR3_negEta;

  // Temporary arrays used to represent the four RCT cards in one SLR one side of eta (positive or eta)
  l1tp2::GCTHadDigiClusterLink buffer_had_GCT1_SLR1_posEta[4];
  l1tp2::GCTHadDigiClusterLink buffer_had_GCT1_SLR1_negEta[4];
  l1tp2::GCTHadDigiClusterLink buffer_had_GCT1_SLR3_posEta[4];
  l1tp2::GCTHadDigiClusterLink buffer_had_GCT1_SLR3_negEta[4];
  l1tp2::GCTHadDigiClusterLink buffer_had_GCT2_SLR1_posEta[4];
  l1tp2::GCTHadDigiClusterLink buffer_had_GCT2_SLR1_negEta[4];
  l1tp2::GCTHadDigiClusterLink buffer_had_GCT2_SLR3_posEta[4];
  l1tp2::GCTHadDigiClusterLink buffer_had_GCT2_SLR3_negEta[4];
  l1tp2::GCTHadDigiClusterLink buffer_had_GCT3_SLR1_posEta[4];
  l1tp2::GCTHadDigiClusterLink buffer_had_GCT3_SLR1_negEta[4];
  l1tp2::GCTHadDigiClusterLink buffer_had_GCT3_SLR3_posEta[4];
  l1tp2::GCTHadDigiClusterLink buffer_had_GCT3_SLR3_negEta[4];

  //***************************************************//
  // Loop over the regions: in order: GCT1 SLR1, GCT1 SLR3, GCT2 SLR1, GCT2 SLR3, GCT3 SLR1, GCT3SLR3
  //***************************************************//

  const int nRegions = 6;
  float regionCentersInDegrees[nRegions] = {10.0, 70.0, 130.0, -170.0, -110.0, -50.0};

  for (int iRegion = 0; iRegion < nRegions; iRegion++) {
    // EM digi clusters
    for (size_t iCluster = 0; iCluster < inputGCTDigiClusters->size(); ++iCluster) {
      l1tp2::DigitizedClusterCorrelator clusterIn = inputGCTDigiClusters->at(iCluster);

      // Check if this cluster falls into each SLR region, i.e. if the cluster is within 120/2 = 60 degrees of the center of the SLR in phi
      float clusterRealPhiAsDegree = clusterIn.realPhi() * 180 / M_PI;
      float phiDifference = p2eg::deltaPhiInDegrees(clusterRealPhiAsDegree, regionCentersInDegrees[iRegion]);
      if (std::abs(phiDifference) < (p2eg::PHI_RANGE_PER_SLR_DEGREES / 2)) {
        // Go from real phi to an index in the SLR
        // The crystal directly above the region center in phi, is iPhi 0. The crystal directly below the region center in phi, is iPhi -1.
        int iPhiCrystalDifference = std::floor(phiDifference);

        // For eta, the eta is already digitized, just needs to be converted from [0, +2*17*5) to [-17*5, +17*5)
        int temp_iEta_signed = clusterIn.eta() - (p2eg::CRYSTALS_IN_TOWER_ETA * p2eg::n_towers_per_link);

        // Default value: for clusters in positive eta, values go from 0, 1, 2, 3, 4
        int iEta = temp_iEta_signed;
        // If cluster is in negative eta, instead of from -5, -4, -3, -2, -1, we want 4, 3, 2, 1, 0
        if (temp_iEta_signed < 0) {
          // If in negative eta, convert to an absolute value, with 0 being the crystal nearest real eta = 0
          iEta = std::abs(temp_iEta_signed + 1);
        }

        // Initialize the new cluster and set the edm::Ref pointing to the underlying float
        l1tp2::GCTEmDigiCluster clusterOut = l1tp2::GCTEmDigiCluster(clusterIn.pt(),
                                                                     iEta,
                                                                     iPhiCrystalDifference,
                                                                     clusterIn.hoe(),
                                                                     clusterIn.hoeFlag(),
                                                                     clusterIn.iso(),
                                                                     clusterIn.isoFlags(),
                                                                     clusterIn.fb(),
                                                                     clusterIn.timing(),
                                                                     clusterIn.shapeFlags(),
                                                                     clusterIn.brems());
        // there is a 1-to-1 mapping between the original float clusters and the first step of digitization, so we can build a ref to the same cluster
        edm::Ref<l1tp2::CaloCrystalClusterCollection> thisRef(inputGCTClusters, iCluster);
        clusterOut.setRef(thisRef);
        edm::Ref<l1tp2::DigitizedClusterCorrelatorCollection> thisDigiRef(inputGCTDigiClusters, iCluster);
        clusterOut.setDigiRef(thisDigiRef);

        // Check which RCT card this falls into, ordered 0, 1, 2, 3 counting from the most negative phi (real phi or iPhi) to the most positive
        // so RCT card 0 is -60 to -30 degrees in phi from the center, RCT card 1 is -30 to 0 degrees in phi from the center, RCT card 2 is 0 to +30 degrees in phi from the center, RCT card 3 is +30 to +60 degrees in phi from the center
        int whichRCTcard = 0;
        if (phiDifference < -30) {
          whichRCTcard = 0;
        } else if (phiDifference < 0) {
          whichRCTcard = 1;
        } else if (phiDifference < 30) {
          whichRCTcard = 2;
        } else {
          whichRCTcard = 3;
        }

        if (iRegion == 0) {
          if (temp_iEta_signed < 0) {
            buffer_eg_GCT1_SLR1_negEta[whichRCTcard].push_back(clusterOut);
          } else {
            buffer_eg_GCT1_SLR1_posEta[whichRCTcard].push_back(clusterOut);
          }
        } else if (iRegion == 1) {
          if (temp_iEta_signed < 0) {
            buffer_eg_GCT1_SLR3_negEta[whichRCTcard].push_back(clusterOut);
          } else {
            buffer_eg_GCT1_SLR3_posEta[whichRCTcard].push_back(clusterOut);
          }
        } else if (iRegion == 2) {
          if (temp_iEta_signed < 0) {
            buffer_eg_GCT2_SLR1_negEta[whichRCTcard].push_back(clusterOut);
          } else {
            buffer_eg_GCT2_SLR1_posEta[whichRCTcard].push_back(clusterOut);
          }
        } else if (iRegion == 3) {
          if (temp_iEta_signed < 0) {
            buffer_eg_GCT2_SLR3_negEta[whichRCTcard].push_back(clusterOut);
          } else {
            buffer_eg_GCT2_SLR3_posEta[whichRCTcard].push_back(clusterOut);
          }
        } else if (iRegion == 4) {
          if (temp_iEta_signed < 0) {
            buffer_eg_GCT3_SLR1_negEta[whichRCTcard].push_back(clusterOut);
          } else {
            buffer_eg_GCT3_SLR1_posEta[whichRCTcard].push_back(clusterOut);
          }
        } else if (iRegion == 5) {
          if (temp_iEta_signed < 0) {
            buffer_eg_GCT3_SLR3_negEta[whichRCTcard].push_back(clusterOut);
          } else {
            buffer_eg_GCT3_SLR3_posEta[whichRCTcard].push_back(clusterOut);
          }
        }
      }
    }

    // Repeat for PF Clusters
    for (size_t iCluster = 0; iCluster < inputPFClusters->size(); ++iCluster) {
      l1tp2::CaloPFCluster pfIn = inputPFClusters->at(iCluster);

      // Skip zero-energy clusters
      if (pfIn.clusterEt() == 0)
        continue;

      // Check if this cluster falls into each GCT card
      float clusterRealPhiAsDegree = pfIn.clusterPhi() * 180 / M_PI;
      float differenceInPhi = p2eg::deltaPhiInDegrees(clusterRealPhiAsDegree, regionCentersInDegrees[iRegion]);
      if (std::abs(differenceInPhi) < (p2eg::PHI_RANGE_PER_SLR_DEGREES / 2)) {
        // Go from real phi to an index in the SLR
        // Calculate the distance in phi from the center of the region
        float phiDifference = clusterRealPhiAsDegree - regionCentersInDegrees[iRegion];
        int iPhiCrystalDifference = std::floor(phiDifference);

        // For PFClusters, the method clusterEta returns a float, so we need to digitize this
        float eta_LSB = p2eg::ECAL_eta_range / (p2eg::N_GCTTOWERS_FIBER * p2eg::CRYSTALS_IN_TOWER_ETA);
        int temp_iEta_signed = std::floor(pfIn.clusterEta() / eta_LSB);
        // Default value (for positive eta)
        int iEta = temp_iEta_signed;
        // If cluster is in negative eta, instead of from -5, -4, -3, -2, -1 we want 4, 3, 2, 1, 0
        if (temp_iEta_signed < 0) {
          // If in negative eta, convert to an absolute value, with 0 being the crystal nearest real eta = 0
          iEta = std::abs(temp_iEta_signed + 1);
        }

        // Initialize the new cluster
        l1tp2::GCTHadDigiCluster pfOut =
            l1tp2::GCTHadDigiCluster(pfIn.clusterEt() / p2eg::ECAL_LSB,  // convert to integer
                                     iEta,
                                     iPhiCrystalDifference,
                                     0  // no HoE value in PF Cluster
            );
        pfOut.setRef(edm::Ref<l1tp2::CaloPFClusterCollection>(inputPFClusters, iCluster));

        // Check which RCT card this falls into, ordered 0, 1, 2, 3 counting from the most negative phi (real phi or iPhi) to the most positive
        // so RCT card 0 is -60 to -30 degrees in phi from the center, RCT card 1 is -30 to 0 degrees in phi from the center, RCT card 2 is 0 to +30 degrees in phi from the center, RCT card 3 is +30 to +60 degrees in phi from the center
        int whichRCTcard = 0;
        if (phiDifference < -30) {
          whichRCTcard = 0;
        } else if (phiDifference < 0) {
          whichRCTcard = 1;
        } else if (phiDifference < 30) {
          whichRCTcard = 2;
        } else {
          whichRCTcard = 3;
        }

        if (iRegion == 0) {
          if (temp_iEta_signed < 0) {
            buffer_had_GCT1_SLR1_negEta[whichRCTcard].push_back(pfOut);
          } else {
            buffer_had_GCT1_SLR1_posEta[whichRCTcard].push_back(pfOut);
          }
        } else if (iRegion == 1) {
          if (temp_iEta_signed < 0) {
            buffer_had_GCT1_SLR3_negEta[whichRCTcard].push_back(pfOut);
          } else {
            buffer_had_GCT1_SLR3_posEta[whichRCTcard].push_back(pfOut);
          }
        } else if (iRegion == 2) {
          if (temp_iEta_signed < 0) {
            buffer_had_GCT2_SLR1_negEta[whichRCTcard].push_back(pfOut);
          } else {
            buffer_had_GCT2_SLR1_posEta[whichRCTcard].push_back(pfOut);
          }
        } else if (iRegion == 3) {
          if (temp_iEta_signed < 0) {
            buffer_had_GCT2_SLR3_negEta[whichRCTcard].push_back(pfOut);
          } else {
            buffer_had_GCT2_SLR3_posEta[whichRCTcard].push_back(pfOut);
          }
        } else if (iRegion == 4) {
          if (temp_iEta_signed < 0) {
            buffer_had_GCT3_SLR1_negEta[whichRCTcard].push_back(pfOut);
          } else {
            buffer_had_GCT3_SLR1_posEta[whichRCTcard].push_back(pfOut);
          }
        } else if (iRegion == 5) {
          if (temp_iEta_signed < 0) {
            buffer_had_GCT3_SLR3_negEta[whichRCTcard].push_back(pfOut);
          } else {
            buffer_had_GCT3_SLR3_posEta[whichRCTcard].push_back(pfOut);
          }
        }
      }
    }
  }

  // Within each RCT card, sort the egamma clusters in descending pT order and add zero-padding
  for (int iRCT = 0; iRCT < 4; iRCT++) {
    p2eg::sortAndPadSLR(buffer_eg_GCT1_SLR1_negEta[iRCT], p2eg::N_EG_CLUSTERS_PER_RCT_CARD);
    p2eg::sortAndPadSLR(buffer_eg_GCT1_SLR3_negEta[iRCT], p2eg::N_EG_CLUSTERS_PER_RCT_CARD);
    p2eg::sortAndPadSLR(buffer_eg_GCT2_SLR1_negEta[iRCT], p2eg::N_EG_CLUSTERS_PER_RCT_CARD);
    p2eg::sortAndPadSLR(buffer_eg_GCT2_SLR3_negEta[iRCT], p2eg::N_EG_CLUSTERS_PER_RCT_CARD);
    p2eg::sortAndPadSLR(buffer_eg_GCT3_SLR1_negEta[iRCT], p2eg::N_EG_CLUSTERS_PER_RCT_CARD);
    p2eg::sortAndPadSLR(buffer_eg_GCT3_SLR3_negEta[iRCT], p2eg::N_EG_CLUSTERS_PER_RCT_CARD);
    p2eg::sortAndPadSLR(buffer_eg_GCT1_SLR1_posEta[iRCT], p2eg::N_EG_CLUSTERS_PER_RCT_CARD);
    p2eg::sortAndPadSLR(buffer_eg_GCT1_SLR3_posEta[iRCT], p2eg::N_EG_CLUSTERS_PER_RCT_CARD);
    p2eg::sortAndPadSLR(buffer_eg_GCT2_SLR1_posEta[iRCT], p2eg::N_EG_CLUSTERS_PER_RCT_CARD);
    p2eg::sortAndPadSLR(buffer_eg_GCT2_SLR3_posEta[iRCT], p2eg::N_EG_CLUSTERS_PER_RCT_CARD);
    p2eg::sortAndPadSLR(buffer_eg_GCT3_SLR1_posEta[iRCT], p2eg::N_EG_CLUSTERS_PER_RCT_CARD);
    p2eg::sortAndPadSLR(buffer_eg_GCT3_SLR3_posEta[iRCT], p2eg::N_EG_CLUSTERS_PER_RCT_CARD);
  }

  // Then build the container for each egamma SLR, by pushing back, in order, the four RCT cards (starting from most negative phi to most positive phi)
  for (int iRCT = 0; iRCT < 4; iRCT++) {
    for (int iCluster = 0; iCluster < p2eg::N_EG_CLUSTERS_PER_RCT_CARD; iCluster++) {
      // outer loop is over RCT cards, inner loop is over the clusters in each RCT card
      out_eg_GCT1_SLR1_posEta.push_back(buffer_eg_GCT1_SLR1_posEta[iRCT][iCluster]);
      out_eg_GCT1_SLR1_negEta.push_back(buffer_eg_GCT1_SLR1_negEta[iRCT][iCluster]);
      out_eg_GCT1_SLR3_posEta.push_back(buffer_eg_GCT1_SLR3_posEta[iRCT][iCluster]);
      out_eg_GCT1_SLR3_negEta.push_back(buffer_eg_GCT1_SLR3_negEta[iRCT][iCluster]);
      out_eg_GCT2_SLR1_posEta.push_back(buffer_eg_GCT2_SLR1_posEta[iRCT][iCluster]);
      out_eg_GCT2_SLR1_negEta.push_back(buffer_eg_GCT2_SLR1_negEta[iRCT][iCluster]);
      out_eg_GCT2_SLR3_posEta.push_back(buffer_eg_GCT2_SLR3_posEta[iRCT][iCluster]);
      out_eg_GCT2_SLR3_negEta.push_back(buffer_eg_GCT2_SLR3_negEta[iRCT][iCluster]);
      out_eg_GCT3_SLR1_posEta.push_back(buffer_eg_GCT3_SLR1_posEta[iRCT][iCluster]);
      out_eg_GCT3_SLR1_negEta.push_back(buffer_eg_GCT3_SLR1_negEta[iRCT][iCluster]);
      out_eg_GCT3_SLR3_posEta.push_back(buffer_eg_GCT3_SLR3_posEta[iRCT][iCluster]);
      out_eg_GCT3_SLR3_negEta.push_back(buffer_eg_GCT3_SLR3_negEta[iRCT][iCluster]);
    }
  }

  // Repeat for PF: Within each RCT card, sort the PF clusters in descending pT order and add zero-padding
  for (int iRCT = 0; iRCT < 4; iRCT++) {
    p2eg::sortAndPadSLR(buffer_had_GCT1_SLR1_negEta[iRCT], p2eg::N_PF_CLUSTERS_PER_RCT_CARD);
    p2eg::sortAndPadSLR(buffer_had_GCT1_SLR3_negEta[iRCT], p2eg::N_PF_CLUSTERS_PER_RCT_CARD);
    p2eg::sortAndPadSLR(buffer_had_GCT2_SLR1_negEta[iRCT], p2eg::N_PF_CLUSTERS_PER_RCT_CARD);
    p2eg::sortAndPadSLR(buffer_had_GCT2_SLR3_negEta[iRCT], p2eg::N_PF_CLUSTERS_PER_RCT_CARD);
    p2eg::sortAndPadSLR(buffer_had_GCT3_SLR1_negEta[iRCT], p2eg::N_PF_CLUSTERS_PER_RCT_CARD);
    p2eg::sortAndPadSLR(buffer_had_GCT3_SLR3_negEta[iRCT], p2eg::N_PF_CLUSTERS_PER_RCT_CARD);
    p2eg::sortAndPadSLR(buffer_had_GCT1_SLR1_posEta[iRCT], p2eg::N_PF_CLUSTERS_PER_RCT_CARD);
    p2eg::sortAndPadSLR(buffer_had_GCT1_SLR3_posEta[iRCT], p2eg::N_PF_CLUSTERS_PER_RCT_CARD);
    p2eg::sortAndPadSLR(buffer_had_GCT2_SLR1_posEta[iRCT], p2eg::N_PF_CLUSTERS_PER_RCT_CARD);
    p2eg::sortAndPadSLR(buffer_had_GCT2_SLR3_posEta[iRCT], p2eg::N_PF_CLUSTERS_PER_RCT_CARD);
    p2eg::sortAndPadSLR(buffer_had_GCT3_SLR1_posEta[iRCT], p2eg::N_PF_CLUSTERS_PER_RCT_CARD);
    p2eg::sortAndPadSLR(buffer_had_GCT3_SLR3_posEta[iRCT], p2eg::N_PF_CLUSTERS_PER_RCT_CARD);
  }

  // Then build the container for each PF SLR, by pushing back, in order, the four RCT cards (starting from most negative phi to most positive phi)
  for (int iRCT = 0; iRCT < 4; iRCT++) {
    for (int iCluster = 0; iCluster < p2eg::N_PF_CLUSTERS_PER_RCT_CARD; iCluster++) {
      // outer loop is over RCT cards, inner loop is over clusters
      out_had_GCT1_SLR1_posEta.push_back(buffer_had_GCT1_SLR1_posEta[iRCT][iCluster]);
      out_had_GCT1_SLR1_negEta.push_back(buffer_had_GCT1_SLR1_negEta[iRCT][iCluster]);
      out_had_GCT1_SLR3_posEta.push_back(buffer_had_GCT1_SLR3_posEta[iRCT][iCluster]);
      out_had_GCT1_SLR3_negEta.push_back(buffer_had_GCT1_SLR3_negEta[iRCT][iCluster]);
      out_had_GCT2_SLR1_posEta.push_back(buffer_had_GCT2_SLR1_posEta[iRCT][iCluster]);
      out_had_GCT2_SLR1_negEta.push_back(buffer_had_GCT2_SLR1_negEta[iRCT][iCluster]);
      out_had_GCT2_SLR3_posEta.push_back(buffer_had_GCT2_SLR3_posEta[iRCT][iCluster]);
      out_had_GCT2_SLR3_negEta.push_back(buffer_had_GCT2_SLR3_negEta[iRCT][iCluster]);
      out_had_GCT3_SLR1_posEta.push_back(buffer_had_GCT3_SLR1_posEta[iRCT][iCluster]);
      out_had_GCT3_SLR1_negEta.push_back(buffer_had_GCT3_SLR1_negEta[iRCT][iCluster]);
      out_had_GCT3_SLR3_posEta.push_back(buffer_had_GCT3_SLR3_posEta[iRCT][iCluster]);
      out_had_GCT3_SLR3_negEta.push_back(buffer_had_GCT3_SLR3_negEta[iRCT][iCluster]);
    }
  }

  // Finally, push back the SLR containers
  outputEmClusters->push_back(out_eg_GCT1_SLR1_posEta);
  outputEmClusters->push_back(out_eg_GCT1_SLR1_negEta);
  outputEmClusters->push_back(out_eg_GCT1_SLR3_posEta);
  outputEmClusters->push_back(out_eg_GCT1_SLR3_negEta);
  outputEmClusters->push_back(out_eg_GCT2_SLR1_posEta);
  outputEmClusters->push_back(out_eg_GCT2_SLR1_negEta);
  outputEmClusters->push_back(out_eg_GCT2_SLR3_posEta);
  outputEmClusters->push_back(out_eg_GCT2_SLR3_negEta);
  outputEmClusters->push_back(out_eg_GCT3_SLR1_posEta);
  outputEmClusters->push_back(out_eg_GCT3_SLR1_negEta);
  outputEmClusters->push_back(out_eg_GCT3_SLR3_posEta);
  outputEmClusters->push_back(out_eg_GCT3_SLR3_negEta);

  outputHadClusters->push_back(out_had_GCT1_SLR1_posEta);
  outputHadClusters->push_back(out_had_GCT1_SLR1_negEta);
  outputHadClusters->push_back(out_had_GCT1_SLR3_posEta);
  outputHadClusters->push_back(out_had_GCT1_SLR3_negEta);
  outputHadClusters->push_back(out_had_GCT2_SLR1_posEta);
  outputHadClusters->push_back(out_had_GCT2_SLR1_negEta);
  outputHadClusters->push_back(out_had_GCT2_SLR3_posEta);
  outputHadClusters->push_back(out_had_GCT2_SLR3_negEta);
  outputHadClusters->push_back(out_had_GCT3_SLR1_posEta);
  outputHadClusters->push_back(out_had_GCT3_SLR1_negEta);
  outputHadClusters->push_back(out_had_GCT3_SLR3_posEta);
  outputHadClusters->push_back(out_had_GCT3_SLR3_negEta);

  iEvent.put(std::move(outputEmClusters), "GCTEmDigiClusters");
  iEvent.put(std::move(outputHadClusters), "GCTHadDigiClusters");
}

//define this as a plug-in
DEFINE_FWK_MODULE(Phase2GCTBarrelToCorrelatorLayer1);
