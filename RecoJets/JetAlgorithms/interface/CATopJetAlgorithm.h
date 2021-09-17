#ifndef RecoJets_JetAlgorithms_CATopJetAlgorithm_h
#define RecoJets_JetAlgorithms_CATopJetAlgorithm_h

/* *********************************************************
 * \class CATopJetAlgorithm
 * Jet producer to produce top jets using the C-A algorithm to break
 * jets into subjets as described here:
 * "Top-tagging: A Method for Identifying Boosted Hadronic Tops"
 * David E. Kaplan, Keith Rehermann, Matthew D. Schwartz, Brock Tweedie
 * arXiv:0806.0848v1 [hep-ph] 
 *
 ************************************************************/

#include <vector>
#include <list>
#include <functional>
#include <TMath.h>
#include <iostream>

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDefs.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "RecoJets/JetAlgorithms/interface/JetAlgoHelper.h"
#include "RecoJets/JetAlgorithms/interface/CompoundPseudoJet.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "FWCore/Framework/interface/Event.h"

#include "RecoJets/JetAlgorithms/interface/CompoundPseudoJet.h"

#include <fastjet/JetDefinition.hh>
#include <fastjet/PseudoJet.hh>
#include <fastjet/ClusterSequence.hh>
#include <fastjet/GhostedAreaSpec.hh>
#include <fastjet/ClusterSequenceArea.hh>

class CATopJetAlgorithm {
public:
  /** Constructor
  */
  CATopJetAlgorithm(const edm::InputTag& mSrc,
                    bool verbose,
                    int algorithm,
                    int useAdjacency,
                    double centralEtaCut,
                    double ptMin,
                    const std::vector<double>& sumEtBins,
                    const std::vector<double>& rBins,
                    const std::vector<double>& ptFracBins,
                    const std::vector<double>& deltarBins,
                    const std::vector<double>& nCellBins,
                    double seedThreshold,
                    bool useMaxTower,
                    double sumEtEtaCut,
                    double etFrac)
      : mSrc_(mSrc),
        verbose_(verbose),
        algorithm_(algorithm),
        useAdjacency_(useAdjacency),
        centralEtaCut_(centralEtaCut),
        ptMin_(ptMin),
        sumEtBins_(sumEtBins),
        rBins_(rBins),
        ptFracBins_(ptFracBins),
        deltarBins_(deltarBins),
        nCellBins_(nCellBins),
        seedThreshold_(seedThreshold),
        useMaxTower_(useMaxTower),
        sumEtEtaCut_(sumEtEtaCut),
        etFrac_(etFrac)

  {}

  /// Find the ProtoJets from the collection of input Candidates.
  void run(const std::vector<fastjet::PseudoJet>& cell_particles,
           std::vector<fastjet::PseudoJet>& hardjetsOutput,
           std::shared_ptr<fastjet::ClusterSequence>& fjClusterSeq);

private:
  edm::InputTag mSrc_;    //<! calo tower input source
  bool verbose_;          //<!
  int algorithm_;         //<! 0 = KT, 1 = CA, 2 = anti-KT
  int useAdjacency_;      //<! choose adjacency requirement:
                          //<! 	0 = no adjacency
                          //<! 	1 = deltar adjacency
                          //<! 	2 = modified adjacency
                          //<! 	3 = calotower neirest neigbor based adjacency (untested)
  double centralEtaCut_;  //<! eta for defining "central" jets
  double ptMin_;          //<! lower pt cut on which jets to reco
  std::vector<double>
      sumEtBins_;              //<! sumEt bins over which cuts vary. vector={bin 0 lower bound, bin 1 lower bound, ...}
  std::vector<double> rBins_;  //<! Jet distance paramter R. R values depend on sumEt bins.
  std::vector<double> ptFracBins_;  //<! deltap = fraction of full jet pt for a subjet to be consider "hard".
  std::vector<double> deltarBins_;  //<! Applicable only if useAdjacency=1. deltar adjacency values for each sumEtBin
  std::vector<double>
      nCellBins_;  //<! Applicable only if useAdjacency=3. number of cells apart for two subjets to be considered "independent"
  // NOT USED:
  double seedThreshold_;  //<! calo tower seed threshold - NOT USED
  bool useMaxTower_;      //<! use max tower for jet adjacency criterion, false is to use the centroid - NOT USED
  double sumEtEtaCut_;    //<! eta for event SumEt - NOT USED
  double etFrac_;         //<! fraction of event sumEt / 2 for a jet to be considered "hard" - NOT USED
  std::string jetType_;   //<! CaloJets or GenJets - NOT USED

  // Decide if the two jets are in adjacent cells
  bool adjacentCells(const fastjet::PseudoJet& jet1,
                     const fastjet::PseudoJet& jet2,
                     const std::vector<fastjet::PseudoJet>& cell_particles,
                     const fastjet::ClusterSequence& theClusterSequence,
                     double nCellMin) const;

  // Attempt to break up one "hard" jet into two "soft" jets

  bool decomposeJet(const fastjet::PseudoJet& theJet,
                    const fastjet::ClusterSequence& theClusterSequence,
                    const std::vector<fastjet::PseudoJet>& cell_particles,
                    double ptHard,
                    double nCellMin,
                    double deltarcut,
                    fastjet::PseudoJet& ja,
                    fastjet::PseudoJet& jb,
                    std::vector<fastjet::PseudoJet>& leftovers) const;
};

#endif
