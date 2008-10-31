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
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "RecoJets/JetAlgorithms/interface/ProtoJet.h"
#include "RecoJets/JetAlgorithms/interface/JetAlgoHelper.h"
#include "RecoJets/JetAlgorithms/interface/CompoundPseudoJet.h"
#include "RecoJets/JetAlgorithms/interface/JetRecoTypes.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "FWCore/Framework/interface/Event.h"

#include "RecoJets/JetAlgorithms/interface/CompoundPseudoJet.h"

#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"

#include <fastjet/JetDefinition.hh>
#include <fastjet/PseudoJet.hh>
#include <fastjet/ClusterSequence.hh>


class CATopJetAlgorithm{
 public:
  /** Constructor
  */
  CATopJetAlgorithm(edm::InputTag mSrc,
		    int algorithm,
		    double seedThreshold,            
		    double centralEtaCut,            
		    double sumEtEtaCut,              
		    double ptMin,                    
		    double etFrac,            
		    bool   useAdjacency,
		    bool   useMaxTower,
		    const std::vector<double> & ptBins,      
		    const std::vector<double> & rBins,       
		    const std::vector<double> & ptFracBins,  
		    const std::vector<int> & nCellBins) :
    mSrc_          (mSrc          ),
    algorithm_     (algorithm     ),
    seedThreshold_ (seedThreshold ), 
    centralEtaCut_ (centralEtaCut ), 
    sumEtEtaCut_   (sumEtEtaCut   ),   
    ptMin_         (ptMin         ),         
    etFrac_        (etFrac        ),
    useAdjacency_  (useAdjacency  ),
    useMaxTower_   (useMaxTower   ),
    ptBins_        (ptBins        ),        
    rBins_         (rBins         ),         
    ptFracBins_    (ptFracBins    ),    
    nCellBins_     (nCellBins     )
      { }

    /// Find the ProtoJets from the collection of input Candidates.
    void run( const std::vector<fastjet::PseudoJet> & cell_particles, 
	      std::vector<CompoundPseudoJet> & hardjetsOutput,
	      edm::EventSetup const & c
	      );


 private:

  edm::InputTag       mSrc_;          //<! calo tower input source
  int                 algorithm_;     //<! 0 = KT, 1 = CA, 2 = anti-KT
  double              seedThreshold_; //<! calo tower seed threshold                                           
  double              centralEtaCut_; //<! eta for defining "central" jets                                     
  double              sumEtEtaCut_;   //<! eta for event SumEt                                                 
  double              ptMin_;	      //<! lower pt cut on which jets to reco                                  
  double              etFrac_;	      //<! fraction of event sumEt / 2 for a jet to be considered "hard"  
  bool                useAdjacency_;  //<! veto adjacent subjets
  bool                useMaxTower_;   //<! use max tower for jet adjacency criterion, false is to use the centroid
  std::vector<double> ptBins_;	      //<! pt bins over which cuts vary                                        
  std::vector<double> rBins_;	      //<! cone size bins                                                      
  std::vector<double> ptFracBins_;    //<! fraction of full jet pt for a subjet to be consider "hard"          
  std::vector<int>    nCellBins_;     //<! number of cells apart for two subjets to be considered "independent"
  std::string         jetType_;       //<! CaloJets or GenJets

  // Decide if the two jets are in adjacent cells    

  bool adjacentCells(const fastjet::PseudoJet & jet1, const fastjet::PseudoJet & jet2, 
		     const std::vector<fastjet::PseudoJet> & cell_particles,
		     const CaloSubdetectorGeometry  * fTowerGeometry,
		     const fastjet::ClusterSequence & theClusterSequence,
		     int nCellMin ) const;

  
  // Get maximum pt tower
  fastjet::PseudoJet getMaxTower( const fastjet::PseudoJet & jet,
				  const std::vector<fastjet::PseudoJet> & cell_particles,
				  const fastjet::ClusterSequence & theClusterSequence
				  ) const;


  // Find the calo tower associated with the jet
  CaloTowerDetId getCaloTower( const fastjet::PseudoJet & jet,
			       const std::vector<fastjet::PseudoJet> & cell_particles,
			       const CaloSubdetectorGeometry  * fTowerGeometry,
			       const fastjet::ClusterSequence & theClusterSequence ) const;


  // Get number of calo towers away that the two calo towers are
  int getDistance ( CaloTowerDetId const & t1, CaloTowerDetId const & t2, 
		    const CaloSubdetectorGeometry  * fTowerGeometry ) const;
    
  // Attempt to break up one "hard" jet into two "soft" jets

  bool decomposeJet(const fastjet::PseudoJet & theJet, 
		    const fastjet::ClusterSequence & theClusterSequence, 
		    const std::vector<fastjet::PseudoJet> & cell_particles,
		    const CaloSubdetectorGeometry  * fTowerGeometry,
		    double ptHard, int nCellMin,
		    fastjet::PseudoJet & ja, fastjet::PseudoJet & jb, 
		    std::vector<fastjet::PseudoJet> & leftovers) const;

};



#endif
