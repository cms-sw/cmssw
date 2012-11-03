#ifndef EgammaTowerIsolation_h
#define EgammaTowerIsolation_h

//*****************************************************************************
// File:      EgammaTowerIsolation.h
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer
// Institute: IIHE-VUB
//  Adding feature to exclude towers used by H/E
//=============================================================================
//*****************************************************************************

//C++ includes
#include <vector>

//CMSSW includes
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"


class EgammaTowerIsolation {
 public:
  
  enum HcalDepth{AllDepths=-1,Undefined=0,Depth1=1,Depth2=2};

  //constructors
  EgammaTowerIsolation (double extRadius,
		double intRadius,
		double etLow,
		signed int depth,
		const CaloTowerCollection* ) ;
 
   //destructor 
  ~EgammaTowerIsolation() ;
    //methods

  double getTowerEtSum (const reco::Candidate*, const std::vector<CaloTowerDetId> * detIdToExclude=0) const ;
  double getTowerESum (const reco::Candidate*, const std::vector<CaloTowerDetId> * detIdToExclude=0 ) const ;
  double getTowerEtSum(const reco::SuperCluster*, const std::vector<CaloTowerDetId> * detIdToExclude=0 ) const;
  double getTowerESum(const reco::SuperCluster*, const std::vector<CaloTowerDetId> * detIdToExclude=0) const;
  double getTowerEtSum(float candEta, float candPhi, const std::vector<CaloTowerDetId> * detIdToExclude=0 ) const;
  double getTowerESum(float candEta, float candPhi, const std::vector<CaloTowerDetId> * detIdToExclude=0) const;


 private:

  double extRadius2_ ;
  double intRadius2_ ;
  double etLow_ ;
  signed int depth_;

  const CaloTowerCollection * towercollection_ ;
};

#endif
