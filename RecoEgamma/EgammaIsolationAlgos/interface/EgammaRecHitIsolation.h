#ifndef EgammaIsolationProducers_EgammaRecHitIsolation_h
#define EgammaIsolationProducers_EgammaRecHitIsolation_h
//*****************************************************************************
// File:      EgammaRecHitIsolation.h
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer, adapted from EgammaHcalIsolation by S. Harper
// Institute: IIHE-VUB, RAL
//=============================================================================
//*****************************************************************************

//C++ includes
#include <vector>
#include <functional>

//CMSSW includes
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollections.h"
#include "RecoCaloTools/Selectors/interface/CaloDualConeSelector.h"


class EgammaRecHitIsolation {
 public:
  
  //constructors
  EgammaRecHitIsolation (double extRadius,
		       double intRadius,
		       double etLow,
		       edm::ESHandle<CaloGeometry> ,
		       CaloRecHitMetaCollectionV* ,
		       DetId::Detector detector);
  
  double getEtSum (const reco::Candidate * emObject) const {return getSum_(emObject,true);}
  double getEnergySum (const reco::Candidate * emObject) const{ return  getSum_(emObject,false);}

  //destructor 
  ~EgammaRecHitIsolation() ;
  
 private:
  double getSum_(const reco::Candidate *,bool returnEt )const;

  double extRadius_ ;
  double intRadius_ ;
  double etLow_ ;

  
  edm::ESHandle<CaloGeometry>  theCaloGeom_ ;
  CaloRecHitMetaCollectionV* caloHits_ ;

  CaloDualConeSelector* doubleConeSel_;
  


};

#endif
