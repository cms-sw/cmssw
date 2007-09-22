#ifndef EgammaIsolationProducers_EgammaHcalIsolation_h
#define EgammaIsolationProducers_EgammaHcalIsolation_h
//*****************************************************************************
// File:      EgammaHcalIsolation.h
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer
// Institute: IIHE-VUB
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


class EgammaHcalIsolation {
 public:
  
  //constructors
  EgammaHcalIsolation (double extRadius,
		double intRadius,
		double etLow,
		edm::ESHandle<CaloGeometry> ,
                HBHERecHitMetaCollection* ) ;
  
  double getHcalEtSum (const reco::Candidate * ) const ;

  //destructor 
  ~EgammaHcalIsolation() ;
  
 private:

  double extRadius_ ;
  double intRadius_ ;
  double etLow_ ;

  
  edm::ESHandle<CaloGeometry>  theCaloGeom_ ;
  HBHERecHitMetaCollection* mhbhe_ ;

  CaloDualConeSelector* doubleConeSel_;
  


};

#endif
