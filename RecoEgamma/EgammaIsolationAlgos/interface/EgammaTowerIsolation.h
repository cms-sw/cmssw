#ifndef EgammaTowerIsolation_h
#define EgammaTowerIsolation_h

//*****************************************************************************
// File:      EgammaTowerIsolation.h
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
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"



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

  int getNumberTracks(const reco::Candidate*) const ;
  double getTowerEtSum (const reco::Candidate*) const ;

 private:

  double extRadius_ ;
  double intRadius_ ;
  double etLow_ ;
  signed int depth_;

  const CaloTowerCollection *towercollection_ ;
  
};

#endif
