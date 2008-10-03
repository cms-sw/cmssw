#ifndef EgammaIsolationAlgos_ElectronTkIsolation_h
#define EgammaIsolationAlgos_ElectronTkIsolation_h
//*****************************************************************************
// File:      ElectronTkIsolation.h
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer
// Institute: IIHE-VUB
//=============================================================================
//*****************************************************************************

//C++ includes
#include <vector>
#include <functional>

//Root includes
#include "TObjArray.h"

//CMSSW includes 
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/TrackReco/interface/Track.h"


class ElectronTkIsolation {
 public:
  
  //constructors
  ElectronTkIsolation ( double extRadius,
			double intRadius,
			double ptLow,
			double lip,
			double drb,
			const reco::TrackCollection*,
			reco::TrackBase::Point beamPoint) ;
  //destructor 
  ~ElectronTkIsolation() ;
 
  //methods

  int getNumberTracks(const reco::GsfElectron*) const ;
  double getPtTracks (const reco::GsfElectron*) const ;
  
 private:
    
  double extRadius_ ;
  double intRadius_ ;
  double ptLow_ ;
  double lip_ ;
  double drb_;

  const reco::TrackCollection *trackCollection_ ;
  reco::TrackBase::Point beamPoint_;

  std::pair<int,double>getIso(const reco::GsfElectron*) const ;
};

#endif
