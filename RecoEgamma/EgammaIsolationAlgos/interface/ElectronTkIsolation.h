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
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTrackSelector.h"

#include<string>


class ElectronTkIsolation {
 public:
  
  //constructors
  ElectronTkIsolation ( double extRadius,
			double intRadius,
			double ptLow,
			double lip,
			double drb,
			const reco::TrackCollection* trackCollection,
			reco::TrackBase::Point beamPoint) :
  extRadius_(extRadius),
  intRadiusBarrel_(intRadius),
  intRadiusEndcap_(intRadius),
  stripBarrel_(0.0),
  stripEndcap_(0.0),
  ptLow_(ptLow),
  lip_(lip),
  drb_(drb),
  trackCollection_(trackCollection),
  beamPoint_(beamPoint) {

        setDzOption("vz");

  }

  ElectronTkIsolation ( double extRadius,
                        double intRadiusBarrel,
			            double intRadiusEndcap,
			            double stripBarrel,
			            double stripEndcap,			
                        double ptLow,
                        double lip,
                        double drb,
                        const reco::TrackCollection* trackCollection,
                        reco::TrackBase::Point beamPoint) :
  extRadius_(extRadius),
  intRadiusBarrel_(intRadiusBarrel),
  intRadiusEndcap_(intRadiusEndcap),
  stripBarrel_(stripBarrel),
  stripEndcap_(stripEndcap),
  ptLow_(ptLow),
  lip_(lip),
  drb_(drb),
  trackCollection_(trackCollection),
  beamPoint_(beamPoint) {

        setDzOption("vz");

  }

  ElectronTkIsolation ( double extRadius,
                        double intRadiusBarrel,
			            double intRadiusEndcap,
			            double stripBarrel,
			            double stripEndcap,			
                        double ptLow,
                        double lip,
                        double drb,
                        const reco::TrackCollection*,
                        reco::TrackBase::Point beamPoint,
                        const std::string&) ;

  //destructor 
  ~ElectronTkIsolation() ;
 
  //methods

    void setDzOption(const std::string &s) {
        if( ! s.compare("dz") )      dzOption_ = egammaisolation::EgammaTrackSelector::dz;
        else if( ! s.compare("vz") ) dzOption_ = egammaisolation::EgammaTrackSelector::vz;
        else if( ! s.compare("bs") ) dzOption_ = egammaisolation::EgammaTrackSelector::bs;
        else if( ! s.compare("vtx") )dzOption_ = egammaisolation::EgammaTrackSelector::vtx;
        else                         dzOption_ = egammaisolation::EgammaTrackSelector::dz;
    }

  int getNumberTracks(const reco::GsfElectron*) const ;
  double getPtTracks (const reco::GsfElectron*) const ;
  std::pair<int,double>getIso(const reco::GsfElectron*) const;
  std::pair<int,double>getIso(const reco::Track*) const ;

 private:
    
  double extRadius_ ;
  double intRadiusBarrel_ ;
  double intRadiusEndcap_ ;
  double stripBarrel_ ;
  double stripEndcap_ ;
  double ptLow_ ;
  double lip_ ;
  double drb_;
  const reco::TrackCollection *trackCollection_ ;
  reco::TrackBase::Point beamPoint_;

  int dzOption_;

  
};

#endif
