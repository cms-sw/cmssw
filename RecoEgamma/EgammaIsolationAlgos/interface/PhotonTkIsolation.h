#ifndef PhotonTkIsolation_h
#define PhotonTkIsolation_h

//*****************************************************************************
// File:      PhotonTkIsolation.h
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer
// Institute: IIHE-VUB
//=============================================================================
//*****************************************************************************

//C++ includes
#include <vector>
#include <string>
#include <functional>

//CMSSW includes
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTrackSelector.h"




class PhotonTkIsolation {
 public:
  
  //constructors
  PhotonTkIsolation (double extRadius,
		     double intRadius,
		     double etLow,
		     double lip,
		     double drb,
		     const reco::TrackCollection* trackCollection,
		     reco::TrackBase::Point beamPoint) :
  extRadius_(extRadius),
  intRadiusBarrel_(intRadius),
  intRadiusEndcap_(intRadius),
  stripBarrel_(0.0),
  stripEndcap_(0.0),
  etLow_(etLow),
  lip_(lip),
  drb_(drb),
  trackCollection_(trackCollection),
  beamPoint_(beamPoint) {
    
    setDzOption("vz");

  }

  PhotonTkIsolation (double extRadius,
                     double intRadius,
                     double strip,
                     double etLow,
                     double lip,
                     double drb,
                     const reco::TrackCollection* trackCollection,
                     reco::TrackBase::Point beamPoint) :
  extRadius_(extRadius),
  intRadiusBarrel_(intRadius),
  intRadiusEndcap_(intRadius),
  stripBarrel_(strip),
  stripEndcap_(strip),
  etLow_(etLow),
  lip_(lip),
  drb_(drb),
  trackCollection_(trackCollection),
  beamPoint_(beamPoint) {
    
    setDzOption("vz");

  }


  PhotonTkIsolation (double extRadius,
                     double intRadiusBarrel,
		             double intRadiusEndcap,
		             double stripBarrel,
                     double stripEndcap,		
                     double etLow,
                     double lip,
                     double drb,
                     const reco::TrackCollection* trackCollection,
                     reco::TrackBase::Point beamPoint) :
  extRadius_(extRadius),
  intRadiusBarrel_(intRadiusBarrel),
  intRadiusEndcap_(intRadiusEndcap),
  stripBarrel_(stripBarrel),
  stripEndcap_(stripEndcap),
  etLow_(etLow),
  lip_(lip),
  drb_(drb),
  trackCollection_(trackCollection),
  beamPoint_(beamPoint) {
    
    setDzOption("vz");

  }



  PhotonTkIsolation (double extRadius,
                     double intRadiusBarrel,
		             double intRadiusEndcap,
		             double stripBarrel,
                     double stripEndcap,		
                     double etLow,
                     double lip,
                     double drb,
                     const reco::TrackCollection*,
                     reco::TrackBase::Point beamPoint,
                     const std::string&) ;

   //destructor 
  ~PhotonTkIsolation() ;
    //methods

  int getNumberTracks(const reco::Candidate*) const ;
  double getPtTracks (const reco::Candidate*) const ;

  void setDzOption(const std::string &s) {
    if( ! s.compare("dz") )      dzOption_ = egammaisolation::EgammaTrackSelector::dz;
    else if( ! s.compare("vz") ) dzOption_ = egammaisolation::EgammaTrackSelector::vz;
    else if( ! s.compare("bs") ) dzOption_ = egammaisolation::EgammaTrackSelector::bs;
    else if( ! s.compare("vtx") )dzOption_ = egammaisolation::EgammaTrackSelector::vtx;
    else                         dzOption_ = egammaisolation::EgammaTrackSelector::dz;
  }

 private:

  double extRadius_ ;
  double intRadiusBarrel_ ;
  double intRadiusEndcap_ ;
  double stripBarrel_;
  double stripEndcap_;
  double etLow_ ;
  double lip_ ;
  double drb_;

  const reco::TrackCollection *trackCollection_ ;
  reco::TrackBase::Point beamPoint_;

  int dzOption_;

  std::pair<int,double>getIso(const reco::Candidate*) const ;

};

#endif
