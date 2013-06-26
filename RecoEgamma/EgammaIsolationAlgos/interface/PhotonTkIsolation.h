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
#include <string>
#
//CMSSW includes
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTrackSelector.h"




class PhotonTkIsolation {
 public:
  
  //constructors
  PhotonTkIsolation (float extRadius,
		     float intRadius,
		     float etLow,
		     float lip,
		     float drb,
		     const reco::TrackCollection* trackCollection,
		     reco::TrackBase::Point beamPoint) :
    extRadius2_(extRadius*extRadius),
    intRadiusBarrel2_(intRadius*intRadius),
    intRadiusEndcap2_(intRadius*intRadius),
    stripBarrel_(0.0),
    stripEndcap_(0.0),
    etLow_(etLow),
    lip_(lip),
    drb_(drb),
    trackCollection_(trackCollection),
    beamPoint_(beamPoint) {
    
    setDzOption("vz");

  }
  
  PhotonTkIsolation (float extRadius,
                     float intRadius,
                     float strip,
                     float etLow,
                     float lip,
                     float drb,
                     const reco::TrackCollection* trackCollection,
                     reco::TrackBase::Point beamPoint) :
    extRadius2_(extRadius*extRadius),
    intRadiusBarrel2_(intRadius*intRadius),
    intRadiusEndcap2_(intRadius*intRadius),
    stripBarrel_(strip),
    stripEndcap_(strip),
    etLow_(etLow),
    lip_(lip),
    drb_(drb),
    trackCollection_(trackCollection),
    beamPoint_(beamPoint) {
    
    setDzOption("vz");
    
  }

  
  PhotonTkIsolation (float extRadius,
                     float intRadiusBarrel,
		     float intRadiusEndcap,
		     float stripBarrel,
                     float stripEndcap,		
                     float etLow,
                     float lip,
                     float drb,
                     const reco::TrackCollection* trackCollection,
                     reco::TrackBase::Point beamPoint) :
    extRadius2_(extRadius*extRadius),
    intRadiusBarrel2_(intRadiusBarrel*intRadiusBarrel),
    intRadiusEndcap2_(intRadiusEndcap*intRadiusEndcap),
    stripBarrel_(stripBarrel),
    stripEndcap_(stripEndcap),
    etLow_(etLow),
    lip_(lip),
    drb_(drb),
    trackCollection_(trackCollection),
    beamPoint_(beamPoint) {
    
    setDzOption("vz");
    
  }
  
  
  
  PhotonTkIsolation (float extRadius,
                     float intRadiusBarrel,
		     float intRadiusEndcap,
		     float stripBarrel,
                     float stripEndcap,		
                     float etLow,
                     float lip,
                     float drb,
                     const reco::TrackCollection*,
                     reco::TrackBase::Point beamPoint,
                     const std::string&) ;
  
  //destructor 
  ~PhotonTkIsolation() ;
  //methods

  std::pair<int,float>getIso(const reco::Candidate*) const ;
  

  void setDzOption(const std::string &s);
private:
  
  float extRadius2_ ;
  float intRadiusBarrel2_ ;
  float intRadiusEndcap2_ ;
  float stripBarrel_;
  float stripEndcap_;
  float etLow_ ;
  float lip_ ;
  float drb_;

  const reco::TrackCollection *trackCollection_ ;
  reco::TrackBase::Point beamPoint_;

  int dzOption_;


};

#endif
