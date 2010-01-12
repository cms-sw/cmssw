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
#include <functional>

//CMSSW includes
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"



class PhotonTkIsolation {
 public:
  
  //constructors
  PhotonTkIsolation (double extRadius,
		     double intRadius,
		     double etLow,
		     double lip,
		     double drb,
		     const reco::TrackCollection*,
		     reco::TrackBase::Point beamPoint) ;

  PhotonTkIsolation (double extRadius,
                     double intRadius,
                     double strip,
                     double etLow,
                     double lip,
                     double drb,
                     const reco::TrackCollection*,
                     reco::TrackBase::Point beamPoint) ;

  PhotonTkIsolation (double extRadius,
                     double intRadiusBarrel,
		     double intRadiusEndcap,
		     double stripBarrel,
                     double stripEndcap,		
                     double etLow,
                     double lip,
                     double drb,
                     const reco::TrackCollection*,
                     reco::TrackBase::Point beamPoint) ;

   //destructor 
  ~PhotonTkIsolation() ;
    //methods

  int getNumberTracks(const reco::Candidate*) const ;
  double getPtTracks (const reco::Candidate*) const ;

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

  std::pair<int,double>getIso(const reco::Candidate*) const ;

};

#endif
