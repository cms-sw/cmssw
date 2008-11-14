// File: TCMETAlgo.cc
// Description:  see TCMETAlgo.h
// Author: F. Golf, A. Yagil
// Creation Date:  Nov 12, 2008 Initial version.
//
//------------------------------------------------------------------------
#include "RecoMET/METAlgorithms/interface/TCMETAlgo.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/METReco/interface/CommonMETData.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

#include <cmath>
#include <iostream>

using namespace std;
using namespace reco;
using namespace math;
//------------------------------------------------------------------------
// Default Constructer
//----------------------------------
TCMETAlgo::TCMETAlgo() {}
//------------------------------------------------------------------------

//------------------------------------------------------------------------
// Default Destructor
//----------------------------------
TCMETAlgo::~TCMETAlgo() {}
//------------------------------------------------------------------------

reco::MET TCMETAlgo::CalculateTCMET(edm::Event& event, const edm::EventSetup& setup, const edm::ParameterSet& iConfig)
{ 
  //Get Appropriate Handles for TCMET Calculation

  edm::Handle<reco::MuonCollection> MuonHandle;
  event.getByLabel( iConfig.getParameter<edm::InputTag>("muonLabel") , MuonHandle);
  const reco::MuonCollection *Muons = MuonHandle.product();
  reco::MuonCollection::const_iterator muon_it;

  edm::Handle<reco::PixelMatchGsfElectronCollection> ElectronHandle;
  event.getByLabel( iConfig.getParameter<edm::InputTag>("electronLable") , ElectronHandle);
  const reco::PixelMatchGsfElectronCollection *Electrons = ElectronHandle.product();
  reco::PixelMatchGsfElectronCollection::const_iterator electron_it;

  edm::Handle<reco::CaloMETCollection> metHandle;
  event.getByLabel( iConfig.getParameter<edm::InputTag>("metLabel") , metHandle);
  const reco::CaloMETCollection *MET = metHandle.product();
  const reco::CaloMET caloMET = MET->front();
  
  edm::Handle<reco::TrackCollection> TrackHandle;
  event.getByLabel( iConfig.getParameter<edm::InputTag>("trackLable") , TrackHandle);
  const reco::TrackCollection *tracks = TrackHandle.product();
  reco::TrackCollection::const_iterator track_it;

  //Insert Code Here
  //Ultimately, you will need to fill the six variables below the get stored to the MET Object   
  CommonMETData TCMETData;
  TCMETData.mex   = 0.0;
  TCMETData.mey   = 0.0;
  TCMETData.mez   = 0.0;
  TCMETData.met   = 0.0;
  TCMETData.sumet = 0.0;
  TCMETData.phi   = 0.0; 

  XYZTLorentzVector p4( TCMETData.mex , TCMETData.mey , 0, TCMETData.met);
  XYZPointD vtx(0,0,0);
  reco::MET tcmet(TCMETData.sumet, p4, vtx);
  return tcmet;
//------------------------------------------------------------------------
}
