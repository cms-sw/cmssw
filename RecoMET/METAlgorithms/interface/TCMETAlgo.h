#ifndef TCMETAlgo_h
#define TCMETAlgo_h

/** \class TCMETAlgo
 *
 * Calculates TCMET based on detector response to charged paricles
 * using the tracker to correct for the non-linearity of the calorimeter
 * and the displacement of charged particles by the B-field.  Given a 
 * track pt, eta the expected energy deposited in the calorimeter is
 * obtained from a lookup table, removed from the calorimeter, and
 * replaced with the track at the vertex.
 *
 * \author    F. Golf
 *
 * \version   2nd Version March 24, 2009
 ************************************************************/

#include <vector>
#include <string>
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/ValueMap.h" 

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/METReco/interface/CommonMETData.h"
#include "DataFormats/METReco/interface/MET.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/MuonReco/interface/MuonMETCorrectionData.h"
#include "TH2D.h"

class TCMETAlgo 
{
 public:
  typedef std::vector<const reco::Candidate> InputCollection;
  TCMETAlgo();
  virtual ~TCMETAlgo();
  reco::MET CalculateTCMET(edm::Event& event, const edm::EventSetup& setup, const edm::ParameterSet& iConfig, TH2D *response_function, TH2D *showerRF);
  TH2D* getResponseFunction_fit ( );
  TH2D* getResponseFunction_mode ( );
  TH2D* getResponseFunction_shower ( );
 private:
  double met_x;
  double met_y;
  double sumEt;

  edm::Handle<reco::MuonCollection> MuonHandle;
  edm::Handle<reco::GsfElectronCollection> ElectronHandle;
  edm::Handle<reco::CaloMETCollection> metHandle;
  edm::Handle<reco::TrackCollection> TrackHandle;
  edm::Handle<reco::BeamSpot> beamSpotHandle;
  edm::Handle<reco::VertexCollection> VertexHandle;

  edm::Handle<edm::ValueMap<reco::MuonMETCorrectionData> > muon_data_h;
  edm::Handle<edm::ValueMap<reco::MuonMETCorrectionData> > tcmet_data_h;

  edm::InputTag muonInputTag_;
  edm::InputTag electronInputTag_;
  edm::InputTag metInputTag_;
  edm::InputTag trackInputTag_;
  edm::InputTag beamSpotInputTag_;
  edm::InputTag vertexInputTag_;

  edm::InputTag muonDepValueMap_;
  edm::InputTag tcmetDepValueMap_;
  
  
  int     rfType_;
  int     nMinOuterHits_;
  double  scaleShowerRF_;
  double  usedeltaRRejection_;
  double  deltaRShower_;
  double  minpt_;
  double  maxpt_;
  double  maxeta_;
  double  maxchi2_;
  double  minhits_;
  double  maxd0_;
  double  maxPtErr_;
  double  radius_;
  double  zdist_;
  double  corner_;
  std::vector<int> trkQuality_;
  std::vector<int> trkAlgos_;

  bool isCosmics_;
  bool correctShowerTracks_;
  bool usePvtxd0_;
  bool propagateToHCAL_;

  const class MagneticField* bField;

  class TH2D* response_function;
  class TH2D* showerRF;
  bool hasValidVertex;
  const reco::VertexCollection *vertexColl;

  edm::ValueMap<reco::MuonMETCorrectionData> muon_data;
  edm::ValueMap<reco::MuonMETCorrectionData> tcmet_data;

  bool isMuon( unsigned int );
  bool isElectron( unsigned int ); 
  bool isGoodTrack( const reco::TrackRef );
  void correctMETforMuon( const reco::TrackRef, const unsigned int );
  void correctSumEtForMuon( const reco::TrackRef, const unsigned int );
  void correctMETforMuon( const unsigned int );
  void correctSumEtForMuon( const unsigned int );
  void correctMETforTrack( const reco::TrackRef , TH2D* rf);
  void correctSumEtForTrack( const reco::TrackRef , TH2D* rf);
  class TVector3 propagateTrack( const reco::TrackRef );
  class TVector3 propagateTrackToHCAL( const reco::TrackRef );
  void findGoodShowerTracks(vector<int>& goodShowerTracks);
  bool nearGoodShowerTrack( const reco::TrackRef , vector<int> goodShowerTracks );
  int nExpectedInnerHits(const reco::TrackRef);
  int nExpectedOuterHits(const reco::TrackRef);

};

#endif // TCMETAlgo_h

