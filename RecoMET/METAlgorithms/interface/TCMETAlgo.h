#ifndef TCMETAlgo_h
#define TCMETAlgo_h

/** \class TCMETAlgo
 *
 * Calculates TCMET based on ... (add details here)
 *
 * \author    F. Golf and A. Yagil
 *
 * \version   1st Version November 12, 2008 
 ************************************************************/

#include <vector>
#include <string>
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

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
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > LorentzVector;

class TCMETAlgo 
{
 public:
  typedef std::vector<const reco::Candidate> InputCollection;
  TCMETAlgo();
  virtual ~TCMETAlgo();
  reco::MET CalculateTCMET(edm::Event& event, const edm::EventSetup& setup, const edm::ParameterSet& iConfig);
 private:
  double met_x;
  double met_y;
  double sumEt;

  edm::Handle<reco::MuonCollection> MuonHandle;
  edm::Handle<reco::PixelMatchGsfElectronCollection> ElectronHandle;
  edm::Handle<reco::CaloMETCollection> metHandle;
  edm::Handle<reco::TrackCollection> TrackHandle;

  const class MagneticField* bField;

  class TH2D* response_function;

  bool isMuon( unsigned int );
  bool isElectron( unsigned int ); 
  bool isGoodTrack( const reco::Track& );
  void correctMETforMuon( const reco::Track& );
  void correctSumEtForMuon( const reco::Track& );
  void correctMETforTrack( const reco::Track& );
  void correctSumEtForTrack( const reco::Track&);
  class TVector3 propagateTrack( const reco::Track& );
  double deltaR( const LorentzVector&, const LorentzVector& );
  TH2D* getResponseFunction ( );
};

#endif // TCMETAlgo_h

