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
  TH2D* getResponseFunction ( );
};

#endif // TCMETAlgo_h

