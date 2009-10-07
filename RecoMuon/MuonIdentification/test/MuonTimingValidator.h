#ifndef RecoMuon_MuonIdentification_MuonTimingValidator_H
#define RecoMuon_MuonIdentification_MuonTimingValidator_H

/** \class MuonTimingValidator
 *  Analyzer of the timing information in the reco::Muon object
 *
 *  $Date: 2009/09/18 09:54:43 $
 *  $Revision: 1.1 $
 *  \author P. Traczyk    CERN
 */

// Base Class Headers
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/TrackReco/interface/TrackToTrackMap.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtra.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtraMap.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include <TROOT.h>
#include <TSystem.h>

namespace edm {
  class ParameterSet;
  //  class Event;
  class EventSetup;
  class InputTag;
}

class TFile;
class TH1F;
class TH2F;
//class TrackRef;
//class SimTrackRef;
//class MuonRef;
class MuonServiceProxy;

using namespace std;
using namespace edm;
using namespace reco;

class MuonTimingValidator : public edm::EDAnalyzer {
public:
  explicit MuonTimingValidator(const edm::ParameterSet&);
  ~MuonTimingValidator();
  
  typedef std::pair< TrackRef, SimTrackRef> CandToSim;
  typedef std::pair< TrackRef, SimTrackRef> CandStaSim;
  typedef std::pair< TrackRef, SimTrackRef> CandMuonSim;
  
private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  virtual float calculateDistance(const math::XYZVector&, const math::XYZVector&);
  virtual TH1F* divideErr(TH1F*, TH1F*, TH1F*);


  // ----------member data ---------------------------

  edm::InputTag TKtrackTags_; 
  edm::InputTag MuonTags_; 
  edm::InputTag TimeTags_; 
  edm::InputTag SIMtrackTags_; 

  string out, open;
  double  theMinEta, theMaxEta, theMinPt, thePtCut, theMinPtres, theMaxPtres;
  int theNBins;

  Handle<reco::MuonCollection> MuCollection;
  Handle<reco::MuonCollection> MuCollectionT;
  Handle<reco::TrackCollection> TKTrackCollection;
  Handle<reco::TrackCollection> STATrackCollection;
  Handle<reco::TrackCollection> GLBTrackCollection;
  Handle<reco::TrackCollection> PMRTrackCollection;
  Handle<reco::TrackCollection> GMRTrackCollection;
  Handle<reco::TrackCollection> FMSTrackCollection;
  Handle<reco::TrackCollection> SLOTrackCollection;
  Handle<edm::SimTrackContainer> SIMTrackCollection;

  Handle<reco::MuonTimeExtraMap> timeMap1;
  Handle<reco::MuonTimeExtraMap> timeMap2;
  Handle<reco::MuonTimeExtraMap> timeMap3;
  
  //ROOT Pointers
  TFile* hFile;
  TStyle* effStyle;

  TH1F* hi_sta_pt  ;
  TH1F* hi_tk_pt  ;
  TH1F* hi_glb_pt  ;

  TH1F* hi_mutime_vtx;
  TH1F* hi_mutime_vtx_err;

  TH1F* hi_cmbtime_ibt;
  TH1F* hi_cmbtime_ibt_err;
  TH1F* hi_cmbtime_ibt_pull;
  TH1F* hi_cmbtime_fib;
  TH1F* hi_cmbtime_fib_err;
  TH1F* hi_cmbtime_fib_pull;
  TH1F* hi_cmbtime_vtx;
  TH1F* hi_cmbtime_vtx_err;
  TH1F* hi_cmbtime_vtx_pull;
  TH1F* hi_cmbtime_vtxr;
  TH1F* hi_cmbtime_vtxr_err;
  TH1F* hi_cmbtime_vtxr_pull;
  TH1F* hi_cmbtime_ndof;

  TH1F* hi_dttime_ibt;
  TH1F* hi_dttime_ibt_err;
  TH1F* hi_dttime_ibt_pull;
  TH1F* hi_dttime_fib;
  TH1F* hi_dttime_fib_err;
  TH1F* hi_dttime_fib_pull;
  TH1F* hi_dttime_vtx;
  TH1F* hi_dttime_vtx_err;
  TH1F* hi_dttime_vtx_pull;
  TH1F* hi_dttime_vtxr;
  TH1F* hi_dttime_vtxr_err;
  TH1F* hi_dttime_vtxr_pull;
  TH1F* hi_dttime_ndof;

  TH1F* hi_csctime_ibt;
  TH1F* hi_csctime_ibt_err;
  TH1F* hi_csctime_ibt_pull;
  TH1F* hi_csctime_fib;
  TH1F* hi_csctime_fib_err;
  TH1F* hi_csctime_fib_pull;
  TH1F* hi_csctime_vtx;
  TH1F* hi_csctime_vtx_err;
  TH1F* hi_csctime_vtx_pull;
  TH1F* hi_csctime_vtxr;
  TH1F* hi_csctime_vtxr_err;
  TH1F* hi_csctime_vtxr_pull;
  TH1F* hi_csctime_ndof;

  TH1F* hi_ecal_time;
  TH1F* hi_ecal_time_err;
  TH1F* hi_ecal_time_pull;
  TH1F* hi_ecal_time_ecut;
  TH1F* hi_ecal_energy;

  TH1F* hi_tk_eta  ;
  TH1F* hi_sta_eta  ;
  TH1F* hi_glb_eta  ;

};
#endif
