#ifndef RecoMuon_MuonIdentification_MuonTimingValidator_H
#define RecoMuon_MuonIdentification_MuonTimingValidator_H

/** \class MuonTimingValidator
 *  Analyzer of the timing information in the reco::Muon object
 *
 *  $Date: 2010/03/25 14:08:50 $
 *  $Revision: 1.6 $
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

class MuonTimingValidator : public edm::EDAnalyzer {
public:
  explicit MuonTimingValidator(const edm::ParameterSet&);
  ~MuonTimingValidator();
  
  typedef std::pair< reco::TrackRef, SimTrackRef> CandToSim;
  typedef std::pair< reco::TrackRef, SimTrackRef> CandStaSim;
  typedef std::pair< reco::TrackRef, SimTrackRef> CandMuonSim;
  
private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  virtual float calculateDistance(const math::XYZVector&, const math::XYZVector&);
  virtual TH1F* divideErr(TH1F*, TH1F*, TH1F*);


  // ----------member data ---------------------------

  edm::InputTag TKtrackTags_; 
  edm::InputTag MuonTags_; 
  edm::InputTag TimeTags_; 
  edm::InputTag SIMtrackTags_; 

  std::string out, open;
  double  theMinEta, theMaxEta, theMinPt, thePtCut, theMinPtres, theMaxPtres, theScale;
  int theDtCut, theCscCut;
  int theNBins;

  edm::Handle<reco::MuonCollection> MuCollection;
  edm::Handle<reco::MuonCollection> MuCollectionT;
  edm::Handle<reco::TrackCollection> TKTrackCollection;
  edm::Handle<reco::TrackCollection> STATrackCollection;
  edm::Handle<reco::TrackCollection> GLBTrackCollection;
  edm::Handle<reco::TrackCollection> PMRTrackCollection;
  edm::Handle<reco::TrackCollection> GMRTrackCollection;
  edm::Handle<reco::TrackCollection> FMSTrackCollection;
  edm::Handle<reco::TrackCollection> SLOTrackCollection;
  edm::Handle<edm::SimTrackContainer> SIMTrackCollection;

  edm::Handle<reco::MuonTimeExtraMap> timeMap1;
  edm::Handle<reco::MuonTimeExtraMap> timeMap2;
  edm::Handle<reco::MuonTimeExtraMap> timeMap3;
  
  //ROOT Pointers
  TFile* hFile;
  TStyle* effStyle;

  TH1F* hi_sta_pt  ;
  TH1F* hi_tk_pt   ;
  TH1F* hi_glb_pt  ;
  TH1F* hi_sta_phi ;
  TH1F* hi_tk_phi  ;
  TH1F* hi_glb_phi ;

  TH1F* hi_mutime_vtx;
  TH1F* hi_mutime_vtx_err;

  TH1F* hi_dtcsc_vtx;
  TH1F* hi_dtecal_vtx;
  TH1F* hi_ecalcsc_vtx;
  TH1F* hi_dthcal_vtx;
  TH1F* hi_hcalcsc_vtx;
  TH1F* hi_hcalecal_vtx;

  TH1F* hi_cmbtime_ibt;
  TH2F* hi_cmbtime_ibt_pt;
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
  TH2F* hi_dttime_ibt_pt;
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
  TH1F* hi_dttime_errdiff;
  TH1F* hi_dttime_ndof;

  TH1F* hi_csctime_ibt;
  TH2F* hi_csctime_ibt_pt;
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

  TH1F* hi_hcal_time;
  TH1F* hi_hcal_time_err;
  TH1F* hi_hcal_time_pull;
  TH1F* hi_hcal_time_ecut;
  TH1F* hi_hcal_energy;

  TH1F* hi_tk_eta  ;
  TH1F* hi_sta_eta  ;
  TH1F* hi_glb_eta  ;

};
#endif
