#include "PhysicsTools/StarterKit/interface/HistoMuon.h"
//#include "DataFormats/MuonReco/interface/MuonEnergy.h"

#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"



#include <iostream>
#include <sstream>

using pat::HistoMuon;
using namespace std;

// Constructor:


HistoMuon::HistoMuon(std::string dir, std::string group,std::string pre,
		   double pt1, double pt2, double m1, double m2,
		     TFileDirectory * parentDir)
  : HistoGroup<Muon>( dir, group, pre, pt1, pt2, m1, m2, parentDir)
{
  addHisto( h_trackIso_ =
	    new PhysVarHisto( pre + "TrackIso", "Muon Track Isolation", 20, 0, 10, currDir_, "", "vD" )
	   );

  addHisto( h_caloIso_  =
	    new PhysVarHisto( pre + "CaloIso",  "Muon Calo Isolation",  20, 0, 10, currDir_, "", "vD" )
	    );

  addHisto( h_leptonID_ =
            new PhysVarHisto( pre + "LeptonID", "Muon Lepton ID",       20, 0, 1, currDir_, "", "vD" )
            );

  addHisto( h_calCompat_ =
            new PhysVarHisto( pre + "CaloCompat", "Muon Calorimetry Compatability", 100, 0, 1, currDir_, "", "vD" )
            );

  addHisto( h_nChambers_ =
            new PhysVarHisto( pre + "NChamber", "Muon # of Chambers", 51, -0.5, 50.5, currDir_, "", "vD" )
            );

  addHisto( h_caloE_ =
            new PhysVarHisto( pre + "CaloE", "Muon Calorimeter Energy", 50, 0, 50, currDir_, "", "vD" )
            );
  addHisto( h_type_ =
            new PhysVarHisto( pre + "Type", "Muon Type", 65, -0.5, 64.5, currDir_, "", "vD" )
            );


  std::string histname = "ecalDepositedEnergy";
  addHisto( ecalDepEnergy_ =
            new PhysVarHisto( pre + histname, histname, 50, 0., 20., currDir_, "", "vD" )
            );

  histname = "ecalS9DepositedEnergy";
  addHisto( ecalS9DepEnergy_ =
            new PhysVarHisto( pre + histname, histname, 80, 0. ,40., currDir_, "", "vD" )
            );
  histname = "hadDepositedEnergy";
  addHisto( hcalDepEnergy_ =
            new PhysVarHisto( pre + histname, histname, 65, -0.5, 64.5, currDir_, "", "vD" )
            );

  histname = "hadS9DepositedEnergy";
  addHisto( hcalS9DepEnergy_ =
            new PhysVarHisto( pre + histname, histname, 80, 0. ,40., currDir_, "", "vD" )
            );

  histname = "hoDepositedEnergy";
  addHisto( hoDepEnergy_ =
            new PhysVarHisto( pre + histname, histname, 50, 0. ,20. , currDir_, "", "vD" )
            );

  histname = "hoS9DepositedEnergy";
  addHisto( hoS9DepEnergy_ =
            new PhysVarHisto( pre + histname, histname, 50, 0. ,20. , currDir_, "", "vD" )
            );



////trajectory seed
/*
  histname = "NumberOfRecHitsPerSeed";
  addHisto( NumberOfRecHitsPerSeed_ =
            new PhysVarHisto( pre + histname, histname, 65, -0.5, 64.5, currDir_, "", "vD" )
            );

  histname = "seedPhi";
  addHisto( seedPhi_ =
            new PhysVarHisto( pre + histname, histname, 65, -0.5, 64.5, currDir_, "", "vD" )
            );
  histname = "seedEta";
  addHisto( seedEta_ =
            new PhysVarHisto( pre + histname, histname, 65, -0.5, 64.5, currDir_, "", "vD" )
            );
  histname = "seedTheta";
  addHisto( seedTheta_ =
            new PhysVarHisto( pre + histname, histname, 65, -0.5, 64.5, currDir_, "", "vD" )
            );
  histname = "seedPt";
  addHisto( seedPt_ =
            new PhysVarHisto( pre + histname, histname, 65, -0.5, 64.5, currDir_, "", "vD" )
            );
 histname = "seedPx";
  addHisto( seedPx_ =
            new PhysVarHisto( pre + histname, histname, 65, -0.5, 64.5, currDir_, "", "vD" )
            );
 histname = "seedPy";
  addHisto( seedPy_ =
            new PhysVarHisto( pre + histname, histname, 65, -0.5, 64.5, currDir_, "", "vD" )
            );
  histname = "seedPz";
  addHisto( seedPz_ =
            new PhysVarHisto( pre + histname, histname, 65, -0.5, 64.5, currDir_, "", "vD" )
            );
  histname = "seedPtErrOverPt";
  addHisto( seedPtErr_ =
            new PhysVarHisto( pre + histname, histname, 65, -0.5, 64.5, currDir_, "", "vD" )
            );
  histname = "seedPErrOverP";
  addHisto( seedPErr_ =
            new PhysVarHisto( pre + histname, histname, 65, -0.5, 64.5, currDir_, "", "vD" )
            );
  histname = "seedPxErrOverPx";
  addHisto( seedPxErr_ =
            new PhysVarHisto( pre + histname, histname, 65, -0.5, 64.5, currDir_, "", "vD" )
            );
  histname = "seedPyErrOverPy";
  addHisto( seedPyErr_ =
            new PhysVarHisto( pre + histname, histname, 65, -0.5, 64.5, currDir_, "", "vD" )
            );
  histname = "seedPzErrOverPz";
  addHisto( seedPzErr_ =
            new PhysVarHisto( pre + histname, histname, 65, -0.5, 64.5, currDir_, "", "vD" )
            );
  histname = "seedPhiErr";
  addHisto( seedPhiErr_ =
            new PhysVarHisto( pre + histname, histname, 65, -0.5, 64.5, currDir_, "", "vD" )
            );
  histname = "seedEtaErr";
  addHisto( seedEtaErr_ =
            new PhysVarHisto( pre + histname, histname, 65, -0.5, 64.5, currDir_, "", "vD" )
            );
*/


//muon reco begin

  addHisto( muReco_ =
            new PhysVarHisto( pre + "Reco", "muReco", 6, 1, 7, currDir_, "", "vD" )
            );

  histname = "GlbMuon";
  PhysVarHisto * temp;
  addHisto( temp =
            new PhysVarHisto( pre + histname+"Glbeta", histname+"Glb_eta", 100, -20., 20., currDir_, "", "vD" )
            );
  etaGlbTrack_.push_back(temp);

  addHisto( temp =
            new PhysVarHisto( pre + histname+"Tketa", histname+"Tk_eta", 50, -4., 4., currDir_, "", "vD" )
            );
  etaGlbTrack_.push_back(temp);

  addHisto( temp  =
            new PhysVarHisto( pre + histname+"Staeta", histname+"Sta_eta", 100, -4., 4., currDir_, "", "vD" )
            );
  etaGlbTrack_.push_back(temp);


  addHisto( temp   =
            new PhysVarHisto( pre + "ResTkGlbeta", "Res_TkGlb_eta", 50,  -4. , 4. , currDir_, "", "vD" )
            );
  etaResolution_.push_back(temp);

  addHisto(  temp  =
            new PhysVarHisto( pre + "ResGlbStaeta", "Res_GlbSta_eta", 100, -4., 4., currDir_, "", "vD" )
            );
  etaResolution_.push_back(temp);

  addHisto(  temp   =
            new PhysVarHisto( pre + "ResTkStaeta", "Res_TkSta_eta", 100, -4., 4., currDir_, "", "vD" )
            );
  etaResolution_.push_back(temp);


  addHisto( etaTrack_ =
            new PhysVarHisto( pre + "TkMuoneta", "TkMuon_eta", 100, -4., 4., currDir_, "", "vD" )
            );

  addHisto( etaStaTrack_ =
            new PhysVarHisto( pre + "StaMuoneta", "StaMuon_eta", 100, -4., 4., currDir_, "", "vD" )
            );

  addHisto( temp =
            new PhysVarHisto( pre + histname+"Glbtheta", histname+"Glb_theta", 50, 0., 4., currDir_, "", "vD" )
            );
  thetaGlbTrack_.push_back(temp);

  addHisto( temp =
            new PhysVarHisto( pre + histname+"Tktheta", histname+"Tk_theta", 50, 0., 4., currDir_, "", "vD" )
            );
  thetaGlbTrack_.push_back(temp);

  addHisto(  temp  =
            new PhysVarHisto( pre + histname+"Statheta", histname+"Sta_theta", 50, 0. , 4., currDir_, "", "vD" )
            );
  thetaGlbTrack_.push_back(temp);

  addHisto(  temp   =
            new PhysVarHisto( pre + "ResTkGlbtheta", "Res_TkGlb_theta", 50, 0. , 4., currDir_, "", "vD" )
            );
  thetaResolution_.push_back(temp);

  addHisto(  temp   =
            new PhysVarHisto( pre + "ResGlbStatheta", "Res_GlbSta_theta", 50, 0. , 4. , currDir_, "", "vD" )
            );
  thetaResolution_.push_back(temp);

  addHisto(  temp   =
            new PhysVarHisto( pre + "ResTkStatheta", "Res_TkSta_theta", 50, 0. , 4. , currDir_, "", "vD" )
            );
  thetaResolution_.push_back(temp);

  addHisto( thetaTrack_ =
            new PhysVarHisto( pre + "TkMuontheta", "TkMuon_theta",  50, 0. , 4. , currDir_, "", "vD" )
            );

  addHisto( thetaStaTrack_ =
            new PhysVarHisto( pre + "StaMuontheta", "StaMuon_theta", 50, 0., 4., currDir_, "", "vD" )
            );

  addHisto( temp =
            new PhysVarHisto( pre + histname+"Glbphi", histname+"Glb_phi", 100, -4., 4., currDir_, "", "vD" )
            );
  phiGlbTrack_.push_back(temp);

  addHisto(  temp  =
            new PhysVarHisto( pre + histname+"Tkphi", histname+"Tk_phi", 50, -4., 4., currDir_, "", "vD" )
            );
  phiGlbTrack_.push_back(temp);

  addHisto(  temp  =
            new PhysVarHisto( pre + histname+"Staphi", histname+"Sta_phi", 100, -4.,  4., currDir_, "", "vD" )
            );
  phiGlbTrack_.push_back(temp);

  addHisto( temp  =
            new PhysVarHisto( pre + "ResTkGlbphi", "Res_TkGlb_phi", 50, 0., 4., currDir_, "", "vD" )
            );
  phiResolution_.push_back(temp);

  addHisto(  temp  =
            new PhysVarHisto( pre + "ResGlbStaphi", "Res_GlbSta_phi", 50, -1., 4., currDir_, "", "vD" )
            );
  phiResolution_.push_back(temp);

  addHisto(  temp   =
            new PhysVarHisto( pre + "ResTkStaphi", "Res_TkSta_phi",  50, -1., 4., currDir_, "", "vD" )
            );
  phiResolution_.push_back(temp);

  addHisto( phiTrack_ =
            new PhysVarHisto( pre + "TkMuonphi", "TkMuon_phi", 50, -4. , 4., currDir_, "", "vD" )
            );
  addHisto( phiStaTrack_ =
            new PhysVarHisto( pre + "StaMuonphi", "StaMuon_phi", 50, -4. , 4., currDir_, "", "vD" )
            );

  addHisto( temp  =
            new PhysVarHisto( pre + histname+"Glbp", histname+"Glb_p", 100, 0., 100, currDir_, "", "vD" )
            );
  pGlbTrack_.push_back(temp);

  addHisto( temp  =
            new PhysVarHisto( pre + histname+"Tkp", histname+"Tk_p", 100, 0., 100., currDir_, "", "vD" )
            );
  pGlbTrack_.push_back(temp);

  addHisto(  temp   =
            new PhysVarHisto( pre + histname+"Stap", histname+"Sta_p", 100, 0., 100., currDir_, "", "vD" )
            );
  pGlbTrack_.push_back(temp);

  addHisto( pTrack_ =
            new PhysVarHisto( pre + "TkMuonp", "TkMuon_p", 100, 0., 100., currDir_, "", "vD" )
            );
  addHisto( pStaTrack_ =
            new PhysVarHisto( pre + "StaMuonp", "StaMuon_p", 100, 0., 100., currDir_, "", "vD" )
            );

  addHisto( temp  =
            new PhysVarHisto( pre + histname+"Glbpt", histname+"Glb_pt", 100,  0., 100., currDir_, "", "vD" )
            );
  ptGlbTrack_.push_back(temp);

  addHisto(  temp   =
            new PhysVarHisto( pre + histname+"Tkpt", histname+"Tk_pt", 100, 0., 100., currDir_, "", "vD" )
            );
  ptGlbTrack_.push_back(temp);

  addHisto( temp   =
            new PhysVarHisto( pre + histname+"Stapt", histname+"Sta_pt", 100, 0., 100., currDir_, "", "vD" )
            );
  ptGlbTrack_.push_back(temp);

  addHisto( ptTrack_ =
            new PhysVarHisto( pre + "TkMuonpt", "TkMuon_pt", 65, -0.5, 64.5, currDir_, "", "vD" )
            );
  addHisto(  ptStaTrack_ =
            new PhysVarHisto( pre + "StaMuonpt", "StaMuon_pt", 65, -0.5, 64.5, currDir_, "", "vD" )
            );

  addHisto( temp  =
            new PhysVarHisto( pre + histname+"Glbq", histname+"Glb_q", 100, -4., 4., currDir_, "", "vD" )
            );
  qGlbTrack_.push_back(temp);

  addHisto(  temp   =
            new PhysVarHisto( pre + histname+"Tkq", histname+"Tk_q", 50, -4., 4., currDir_, "", "vD" )
            );
  qGlbTrack_.push_back(temp);

  addHisto(  temp    =
            new PhysVarHisto( pre + histname+"Staq", histname+"Sta_q", 50, -4. , 4., currDir_, "", "vD" )
            );
  qGlbTrack_.push_back(temp);

  addHisto( temp  =
            new PhysVarHisto( pre + histname+"qComparison", histname+"qComparison", 50, 0., 20., currDir_, "", "vD" )
            );
  qGlbTrack_.push_back(temp);

  addHisto( qTrack_ =
            new PhysVarHisto( pre + "TkMuonq", "TkMuon_q", 50, -4. ,4., currDir_, "", "vD" )
            );
  addHisto( qStaTrack_ =
            new PhysVarHisto( pre + "StaMuonq", "StaMuon_q", 50, -4. ,4.,  currDir_, "", "vD" )
            );

  addHisto( temp =
            new PhysVarHisto( pre + "ResTkGlbqOverp", "Res_TkGlb_qOverp", 50, 0., 4., currDir_, "", "vD" )
            );
  qOverpResolution_.push_back(temp);

  addHisto(  temp  =
            new PhysVarHisto( pre + "ResGlbStaqOverp", "Res_GlbSta_qOverp", 50, 0., 4., currDir_, "", "vD" )
            );
  qOverpResolution_.push_back(temp);

  addHisto(  temp  =
            new PhysVarHisto( pre + "ResTkStaqOverp", "Res_TkSta_qOverp", 50, 0., 4., currDir_, "", "vD" )
            );
  qOverpResolution_.push_back(temp);

  addHisto( temp  =
            new PhysVarHisto( pre + "ResTkGlboneOverp", "Res_TkGlb_oneOverp", 50, 0., 4., currDir_, "", "vD" )
            );
  oneOverpResolution_.push_back(temp);

  addHisto( temp  =
            new PhysVarHisto( pre + "ResGlbStaoneOverp", "Res_GlbSta_oneOverp", 50, 0., 4., currDir_, "", "vD" )
            );
  oneOverpResolution_.push_back(temp);

  addHisto(  temp =
            new PhysVarHisto( pre + "ResTkStaoneOverp", "Res_TkSta_oneOverp", 50, 0., 4., currDir_, "", "vD" )
            );
  oneOverpResolution_.push_back(temp);

  addHisto( temp  =
            new PhysVarHisto( pre + "ResTkGlbqOverpt", "Res_TkGlb_qOverpt", 50, 0., 4., currDir_, "", "vD" )
            );
  qOverptResolution_.push_back(temp);

  addHisto( temp =
            new PhysVarHisto( pre + "ResGlbStaqOverpt", "Res_GlbSta_qOverpt", 50, 0., 4., currDir_, "", "vD" )
            );
  qOverptResolution_.push_back(temp);

  addHisto( temp =
            new PhysVarHisto( pre + "ResTkStaqOverpt", "Res_TkSta_qOverpt", 50, 0., 4., currDir_, "", "vD" )
            );
  qOverptResolution_.push_back(temp);

  addHisto( temp =
            new PhysVarHisto( pre + "ResTkGlboneOverpt", "Res_TkGlb_oneOverpt", 50, 0., 4., currDir_, "", "vD" )
            );
  oneOverptResolution_.push_back(temp);

  addHisto( temp =
            new PhysVarHisto( pre + "ResGlbStaoneOverpt", "Res_GlbSta_oneOverpt", 50, 0., 4., currDir_, "", "vD" )
            );
  oneOverptResolution_.push_back(temp);

  addHisto( temp =
            new PhysVarHisto( pre + "ResTkStaoneOverpt", "Res_TkSta_oneOverpt", 50, 0., 4., currDir_, "", "vD" )
            );
  oneOverptResolution_.push_back(temp);
// muon reco end


// Muon Track
//Global Muon
  addHisto( GlbhitsNotUsed_ =
            new PhysVarHisto( pre + "GlbHitsNotUsedForGlobalTracking", "GlbHitsNotUsedForGlobalTracking", 50, -0.5, 49.5, currDir_, "", "vD" )
            );
  addHisto( GlbhitsNotUsedPercentual_ =
            new PhysVarHisto( pre + "GlbHitsNotUsedForGlobalTrackingDvHitUsed", "GlbHitsNotUsedForGlobalTrackingDvHitUsed" , 100, 0, 1., currDir_, "", "vD" )
            );
  addHisto( GlbhitStaProvenance_ =
            new PhysVarHisto( pre + "GlbtrackHitStaProvenance", "GlbtrackHitStaProvenance",  7, 0.5, 7.5, currDir_, "", "vD" )
            );
  addHisto( GlbhitTkrProvenance_ =
            new PhysVarHisto( pre + "GlbtrackHitTkrProvenance", "GlbtrackHitTkrProvenance",  6, 0.5, 6.5, currDir_, "", "vD" )
            );

//StandAlone Muon
  addHisto( StahitsNotUsed_ =
            new PhysVarHisto( pre + "StaHitsNotUsedForGlobalTracking", "StaHitsNotUsedForGlobalTracking", 50, -0.5, 49.5, currDir_, "", "vD" )
            );
  addHisto( StahitsNotUsedPercentual_ =
            new PhysVarHisto( pre + "StaHitsNotUsedForGlobalTrackingDvHitUsed", "StaHitsNotUsedForGlobalTrackingDvHitUsed" , 100, 0, 1., currDir_, "", "vD" )
            );
  addHisto( StahitStaProvenance_ =
            new PhysVarHisto( pre + "StatrackHitStaProvenance", "StatrackHitStaProvenance",  7, 0.5, 7.5, currDir_, "", "vD" )
            );
  addHisto( StahitTkrProvenance_ =
            new PhysVarHisto( pre + "StatrackHitTkrProvenance", "StatrackHitTkrProvenance",  6, 0.5, 6.5, currDir_, "", "vD" )
            );


}


// fill a plain ol' muon
void HistoMuon::fill( const Muon *muon, uint iMu, double weight )
{

  // First fill common 4-vector histograms

  HistoGroup<Muon>::fill( muon, iMu, weight);

  // fill relevant muon histograms
  h_trackIso_->fill( muon->trackIso(), iMu , weight);
  h_caloIso_ ->fill( muon->caloIso() , iMu , weight);
  h_leptonID_->fill( muon->isGood(), iMu , weight);

  h_nChambers_->fill( muon->numberOfChambers(), iMu , weight);

  h_calCompat_->fill( muon->caloCompatibility(), iMu, weight );
  h_type_->fill( muon->type(), iMu, weight );
  reco::MuonEnergy muEnergy = muon->calEnergy();

  h_caloE_->fill( muEnergy.em+muEnergy.had+muEnergy.ho, iMu , weight);



///////////////////////////////////////

  // get all the mu energy deposits
  ecalDepEnergy_->fill(muEnergy.em, iMu, weight);

  hcalDepEnergy_->fill(muEnergy.had, iMu, weight);

  hoDepEnergy_->fill(muEnergy.ho, iMu, weight);

  ecalS9DepEnergy_->fill(muEnergy.emS9, iMu, weight);

  hcalS9DepEnergy_->fill(muEnergy.hadS9, iMu, weight);

  hoS9DepEnergy_->fill(muEnergy.hoS9, iMu, weight);

/////
// Muon reco fill

  if( muon->isGlobalMuon()) {
    if(muon->isTrackerMuon() &&  muon->isStandAloneMuon())
      muReco_->fill(1, iMu, weight);
    if(!( muon->isTrackerMuon()) &&  muon->isStandAloneMuon())
      muReco_->fill(2, iMu, weight);
    if(! muon->isStandAloneMuon());


    // get the track combinig the information from both the Tracker and the Spectrometer
    reco::TrackRef recoCombinedGlbTrack = muon->combinedMuon();
    // get the track using only the tracker data
    reco::TrackRef recoGlbTrack = muon->track();
    // get the track using only the mu spectrometer data
    reco::TrackRef recoStaGlbTrack = muon->standAloneMuon();
  
    etaGlbTrack_[0]->fill(recoCombinedGlbTrack->eta(), iMu, weight);
    etaGlbTrack_[1]->fill(recoGlbTrack->eta(), iMu, weight);
    etaGlbTrack_[2]->fill(recoStaGlbTrack->eta(), iMu, weight);
    etaResolution_[0]->fill(recoGlbTrack->eta()-recoCombinedGlbTrack->eta(), iMu, weight);
    etaResolution_[1]->fill(-recoStaGlbTrack->eta()+recoCombinedGlbTrack->eta(), iMu, weight);
    etaResolution_[2]->fill(recoGlbTrack->eta()-recoStaGlbTrack->eta(), iMu, weight);
// two dimensional histos
//    etaResolution_[3]->fill(recoCombinedGlbTrack->eta(), recoGlbTrack->eta()-recoCombinedGlbTrack->eta());
//    etaResolution_[4]->fill(recoCombinedGlbTrack->eta(), -recoStaGlbTrack->eta()+recoCombinedGlbTrack->eta());
//    etaResolution_[5]->fill(recoCombinedGlbTrack->eta(), recoGlbTrack->eta()-recoStaGlbTrack->eta());

    thetaGlbTrack_[0]->fill(recoCombinedGlbTrack->theta(), iMu, weight);
    thetaGlbTrack_[1]->fill(recoGlbTrack->theta(), iMu, weight);
    thetaGlbTrack_[2]->fill(recoStaGlbTrack->theta(), iMu, weight);
    thetaResolution_[0]->fill(recoGlbTrack->theta()-recoCombinedGlbTrack->theta(), iMu, weight);
    thetaResolution_[1]->fill(-recoStaGlbTrack->theta()+recoCombinedGlbTrack->theta(), iMu, weight);
    thetaResolution_[2]->fill(recoGlbTrack->theta()-recoStaGlbTrack->theta(), iMu, weight);
// two dimensional histos
//    thetaResolution_[3]->fill(recoCombinedGlbTrack->theta(), recoGlbTrack->theta()-recoCombinedGlbTrack->theta());
//    thetaResolution_[4]->fill(recoCombinedGlbTrack->theta(), -recoStaGlbTrack->theta()+recoCombinedGlbTrack->theta());
//    thetaResolution_[5]->fill(recoCombinedGlbTrack->theta(), recoGlbTrack->theta()-recoStaGlbTrack->theta());
      
    phiGlbTrack_[0]->fill(recoCombinedGlbTrack->phi(), iMu, weight);
    phiGlbTrack_[1]->fill(recoGlbTrack->phi(), iMu, weight);
    phiGlbTrack_[2]->fill(recoStaGlbTrack->phi(), iMu, weight);
    phiResolution_[0]->fill(recoGlbTrack->phi()-recoCombinedGlbTrack->phi(), iMu, weight);
    phiResolution_[1]->fill(-recoStaGlbTrack->phi()+recoCombinedGlbTrack->phi(), iMu, weight);
    phiResolution_[2]->fill(recoGlbTrack->phi()-recoStaGlbTrack->phi(), iMu, weight);
// two dimensional histos
//    phiResolution_[3]->fill(recoCombinedGlbTrack->phi(), recoGlbTrack->phi()-recoCombinedGlbTrack->phi());
//    phiResolution_[4]->fill(recoCombinedGlbTrack->phi(), -recoStaGlbTrack->phi()+recoCombinedGlbTrack->phi());
//    phiResolution_[5]->fill(recoCombinedGlbTrack->phi(), recoGlbTrack->phi()-recoStaGlbTrack->phi());
     
    pGlbTrack_[0]->fill(recoCombinedGlbTrack->p(), iMu, weight);
    pGlbTrack_[1]->fill(recoGlbTrack->p(), iMu, weight);
    pGlbTrack_[2]->fill(recoStaGlbTrack->p(), iMu, weight);
 
    ptGlbTrack_[0]->fill(recoCombinedGlbTrack->pt(), iMu, weight);
    ptGlbTrack_[1]->fill(recoGlbTrack->pt(), iMu, weight);
    ptGlbTrack_[2]->fill(recoStaGlbTrack->pt(), iMu, weight);
 
    qGlbTrack_[0]->fill(recoCombinedGlbTrack->charge(), iMu, weight);
    qGlbTrack_[1]->fill(recoGlbTrack->charge(), iMu, weight);
    qGlbTrack_[2]->fill(recoStaGlbTrack->charge(), iMu, weight);
    if(recoCombinedGlbTrack->charge()==recoStaGlbTrack->charge()) qGlbTrack_[3]->fill(1, iMu, weight);
    else qGlbTrack_[3]->fill(2, iMu, weight);
    if(recoCombinedGlbTrack->charge()==recoGlbTrack->charge()) qGlbTrack_[3]->fill(3, iMu, weight);
    else qGlbTrack_[3]->fill(4, iMu, weight);
    if(recoStaGlbTrack->charge()==recoGlbTrack->charge()) qGlbTrack_[3]->fill(5, iMu, weight);
    else qGlbTrack_[3]->fill(6, iMu, weight);
    if(recoCombinedGlbTrack->charge()!=recoStaGlbTrack->charge() && recoCombinedGlbTrack->charge()!=recoGlbTrack->charge()) qGlbTrack_[3]->fill(7, iMu, weight);
    if(recoCombinedGlbTrack->charge()==recoStaGlbTrack->charge() && recoCombinedGlbTrack->charge()==recoGlbTrack->charge()) qGlbTrack_[3]->fill(8, iMu, weight);
     
    qOverpResolution_[0]->fill((recoGlbTrack->charge()/recoGlbTrack->p())-(recoCombinedGlbTrack->charge()/recoCombinedGlbTrack->p()), iMu, weight);
    qOverpResolution_[1]->fill(-(recoStaGlbTrack->charge()/recoStaGlbTrack->p())+(recoCombinedGlbTrack->charge()/recoCombinedGlbTrack->p()), iMu, weight);
    qOverpResolution_[2]->fill((recoGlbTrack->charge()/recoGlbTrack->p())-(recoStaGlbTrack->charge()/recoStaGlbTrack->p()), iMu, weight);
    oneOverpResolution_[0]->fill((1/recoGlbTrack->p())-(1/recoCombinedGlbTrack->p()), iMu, weight);
    oneOverpResolution_[1]->fill(-(1/recoStaGlbTrack->p())+(1/recoCombinedGlbTrack->p()), iMu, weight);
    oneOverpResolution_[2]->fill((1/recoGlbTrack->p())-(1/recoStaGlbTrack->p()), iMu, weight);
    qOverptResolution_[0]->fill((recoGlbTrack->charge()/recoGlbTrack->pt())-(recoCombinedGlbTrack->charge()/recoCombinedGlbTrack->pt()), iMu, weight);
    qOverptResolution_[1]->fill(-(recoStaGlbTrack->charge()/recoStaGlbTrack->pt())+(recoCombinedGlbTrack->charge()/recoCombinedGlbTrack->pt()), iMu, weight);
    qOverptResolution_[2]->fill((recoGlbTrack->charge()/recoGlbTrack->pt())-(recoStaGlbTrack->charge()/recoStaGlbTrack->pt()), iMu, weight);
    oneOverptResolution_[0]->fill((1/recoGlbTrack->pt())-(1/recoCombinedGlbTrack->pt()), iMu, weight);
    oneOverptResolution_[1]->fill(-(1/recoStaGlbTrack->pt())+(1/recoCombinedGlbTrack->pt()), iMu, weight);
    oneOverptResolution_[2]->fill((1/recoGlbTrack->pt())-(1/recoStaGlbTrack->pt()), iMu, weight);
//  two dimensional histos
//    oneOverptResolution_[3]->fill(recoCombinedGlbTrack->eta(),(1/recoGlbTrack->pt())-(1/recoCombinedGlbTrack->pt()));
//    oneOverptResolution_[4]->fill(recoCombinedGlbTrack->eta(),-(1/recoStaGlbTrack->pt())+(1/recoCombinedGlbTrack->pt()));
//    oneOverptResolution_[5]->fill(recoCombinedGlbTrack->eta(),(1/recoGlbTrack->pt())-(1/recoStaGlbTrack->pt()));
//    oneOverptResolution_[6]->fill(recoCombinedGlbTrack->phi(),(1/recoGlbTrack->pt())-(1/recoCombinedGlbTrack->pt()));
//    oneOverptResolution_[7]->fill(recoCombinedGlbTrack->phi(),-(1/recoStaGlbTrack->pt())+(1/recoCombinedGlbTrack->pt()));
//    oneOverptResolution_[8]->fill(recoCombinedGlbTrack->phi(),(1/recoGlbTrack->pt())-(1/recoStaGlbTrack->pt()));
//    oneOverptResolution_[9]->fill(recoCombinedGlbTrack->pt(),(1/recoGlbTrack->pt())-(1/recoCombinedGlbTrack->pt()));
//    oneOverptResolution_[10]->fill(recoCombinedGlbTrack->pt(),-(1/recoStaGlbTrack->pt())+(1/recoCombinedGlbTrack->pt()));
//    oneOverptResolution_[11]->fill(recoCombinedGlbTrack->pt(),(1/recoGlbTrack->pt())-(1/recoStaGlbTrack->pt()));

  }



  if(muon->isTrackerMuon() && !(muon->isGlobalMuon())) {
     if(muon->isStandAloneMuon())
       muReco_->fill(3, iMu, weight);
     if(!(muon->isStandAloneMuon()))
       muReco_->fill(4, iMu, weight);
 
    // get the track using only the tracker data
    reco::TrackRef recoTrack = muon->track();

    etaTrack_->fill(recoTrack->eta(), iMu, weight);
    thetaTrack_->fill(recoTrack->theta(), iMu, weight);
    phiTrack_->fill(recoTrack->phi(), iMu, weight);
    pTrack_->fill(recoTrack->p(), iMu, weight);
    ptTrack_->fill(recoTrack->pt(), iMu, weight);
    qTrack_->fill(recoTrack->charge(), iMu, weight);
  }


  if(muon->isStandAloneMuon() && !(muon->isGlobalMuon())) {
    if(!(muon->isTrackerMuon()))
      muReco_->fill(5, iMu, weight);
     
    // get the track using only the mu spectrometer data
    reco::TrackRef recoStaTrack = muon->standAloneMuon();
 
    etaStaTrack_->fill(recoStaTrack->eta(), iMu, weight);
    thetaStaTrack_->fill(recoStaTrack->theta(), iMu, weight);
    phiStaTrack_->fill(recoStaTrack->phi(), iMu, weight);
    pStaTrack_->fill(recoStaTrack->p(), iMu, weight);
    ptStaTrack_->fill(recoStaTrack->pt(), iMu, weight);
    qStaTrack_->fill(recoStaTrack->charge(), iMu, weight);
 
  }
     
  if(muon->isCaloMuon() && !(muon->isGlobalMuon()) && !(muon->isTrackerMuon()) && !(muon->isStandAloneMuon()))
    muReco_->fill(6, iMu, weight);

// muon track
//Global track
  if(muon->isGlobalMuon()){
   reco::Track const & GlbrecoTrack = *(muon->globalTrack());
   // hit counters
   int hitsFromDt=0;
   int hitsFromCsc=0;
   int hitsFromRpc=0;
   int hitsFromTk=0;
   int hitsFromTrack=0;
//   int hitsFromSegmDt=0;
//   int hitsFromSegmCsc=0;
   // segment counters
//   int segmFromDt=0;
//   int segmFromCsc=0;


   // hits from track
   for(trackingRecHit_iterator GlbrecHit =  GlbrecoTrack.recHitsBegin(); GlbrecHit != GlbrecoTrack.recHitsEnd(); ++GlbrecHit){
 
     hitsFromTrack++;
      DetId id = (*GlbrecHit)->geographicalId();
      // hits from DT
      if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::DT ) 
        hitsFromDt++;   
      // hits from CSC
       if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::CSC ) 
        hitsFromCsc++;
      // hits from RPC
      if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::RPC ) 
        hitsFromRpc++;
      // hits from Tracker
      if (id.det() == DetId::Tracker){
        hitsFromTk++;
        if(id.subdetId() == PixelSubdetector::PixelBarrel )
          GlbhitTkrProvenance_->fill(1,  iMu,   weight);
        if(id.subdetId() == PixelSubdetector::PixelEndcap )
          GlbhitTkrProvenance_->fill(2,  iMu,   weight);
        if(id.subdetId() == SiStripDetId::TIB )
          GlbhitTkrProvenance_->fill(3,  iMu,   weight);
        if(id.subdetId() == SiStripDetId::TID )
          GlbhitTkrProvenance_->fill(4,  iMu,   weight);
        if(id.subdetId() == SiStripDetId::TOB )
          GlbhitTkrProvenance_->fill(5,  iMu,   weight);
        if(id.subdetId() == SiStripDetId::TEC )
          GlbhitTkrProvenance_->fill(6,  iMu,   weight);
      }
 
   }
 
   // fill the histos

//   GlbhitsNotUsed_->fill(hitsFromSegmDt+hitsFromSegmCsc+hitsFromRpc+hitsFromTk-hitsFromTrack,  iMu,   weight);
//   GlbhitsNotUsedPercentual_->fill(double(hitsFromSegmDt+hitsFromSegmCsc+hitsFromRpc+hitsFromTk-hitsFromTrack)/hitsFromTrack,  iMu,   weight);
 
   if(hitsFromDt!=0 && hitsFromCsc==0 && hitsFromRpc==0) GlbhitStaProvenance_->fill(1,  iMu,   weight);
   if(hitsFromCsc!=0 && hitsFromDt==0 && hitsFromRpc==0) GlbhitStaProvenance_->fill(2,  iMu,   weight);
   if(hitsFromRpc!=0 && hitsFromDt==0 && hitsFromCsc==0) GlbhitStaProvenance_->fill(3,  iMu,   weight);
   if(hitsFromDt!=0 && hitsFromCsc!=0 && hitsFromRpc==0) GlbhitStaProvenance_->fill(4,  iMu,   weight);
   if(hitsFromDt!=0 && hitsFromRpc!=0 && hitsFromCsc==0) GlbhitStaProvenance_->fill(5,  iMu,   weight);
   if(hitsFromCsc!=0 && hitsFromRpc!=0 && hitsFromDt==0) GlbhitStaProvenance_->fill(6,  iMu,   weight);
   if(hitsFromDt!=0 && hitsFromCsc!=0 && hitsFromRpc!=0) GlbhitStaProvenance_->fill(7,  iMu,   weight);
  }





//StandAlone track
  if(muon->isStandAloneMuon()){
   reco::Track const & StarecoTrack = *(muon->standAloneMuon());
   // hit counters
   int hitsFromDt=0;
   int hitsFromCsc=0;
   int hitsFromRpc=0;
   int hitsFromTk=0;
   int hitsFromTrack=0;
//   int hitsFromSegmDt=0;
//   int hitsFromSegmCsc=0;
   // segment counters
//   int segmFromDt=0;
//   int segmFromCsc=0;

   // hits from track
   for(trackingRecHit_iterator StarecHit =  StarecoTrack.recHitsBegin(); StarecHit != StarecoTrack.recHitsEnd(); ++StarecHit){

     hitsFromTrack++;
      DetId id = (*StarecHit)->geographicalId();
      // hits from DT
      if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::DT )
        hitsFromDt++;
      // hits from CSC
       if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::CSC )
        hitsFromCsc++;
      // hits from RPC
      if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::RPC )
        hitsFromRpc++;
      // hits from Tracker
      if (id.det() == DetId::Tracker){
        hitsFromTk++;
        if(id.subdetId() == PixelSubdetector::PixelBarrel )
          StahitTkrProvenance_->fill(1,  iMu,   weight);
        if(id.subdetId() == PixelSubdetector::PixelEndcap )
          StahitTkrProvenance_->fill(2,  iMu,   weight);
        if(id.subdetId() == SiStripDetId::TIB )
          StahitTkrProvenance_->fill(3,  iMu,   weight);
        if(id.subdetId() == SiStripDetId::TID )
          StahitTkrProvenance_->fill(4,  iMu,   weight);
        if(id.subdetId() == SiStripDetId::TOB )
          StahitTkrProvenance_->fill(5,  iMu,   weight);
        if(id.subdetId() == SiStripDetId::TEC )
          StahitTkrProvenance_->fill(6,  iMu,   weight);
      }
 
   }

   // fill the histos

//   StahitsNotUsed_->fill(hitsFromSegmDt+hitsFromSegmCsc+hitsFromRpc+hitsFromTk-hitsFromTrack,  iMu,   weight);
//   StahitsNotUsedPercentual_->fill(double(hitsFromSegmDt+hitsFromSegmCsc+hitsFromRpc+hitsFromTk-hitsFromTrack)/hitsFromTrack,  iMu,   weight);

   if(hitsFromDt!=0 && hitsFromCsc==0 && hitsFromRpc==0) StahitStaProvenance_->fill(1,  iMu,   weight);
   if(hitsFromCsc!=0 && hitsFromDt==0 && hitsFromRpc==0) StahitStaProvenance_->fill(2,  iMu,   weight);
   if(hitsFromRpc!=0 && hitsFromDt==0 && hitsFromCsc==0) StahitStaProvenance_->fill(3,  iMu,   weight);
   if(hitsFromDt!=0 && hitsFromCsc!=0 && hitsFromRpc==0) StahitStaProvenance_->fill(4,  iMu,   weight);
   if(hitsFromDt!=0 && hitsFromRpc!=0 && hitsFromCsc==0) StahitStaProvenance_->fill(5,  iMu,   weight);
   if(hitsFromCsc!=0 && hitsFromRpc!=0 && hitsFromDt==0) StahitStaProvenance_->fill(6,  iMu,   weight);
   if(hitsFromDt!=0 && hitsFromCsc!=0 && hitsFromRpc!=0) StahitStaProvenance_->fill(7,  iMu,   weight);
  }





}


// fill a muon that is a shallow clone, and take kinematics from 
// shallow clone but detector plots from the muon itself
void HistoMuon::fill( const reco::ShallowClonePtrCandidate *pshallow, uint iMu, double weight )
{

  // Get the underlying object that the shallow clone represents
  const pat::Muon * muon = dynamic_cast<const pat::Muon*>(pshallow);

  if ( muon == 0 ) {
    cout << "Error! Was passed a shallow clone that is not at heart a muon" << endl;
    return;
  }

  

  // First fill common 4-vector histograms from shallow clone

  HistoGroup<Muon>::fill( pshallow, iMu, weight);

  // fill relevant muon histograms from muon
  h_trackIso_->fill( muon->trackIso(), iMu , weight);
  h_caloIso_ ->fill( muon->caloIso() , iMu , weight);
  h_leptonID_->fill( muon->isGood(), iMu , weight);

  h_nChambers_->fill( muon->numberOfChambers(), iMu , weight);

  h_calCompat_->fill( muon->caloCompatibility(), iMu, weight );
  h_type_->fill( muon->type(), iMu, weight );
  reco::MuonEnergy muEnergy = muon->calEnergy();

  h_caloE_->fill( muEnergy.em+muEnergy.had+muEnergy.ho, iMu , weight);

///////////////////////////////////////

  // get all the mu energy deposits
  ecalDepEnergy_->fill(muEnergy.em, iMu, weight);

  hcalDepEnergy_->fill(muEnergy.had, iMu, weight);

  hoDepEnergy_->fill(muEnergy.ho, iMu, weight);

  ecalS9DepEnergy_->fill(muEnergy.emS9, iMu, weight);

  hcalS9DepEnergy_->fill(muEnergy.hadS9, iMu, weight);

  hoS9DepEnergy_->fill(muEnergy.hoS9, iMu, weight);



/////
// Muon reco fill

  if( muon->isGlobalMuon()) {
    if(muon->isTrackerMuon() &&  muon->isStandAloneMuon())
      muReco_->fill(1, iMu, weight);
    if(!( muon->isTrackerMuon()) &&  muon->isStandAloneMuon())
      muReco_->fill(2, iMu, weight);
    if(! muon->isStandAloneMuon());


    // get the track combinig the information from both the Tracker and the Spectrometer
    reco::TrackRef recoCombinedGlbTrack = muon->combinedMuon();
    // get the track using only the tracker data
    reco::TrackRef recoGlbTrack = muon->track();
    // get the track using only the mu spectrometer data
    reco::TrackRef recoStaGlbTrack = muon->standAloneMuon();
  
    etaGlbTrack_[0]->fill(recoCombinedGlbTrack->eta(), iMu, weight);
    etaGlbTrack_[1]->fill(recoGlbTrack->eta(), iMu, weight);
    etaGlbTrack_[2]->fill(recoStaGlbTrack->eta(), iMu, weight);
    etaResolution_[0]->fill(recoGlbTrack->eta()-recoCombinedGlbTrack->eta(), iMu, weight);
    etaResolution_[1]->fill(-recoStaGlbTrack->eta()+recoCombinedGlbTrack->eta(), iMu, weight);
    etaResolution_[2]->fill(recoGlbTrack->eta()-recoStaGlbTrack->eta(), iMu, weight);
//    etaResolution_[3]->fill(recoCombinedGlbTrack->eta(), recoGlbTrack->eta()-recoCombinedGlbTrack->eta());
//    etaResolution_[4]->fill(recoCombinedGlbTrack->eta(), -recoStaGlbTrack->eta()+recoCombinedGlbTrack->eta());
//    etaResolution_[5]->fill(recoCombinedGlbTrack->eta(), recoGlbTrack->eta()-recoStaGlbTrack->eta());

    thetaGlbTrack_[0]->fill(recoCombinedGlbTrack->theta(), iMu, weight);
    thetaGlbTrack_[1]->fill(recoGlbTrack->theta(), iMu, weight);
    thetaGlbTrack_[2]->fill(recoStaGlbTrack->theta(), iMu, weight);
    thetaResolution_[0]->fill(recoGlbTrack->theta()-recoCombinedGlbTrack->theta(), iMu, weight);
    thetaResolution_[1]->fill(-recoStaGlbTrack->theta()+recoCombinedGlbTrack->theta(), iMu, weight);
    thetaResolution_[2]->fill(recoGlbTrack->theta()-recoStaGlbTrack->theta(), iMu, weight);
//    thetaResolution_[3]->fill(recoCombinedGlbTrack->theta(), recoGlbTrack->theta()-recoCombinedGlbTrack->theta());
//    thetaResolution_[4]->fill(recoCombinedGlbTrack->theta(), -recoStaGlbTrack->theta()+recoCombinedGlbTrack->theta());
//    thetaResolution_[5]->fill(recoCombinedGlbTrack->theta(), recoGlbTrack->theta()-recoStaGlbTrack->theta());
      
    phiGlbTrack_[0]->fill(recoCombinedGlbTrack->phi(), iMu, weight);
    phiGlbTrack_[1]->fill(recoGlbTrack->phi(), iMu, weight);
    phiGlbTrack_[2]->fill(recoStaGlbTrack->phi(), iMu, weight);
    phiResolution_[0]->fill(recoGlbTrack->phi()-recoCombinedGlbTrack->phi(), iMu, weight);
    phiResolution_[1]->fill(-recoStaGlbTrack->phi()+recoCombinedGlbTrack->phi(), iMu, weight);
    phiResolution_[2]->fill(recoGlbTrack->phi()-recoStaGlbTrack->phi(), iMu, weight);
//    phiResolution_[3]->fill(recoCombinedGlbTrack->phi(), recoGlbTrack->phi()-recoCombinedGlbTrack->phi());
//    phiResolution_[4]->fill(recoCombinedGlbTrack->phi(), -recoStaGlbTrack->phi()+recoCombinedGlbTrack->phi());
//    phiResolution_[5]->fill(recoCombinedGlbTrack->phi(), recoGlbTrack->phi()-recoStaGlbTrack->phi());
     
    pGlbTrack_[0]->fill(recoCombinedGlbTrack->p(), iMu, weight);
    pGlbTrack_[1]->fill(recoGlbTrack->p(), iMu, weight);
    pGlbTrack_[2]->fill(recoStaGlbTrack->p(), iMu, weight);
 
    ptGlbTrack_[0]->fill(recoCombinedGlbTrack->pt(), iMu, weight);
    ptGlbTrack_[1]->fill(recoGlbTrack->pt(), iMu, weight);
    ptGlbTrack_[2]->fill(recoStaGlbTrack->pt(), iMu, weight);
 
    qGlbTrack_[0]->fill(recoCombinedGlbTrack->charge(), iMu, weight);
    qGlbTrack_[1]->fill(recoGlbTrack->charge(), iMu, weight);
    qGlbTrack_[2]->fill(recoStaGlbTrack->charge(), iMu, weight);
    if(recoCombinedGlbTrack->charge()==recoStaGlbTrack->charge()) qGlbTrack_[3]->fill(1, iMu, weight);
    else qGlbTrack_[3]->fill(2, iMu, weight);
    if(recoCombinedGlbTrack->charge()==recoGlbTrack->charge()) qGlbTrack_[3]->fill(3, iMu, weight);
    else qGlbTrack_[3]->fill(4, iMu, weight);
    if(recoStaGlbTrack->charge()==recoGlbTrack->charge()) qGlbTrack_[3]->fill(5, iMu, weight);
    else qGlbTrack_[3]->fill(6, iMu, weight);
    if(recoCombinedGlbTrack->charge()!=recoStaGlbTrack->charge() && recoCombinedGlbTrack->charge()!=recoGlbTrack->charge()) qGlbTrack_[3]->fill(7, iMu, weight);
    if(recoCombinedGlbTrack->charge()==recoStaGlbTrack->charge() && recoCombinedGlbTrack->charge()==recoGlbTrack->charge()) qGlbTrack_[3]->fill(8, iMu, weight);
     
    qOverpResolution_[0]->fill((recoGlbTrack->charge()/recoGlbTrack->p())-(recoCombinedGlbTrack->charge()/recoCombinedGlbTrack->p()), iMu, weight);
    qOverpResolution_[1]->fill(-(recoStaGlbTrack->charge()/recoStaGlbTrack->p())+(recoCombinedGlbTrack->charge()/recoCombinedGlbTrack->p()), iMu, weight);
    qOverpResolution_[2]->fill((recoGlbTrack->charge()/recoGlbTrack->p())-(recoStaGlbTrack->charge()/recoStaGlbTrack->p()), iMu, weight);
    oneOverpResolution_[0]->fill((1/recoGlbTrack->p())-(1/recoCombinedGlbTrack->p()), iMu, weight);
    oneOverpResolution_[1]->fill(-(1/recoStaGlbTrack->p())+(1/recoCombinedGlbTrack->p()), iMu, weight);
    oneOverpResolution_[2]->fill((1/recoGlbTrack->p())-(1/recoStaGlbTrack->p()), iMu, weight);
    qOverptResolution_[0]->fill((recoGlbTrack->charge()/recoGlbTrack->pt())-(recoCombinedGlbTrack->charge()/recoCombinedGlbTrack->pt()), iMu, weight);
    qOverptResolution_[1]->fill(-(recoStaGlbTrack->charge()/recoStaGlbTrack->pt())+(recoCombinedGlbTrack->charge()/recoCombinedGlbTrack->pt()), iMu, weight);
    qOverptResolution_[2]->fill((recoGlbTrack->charge()/recoGlbTrack->pt())-(recoStaGlbTrack->charge()/recoStaGlbTrack->pt()), iMu, weight);
    oneOverptResolution_[0]->fill((1/recoGlbTrack->pt())-(1/recoCombinedGlbTrack->pt()), iMu, weight);
    oneOverptResolution_[1]->fill(-(1/recoStaGlbTrack->pt())+(1/recoCombinedGlbTrack->pt()), iMu, weight);
    oneOverptResolution_[2]->fill((1/recoGlbTrack->pt())-(1/recoStaGlbTrack->pt()), iMu, weight);
//    oneOverptResolution_[3]->fill(recoCombinedGlbTrack->eta(),(1/recoGlbTrack->pt())-(1/recoCombinedGlbTrack->pt()));
//    oneOverptResolution_[4]->fill(recoCombinedGlbTrack->eta(),-(1/recoStaGlbTrack->pt())+(1/recoCombinedGlbTrack->pt()));
//    oneOverptResolution_[5]->fill(recoCombinedGlbTrack->eta(),(1/recoGlbTrack->pt())-(1/recoStaGlbTrack->pt()));
//    oneOverptResolution_[6]->fill(recoCombinedGlbTrack->phi(),(1/recoGlbTrack->pt())-(1/recoCombinedGlbTrack->pt()));
//    oneOverptResolution_[7]->fill(recoCombinedGlbTrack->phi(),-(1/recoStaGlbTrack->pt())+(1/recoCombinedGlbTrack->pt()));
//    oneOverptResolution_[8]->fill(recoCombinedGlbTrack->phi(),(1/recoGlbTrack->pt())-(1/recoStaGlbTrack->pt()));
//    oneOverptResolution_[9]->fill(recoCombinedGlbTrack->pt(),(1/recoGlbTrack->pt())-(1/recoCombinedGlbTrack->pt()));
//    oneOverptResolution_[10]->fill(recoCombinedGlbTrack->pt(),-(1/recoStaGlbTrack->pt())+(1/recoCombinedGlbTrack->pt()));
//    oneOverptResolution_[11]->fill(recoCombinedGlbTrack->pt(),(1/recoGlbTrack->pt())-(1/recoStaGlbTrack->pt()));

  }



  if(muon->isTrackerMuon() && !(muon->isGlobalMuon())) {
     if(muon->isStandAloneMuon())
       muReco_->fill(3, iMu, weight);
     if(!(muon->isStandAloneMuon()))
       muReco_->fill(4, iMu, weight);
 
    // get the track using only the tracker data
    reco::TrackRef recoTrack = muon->track();

    etaTrack_->fill(recoTrack->eta(), iMu, weight);
    thetaTrack_->fill(recoTrack->theta(), iMu, weight);
    phiTrack_->fill(recoTrack->phi(), iMu, weight);
    pTrack_->fill(recoTrack->p(), iMu, weight);
    ptTrack_->fill(recoTrack->pt(), iMu, weight);
    qTrack_->fill(recoTrack->charge(), iMu, weight);
  }


  if(muon->isStandAloneMuon() && !(muon->isGlobalMuon())) {
    if(!(muon->isTrackerMuon()))
      muReco_->fill(5, iMu, weight);
     
    // get the track using only the mu spectrometer data
    reco::TrackRef recoStaTrack = muon->standAloneMuon();
 
    etaStaTrack_->fill(recoStaTrack->eta(), iMu, weight);
    thetaStaTrack_->fill(recoStaTrack->theta(), iMu, weight);
    phiStaTrack_->fill(recoStaTrack->phi(), iMu, weight);
    pStaTrack_->fill(recoStaTrack->p(), iMu, weight);
    ptStaTrack_->fill(recoStaTrack->pt(), iMu, weight);
    qStaTrack_->fill(recoStaTrack->charge(), iMu, weight);
 
  }
     
  if(muon->isCaloMuon() && !(muon->isGlobalMuon()) && !(muon->isTrackerMuon()) && !(muon->isStandAloneMuon()))
    muReco_->fill(6, iMu, weight);

// muon track
//Global track
  if(muon->isGlobalMuon()){
   reco::Track const & GlbrecoTrack = *(muon->globalTrack());
   // hit counters
   int hitsFromDt=0;
   int hitsFromCsc=0;
   int hitsFromRpc=0;
   int hitsFromTk=0;
   int hitsFromTrack=0;
//   int hitsFromSegmDt=0;
//   int hitsFromSegmCsc=0;
   // segment counters
//   int segmFromDt=0;
//   int segmFromCsc=0;


   // hits from track
   for(trackingRecHit_iterator GlbrecHit =  GlbrecoTrack.recHitsBegin(); GlbrecHit != GlbrecoTrack.recHitsEnd(); ++GlbrecHit){

     hitsFromTrack++;
      DetId id = (*GlbrecHit)->geographicalId();
      // hits from DT
      if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::DT )
        hitsFromDt++;
      // hits from CSC
       if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::CSC )
        hitsFromCsc++;
      // hits from RPC
      if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::RPC )
        hitsFromRpc++;
      // hits from Tracker
      if (id.det() == DetId::Tracker){
        hitsFromTk++;
        if(id.subdetId() == PixelSubdetector::PixelBarrel )
          GlbhitTkrProvenance_->fill(1,  iMu,   weight);
        if(id.subdetId() == PixelSubdetector::PixelEndcap )
          GlbhitTkrProvenance_->fill(2,  iMu,   weight);
        if(id.subdetId() == SiStripDetId::TIB )
          GlbhitTkrProvenance_->fill(3,  iMu,   weight);
        if(id.subdetId() == SiStripDetId::TID )
          GlbhitTkrProvenance_->fill(4,  iMu,   weight);
        if(id.subdetId() == SiStripDetId::TOB )
          GlbhitTkrProvenance_->fill(5,  iMu,   weight);
        if(id.subdetId() == SiStripDetId::TEC )
          GlbhitTkrProvenance_->fill(6,  iMu,   weight);
      }

   }

   // fill the histos

//   GlbhitsNotUsed_->fill(hitsFromSegmDt+hitsFromSegmCsc+hitsFromRpc+hitsFromTk-hitsFromTrack,  iMu,   weight);
//   GlbhitsNotUsedPercentual_->fill(double(hitsFromSegmDt+hitsFromSegmCsc+hitsFromRpc+hitsFromTk-hitsFromTrack)/hitsFromTrack,  iMu,   weight);

   if(hitsFromDt!=0 && hitsFromCsc==0 && hitsFromRpc==0) GlbhitStaProvenance_->fill(1,  iMu,   weight);
   if(hitsFromCsc!=0 && hitsFromDt==0 && hitsFromRpc==0) GlbhitStaProvenance_->fill(2,  iMu,   weight);
   if(hitsFromRpc!=0 && hitsFromDt==0 && hitsFromCsc==0) GlbhitStaProvenance_->fill(3,  iMu,   weight);
   if(hitsFromDt!=0 && hitsFromCsc!=0 && hitsFromRpc==0) GlbhitStaProvenance_->fill(4,  iMu,   weight);
   if(hitsFromDt!=0 && hitsFromRpc!=0 && hitsFromCsc==0) GlbhitStaProvenance_->fill(5,  iMu,   weight);
   if(hitsFromCsc!=0 && hitsFromRpc!=0 && hitsFromDt==0) GlbhitStaProvenance_->fill(6,  iMu,   weight);
   if(hitsFromDt!=0 && hitsFromCsc!=0 && hitsFromRpc!=0) GlbhitStaProvenance_->fill(7,  iMu,   weight);
  }





 //StandAlone track
  if(muon->isStandAloneMuon()){
   reco::Track const & StarecoTrack = *(muon->standAloneMuon());
   // hit counters
   int hitsFromDt=0;
   int hitsFromCsc=0;
   int hitsFromRpc=0;
   int hitsFromTk=0;
   int hitsFromTrack=0;
//   int hitsFromSegmDt=0;
//   int hitsFromSegmCsc=0;
   // segment counters
//   int segmFromDt=0;
//   int segmFromCsc=0;

   // hits from track
   for(trackingRecHit_iterator StarecHit =  StarecoTrack.recHitsBegin(); StarecHit != StarecoTrack.recHitsEnd(); ++StarecHit){

     hitsFromTrack++;
      DetId id = (*StarecHit)->geographicalId();
      // hits from DT
      if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::DT )
        hitsFromDt++;
      // hits from CSC
       if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::CSC )
        hitsFromCsc++;
      // hits from RPC
      if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::RPC )
        hitsFromRpc++;
      // hits from Tracker
      if (id.det() == DetId::Tracker){
        hitsFromTk++;
        if(id.subdetId() == PixelSubdetector::PixelBarrel )
          StahitTkrProvenance_->fill(1,  iMu,   weight);
        if(id.subdetId() == PixelSubdetector::PixelEndcap )
          StahitTkrProvenance_->fill(2,  iMu,   weight);
        if(id.subdetId() == SiStripDetId::TIB )
          StahitTkrProvenance_->fill(3,  iMu,   weight);
        if(id.subdetId() == SiStripDetId::TID )
          StahitTkrProvenance_->fill(4,  iMu,   weight);
        if(id.subdetId() == SiStripDetId::TOB )
          StahitTkrProvenance_->fill(5,  iMu,   weight);
        if(id.subdetId() == SiStripDetId::TEC )
          StahitTkrProvenance_->fill(6,  iMu,   weight);
      }

   }

   // fill the histos

//   StahitsNotUsed_->fill(hitsFromSegmDt+hitsFromSegmCsc+hitsFromRpc+hitsFromTk-hitsFromTrack,  iMu,   weight);
//   StahitsNotUsedPercentual_->fill(double(hitsFromSegmDt+hitsFromSegmCsc+hitsFromRpc+hitsFromTk-hitsFromTrack)/hitsFromTrack,  iMu,   weight);

   if(hitsFromDt!=0 && hitsFromCsc==0 && hitsFromRpc==0) StahitStaProvenance_->fill(1,  iMu,   weight);
   if(hitsFromCsc!=0 && hitsFromDt==0 && hitsFromRpc==0) StahitStaProvenance_->fill(2,  iMu,   weight);
   if(hitsFromRpc!=0 && hitsFromDt==0 && hitsFromCsc==0) StahitStaProvenance_->fill(3,  iMu,   weight);
   if(hitsFromDt!=0 && hitsFromCsc!=0 && hitsFromRpc==0) StahitStaProvenance_->fill(4,  iMu,   weight);
   if(hitsFromDt!=0 && hitsFromRpc!=0 && hitsFromCsc==0) StahitStaProvenance_->fill(5,  iMu,   weight);
   if(hitsFromCsc!=0 && hitsFromRpc!=0 && hitsFromDt==0) StahitStaProvenance_->fill(6,  iMu,   weight);
   if(hitsFromDt!=0 && hitsFromCsc!=0 && hitsFromRpc!=0) StahitStaProvenance_->fill(7,  iMu,   weight);
  } 




}

void HistoMuon::fillCollection( const std::vector<Muon> & coll, double weight )
{

  h_size_->fill( coll.size(), 1, weight );     //! Save the size of the collection.

  std::vector<Muon>::const_iterator
    iobj = coll.begin(),
    iend = coll.end();

  uint i = 1;              //! Fortran-style indexing
  for ( ; iobj != iend; ++iobj, ++i ) {
    fill( &*iobj, i, weight);      //! &*iobj dereferences to the pointer to a PHYS_OBJ*
  }
}


void HistoMuon::clearVec()
{
  HistoGroup<Muon>::clearVec();

  h_trackIso_->clearVec();
  h_caloIso_->clearVec();
  h_leptonID_->clearVec();
  h_calCompat_->clearVec();
  h_caloE_->clearVec();
  h_type_->clearVec();
  h_nChambers_->clearVec();
  
//muon energy deposit analyzer
  ecalDepEnergy_->clearVec();
  ecalS9DepEnergy_->clearVec();
  hcalDepEnergy_->clearVec();
  hcalS9DepEnergy_->clearVec();
  hoDepEnergy_->clearVec();
  hoS9DepEnergy_->clearVec();

// muon reco  
  muReco_->clearVec(); 
  while(!etaGlbTrack_.empty())
    etaGlbTrack_.pop_back();
  while(!etaResolution_.empty())
    etaResolution_.pop_back();
  while(!thetaGlbTrack_.empty())
    thetaGlbTrack_.pop_back();
  while(!thetaResolution_.empty())
    thetaResolution_.pop_back();
  while(!phiGlbTrack_.empty())
    phiGlbTrack_.pop_back();
  while(!phiResolution_.empty())
    phiResolution_.pop_back();
  while(!pGlbTrack_.empty())
    pGlbTrack_.pop_back();
  while(!ptGlbTrack_.empty())
    ptGlbTrack_.pop_back();
  while(!qGlbTrack_.empty())
    qGlbTrack_.pop_back();
  while(!qOverpResolution_.empty())
    qOverpResolution_.pop_back();
  while(!qOverptResolution_.empty())
    qOverptResolution_.pop_back();
  while(!oneOverpResolution_.empty())
    oneOverpResolution_.pop_back();
  while(!oneOverptResolution_.empty())
    oneOverptResolution_.pop_back();

// tracker muon
  etaTrack_->clearVec();
  thetaTrack_->clearVec();
  phiTrack_->clearVec();
  pTrack_->clearVec();
  ptTrack_->clearVec();
  qTrack_->clearVec();

// sta muon
  etaStaTrack_->clearVec();
  thetaStaTrack_->clearVec();
  phiStaTrack_->clearVec();
  pStaTrack_->clearVec();
  ptStaTrack_->clearVec();
  qStaTrack_->clearVec();

// segment track 

  GlbhitsNotUsed_->clearVec();
  GlbhitsNotUsedPercentual_->clearVec();
  GlbhitStaProvenance_->clearVec();
  GlbhitTkrProvenance_->clearVec();

  StahitsNotUsed_->clearVec();
  StahitsNotUsedPercentual_->clearVec();
  StahitStaProvenance_->clearVec();
  StahitTkrProvenance_->clearVec();

/*
  TrackSegm_->clearVec();
  trackHitPercentualVsEta_->clearVec();
  trackHitPercentualVsPhi_->clearVec();
  trackHitPercentualVsPt_->clearVec();
  dtTrackHitPercentualVsEta_->clearVec();
  dtTrackHitPercentualVsPhi_->clearVec();
  dtTrackHitPercentualVsPt_->clearVec();
  cscTrackHitPercentualVsEta_->clearVec();
  cscTrackHitPercentualVsPhi_->clearVec();
  cscTrackHitPercentualVsPt_->clearVec();
*/

}
