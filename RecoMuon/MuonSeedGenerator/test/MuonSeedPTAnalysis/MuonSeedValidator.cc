// Class Header
#include "MuonSeedValidator.h"
#include "SegSelector.h"

// for MuonSeedBuilder
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "TFile.h"
#include "TVector3.h"

#include <iostream>
#include <fstream>
#include <map>
#include <utility>
#include <string>
#include <stdio.h>
#include <algorithm>

DEFINE_FWK_MODULE(MuonSeedValidator);
using namespace std;
using namespace edm;
using namespace reco;


// constructors
MuonSeedValidator::MuonSeedValidator(const ParameterSet& pset){ 

  rootFileName      = pset.getUntrackedParameter<string>("rootFileName");
  recHitLabel       = pset.getUntrackedParameter<string>("recHitLabel");
  cscSegmentLabel   = pset.getUntrackedParameter<string>("cscSegmentLabel");
  dtrecHitLabel     = pset.getUntrackedParameter<string>("dtrecHitLabel");
  dtSegmentLabel    = pset.getUntrackedParameter<string>("dtSegmentLabel");
  simHitLabel       = pset.getUntrackedParameter<string>("simHitLabel");
  simTrackLabel     = pset.getUntrackedParameter<string>("simTrackLabel");
  muonseedLabel     = pset.getUntrackedParameter<string>("muonseedLabel");
  staTrackLabel     = pset.getParameter<InputTag>("staTrackLabel");
  glbTrackLabel     = pset.getParameter<InputTag>("glbTrackLabel");

  debug             = pset.getUntrackedParameter<bool>("debug");
  dtMax             = pset.getUntrackedParameter<double>("dtMax");
  dfMax             = pset.getUntrackedParameter<double>("dfMax");
  scope             = pset.getUntrackedParameter<bool>("scope");
  pTCutMax          = pset.getUntrackedParameter<double>("pTCutMax");
  pTCutMin          = pset.getUntrackedParameter<double>("pTCutMin");
  eta_Low           = pset.getUntrackedParameter<double>("eta_Low");
  eta_High          = pset.getUntrackedParameter<double>("eta_High");

  ParameterSet serviceParameters = pset.getParameter<ParameterSet>("ServiceParameters");
  theService        = new MuonServiceProxy(serviceParameters);

  recsegSelector    = new SegSelector(pset);

  // Create the root file
  theFile = new TFile(rootFileName.c_str(), "RECREATE");
  theFile->mkdir("AllMuonSys");
  theFile->cd();
  theFile->mkdir("No_Seed");
  theFile->cd();
  theFile->mkdir("No_STA");
  theFile->cd();
  theFile->mkdir("EventScope");
  theFile->cd();
  theFile->mkdir("UnRelated");
  theFile->cd();
  // TTree test
  tr_muon = new TNtuple1();

  h_all     = new H2DRecHit1("AllMu_");
  h_NoSeed  = new H2DRecHit2("NoSeed");
  h_NoSta   = new H2DRecHit3("NoSta");
  h_Scope   = new H2DRecHit4();
  h_UnRel   = new H2DRecHit5();
}

// destructor
MuonSeedValidator::~MuonSeedValidator(){

  if (debug) cout << "[Seed Validation] Destructor called" << endl;
  delete recsegSelector;
  //delete muonSeedBuilder_; 
  // Write the histos to file
  theFile->cd();
  theFile->cd("AllMuonSys");
  h_all->Write();

  theFile->cd();
  theFile->cd("No_Seed");
  h_NoSeed->Write();

  theFile->cd();
  theFile->cd("No_STA");
  h_NoSta->Write();

  theFile->cd();
  theFile->cd("EventScope");
  h_Scope->Write();

  theFile->cd();
  theFile->cd("UnRelated");
  h_UnRel->Write();
  // for tree
  theFile->cd();
  tr_muon->Write();

  // Release the memory ...
  delete h_all;
  delete h_NoSeed;
  delete h_NoSta;
  delete h_Scope;
  delete h_UnRel;
  delete tr_muon;

  theFile->Close();
  if (debug) cout << "************* Finished writing histograms to file" << endl;
  if (theService) delete theService;

}

// The Main...Aanlysis...

void MuonSeedValidator::analyze(const Event& event, const EventSetup& eventSetup)
{
  //Get the CSC Geometry :
  theService->update(eventSetup);

  ESHandle<CSCGeometry> cscGeom;
  eventSetup.get<MuonGeometryRecord>().get(cscGeom);

  //Get the DT Geometry :
  ESHandle<DTGeometry> dtGeom;
  eventSetup.get<MuonGeometryRecord>().get(dtGeom);

  // Get the RecHits collection :
  Handle<CSCRecHit2DCollection> csc2DRecHits; 
  event.getByLabel(recHitLabel, csc2DRecHits);

  // Get the CSC Segments collection :
  Handle<CSCSegmentCollection> cscSegments; 
  event.getByLabel(cscSegmentLabel, cscSegments);

  // Get the DT RecHits collection :
  Handle<DTRecHitCollection> dt1DRecHits; 
  event.getByLabel(dtrecHitLabel, dt1DRecHits);

  // Get the DT Segments collection :
  Handle<DTRecSegment4DCollection> dt4DSegments;
  event.getByLabel(dtSegmentLabel, dt4DSegments);

  // Get the SimHit collection :
  Handle<PSimHitContainer> csimHits;
  event.getByLabel(simHitLabel,"MuonCSCHits", csimHits);  
  Handle<PSimHitContainer> dsimHits;
  event.getByLabel(simHitLabel,"MuonDTHits", dsimHits);  
 
  // Get the SimTrack
  Handle<SimTrackContainer> simTracks;
  event.getByLabel(simTrackLabel, simTracks);

  // Get the muon seeds
  Handle<TrajectorySeedCollection> muonseeds;
  event.getByLabel(muonseedLabel, muonseeds);
  // Get sta muon tracks
  Handle<TrackCollection> standAloneMuons;
  event.getByLabel(staTrackLabel, standAloneMuons);
  // Get global muon tracks
  Handle<TrackCollection> globalMuons;
  event.getByLabel(glbTrackLabel, globalMuons);

  // Magnetic field
  ESHandle<MagneticField> field;
  eventSetup.get<IdealMagneticFieldRecord>().get(field);

  H2DRecHit1 *histo1 = 0;   
  H2DRecHit2 *histo2 = 0;   
  H2DRecHit3 *histo3 = 0;   
  H2DRecHit4 *histo4 = 0;   
  H2DRecHit5 *histo5 = 0;   
  TNtuple1 *tt = 0;
 
  // Get sim track information  
  // return  theta_v, theta_p, phi_v, phi_p :  theta and phi of position and momentum 
  //                                           of first simhit from CSC and DT  
  //         theQ, eta_trk, phi_trk, pt_trk, trackId : Q,eta,phi,pt and trackID of simtrack
  //         pt_layer,palayer : the pt or pa in each layer
  SimInfo(simTracks,dsimHits,csimHits,dtGeom,cscGeom);

  /// A. statistic information for segment, seed and sta muon w.r.t eta
  // 2. Reading the damn seeds
  RecSeedReader(muonseeds);

  // 3. Get muon track information ; sta=0, glb=1
  StaTrackReader(standAloneMuons, 0);
  
  //// associate seeds with sim tracks 
  //// seed_trk => the simtrack number j associate with seed i 
  std::vector<int> seeds_simtrk(seed_gp.size(), -1);
  for (unsigned int i=0; i < seed_gp.size(); i++) {
    double dR1= 99.0 ;
    int preferTrack = -1;
    for (unsigned int j=0; j < theta_p.size(); j++) {
      double dt = fabs(seed_gp[i].theta() - theta_p[j]) ; 
      double df = fabs(seed_gp[i].phi() - phi_p[j]) ;
      if (df > (6.283 - dfMax) ) df = 6.283 - df ;
      double dR2 = sqrt( dt*dt + df*df ) ;

      if (  dR2 < dR1 && dt < dtMax && df < dfMax ) {
         preferTrack = static_cast<int>(j) ;
         dR1 = dR2;
      }
    }
    seeds_simtrk[i] = preferTrack ;
  }

  //// associate sta with sim tracks  
  //// sta_simtrk => the simtrack number j associate with sta  i 
  std::vector<int> sta_simtrk(sta_thetaV.size(), -1);
  for (unsigned int i=0; i < sta_thetaV.size(); i++) {
    double dR1 = 99.0 ;
    int preferTrack = -1;
    for (unsigned int j=0; j < theta_p.size(); j++) {
      double dt   = fabs(sta_thetaP[i] - theta_p[j]) ; 
      double df   = fabs(sta_phiP[i] - phi_p[j]) ;
      if (df > (6.283 - dfMax) ) df = 6.283 - df ;
      double dR2  = sqrt( dt*dt + df*df ) ;
      if ( dR2 < dR1  && dt < dtMax && df < dfMax ) {
         preferTrack = static_cast<int>(j) ;
         dR1 = dR2;
      }
    }
    sta_simtrk[i] = preferTrack ;
  }


  // look at the number of reconstructed seed/sta for each sim-track
  histo1 = h_all;
  int sta_Evt  =0;
  int seed_Evt =0;
  for (size_t i=0; i < theta_p.size(); i++) {

      int nu_seeds_trk =0;
      for(size_t j=0; j < seeds_simtrk.size(); j++){
         if ( seeds_simtrk[j]== static_cast<int>(i) ) nu_seeds_trk++;
         if ( nu_seeds_trk > 19 ) nu_seeds_trk = 19;
      }

      int nu_sta_trk =0;
      for(size_t j=0; j < sta_simtrk.size(); j++){
         if ( sta_simtrk[j]== static_cast<int>(i) ) nu_sta_trk++;
      }
      std::vector<double> pa_tmp = palayer[i] ; 
      histo1->Fill1b( getEta(theta_p[i]), nu_seeds_trk, nu_sta_trk, pt_trk[i],
                      theQ[i]/pt_trk[i], theQ[i]/pa_tmp[0] );

      if (nu_seeds_trk > 0) { seed_Evt++ ; }
      if (nu_sta_trk > 0)   { sta_Evt++  ; } 
  }
 
  histo1->Fill1o( sta_mT.size(), seed_mT.size() );

  // look at those un-associated seeds and sta
  histo5 = h_UnRel;
  for (size_t i=0; i < seeds_simtrk.size(); i++ ) {
      // un-related !
      if ( seeds_simtrk[i]== -1 && seed_Evt != 0) { 
           histo5->Fill5a( seed_gp[i].eta(), seed_mT[i] );
      }
      // Orphan
      if ( seeds_simtrk[i]== -1 && seed_Evt == 0 ) {
         for (size_t j=0; j < theta_p.size(); j++ ) { 
             histo5->Fill5b( seed_gp[i].eta(), getEta(theta_p[j]), seed_mT[i] );
         }
      }
  }
  for (size_t i=0; i < sta_simtrk.size(); i++ ) {
      // un-related !
      if ( sta_simtrk[i]== -1 && sta_Evt != 0 ) {
           histo5->Fill5c( getEta(sta_thetaV[i]), sta_mT[i] );
      }
      // Orphan
      if ( sta_simtrk[i]== -1 && sta_Evt == 0 ) {
         for (unsigned int j=0; j < theta_p.size(); j++ ) { 
             histo5->Fill5d( getEta(sta_thetaV[i]), getEta(theta_p[j]), sta_mT[i], sta_phiV[i], phi_v[j]  );
         }
      }
  }
 
  /// B. seeds  Q/PT pulls
  if (nu_seed > 0) {
    int bestSeed = -1;
    double dSeedPT = 99999.9 ;
    // look the seed for every sta tracks
    for (unsigned int i=0; i < eta_trk.size(); i++) {

        // find the best seed whose pt is closest to a simTrack which associate with 
        for(unsigned int j=0; j < seeds_simtrk.size(); j++){
           if ( ( seeds_simtrk[j] == static_cast<int>(i) ) && ( fabs(seed_mT[j] - pt_trk[i]) < dSeedPT ) ){
              dSeedPT = fabs( seed_mT[j] - pt_trk[i] );
              bestSeed = static_cast<int>(j);
           }
        }

        for(unsigned int j=0; j < seeds_simtrk.size(); j++){

           if (  seeds_simtrk[j]!= static_cast<int>(i) ) continue;

           // put the rest seeds in the minus side 
           double bestSeedPt = 99999.9;
           if (bestSeed == static_cast<int>(j) ) {
              bestSeedPt = seed_mT[j];
           }else {
              bestSeedPt = -1.0*seed_mT[j];
           }

           std::vector<double> pa1 = palayer[i];
           std::vector<double> pt1 = ptlayer[i];
	   double pull_qbp   = ( qbp[j]  - (theQ[j]/pa1[ seed_layer[j] ]) ) / err_qbp[j] ;
	   double pull_qbpt  = ( qbpt[j] - (theQ[j]/pt1[ seed_layer[j] ]) ) / err_qbpt[j] ;
           //double resol_qbpt = ( qbpt[j] - (theQ[j]/pt_trk[j]) ) / (theQ[j]/pt_trk[j]) ;
           double resol_qbpt = ( qbpt[j] - (theQ[j]/pt1[ seed_layer[j] ]) ) / (theQ[j]/pt1[ seed_layer[j] ]) ;
           double ptLoss =  pt1[ seed_layer[j] ] / pt1[0] ;
	   histo1->Fill1g( seed_mT[j], seed_mA[j], bestSeedPt, seed_gp[j].eta(), ptLoss , pt1[0]) ;
	   histo1->Fill1i( pull_qbp, seed_gp[j].eta(), qbpt[j], pull_qbpt, err_qbp[j], err_qbpt[j], resol_qbpt);
	   histo1->Fill1f( seed_gp[j].eta(), err_dx[j], err_dy[j], err_x[j], err_y[j]);

        }
    }

  }

  /// C. sta track information
  if (nu_sta > 0 ) {
    double dPT = 9999999.9 ;
    int best = 0;
    double expectPT = -1.0;
    for (unsigned int i=0; i < eta_trk.size(); i++) {
        // find the best sta whose pt is closest to a simTrack which associate with 
        for(unsigned int j=0; j < sta_simtrk.size(); j++){
           if ( ( sta_simtrk[j]==static_cast<int>(i) ) && ( fabs(sta_mT[j] - pt_trk[i]) < dPT ) ){
              dPT = fabs( seed_mT[j] - pt_trk[i] );
              best = j;
              std::vector<double> pt1 = ptlayer[i];
              //expectPT = pt1[1];
              expectPT = pt_trk[i];
           }
        }
       // looking for the sta muon which is closest to simTrack pt 
       if (expectPT > 0) {
          double sim_qbpt = theQ[best]/expectPT;
          double resol_qbpt = (sta_qbpt[best] - sim_qbpt ) / sim_qbpt ;
          /*
          double sta_q = 1.0;
          if ( sta_qbpt[best] < 0 ) sta_q = -1.0;
          double sta_qbmt = sta_q / sta_mT[best];
          double resol_qbpt = (sta_qbmt - sim_qbpt ) / sim_qbpt ;
          */
          histo1->Fill1j(getEta(theta_p[i]), sta_qbp[best], sta_qbpt[best], sta_mT[best], sta_mA[best], pt_trk[i], resol_qbpt );
       }
    }


    // look at the dPhi, dEta, dx, dy from sim-segment and reco-segment of a seed
    for (unsigned int i=0; i<nSegInSeed.size(); i++) {

        int ii = seeds_simtrk[i]  ;
        std::vector<SimSegment> sCSC_v = recsegSelector->Sim_CSCSegments( trackID[ii], csimHits, cscGeom);
        std::vector<SimSegment> sDT_v  = recsegSelector->Sim_DTSegments( trackID[ii], dsimHits, dtGeom);

        SegOfRecSeed(muonseeds,i,sCSC_v,sDT_v); 
        for ( unsigned int j=0; j<geoID.size(); j++  ) {
            if (geoID[j].subdetId() ==  MuonSubdetId::DT ) {
               DTChamberId MB_Id = DTChamberId( geoID[j]  );
               histo1->Fill1e(-1*MB_Id.station() ,d_h[j],d_f[j],d_x[j],d_y[j]); 
            }
            if (geoID[j].subdetId() ==  MuonSubdetId::CSC ) {
               CSCDetId  ME_Id = CSCDetId( geoID[j] );
               histo1->Fill1e(ME_Id.station(),d_h[j],d_f[j],d_x[j],d_y[j]); 
            }
        }
    }
  }

  /// open a scope to check all information
  if ( scope ) { cout <<" ************ pT-eta scope *************** " <<endl; }
  for ( size_t i=0; i < theta_p.size(); i++ ) {
    
     if ( fabs( getEta(theta_p[i]) ) < eta_Low || fabs( getEta(theta_p[i]) ) > eta_High ) continue;


        histo4 = h_Scope;
        // check the seed phi eta dirstribition
        for ( size_t j=0; j< seeds_simtrk.size(); j++ ) { 
            if ( seeds_simtrk[j] != static_cast<int>(i) ) continue;
            if ( seed_mT[j] < pTCutMin || seed_mT[j] > pTCutMax   ) continue;

            histo4->Fill4b( seed_gp[j].phi(), seed_gp[j].eta(), getEta(theta_p[i])  );

         // look at the dPhi, dEta, dx, dy from sim-segment and reco-segment of a seed

            if (scope) {
               cout <<" "<<j<<"st seed w/ "<<nSegInSeed[j]<<" segs";
	       cout <<" & pt= "<<seed_mT[j]<<" +/- "<<err_qbp[j]*seed_mT[j]*seed_mA[j];
	       cout <<" q/p= "<<qbp[j]<<" @ "<<seed_layer[j];
	       cout <<" dx= "<<err_dx[j]<<" dy= "<<err_dy[j]<<" x= "<<err_x[j]<<" y= "<<err_y[j]<<endl;
            }
    
            // check the segments of the seeds in the scope
            bool flip_debug = false;
            if (!debug)  {
               debug = true;
               flip_debug = true;
            }

            std::vector<SimSegment> sCSC_v = recsegSelector->Sim_CSCSegments( trackID[i], csimHits, cscGeom);
            std::vector<SimSegment> sDT_v  = recsegSelector->Sim_DTSegments( trackID[i], dsimHits, dtGeom);
	    SegOfRecSeed(muonseeds,j,sCSC_v,sDT_v);

            if (flip_debug) {
               debug = false;
            }
            // check the dEta, dPhi, dx, dy resolution
            for ( size_t k=0; k<geoID.size(); k++  ) {
	        if (geoID[k].subdetId() ==  MuonSubdetId::DT ) {
      	           DTChamberId MB_Id = DTChamberId( geoID[k]  );
	           histo4->Fill4c( -1*MB_Id.station(), d_h[k], d_f[k], d_x[k], d_y[k] ); 
                }
        	if (geoID[k].subdetId() ==  MuonSubdetId::CSC ) {
	           CSCDetId  ME_Id = CSCDetId( geoID[k] );
                   histo4->Fill4c(  ME_Id.station(), d_h[k], d_f[k], d_x[k], d_y[k] ); 
                }
	    }

        }
        // check the sta information 
        for ( size_t j=0; j< sta_simtrk.size(); j++ ) { 

            if ( sta_simtrk[j] != static_cast<int>(i) ) continue;
            if ( sta_mT[j] < pTCutMin || sta_mT[j] > pTCutMax   ) continue;

            histo4->Fill4a( sta_phiV[j], getEta(sta_thetaV[j]), getEta(theta_p[i])  );

            // look at the sta pt vs. #_of_hits and chi2
            double ndf = static_cast<double>(2*sta_nHits[j]-4);
            double chi2_ndf = sta_chi2[j]/ndf;
            histo4->Fill4d(sta_mT[j], sta_nHits[j], chi2_ndf);

            if (scope) {
               cout <<" sta_pt= "<<sta_mT[j]<<" q/p= "<<sta_qbp[j]<<" w/"<<sta_nHits[j] <<endl;
            }
            //cout <<"************************************************"<<endl;
            //cout <<"  "<<endl;
        }

  }

  // D. fail sta cases
  if (nu_sta == 0 && nu_seed!=0) {
     double sim_eta = -9.0 ; 
     for (int i=0; i < nu_seed; i++) {

         histo3 = h_NoSta;
         double pull_qbpt = ( qbpt[i] - (theQ[0]/pt_trk[0]) ) / err_qbpt[i] ;
         double pt_err = err_qbpt[i]*seed_mT[i]*seed_mT[i];
         //if (seeds_simtrk[i] != -1 )  sim_eta = eta_trk[ seeds_simtrk[i] ] ; 
         if (seeds_simtrk[i] != -1 )  sim_eta = getEta( theta_p[ seeds_simtrk[i] ] ) ; 
         histo3->Fill3a( sim_eta , seed_gp[i].eta(), nu_seed, seed_mT[i] , pt_err, pull_qbpt);
    
         CSCsegment_stat(cscSegments, cscGeom, seed_gp[i].theta(), seed_gp[i].phi() );
         DTsegment_stat(dt4DSegments, dtGeom,  seed_gp[i].theta(), seed_gp[i].phi() );
         // compare segments of seed with seed itself
         SegOfRecSeed(muonseeds,i); 

         int allseg1 = cscseg_stat1[5]+dtseg_stat1[5]; // # of stations which have good segments
         int allseg  = cscseg_stat[0]+dtseg_stat[0];   // # of segments
         histo3->Fill3b(sim_eta,allseg1,allseg );

         for (unsigned j=0; j<d_h.size(); j++) {
             histo3->Fill3d(sim_eta,d_h[j],d_f[j]);
         }

         int types = RecSegReader(cscSegments,dt4DSegments,cscGeom,dtGeom, seed_gp[i].eta(), seed_gp[i].phi());
         if (debug) cout<<" seed type for fail STA= "<<types<<endl;
         for (unsigned i=0; i< phi_resid.size(); i++) {
             histo3->Fill3c(phi_resid[i],eta_resid[i]);
         }
     }
     
  }
  
  //cout <<" ================================================== "<<endl; 
  //cout <<" # of seeds: "<<nu_seed<<" # of STA: "<<nu_sta<<endl;
  // Basic simulation and reco information
  int idx=0;
  for (SimTrackContainer::const_iterator stk = simTracks->begin(); stk != simTracks->end(); stk++)
  {

      bool rechitSize = (dsimHits->size() < 8 && csimHits->size() < 4 )? true:false ;
      if (abs((*stk).type())!=13 || rechitSize || (*stk).vertIndex() != 0 ) continue;

      // 1. Run the class SegSelector
      int trkId = static_cast<int>( (*stk).trackId() );
      std::vector<SimSegment>     sCSC_v = recsegSelector->Sim_CSCSegments( trkId, csimHits, cscGeom);
      std::vector<SimSegment>     sDT_v = recsegSelector->Sim_DTSegments( trkId, dsimHits, dtGeom);
      std::vector<CSCSegment>     cscseg_V = recsegSelector->Select_CSCSeg(cscSegments,cscGeom, sCSC_v);
      std::vector<DTRecSegment4D> dtseg_V = recsegSelector->Select_DTSeg(dt4DSegments,dtGeom, sDT_v);

      // 2. Reading the reco segmenst
      //   get the appropriate eta and phi to select reco-segments
      double px = ((*stk).momentum()).x();
      double py = ((*stk).momentum()).y();
      double pz = ((*stk).momentum()).z();
      double pt = sqrt(px*px + py*py);
      double pa = sqrt(px*px + py*py + pz*pz);
       
      //cout<<" <<<<< Idx : "<<idx<<" >>>>>>>>>>>>>>>>>>>>> "<<endl;
      //cout<<"  eta from SimTrack= "<<getEta(px,py,pz)<<"  theta = "<<theta<<"("<<(180.0*theta)/3.1415 <<")"<<endl;
 
      // 3. Read out reco segments
      //   return : the ave_phi, ave_eta, phi_resid, eta_resid, dx_error, dy_error, x_error, y_error 
      int types = RecSegReader(cscSegments,dt4DSegments,cscGeom,dtGeom,theta_p[idx],phi_p[idx]);

      // 4. Check # of segments and rechits in each chambers for this track
      CSCsegment_stat(cscSegments, cscGeom, theta_p[idx], phi_p[idx]);
      DTsegment_stat(dt4DSegments, dtGeom, theta_p[idx], phi_p[idx]);
      Simsegment_stat(sCSC_v,sDT_v);
      // 5. if less than 1 segment, check the # of rechits in each chambers for this track
      if ( (cscseg_stat[5] < 2)&&(dtseg_stat[5] < 2) ) {
         CSCRecHit_Stat(csc2DRecHits, cscGeom, getEta(theta_p[idx]), phi_p[idx]);
         DTRecHit_Stat(dt1DRecHits, dtGeom, getEta(theta_p[idx]), phi_p[idx]);
      }

      // seg_stat[0] = total segments in all stations
      // seg_stat[5] = the number of stations which have segments
      // seg_stat1[x] = the number of stations/segments which only count segments w/ more than 4 rechits
      int layer_sum  = cscseg_stat1[5] + dtseg_stat1[5];
      int seg_sum    = cscseg_stat[0]  + dtseg_stat[0];
      int leftrh_sum = cscrh_sum[5]    + dtrh_sum[5];
      //cout <<"  sims:"<<simseg_sum<<" seg:"<<seg_sum<<" layer:"<<layer_sum<<endl;

      histo1 = h_all;
      // look at all information
      histo1->Fill1(layer_sum, simseg_sum, getEta(theta_p[idx]) );
      // 6. pt vs. # of segments in a event
      histo1->Fill1c(pt, pa, cscseg_stat[0]+dtseg_stat[0]);
      // 7. look at those events without any reco segments => no seed can be produced!!
      if ((cscseg_stat[5]==0)&&(dtseg_stat[5]==0)) { 

	 histo1->Fill1a(leftrh_sum, getEta(theta_p[idx]));
         if (cscrh_sum[0] !=0 ) {
            histo2 = h_NoSeed;
            histo2 -> Fill2b(getEta(theta_p[idx]), cscrh_sum[0]);
         }
         if (dtrh_sum[0] !=0 ) {
            histo2 = h_NoSeed;
            histo2 -> Fill2c(getEta(theta_p[idx]), dtrh_sum[0]);
         }
      }

      if ( nu_seed == 0 ) {
         //cout<<" seed type for no seed : "<<types<<" h: "<< getEta(px,py,pz) <<" w/ seg# "<<layer_sum<<endl; 
         //cout<<" idx : "<<idx<<endl;
         histo2 = h_NoSeed;
         histo2->Fill2a( getEta(px,py,pz), layer_sum, seg_sum , simseg_sum, 
                        simseg_sum - seg_sum, simseg_sum - layer_sum );
      }
      idx++;
  }


}

// ********************************************
// ***********  Utility functions  ************
// ********************************************

// number of csc segments in one chamber for each station
// cscseg_stat[0] = total segments in all stations
// cscseg_stat[5] = the number of stations which have segments
void MuonSeedValidator::CSCsegment_stat( Handle<CSCSegmentCollection> cscSeg , ESHandle<CSCGeometry> cscGeom, double trkTheta, double trkPhi) {

     for (int i=0; i<6; i++) {
         cscseg_stat[i]=0;
         cscseg_stat1[i]=0;
     }
     for(CSCSegmentCollection::const_iterator seg_It = cscSeg->begin(); seg_It != cscSeg->end(); seg_It++)
     { 
        CSCDetId DetId = (CSCDetId)(*seg_It).cscDetId();
	const CSCChamber* cscchamber = cscGeom->chamber( DetId );
	GlobalPoint  gp = cscchamber->toGlobal((*seg_It).localPosition() );
	GlobalVector gv = cscchamber->toGlobal((*seg_It).localDirection() );
        if (( fabs(gp.theta()- trkTheta) > dtMax  ) || ( fabs(gv.phi()- trkPhi) > dfMax ) ) continue;

        cscseg_stat[DetId.station()] += 1;
        if ((*seg_It).nRecHits() > 3 ) {
           cscseg_stat1[DetId.station()] += 1;
        }
     }
     cscseg_stat[0] = cscseg_stat[1]+cscseg_stat[2]+cscseg_stat[3]+cscseg_stat[4];
     cscseg_stat1[0] = cscseg_stat1[1]+cscseg_stat1[2]+cscseg_stat1[3]+cscseg_stat1[4];
     for (int i =1; i<5; i++){
         if(cscseg_stat[i]!=0)  { cscseg_stat[5]++  ;}
         if(cscseg_stat1[i]!=0) { cscseg_stat1[5]++ ;}
     }
     
}
// number of dt segments in one chamber for each station
void MuonSeedValidator::DTsegment_stat( Handle<DTRecSegment4DCollection> dtSeg, ESHandle<DTGeometry> dtGeom, double trkTheta, double trkPhi)  {

     for (int i=0; i<6; i++) {
         dtseg_stat[i]=0;
         dtseg_stat1[i]=0;
     }
     for(DTRecSegment4DCollection::const_iterator seg_It = dtSeg->begin(); seg_It != dtSeg->end(); seg_It++)
     { 
        if ( !(*seg_It).hasPhi() || !(*seg_It).hasZed()  ) continue;
        DTChamberId DetId = (*seg_It).chamberId();
        const DTChamber* dtchamber = dtGeom->chamber( DetId );
        GlobalPoint  gp = dtchamber->toGlobal( (*seg_It).localPosition() );
        GlobalVector gv = dtchamber->toGlobal( (*seg_It).localDirection() );
        if ( ( fabs(gp.theta()- trkTheta) > dtMax  ) || ( fabs(gv.phi()- trkPhi) > dfMax ) ) continue;

        dtseg_stat[DetId.station()] += 1;
        int n_phiHits = ((*seg_It).phiSegment())->specificRecHits().size();
        if ( (*seg_It).hasZed() && (n_phiHits > 4) ) {
           dtseg_stat1[DetId.station()] += 1;
        }
     }
     dtseg_stat[0]  = dtseg_stat[1] +dtseg_stat[2] +dtseg_stat[3] +dtseg_stat[4];
     dtseg_stat1[0] = dtseg_stat1[1]+dtseg_stat1[2]+dtseg_stat1[3]+dtseg_stat1[4];
     for (int i =1; i<5; i++){
         if(dtseg_stat[i]!=0)  { dtseg_stat[5]++ ;}
         if(dtseg_stat1[i]!=0) {
            if ( i !=4 ) { dtseg_stat1[5]++ ;}
            // because no eta/Z measurement at station 4
            if ((i==4)&&(dtseg_stat[4]!=0)) { dtseg_stat1[5]++ ;} 
         }
     }
}

// number of sim segments in one chamber for each station
void MuonSeedValidator::Simsegment_stat( std::vector<SimSegment> sCSC, std::vector<SimSegment> sDT ) {

     for (int i=0; i<6; i++) {
         simcscseg[i] = 0;
         simdtseg[i]=0;
     }

     double ns1 =0.0;
     double eta_sim1 =0;
     for (std::vector<SimSegment>::const_iterator it = sCSC.begin(); it != sCSC.end(); it++) {
	     int st = ((*it).csc_DetId).station();
	     eta_sim1 += ((*it).sGlobalOrg).eta();
	     simcscseg[st]++;
	     ns1++;
     }
     simcscseg[0]=simcscseg[1]+simcscseg[2]+simcscseg[3]+simcscseg[4];
     for (int i=1; i<5; i++) {
	 if (simcscseg[i]!=0)  simcscseg[5]++; 
     }
     eta_sim1 = eta_sim1/ns1 ;

     double ns2 =0.0;
     double eta_sim2 =0;
     for (std::vector<SimSegment>::const_iterator it = sDT.begin(); it != sDT.end(); it++) {
	     int st = ((*it).dt_DetId).station();
	     eta_sim2 += ((*it).sGlobalOrg).eta();
	     simdtseg[st]++;
	     ns2++;
     }
     simdtseg[0]=simdtseg[1]+simdtseg[2]+simdtseg[3]+simdtseg[4];
     for (int i=1; i<5; i++) {
	 if (simdtseg[i]!=0) simdtseg[5]++; 
     }
     eta_sim2 = eta_sim2/ns2 ;

     simseg_sum = simcscseg[5]+ simdtseg[5];
     simseg_eta = -9.0;
     if      ((simcscseg[0]==0)&&(simdtseg[0]!=0)) { simseg_eta = eta_sim2; }
     else if ((simdtseg[0]==0)&&(simcscseg[0]!=0)) { simseg_eta = eta_sim1; }
     else { simseg_eta = (eta_sim1 + eta_sim2)/2.0 ; }
}

void MuonSeedValidator::CSCRecHit_Stat(Handle<CSCRecHit2DCollection> cscrechit, ESHandle<CSCGeometry> cscGeom, double trkEta, double trkPhi){
     for (int i=0; i <6; i++) {
         cscrh_sum[i]=0;
     }
     for(CSCRecHit2DCollection::const_iterator r_it = cscrechit->begin(); r_it != cscrechit->end(); r_it++)
     { 
        CSCDetId det_id = (CSCDetId)(*r_it).cscDetId();
	const CSCChamber* cscchamber = cscGeom->chamber( det_id );
	GlobalPoint gp = cscchamber->toGlobal((*r_it).localPosition() );
        if (( fabs(gp.eta()- trkEta) > dtMax  ) || ( fabs(gp.phi()- trkPhi) > dfMax ) ) continue;

        cscrh_sum[det_id.station()]++;
     }
     cscrh_sum[0] = cscrh_sum[1]+cscrh_sum[2]+cscrh_sum[3]+cscrh_sum[4];
     for (int i =1; i<5; i++){
         if(cscrh_sum[i]!=0) {
            cscrh_sum[5]++ ;
         }
     }
}

void MuonSeedValidator::DTRecHit_Stat(Handle<DTRecHitCollection> dtrechit, ESHandle<DTGeometry> dtGeom, double trkEta, double trkPhi){

     //double phi[4]={999.0};
     for (int i=0; i <6; i++) {
         dtrh_sum[i]=0;
     }

     double eta=-9.0;
     double nn=0.0;
     for (DTRecHitCollection::const_iterator r_it = dtrechit->begin(); r_it != dtrechit->end(); r_it++){
         DTWireId det_id = (*r_it).wireId();
         const DTChamber* dtchamber = dtGeom->chamber( det_id );
         LocalPoint lrh = (*r_it).localPosition();
         GlobalPoint grh = dtchamber->toGlobal( lrh );
         if ( ( fabs(grh.eta()- trkEta) > dtMax  ) || ( fabs(grh.phi()- trkPhi) > dfMax ) ) continue;

         dtrh_sum[det_id.station()]++;
         eta += grh.eta();
         nn += 1.0;
     }
     eta = eta/nn ;

     dtrh_sum[0] = dtrh_sum[1]+dtrh_sum[2]+dtrh_sum[3]+dtrh_sum[4];
     for (int i =1; i<5; i++){
         if (dtrh_sum[i]!=0) {
            dtrh_sum[5]++ ;
         }
     }
}

int MuonSeedValidator::ChargeAssignment(GlobalVector Va, GlobalVector Vb){
     int charge = 0;
     float axb = ( Va.x()*Vb.y() ) - ( Vb.x()*Va.y() );
     if (axb != 0.0) {
        charge = ( (axb > 0.0) ?  1:-1 ) ;
     }
     return charge;
}

void MuonSeedValidator::RecSeedReader( Handle<TrajectorySeedCollection> rec_seeds ){

     nu_seed = 0;
     seed_gp.clear();
     seed_gm.clear();
     seed_lp.clear();
     seed_lv.clear();
     qbp.clear();
     qbpt.clear();
     err_qbp.clear();
     err_qbpt.clear();
     err_dx.clear();
     err_dy.clear();
     err_x.clear();
     err_y.clear();
     seed_mT.clear();
     seed_mA.clear();
     seed_layer.clear();
     nSegInSeed.clear();

     TrajectorySeedCollection::const_iterator seed_it;
     for (seed_it = rec_seeds->begin(); seed_it !=  rec_seeds->end(); seed_it++) {
         PTrajectoryStateOnDet pTSOD = (*seed_it).startingState();

         nSegInSeed.push_back( (*seed_it).nHits() );
         // Get the tsos of the seed
         TrajectoryStateTransform tsTransform;
         DetId seedDetId(pTSOD.detId());
         const GeomDet* gdet = theService->trackingGeometry()->idToDet( seedDetId );
         TrajectoryStateOnSurface seedTSOS = tsTransform.transientState(pTSOD, &(gdet->surface()), &*theService->magneticField());
         // seed global position and momentum(direction)
         seed_gp.push_back( seedTSOS.globalPosition() );
         seed_gm.push_back( seedTSOS.globalMomentum() );
         seed_lp.push_back( seedTSOS.localPosition()  );
         seed_lv.push_back( seedTSOS.localDirection() );
        
         LocalTrajectoryParameters seed_para = pTSOD.parameters();

         // the error_vector v[15]-> [0] , [2] , [5] , [9] , [14]
         //                          q/p   dx    dy     x      y
         std::vector<float> err_mx = pTSOD.errorMatrix();
         err_qbp.push_back( sqrt(err_mx[0]) );
         err_dx.push_back( sqrt(err_mx[2])  );
         err_dy.push_back( sqrt(err_mx[5])  );
         err_x.push_back(  sqrt(err_mx[9])  );
         err_y.push_back(  sqrt(err_mx[14]) );
         //for (unsigned i=0; i< err_mx.size(); i++) {
         //    cout <<"Err"<<i<<" = "<<err_mx[i]<<"  -> "<<sqrt(err_mx[i])<<endl;
         //}

         // seed layer
         DetId pdid(pTSOD.detId()); 
         if ( pdid.subdetId()  == MuonSubdetId::DT ) {
            DTChamberId MB_Id = DTChamberId( pTSOD.detId() );
            seed_layer.push_back( MB_Id.station() );
         }
         if ( pdid.subdetId()  == MuonSubdetId::CSC ) {
            CSCDetId  ME_Id = CSCDetId( pTSOD.detId() );
            seed_layer.push_back( ME_Id.station() );
         }
         double seed_mx = seed_gm[nu_seed].x();
         double seed_my = seed_gm[nu_seed].y();
         double seed_mz = seed_gm[nu_seed].z();
         double seed_q  = 1.0*seed_para.charge();
         double seed_sin = sqrt((seed_mx*seed_mx)+(seed_my*seed_my)) / sqrt((seed_mx*seed_mx)+(seed_my*seed_my)+(seed_mz*seed_mz));
         // seed pt, pa, and q/pt or q/pa
         seed_mT.push_back( sqrt((seed_mx*seed_mx)+(seed_my*seed_my)) );
         seed_mA.push_back(  sqrt((seed_mx*seed_mx)+(seed_my*seed_my)+(seed_mz*seed_mz)) );
         qbpt.push_back( seed_q / sqrt((seed_mx*seed_mx)+(seed_my*seed_my)) );
         qbp.push_back( seed_para.signedInverseMomentum() );
         err_qbpt.push_back( err_qbp[nu_seed]/seed_sin );
         nu_seed++;
         if (debug) { cout <<" seed pt: "<<sqrt((seed_mx*seed_mx)+(seed_my*seed_my)) <<endl; }

     }
}

 
// read the segments associated with the seed and compare with seed
void MuonSeedValidator::SegOfRecSeed( Handle<TrajectorySeedCollection> rec_seeds, int seed_idx){
 
     int idx = 0;
     d_h.clear();
     d_f.clear();
     d_x.clear();
     d_y.clear();
     geoID.clear();

     TrajectorySeedCollection::const_iterator seed_it;
     for (seed_it = rec_seeds->begin(); seed_it !=  rec_seeds->end(); seed_it++) {
 
         idx++; 
         if (seed_idx != (idx-1) ) continue;
 
         PTrajectoryStateOnDet pTSOD = (*seed_it).startingState();
         TrajectoryStateTransform tsTransform;
         DetId seedDetId(pTSOD.detId());
         const GeomDet* seedDet = theService->trackingGeometry()->idToDet( seedDetId );
         TrajectoryStateOnSurface seedTSOS = tsTransform.transientState(pTSOD, &(seedDet->surface()), &*theService->magneticField());
         GlobalPoint seedgp = seedTSOS.globalPosition();

         for (edm::OwnVector<TrackingRecHit>::const_iterator rh_it = seed_it->recHits().first; rh_it != seed_it->recHits().second; rh_it++) {

             const GeomDet* gdet = theService->trackingGeometry()->idToDet( (*rh_it).geographicalId() );
             LocalPoint  lp = (*rh_it).localPosition();
             GlobalPoint gp = gdet->toGlobal( lp );
             LocalPoint slp = gdet->toLocal( gp );
             DetId pdid = (*rh_it).geographicalId() ;

             geoID.push_back( pdid );
             d_h.push_back( gp.eta() - seedgp.eta() ); 
             d_f.push_back( gp.phi() - seedgp.phi() ); 
             d_x.push_back( lp.x() - slp.x() );
             d_y.push_back( lp.y() - slp.y() );
         }

     }
}

// read the segments associated with the seed and compare with sim-segment
void MuonSeedValidator::SegOfRecSeed( Handle<TrajectorySeedCollection> rec_seeds, int seed_idx,
                                  std::vector<SimSegment> sCSC, std::vector<SimSegment> sDT ){
     int idx = 0;
     d_h.clear();
     d_f.clear();
     d_x.clear();
     d_y.clear();
     geoID.clear();

     TrajectorySeedCollection::const_iterator seed_it;
     for (seed_it = rec_seeds->begin(); seed_it !=  rec_seeds->end(); seed_it++) {
 
         idx++; 
         if (seed_idx != (idx-1) ) continue;
         if (debug && scope) {  cout<<" "<<endl; }
 
         for (edm::OwnVector<TrackingRecHit>::const_iterator rh_it = seed_it->recHits().first; rh_it != seed_it->recHits().second; rh_it++) {

             const GeomDet* gdet = theService->trackingGeometry()->idToDet( (*rh_it).geographicalId() );
             GlobalPoint gp = gdet->toGlobal( (*rh_it).localPosition() );
             LocalPoint  lp = (*rh_it).localPosition();
             DetId pdid = (*rh_it).geographicalId() ; 
             

             // for parameters [1]:dx/dz, [2]:dy/dz, [3]:x, [4]:y
             double dxz = (*rh_it).parameters()[0] ;
             double dyz = (*rh_it).parameters()[1] ;
             double dz  = 1.0 / sqrt( (dxz*dxz) + (dyz*dyz) + 1.0 );
             if ( pdid.subdetId()  == MuonSubdetId::DT ) {
                dz = -1.0*dz;
             }
             double dx  = dxz*dz;
             double dy  = dyz*dz;
             LocalVector lv = LocalVector(dx,dy,dz);
             GlobalVector gv = gdet->toGlobal( lv );

	     if ( pdid.subdetId()  == MuonSubdetId::CSC ) {
                double directionSign = gp.z() * gv.z();
		lv =  (directionSign*lv).unit();
		gv = gdet->toGlobal( lv );
             }

             if (debug && scope) { 
                cout<<"============= segs from seed ============== "<<endl;
             }

	     if ( pdid.subdetId()  == MuonSubdetId::DT ) {

	        DTChamberId MB_Id = DTChamberId( pdid );

                if (debug && scope) { cout <<"DId: "<< MB_Id <<endl; }

                // find the sim-reco match case and store the difference
                if ( sDT.size() ==0 ) {
                   geoID.push_back( pdid );
                   d_h.push_back( 999.0 );
                   d_f.push_back( 999.0 );
                   d_x.push_back( 999.0 );
                   d_y.push_back( 999.0 );
                }
                else {
                   bool match=false;
                   for (std::vector<SimSegment>::const_iterator it = sDT.begin(); it != sDT.end(); it++) {
                       if ( (*it).dt_DetId == MB_Id ) {
			  geoID.push_back( pdid );
			  d_h.push_back( gv.eta() - (*it).sGlobalVec.eta() );
			  d_f.push_back( gv.phi() - (*it).sGlobalVec.phi() );
			  d_x.push_back( lp.x() - (*it).sLocalOrg.x() );
			  d_y.push_back( lp.y() - (*it).sLocalOrg.y() );
                          match = true;
                       }
                   }
                   if (!match) {
                      geoID.push_back( pdid );
		      d_h.push_back( 999.0 );
		      d_f.push_back( 999.0 );
		      d_x.push_back( 999.0 );
		      d_y.push_back( 999.0 );
                   }
                }
	     }
	     if ( pdid.subdetId()  == MuonSubdetId::CSC ) {

		CSCDetId  ME_Id = CSCDetId( pdid ) ;
                if (debug && scope) { cout <<"DId: "<< ME_Id <<endl; }

                /*/ flip the z-sign for station 1 and 2 because of the chamber orientation
                if ( (ME_Id.station()==1)||(ME_Id.station()==2) ) {
                   gv = GlobalVector( gv.x(), gv.y(), (-1.*gv.z()) );
                }*/

                // find the sim-reco match case and store the difference
                if ( sCSC.size() ==0 ) {
                   geoID.push_back( pdid );
                   d_h.push_back( 999.0 );
                   d_f.push_back( 999.0 );
                   d_x.push_back( 999.0 );
                   d_y.push_back( 999.0 );
                }
                else {
                   bool match=false;
                   for (std::vector<SimSegment>::const_iterator it = sCSC.begin(); it != sCSC.end(); it++) {
                       if ( (*it).csc_DetId == ME_Id ) {
			  geoID.push_back( pdid );
			  d_h.push_back( gv.eta() - (*it).sGlobalVec.eta() );
			  d_f.push_back( gv.phi() - (*it).sGlobalVec.phi() );
			  d_x.push_back( lp.x() - (*it).sLocalOrg.x() );
			  d_y.push_back( lp.y() - (*it).sLocalOrg.y() );
                          match = true;
                       }
                   }
                   if (!match) {
                      geoID.push_back( pdid );
		      d_h.push_back( 999.0 );
		      d_f.push_back( 999.0 );
		      d_x.push_back( 999.0 );
		      d_y.push_back( 999.0 );
                   }
                }
	     }

             if (debug && scope) {
                cout<<"  h0= "<< gp.eta() <<"  f0= "<< gp.phi()<<"   gp= "<< gp << endl;
                cout<<"  h1= "<< gv.eta() <<"  f1= "<< gv.phi()<<"   gv= "<< gv << endl;
             }
         }

     }
}

void MuonSeedValidator::StaTrackReader( Handle<reco::TrackCollection> sta_trk, int sta_glb){

     // look at the inner most momentum and position
     nu_sta=0;

     sta_phiP.clear();
     sta_thetaP.clear();

     sta_mT.clear();
     sta_mA.clear();
     sta_thetaV.clear();
     sta_phiV.clear();
     sta_qbp.clear();
     sta_qbpt.clear();
     sta_chi2.clear();
     sta_nHits.clear();   

     TrackCollection::const_iterator iTrk;
     for ( iTrk = sta_trk->begin(); iTrk !=  sta_trk->end(); iTrk++) {
         nu_sta++;
 
         // get the inner poistion( eta & phi )
         math::XYZPoint staPos = (*iTrk).innerPosition();
         double posMag = sqrt( staPos.x()*staPos.x() + staPos.y()*staPos.y() + staPos.z()*staPos.z() );
         math::XYZVector staMom = (*iTrk).innerMomentum();
         double innerMt = sqrt( (staMom.x()*staMom.x()) + (staMom.y()*staMom.y()) );

         sta_phiP.push_back( atan2( staPos.y(), staPos.x() ) );
         sta_thetaP.push_back( acos( staPos.z()/posMag ) );

         cout<<" momentum= "<<(*iTrk).momentum()<<"  pt= "<<(*iTrk).pt()<<endl;
         cout<<" innerMom= "<<(*iTrk).innerMomentum()<<" iMt= "<<innerMt<<endl;

         sta_mA.push_back( (*iTrk).p() );
         //sta_mT.push_back( (*iTrk).pt() );
         sta_mT.push_back( innerMt );
         sta_thetaV.push_back( (*iTrk).theta() );
         sta_phiV.push_back( (*iTrk).phi() );
         sta_qbp.push_back( (*iTrk).qoverp() );
         sta_qbpt.push_back( ( (*iTrk).qoverp()/(*iTrk).pt() )*(*iTrk).p() );
         sta_chi2.push_back( (*iTrk).chi2() );
         sta_nHits.push_back( (*iTrk).recHitsSize() );

         if (debug) { 
            if (sta_glb==0) cout<<"sta";
            if (sta_glb==1) cout<<"glb";

            cout <<" track  pt: "<< (*iTrk).pt() <<endl; 
         }
     }
}


void MuonSeedValidator::SimInfo(Handle<edm::SimTrackContainer> simTracks,
                            Handle<edm::PSimHitContainer> dsimHits, Handle<edm::PSimHitContainer> csimHits,
                            ESHandle<DTGeometry> dtGeom, ESHandle<CSCGeometry> cscGeom){

  // theta and phi at inner-most layer of Muon System
  theta_p.clear();
  theta_v.clear();
  phi_p.clear();
  phi_v.clear();
  // basic sim track infomation 
  eta_trk.clear();
  theta_trk.clear();
  phi_trk.clear();
  theQ.clear();
  pt_trk.clear();
  ptlayer.clear();
  palayer.clear();
  trackID.clear();

  for (SimTrackContainer::const_iterator simTk_It = simTracks->begin(); simTk_It != simTracks->end(); simTk_It++)
  {

      //if (abs((*simTk_It).type())!=13 || (*simTk_It).vertIndex() != 0 ) continue;
      bool rechitSize = (dsimHits->size() <8 && csimHits->size() <4) ? true:false ;
      if (abs((*simTk_It).type())!=13 || rechitSize || (*simTk_It).vertIndex() != 0 ) continue;
    
      trackID.push_back( static_cast<int>((*simTk_It).trackId())  );
      if ((*simTk_It).type()==13) {
         theQ.push_back( -1.0 );
      }else {
         theQ.push_back(  1.0 );
      }

      std::vector<double> pt1(5,0.0);
      std::vector<double> pa1(5,0.0);

      double px = ((*simTk_It).momentum()).x();
      double py = ((*simTk_It).momentum()).y();
      double pz = ((*simTk_It).momentum()).z();
      pa1[0] = sqrt( px*px + py*py + pz*pz );
      pt1[0] = sqrt( px*px + py*py );

      eta_trk.push_back( getEta(px,py,pz)  );
      theta_trk.push_back( acos(pz/pa1[0])  );
      phi_trk.push_back( atan2(py,px) );
      pt_trk.push_back( pt1[0] );
   
      double enu2   = 0.0;
      for (PSimHitContainer::const_iterator ds_It = dsimHits->begin(); ds_It != dsimHits->end(); ds_It++)
      {          
          Local3DPoint lp = (*ds_It).localPosition(); 

          DTLayerId D_Id = DTLayerId( (*ds_It).detUnitId() );
          const DTLayer* dtlayer = dtGeom->layer(D_Id);
          GlobalVector m2 = dtlayer->toGlobal((*ds_It).momentumAtEntry() );
          GlobalPoint gp = dtlayer->toGlobal(lp );

          if ( ( abs((*ds_It).particleType())==13 ) && ( (*ds_It).trackId()==(*simTk_It).trackId() )) {
 
             pt1[ D_Id.station() ] = sqrt( (m2.x()*m2.x()) + (m2.y()*m2.y()) );
             pa1[ D_Id.station() ] = sqrt( (m2.x()*m2.x()) + (m2.y()*m2.y()) + (m2.z()*m2.z()) );
                       
             if ( enu2 == 0 ) { 
                theta_p.push_back( gp.theta() );
                theta_v.push_back( m2.theta() );
                phi_p.push_back( gp.phi() );
                phi_v.push_back( m2.phi() );
             }
             enu2  += 1.0;
          } 
      }

      double enu1   = 0.0;
      for (PSimHitContainer::const_iterator cs_It = csimHits->begin(); cs_It != csimHits->end(); cs_It++)
      {
          CSCDetId C_Id = CSCDetId((*cs_It).detUnitId());
          const CSCChamber* cscchamber = cscGeom->chamber( C_Id );
          GlobalVector m1 = cscchamber->toGlobal((*cs_It).momentumAtEntry() );
          Local3DPoint lp = (*cs_It).localPosition(); 
          GlobalPoint gp = cscchamber->toGlobal(lp );

          if ( ( abs((*cs_It).particleType())==13 ) && ( (*cs_It).trackId()==(*simTk_It).trackId() )) {

             if (enu2 == 0.0) {
                pt1[C_Id.station()] = sqrt( (m1.x()*m1.x()) + (m1.y()*m1.y()) ) ; 
                pa1[C_Id.station()] = sqrt( (m1.x()*m1.x()) + (m1.y()*m1.y()) + (m1.z()*m1.z()) );
                
                if ( enu1 == 0 ) { 
                   theta_p.push_back( gp.theta() );
                   theta_v.push_back( m1.theta() );
                   phi_p.push_back( gp.phi() );
                   phi_v.push_back( m1.phi() );
                }
             }
             enu1   += 1.0;
          }
      }

      ptlayer.push_back(pt1);
      palayer.push_back(pa1);
      cout<<" simTrk momentum= "<<(*simTk_It).momentum()<<" pa= "<<pa1[0]<<" pt= "<<pt1[0]<<endl;
      cout<<" simhit momentum= "<< pa1[1] <<" pt= "<<pt1[1]<<endl;
  }
}

// Look up what segments we have in a event
int MuonSeedValidator::RecSegReader( Handle<CSCSegmentCollection> cscSeg, Handle<DTRecSegment4DCollection> dtSeg                                , ESHandle<CSCGeometry> cscGeom, ESHandle<DTGeometry> dtGeom, double trkTheta, double trkPhi) {

     // Calculate the ave. eta & phi
     ave_phi = 0.0;
     ave_eta = 0.0;
     phi_resid.clear();
     eta_resid.clear();

     double n=0.0;
     double m=0.0;
     for(CSCSegmentCollection::const_iterator it = cscSeg->begin(); it != cscSeg->end(); it++)
     {
        if ( (*it).nRecHits() < 4) continue;
        CSCDetId DetId = (CSCDetId)(*it).cscDetId();
	const CSCChamber* cscchamber = cscGeom->chamber( DetId );
	GlobalPoint gp = cscchamber->toGlobal((*it).localPosition() );
        if (( fabs(gp.theta()- trkTheta) > 0.5  ) || ( fabs(gp.phi()- trkPhi) > 0.5)  ) continue;
	GlobalVector gv = cscchamber->toGlobal((*it).localDirection() );


        ave_phi += gp.phi();
        ave_eta += gp.eta();
        dx_error.push_back( (*it).localDirectionError().xx() );
        dy_error.push_back( (*it).localDirectionError().yy() );
        x_error.push_back( (*it).localPositionError().xx() );
        y_error.push_back( (*it).localPositionError().yy() );
        n++;
        if (debug) {
           cout <<"~~~~~~~~~~~~~~~~  reco segs  ~~~~~~~~~~~~~~~~  " <<endl;
	   cout <<"DId: "<<DetId<<endl;
	   cout <<"  h0= "<<gp.eta()<<"  f0= "<<gp.phi()<<"   gp= "<< gp <<endl;
	   cout <<"  h1= "<<gv.eta()<<"  f1= "<<gv.phi()<<"   gv= "<< gv <<endl;
        }
     }
     for(DTRecSegment4DCollection::const_iterator it = dtSeg->begin(); it != dtSeg->end(); it++)
     {
        if ( !(*it).hasPhi() || !(*it).hasZed()  ) continue;
        DTChamberId DetId = (*it).chamberId();
        const DTChamber* dtchamber = dtGeom->chamber( DetId );
        GlobalPoint  gp = dtchamber->toGlobal( (*it).localPosition() );
        if (( fabs(gp.eta()- trkTheta) > 0.5  ) || ( fabs(gp.phi()- trkPhi) > 0.5 ) ) continue;
	GlobalVector gv = dtchamber->toGlobal((*it).localDirection() );

        ave_phi += gp.phi();
        ave_eta += gp.eta();
        m++;
        if (debug) {
           cout <<"~~~~~~~~~~~~~~~~  reco segs  ~~~~~~~~~~~~~~~~  " <<endl;
	   cout <<"DId: "<<DetId<<endl;
	   cout <<"  h0= "<<gp.eta()<<"  f0= "<<gp.phi()<<"   gp= "<< gp <<endl;
	   cout <<"  h1= "<<gv.eta()<<"  f1= "<<gv.phi()<<"   gv= "<< gv <<endl;
        }
     }
     ave_phi = ave_phi / (n+m) ;
     ave_eta = ave_eta / (n+m) ;

     // Calculate the residual of phi and eta
     for(CSCSegmentCollection::const_iterator it = cscSeg->begin(); it != cscSeg->end(); it++)
     {
        if ( (*it).nRecHits() < 4) continue;
        CSCDetId DetId = (CSCDetId)(*it).cscDetId();
        const CSCChamber* cscchamber = cscGeom->chamber( DetId );
        GlobalPoint gp = cscchamber->toGlobal((*it).localPosition() );
        if ( (fabs(gp.theta()- trkTheta) > 0.5  ) || ( fabs(gp.phi()- trkPhi) > 0.5 ) ) continue;
        phi_resid.push_back( gp.phi()- ave_phi );
        eta_resid.push_back( gp.eta()- ave_eta );
     }
     for(DTRecSegment4DCollection::const_iterator it = dtSeg->begin(); it != dtSeg->end(); it++)
     {
        if ( !(*it).hasPhi() || !(*it).hasZed()  ) continue;
        DTChamberId DetId = (*it).chamberId();
        const DTChamber* dtchamber = dtGeom->chamber( DetId );
        GlobalPoint  gp = dtchamber->toGlobal( (*it).localPosition() );
        if ( (fabs(gp.eta()- trkTheta) > 0.5  ) || ( fabs(gp.phi()- trkPhi) > 0.5 ) ) continue;
        phi_resid.push_back( gp.phi()- ave_phi );
        eta_resid.push_back( gp.eta()- ave_eta );
     }

     if (n!=0 && m== 0) {
        return 1; // csc 
     } else if (n==0 && m!=0 ) {
        return 3; // dt
     } else if (n!=0 && m!=0 ) {
        return 2; // overlap
     } else {
        return 0; // fail
     }

}

double MuonSeedValidator::getEta(double vx, double vy, double vz ) {

      double va = sqrt( vx*vx + vy*vy + vz*vz );

      double theta = acos( vz/va );
      double eta = (-1.0)*log( tan(theta/2.0) )  ;
      return eta;
}
double MuonSeedValidator::getEta(double theta ) {

      double eta = (-1.0)*log( tan(theta/2.0) )  ;
      return eta;
}
