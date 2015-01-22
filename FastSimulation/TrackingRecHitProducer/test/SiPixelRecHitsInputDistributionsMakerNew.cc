// SiPixelRecHitsInputDistributionsMaker.cc
// Description: see SiPixelRecHitsInputDistributionsMaker.h
// Author: Mario Galanti
// Created 31/1/2008
//
// Make resolution histograms for the new pixel resolution parameterization
// G. Hu
//
// Based on SiPixelRecHitsValid.cc by Jason Shaev

#include "FastSimulation/TrackingRecHitProducer/test/SiPixelRecHitsInputDistributionsMakerNew.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include <DQMServices/Core/interface/MonitorElement.h>

#include <math.h>

// To plot resolutions, angles, etc. for muons only
#define MUONSONLY

using namespace std;
using namespace edm;

SiPixelRecHitsInputDistributionsMakerNew::SiPixelRecHitsInputDistributionsMakerNew(const ParameterSet& ps): 
  conf_(ps),
  src_( ps.getParameter<edm::InputTag>( "src" ) ) 
{
  cotAlphaLowEdgeBarrel_  = -0.2;
  cotAlphaBinWidthBarrel_ = 0.08;
  cotBetaLowEdgeBarrel_   = -5.5;
  cotBetaBinWidthBarrel_  = 1.0;

  cotAlphaLowEdgeForward_	= 0.1;
  cotAlphaBinWidthForward_	= 0.1;
  cotBetaLowEdgeForward_	= 0.;
  cotBetaBinWidthForward_	= 0.15;
}


void SiPixelRecHitsInputDistributionsMakerNew::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run& run, const edm::EventSetup & ev) {

  

  ibooker.setCurrentFolder("clustBPIX");
  
  Char_t histo[200];
  Char_t title[200];


  
  // RecHit resolutions in barrel according to cotAlpha, cotBeta, qBin
  for ( int ii=0; ii < cotAlphaBinsBarrel_ ; ii++ )
     for( int jj=0; jj < cotBetaBinsBarrel_; jj++ )
        for ( int kk=0; kk < qBins_ ;	kk++ )
	  {
	     double x1_ = cotAlphaLowEdgeBarrel_ + ii*cotAlphaBinWidthBarrel_ ;
	     double x2_ = cotAlphaLowEdgeBarrel_ + (ii+1)*cotAlphaBinWidthBarrel_ ;
	     double y1_ = cotBetaLowEdgeBarrel_ + jj*cotBetaBinWidthBarrel_ ;
	     double y2_ = cotBetaLowEdgeBarrel_ +  (jj+1)*cotBetaBinWidthBarrel_ ;
	     sprintf( histo, "hx0%d%02d%d", ii+1, jj+1, kk+1 );
	     sprintf( title, "cotalpha %.2f-%.2f cotabeta %.1f-%.1f qbin %d Edge X", x1_, x2_, y1_, y2_, kk+1 );
	     recHitResBarrelEdgeX[ii][jj][kk]  = ibooker.book1D(histo,title, 2000, -0.10, 0.10);
	     sprintf( histo, "hy0%d%02d%d", ii+1, jj+1, kk+1 );
	     sprintf( title, "cotalpha %.2f-%.2f cotabeta %.1f-%.1f qbin %d Edge Y", x1_, x2_, y1_, y2_, kk+1 );
	     recHitResBarrelEdgeY[ii][jj][kk]  = ibooker.book1D(histo,title, 2000, -0.10, 0.10);
	     sprintf( histo, "hx110%d%02d%d", ii+1, jj+1, kk+1 );
	     sprintf( title, "cotalpha %.2f-%.2f cotabeta %.1f-%.1f qbin %d nxpix > 1, BigX", x1_, x2_, y1_, y2_, kk+1 );
	     recHitResBarrelMultiPixBigX[ii][jj][kk]  = ibooker.book1D(histo,title, 2000, -0.10, 0.10);
	     sprintf( histo, "hx111%d%02d%d", ii+1, jj+1, kk+1 );
	     sprintf( title, "cotalpha %.2f-%.2f cotabeta %.1f-%.1f qbin %d nxpix > 1", x1_, x2_, y1_, y2_, kk+1 );
	     recHitResBarrelMultiPixX[ii][jj][kk]  = ibooker.book1D(histo,title, 2000, -0.10, 0.10);
	     sprintf( histo, "hy110%d%02d%d", ii+1, jj+1, kk+1 );
	     sprintf( title, "cotalpha %.2f-%.2f cotabeta %.1f-%.1f qbin %d nypix > 1, BigY", x1_, x2_, y1_, y2_, kk+1 );
	     recHitResBarrelMultiPixBigY[ii][jj][kk]  = ibooker.book1D(histo,title, 2000, -0.10, 0.10);
	     sprintf( histo, "hy111%d%02d%d", ii+1, jj+1, kk+1 );
	     sprintf( title, "cotalpha %.2f-%.2f cotabeta %.1f-%.1f qbin %d nypix > 1", x1_, x2_, y1_, y2_, kk+1 );
	     recHitResBarrelMultiPixY[ii][jj][kk]  = ibooker.book1D(histo,title, 2000, -0.10, 0.10);
	  }

  for ( int ii=0; ii < cotAlphaBinsBarrel_ ; ii++ )
     for( int jj=0; jj < cotBetaBinsBarrel_; jj++ )
     {
        double x1_ = cotAlphaLowEdgeBarrel_ + ii*cotAlphaBinWidthBarrel_ ;
	double x2_ = cotAlphaLowEdgeBarrel_ + (ii+1)*cotAlphaBinWidthBarrel_ ;
	double y1_ = cotBetaLowEdgeBarrel_ + jj*cotBetaBinWidthBarrel_ ;
	double y2_ = cotBetaLowEdgeBarrel_ +  (jj+1)*cotBetaBinWidthBarrel_ ;
	sprintf( histo, "hx100%d%02d", ii+1, jj+1 );
	sprintf( title, "cotalpha %.2f-%.2f cotabeta %.1f-%.1f nxpix = 1, BigX", x1_, x2_, y1_, y2_ );
	recHitResBarrelSinglePixBigX[ii][jj]  = ibooker.book1D(histo,title, 2000, -0.10, 0.10);
	sprintf( histo, "hx101%d%02d", ii+1, jj+1 );
	sprintf( title, "cotalpha %.2f-%.2f cotabeta %.1f-%.1f nxpix = 1", x1_, x2_, y1_, y2_ );
	recHitResBarrelSinglePixX[ii][jj]  = ibooker.book1D(histo,title, 2000, -0.10, 0.10);
	sprintf( histo, "hy100%d%02d", ii+1, jj+1 );
	sprintf( title, "cotalpha %.2f-%.2f cotabeta %.1f-%.1f nypix = 1, BigY", x1_, x2_, y1_, y2_ );
	recHitResBarrelSinglePixBigY[ii][jj]  = ibooker.book1D(histo,title, 2000, -0.10, 0.10);
	sprintf( histo, "hy101%d%02d", ii+1, jj+1 );
	sprintf( title, "cotalpha %.2f-%.2f cotabeta %.1f-%.1f nypix = 1", x1_, x2_, y1_, y2_ );
	recHitResBarrelSinglePixY[ii][jj]  = ibooker.book1D(histo,title, 2000, -0.10, 0.10);
     }

  ibooker.setCurrentFolder("clustBPIXInfo");
  recHitClusterInfo = ibooker.book1D("Cluster Info", "Cluster Info", 10, -0.5, 9.5 );
  recHitcotAlpha[0] = ibooker.book1D("cotAlpha all", "cotAlpha all", 5, -0.2, 0.2 );
  recHitcotAlpha[1] = ibooker.book1D("cotAlpha edge", "cotAlpha edge", 5, -0.2, 0.2 );
  recHitcotAlpha[2] = ibooker.book1D("cotAlpha MultiPixBigX", "cotAlpha MultiPixBigX", 5, -0.2, 0.2 );
  recHitcotAlpha[3] = ibooker.book1D("cotAlpha SinglePixBigX", "cotAlpha SinglePixBigX", 5, -0.2, 0.2 );
  recHitcotAlpha[4] = ibooker.book1D("cotAlpha MultiPixX", "cotAlpha MultiPixX", 5, -0.2, 0.2 );
  recHitcotAlpha[5] = ibooker.book1D("cotAlpha SinglePixX", "cotAlpha SinglePixX", 5, -0.2, 0.2 );
  recHitcotAlpha[6] = ibooker.book1D("cotAlpha MultiPixBigY", "cotAlpha MultiPixBigY", 5, -0.2, 0.2 );
  recHitcotAlpha[7] = ibooker.book1D("cotAlpha SinglePixBigY", "cotAlpha SinglePixBigY", 5, -0.2, 0.2 );
  recHitcotAlpha[8] = ibooker.book1D("cotAlpha MultiPixY", "cotAlpha MultiPixY", 5, -0.2, 0.2 );
  recHitcotAlpha[9] = ibooker.book1D("cotAlpha SinglePixY", "cotAlpha SinglePixY",  5, -0.2, 0.2 );

  recHitcotBeta[0] = ibooker.book1D("cotBeta all", "cotBeta all", 11, -5.5, 5.5 );
  recHitcotBeta[1] = ibooker.book1D("cotBeta edge", "cotBeta edge", 11, -5.5, 5.5 );
  recHitcotBeta[2] = ibooker.book1D("cotBeta MultiPixBigX", "cotBeta MultiPixBigX",11, -5.5, 5.5 );
  recHitcotBeta[3] = ibooker.book1D("cotBeta SinglePixBigX", "cotBeta SinglePixBigX",11, -5.5, 5.5 );
  recHitcotBeta[4] = ibooker.book1D("cotBeta MultiPixX", "cotBeta MultiPixX",11, -5.5, 5.5 );
  recHitcotBeta[5] = ibooker.book1D("cotBeta SinglePixX", "cotBeta SinglePixX",11, -5.5, 5.5 );
  recHitcotBeta[6] = ibooker.book1D("cotBeta MultiPixBigY", "cotBeta MultiPixBigY",11, -5.5, 5.5 );
  recHitcotBeta[7] = ibooker.book1D("cotBeta SinglePixBigY", "cotBeta SinglePixBigY",11, -5.5, 5.5 );
  recHitcotBeta[8] = ibooker.book1D("cotBeta MultiPixY", "cotBeta MultiPixY",11, -5.5, 5.5 );
  recHitcotBeta[9] = ibooker.book1D("cotBeta SinglePixY", "cotBeta SinglePixY", 11, -5.5, 5.5 );

  recHitqBin[0] = ibooker.book1D("qBin all", "qBin all", 4, -0.5, 3.5 );
  recHitqBin[1] = ibooker.book1D("qBin edge", "qBin edge",4, -0.5, 3.5 );
  recHitqBin[2] = ibooker.book1D("qBin MultiPixBigX", "qBin MultiPixBigX",4, -0.5, 3.5 );
  recHitqBin[3] = ibooker.book1D("qBin MultiPixX", "qBin MultiPixX",4, -0.5, 3.5 );
  recHitqBin[4] = ibooker.book1D("qBin MultiPixBigY","qBin MultiPixBigY",4, -0.5, 3.5 );
  recHitqBin[5] = ibooker.book1D("qBin MultiPixY", "qBin MultiPixY",4, -0.5, 3.5 );
  recHitqBin[6] = ibooker.book1D("qBin", "qBin all", 10, -0.5, 9.5 );
  recHitqBin[7] = ibooker.book1D("charge", "charge distribution", 100, 0, 700000 );


  ibooker.setCurrentFolder("clustFPIX");
  // RecHit resolutions in forward according to cotAlpha, cotBeta, qBin
  for ( int ii=0; ii < cotAlphaBinsForward_ ; ii++ )
    for( int jj=0; jj < cotBetaBinsForward_ ; jj++ )
      for( int kk=0; kk < qBins_ ; kk++ )
      {
	  double x1_ = cotAlphaLowEdgeForward_ + ii*cotAlphaBinWidthForward_ ;
	  double x2_ = cotAlphaLowEdgeForward_ + (ii+1)*cotAlphaBinWidthForward_ ;
	  double y1_ = cotBetaLowEdgeForward_ + jj*cotBetaBinWidthForward_ ;
	  double y2_ = cotBetaLowEdgeForward_ +  (jj+1)*cotBetaBinWidthForward_ ;
	  sprintf( histo, "fhx0%d%02d%d", ii+1, jj+1, kk+1 );
	  sprintf( title, "cotalpha %.2f-%.2f cotbeta %.2f-%.2f qbin %d Edge X", x1_, x2_, y1_, y2_, kk+1 );
	  recHitResForwardEdgeX[ii][jj][kk]  = ibooker.book1D(histo,title, 2000, -0.10, 0.10);
	  sprintf( histo, "fhy0%d%02d%d", ii+1, jj+1, kk+1 );
	  sprintf( title, "cotalpha %.2f-%.2f cotbeta %.2f-%.2f qbin %d Edge Y", x1_, x2_, y1_, y2_, kk+1 );
	  recHitResForwardEdgeY[ii][jj][kk]  = ibooker.book1D(histo,title, 2000, -0.10, 0.10);

          sprintf( histo, "fhx11%d%02d%d", ii+1, jj+1, kk+1 );
          sprintf( title, "cotalpha %.2f-%.2f cotbeta %.2f-%.2f qbin %d X", x1_, x2_, y1_, y2_, kk+1 );
          recHitResForwardX[ii][jj][kk]  = ibooker.book1D(histo,title, 2000, -0.10, 0.10);
          sprintf( histo, "fhy11%d%02d%d", ii+1, jj+1, kk+1 );
          sprintf( title, "cotalpha %.2f-%.2f cotbeta %.2f-%.2f qbin %d Y", x1_, x2_, y1_, y2_, kk+1 );
          recHitResForwardY[ii][jj][kk]  = ibooker.book1D(histo,title, 2000, -0.10, 0.10);
      }

  for ( int ii=0; ii < cotAlphaBinsForward_ ; ii++ )
    for( int jj=0; jj < cotBetaBinsForward_ ; jj++ )
      {
        double x1_ = cotAlphaLowEdgeForward_ + ii*cotAlphaBinWidthForward_ ;
	double x2_ = cotAlphaLowEdgeForward_ + (ii+1)*cotAlphaBinWidthForward_ ;
	double y1_ = cotBetaLowEdgeForward_ + jj*cotBetaBinWidthForward_ ;
	double y2_ = cotBetaLowEdgeForward_ +  (jj+1)*cotBetaBinWidthForward_ ;
	sprintf( histo, "fhx100%d%02d", ii+1, jj+1 );
	sprintf( title, "cotalpha %.2f-%.2f cotbeta %.2f-%.2f sizex=1 BigX",  x1_, x2_, y1_, y2_);
	recHitResForwardSingleBX[ii][jj] = ibooker.book1D(histo,title, 2000, -0.10, 0.10);
	sprintf( histo, "fhy100%d%02d", ii+1, jj+1 );
	sprintf( title, "cotalpha %.2f-%.2f cotbeta %.2f-%.2f sizey=1 BigY", x1_, x2_, y1_, y2_);
	recHitResForwardSingleBY[ii][jj] = ibooker.book1D(histo,title, 2000, -0.10, 0.10);
	sprintf( histo, "fhx101%d%02d", ii+1, jj+1 );
	sprintf( title, "cotalpha %.2f-%.2f cotbeta %.2f-%.2f sizex=1",  x1_, x2_, y1_, y2_);
	recHitResForwardSingleX[ii][jj] = ibooker.book1D(histo,title, 2000, -0.10, 0.10);
	sprintf( histo, "fhy101%d%02d", ii+1, jj+1 );
	sprintf( title, "cotalpha %.2f-%.2f cotbeta %.2f-%.2f sizey=1",  x1_, x2_, y1_, y2_);
	recHitResForwardSingleY[ii][jj] = ibooker.book1D(histo,title, 2000, -0.10, 0.10);
      }
  


  ibooker.setCurrentFolder("simHitBPIX");
  //SimHit alpha in barrel
  simHitAlphaBarrel = ibooker.book1D("simHit_alpha_AllModules_Barrel", "SimHit Alpha distribution for all modules in barrel", 14, -0.28, 0.28);
  simHitAlphaBarrelBigPixel = ibooker.book1D("simHit_alpha_AllModules_Barrel_BigPixel", "SimHit Alpha distribution for all modules in barrel for bigpixels", 14, -0.28, 0.28);
  //SimHit beta in barrel
  simHitBetaBarrel = ibooker.book1D("simHit_beta_AllModules_Barrel", "SimHit beta distribution for all modules in barrel<", 55, 0., 1.595);
  simHitBetaBarrelBigPixel = ibooker.book1D("simHit_beta_AllModules_Barrel_BigPixel", "SimHit beta distribution for all modules in barrel for bigpixels", 55, 0., 1.595);

  // Alpha and beta probabilities according as multiplicities
  simHitAlphaMultiBarrel[0] = ibooker.book1D("hist_alpha_barrel_0", "#alpha probability (barrel)", 14, -0.28, 0.28);
  simHitAlphaMultiBarrelBigPixel[0] = ibooker.book1D("hist_alpha_barrel_big_0", "#alpha probability bigpix (barrel)", 14, -0.28, 0.28);
  simHitBetaMultiBarrel[0] = ibooker.book1D("hist_beta_barrel_0", "#beta probability (barrel)", 55, 0., 1.595);
  simHitBetaMultiBarrelBigPixel[0] = ibooker.book1D("hist_beta_barrel_big_0", "#beta probability bigpix (barrel)", 55, 0., 1.595);
  for(int i=1; i<4; i++) {
    sprintf(histo, "hist_alpha_barrel_%d", i);
    sprintf(title, "#alpha probability multiplicity=%d (barrel)",i);
    simHitAlphaMultiBarrel[i] = ibooker.book1D(histo, title, 14, -0.28, 0.28);
    sprintf(histo, "hist_alpha_barrel_big_%d", i);
    sprintf(title, "#alpha probability multiplicity=%d bigpixel (barrel)",i);
    simHitAlphaMultiBarrelBigPixel[i] = ibooker.book1D(histo, title, 14, -0.28, 0.28);
  }
  for(int i=1; i<7; i++) {
    sprintf(histo, "hist_beta_barrel_%d", i);
    sprintf(title, "#beta probability multiplicity=%d (barrel)",i);
    simHitBetaMultiBarrel[i] = ibooker.book1D(histo, title, 55, 0., 1.595);
    sprintf(histo, "hist_beta_barrel_big_%d", i);
    sprintf(title, "#beta probability multiplicity=%d bigpixel (barrel)",i);
    simHitBetaMultiBarrelBigPixel[i] = ibooker.book1D(histo, title, 55, 0., 1.595);
  }
  simHitAlphaMultiBarrel[4] = ibooker.book1D("hist_alpha_barrel_4", "#alpha probability multiplicity>3 (barrel)", 14, -0.28, 0.28);
  simHitAlphaMultiBarrelBigPixel[4] = ibooker.book1D("hist_alpha_barrel_big_4", "#alpha probability multiplicity>3 bigpixel (barrel)", 14, -0.28, 0.28);
  simHitBetaMultiBarrel[7] = ibooker.book1D("hist_beta_barrel_7", "#beta probability multiplicity>6 (barrel)", 55, 0., 1.595);
  simHitBetaMultiBarrelBigPixel[7] = ibooker.book1D("hist_beta_barrel_big_7", "#beta probability multiplicity>6 bigpixel (barrel)", 55, 0., 1.595);

  ibooker.setCurrentFolder("simHitFPIX");
  //SimHit alpha in forward
  simHitAlphaForward = ibooker.book1D("simHit_alpha_AllModules_Forward", "SimHit Alpha distribution for all modules in forward", 14, 0.104, 0.566);
  simHitAlphaForwardBigPixel = ibooker.book1D("simHit_alpha_AllModules_Forward_BigPixel", "SimHit Alpha distribution for all modules in forward for bigpixels", 14, 0.104, 0.566);
  //SimHit beta in forward
  simHitBetaForward = ibooker.book1D("simHit_beta_AllModules_Forward", "SimHit beta distribution for all modules in forward", 10, 0.203, 0.493);
  simHitBetaForwardBigPixel = ibooker.book1D("simHit_beta_AllModules_Forward_BigPixel", "SimHit beta distribution for all modules in forward for bigpixels", 10, 0.203, 0.493);

  // Alpha and beta probabilities according as multiplicities
  simHitAlphaMultiForward[0] = ibooker.book1D("hist_alpha_forward_0", "#alpha probability (forward)", 14, 0.104, 0.566);
  simHitAlphaMultiForwardBigPixel[0] = ibooker.book1D("hist_alpha_forward_big_0", "#alpha probability bigpixel (forward)", 14, 0.104, 0.566);
  simHitBetaMultiForward[0] = ibooker.book1D("hist_beta_forward_0", "#beta probability (forward)", 10, 0.203, 0.493);
  simHitBetaMultiForwardBigPixel[0] = ibooker.book1D("hist_beta_forward_big_0", "#beta probability bigpixel (forward)", 10, 0.203, 0.493);
  for (int i=1; i<3; i++) {
    sprintf(histo, "hist_alpha_forward_%d", i);
    sprintf(title, "#alpha probability multiplicity=%d (forward)", i);
    simHitAlphaMultiForward[i] = ibooker.book1D(histo, title, 14, 0.104, 0.566);
    sprintf(histo, "hist_alpha_forward_big_%d", i);
    sprintf(title, "#alpha probability multiplicity=%d bigpixel (forward)", i);
    simHitAlphaMultiForwardBigPixel[i] = ibooker.book1D(histo, title, 14, 0.104, 0.566);
    sprintf(histo, "hist_beta_forward_%d", i);
    sprintf(title, "#beta probability multiplicity=%d (forward)", i);
    simHitBetaMultiForward[i] = ibooker.book1D(histo, title, 10, 0.203, 0.493);
    sprintf(histo, "hist_beta_forward_big_%d", i);
    sprintf(title, "#beta probability multiplicity=%d bigpixel (forward)", i);
    simHitBetaMultiForwardBigPixel[i] = ibooker.book1D(histo, title, 10, 0.203, 0.493);
  }
  simHitAlphaMultiForward[3] = ibooker.book1D("hist_alpha_forward_3", "#alpha probability multiplicity>2 (forward)", 14, 0.104, 0.566);
  simHitAlphaMultiForwardBigPixel[3] = ibooker.book1D("hist_alpha_forward_big_3", "#alpha probability multiplicity>2 bigpixel (forward)", 14, 0.104, 0.566);
  simHitBetaMultiForward[3] = ibooker.book1D("hist_beta_forward_3", "#beta probability multiplicity>2 (forward)", 10, 0.203, 0.493);
  simHitBetaMultiForwardBigPixel[3] = ibooker.book1D("hist_beta_forward_big_3", "#beta probability multiplicity>2 bigpixel (forward)", 10, 0.203, 0.493);
}

SiPixelRecHitsInputDistributionsMakerNew::~SiPixelRecHitsInputDistributionsMakerNew() {
}

void SiPixelRecHitsInputDistributionsMakerNew::analyze(const edm::Event& e, const edm::EventSetup& es) 
{
  
  LogInfo("EventInfo") << " Run = " << e.id().run() << " Event = " << e.id().event();
  if ( (int) e.id().event() % 1000 == 0 )
    cout << " Run = " << e.id().run() << " Event = " << e.id().event() << endl;
  
  //Get event setup
  edm::ESHandle<TrackerGeometry> geom;
  es.get<TrackerDigiGeometryRecord>().get(geom);
  const TrackerGeometry& theTracker(*geom);

  TrackerHitAssociator associate( e, conf_ );


  edm::Handle<reco::TrackCollection> trackCollection;
  e.getByLabel( "generalTracks" , trackCollection);
  const reco::TrackCollection *tracks = trackCollection.product();
  reco::TrackCollection::const_iterator tciter;
  std::vector<PSimHit> matched;

  if ( tracks->size() > 0 )
  {
      for ( tciter=tracks->begin(); tciter!=tracks->end(); ++tciter)
      {
        // First loop on hits: find matched hits
        for ( trackingRecHit_iterator it = tciter->recHitsBegin(); it != tciter->recHitsEnd(); ++it)
	{
	  const TrackingRecHit &thit = **it;
	  bool  found_match = false;
	  // Is it a matched hit?
	  const SiPixelRecHit* pixelhit = dynamic_cast<const SiPixelRecHit*>(&thit);
	  DetId detId = (*it)->geographicalId();
	  const PixelGeomDetUnit * theGeomDet = dynamic_cast<const PixelGeomDetUnit*>(theTracker.idToDet(detId) );
	  int subdetId = (int)detId.subdetId();
	  
	  if ( pixelhit )
	  {
	    matched.clear();
	    matched = associate.associateHit(*pixelhit);
	    float closest = 9999.9;
	    vector<PSimHit>::const_iterator closestit;
	    float rechit_x = (pixelhit->localPosition()).x();
	    float rechit_y = (pixelhit->localPosition()).y();
	    if ( !matched.empty() )
	    {
	      for (vector<PSimHit>::const_iterator m=matched.begin(); m!=matched.end(); ++m) 
	      { 
#ifdef	MUONSONLY
		int pid = (*m).particleType();
		if( abs(pid) != 13 )  continue;
#endif
	        float sim_x1 = (*m).entryPoint().x();
		float sim_x2 = (*m).exitPoint().x();
                float sim_xpos = 0.5*(sim_x1+sim_x2);
                float sim_y1 = (*m).entryPoint().y();
                float sim_y2 = (*m).exitPoint().y();
                float sim_ypos = 0.5*(sim_y1+sim_y2);
		float x_res = fabs( sim_xpos - rechit_x );
		float y_res = fabs( sim_ypos - rechit_y );
		
		float dist = sqrt( x_res*x_res + y_res*y_res );
                if ( dist < closest ) 
                   {
                      closest = dist;
                      closestit = m;
		      found_match = true;
                   }
	      } // end sim hits loop

	      if( found_match )
	      {
	        if( subdetId == (int)PixelSubdetector::PixelBarrel )
		  fillBarrel(*pixelhit, *closestit, detId, theGeomDet);
		if( subdetId == (int)PixelSubdetector::PixelEndcap )
		  fillForward(*pixelhit, *closestit, detId, theGeomDet);
		  
	      } // if found_match
	    }//find some match 
	  }// find a pixel hit
	}// end loop on hits
      } // end loop on tracks
  }  // tracks_size > 0
}


void SiPixelRecHitsInputDistributionsMakerNew::fillBarrel(const SiPixelRecHit& recHit, const PSimHit& simHit, 
                                                       DetId detId, const PixelGeomDetUnit* theGeomDet) 
{
  
  LocalPoint lp = recHit.localPosition();

  //  LocalError lerr = recHit.localPositionError();

  recHitClusterInfo -> Fill(0);
  
  float sim_x1 = simHit.entryPoint().x();
  float sim_x2 = simHit.exitPoint().x();
  float sim_xpos = 0.5*(sim_x1 + sim_x2);
  float res_x = lp.x() - sim_xpos;
  
  float sim_xdir = simHit.localDirection().x();
  float sim_ydir = simHit.localDirection().y();
  float sim_zdir = simHit.localDirection().z();

  // alpha: angle with respect to local x axis in local (x,z) plane
  float cotalpha = sim_xdir/sim_zdir;
  // beta: angle with respect to local y axis in local (y,z) plane
  float cotbeta = sim_ydir/sim_zdir;
  
  float sim_y1 = simHit.entryPoint().y();
  float sim_y2 = simHit.exitPoint().y();
  float sim_ypos = 0.5*(sim_y1 + sim_y2);
  float res_y = lp.y() - sim_ypos;
  
  //get cluster
  SiPixelRecHit::ClusterRef const& clust = recHit.cluster();

  int sizeX = (*clust).sizeX();
  int sizeY = (*clust).sizeY();
  int firstPixelInX = (*clust).minPixelRow();
  int lastPixelInX = (*clust).maxPixelRow();
  int firstPixelInY = (*clust).minPixelCol();
  int lastPixelInY = (*clust).maxPixelCol();
  int iqbin = recHit.qBin();
  recHitqBin[6]-> Fill( iqbin );
  recHitqBin[7]-> Fill( (*clust).charge() );
  iqbin = iqbin > 2 ? 3 : iqbin ;
  int icotalpha = int ( ( cotalpha - cotAlphaLowEdgeBarrel_ ) / cotAlphaBinWidthBarrel_ );
  int icotbeta = int ( ( cotbeta - cotBetaLowEdgeBarrel_ ) / cotBetaBinWidthBarrel_ );
  icotalpha = icotalpha < 0 ? 0 : icotalpha;
  icotalpha = icotalpha > (cotAlphaBinsBarrel_-1 ) ? cotAlphaBinsBarrel_-1 : icotalpha;
  icotbeta  = icotbeta < 0 ? 0 : icotbeta;
  icotbeta  = icotbeta > (cotBetaBinsBarrel_-1 ) ? cotBetaBinsBarrel_-1 : icotbeta;

// Use this definition if you want to consider an ideal topology 
  const RectangularPixelTopology *rectPixelTopology = static_cast<const RectangularPixelTopology*>(&(theGeomDet->specificType().specificTopology()));
// Use the following instead of the previous line if you want the real topology that will take care of the 
// module deformations when mapping from ideal to real coordinates:
//  PixelTopology const & rectPixelTopology = theGeomDet->specificTopology();

  bool edgex = 
       ( rectPixelTopology->isItEdgePixelInX( firstPixelInX ) || rectPixelTopology->isItEdgePixelInX( lastPixelInX ) );
  bool edgey = 
       ( rectPixelTopology->isItEdgePixelInY( firstPixelInY ) || rectPixelTopology->isItEdgePixelInY( lastPixelInY ) );
  bool bigx =  ( rectPixelTopology->isItBigPixelInX( firstPixelInX ) || 
		 rectPixelTopology->isItBigPixelInX( lastPixelInX ) );
  bool bigy =  ( rectPixelTopology->isItBigPixelInY( firstPixelInY ) ||
                 rectPixelTopology->isItBigPixelInY( lastPixelInY ) );
//  bool bigx =  ( RectangularPixelTopology::isItBigPixelInX( firstPixelInX ) || 
//	 	 RectangularPixelTopology::isItBigPixelInX( lastPixelInX ) );
//  bool bigy =  ( RectangularPixelTopology::isItBigPixelInY( firstPixelInY ) ||
//                 RectangularPixelTopology::isItBigPixelInY( lastPixelInY ) );
  bool edge = ( edgex || edgey );

  recHitcotAlpha[0] -> Fill( cotalpha );
  recHitcotBeta[0]  -> Fill( cotbeta  );
  recHitqBin[0]     -> Fill( iqbin );
 
  if ( fabs(res_x) > 0.10  || fabs(res_y) > 0.10 )  return;
#ifdef MUONSONLY 
  if(abs(simHit.particleType()) == 13)
  {
#endif
    if( edge )
    {   
           recHitResBarrelEdgeX[icotalpha][icotbeta][iqbin] -> Fill( res_x );
	   recHitResBarrelEdgeY[icotalpha][icotbeta][iqbin] -> Fill( res_y );
	   recHitClusterInfo -> Fill(1) ;
	   recHitcotAlpha[1] -> Fill( cotalpha );
	   recHitcotBeta[1]  -> Fill( cotbeta );
	   recHitqBin[1]     -> Fill( iqbin  );
    }
    else
    {
       if ( sizeX == 1 )
       {
          if( bigx ) {
	     recHitResBarrelSinglePixBigX[icotalpha][icotbeta] -> Fill( res_x );
	     recHitClusterInfo -> Fill(3);
	     recHitcotAlpha[3] -> Fill( cotalpha );
	     recHitcotBeta[3] -> Fill( cotbeta );
	  }
	  else {
	     recHitResBarrelSinglePixX[icotalpha][icotbeta] -> Fill( res_x );
	     recHitClusterInfo -> Fill(5);
	     recHitcotAlpha[5] -> Fill( cotalpha );
	     recHitcotBeta[5] -> Fill( cotbeta );
	  }
       }
       else
       {
           if( bigx ) {
	      recHitResBarrelMultiPixBigX[icotalpha][icotbeta][iqbin] -> Fill( res_x );
	      recHitClusterInfo -> Fill(2);
	      recHitcotAlpha[2] -> Fill( cotalpha );
	      recHitcotBeta[2] -> Fill( cotbeta );
	      recHitqBin[2]    -> Fill( iqbin );
	   }
	   else {
	      recHitResBarrelMultiPixX[icotalpha][icotbeta][iqbin] -> Fill( res_x );
	      recHitClusterInfo -> Fill(4);
	      recHitcotAlpha[4] -> Fill( cotalpha );
	      recHitcotBeta[4] -> Fill( cotbeta );
	      recHitqBin[3]    -> Fill( iqbin );
	   }
       }
       if ( sizeY == 1 )
       {
          if( bigy ) {
	     recHitResBarrelSinglePixBigY[icotalpha][icotbeta] -> Fill( res_y );
	     recHitClusterInfo -> Fill(7);
             recHitcotAlpha[7]  -> Fill ( cotalpha );
             recHitcotBeta[7]  -> Fill ( cotbeta );
	  }
	  else  {
	     recHitResBarrelSinglePixY[icotalpha][icotbeta] -> Fill( res_y );
	     recHitClusterInfo -> Fill(9);
             recHitcotAlpha[9]  -> Fill ( cotalpha );
             recHitcotBeta[9]  -> Fill ( cotbeta );
	  }
       }
       else
       {
          if( bigy )  {
	     recHitResBarrelMultiPixBigY[icotalpha][icotbeta][iqbin] -> Fill( res_y );
	     recHitClusterInfo -> Fill(6);
	     recHitcotAlpha[6]  -> Fill ( cotalpha );
	     recHitcotBeta[6]  -> Fill ( cotbeta );
	     recHitqBin[4]  -> Fill ( iqbin );
	  }
	  else  {
	     recHitResBarrelMultiPixY[icotalpha][icotbeta][iqbin] -> Fill( res_y );
	     recHitClusterInfo-> Fill(8);
             recHitcotAlpha[8]  -> Fill ( cotalpha );
             recHitcotBeta[8]  -> Fill ( cotbeta );
             recHitqBin[5]  -> Fill ( iqbin );
	  }
       }
    }
#ifdef MUONSONLY
  }
#endif
}

void SiPixelRecHitsInputDistributionsMakerNew::fillForward(const SiPixelRecHit & recHit, const PSimHit & simHit, 
                                                        DetId detId,const PixelGeomDetUnit * theGeomDet ) 
{
  LocalPoint lp = recHit.localPosition();
  
  //  LocalError lerr = recHit.localPositionError();

  float sim_x1 = simHit.entryPoint().x();
  float sim_x2 = simHit.exitPoint().x();
  float sim_xpos = 0.5*(sim_x1 + sim_x2);
  
  float sim_y1 = simHit.entryPoint().y();
  float sim_y2 = simHit.exitPoint().y();
  float sim_ypos = 0.5*(sim_y1 + sim_y2);
  
  float sim_xdir = simHit.localDirection().x();
  float sim_ydir = simHit.localDirection().y();
  float sim_zdir = simHit.localDirection().z();

  // alpha: angle with respect to local x axis in local (x,z) plane
  float cotalpha = sim_xdir/sim_zdir;
  // beta: angle with respect to local y axis in local (y,z) plane
  float cotbeta = sim_ydir/sim_zdir;

  float res_x = (lp.x() - sim_xpos);
  
  float res_y = (lp.y() - sim_ypos);
  if( cotbeta < 0. )
  {
    cotbeta = fabs(cotbeta);
    res_y = -1.*res_y;
  }
  
  // get cluster
  SiPixelRecHit::ClusterRef const& clust = recHit.cluster();

  int sizeX = (*clust).sizeX();
  int sizeY = (*clust).sizeY();
  int firstPixelInX = (*clust).minPixelRow();
  int lastPixelInX = (*clust).maxPixelRow();
  int firstPixelInY = (*clust).minPixelCol();
  int lastPixelInY = (*clust).maxPixelCol();
  const RectangularPixelTopology *rectPixelTopology = static_cast<const RectangularPixelTopology*>(&(theGeomDet->specificType().specificTopology()));
  bool hasBigPixelInX = rectPixelTopology->containsBigPixelInX(firstPixelInX,lastPixelInX);
  bool hasBigPixelInY = rectPixelTopology->containsBigPixelInY(firstPixelInY,lastPixelInY);
  bool edgex =
         ( rectPixelTopology->isItEdgePixelInX( firstPixelInX ) || rectPixelTopology->isItEdgePixelInX( lastPixelInX ) );
  bool edgey = 
  	( rectPixelTopology->isItEdgePixelInY( firstPixelInY ) || rectPixelTopology->isItEdgePixelInY( lastPixelInY  ) );

  bool edge = edgex || edgey;
  int icotalpha = int ( ( cotalpha - cotAlphaLowEdgeForward_ ) / cotAlphaBinWidthForward_ );
  int icotbeta = int ( ( cotbeta - cotBetaLowEdgeForward_ ) / cotBetaBinWidthForward_ );
  icotalpha = icotalpha < 0 ? 0 : icotalpha;
  icotalpha = icotalpha > (cotAlphaBinsForward_-1 ) ? cotAlphaBinsForward_-1 : icotalpha;
  icotbeta  = icotbeta < 0 ? 0 : icotbeta;
  icotbeta  = icotbeta > (cotBetaBinsForward_-1 ) ? cotBetaBinsForward_-1 : icotbeta;

  int iqbin = recHit.qBin();
  iqbin = iqbin > 2 ? 3 : iqbin ;

  if ( fabs(res_x) > 0.10  || fabs(res_y) > 0.10 )  return;
  if ( edge )
  {
    recHitResForwardEdgeX[icotalpha][icotbeta][iqbin] -> Fill( res_x );
    recHitResForwardEdgeY[icotalpha][icotbeta][iqbin] -> Fill( res_y );
  }
  else
  {
    if( sizeX == 1 )
      if( hasBigPixelInX )
        recHitResForwardSingleBX[icotalpha][icotbeta] -> Fill( res_x );
      else
        recHitResForwardSingleX[icotalpha][icotbeta] -> Fill( res_x );
    else
      recHitResForwardX[icotalpha][icotbeta][iqbin] -> Fill( res_x );

    if( sizeY == 1 )
      if( hasBigPixelInY )
        recHitResForwardSingleBY[icotalpha][icotbeta] -> Fill( res_y );
      else
        recHitResForwardSingleY[icotalpha][icotbeta] -> Fill( res_y );
    else
      recHitResForwardY[icotalpha][icotbeta][iqbin] -> Fill( res_y );
  }

}

//
DEFINE_FWK_MODULE(SiPixelRecHitsInputDistributionsMakerNew);
