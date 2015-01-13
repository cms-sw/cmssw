// SiPixelRecHitsInputDistributionsMaker.cc
// Description: see SiPixelRecHitsInputDistributionsMaker.h
// Author: Mario Galanti
// Created 31/1/2008
//
// migrated to MT compatible DQM module by Lukas Vanelderen on 9/1/2015
// only tested compilation
//
// Based on SiPixelRecHitsValid.cc by Jason Shaev

#include "FastSimulation/TrackingRecHitProducer/test/SiPixelRecHitsInputDistributionsMaker.h"

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

#include "DQMServices/Core/interface/MonitorElement.h"

#include <math.h>

// To plot resolutions, angles, etc. for muons only
#define MUONSONLY

using namespace std;
using namespace edm;

const double PI = 3.14159265358979323;

SiPixelRecHitsInputDistributionsMaker::SiPixelRecHitsInputDistributionsMaker(const ParameterSet& ps): 
  conf_(ps),
  src_( ps.getParameter<edm::InputTag>( "src" ) ) 
{
  trackerContainers.clear();
  trackerContainers = ps.getParameter<std::vector<std::string> >("ROUList");
}

void SiPixelRecHitsInputDistributionsMaker::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run& run, const edm::EventSetup & ev) {

  ibooker.setCurrentFolder("clustBPIX");
  
  Char_t histo[200];
  Char_t title[200];
  
  // RecHit resolutions in barrel according to beta multiplicity
  for (int i=0; i<7; i++) {
    // Resolution for 0<beta<0.2
    sprintf(histo, "h10%d", i+1);
    sprintf(title, "beta 0-0.2 pix %d", i+1);
    recHitResBetaBarrel00[i] = ibooker.book1D(histo,title, 1000, -0.05, 0.05);
    
    // Resolution for 0.2<beta<0.4
    sprintf(histo, "h20%d", i+1);
    sprintf(title, "beta 0.2-0.4 pix %d", i+1);
    recHitResBetaBarrel02[i] = ibooker.book1D(histo,title, 1000, -0.05, 0.05);

    // Resolution for 0.4<beta<0.6
    sprintf(histo, "h30%d", i+1);
    sprintf(title, "beta 0.4-0.6 pix %d", i+1);
    recHitResBetaBarrel04[i] = ibooker.book1D(histo,title, 1000, -0.05, 0.05);

    // Resolution for 0.6<beta<0.8
    sprintf(histo, "h40%d", i+1);
    sprintf(title, "beta 0.6-0.8 pix %d", i+1);
    recHitResBetaBarrel06[i] = ibooker.book1D(histo,title, 1000, -0.05, 0.05);

    // Resolution for 0.8<beta<1.0
    sprintf(histo, "h50%d", i+1);
    sprintf(title, "beta 0.8-1.0 pix %d", i+1);
    recHitResBetaBarrel08[i] = ibooker.book1D(histo,title, 1000, -0.05, 0.05);

    // Resolution for 1.0<beta<1.2
    sprintf(histo, "h60%d", i+1);
    sprintf(title, "beta 1.0-1.2 pix %d", i+1);
    recHitResBetaBarrel10[i] = ibooker.book1D(histo,title, 1000, -0.05, 0.05);

    // Resolution for 1.2<beta<1.4
    sprintf(histo, "h70%d", i+1);
    sprintf(title, "beta 1.2-1.4 pix %d", i+1);
    recHitResBetaBarrel12[i] = ibooker.book1D(histo,title, 1000, -0.05, 0.05);
  }
 
  for (int i=0; i<7; i++) {
    // Resolution for 0<beta<0.2
    sprintf(histo, "h10%db", i+1);
    sprintf(title, "beta 0-0.2 big pix %d", i+1);
    recHitResBetaBarrelBigPixel00[i] = ibooker.book1D(histo,title, 1000, -0.05, 0.05);
    
    // Resolution for 0.2<beta<0.4
    sprintf(histo, "h20%db", i+1);
    sprintf(title, "beta 0.2-0.4 big pix %d", i+1);
    recHitResBetaBarrelBigPixel02[i] = ibooker.book1D(histo,title, 1000, -0.05, 0.05);

    // Resolution for 0.4<beta<0.6
    sprintf(histo, "h30%db", i+1);
    sprintf(title, "beta 0.4-0.6 big pix %d", i+1);
    recHitResBetaBarrelBigPixel04[i] = ibooker.book1D(histo,title, 1000, -0.05, 0.05);

    // Resolution for 0.6<beta<0.8
    sprintf(histo, "h40%db", i+1);
    sprintf(title, "beta 0.6-0.8 big pix %d", i+1);
    recHitResBetaBarrelBigPixel06[i] = ibooker.book1D(histo,title, 1000, -0.05, 0.05);

    // Resolution for 0.8<beta<1.0
    sprintf(histo, "h50%db", i+1);
    sprintf(title, "beta 0.8-1.0 big pix %d", i+1);
    recHitResBetaBarrelBigPixel08[i] = ibooker.book1D(histo,title, 1000, -0.05, 0.05);

    // Resolution for 1.0<beta<1.2
    sprintf(histo, "h60%db", i+1);
    sprintf(title, "beta 1.0-1.2 big pix %d", i+1);
    recHitResBetaBarrelBigPixel10[i] = ibooker.book1D(histo,title, 1000, -0.05, 0.05);

    // Resolution for 1.2<beta<1.4
    sprintf(histo, "h70%db", i+1);
    sprintf(title, "beta 1.2-1.4 big pix %d", i+1);
    recHitResBetaBarrelBigPixel12[i] = ibooker.book1D(histo,title, 1000, -0.05, 0.05);
  }
 
  // RecHit resolutions in barrel according to alpha multiplicity
  for (int i=0; i<4; i++) {
    // Resolution for -0.2<alpha<-0.1
    sprintf(histo, "h11%d", i+1);
    sprintf(title, "alpha -0.2 -0.1 pix %d", i+1);
    recHitResAlphaBarrel0201[i] = ibooker.book1D(histo,title, 1000, -0.05, 0.05);

    // Resolution for -0.1<alpha<0
    sprintf(histo, "h21%d", i+1);
    sprintf(title, "alpha -0.1  0.0 pix %d", i+1);
    recHitResAlphaBarrel0100[i] = ibooker.book1D(histo,title, 1000, -0.05, 0.05);

    // Resolution for 0<alpha<0.1
    sprintf(histo, "h31%d", i+1);
    sprintf(title, "alpha  0.0  0.1 pix %d", i+1);
    recHitResAlphaBarrel0001[i] = ibooker.book1D(histo,title, 1000, -0.05, 0.05);

    // Resolution for 0.1<alpha<0.2
    sprintf(histo, "h41%d", i+1);
    sprintf(title, "alpha  0.1  0.2 pix %d", i+1);
    recHitResAlphaBarrel0102[i] = ibooker.book1D(histo,title, 1000, -0.05, 0.05);
  }

  for (int i=0; i<4; i++) {
    // Resolution for -0.2<alpha<-0.1
    sprintf(histo, "h11%db", i+1);
    sprintf(title, "alpha -0.2 -0.1 big pix %d", i+1);
    recHitResAlphaBarrelBigPixel0201[i] = ibooker.book1D(histo,title, 1000, -0.05, 0.05);

    // Resolution for -0.1<alpha<0
    sprintf(histo, "h21%db", i+1);
    sprintf(title, "alpha -0.1  0.0 big pix %d", i+1);
    recHitResAlphaBarrelBigPixel0100[i] = ibooker.book1D(histo,title, 1000, -0.05, 0.05);

    // Resolution for 0<alpha<0.1
    sprintf(histo, "h31%db", i+1);
    sprintf(title, "alpha  0.0  0.1 big pix %d", i+1);
    recHitResAlphaBarrelBigPixel0001[i] = ibooker.book1D(histo,title, 1000, -0.05, 0.05);

    // Resolution for 0.1<alpha<0.2
    sprintf(histo, "h41%db", i+1);
    sprintf(title, "alpha  0.1  0.2 big pix %d", i+1);
    recHitResAlphaBarrelBigPixel0102[i] = ibooker.book1D(histo,title, 1000, -0.05, 0.05);
  }

  ibooker.setCurrentFolder("clustFPIX");

  // RecHit X resolutions in forward
  for (int i=0; i<3; i++) {
    sprintf(histo, "h111%d", i+1);
    sprintf(title, "pixx %d", i+1);
    recHitResXForward[i] = ibooker.book1D(histo, title, 1000, -0.05, 0.05);
  }

  for (int i=0; i<3; i++) {
    sprintf(histo, "h111%db", i+1);
    sprintf(title, "big pixx %d", i+1);
    recHitResXForwardBigPixel[i] = ibooker.book1D(histo, title, 1000, -0.05, 0.05);
  }

  // RecHit Y resolutions in forward
  for (int i=0; i<3; i++) {
    sprintf(histo, "h110%d", i+1);
    sprintf(title, "pixy %d", i+1);
    recHitResYForward[i] = ibooker.book1D(histo, title, 1000, -0.05, 0.05);
  }

  for (int i=0; i<3; i++) {
    sprintf(histo, "h110%db", i+1);
    sprintf(title, "big pixy %d", i+1);
    recHitResYForwardBigPixel[i] = ibooker.book1D(histo, title, 1000, -0.05, 0.05);
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

SiPixelRecHitsInputDistributionsMaker::~SiPixelRecHitsInputDistributionsMaker() {
}

void SiPixelRecHitsInputDistributionsMaker::beginJob() { }

void SiPixelRecHitsInputDistributionsMaker::endJob() {
}

void SiPixelRecHitsInputDistributionsMaker::analyze(const edm::Event& e, const edm::EventSetup& es) 
{
  
  LogInfo("EventInfo") << " Run = " << e.id().run() << " Event = " << e.id().event();
  if ( (int) e.id().event() % 1000 == 0 )
    cout << " Run = " << e.id().run() << " Event = " << e.id().event() << endl;
  
  //Get RecHits
  edm::Handle<SiPixelRecHitCollection> recHitColl;
  e.getByLabel( src_, recHitColl);
  
  //Get event setup
  edm::ESHandle<TrackerGeometry> geom;
  es.get<TrackerDigiGeometryRecord>().get(geom);
  const TrackerGeometry& theTracker(*geom);

  TrackerHitAssociator associate( e, conf_ );

  //iterate over detunits
  for (TrackerGeometry::DetContainer::const_iterator it = geom->dets().begin(); it != geom->dets().end(); it++) 
    {
      DetId detId = ((*it)->geographicalId());
      unsigned int subid=detId.subdetId();
      
      if (! ((subid==1) || (subid==2))) continue;
      
      const PixelGeomDetUnit * theGeomDet = dynamic_cast<const PixelGeomDetUnit*>(theTracker.idToDet(detId) );
     
      SiPixelRecHitCollection::const_iterator pixelrechitMatch = recHitColl->find(detId);
      if (pixelrechitMatch == recHitColl->end()) continue; 
      SiPixelRecHitCollection::DetSet pixelrechitRange = *pixelrechitMatch;
      SiPixelRecHitCollection::DetSet::const_iterator pixelrechitRangeIteratorBegin = pixelrechitRange.begin();
      SiPixelRecHitCollection::DetSet::const_iterator pixelrechitRangeIteratorEnd = pixelrechitRange.end();
      SiPixelRecHitCollection::DetSet::const_iterator pixeliter = pixelrechitRangeIteratorBegin;
      std::vector<PSimHit> matched;
      
      //----Loop over rechits for this detId
      for ( ; pixeliter != pixelrechitRangeIteratorEnd; pixeliter++)
	{
	  matched.clear();
	  matched = associate.associateHit(*pixeliter);
	  
	  if ( !matched.empty() ) 
	    {
	      float closest = 9999.9;
	      std::vector<PSimHit>::const_iterator closestit = matched.begin();
	      LocalPoint lp = pixeliter->localPosition();
	      float rechit_x = lp.x();
	      float rechit_y = lp.y();

	      //loop over sim hits and fill closet
	      for (std::vector<PSimHit>::const_iterator m = matched.begin(); m<matched.end(); m++) 
		{
		  float sim_x1 = (*m).entryPoint().x();
		  float sim_x2 = (*m).exitPoint().x();
		  float sim_xpos = 0.5*(sim_x1+sim_x2);

		  float sim_y1 = (*m).entryPoint().y();
		  float sim_y2 = (*m).exitPoint().y();
		  float sim_ypos = 0.5*(sim_y1+sim_y2);
		  
		  float x_res = fabs(sim_xpos - rechit_x);
		  float y_res = fabs(sim_ypos - rechit_y);
		  
		  float dist = sqrt(x_res*x_res + y_res*y_res);

		  if ( dist < closest ) 
		    {
		      closest = x_res;
		      closestit = m;
		    }
		} // end sim hit loop
	      
	      if (subid==1) 
		{ //<----------barrel
		  fillBarrel(*pixeliter, *closestit, detId, theGeomDet);	
		} // end barrel
	      if (subid==2) 
		{ // <-------forward
		  fillForward(*pixeliter, *closestit, detId, theGeomDet);
		}
	      
	    } // end matched emtpy
	} // <-----end rechit loop 
    } // <------ end detunit loop
}

void SiPixelRecHitsInputDistributionsMaker::fillBarrel(const SiPixelRecHit& recHit, const PSimHit& simHit, 
                                                       DetId detId, const PixelGeomDetUnit* theGeomDet) 
{
  const float cmtomicron = 10000.0; 
  
  LocalPoint lp = recHit.localPosition();

  //  LocalError lerr = recHit.localPositionError();
  
  float sim_x1 = simHit.entryPoint().x();
  float sim_x2 = simHit.exitPoint().x();
  float sim_xpos = 0.5*(sim_x1 + sim_x2);
  float res_x = (lp.x() - sim_xpos)*cmtomicron;
  
  float sim_xdir = simHit.localDirection().x();
  float sim_ydir = simHit.localDirection().y();
  float sim_zdir = simHit.localDirection().z();

  // alpha: angle with respect to local x axis in local (x,z) plane
  float alpha = acos(sim_xdir/sqrt(sim_xdir*sim_xdir+sim_zdir*sim_zdir));
  // beta: angle with respect to local y axis in local (y,z) plane
  float beta = acos(sim_ydir/sqrt(sim_ydir*sim_ydir+sim_zdir*sim_zdir));
  
  float sim_y1 = simHit.entryPoint().y();
  float sim_y2 = simHit.exitPoint().y();
  float sim_ypos = 0.5*(sim_y1 + sim_y2);
  float res_y = (lp.y() - sim_ypos)*cmtomicron;
  
  float tmp1 = theGeomDet->surface().toGlobal(Local3DPoint(0.,0.,0.)).perp();
  float tmp2 = theGeomDet->surface().toGlobal(Local3DPoint(0.,0.,1.)).perp();
  
  bool isFlipped;
  
  if (tmp2<tmp1) 
    { // flipped
      isFlipped=true;
    }
  else 
    {
      isFlipped=false;
    }
    
  if ( isFlipped ) {
      alpha = PI - alpha ;
  }
  
  float alphaToBeUsedForRootFiles = PI/2. - alpha;
  float betaToBeUsedForRootFiles  = fabs( PI/2. - beta );

  //get cluster
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

#ifdef MUONSONLY
  if(abs(simHit.particleType()) == 13)
  {
#endif
    if(!hasBigPixelInX)
      simHitAlphaBarrel->Fill(alphaToBeUsedForRootFiles);
    else
      simHitAlphaBarrelBigPixel->Fill(alphaToBeUsedForRootFiles);
    if(!hasBigPixelInY)
      simHitBetaBarrel->Fill(betaToBeUsedForRootFiles);
    else
      simHitBetaBarrelBigPixel->Fill(betaToBeUsedForRootFiles);
#ifdef MUONSONLY
  }
#endif

  // Protection against overflow in size
  if(sizeY>7)
    sizeY = 7;
#ifdef MUONSONLY 
  if(abs(simHit.particleType()) == 13)
  {
#endif
    if(fabs(res_y/cmtomicron)<0.05)
    {
      if(hasBigPixelInY)
      {
        if(betaToBeUsedForRootFiles<0.2)
          recHitResBetaBarrelBigPixel00[sizeY - 1]->Fill(res_y/cmtomicron);
        if(betaToBeUsedForRootFiles>=0.2 && betaToBeUsedForRootFiles<0.4)
          recHitResBetaBarrelBigPixel02[sizeY - 1]->Fill(res_y/cmtomicron);
        if(betaToBeUsedForRootFiles>=0.4 && betaToBeUsedForRootFiles<0.6)
          recHitResBetaBarrelBigPixel04[sizeY - 1]->Fill(res_y/cmtomicron);
        if(betaToBeUsedForRootFiles>=0.6 && betaToBeUsedForRootFiles<0.8)
          recHitResBetaBarrelBigPixel06[sizeY - 1]->Fill(res_y/cmtomicron);
        if(betaToBeUsedForRootFiles>=0.8 && betaToBeUsedForRootFiles<1.0)
          recHitResBetaBarrelBigPixel08[sizeY - 1]->Fill(res_y/cmtomicron);
        if(betaToBeUsedForRootFiles>=1.0 && betaToBeUsedForRootFiles<1.2)
          recHitResBetaBarrelBigPixel10[sizeY - 1]->Fill(res_y/cmtomicron);
        if(betaToBeUsedForRootFiles>=1.2)
          recHitResBetaBarrelBigPixel12[sizeY - 1]->Fill(res_y/cmtomicron);
      }
      else
      {
        if(betaToBeUsedForRootFiles<0.2)
          recHitResBetaBarrel00[sizeY - 1]->Fill(res_y/cmtomicron);
        if(betaToBeUsedForRootFiles>=0.2 && betaToBeUsedForRootFiles<0.4)
          recHitResBetaBarrel02[sizeY - 1]->Fill(res_y/cmtomicron);
        if(betaToBeUsedForRootFiles>=0.4 && betaToBeUsedForRootFiles<0.6)
          recHitResBetaBarrel04[sizeY - 1]->Fill(res_y/cmtomicron);
        if(betaToBeUsedForRootFiles>=0.6 && betaToBeUsedForRootFiles<0.8)
          recHitResBetaBarrel06[sizeY - 1]->Fill(res_y/cmtomicron);
        if(betaToBeUsedForRootFiles>=0.8 && betaToBeUsedForRootFiles<1.0)
          recHitResBetaBarrel08[sizeY - 1]->Fill(res_y/cmtomicron);
        if(betaToBeUsedForRootFiles>=1.0 && betaToBeUsedForRootFiles<1.2)
          recHitResBetaBarrel10[sizeY - 1]->Fill(res_y/cmtomicron);
        if(betaToBeUsedForRootFiles>=1.2)
          recHitResBetaBarrel12[sizeY - 1]->Fill(res_y/cmtomicron);
      }
    }
#ifdef MUONSONLY
  }
#endif
//  // Limit for sizeY is 6 when computing beta probability in barrel
//  if(sizeY>6)
//    sizeY = 6;
#ifdef MUONSONLY
  if(abs(simHit.particleType()) == 13)
  {
#endif
    if(hasBigPixelInY)
      simHitBetaMultiBarrelBigPixel[sizeY]->Fill(betaToBeUsedForRootFiles);
    else
      simHitBetaMultiBarrel[sizeY]->Fill(betaToBeUsedForRootFiles);
#ifdef MUONSONLY
  }
#endif
  // Limit for sizeX is 4 when computing alpha probability in barrel
  if(sizeX>4)
    sizeX = 4;
#ifdef MUONSONLY
  if(abs(simHit.particleType()) == 13) 
  {
#endif
    if(hasBigPixelInX)
      simHitAlphaMultiBarrelBigPixel[sizeX]->Fill(alphaToBeUsedForRootFiles);
    else                                                                       
      simHitAlphaMultiBarrel[sizeX]->Fill(alphaToBeUsedForRootFiles);
#ifdef MUONSONLY
  }
#endif
//  // Protection against overflow if size
//  if(sizeX>4)
//    sizeX = 4;
#ifdef MUONSONLY
  if(abs(simHit.particleType()) == 13)
  {
#endif
    if(fabs(res_x/cmtomicron)<0.05)
    {
      if(hasBigPixelInX)
      {
        if(alphaToBeUsedForRootFiles<-0.1)
          recHitResAlphaBarrelBigPixel0201[sizeX - 1]->Fill(res_x/cmtomicron);
        if(alphaToBeUsedForRootFiles>=-0.1 && alphaToBeUsedForRootFiles<0)
          recHitResAlphaBarrelBigPixel0100[sizeX - 1]->Fill(res_x/cmtomicron);
        if(alphaToBeUsedForRootFiles>=0 && alphaToBeUsedForRootFiles<0.1)
          recHitResAlphaBarrelBigPixel0001[sizeX - 1]->Fill(res_x/cmtomicron);
        if(alphaToBeUsedForRootFiles>=0.1 && alphaToBeUsedForRootFiles<0.2)
          recHitResAlphaBarrelBigPixel0102[sizeX - 1]->Fill(res_x/cmtomicron);
      }
      else
      {
        if(alphaToBeUsedForRootFiles<-0.1)
          recHitResAlphaBarrel0201[sizeX - 1]->Fill(res_x/cmtomicron);
        if(alphaToBeUsedForRootFiles>=-0.1 && alphaToBeUsedForRootFiles<0)
          recHitResAlphaBarrel0100[sizeX - 1]->Fill(res_x/cmtomicron);
        if(alphaToBeUsedForRootFiles>=0 && alphaToBeUsedForRootFiles<0.1)
          recHitResAlphaBarrel0001[sizeX - 1]->Fill(res_x/cmtomicron);
        if(alphaToBeUsedForRootFiles>=0.1 && alphaToBeUsedForRootFiles<0.2)
          recHitResAlphaBarrel0102[sizeX - 1]->Fill(res_x/cmtomicron);
      }
    }
#ifdef MUONSONLY
  }
#endif
}

void SiPixelRecHitsInputDistributionsMaker::fillForward(const SiPixelRecHit & recHit, const PSimHit & simHit, 
                                                        DetId detId,const PixelGeomDetUnit * theGeomDet ) 
{
  const float cmtomicron = 10000.0;
  
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
  float alpha = acos(sim_xdir/sqrt(sim_xdir*sim_xdir+sim_zdir*sim_zdir));
  // beta: angle with respect to local y axis in local (y,z) plane
  float beta = acos(sim_ydir/sqrt(sim_ydir*sim_ydir+sim_zdir*sim_zdir));
  
  float res_x = (lp.x() - sim_xpos)*cmtomicron;
  
  float res_y = (lp.y() - sim_ypos)*cmtomicron;
  
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

#ifdef MUONSONLY
  if(abs(simHit.particleType()) == 13) 
  {
#endif
    if(hasBigPixelInX)
      simHitAlphaForwardBigPixel->Fill(fabs( PI/2. - alpha ));
    else
      simHitAlphaForward->Fill(fabs( PI/2. - alpha ));
    if(hasBigPixelInY)
      simHitBetaForwardBigPixel->Fill(fabs( PI/2. - beta ));
    else
      simHitBetaForward->Fill(fabs( PI/2. - beta ));
#ifdef MUONSONLY
  }
#endif

  // Limit for sizeX is 3 when computing alpha probability in forward
  if(sizeX>3)
    sizeX = 3;
#ifdef MUONSONLY
  if(abs(simHit.particleType()) == 13)
  { 
#endif
    if(hasBigPixelInX)
      simHitAlphaMultiForwardBigPixel[sizeX]->Fill(fabs( PI/2. - alpha ));
    else
      simHitAlphaMultiForward[sizeX]->Fill(fabs( PI/2. - alpha ));      
#ifdef MUONSONLY
  }
#endif

#ifdef MUONSONLY
  if(abs(simHit.particleType()) == 13)
  {
#endif
    if(fabs(res_x/cmtomicron)<0.05) {
      if(hasBigPixelInX)
        recHitResXForwardBigPixel[sizeX - 1]->Fill(res_x/cmtomicron);
      else
        recHitResXForward[sizeX - 1]->Fill(res_x/cmtomicron);
    }        
#ifdef MUONSONLY
  }
#endif

  // Protection against overflows
  if(sizeY>3)
    sizeY=3;
#ifdef MUONSONLY
  if(abs(simHit.particleType()) == 13)
  {
#endif
    if(fabs(res_y/cmtomicron)<0.05) {
      if(hasBigPixelInY)
        recHitResYForwardBigPixel[sizeY - 1]->Fill(res_y/cmtomicron);
      else
        recHitResYForward[sizeY - 1]->Fill(res_y/cmtomicron);
    }
#ifdef MUONSONLY
  }
#endif

#ifdef MUONSONLY
  if(abs(simHit.particleType()) == 13)
  {
#endif
    if(hasBigPixelInY)
      simHitBetaMultiForwardBigPixel[sizeY]->Fill(fabs( PI/2. - beta ));
    else
      simHitBetaMultiForward[sizeY]->Fill(fabs( PI/2. - beta ));
#ifdef MUONSONLY
  }
#endif
}

//

DEFINE_FWK_MODULE(SiPixelRecHitsInputDistributionsMaker);
