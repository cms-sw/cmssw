// SiPixelRecHitsValid.cc
// Description: see SiPixelRecHitsValid.h
// Author: Jason Shaev, JHU
// Created 6/7/06
//
// G. Giurgiu, JHU (ggiurgiu@pha.jhu.edu)
//             added pull distributions (12/27/06)
//--------------------------------
// K. Rose, RU
//  modified for DQM suite July 2007
//--------------------------------
//

#include "DQM/SiPixelMonitorRecHit/interface/SiPixelMonitorRecHit.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

#include <math.h>

using namespace std;
using namespace edm;


SiPixelMonitorRecHit::SiPixelMonitorRecHit(const ParameterSet& ps): 
  dbe_(0), 
  conf_(ps),
  src_( ps.getParameter<edm::InputTag>( "src" ) ) 
{
  outputFile_ = ps.getUntrackedParameter<string>("outputFile", "pixelrechitshisto.root");
  dbe_ = Service<DaqMonitorBEInterface>().operator->();
  dbe_->showDirStructure();
  
  Char_t histo[200];
  
  // ---------------------------------------------------------------
  // All histograms that depend on plaquette number have 7 indexes.
  // The first 4 (0-3) correspond to Panel 1 plaquettes 1-4.
  // The last 3 (4-6) correspond to Panel 2 plaquettes 1-3.
  // ---------------------------------------------------------------
  
  std::ostringstream dfolder;
  dfolder << "PixelRecHits/Summary";
  
  dbe_->setCurrentFolder(dfolder.str().c_str());
  //RecHit X Resolution all barrel hits
  recHitXResAllB = dbe_->book1D("RecHit_xres_b_All","RecHit X Res All Modules in Barrel", 100, -200., 200.);
  
  //RecHit Y Resolution all barrel hits
  recHitYResAllB = dbe_->book1D("RecHit_yres_b_All","RecHit Y Res All Modules in Barrel", 100, -200., 200.);
  
  //RecHit X distribution for full modules for barrel
  recHitXFullModules = dbe_->book1D("RecHit_x_FullModules", "RecHit X distribution for full modules", 100,-2., 2.);
  
  //RecHit X distribution for half modules for barrel
  recHitXHalfModules = dbe_->book1D("RecHit_x_HalfModules", "RecHit X distribution for half modules", 100, -1., 1.);
  
  //RecHit Y distribution all modules for barrel
  recHitYAllModules = dbe_->book1D("RecHit_y_AllModules", "RecHit Y distribution for all modules", 100, -4., 4.);
  

  //RecHit X resolution for flipped and unflipped ladders by layer for barrel
  for (int i=0; i<3; i++) {

    char slayer[80]; sprintf(slayer, "Layer_%i", (i+1));
    std::ostringstream sfolder;
    sfolder << "Tracker/PixelBarrel/" << slayer << "/FlippedLadders";
    dbe_->setCurrentFolder(sfolder.str().c_str());

    //RecHit X resolution for flipped ladders by layer
    sprintf(histo, "RecHit_XRes_FlippedLadder_Layer%d", i+1);
    recHitXResFlippedLadderLayers[i] = dbe_->book1D(histo, "RecHit XRes Flipped Ladders by Layer", 100, -200., 200.);

    sprintf(histo, "RecHit_XPull_FlippedLadder_Layer%d", i+1);
    recHitXPullFlippedLadderLayers[i] = dbe_->book1D(histo, "RecHit XPull Flipped Ladders by Layer", 100, -10.0, 10.0);
    
    std::ostringstream sfolder2;
    sfolder2 << "Tracker/PixelBarrel/" << slayer << "/UnflippedLadders";
    dbe_->setCurrentFolder(sfolder2.str().c_str());
    
    //RecHit X resolution for unflipped ladders by layer
    sprintf(histo, "RecHit_XRes_UnFlippedLadder_Layer%d", i+1);
    recHitXResNonFlippedLadderLayers[i] = dbe_->book1D(histo, "RecHit XRes NonFlipped Ladders by Layer", 100, -200., 200.);

    sprintf(histo, "RecHit_XPull_UnFlippedLadder_Layer%d", i+1);
    recHitXPullNonFlippedLadderLayers[i] = dbe_->book1D(histo, "RecHit XPull NonFlipped Ladders by Layer", 100, -10.0, 10.0);

  } // end for
  
  //RecHit Y resolutions for layers by module for barrel
  for (int i=0; i<8; i++) {
    
    char smodule[80]; sprintf(smodule, "Module_%i", (i+1));
    std::ostringstream sfolder;
    sfolder << "Tracker/PixelBarrel/Layer_1/" << smodule;
    dbe_->setCurrentFolder(sfolder.str().c_str());

    //Rec Hit Y resolution by module for Layer1
    sprintf(histo, "RecHit_YRes_Layer1_Module%d", i+1);
    recHitYResLayer1Modules[i] = dbe_->book1D(histo, "RecHit YRes Layer1 by module", 100, -200., 200.);

    std::ostringstream sfolder2;
    sfolder2 << "Tracker/PixelBarrel/Layer_2/" << smodule;
    dbe_->setCurrentFolder(sfolder2.str().c_str());
    
    //RecHit Y resolution by module for Layer2
    sprintf(histo, "RecHit_YRes_Layer2_Module%d", i+1);
    recHitYResLayer2Modules[i] = dbe_->book1D(histo, "RecHit YRes Layer2 by module", 100, -200., 200.);
    
    std::ostringstream sfolder3;
    sfolder3 << "Tracker/PixelBarrel/Layer_3/" << smodule;
    dbe_->setCurrentFolder(sfolder3.str().c_str());

    //RecHit Y resolution by module for Layer3
    sprintf(histo, "RecHit_YRes_Layer3_Module%d", i+1);
    recHitYResLayer3Modules[i] = dbe_->book1D(histo, "RecHit YRes Layer3 by module", 100, -200., 200.); 
  } // end for
  
  dbe_->setCurrentFolder(dfolder.str().c_str());
  //RecHit X resolution all plaquettes
  recHitXResAllF = dbe_->book1D("RecHit_xres_f_All", "RecHit X Res All in Forward", 100, -200., 200.);
  
  //RecHit Y resolution all plaquettes
  recHitYResAllF = dbe_->book1D("RecHit_yres_f_All", "RecHit Y Res All in Forward", 100, -200., 200.);
  
  //RecHit X distribution for plaquette with x-size 1 in forward
  recHitXPlaquetteSize1 = dbe_->book1D("RecHit_x_Plaquette_xsize1", "RecHit X Distribution for plaquette x-size1", 100, -2., 2.);
  
  //RecHit X distribution for plaquette with x-size 2 in forward
  recHitXPlaquetteSize2 = dbe_->book1D("RecHit_x_Plaquette_xsize2", "RecHit X Distribution for plaquette x-size2", 100, -2., 2.);
  
  //RecHit Y distribution for plaquette with y-size 2 in forward
  recHitYPlaquetteSize2 = dbe_->book1D("RecHit_y_Plaquette_ysize2", "RecHit Y Distribution for plaquette y-size2", 100, -4., 4.);
  
  //RecHit Y distribution for plaquette with y-size 3 in forward
  recHitYPlaquetteSize3 = dbe_->book1D("RecHit_y_Plaquette_ysize3", "RecHit Y Distribution for plaquette y-size3", 100, -4., 4.);
  
  //RecHit Y distribution for plaquette with y-size 4 in forward
  recHitYPlaquetteSize4 = dbe_->book1D("RecHit_y_Plaquette_ysize4", "RecHit Y Distribution for plaquette y-size4", 100, -4., 4.);
  
  //RecHit Y distribution for plaquette with y-size 5 in forward
  recHitYPlaquetteSize5 = dbe_->book1D("RecHit_y_Plaquette_ysize5", "RecHit Y Distribution for plaquette y-size5", 100, -4., 4.);
  
  //X and Y resolutions for both disks by plaquette in forward
  for (int i=0; i<7; i++) {

    char splaquette[80]; sprintf(splaquette, "Plaquette_%i", (i+1));
    std::ostringstream sfolder;
    sfolder << "Tracker/PixelForward/Disk_1/" << splaquette;
    dbe_->setCurrentFolder(sfolder.str().c_str());

    //X-Y distribution for Disk1 by plaquette
    sprintf(histo, "RecHit_XYPos_Disk1_Plaquette%d", i+1);
    recHitXYPosDisk1Plaquettes[i] = dbe_->book2D(histo, "RecHit XYPos Disk1 by plaquette", 100, -4, 4, 100, -4, 4);

    //X resolution for Disk1 by plaquette
    sprintf(histo, "RecHit_XRes_Disk1_Plaquette%d", i+1);
    recHitXResDisk1Plaquettes[i] = dbe_->book1D(histo, "RecHit XRes Disk1 by plaquette", 100, -200., 200.); 

    //Y resolution for Disk1 by plaquette
    sprintf(histo, "RecHit_YRes_Disk1_Plaquette%d", i+1);
    recHitYResDisk1Plaquettes[i] = dbe_->book1D(histo, "RecHit YRes Disk1 by plaquette", 100, -200., 200.);

    sprintf(histo, "RecHit_XPull_Disk1_Plaquette%d", i+1);
    recHitXPullDisk1Plaquettes[i] = dbe_->book1D(histo, "RecHit XPull Disk1 by plaquette", 100, -10.0, 10.0); 

    sprintf(histo, "RecHit_YPull_Disk1_Plaquette%d", i+1);
    recHitYPullDisk1Plaquettes[i] = dbe_->book1D(histo, "RecHit YPull Disk1 by plaquette", 100, -10.0, 10.0);

    std::ostringstream sfolder2;
    sfolder2 << "Tracker/PixelForward/Disk_2/" << splaquette;
    dbe_->setCurrentFolder(sfolder2.str().c_str());

     sprintf(histo, "RecHit_XYPos_Disk2_Plaquette%d", i+1);
    recHitXYPosDisk2Plaquettes[i] = dbe_->book2D(histo, "RecHit XYPos Disk2 by plaquette", 100, -4, 4, 100, -4, 4);

    //X resolution for Disk2 by plaquette
    sprintf(histo, "RecHit_XRes_Disk2_Plaquette%d", i+1);
    recHitXResDisk2Plaquettes[i] = dbe_->book1D(histo, "RecHit XRes Disk2 by plaquette", 100, -200., 200.);  
    
    //Y resolution for Disk2 by plaquette
    sprintf(histo, "RecHit_YRes_Disk2_Plaquette%d", i+1);
    recHitYResDisk2Plaquettes[i] = dbe_->book1D(histo, "RecHit YRes Disk2 by plaquette", 100, -200., 200.);

    sprintf(histo, "RecHit_XPull_Disk2_Plaquette%d", i+1);
    recHitXPullDisk2Plaquettes[i] = dbe_->book1D(histo, "RecHit XPull Disk2 by plaquette", 100, -10.0, 10.0);  
        
    sprintf(histo, "RecHit_YPull_Disk2_Plaquette%d", i+1);
    recHitYPullDisk2Plaquettes[i] = dbe_->book1D(histo, "RecHit YPull Disk2 by plaquette", 100, -10.0, 10.0);
    
  }


  dbe_->setCurrentFolder(dfolder.str().c_str());
  recHitXPullAllB = dbe_->book1D("RecHit_XPull_b_All", "RecHit X Pull All Modules in Barrel", 100, -10.0, 10.0);
  recHitYPullAllB = dbe_->book1D("RecHit_YPull_b_All", "RecHit Y Pull All Modules in Barrel", 100, -10.0, 10.0);
  recHitMultLayer1 = dbe_->book1D("RecHit_Mult_Layer1", "Rec Hit Multiplicity in Layer 1", 10, 0, 10);
  recHitMultLayer2 = dbe_->book1D("RecHit_Mult_Layer2", "Rec Hit Multiplicity in Layer 2", 10, 0, 10);
  recHitMultLayer3 = dbe_->book1D("RecHit_Mult_Layer3", "Rec Hit Multiplicity in Layer 3", 10, 0, 10);


  for (int i=0; i<8; i++) 
    {

      char smodule[80]; sprintf(smodule, "Module_%i", (i+1));
      std::ostringstream sfolder;
      sfolder << "Tracker/PixelBarrel/Layer_1/" << smodule;
      dbe_->setCurrentFolder(sfolder.str().c_str());

      sprintf(histo, "RecHit_XYPos_Layer1_Module%d", i+1);
      recHitXYPosLayer1Modules[i] = dbe_->book2D(histo, "RecHit XYPos Layer 1 by module", 100, -4, 4, 100, -4, 4);

      sprintf(histo, "RecHit_YPull_Layer1_Module%d", i+1);
      recHitYPullLayer1Modules[i] = dbe_->book1D(histo, "RecHit YPull Layer1 by module", 100, -10.0, 10.0);

      std::ostringstream sfolder2;
      sfolder2 << "Tracker/PixelBarrel/Layer_2/" << smodule;
      dbe_->setCurrentFolder(sfolder2.str().c_str());

      sprintf(histo, "RecHit_XYPos_Layer2_Module%d", i+1);
      recHitXYPosLayer2Modules[i] = dbe_->book2D(histo, "RecHit XYPos Layer 2 by module", 100, -4, 4, 100, -4, 4);
      
      sprintf(histo, "RecHit_YPull_Layer2_Module%d", i+1);
      recHitYPullLayer2Modules[i] = dbe_->book1D(histo, "RecHit YPull Layer2 by module", 100, -10.0, 10.0);
      
      std::ostringstream sfolder3;
      sfolder3 << "Tracker/PixelBarrel/Layer_3/" << smodule;
      dbe_->setCurrentFolder(sfolder3.str().c_str());
      
      sprintf(histo, "RecHit_XYPos_Layer3_Module%d", i+1);
      recHitXYPosLayer3Modules[i] = dbe_->book2D(histo, "RecHit XYPos Layer 3 by module", 100, -4, 4, 100, -4, 4);

      sprintf(histo, "RecHit_YPull_Layer3_Module%d", i+1);
      recHitYPullLayer3Modules[i] = dbe_->book1D(histo, "RecHit YPull Layer3 by module", 100, -10.0, 10.0); 
    }
  
  dbe_->setCurrentFolder(dfolder.str().c_str());
  recHitXPullAllF = dbe_->book1D("RecHit_XPull_f_All", "RecHit X Pull All in Forward", 100, -10.0, 10.0);
  
  recHitYPullAllF = dbe_->book1D("RecHit_YPull_f_All", "RecHit Y Pull All in Forward", 100, -10.0, 10.0);
  

}

SiPixelMonitorRecHit::~SiPixelMonitorRecHit() {
}

void SiPixelMonitorRecHit::beginJob(const EventSetup& c) {
  
}

void SiPixelMonitorRecHit::endJob() {
  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}

void SiPixelMonitorRecHit::analyze(const edm::Event& e, const edm::EventSetup& es) 
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
      
      SiPixelRecHitCollection::range pixelrechitRange = (recHitColl.product())->get(detId);
      SiPixelRecHitCollection::const_iterator pixelrechitRangeIteratorBegin = pixelrechitRange.first;
      SiPixelRecHitCollection::const_iterator pixelrechitRangeIteratorEnd = pixelrechitRange.second;
      SiPixelRecHitCollection::const_iterator pixeliter = pixelrechitRangeIteratorBegin;
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

void SiPixelMonitorRecHit::fillBarrel(const SiPixelRecHit& recHit, const PSimHit& simHit, 
				     DetId detId, const PixelGeomDetUnit* theGeomDet) 
{
  const float cmtomicron = 10000.0; 
  LocalPoint lp = recHit.localPosition();
  float lp_y = lp.y();  
  float lp_x = lp.x();
  LocalError lerr = recHit.localPositionError();
  float lerr_x = sqrt(lerr.xx());
  float lerr_y = sqrt(lerr.yy());
  
  recHitYAllModules->Fill(lp_y);
  float sim_x1 = simHit.entryPoint().x();
  float sim_x2 = simHit.exitPoint().x();
  float sim_xpos = 0.5*(sim_x1 + sim_x2);
  float res_x = (lp.x() - sim_xpos)*cmtomicron;
  
  recHitXResAllB->Fill(res_x);
  float sim_y1 = simHit.entryPoint().y();
  float sim_y2 = simHit.exitPoint().y();
  float sim_ypos = 0.5*(sim_y1 + sim_y2);
  float res_y = (lp.y() - sim_ypos)*cmtomicron;
  
  recHitYResAllB->Fill(res_y);
  float pull_x = 0;
  float pull_y = 0;
  pull_x = ( lp_x - sim_xpos ) / lerr_x;
  pull_y = ( lp_y - sim_ypos ) / lerr_y;
  recHitXPullAllB->Fill(pull_x);  
  recHitYPullAllB->Fill(pull_y);
  int rows = theGeomDet->specificTopology().nrows();
  
  if (rows == 160) 
    {
      recHitXFullModules->Fill(lp_x);
    } 
  else if (rows == 80) 
    {
      recHitXHalfModules->Fill(lp_x);
    }
  float tmp1 = theGeomDet->surface().toGlobal(Local3DPoint(0.,0.,0.)).perp();
  float tmp2 = theGeomDet->surface().toGlobal(Local3DPoint(0.,0.,1.)).perp();
  if (tmp2<tmp1) 
    { // flipped
      for (unsigned int i=0; i<3; i++) 
	{
	  if (PXBDetId::PXBDetId(detId).layer() == i+1) 
	    {
	      recHitXResFlippedLadderLayers[i]->Fill(res_x);
	      recHitXPullFlippedLadderLayers[i]->Fill(pull_x);
	    }
	}
    } 
  else 
    {
      for (unsigned int i=0; i<3; i++) 
	{
	  if (PXBDetId::PXBDetId(detId).layer() == i+1) 
	    {
	      recHitXResNonFlippedLadderLayers[i]->Fill(res_x);
	      recHitXPullNonFlippedLadderLayers[i]->Fill(pull_x);
	    }
	}
    }
  
  // fill module dependent info
  for (unsigned int i=0; i<8; i++) 
    {
      if (PXBDetId::PXBDetId(detId).module() == i+1) 
	{
	  if (PXBDetId::PXBDetId(detId).layer() == 1) 
	    {
	      recHitMultLayer1->Fill(i);
	      recHitXYPosLayer1Modules[i]->Fill(lp_x, lp_y);
	      recHitYResLayer1Modules[i]->Fill(res_y);
	      recHitYPullLayer1Modules[i]->Fill(pull_y);
	    }
	  else if (PXBDetId::PXBDetId(detId).layer() == 2) 
	    {
	      recHitMultLayer2->Fill(i);
	      recHitXYPosLayer2Modules[i]->Fill(lp_x, lp_y);
	      recHitYResLayer2Modules[i]->Fill(res_y);
	      recHitYPullLayer2Modules[i]->Fill(pull_y);
	    }
	  else if (PXBDetId::PXBDetId(detId).layer() == 3) 
	    {
	      recHitMultLayer3->Fill(i);
	      recHitXYPosLayer3Modules[i]->Fill(lp_x, lp_y);
	      recHitYResLayer3Modules[i]->Fill(res_y);
	      recHitYPullLayer3Modules[i]->Fill(pull_y);
	    }
	}
    }
}

void SiPixelMonitorRecHit::fillForward(const SiPixelRecHit & recHit, const PSimHit & simHit, 
				      DetId detId,const PixelGeomDetUnit * theGeomDet ) 
{
  int rows = theGeomDet->specificTopology().nrows();
  int cols = theGeomDet->specificTopology().ncolumns();
  
  const float cmtomicron = 10000.0;
  
  LocalPoint lp = recHit.localPosition();
  float lp_x = lp.x();
  float lp_y = lp.y();
  
  LocalError lerr = recHit.localPositionError();
  float lerr_x = sqrt(lerr.xx());
  float lerr_y = sqrt(lerr.yy());

  float sim_x1 = simHit.entryPoint().x();
  float sim_x2 = simHit.exitPoint().x();
  float sim_xpos = 0.5*(sim_x1 + sim_x2);
  
  float sim_y1 = simHit.entryPoint().y();
  float sim_y2 = simHit.exitPoint().y();
  float sim_ypos = 0.5*(sim_y1 + sim_y2);
  
  float pull_x = ( lp_x - sim_xpos ) / lerr_x;
  float pull_y = ( lp_y - sim_ypos ) / lerr_y;


  if (rows == 80) 
    {
      recHitXPlaquetteSize1->Fill(lp_x);
    } 
  else if (rows == 160) 
    {
      recHitXPlaquetteSize2->Fill(lp_x);
    }
  
  if (cols == 104) 
    {
      recHitYPlaquetteSize2->Fill(lp_y);
    } 
  else if (cols == 156) 
    {
      recHitYPlaquetteSize3->Fill(lp_y);
    } 
  else if (cols == 208) 
    {
      recHitYPlaquetteSize4->Fill(lp_y);
    } 
  else if (cols == 260) 
    {
      recHitYPlaquetteSize5->Fill(lp_y);
    }
  
  float res_x = (lp.x() - sim_xpos)*cmtomicron;
  
  recHitXResAllF->Fill(res_x);
  recHitXPullAllF->Fill(pull_x);

  float res_y = (lp.y() - sim_ypos)*cmtomicron;
  
  recHitYPullAllF->Fill(pull_y);
  
  // fill plaquette dependent info
  for (unsigned int i=0; i<7; i++) 
    {
      if (PXFDetId::PXFDetId(detId).module() == i+1) 
	{
	  if (PXFDetId::PXFDetId(detId).disk() == 1) 
	    {

	      recHitXYPosDisk1Plaquettes[i]->Fill(lp_x, lp_y);
	      
	      recHitXResDisk1Plaquettes[i]->Fill(res_x);
	      recHitYResDisk1Plaquettes[i]->Fill(res_y);

	      recHitXPullDisk1Plaquettes[i]->Fill(pull_x);
	      recHitYPullDisk1Plaquettes[i]->Fill(pull_y);
	    }
	  else 
	    {

	      recHitXYPosDisk2Plaquettes[i]->Fill(lp_x, lp_y);
 
	      recHitXResDisk2Plaquettes[i]->Fill(res_x);
	      recHitYResDisk2Plaquettes[i]->Fill(res_y);

	      recHitXPullDisk2Plaquettes[i]->Fill(pull_x);
	      recHitYPullDisk2Plaquettes[i]->Fill(pull_y);
	      
	    } // end else
	} // end if module
      else if (PXFDetId(detId).panel() == 2 && (PXFDetId(detId).module()+4) == i+1) 
	{
	  if (PXFDetId::PXFDetId(detId).disk() == 1) 
	    {	      
	      recHitXResDisk1Plaquettes[i]->Fill(res_x);
	      recHitYResDisk1Plaquettes[i]->Fill(res_y);

	      recHitXPullDisk1Plaquettes[i]->Fill(pull_x);
	      recHitYPullDisk1Plaquettes[i]->Fill(pull_y);
	    }
	  else 
	    { 
	      recHitXResDisk2Plaquettes[i]->Fill(res_x);
	      recHitYResDisk2Plaquettes[i]->Fill(res_y);

	      recHitXPullDisk2Plaquettes[i]->Fill(pull_x);
	      recHitYPullDisk2Plaquettes[i]->Fill(pull_y);

	    } // end else
        } // end else
    } // end for
}
