
#include <memory>
#include <string>
#include <iostream>
#include <fstream>

#include "CalibTracker/SiStripLorentzAngle/interface/SiStripLorentzAngle.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/Common/interface/OwnVector.h"

using namespace std;

  //Constructor

SiStripLorentzAngle::SiStripLorentzAngle(edm::ParameterSet const& conf) : 
  conf_(conf), filename_(conf.getParameter<std::string>("fileName"))
{
//  anglefinder_=new  TrackLocalAngle(conf);  
}

  //BeginJob

void SiStripLorentzAngle::beginJob(const edm::EventSetup& c){

  hFile = new TFile (filename_.c_str(), "RECREATE" );
  
  CollHitSizeTIBL2mono = new TH1F("CollHitSizeTIBL2mono","Hit Size of TIB Layer 2 MONO", 10, 0, 10);
  CollHitSizeTIBL2stereo = new TH1F("CollHitSizeTIBL2stereo","Hit Size of TIB Layer 2 STEREO", 10, 0, 10);
  CollHitSizeTIBL3 = new TH1F("CollHitSizeTIBL3","Hit Size of TIB Layer 3", 10, 0, 10);
  CollHitSizeTOBL1 = new TH1F("CollHitSizeTOBL1","Hit Size of TOB Layer 1", 10, 0, 10);
  CollHitSizeTOBL5 = new TH1F("CollHitSizeTOBL5","Hit Size of TOB Layer 5", 10, 0, 10);
  TrackHitSizeTIBL2mono = new TH1F("TrackHitSizeTIBL2mono","Hit Size of TIB Layer 2 MONO", 10, 0, 10);
  TrackHitSizeTIBL2stereo = new TH1F("TrackHitSizeTIBL2stereo","Hit Size of TIB Layer 2 STEREO", 10, 0, 10);
  TrackHitSizeTIBL3 = new TH1F("TrackHitSizeTIBL3","Hit Size of TIB Layer 3", 10, 0, 10);
  TrackHitSizeTOBL1 = new TH1F("TrackHitSizeTOBL1","Hit Size of TOB Layer 1", 10, 0, 10);
  TrackHitSizeTOBL5 = new TH1F("TrackHitSizeTOBL5","Hit Size of TOB Layer 5", 10, 0, 10);
  
  TAxis *xaxis, *yaxis;
        
  SiStripLorentzAngleTree = new TTree("SiStripLorentzAngleTree","Lorentz Angle Tree");
  SiStripLorentzAngleTree->Branch("run", &run, "run/I");
  SiStripLorentzAngleTree->Branch("event", &event, "event/I");
  SiStripLorentzAngleTree->Branch("module", &module, "module/I");
  SiStripLorentzAngleTree->Branch("type", &type, "type/I");
  SiStripLorentzAngleTree->Branch("layer", &layer, "layer/I");
  SiStripLorentzAngleTree->Branch("string", &string, "string/I");
  SiStripLorentzAngleTree->Branch("rod", &rod, "rod/I");
  SiStripLorentzAngleTree->Branch("extint", &extint, "extint/I");
  SiStripLorentzAngleTree->Branch("size", &size, "size/I");
  SiStripLorentzAngleTree->Branch("TrackLocalAngle", &TrackLocalAngle, "TrackLocalAngle/F");
  SiStripLorentzAngleTree->Branch("bwfw", &bwfw, "bwfw/I");
  SiStripLorentzAngleTree->Branch("wheel", &wheel, "wheel/I");
  SiStripLorentzAngleTree->Branch("monostereo", &monostereo, "monostereo/I");
  SiStripLorentzAngleTree->Branch("signprojcorrection", &signprojcorrection, "signprojcorrection/F");
  SiStripLorentzAngleTree->Branch("localmagfield", &localmagfield, "localmagfield/F");
  SiStripLorentzAngleTree->Branch("momentum", &momentum, "momentum/F");
  SiStripLorentzAngleTree->Branch("pt", &pt, "pt/F");
  SiStripLorentzAngleTree->Branch("ParticleCharge", &ParticleCharge, "ParticleCharge/I");
  SiStripLorentzAngleTree->Branch("chi2norm", &chi2norm, "chi2norm/F");
  SiStripLorentzAngleTree->Branch("chi2", &chi2, "chi2/F");
  SiStripLorentzAngleTree->Branch("ndof", &ndof, "ndof/F");
  SiStripLorentzAngleTree->Branch("tangent", &tangent, "tangent/F");
  SiStripLorentzAngleTree->Branch("trackproj", &trackproj, "trackproj/F");
  SiStripLorentzAngleTree->Branch("hitscharge", &hitscharge, "hitscharge/I");
  SiStripLorentzAngleTree->Branch("ThetaTrack", &ThetaTrack, "ThetaTrack/F");
  SiStripLorentzAngleTree->Branch("PhiTrack", &PhiTrack, "PhiTrack/F");
  SiStripLorentzAngleTree->Branch("SeedLayer", &SeedLayer, "SeedLayer/I");
  SiStripLorentzAngleTree->Branch("TOB_YtoGlobalSign", &TOB_YtoGlobalSign, "TOB_YtoGlobalSign/I");
  
  TrackHitTree = new TTree("TrackHitTree", "TrackHitTree");
  TrackHitTree->Branch("run", &run, "run/I");
  TrackHitTree->Branch("event", &event, "event/I");
  TrackHitTree->Branch("hitspertrack", &hitspertrack, "hitspertrack/I");
  TrackHitTree->Branch("trackcollsize", &trackcollsize, "trackcollsize/I");
  TrackHitTree->Branch("trajsize", &trajsize, "trajsize/I");
  TrackHitTree->Branch("TIBlayer2", &TIBlayer2, "TIBlayer2/I");
  TrackHitTree->Branch("TIBlayer3", &TIBlayer3, "TIBlayer3/I");
  TrackHitTree->Branch("TOBlayer1", &TOBlayer1, "TOBlayer1/I");
  TrackHitTree->Branch("TOBlayer5", &TOBlayer5, "TOBlayer5/I");
  TrackHitTree->Branch("TECwheel1", &TECwheel1, "TECwheel1/I");
  
  TrackHitTree->Branch("MONOhitsTIBL2collection", &MONOhitsTIBL2collection, "MONOhitsTIBL2collection/I");
  TrackHitTree->Branch("STEREOhitsTIBL2collection", &STEREOhitsTIBL2collection, "STEREOhitsTIBL2collection/I");
  TrackHitTree->Branch("hitsTIBL3collection", &hitsTIBL3collection, "hitsTIBL3collection/I");
  TrackHitTree->Branch("hitsTOBL1collection", &hitsTOBL1collection, "hitsTOBL1collection/I");
  TrackHitTree->Branch("hitsTOBL5collection", &hitsTOBL5collection, "hitsTOBL5collection/I");
  TrackHitTree->Branch("MONOhitsTECcollection", &MONOhitsTECcollection, "MONOhitsTECcollection/I");
  TrackHitTree->Branch("STEREOhitsTECcollection", &STEREOhitsTECcollection, "STEREOhitsTECcollection/I");
  
  TrackHitTree->Branch("MONOhitschargeTIBL2", &MONOhitschargeTIBL2, "MONOhitschargeTIBL2[MONOhitsTIBL2collection]/I");
  TrackHitTree->Branch("STEREOhitschargeTIBL2", &STEREOhitschargeTIBL2, "STEREOhitschargeTIBL2[STEREOhitsTIBL2collection]/I");
  TrackHitTree->Branch("hitschargeTIBL3", &hitschargeTIBL3, "hitschargeTIBL3[hitsTIBL3collection]/I");
  TrackHitTree->Branch("hitschargeTOBL1", &hitschargeTOBL1, "hitschargeTOBL1[hitsTOBL1collection]/I");
  TrackHitTree->Branch("hitschargeTOBL5", &hitschargeTOBL5, "hitschargeTIBL3[hitsTOBL5collection]/I");
  TrackHitTree->Branch("MONOhitschargeTEC", &MONOhitschargeTEC, "MONOhitschargeTEC[MONOhitsTECcollection]/I");
  TrackHitTree->Branch("STEREOhitschargeTEC", &STEREOhitschargeTEC, "STEREOhitschargeTEC[STEREOhitsTECcollection]/I");
    
  TrackHitTree->Branch("hitsTOBcoll", &hitsTOBcoll, "hitsTOBcoll/I");
  TrackHitTree->Branch("TOB_YtoGlobalSignColl", &TOB_YtoGlobalSignColl, "TOB_YtoGlobalSignColl[hitsTOBcoll]/I");
  TrackHitTree->Branch("ThetaTrack", &ThetaTrack, "ThetaTrack/F");
  TrackHitTree->Branch("PhiTrack", &PhiTrack, "PhiTrack/F");
  TrackHitTree->Branch("SeedLayer", &SeedLayer, "SeedLayer/I");
  TrackHitTree->Branch("SeedSize", &SeedSize, "SeedSize/I");
     
  monodscounter=0;
  monosscounter=0;
  stereocounter=0;
  
  mtcctibcorr = 0;
  mtcctobcorr = 0;
  
  if(conf_.getParameter<bool>("MTCCtrack")){  //MTCCtrack TRUE
  mtcctibcorr = 1;
  mtcctobcorr = 3;
  }
  
   //Get binning
  int TIBbinning=conf_.getParameter<int>("TIBbinning");
  int TOBbinning=conf_.getParameter<int>("TOBbinning");
        	   
  edm::ESHandle<MagneticField> esmagfield;
  c.get<IdealMagneticFieldRecord>().get(esmagfield);
  magfield=&(*esmagfield);
      
  edm::ESHandle<TrackerGeometry> estracker;
  c.get<TrackerDigiGeometryRecord>().get(estracker);
  tracker=&(*estracker); 
  
  //Get Ids;
  
  const TrackerGeometry::DetIdContainer& Id = estracker->detIds();
   
  TrackerGeometry::DetIdContainer::iterator Iditer;
       
for(Iditer=Id.begin();Iditer!=Id.end();Iditer++){
  
  	if((Iditer->subdetId() != PixelSubdetector::PixelBarrel) && (Iditer->subdetId() != PixelSubdetector::PixelEndcap)){
	   
		StripSubdetector subid(*Iditer);
				
		//Mono single sided detectors
		
		if(subid.glued() == 0){
		
		        if(conf_.getParameter<bool>("MTCCtrack")){  //MTCCtrack TRUE
			
			if(subid.subdetId() == int (StripSubdetector::TIB)){ 
			TIBDetId TIBid=TIBDetId(*Iditer);
			if(TIBid.layer()!=1){
			const GeomDetUnit * stripdet=(const GeomDetUnit*)tracker->idToDetUnit(subid);
			const StripTopology& topol=(StripTopology&)stripdet->topology();
			float thickness=stripdet->specificSurface().bounds().thickness();		
			monosscounter++;
    			Detvector.push_back(*Iditer);
    			histos[Iditer->rawId()] = new TProfile(makename(*Iditer),makedescription(*Iditer),30,-0.6,0.6);
			detmap[Iditer->rawId()] = new detparameters;
			detmap[Iditer->rawId()]->thickness = thickness*10000;
			detmap[Iditer->rawId()]->pitch = topol.pitch()*10000;
			xaxis = histos[Iditer->rawId()]->GetXaxis();
    			xaxis->SetTitle("tan(#theta_{t})");
    			yaxis = histos[Iditer->rawId()]->GetYaxis();
    			yaxis->SetTitle("Cluster size");
			}
			}
			
			}
		
		        if(!(conf_.getParameter<bool>("MTCCtrack"))){  //MTCCtrack FALSE
			
			if(subid.subdetId() == int (StripSubdetector::TIB)){
			TIBDetId TIBid=TIBDetId(*Iditer);
			if((TIBid.layer()!=1) && (TIBid.layer()!=2)){
			const GeomDetUnit * stripdet=(const GeomDetUnit*)tracker->idToDetUnit(subid);
			const StripTopology& topol=(StripTopology&)stripdet->topology();
			float thickness=stripdet->specificSurface().bounds().thickness();			
		   	monosscounter++;
    			Detvector.push_back(*Iditer);
    			histos[Iditer->rawId()] = new TProfile(makename(*Iditer),makedescription(*Iditer),30,-0.6,0.6);
			detmap[Iditer->rawId()] = new detparameters;
			detmap[Iditer->rawId()]->thickness = thickness*10000;
                        detmap[Iditer->rawId()]->pitch = topol.pitch()*10000;
    			xaxis = histos[Iditer->rawId()]->GetXaxis();
    			xaxis->SetTitle("tan(#theta_{t})");
    			yaxis = histos[Iditer->rawId()]->GetYaxis();
    			yaxis->SetTitle("Cluster size");
			}
			}
			
			}      
			
			if(conf_.getParameter<bool>("MTCCtrack")){  //MTCCtrack TRUE
						
			if(subid.subdetId() == int (StripSubdetector::TOB)){ 
			const GeomDetUnit * stripdet=(const GeomDetUnit*)tracker->idToDetUnit(subid);
			const StripTopology& topol=(StripTopology&)stripdet->topology();
			float thickness=stripdet->specificSurface().bounds().thickness(); 
			monosscounter++;
    			Detvector.push_back(*Iditer);
    			histos[Iditer->rawId()] = new TProfile(makename(*Iditer),makedescription(*Iditer),30,-0.6,0.6);
			detmap[Iditer->rawId()] = new detparameters;
			detmap[Iditer->rawId()]->thickness = thickness*10000;
                        detmap[Iditer->rawId()]->pitch = topol.pitch()*10000;
    			xaxis = histos[Iditer->rawId()]->GetXaxis();
    			xaxis->SetTitle("tan(#theta_{t})");
    			yaxis = histos[Iditer->rawId()]->GetYaxis();
    			yaxis->SetTitle("Cluster size");
			}
			
			}
			
			if(!(conf_.getParameter<bool>("MTCCtrack"))){  //MTCCtrack FALSE
			
			if(subid.subdetId() == int (StripSubdetector::TOB)){
			TOBDetId TOBid=TOBDetId(*Iditer);
			if((TOBid.layer()!=1) && (TOBid.layer()!=2)){
			const GeomDetUnit * stripdet=(const GeomDetUnit*)tracker->idToDetUnit(subid);
			const StripTopology& topol=(StripTopology&)stripdet->topology();
			float thickness=stripdet->specificSurface().bounds().thickness();
			monosscounter++;
    			Detvector.push_back(*Iditer);
    			histos[Iditer->rawId()] = new TProfile(makename(*Iditer),makedescription(*Iditer),30,-0.6,0.6);
			detmap[Iditer->rawId()] = new detparameters;
			detmap[Iditer->rawId()]->thickness = thickness*10000;
                        detmap[Iditer->rawId()]->pitch = topol.pitch()*10000;
    			xaxis = histos[Iditer->rawId()]->GetXaxis();
    			xaxis->SetTitle("tan(#theta_{t})");
    			yaxis = histos[Iditer->rawId()]->GetYaxis();
    			yaxis->SetTitle("Cluster size");
			}
			}
			
			}
			
			if(subid.subdetId() == int (StripSubdetector::TID)){
			TIDDetId TIDid=TIDDetId(*Iditer);
			if((TIDid.ring()!=1) && (TIDid.ring()!=2)){
			const GeomDetUnit * stripdet=(const GeomDetUnit*)tracker->idToDetUnit(subid);
			const StripTopology& topol=(StripTopology&)stripdet->topology();
			float thickness=stripdet->specificSurface().bounds().thickness();
			monosscounter++;
    			Detvector.push_back(*Iditer);
    			histos[Iditer->rawId()] = new TProfile(makename(*Iditer),makedescription(*Iditer),30,-0.6,0.6);
			detmap[Iditer->rawId()] = new detparameters;
			detmap[Iditer->rawId()]->thickness = thickness*10000;
                        detmap[Iditer->rawId()]->pitch = topol.pitch()*10000;
    			xaxis = histos[Iditer->rawId()]->GetXaxis();
    			xaxis->SetTitle("tan(#theta_{t})");
    			yaxis = histos[Iditer->rawId()]->GetYaxis();
    			yaxis->SetTitle("Cluster size");
			}
			}
			
			if(subid.subdetId() == int (StripSubdetector::TEC)){
			TECDetId TECid=TECDetId(*Iditer);
			if((TECid.ring()!=1) && (TECid.ring()!=2) && (TECid.ring()!=5)){
			const GeomDetUnit * stripdet=(const GeomDetUnit*)tracker->idToDetUnit(subid);
			const StripTopology& topol=(StripTopology&)stripdet->topology();
			float thickness=stripdet->specificSurface().bounds().thickness();
			monosscounter++;
    			Detvector.push_back(*Iditer);
    			histos[Iditer->rawId()] = new TProfile(makename(*Iditer),makedescription(*Iditer),30,-0.6,0.6);
			detmap[Iditer->rawId()] = new detparameters;
			detmap[Iditer->rawId()]->thickness = thickness*10000;
                        detmap[Iditer->rawId()]->pitch = topol.pitch()*10000;
    			xaxis = histos[Iditer->rawId()]->GetXaxis();
    			xaxis->SetTitle("tan(#theta_{t})");
    			yaxis = histos[Iditer->rawId()]->GetYaxis();
    			yaxis->SetTitle("Cluster size");
			}
			}
		}
		
		//Mono double sided detectors
		
		if((subid.glued() != 0) && (subid.stereo() == 0)){
		
		        const GeomDetUnit * stripdet=(const GeomDetUnit*)tracker->idToDetUnit(subid);
			const StripTopology& topol=(StripTopology&)stripdet->topology();
			float thickness=stripdet->specificSurface().bounds().thickness();
		        monodscounter++;
    			Detvector.push_back(*Iditer);
    			histos[Iditer->rawId()] = new TProfile(makename(*Iditer),makedescription(*Iditer),30,-0.6,0.6);
			detmap[Iditer->rawId()] = new detparameters;
			detmap[Iditer->rawId()]->thickness = thickness*10000;
                        detmap[Iditer->rawId()]->pitch = topol.pitch()*10000;
    			xaxis = histos[Iditer->rawId()]->GetXaxis();
    			xaxis->SetTitle("tan(#theta_{t})");
    			yaxis = histos[Iditer->rawId()]->GetYaxis();
    			yaxis->SetTitle("Cluster size");
			}
		
		//Stereo detectors
				
		if((subid.glued() != 0) && (subid.stereo() == 1)){
		
		        const GeomDetUnit * stripdet=(const GeomDetUnit*)tracker->idToDetUnit(subid);
			const StripTopology& topol=(StripTopology&)stripdet->topology();
			float thickness=stripdet->specificSurface().bounds().thickness();
		        stereocounter++;
    			Detvector.push_back(*Iditer);
    			histos[Iditer->rawId()] = new TProfile(makename(*Iditer),makedescription(*Iditer),30,-0.6,0.6);
			detmap[Iditer->rawId()] = new detparameters;
			detmap[Iditer->rawId()]->thickness = thickness*10000;
                        detmap[Iditer->rawId()]->pitch = topol.pitch()*10000;
    			xaxis = histos[Iditer->rawId()]->GetXaxis();
    			xaxis->SetTitle("tan(#theta_{t})");
    			yaxis = histos[Iditer->rawId()]->GetYaxis();
    			yaxis->SetTitle("Cluster size");
			} 
	}	
  } 
      
  //Summary histograms
  
  histos[1] = new TProfile("TIBL2_widthvstan", "Cluster width vs tan(track local angle): TIB layer 2",TIBbinning,-0.6,0.6);
  detmap[1] = new detparameters;
  detmap[1]->thickness = 320; //[um]
  detmap[1]->pitch = 80; //[um]
  xaxis = histos[1]->GetXaxis();
  xaxis->SetTitle("tan(#theta_{t})");
  yaxis = histos[1]->GetYaxis();
  yaxis->SetTitle("Cluster size");
  yaxis->SetRangeUser(1.2,3.0);
  
  histos[2] = new TProfile("TIBL3_widthvstan", "Cluster width vs tan(track local angle): TIB layer 3",TIBbinning,-0.6,0.6);
  detmap[2] = new detparameters;
  detmap[2]->thickness = 320; //[um]
  detmap[2]->pitch = 120; //[um]
  xaxis = histos[2]->GetXaxis();
  xaxis->SetTitle("tan(#theta_{t})");
  yaxis = histos[2]->GetYaxis();
  yaxis->SetTitle("Cluster size");
  yaxis->SetRangeUser(1.2,3.0);  
    
  histos[3] = new TProfile("TOBL1_widthvstan", "Cluster width vs tan(track local angle): TOB layer 1",TOBbinning,-0.6,0.6);
  detmap[3] = new detparameters;
  detmap[3]->thickness = 500; //[um]
  detmap[3]->pitch = 183; //[um]
  xaxis = histos[3]->GetXaxis();
  xaxis->SetTitle("tan(#theta_{t})");
  yaxis = histos[3]->GetYaxis();
  yaxis->SetTitle("Cluster size");
  yaxis->SetRangeUser(1.2,3.0);
  
  histos[4] = new TProfile("TOBL1_widthvstan_TIBseed", "Cluster width vs tan(track local angle): TOB layer 1 TIBseed",TOBbinning,-0.6,0.6);
  detmap[4] = new detparameters;
  detmap[4]->thickness = 500; //[um]
  detmap[4]->pitch = 183; //[um]
  xaxis = histos[4]->GetXaxis();
  xaxis->SetTitle("tan(#theta_{t})");
  yaxis = histos[4]->GetYaxis();
  yaxis->SetTitle("Cluster size");
  yaxis->SetRangeUser(1.2,3.0);
  
  histos[5] = new TProfile("TOBL1_widthvstan_TOBseed", "Cluster width vs tan(track local angle): TOB layer 1 TOBseed",TOBbinning,-0.6,0.6);
  detmap[5] = new detparameters;
  detmap[5]->thickness = 500; //[um]
  detmap[5]->pitch = 183; //[um]
  xaxis = histos[5]->GetXaxis();
  xaxis->SetTitle("tan(#theta_{t})");
  yaxis = histos[5]->GetYaxis();
  yaxis->SetTitle("Cluster size");
  yaxis->SetRangeUser(1.2,3.0);
  
  histos[6] = new TProfile("TOBL5_widthvstan", "Cluster width vs tan(track local angle): TOB layer 5",TOBbinning,-0.6,0.6);
  detmap[6] = new detparameters;
  detmap[6]->thickness = 500; //[um]
  detmap[6]->pitch = 122; //[um]
  xaxis = histos[6]->GetXaxis();
  xaxis->SetTitle("tan(#theta_{t})");
  yaxis = histos[6]->GetYaxis();
  yaxis->SetTitle("Cluster size");
  yaxis->SetRangeUser(1.2,3.0);
  
  histos[7] = new TProfile("TOBL5_widthvstan_TIBseed", "Cluster width vs tan(track local angle): TOB layer 5 TIBseed",TOBbinning,-0.6,0.6);
  detmap[7] = new detparameters;
  detmap[7]->thickness = 500; //[um]
  detmap[7]->pitch = 122; //[um]
  xaxis = histos[7]->GetXaxis();
  xaxis->SetTitle("tan(#theta_{t})");
  yaxis = histos[7]->GetYaxis();
  yaxis->SetTitle("Cluster size");
  yaxis->SetRangeUser(1.2,3.0);
  
  histos[8] = new TProfile("TOBL5_widthvstan_TOBseed", "Cluster width vs tan(track local angle): TOB layer 5 TOBseed",TOBbinning,-0.6,0.6);
  detmap[8] = new detparameters;
  detmap[8]->thickness = 500; //[um]
  detmap[8]->pitch = 122; //[um]
  xaxis = histos[8]->GetXaxis();
  xaxis->SetTitle("tan(#theta_{t})");
  yaxis = histos[8]->GetYaxis();
  yaxis->SetTitle("Cluster size");
  yaxis->SetRangeUser(1.2,3.0);
  
  histos[9] = new TProfile("TOBL1_widthvstan_Rod1", "Cluster width vs tan(track local angle): TOB layer 1 Rod 1",TOBbinning,-0.6,0.6);
  detmap[9] = new detparameters;
  detmap[9]->thickness = 500; //[um]
  detmap[9]->pitch = 183; //[um]
  xaxis = histos[9]->GetXaxis();
  xaxis->SetTitle("tan(#theta_{t})");
  yaxis = histos[9]->GetYaxis();
  yaxis->SetTitle("Cluster size");
  yaxis->SetRangeUser(1.2,3.0);
  
  histos[10] = new TProfile("TOBL1_widthvstan_Rod2", "Cluster width vs tan(track local angle): TOB layer 1 Rod 2",TOBbinning,-0.6,0.6);
  detmap[10] = new detparameters;
  detmap[10]->thickness = 500; //[um]
  detmap[10]->pitch = 183; //[um]
  xaxis = histos[10]->GetXaxis();
  xaxis->SetTitle("tan(#theta_{t})");
  yaxis = histos[10]->GetYaxis();
  yaxis->SetTitle("Cluster size");
  yaxis->SetRangeUser(1.2,3.0);
  
  histos[11] = new TProfile("TOBL5_widthvstan_Rod1", "Cluster width vs tan(track local angle): TOB layer 5 Rod 1",TOBbinning,-0.6,0.6);
  detmap[11] = new detparameters;
  detmap[11]->thickness = 500; //[um]
  detmap[11]->pitch = 122; //[um]
  xaxis = histos[11]->GetXaxis();
  xaxis->SetTitle("tan(#theta_{t})");
  yaxis = histos[11]->GetYaxis();
  yaxis->SetTitle("Cluster size");
  yaxis->SetRangeUser(1.2,3.0);
  
  histos[12] = new TProfile("TOBL5_widthvstan_Rod2", "Cluster width vs tan(track local angle): TOB layer 5 Rod 2",TOBbinning,-0.6,0.6);
  detmap[12] = new detparameters;
  detmap[12]->thickness = 500; //[um]
  detmap[12]->pitch = 122; //[um]
  xaxis = histos[12]->GetXaxis();
  xaxis->SetTitle("tan(#theta_{t})");
  yaxis = histos[12]->GetYaxis();
  yaxis->SetTitle("Cluster size");
  yaxis->SetRangeUser(1.2,3.0);
      
  if(!(conf_.getParameter<bool>("MTCCtrack"))){  //MTCCtrack FALSE
  
  histos[13] = new TProfile("TIBL1_widthvstan", "Cluster width vs tan(track local angle): TIB layer 1",TIBbinning,-0.6,0.6);
  detmap[13] = new detparameters;
  detmap[13]->thickness = 320; //[um]
  detmap[13]->pitch = 120; //[um]
  xaxis = histos[13]->GetXaxis();
  xaxis->SetTitle("tan(#theta_{t})");
  yaxis = histos[13]->GetYaxis();
  yaxis->SetTitle("Cluster size");
  yaxis->SetRangeUser(1.4,3.0);
  
  histos[14] = new TProfile("TIBL4_widthvstan", "Cluster width vs tan(track local angle): TIB layer 4",TIBbinning,-0.6,0.6);
  detmap[14] = new detparameters;
  detmap[14]->thickness = 320; //[um]
  detmap[14]->pitch = 120; //[um]
  xaxis = histos[14]->GetXaxis();
  xaxis->SetTitle("tan(#theta_{t})");
  yaxis = histos[14]->GetYaxis();
  yaxis->SetTitle("Cluster size");
  yaxis->SetRangeUser(1.4,3.0);
  
  histos[15] = new TProfile("TOBL4_widthvstan", "Cluster width vs tan(track local angle): TOB layer 4",TOBbinning,-0.6,0.6);
  detmap[15] = new detparameters;
  detmap[15]->thickness = 500; //[um]
  detmap[15]->pitch = 183; //[um]
  xaxis = histos[15]->GetXaxis();
  xaxis->SetTitle("tan(#theta_{t})");
  yaxis = histos[15]->GetYaxis();
  yaxis->SetTitle("Cluster size");
  yaxis->SetRangeUser(1.4,3.0);
  
  histos[16] = new TProfile("TOBL2_widthvstan", "Cluster width vs tan(track local angle): TOB layer 2",TOBbinning,-0.6,0.6);
  detmap[16] = new detparameters;
  detmap[16]->thickness = 500; //[um]
  detmap[16]->pitch = 183; //[um]
  xaxis = histos[16]->GetXaxis();
  xaxis->SetTitle("tan(#theta_{t})");
  yaxis = histos[16]->GetYaxis();
  yaxis->SetTitle("Cluster size");
  yaxis->SetRangeUser(1.45,3.0);
  
  histos[17] = new TProfile("TOBL3_widthvstan", "Cluster width vs tan(track local angle): TOB layer 3",TOBbinning,-0.6,0.6);
  detmap[17] = new detparameters;
  detmap[17]->thickness = 500; //[um]
  detmap[17]->pitch = 183; //[um]
  xaxis = histos[17]->GetXaxis();
  xaxis->SetTitle("tan(#theta_{t})");
  yaxis = histos[17]->GetYaxis();
  yaxis->SetTitle("Cluster size");  
  yaxis->SetRangeUser(1.4,3.0);
  
  histos[18] = new TProfile("TOBL6_widthvstan", "Cluster width vs tan(track local angle): TOB layer 6",TOBbinning,-0.6,0.6);
  detmap[18] = new detparameters;
  detmap[18]->thickness = 500; //[um]
  detmap[18]->pitch = 122; //[um]
  xaxis = histos[18]->GetXaxis();
  xaxis->SetTitle("tan(#theta_{t})");
  yaxis = histos[18]->GetYaxis();
  yaxis->SetTitle("Cluster size");
  yaxis->SetRangeUser(1.4,3.0);
 
  }
    
  //Directory hierarchy  
  
  histograms = new TDirectory("Histograms", "Histograms", "");
  summary = new TDirectory("LorentzAngleSummary", "LorentzAngleSummary", "");
  sizesummary = new TDirectory("ClSizeSummary", "ClSizeSummary", "");  
  
  //TIB-TID-TOB-TEC    
  
  TIB = histograms->mkdir("TIB");
  TOB = histograms->mkdir("TOB");
  TID = histograms->mkdir("TID");
  TEC = histograms->mkdir("TEC");
  
  //Forward-Backward
  
  TIBfw = TIB->mkdir("TIB forward");
  TIDfw = TID->mkdir("TID forward");
  TOBfw = TOB->mkdir("TOB forward");
  TECfw = TEC->mkdir("TEC forward");
  
  TIBbw = TIB->mkdir("TIB backward");
  TIDbw = TID->mkdir("TID backward");
  TOBbw = TOB->mkdir("TOB backward");
  TECbw = TEC->mkdir("TEC backward"); 
  
  //TIB directories
  
  TIBfw1 = TIBfw->mkdir("TIB forward layer 1");
  TIBfw2 = TIBfw->mkdir("TIB forward layer 2");
  TIBfw3 = TIBfw->mkdir("TIB forward layer 3");
  TIBfw4 = TIBfw->mkdir("TIB forward layer 4");
  
  TIBbw1 = TIBbw->mkdir("TIB backward layer 1");
  TIBbw2 = TIBbw->mkdir("TIB backward layer 2");
  TIBbw3 = TIBbw->mkdir("TIB backward layer 3");
  TIBbw4 = TIBbw->mkdir("TIB backward layer 4");
  
  //TID directories
  
  TIDfw1 = TIDfw->mkdir("TID forward wheel 1");
  TIDfw2 = TIDfw->mkdir("TID forward wheel 2");
  TIDfw3 = TIDfw->mkdir("TID forward wheel 3");
  
  TIDbw1 = TIDbw->mkdir("TID backward wheel 1");
  TIDbw2 = TIDbw->mkdir("TID backward wheel 2");
  TIDbw3 = TIDbw->mkdir("TID backward wheel 3"); 
  
  //TOB directories
  
  TOBfw1 = TOBfw->mkdir("TOB forward layer 1");
  TOBfw2 = TOBfw->mkdir("TOB forward layer 2");
  TOBfw3 = TOBfw->mkdir("TOB forward layer 3");
  TOBfw4 = TOBfw->mkdir("TOB forward layer 4");
  TOBfw5 = TOBfw->mkdir("TOB forward layer 5");
  TOBfw6 = TOBfw->mkdir("TOB forward layer 6");
  
  TOBbw1 = TOBbw->mkdir("TOB backward layer 1");
  TOBbw2 = TOBbw->mkdir("TOB backward layer 2");
  TOBbw3 = TOBbw->mkdir("TOB backward layer 3");
  TOBbw4 = TOBbw->mkdir("TOB backward layer 4");
  TOBbw5 = TOBbw->mkdir("TOB backward layer 5");
  TOBbw6 = TOBbw->mkdir("TOB backward layer 6");
  
  //TEC directories
  
  TECfw1 = TECfw->mkdir("TEC forward wheel 1");
  TECfw2 = TECfw->mkdir("TEC forward wheel 2");
  TECfw3 = TECfw->mkdir("TEC forward wheel 3");
  TECfw4 = TECfw->mkdir("TEC forward wheel 4");
  TECfw5 = TECfw->mkdir("TEC forward wheel 5");
  TECfw6 = TECfw->mkdir("TEC forward wheel 6");
  TECfw7 = TECfw->mkdir("TEC forward wheel 7");
  TECfw8 = TECfw->mkdir("TEC forward wheel 8");
  TECfw9 = TECfw->mkdir("TEC forward wheel 9");
  
  TECbw1 = TECbw->mkdir("TEC backward layer 1");
  TECbw2 = TECbw->mkdir("TEC backward layer 2");
  TECbw3 = TECbw->mkdir("TEC backward layer 3");
  TECbw4 = TECbw->mkdir("TEC backward layer 4");
  TECbw5 = TECbw->mkdir("TEC backward layer 5");
  TECbw6 = TECbw->mkdir("TEC backward layer 6");
  TECbw7 = TECbw->mkdir("TEC backward layer 7");
  TECbw8 = TECbw->mkdir("TEC backward layer 8");
  TECbw9 = TECbw->mkdir("TEC backward layer 9");
  
  eventcounter = 0;
  trackcounter = 0;
  hitcounter = 0;
  run = 0;
  runcounter = 0;
  
  int m;
  for(m=0;m!=1000;m++) runvector[m]=0;
  
} 

// Virtual destructor needed.

SiStripLorentzAngle::~SiStripLorentzAngle() {  }  

// Analyzer: Functions that gets called by framework every event

void SiStripLorentzAngle::analyze(const edm::Event& e, const edm::EventSetup& es)
{
  
  if(e.id().run() != run){
  runvector[runcounter]=e.id().run();
  runcounter++;
  }
  
  run       = e.id().run();
  event     = e.id().event();
       
  cout<<"Run number = "<<run<<endl;
  cout<<"Event number = "<<event<<endl;
      
  eventcounter++;
   
  using namespace edm;
  
  // Analysis of RecHits Collections
  
  std::string rechitProducer = conf_.getParameter<std::string>("RecHitProducer");
  
  edm::Handle<SiStripMatchedRecHit2DCollection> rechitsmatched;
  edm::Handle<SiStripRecHit2DCollection> rechitsrphi;
  edm::Handle<SiStripRecHit2DCollection> rechitsstereo;
  e.getByLabel(rechitProducer,"matchedRecHit", rechitsmatched);
  e.getByLabel(rechitProducer,"rphiRecHit", rechitsrphi);
  e.getByLabel(rechitProducer,"stereoRecHit", rechitsstereo);
       
  MONOhitsTIBL2collection = 0;
  STEREOhitsTIBL2collection = 0;
  hitsTIBL3collection = 0;
  hitsTOBL1collection = 0;
  hitsTOBL5collection = 0;
  hitsTOBcoll = 0;
  MONOhitsTECcollection = 0;
  STEREOhitsTECcollection = 0;
    
  int n;
  for(n=0;n<1000;n++){
  
  MONOhitschargeTIBL2[n]=-99;
  STEREOhitschargeTIBL2[n]=-99;
  hitschargeTIBL3[n]=-99;
  hitschargeTOBL1[n]=-99;
  hitschargeTOBL5[n]=-99;
  MONOhitschargeTEC[n]=-99;  
  STEREOhitschargeTEC[n]=-99;
 
  TOB_YtoGlobalSignColl[n]=-99;
  } 
    
  std::vector<DetId>::iterator Iditer;
  
  for(Iditer=Detvector.begin(); Iditer!=Detvector.end(); Iditer++){
  
  SiStripRecHit2DCollection::range          rechitrphiRange = rechitsrphi->get(*Iditer);
  SiStripRecHit2DCollection::const_iterator rechitrphiRangeIteratorBegin = rechitrphiRange.first;
  SiStripRecHit2DCollection::const_iterator rechitrphiRangeIteratorEnd   = rechitrphiRange.second; 
  SiStripRecHit2DCollection::const_iterator iterrphi;
  
    for(iterrphi=rechitrphiRangeIteratorBegin;iterrphi!=rechitrphiRangeIteratorEnd;iterrphi++){
    
    StripSubdetector detid=(StripSubdetector)iterrphi->geographicalId();
    
    std::vector<uint16_t> amplitudes = (iterrphi->cluster())->amplitudes();
    HitSize = amplitudes.size();
    std::vector<uint16_t>::iterator stripiter;
    hitcharge = 0;
    for(stripiter=amplitudes.begin();stripiter!=amplitudes.end();stripiter++){
    hitcharge+= *stripiter;}
    
    if(detid.subdetId() == int (StripSubdetector::TIB)){
    TIBDetId TIBid=TIBDetId(iterrphi->geographicalId());
    int detlayer = TIBid.layer() + mtcctibcorr;
        
    if(detlayer==2){
    MONOhitschargeTIBL2[MONOhitsTIBL2collection]=hitcharge;
    CollHitSizeTIBL2mono->Fill(HitSize);
    MONOhitsTIBL2collection++;}
    
    if(detlayer==3){
    hitschargeTIBL3[hitsTIBL3collection]=hitcharge;
    CollHitSizeTIBL3->Fill(HitSize);
    hitsTIBL3collection++;}
    
    }
    
    if(detid.subdetId() == int (StripSubdetector::TOB)){
    TOBDetId TOBid=TOBDetId(iterrphi->geographicalId());
    int detlayer = TOBid.layer();
    if(detlayer==2){
    detlayer+=mtcctobcorr;}
    
    const GeomDet *moduledet = tracker->idToDet(iterrphi->geographicalId());    
    LocalVector localY(0,1,0);
    GlobalVector globalZ(0,0,1);
    LocalVector GtoLocalZ = (moduledet->surface()).toLocal(globalZ);
    float sign=localY*GtoLocalZ;
    
    if(sign>0){
    TOB_YtoGlobalSignColl[hitsTOBcoll]=1;}
    if(sign<0){
    TOB_YtoGlobalSignColl[hitsTOBcoll]=-1;}
    
    if(detlayer==1){
    hitschargeTOBL1[hitsTOBL1collection]=hitcharge;
    CollHitSizeTOBL1->Fill(HitSize);
    hitsTOBL1collection++;}
    
    if(detlayer==5){
    hitschargeTOBL5[hitsTOBL5collection]=hitcharge;
    CollHitSizeTOBL5->Fill(HitSize);
    hitsTOBL5collection++;}
    
    hitsTOBcoll++;
    }
    
    if(detid.subdetId() == int (StripSubdetector::TEC)){
    TECDetId TECid=TECDetId(iterrphi->geographicalId());
    int detwheel = TECid.wheel();
    
    if(detwheel==1){
    MONOhitschargeTEC[MONOhitsTECcollection]=hitcharge;
    MONOhitsTECcollection++;}
    }
    
    }   
     
  SiStripRecHit2DCollection::range          rechitstereoRange = rechitsstereo->get(*Iditer);
  SiStripRecHit2DCollection::const_iterator rechitstereoRangeIteratorBegin = rechitstereoRange.first;
  SiStripRecHit2DCollection::const_iterator rechitstereoRangeIteratorEnd   = rechitstereoRange.second; 
  SiStripRecHit2DCollection::const_iterator iterstereo;
        
    for(iterstereo=rechitstereoRangeIteratorBegin;iterstereo!=rechitstereoRangeIteratorEnd;iterstereo++){
    
    StripSubdetector detid=(StripSubdetector)iterstereo->geographicalId();
    
    std::vector<uint16_t> amplitudes = (iterstereo->cluster())->amplitudes();
    HitSize = amplitudes.size();
    std::vector<uint16_t>::iterator stripiter;
    hitcharge = 0;
    for(stripiter=amplitudes.begin();stripiter!=amplitudes.end();stripiter++){
    hitcharge+= *stripiter;}
    
    if(detid.subdetId() == int (StripSubdetector::TIB)){
    TIBDetId TIBid=TIBDetId(iterstereo->geographicalId());
    int detlayer = TIBid.layer() + mtcctibcorr;
        
    if(detlayer==2){
    STEREOhitschargeTIBL2[STEREOhitsTIBL2collection]=hitcharge;
    CollHitSizeTIBL2stereo->Fill(HitSize);
    STEREOhitsTIBL2collection++;}
    
    }
    
    if(detid.subdetId() == int (StripSubdetector::TEC)){
    TECDetId TECid=TECDetId(iterstereo->geographicalId());
    int detwheel = TECid.wheel();
    
    if(detwheel==1){
    STEREOhitschargeTEC[STEREOhitsTECcollection]=hitcharge;
    STEREOhitsTECcollection++;}
    }
    
    }    
    }
    
    //Analysis of Trajectory-RecHits
        
    edm::InputTag TkTag = conf_.getParameter<edm::InputTag>("cosmicTracks");
  
    edm::Handle<reco::TrackCollection> trackCollection;
    e.getByLabel(TkTag,trackCollection);

    edm::Handle<TrackingRecHitCollection> trackerchitCollection;
    e.getByLabel(TkTag,trackerchitCollection);
  
    edm::Handle<std::vector<Trajectory> > TrajectoryCollection;
    e.getByLabel(TkTag,TrajectoryCollection);
   
    const reco::TrackCollection *tracks=trackCollection.product();
 
    std::vector<std::pair<const TrackingRecHit*,float> >hitangleassociation;
    Trajectory * theTraj; 
    
    trackcollsize = 0;
    trajsize = 0;
    ThetaTrack = -9999;
    PhiTrack = -9999;
    
    trackcollsize=tracks->size();
    trajsize=TrajectoryCollection->size();
        
    LogDebug("SiStripLorentzAngle::analyze") <<" Number of tracks in event = "<<trackcollsize<<"\n";
    LogDebug("SiStripLorentzAngle::analyze") <<" Number of trajectories in event = "<<trajsize<<"\n";
    
  if (trajsize != 0){

    theTraj = new Trajectory(TrajectoryCollection->front());
    
    LogDebug("SiStripLorentzAngle::analyze") <<"trajectory done";
    
    //Seed
    
    SeedSize=0;
    TrajectorySeed trajSeed = theTraj->seed();
    SeedSize = trajSeed.nHits();
    LogDebug("SiStripLorentzAngle::analyze") <<"Seed Size = "<<SeedSize;

    typedef edm::OwnVector<TrackingRecHit>::const_iterator SeedIter;
    std::pair<SeedIter, SeedIter> IterPair = trajSeed.recHits();
    
    SeedIter SeedBegin = IterPair.first;
    StripSubdetector detid=(StripSubdetector)SeedBegin->geographicalId();
    SeedType = detid.subdetId();
    if(detid.subdetId() == int (StripSubdetector::TIB)){
    LogDebug("SiStripLorentzAngle::analyze") <<"Seed in TIB";
    TIBDetId TIBid=TIBDetId(SeedBegin->geographicalId());
    SeedLayer = TIBid.layer() + mtcctibcorr;}
    if(detid.subdetId() == int (StripSubdetector::TOB)){
    LogDebug("SiStripLorentzAngle::analyze") <<"Seed in TOB";
    TOBDetId TOBid=TOBDetId(SeedBegin->geographicalId());
    SeedLayer = TOBid.layer();
    if(SeedLayer==2){
    SeedLayer+=mtcctobcorr;}
    }

    std::vector<TrajectoryMeasurement> TMeas=theTraj->measurements();
    std::vector<TrajectoryMeasurement>::iterator itm;
    
    LogDebug("SiStripLorentzAngle::analyze")<<"Loop on rechit and TSOS";
    for (itm=TMeas.begin();itm!=TMeas.end();itm++){
      TrajectoryStateOnSurface tsos=itm->updatedState();
      const TransientTrackingRecHit::ConstRecHitPointer thit=itm->recHit();
      const SiStripMatchedRecHit2D* matchedhit=dynamic_cast<const SiStripMatchedRecHit2D*>((*thit).hit());
      const SiStripRecHit2D* hit=dynamic_cast<const SiStripRecHit2D*>((*thit).hit());
      LocalVector trackdirection=tsos.localDirection();
	
      if(matchedhit){//if matched hit...
      
      GluedGeomDet * gdet=(GluedGeomDet *)tracker->idToDet(matchedhit->geographicalId());
	
      GlobalVector gtrkdir=gdet->toGlobal(trackdirection);	
	
	LogDebug("SiStripLorentzAngle::analyze") <<"Matched hits used";
	
	//cluster and trackdirection on mono det
	
	// THIS THE POINTER TO THE MONO HIT OF A MATCHED HIT 
	const SiStripRecHit2D *monohit=matchedhit->monoHit();
	    
	const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > monocluster=monohit->cluster();
	const GeomDetUnit * monodet=gdet->monoDet();
	
	LocalVector monotkdir=monodet->toLocal(gtrkdir);
	//size=(monocluster->amplitudes()).size();
	if(monotkdir.z()!=0){
	  
	  // THE LOCAL ANGLE (MONO)
	  float angle = atan(monotkdir.x()/monotkdir.z())*180/TMath::Pi();
	  
	  hitangleassociation.push_back(make_pair(monohit, angle)); 
	      
	    //cluster and trackdirection on stereo det
	    
	    // THIS THE POINTER TO THE STEREO HIT OF A MATCHED HIT 
	  const SiStripRecHit2D *stereohit=matchedhit->stereoHit();
	  const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > stereocluster=stereohit->cluster();
	  const GeomDetUnit * stereodet=gdet->stereoDet(); 
	  LocalVector stereotkdir=stereodet->toLocal(gtrkdir);
	  
	  if(stereotkdir.z()!=0){
	    
	    // THE LOCAL ANGLE (STEREO)
		  float angle = atan(stereotkdir.x()/stereotkdir.z())*180/TMath::Pi();
		  		  
		  hitangleassociation.push_back(make_pair(stereohit, angle)); 		  
	  }
	}
      }
      else if(hit){
	//  hit= POINTER TO THE RECHIT
	const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > cluster=hit->cluster();
	
	if(trackdirection.z()!=0){
	  
	    // THE LOCAL ANGLE (STEREO)
	  float angle = atan(trackdirection.x()/trackdirection.z())*180/TMath::Pi();
	  	  
	  hitangleassociation.push_back(make_pair(hit, angle)); 
	}
      }
    }
  }
         
    bool noise_rechit = false; 
	 
    std::vector<std::pair<const TrackingRecHit *,float> >::iterator hitangleiter;
    
    if(hitangleassociation.size()!=0){
    
     LogDebug("TrackLocalAngle")<<"Hitangleassocitaion size = "<<hitangleassociation.size();
                
    for(hitangleiter=hitangleassociation.begin();hitangleiter!=hitangleassociation.end();hitangleiter++){ 
      const SiStripRecHit2D* hit=dynamic_cast<const SiStripRecHit2D*>(hitangleiter->first);
      const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > cluster=hit->cluster();
      std::vector<uint16_t> amplitudes = cluster->amplitudes();
      std::vector<uint16_t>::iterator stripiter;
      int hit_charge = 0; 
      for(stripiter=amplitudes.begin();stripiter!=amplitudes.end();stripiter++){
      hit_charge+= *stripiter;}
      StripSubdetector detid=(StripSubdetector)hit->geographicalId();
      if((detid.subdetId() == int (StripSubdetector::TOB)) && (hit_charge<80)){
      noise_rechit = true;}
      }
      } 
      	 
    trackhitmap trackhits; 
    
    if(noise_rechit==false){      
    reco::TrackCollection::const_iterator ibeg=tracks->begin();
    
    PhiTrack = (*ibeg).outerPhi()*180/TMath::Pi();
    ThetaTrack = (*ibeg).outerTheta()*180/TMath::Pi();
        
    trackhits[&(*ibeg)] = hitangleassociation;
    }          
    
  if(trackhits.size()!=0){
           
    trackhitmap::iterator mapiter;
    
    hitspertrack=0;
    
    for(mapiter = trackhits.begin(); mapiter != trackhits.end(); mapiter++){
    
    trackcounter++;
           
      hitspertrack=0;
      TECwheel1=0;
      TIBlayer2=0;
      TIBlayer3=0;
      TOBlayer1=0;
      TOBlayer5=0;
      momentum=-99;
      pt=-99;
      ParticleCharge=-99;
      chi2=-99;
      ndof=-99;
      chi2norm=-99;
                 
      momentum = (*mapiter).first->p();
      pt = (*mapiter).first->pt();
      ParticleCharge = (*mapiter).first->charge();
      chi2 = (*mapiter).first->chi2();
      ndof = (*mapiter).first->ndof();
      chi2norm = chi2/ndof;
                      
      std::vector<std::pair<const TrackingRecHit *,float> > hitangle = (*mapiter).second;
      std::vector<std::pair<const TrackingRecHit *,float> >::iterator hitsiter;
    
    if(hitangle.size()!=0){
                
    for(hitsiter=hitangle.begin();hitsiter!=hitangle.end();hitsiter++){
    
    hitcounter++;
    
      module=-99;
      type=-99;
      layer=-99;
      wheel=-99;
      string=-99;
      size=-99;
      extint=-99;
      bwfw=-99;
      rod=-99;
      TrackLocalAngle=-9999;
      tangent=-9999;
      trackproj=-9999;
      signprojcorrection=-9999;
      localmagfield=-99;
      monostereo=-99;
      hitscharge=0;
      TOB_YtoGlobalSign=0;
     
      const SiStripRecHit2D* hit=dynamic_cast<const SiStripRecHit2D*>(hitsiter->first);
      const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > cluster=hit->cluster();
      
      std::vector<uint16_t> amplitudes = cluster->amplitudes();
      std::vector<uint16_t>::iterator stripiter;
      for(stripiter=amplitudes.begin();stripiter!=amplitudes.end();stripiter++){
      hitscharge+= *stripiter;}
      
      size=(cluster->amplitudes()).size();
      
      StripSubdetector detid=(StripSubdetector)hit->geographicalId();
      
      type = detid.subdetId();
      
      module = (hit->geographicalId()).rawId();
            
      TrackLocalAngle=hitsiter->second;
      
      tangent = tan((TrackLocalAngle*TMath::Pi())/180);
      
      monostereo=detid.stereo();
      
      //Sign and XZ plane projection correction applied in TrackLocalAngle (TIB|TOB layers)
      
      const GeomDet *geomdet = tracker->idToDet(hit->geographicalId());
      LocalPoint localp(0,0,0);
      const GlobalPoint globalp = (geomdet->surface()).toGlobal(localp);
      GlobalVector globalmagdir = magfield->inTesla(globalp);
      localmagdir = (geomdet->surface()).toLocal(globalmagdir);
      localmagfield = localmagdir.mag();
      
      if(localmagfield != 0.){
     
      if((detid.subdetId() == int (StripSubdetector::TIB)) || (detid.subdetId() == int (StripSubdetector::TOB))){
      
      LocalVector ylocal(0,1,0);
      
      float normprojection = (localmagdir * ylocal)/(localmagfield);
                        
      if(normprojection == 0.){
      LogDebug("SiStripLorentzAngle::analyze")<<"Error: TIB|TOB YBprojection = 0";
      }
      
      if(normprojection != 0.){
      
      signprojcorrection = 1/normprojection;
        
      tangent*=signprojcorrection;
      
      TrackLocalAngle = atan(tangent)*180/TMath::Pi();
           
      }
      }
      }
            
      float thickness = detmap[module]->thickness;
      float pitch = detmap[module]->pitch;
	
      trackproj=(tangent*thickness)/pitch;
      	
      //Filling histograms
       
      histos[module]->Fill(tangent,size);
                          
      //Summary histograms
	
	if(detid.subdetId() == int (StripSubdetector::TIB)){
		TIBDetId TIBid=TIBDetId(hit->geographicalId());
		
		extint = TIBid.string()[1];
		string = TIBid.string()[2];
		bwfw= TIBid.string()[0];
		layer = TIBid.layer() + mtcctibcorr;
		
		if(layer == 1){
		histos[13]->Fill(tangent,size);
		}
		
		if(layer == 2){
		histos[1]->Fill(tangent,size);
		
		if(detid.stereo()==1){
		TrackHitSizeTIBL2mono->Fill(size);}
		else{
		TrackHitSizeTIBL2stereo->Fill(size);}
		
		TIBlayer2++;}
		
		if(layer == 3){
		histos[2]->Fill(tangent,size);
		TrackHitSizeTIBL3->Fill(size);
		TIBlayer3++;}
		
		if(layer == 4){
		histos[14]->Fill(tangent,size);
		}
				
		}
		
	if(detid.subdetId() == int (StripSubdetector::TOB)){
		TOBDetId TOBid=TOBDetId(hit->geographicalId());
		
		layer = TOBid.layer();
		if(layer==2){
		layer+=mtcctobcorr;}
		
		rod = TOBid.rod()[1];
		bwfw = TOBid.rod()[0];
		
		//Orientation of local Y axis with respect to the global Z axis
		
		LocalVector localY(0,1,0);
		GlobalVector globalZ(0,0,1);
		LocalVector GtoLocalZ = (geomdet->surface()).toLocal(globalZ);
		float sign=localY*GtoLocalZ;
		if(sign>0){TOB_YtoGlobalSign=1;}
		if(sign<0){TOB_YtoGlobalSign=-1;}
		
		if(layer == 1){
		histos[3]->Fill(tangent,size);
		TOBlayer1++;
		TrackHitSizeTOBL1->Fill(size);
		
		if(rod==1){
		histos[9]->Fill(tangent,size);
		}
		if(rod==2){
		histos[10]->Fill(tangent,size);
		}
		
		if(SeedType==3){
		histos[4]->Fill(tangent,size);
		}
		if(SeedType==5){
		histos[5]->Fill(tangent,size);
		}		
		}
		
		if(layer == 2){
		histos[16]->Fill(tangent,size);		
		}
		
		if(layer == 3){
		histos[17]->Fill(tangent,size);
		}
		
		if(layer == 4){
		histos[15]->Fill(tangent,size);
		}
		
		if(layer == 5){
		histos[6]->Fill(tangent,size);
		TOBlayer5++;
		TrackHitSizeTOBL5->Fill(size);
		
		if(rod==1){
		histos[11]->Fill(tangent,size);
		}
		if(rod==2){
		histos[12]->Fill(tangent,size);
		}
		
		if(SeedType==3){
		histos[7]->Fill(tangent,size);
		}
		if(SeedType==5){
		histos[8]->Fill(tangent,size);
		}
		}
		
		if(layer == 6){
		histos[18]->Fill(tangent,size);
		}		
				
		}
		
	if(detid.subdetId() == int (StripSubdetector::TID)){
		TIDDetId TIDid=TIDDetId(hit->geographicalId());
		bwfw = TIDid.module()[0];
		wheel = TIDid.wheel();
		}
		
	if(detid.subdetId() == int (StripSubdetector::TEC)){
		TECDetId TECid=TECDetId(hit->geographicalId());
		bwfw = TECid.petal()[0];
		wheel = TECid.wheel();
		TECwheel1++;
		}
      	
	//Filling Tree
	
        SiStripLorentzAngleTree->Fill();
	
	LogDebug("SiStripLorentzAngle::analyze")<<"Tree Filled";
	
	hitspertrack++;
	
	}
      }
                           
      TrackHitTree->Fill();

    }
  }
}

//Makename function

const char* SiStripLorentzAngle::makename(DetId detid){
  
  std::string name;
  
  stringstream idnum;
  stringstream layernum;
  stringstream wheelnum;
  stringstream stringnum;
  stringstream rodnum;
  stringstream ringnum;
  stringstream petalnum;
  
  idnum << detid.rawId();
  
  StripSubdetector DetId(detid.rawId());
  
  //TIB
  
  if(detid.subdetId() == int (StripSubdetector::TIB)){
    name="TIB";
    
    TIBDetId TIBid=TIBDetId(DetId);
    
    if(TIBid.string()[0] == 0){
      name+="bw";}
    if(TIBid.string()[0] == 1){
      name+="fw";}
    
    name+="L";
    int layer = TIBid.layer() + mtcctibcorr;    
    layernum << layer;
    name+=layernum.str();
    
    if(TIBid.string()[1] == 0){
      name+="int";}
    if(TIBid.string()[1] == 1){
      name+="ext";}
    
    name+="string";
    int string = TIBid.string()[2];
    stringnum << string;
    name+=stringnum.str();
    
    if(TIBid.stereo() == 0){
      name+="mono";}
    if(TIBid.stereo() == 1){
      name+="stereo";}    
  }
  
  //TID
   
  if(detid.subdetId() == int (StripSubdetector::TID)){
    name="TID";
    
    TIDDetId TIDid=TIDDetId(DetId);
    
    if(TIDid.module()[0] == 0){
      name+="bw";}
    if(TIDid.module()[0] == 1){
      name+="fw";}
      
    name+="W";
    int wheel = TIDid.wheel();    
    wheelnum << wheel;
    name+=wheelnum.str();
    
    if(TIDid.side() == 1){
      name+="neg";}
    if(TIDid.side() == 2){
      name+="pos";}
      
    name+="ring";
    int ring = TIDid.ring();
    ringnum << ring;
    name+=ringnum.str();
    
    if(TIDid.stereo() == 0){
      name+="mono";}
    if(TIDid.stereo() == 1){
      name+="stereo";}    
  }
  
  //TOB
  
  if(detid.subdetId() == int (StripSubdetector::TOB)){
    name="TOB";
    
    TOBDetId TOBid=TOBDetId(DetId);
    
    if(TOBid.rod()[0] == 0){
      name+="bw";}
    if(TOBid.rod()[0] == 1){
      name+="fw";}
    
    name+="L";
    int layer = TOBid.layer();
    if(layer==2){
    layer+=mtcctobcorr;} 
    layernum << layer;
    name+=layernum.str();
 
    name+="rod";
    int rod = TOBid.rod()[1];
    rodnum << rod;
    name+=rodnum.str();
    
    if(TOBid.stereo() == 0){
      name+="mono";}
    if(TOBid.stereo() == 1){
      name+="stereo";}    
   }
   
  //TEC
  
  if(detid.subdetId() == int (StripSubdetector::TEC)){
    name="TEC";
    
    TECDetId TECid=TECDetId(DetId);
    
    if(TECid.petal()[0] == 0){
      name+="bw";}
    if(TECid.petal()[0] == 1){
      name+="fw";}
      
    name+="W";
    int wheel = TECid.wheel();    
    wheelnum << wheel;
    name+=wheelnum.str();
    
    if(TECid.side() == 1){
      name+="neg";}
    if(TECid.side() == 2){
      name+="pos";}
      
    name+="ring";
    int ring = TECid.ring();
    ringnum << ring;
    name+=ringnum.str();
    
    name+="petal";
    int petal = TECid.petal()[1];
    petalnum << petal;
    name+=petalnum.str();
    
    if(TECid.stereo() == 0){
      name+="mono";}
    if(TECid.stereo() == 1){
      name+="stereo";}    
  }
    
  name+="_";
  
  name+=idnum.str();
  
  return name.c_str();
  
}

//Makedescription function

const char* SiStripLorentzAngle::makedescription(DetId detid){
  
  std::string name;
  
  name="Cluster width vs tan(track local angle) (";
  
  stringstream idnum;
  stringstream layernum;
  stringstream wheelnum;
  stringstream stringnum;
  stringstream rodnum;
  stringstream ringnum;
  stringstream petalnum;
  
  idnum << detid.rawId();
  
  StripSubdetector DetId(detid.rawId());
  
  //TIB
  
  if(detid.subdetId() == int (StripSubdetector::TIB)){
    name+="TIB ";
    
    TIBDetId TIBid=TIBDetId(DetId);
    
    if(TIBid.string()[0] == 0){
      name+="backward, ";}
    if(TIBid.string()[0] == 1){
      name+="forward, ";}
    
    name+="Layer n.";
    int layer = TIBid.layer() + mtcctibcorr;  
    layernum << layer;
    name+=layernum.str();
    
    if(TIBid.string()[1] == 0){
      name+=", internal ";}
    if(TIBid.string()[1] == 1){
      name+=", external ";}
    
    name+="string n.";
    int string = TIBid.string()[2];
    stringnum << string;
    name+=stringnum.str();
    
    if(TIBid.stereo() == 0){
      name+=", mono,";}
    if(TIBid.stereo() == 1){
      name+=", stereo,";} 
         
  }
  
  //TID
  
  if(detid.subdetId() == int (StripSubdetector::TID)){
    name+="TID ";
    
    TIDDetId TIDid=TIDDetId(DetId);
    
    if(TIDid.module()[0] == 0){
      name+="bacward, ";}
    if(TIDid.module()[0] == 1){
      name+="forward, ";}
      
    name+="Wheel n.";
    int wheel = TIDid.wheel();    
    wheelnum << wheel;
    name+=wheelnum.str();
    
    if(TIDid.side() == 1){
      name+=", negative ";}
    if(TIDid.side() == 2){
      name+=", positive ";}
      
    name+="ring n.";
    int ring = TIDid.ring();
    ringnum << ring;
    name+=ringnum.str();
    
    if(TIDid.stereo() == 0){
      name+=", mono";}
    if(TIDid.stereo() == 1){
      name+=", stereo";}    
      
  }
    
  //TOB  
  
  if(detid.subdetId() == int (StripSubdetector::TOB)){
    name+="TOB ";
    
    TOBDetId TOBid=TOBDetId(DetId);
    
    if(TOBid.rod()[0] == 0){
      name+="backward, ";}
    if(TOBid.rod()[0] == 1){
      name+="forward, ";}
    
    name+="Layer n.";
    int layer = TOBid.layer();
    if(layer==2){
    layer+=mtcctobcorr;} 
    layernum << layer;
    name+=layernum.str();
    
    name+=", rod n.";
    int rod = TOBid.rod()[1];
    rodnum << rod;
    name+=rodnum.str();
    
    if(TOBid.stereo() == 0){
      name+=", mono,";}
    if(TOBid.stereo() == 1){
      name+=", stereo,";}    
  }
  
  //TEC
  
  if(detid.subdetId() == int (StripSubdetector::TEC)){
    name+="TEC ";
    
    TECDetId TECid=TECDetId(DetId);
    
    if(TECid.petal()[0] == 0){
      name+="backward, ";}
    if(TECid.petal()[0] == 1){
      name+="forward, ";}
      
    name+="Wheel n.";
    int wheel = TECid.wheel();    
    wheelnum << wheel;
    name+=wheelnum.str();
    
    if(TECid.side() == 1){
      name+=", negative ";}
    if(TECid.side() == 2){
      name+=", positive ";}
      
    name+="ring n.";
    int ring = TECid.ring();
    ringnum << ring;
    name+=ringnum.str();
    
    name+=", petal n.";
    int petal = TECid.petal()[1];
    petalnum << petal;
    name+=petalnum.str();
    
    if(TECid.stereo() == 0){
      name+=", mono";}
    if(TECid.stereo() == 1){
      name+=", stereo";}  
        
  }
      
  name+=" IdNumber = ";
  
  name+=idnum.str();
  
  name+=")";
  
  return name.c_str();
  
}

//EndJob

void SiStripLorentzAngle::endJob(){

  std::vector<DetId>::iterator Iditer;
  
  //Histograms fit
  
  int histonum = Detvector.size();
  
  double ModuleRangeMin=conf_.getParameter<double>("ModuleRangeMin");
  double ModuleRangeMax=conf_.getParameter<double>("ModuleRangeMax");
  double TIBRangeMin=conf_.getParameter<double>("TIBRangeMin");
  double TIBRangeMax=conf_.getParameter<double>("TIBRangeMax");
  double TOBRangeMin=conf_.getParameter<double>("TOBRangeMin");
  double TOBRangeMax=conf_.getParameter<double>("TOBRangeMax");
  

  for(Iditer=Detvector.begin(); Iditer!=Detvector.end(); Iditer++){
  
    float thickness = detmap[Iditer->rawId()]->thickness;
    float pitch = detmap[Iditer->rawId()]->pitch;
    
    fitfunc = new TF1("fitfunc","([4]/[3])*[1]*(TMath::Abs(x-[0]))+[2]",-1,1);
    
    fitfunc->SetParameter(0, 0);
    fitfunc->SetParameter(1, 0);
    fitfunc->SetParameter(2, 1);
    fitfunc->FixParameter(3, pitch);
    fitfunc->FixParameter(4, thickness);
    
    histos[Iditer->rawId()]->Fit("fitfunc","E","",ModuleRangeMin, ModuleRangeMax);
    
    TF1 *fitfunction = histos[Iditer->rawId()]->GetFunction("fitfunc");
    
    fits[Iditer->rawId()] = new histofit;
    
    fits[Iditer->rawId()]->chi2 = fitfunction->GetChisquare();
    fits[Iditer->rawId()]->ndf  = fitfunction->GetNDF();
    fits[Iditer->rawId()]->p0   = fitfunction->GetParameter(0);
    fits[Iditer->rawId()]->p1   = fitfunction->GetParameter(1);
    fits[Iditer->rawId()]->p2   = fitfunction->GetParameter(2);
    fits[Iditer->rawId()]->errp0   = fitfunction->GetParError(0);
    fits[Iditer->rawId()]->errp1   = fitfunction->GetParError(1);
    fits[Iditer->rawId()]->errp2   = fitfunction->GetParError(2);
        
  }
  
  int n;
  int nmax = 19;
  
  if(conf_.getParameter<bool>("MTCCtrack")){  //MTCCtrack TRUE
  nmax=13;}
  
  for(n=1; n<nmax; n++){
  
    float thickness = detmap[n]->thickness;
    float pitch = detmap[n]->pitch;
    
    fitfunc = new TF1("fitfunc","([4]/[3])*[1]*(TMath::Abs(x-[0]))+[2]",-1,1);
        
    fitfunc->SetParameter(0, 0);
    fitfunc->SetParameter(1, 0);
    fitfunc->SetParameter(2, 1);
    fitfunc->FixParameter(3, pitch);
    fitfunc->FixParameter(4, thickness);
    
    if(n<3){
    histos[n]->Fit("fitfunc","E","",TIBRangeMin, TIBRangeMax);
    }else{
    histos[n]->Fit("fitfunc","E","",TOBRangeMin, TOBRangeMax);
    }
    
    TF1 *fitfunction = histos[n]->GetFunction("fitfunc");
    
    fits[n] = new histofit;
    
    fits[n]->chi2 = fitfunction->GetChisquare();
    fits[n]->ndf  = fitfunction->GetNDF();
    fits[n]->p0   = fitfunction->GetParameter(0);
    fits[n]->p1   = fitfunction->GetParameter(1);
    fits[n]->p2   = fitfunction->GetParameter(2);
    fits[n]->errp0   = fitfunction->GetParError(0);
    fits[n]->errp1   = fitfunction->GetParError(1);
    fits[n]->errp2   = fitfunction->GetParError(2);
        
  }
  
  //File with fit parameters  
  
  std::string fitName=conf_.getParameter<std::string>("fitName");
  fitName+=".txt";
  
  ofstream fit;
  fit.open(fitName.c_str());
  
  if(conf_.getParameter<bool>("MTCCtrack")){  //MTCCtrack TRUE
  fit<<endl<<">>> MTCCtrack = TRUE"<<endl<<endl;
  fit<<">>> MAGNETIC FIELD = "<<localmagfield<<endl;
  }else{
  fit<<endl<<">>> MTCCtrack = FALSE"<<endl<<endl;
  }
  
  fit<<">>> ANALYZED RUNS = ";
  for(n=0;n!=runcounter;n++){
  fit<<runvector[n]<<", ";}
  fit<<endl;
  
  fit<<">>> TOTAL EVENTS = "<<eventcounter<<endl;
  fit<<">>> NUMBER OF TRACKS = "<<trackcounter<<endl<<endl;
  fit<<">>> NUMBER OF RECHITS = "<<hitcounter<<endl<<endl;
  
  fit<<">>> NUMBER OF DETECTOR HISTOGRAMS = "<<histonum<<endl;
  fit<<">>> NUMBER OF MONO SINGLE SIDED DETECTORS = "<<monosscounter<<endl;
  fit<<">>> NUMBER OF MONO DOUBLE SIDED DETECTORS = "<<monodscounter<<endl;
  fit<<">>> NUMBER OF STEREO DETECTORS = "<<stereocounter<<endl<<endl;
    
    if(!(conf_.getParameter<bool>("MTCCtrack"))){  //MTCCtrack FALSE
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TIB LAYER 1 -------------------------"<<endl<<endl;
    fit<<"Number of entries = "<<histos[13]->GetEntries()<<endl<<endl;
    fit<<"Detector thickness = "<<detmap[13]->thickness<<" um "<<endl;
    fit<<"Detector pitch = "<<detmap[13]->pitch<<" um "<<endl<<endl;    
    fit<<"Chi Square/ndf = "<<(fits[13]->chi2)/(fits[13]->ndf)<<endl;
    fit<<"NdF        = "<<fits[13]->ndf<<endl;
    fit<<"p0 = "<<fits[13]->p0<<"     err p0 = "<<fits[13]->errp0<<endl;
    fit<<"p1 = "<<fits[13]->p1<<"     err p1 = "<<fits[13]->errp1<<endl;
    fit<<"p2 = "<<fits[13]->p2<<"     err p2 = "<<fits[13]->errp2<<endl<<endl;
    fit<<"Minimum at tan(track local angle) = "<<fits[13]->p0<<"  +-  "<<fits[13]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[13]->p2<<"  +-  "<<fits[13]->errp2<<endl<<endl;  
    }
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TIB LAYER 2 -------------------------"<<endl<<endl;
    fit<<"Number of entries = "<<histos[1]->GetEntries()<<endl<<endl;
    fit<<"Detector thickness = "<<detmap[1]->thickness<<" um "<<endl;
    fit<<"Detector pitch = "<<detmap[1]->pitch<<" um "<<endl<<endl;    
    fit<<"Chi Square/ndf = "<<(fits[1]->chi2)/(fits[1]->ndf)<<endl;
    fit<<"NdF        = "<<fits[1]->ndf<<endl;
    fit<<"p0 = "<<fits[1]->p0<<"     err p0 = "<<fits[1]->errp0<<endl;
    fit<<"p1 = "<<fits[1]->p1<<"     err p1 = "<<fits[1]->errp1<<endl;
    fit<<"p2 = "<<fits[1]->p2<<"     err p2 = "<<fits[1]->errp2<<endl<<endl;
    fit<<"Minimum at tan(track local angle) = "<<fits[1]->p0<<"  +-  "<<fits[1]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[1]->p2<<"  +-  "<<fits[1]->errp2<<endl<<endl;
        
    fit<<endl<<"--------------------------- SUMMARY FIT: TIB LAYER 3 -------------------------"<<endl<<endl;
    fit<<"Number of entries = "<<histos[2]->GetEntries()<<endl<<endl;
    fit<<"Detector thickness = "<<detmap[2]->thickness<<" um "<<endl;
    fit<<"Detector pitch = "<<detmap[2]->pitch<<" um "<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[2]->chi2)/(fits[2]->ndf)<<endl;
    fit<<"NdF        = "<<fits[2]->ndf<<endl;
    fit<<"p0 = "<<fits[2]->p0<<"     err p0 = "<<fits[2]->errp0<<endl;
    fit<<"p1 = "<<fits[2]->p1<<"     err p1 = "<<fits[2]->errp1<<endl;
    fit<<"p2 = "<<fits[2]->p2<<"     err p2 = "<<fits[2]->errp2<<endl<<endl;
    fit<<"Minimum at tan(track local angle) = "<<fits[2]->p0<<"  +-  "<<fits[2]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[2]->p2<<"  +-  "<<fits[2]->errp2<<endl<<endl;
        
    if(!(conf_.getParameter<bool>("MTCCtrack"))){  //MTCCtrack FALSE 
   
    fit<<endl<<"--------------------------- SUMMARY FIT: TIB LAYER 4 -------------------------"<<endl<<endl;
    fit<<"Number of entries = "<<histos[14]->GetEntries()<<endl<<endl;
    fit<<"Detector thickness = "<<detmap[14]->thickness<<" um "<<endl;
    fit<<"Detector pitch = "<<detmap[14]->pitch<<" um "<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[14]->chi2)/(fits[14]->ndf)<<endl;
    fit<<"NdF        = "<<fits[14]->ndf<<endl;
    fit<<"p0 = "<<fits[14]->p0<<"     err p0 = "<<fits[14]->errp0<<endl;
    fit<<"p1 = "<<fits[14]->p1<<"     err p1 = "<<fits[14]->errp1<<endl;
    fit<<"p2 = "<<fits[14]->p2<<"     err p2 = "<<fits[14]->errp2<<endl<<endl;
    fit<<"Minimum at tan(track local angle) = "<<fits[14]->p0<<"  +-  "<<fits[14]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[14]->p2<<"  +-  "<<fits[14]->errp2<<endl<<endl;
    }             
    fit<<endl<<"--------------------------- SUMMARY FIT: TOB LAYER 1 -------------------------"<<endl<<endl;
    fit<<"Number of entries = "<<histos[3]->GetEntries()<<endl<<endl;
    fit<<"Detector thickness = "<<detmap[3]->thickness<<" um "<<endl;
    fit<<"Detector pitch = "<<detmap[3]->pitch<<" um "<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[3]->chi2)/(fits[3]->ndf)<<endl;
    fit<<"NdF        = "<<fits[3]->ndf<<endl;
    fit<<"p0 = "<<fits[3]->p0<<"     err p0 = "<<fits[3]->errp0<<endl;
    fit<<"p1 = "<<fits[3]->p1<<"     err p1 = "<<fits[3]->errp1<<endl;
    fit<<"p2 = "<<fits[3]->p2<<"     err p2 = "<<fits[3]->errp2<<endl<<endl;
    fit<<"Minimum at tan(track local angle) = "<<fits[3]->p0<<"  +-  "<<fits[3]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[3]->p2<<"  +-  "<<fits[3]->errp2<<endl<<endl;
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TOB LAYER 1 TIBseed -------------------------"<<endl<<endl;
    fit<<"Number of entries = "<<histos[4]->GetEntries()<<endl<<endl;
    fit<<"Detector thickness = "<<detmap[4]->thickness<<" um "<<endl;
    fit<<"Detector pitch = "<<detmap[4]->pitch<<" um "<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[4]->chi2)/(fits[4]->ndf)<<endl;
    fit<<"NdF        = "<<fits[4]->ndf<<endl;
    fit<<"p0 = "<<fits[4]->p0<<"     err p0 = "<<fits[4]->errp0<<endl;
    fit<<"p1 = "<<fits[4]->p1<<"     err p1 = "<<fits[4]->errp1<<endl;
    fit<<"p2 = "<<fits[4]->p2<<"     err p2 = "<<fits[4]->errp2<<endl<<endl;
    fit<<"Minimum at tan(track local angle) = "<<fits[4]->p0<<"  +-  "<<fits[4]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[4]->p2<<"  +-  "<<fits[4]->errp2<<endl<<endl;
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TOB LAYER 1 TOBseed -------------------------"<<endl<<endl;
    fit<<"Number of entries = "<<histos[5]->GetEntries()<<endl<<endl;
    fit<<"Detector thickness = "<<detmap[5]->thickness<<" um "<<endl;
    fit<<"Detector pitch = "<<detmap[5]->pitch<<" um "<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[5]->chi2)/(fits[5]->ndf)<<endl;
    fit<<"NdF        = "<<fits[5]->ndf<<endl;
    fit<<"p0 = "<<fits[5]->p0<<"     err p0 = "<<fits[5]->errp0<<endl;
    fit<<"p1 = "<<fits[5]->p1<<"     err p1 = "<<fits[5]->errp1<<endl;
    fit<<"p2 = "<<fits[5]->p2<<"     err p2 = "<<fits[5]->errp2<<endl<<endl;
    fit<<"Minimum at tan(track local angle) = "<<fits[5]->p0<<"  +-  "<<fits[5]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[5]->p2<<"  +-  "<<fits[5]->errp2<<endl<<endl;
    
    if(!(conf_.getParameter<bool>("MTCCtrack"))){  //MTCCtrack FALSE 
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TOB LAYER 2 -------------------------"<<endl<<endl;
    fit<<"Number of entries = "<<histos[16]->GetEntries()<<endl<<endl;
    fit<<"Detector thickness = "<<detmap[16]->thickness<<" um "<<endl;
    fit<<"Detector pitch = "<<detmap[16]->pitch<<" um "<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[16]->chi2)/(fits[16]->ndf)<<endl;
    fit<<"NdF        = "<<fits[16]->ndf<<endl;
    fit<<"p0 = "<<fits[16]->p0<<"     err p0 = "<<fits[16]->errp0<<endl;
    fit<<"p1 = "<<fits[16]->p1<<"     err p1 = "<<fits[16]->errp1<<endl;
    fit<<"p2 = "<<fits[16]->p2<<"     err p2 = "<<fits[16]->errp2<<endl<<endl;
    fit<<"Minimum at tan(track local angle) = "<<fits[16]->p0<<"  +-  "<<fits[16]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[16]->p2<<"  +-  "<<fits[16]->errp2<<endl<<endl;
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TOB LAYER 3 -------------------------"<<endl<<endl;
    fit<<"Number of entries = "<<histos[17]->GetEntries()<<endl<<endl;
    fit<<"Detector thickness = "<<detmap[17]->thickness<<" um "<<endl;
    fit<<"Detector pitch = "<<detmap[17]->pitch<<" um "<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[17]->chi2)/(fits[17]->ndf)<<endl;
    fit<<"NdF        = "<<fits[17]->ndf<<endl;
    fit<<"p0 = "<<fits[17]->p0<<"     err p0 = "<<fits[17]->errp0<<endl;
    fit<<"p1 = "<<fits[17]->p1<<"     err p1 = "<<fits[17]->errp1<<endl;
    fit<<"p2 = "<<fits[17]->p2<<"     err p2 = "<<fits[17]->errp2<<endl<<endl;
    fit<<"Minimum at tan(track local angle) = "<<fits[17]->p0<<"  +-  "<<fits[17]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[17]->p2<<"  +-  "<<fits[17]->errp2<<endl<<endl;
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TOB LAYER 4 -------------------------"<<endl<<endl;
    fit<<"Number of entries = "<<histos[3]->GetEntries()<<endl<<endl;
    fit<<"Detector thickness = "<<detmap[3]->thickness<<" um "<<endl;
    fit<<"Detector pitch = "<<detmap[3]->pitch<<" um "<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[3]->chi2)/(fits[3]->ndf)<<endl;
    fit<<"NdF        = "<<fits[3]->ndf<<endl;
    fit<<"p0 = "<<fits[3]->p0<<"     err p0 = "<<fits[3]->errp0<<endl;
    fit<<"p1 = "<<fits[3]->p1<<"     err p1 = "<<fits[3]->errp1<<endl;
    fit<<"p2 = "<<fits[3]->p2<<"     err p2 = "<<fits[3]->errp2<<endl<<endl;
    fit<<"Minimum at tan(track local angle) = "<<fits[3]->p0<<"  +-  "<<fits[3]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[3]->p2<<"  +-  "<<fits[3]->errp2<<endl<<endl;
    }
            
    fit<<endl<<"--------------------------- SUMMARY FIT: TOB LAYER 5 -------------------------"<<endl<<endl;
    fit<<"Number of entries = "<<histos[6]->GetEntries()<<endl<<endl;
    fit<<"Detector thickness = "<<detmap[6]->thickness<<" um "<<endl;
    fit<<"Detector pitch = "<<detmap[6]->pitch<<" um "<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[6]->chi2)/(fits[6]->ndf)<<endl;
    fit<<"NdF        = "<<fits[6]->ndf<<endl;
    fit<<"p0 = "<<fits[6]->p0<<"     err p0 = "<<fits[6]->errp0<<endl;
    fit<<"p1 = "<<fits[6]->p1<<"     err p1 = "<<fits[6]->errp1<<endl;
    fit<<"p2 = "<<fits[6]->p2<<"     err p2 = "<<fits[6]->errp2<<endl<<endl;
    fit<<"Minimum at tan(track local angle) = "<<fits[6]->p0<<"  +-  "<<fits[6]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[6]->p2<<"  +-  "<<fits[6]->errp2<<endl<<endl;
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TOB LAYER 5 TIBseed -------------------------"<<endl<<endl;
    fit<<"Number of entries = "<<histos[7]->GetEntries()<<endl<<endl;
    fit<<"Detector thickness = "<<detmap[7]->thickness<<" um "<<endl;
    fit<<"Detector pitch = "<<detmap[7]->pitch<<" um "<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[7]->chi2)/(fits[7]->ndf)<<endl;
    fit<<"NdF        = "<<fits[7]->ndf<<endl;
    fit<<"p0 = "<<fits[7]->p0<<"     err p0 = "<<fits[7]->errp0<<endl;
    fit<<"p1 = "<<fits[7]->p1<<"     err p1 = "<<fits[7]->errp1<<endl;
    fit<<"p2 = "<<fits[7]->p2<<"     err p2 = "<<fits[7]->errp2<<endl<<endl;
    fit<<"Minimum at tan(track local angle) = "<<fits[7]->p0<<"  +-  "<<fits[7]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[7]->p2<<"  +-  "<<fits[7]->errp2<<endl<<endl;
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TOB LAYER 5 TOBseed -------------------------"<<endl<<endl;
    fit<<"Number of entries = "<<histos[8]->GetEntries()<<endl<<endl;
    fit<<"Detector thickness = "<<detmap[8]->thickness<<" um "<<endl;
    fit<<"Detector pitch = "<<detmap[8]->pitch<<" um "<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[8]->chi2)/(fits[6]->ndf)<<endl;
    fit<<"NdF        = "<<fits[8]->ndf<<endl;
    fit<<"p0 = "<<fits[8]->p0<<"     err p0 = "<<fits[8]->errp0<<endl;
    fit<<"p1 = "<<fits[8]->p1<<"     err p1 = "<<fits[8]->errp1<<endl;
    fit<<"p2 = "<<fits[8]->p2<<"     err p2 = "<<fits[8]->errp2<<endl<<endl;
    fit<<"Minimum at tan(track local angle) = "<<fits[8]->p0<<"  +-  "<<fits[8]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[8]->p2<<"  +-  "<<fits[8]->errp2<<endl<<endl;
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TOB LAYER 1 Rod 1 -------------------------"<<endl<<endl;
    fit<<"Number of entries = "<<histos[9]->GetEntries()<<endl<<endl;
    fit<<"Detector thickness = "<<detmap[9]->thickness<<" um "<<endl;
    fit<<"Detector pitch = "<<detmap[9]->pitch<<" um "<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[9]->chi2)/(fits[9]->ndf)<<endl;
    fit<<"NdF        = "<<fits[9]->ndf<<endl;
    fit<<"p0 = "<<fits[9]->p0<<"     err p0 = "<<fits[9]->errp0<<endl;
    fit<<"p1 = "<<fits[9]->p1<<"     err p1 = "<<fits[9]->errp1<<endl;
    fit<<"p2 = "<<fits[9]->p2<<"     err p2 = "<<fits[9]->errp2<<endl<<endl;
    fit<<"Minimum at tan(track local angle) = "<<fits[9]->p0<<"  +-  "<<fits[9]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[9]->p2<<"  +-  "<<fits[9]->errp2<<endl<<endl;
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TOB LAYER 1 Rod 2 -------------------------"<<endl<<endl;
    fit<<"Number of entries = "<<histos[10]->GetEntries()<<endl<<endl;
    fit<<"Detector thickness = "<<detmap[10]->thickness<<" um "<<endl;
    fit<<"Detector pitch = "<<detmap[10]->pitch<<" um "<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[10]->chi2)/(fits[10]->ndf)<<endl;
    fit<<"NdF        = "<<fits[10]->ndf<<endl;
    fit<<"p0 = "<<fits[10]->p0<<"     err p0 = "<<fits[10]->errp0<<endl;
    fit<<"p1 = "<<fits[10]->p1<<"     err p1 = "<<fits[10]->errp1<<endl;
    fit<<"p2 = "<<fits[10]->p2<<"     err p2 = "<<fits[10]->errp2<<endl<<endl;
    fit<<"Minimum at tan(track local angle) = "<<fits[10]->p0<<"  +-  "<<fits[10]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[10]->p2<<"  +-  "<<fits[10]->errp2<<endl<<endl;
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TOB LAYER 5 Rod 1 -------------------------"<<endl<<endl;
    fit<<"Number of entries = "<<histos[11]->GetEntries()<<endl<<endl;
    fit<<"Detector thickness = "<<detmap[11]->thickness<<" um "<<endl;
    fit<<"Detector pitch = "<<detmap[11]->pitch<<" um "<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[11]->chi2)/(fits[11]->ndf)<<endl;
    fit<<"NdF        = "<<fits[11]->ndf<<endl;
    fit<<"p0 = "<<fits[11]->p0<<"     err p0 = "<<fits[11]->errp0<<endl;
    fit<<"p1 = "<<fits[11]->p1<<"     err p1 = "<<fits[11]->errp1<<endl;
    fit<<"p2 = "<<fits[11]->p2<<"     err p2 = "<<fits[11]->errp2<<endl<<endl;
    fit<<"Minimum at tan(track local angle) = "<<fits[11]->p0<<"  +-  "<<fits[11]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[11]->p2<<"  +-  "<<fits[11]->errp2<<endl<<endl;
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TOB LAYER 5 Rod 2 -------------------------"<<endl<<endl;
    fit<<"Number of entries = "<<histos[12]->GetEntries()<<endl<<endl;
    fit<<"Detector thickness = "<<detmap[12]->thickness<<" um "<<endl;
    fit<<"Detector pitch = "<<detmap[12]->pitch<<" um "<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[12]->chi2)/(fits[12]->ndf)<<endl;
    fit<<"NdF        = "<<fits[12]->ndf<<endl;
    fit<<"p0 = "<<fits[12]->p0<<"     err p0 = "<<fits[12]->errp0<<endl;
    fit<<"p1 = "<<fits[12]->p1<<"     err p1 = "<<fits[12]->errp1<<endl;
    fit<<"p2 = "<<fits[12]->p2<<"     err p2 = "<<fits[12]->errp2<<endl<<endl;
    fit<<"Minimum at tan(track local angle) = "<<fits[12]->p0<<"  +-  "<<fits[12]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[12]->p2<<"  +-  "<<fits[12]->errp2<<endl<<endl;
    
    if(!(conf_.getParameter<bool>("MTCCtrack"))){  //MTCCtrack FALSE
    fit<<endl<<"--------------------------- SUMMARY FIT: TOB LAYER 6 -------------------------"<<endl<<endl;
    fit<<"Number of entries = "<<histos[18]->GetEntries()<<endl<<endl;
    fit<<"Detector thickness = "<<detmap[18]->thickness<<" um "<<endl;
    fit<<"Detector pitch = "<<detmap[18]->pitch<<" um "<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[18]->chi2)/(fits[18]->ndf)<<endl;
    fit<<"NdF        = "<<fits[18]->ndf<<endl;
    fit<<"p0 = "<<fits[18]->p0<<"     err p0 = "<<fits[18]->errp0<<endl;
    fit<<"p1 = "<<fits[18]->p1<<"     err p1 = "<<fits[18]->errp1<<endl;
    fit<<"p2 = "<<fits[18]->p2<<"     err p2 = "<<fits[18]->errp2<<endl<<endl;
    fit<<"Minimum at tan(track local angle) = "<<fits[18]->p0<<"  +-  "<<fits[18]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[18]->p2<<"  +-  "<<fits[18]->errp2<<endl<<endl;
    }
    
  for(Iditer=Detvector.begin(); Iditer!=Detvector.end(); Iditer++){
    
    fit<<endl<<"-------------------------- MODULE HISTOGRAM FIT ------------------------"<<endl<<endl;
    fit<<makedescription(*Iditer)<<endl<<endl;
    fit<<"Number of entries = "<<histos[Iditer->rawId()]->GetEntries()<<endl<<endl;
    fit<<"Detector thickness = "<<detmap[Iditer->rawId()]->thickness<<" um "<<endl;
    fit<<"Detector pitch = "<<detmap[Iditer->rawId()]->pitch<<" um "<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[Iditer->rawId()]->chi2)/(fits[Iditer->rawId()]->ndf)<<endl;
    fit<<"NdF        = "<<fits[Iditer->rawId()]->ndf<<endl;
    fit<<"p0 = "<<fits[Iditer->rawId()]->p0<<"     err p0 = "<<fits[Iditer->rawId()]->errp0<<endl;
    fit<<"p1 = "<<fits[Iditer->rawId()]->p1<<"     err p1 = "<<fits[Iditer->rawId()]->errp1<<endl;
    fit<<"p2 = "<<fits[Iditer->rawId()]->p2<<"     err p2 = "<<fits[Iditer->rawId()]->errp2<<endl<<endl;
    fit<<"Minimum at tan(track local angle) = "<<fits[Iditer->rawId()]->p0<<"  +-  "<<fits[Iditer->rawId()]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[Iditer->rawId()]->p2<<"  +-  "<<fits[Iditer->rawId()]->errp2<<endl<<endl;
    
  }
  
  fit.close();
    
  //Set directories
  
  CollHitSizeTIBL2mono->SetDirectory(sizesummary);
  CollHitSizeTIBL2stereo->SetDirectory(sizesummary);
  CollHitSizeTIBL3->SetDirectory(sizesummary);
  CollHitSizeTOBL1->SetDirectory(sizesummary);
  CollHitSizeTOBL5->SetDirectory(sizesummary);
  TrackHitSizeTIBL2mono->SetDirectory(sizesummary);
  TrackHitSizeTIBL2stereo->SetDirectory(sizesummary);
  TrackHitSizeTIBL3->SetDirectory(sizesummary);
  TrackHitSizeTOBL1->SetDirectory(sizesummary);
  TrackHitSizeTOBL5->SetDirectory(sizesummary);
  
  for(n=1;n<nmax;n++){
  histos[n]->SetDirectory(summary);}
   
  for(Iditer=Detvector.begin(); Iditer!=Detvector.end(); Iditer++){
  
  StripSubdetector DetId(Iditer->rawId());
  
  if(Iditer->subdetId() == int (StripSubdetector::TIB)){
      
    TIBDetId TIBid=TIBDetId(DetId);
    
    int correctedlayer = TIBid.layer() + mtcctibcorr;
      
    if(TIBid.string()[0] == 0){   
      if(correctedlayer == 1){
      histos[Iditer->rawId()]->SetDirectory(TIBbw1);}
      if(correctedlayer == 2){
      histos[Iditer->rawId()]->SetDirectory(TIBbw2);}
      if(correctedlayer == 3){
      histos[Iditer->rawId()]->SetDirectory(TIBbw3);}
      if(correctedlayer == 4){
      histos[Iditer->rawId()]->SetDirectory(TIBbw4);}
      }
    if(TIBid.string()[0] == 1){
      if(correctedlayer == 1){
      histos[Iditer->rawId()]->SetDirectory(TIBfw1);}
      if(correctedlayer == 2){
      histos[Iditer->rawId()]->SetDirectory(TIBfw2);}
      if(correctedlayer == 3){
      histos[Iditer->rawId()]->SetDirectory(TIBfw3);}
      if(correctedlayer == 4){
      histos[Iditer->rawId()]->SetDirectory(TIBfw4);}
      }
      } 
      
  if(Iditer->subdetId() == int (StripSubdetector::TID)){
  
     TIDDetId TIDid=TIDDetId(DetId);
     
      if(TIDid.module()[0] == 0){ 
        if(TIDid.wheel() == 1){
        histos[Iditer->rawId()]->SetDirectory(TIDbw1);}
	if(TIDid.wheel() == 2){
        histos[Iditer->rawId()]->SetDirectory(TIDbw2);}
	if(TIDid.wheel() == 3){
        histos[Iditer->rawId()]->SetDirectory(TIDbw3);}
	} 
      if(TIDid.module()[0] == 1){
        if(TIDid.wheel() == 1){
        histos[Iditer->rawId()]->SetDirectory(TIDfw1);}
	if(TIDid.wheel() == 2){
        histos[Iditer->rawId()]->SetDirectory(TIDfw2);}
	if(TIDid.wheel() == 3){
        histos[Iditer->rawId()]->SetDirectory(TIDfw3);}
	}
	}
      
  if(Iditer->subdetId() == int (StripSubdetector::TOB)){
    
    TOBDetId TOBid=TOBDetId(DetId);
    
    int correctedlayer = TOBid.layer();
    if(correctedlayer==2){
    correctedlayer+=mtcctobcorr;}
    
     if(TOBid.rod()[0] == 0){      
       if(correctedlayer == 1){
       histos[Iditer->rawId()]->SetDirectory(TOBbw1);}
       if(correctedlayer == 2){
       histos[Iditer->rawId()]->SetDirectory(TOBbw2);}
       if(correctedlayer == 3){
       histos[Iditer->rawId()]->SetDirectory(TOBbw3);}
       if(correctedlayer == 4){
       histos[Iditer->rawId()]->SetDirectory(TOBbw4);}
       if(correctedlayer == 5){
       histos[Iditer->rawId()]->SetDirectory(TOBbw5);}
       if(correctedlayer == 6){
       histos[Iditer->rawId()]->SetDirectory(TOBbw6);}
       }
     if(TOBid.rod()[0] == 1){
       if(correctedlayer == 1){
       histos[Iditer->rawId()]->SetDirectory(TOBfw1);}
       if(correctedlayer == 2){
       histos[Iditer->rawId()]->SetDirectory(TOBfw2);}
       if(correctedlayer == 3){
       histos[Iditer->rawId()]->SetDirectory(TOBfw3);}
       if(correctedlayer == 4){
       histos[Iditer->rawId()]->SetDirectory(TOBfw4);}
       if(correctedlayer == 5){
       histos[Iditer->rawId()]->SetDirectory(TOBfw5);}
       if(correctedlayer == 6){
       histos[Iditer->rawId()]->SetDirectory(TOBfw6);}
       }
       }
       
   if(Iditer->subdetId() == int (StripSubdetector::TEC)){
    
     TECDetId TECid=TECDetId(DetId);
     
       if(TECid.petal()[0] == 0){     
         if(TECid.wheel() == 1){
         histos[Iditer->rawId()]->SetDirectory(TECbw1);}
	 if(TECid.wheel() == 2){
         histos[Iditer->rawId()]->SetDirectory(TECbw2);}
	 if(TECid.wheel() == 3){
         histos[Iditer->rawId()]->SetDirectory(TECbw3);}
	 if(TECid.wheel() == 4){
         histos[Iditer->rawId()]->SetDirectory(TECbw4);}
	 if(TECid.wheel() == 5){
         histos[Iditer->rawId()]->SetDirectory(TECbw5);}
	 if(TECid.wheel() == 6){
         histos[Iditer->rawId()]->SetDirectory(TECbw6);}
	 if(TECid.wheel() == 7){
         histos[Iditer->rawId()]->SetDirectory(TECbw7);}
	 if(TECid.wheel() == 8){
         histos[Iditer->rawId()]->SetDirectory(TECbw8);}
	 if(TECid.wheel() == 9){
         histos[Iditer->rawId()]->SetDirectory(TECbw9);}
	 }
       if(TECid.petal()[0] == 1){
         if(TECid.wheel() == 1){
         histos[Iditer->rawId()]->SetDirectory(TECfw1);}
	 if(TECid.wheel() == 2){
         histos[Iditer->rawId()]->SetDirectory(TECfw2);}
	 if(TECid.wheel() == 3){
         histos[Iditer->rawId()]->SetDirectory(TECfw3);}
	 if(TECid.wheel() == 4){
         histos[Iditer->rawId()]->SetDirectory(TECfw4);}
	 if(TECid.wheel() == 5){
         histos[Iditer->rawId()]->SetDirectory(TECfw5);}
	 if(TECid.wheel() == 6){
         histos[Iditer->rawId()]->SetDirectory(TECfw6);}
	 if(TECid.wheel() == 7){
         histos[Iditer->rawId()]->SetDirectory(TECfw7);}
	 if(TECid.wheel() == 8){
         histos[Iditer->rawId()]->SetDirectory(TECfw8);}
	 if(TECid.wheel() == 9){
         histos[Iditer->rawId()]->SetDirectory(TECfw9);}
	 }
	 }
	  
 }        
  
  hFile->Write();
  hFile->Close();
  
}
