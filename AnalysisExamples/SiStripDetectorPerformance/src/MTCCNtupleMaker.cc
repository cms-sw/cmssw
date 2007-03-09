#include <memory>
#include <string>
#include <iostream>
#include <fstream>

#include "AnalysisExamples/SiStripDetectorPerformance/interface/MTCCNtupleMaker.h"

#include "DataFormats/Common/interface/Handle.h"
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
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "AnalysisDataFormats/SiStripClusterInfo/interface/SiStripClusterInfo.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/LTCDigi/interface/LTCDigi.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "MagneticField/Engine/interface/MagneticField.h"


using namespace std;

  //Constructor

MTCCNtupleMaker::MTCCNtupleMaker(edm::ParameterSet const& conf) : 
  conf_(conf), 
  filename_(conf.getParameter<std::string>("fileName")),
  oSiStripDigisLabel_( conf.getUntrackedParameter<std::string>( "oSiStripDigisLabel")),
  oSiStripDigisProdInstName_( conf.getUntrackedParameter<std::string>( "oSiStripDigisProdInstName")),
  oSeedsLabel_( conf.getUntrackedParameter<std::string>( "SeedsLabel")),
  oSeedsProdInstName_( conf.getUntrackedParameter<std::string>( "SeedsProdInstName")),
  oTracksLabel_( conf.getUntrackedParameter<std::string>( "TracksLabel")),
  oTracksProdInstName_( conf.getUntrackedParameter<std::string>( "TracksProdInstName")),
  bUseLTCDigis_( conf.getUntrackedParameter<bool>( "bUseLTCDigis")),
  dCROSS_TALK_ERR( conf.getUntrackedParameter<double>( "dCrossTalkErr")),
  bTriggerDT( false),
  bTriggerCSC( false),
  bTriggerRBC1( false),
  bTriggerRBC2( false),
  bTriggerRPC( false)
  //m_oSiStripNoiseService( conf_)
{
  anglefinder_=new  TrackLocalAngleNew(conf);  
}

  //BeginJob

void MTCCNtupleMaker::beginJob(const edm::EventSetup& c){

  hFile = new TFile (filename_.c_str(), "RECREATE" );
  
  TAxis *xaxis, *yaxis;
  
  hphi = new TH1F("hphi","Phi distribution",20,-3.14,3.14);
  hnhit = new TH1F("hnhit","Number of Hits per Track ",18,2,20);
     
  hwvst = new TProfile("WidthvsTrackProjection","Cluster width vs track projection ",120,-60.,60.);
  fitfunc = new TF1("fitfunc","[1]*((x-[0])^2)+[2]",-30,30); 
   
  MTCCNtupleMakerTree = new TTree("MTCCNtupleMakerTree","SiStrip LorentzAngle tree");
  MTCCNtupleMakerTree->Branch("run", &run, "run/I");
  MTCCNtupleMakerTree->Branch( "eventcounter", &eventcounter, "eventcounter/I");
  MTCCNtupleMakerTree->Branch("event", &event, "event/I");
  MTCCNtupleMakerTree->Branch("module", &module, "module/I");
  MTCCNtupleMakerTree->Branch("type", &type, "type/I");
  MTCCNtupleMakerTree->Branch("layer", &layer, "layer/I");
  MTCCNtupleMakerTree->Branch("string", &string, "string/I");
  MTCCNtupleMakerTree->Branch("rod", &rod, "rod/I");
  MTCCNtupleMakerTree->Branch("extint", &extint, "extint/I");
  MTCCNtupleMakerTree->Branch("size", &size, "size/I");
  MTCCNtupleMakerTree->Branch("angle", &angle, "angle/F");
  MTCCNtupleMakerTree->Branch("sign", &sign, "sign/I");
  MTCCNtupleMakerTree->Branch("bwfw", &bwfw, "bwfw/I");
  MTCCNtupleMakerTree->Branch("wheel", &wheel, "wheel/I");
  MTCCNtupleMakerTree->Branch("monostereo", &monostereo, "monostereo/I");
  MTCCNtupleMakerTree->Branch("stereocorrection", &stereocorrection, "stereocorrection/F");
  MTCCNtupleMakerTree->Branch("localmagfield", &localmagfield, "localmagfield/F");
  MTCCNtupleMakerTree->Branch("momentum", &momentum, "momentum/F");
  MTCCNtupleMakerTree->Branch("pt", &pt, "pt/F");
  MTCCNtupleMakerTree->Branch("charge", &charge, "charge/I");
  MTCCNtupleMakerTree->Branch("eta", &eta, "eta/F");
  MTCCNtupleMakerTree->Branch("phi", &phi, "phi/F");
  MTCCNtupleMakerTree->Branch("hitspertrack", &hitspertrack, "hitspertrack/I");
  MTCCNtupleMakerTree->Branch( "normchi2", &normchi2, "normchi2/F");
  MTCCNtupleMakerTree->Branch( "chi2", &chi2, "chi2/F");
  MTCCNtupleMakerTree->Branch( "ndof", &ndof, "ndof/F");
  MTCCNtupleMakerTree->Branch( "bTrack", &bTrack, "bTrack/O");
  MTCCNtupleMakerTree->Branch( "clusterpos", &clusterpos, "clusterpos/F");
  MTCCNtupleMakerTree->Branch( "clustereta", &clustereta, "clustereta/F");
  MTCCNtupleMakerTree->Branch( "clusterchg", &clusterchg, "clusterchg/F");
  MTCCNtupleMakerTree->Branch( "clusterchgl", &clusterchgl, "clusterchgl/F");
  MTCCNtupleMakerTree->Branch( "clusterchgr", &clusterchgr, "clusterchgr/F");
  MTCCNtupleMakerTree->Branch( "clusternoise", &clusternoise, "clusternoise/F");
  MTCCNtupleMakerTree->Branch( "clusterbarycenter", &clusterbarycenter, "clusterbarycenter/F");
  MTCCNtupleMakerTree->Branch( "clustermaxchg", &clustermaxchg, "clustermaxchg/F");
  MTCCNtupleMakerTree->Branch( "clusterseednoise", &clusterseednoise, "clusterseednoise/F");
  MTCCNtupleMakerTree->Branch( "clustercrosstalk", &clustercrosstalk, "clustercrosstalk/F");
  // Trigger Bits
  MTCCNtupleMakerTree->Branch( "bTriggerDT",   &bTriggerDT,	  "bTriggerDT/O");
  MTCCNtupleMakerTree->Branch( "bTriggerCSC",  &bTriggerCSC,  "bTriggerCSC/O");
  MTCCNtupleMakerTree->Branch( "bTriggerRBC1", &bTriggerRBC1, "bTriggerRBC1/O");
  MTCCNtupleMakerTree->Branch( "bTriggerRBC2", &bTriggerRBC2, "bTriggerRBC2/O");
  MTCCNtupleMakerTree->Branch( "bTriggerRPC",  &bTriggerRPC,  "bTriggerRPC/O");
  MTCCNtupleMakerTree->Branch( "dLclX", &dLclX, "dLclX/F");
  MTCCNtupleMakerTree->Branch( "dLclY", &dLclY, "dLclY/F");
  MTCCNtupleMakerTree->Branch( "dLclZ", &dLclZ, "dLclZ/F");
  MTCCNtupleMakerTree->Branch( "dGlbX", &dGlbX, "dGlbX/F");
  MTCCNtupleMakerTree->Branch( "dGlbY", &dGlbY, "dGlbY/F");
  MTCCNtupleMakerTree->Branch( "dGlbZ", &dGlbZ, "dGlbZ/F");

  poTrackTree = new TTree( "TrackTree", "This is a Track specific variables tree");
  poTrackTree->Branch( "run",	       &run,	      "run/I");
  poTrackTree->Branch( "pt",	       &pt,	      "pt/F");
  poTrackTree->Branch( "eta",	       &eta,	      "eta/F");
  poTrackTree->Branch( "phi",	       &phi,	      "phi/F");
  poTrackTree->Branch( "ndof",	       &ndof,	      "ndof/F");
  poTrackTree->Branch( "chi2",	       &chi2,	      "chi2/F");
  poTrackTree->Branch( "event",	       &event,	      "event/I");
  poTrackTree->Branch( "charge",       &charge,	      "charge/I");
  poTrackTree->Branch( "momentum",     &momentum,     "momentum/F");
  poTrackTree->Branch( "eventcounter", &eventcounter, "eventcounter/I");
  poTrackTree->Branch( "hitspertrack", &hitspertrack, "hitspertrack/I");
  // Trigger Bits
  poTrackTree->Branch( "bTriggerDT",   &bTriggerDT,   "bTriggerDT/O");
  poTrackTree->Branch( "bTriggerCSC",  &bTriggerCSC,  "bTriggerCSC/O");
  poTrackTree->Branch( "bTriggerRBC1", &bTriggerRBC1, "bTriggerRBC1/O");
  poTrackTree->Branch( "bTriggerRBC2", &bTriggerRBC2, "bTriggerRBC2/O");
  poTrackTree->Branch( "bTriggerRPC",  &bTriggerRPC,  "bTriggerRPC/O");

  eventcounter = 0;
  trackcounter = 0;
  hitcounter = 0;
  
  mtcctibcorr = 0;
  mtcctobcorr = 0;
  
  if(conf_.getParameter<bool>("MTCCtrack")){  //MTCCtrack TRUE
    mtcctibcorr = 1;
    mtcctobcorr = 2;
  }
  
  if(!(conf_.getParameter<bool>("MTCCtrack"))){  //MTCCtrack FALSE  
    htaTIBL1mono = new TH1F("TIBL1angle_mono","Track angle (TIB L1) MONO",120,-60.,60.);
    xaxis = htaTIBL1mono->GetXaxis();
    xaxis->SetTitle("degree");
    htaTIBL1stereo = new TH1F("TIBL1angle_stereo","Track angle (TIB L1) STEREO",120,-60.,60.);
    xaxis = htaTIBL1stereo->GetXaxis();
    xaxis->SetTitle("degree");
  }
    
  htaTIBL2mono = new TH1F("TIBL2angle_mono","Track angle (TIB L2) MONO",120,-60.,60.);
  xaxis = htaTIBL2mono->GetXaxis();
  xaxis->SetTitle("degree");
  htaTIBL2stereo = new TH1F("TIBL2angle_stereo","Track angle (TIB L2) STEREO",120,-60.,60.);
  xaxis = htaTIBL2stereo->GetXaxis();
  xaxis->SetTitle("degree");
  htaTIBL3 = new TH1F("TIBL3angle","Track angle (TIB L3)",120,-60.,60.);
  xaxis = htaTIBL3->GetXaxis();
  xaxis->SetTitle("degree");
  
  if(!(conf_.getParameter<bool>("MTCCtrack"))){  //MTCCtrack FALSE
    htaTIBL4 = new TH1F("TIBL4angle","Track angle (TIB L4)",120,-60.,60.);
    xaxis = htaTIBL4->GetXaxis();
    xaxis->SetTitle("degree");
    
    htaTOBL1mono = new TH1F("TOBL1angle_mono","Track angle (TOB L1) MONO",120,-60.,60.);
    xaxis = htaTOBL1mono->GetXaxis();
    xaxis->SetTitle("degree");
    htaTOBL1stereo = new TH1F("TOBL1angle_stereo","Track angle (TOB L1) STEREO",120,-60.,60.);
    xaxis = htaTOBL1stereo->GetXaxis();
    xaxis->SetTitle("degree");
    htaTOBL2mono = new TH1F("TOBL2angle_mono","Track angle (TOB L2) MONO",120,-60.,60.);
    xaxis = htaTOBL2mono->GetXaxis();
    xaxis->SetTitle("degree");
    htaTOBL2stereo = new TH1F("TOBL2angle_stereo","Track angle (TOB L2) STEREO",120,-60.,60.);
    xaxis = htaTOBL2stereo->GetXaxis();
    xaxis->SetTitle("degree");
  }
    
  htaTOBL3 = new TH1F("TOBL3","Track angle (TOB L3)",120,-60.,60.);
  xaxis = htaTOBL3->GetXaxis();
  xaxis->SetTitle("degree");
  htaTOBL4 = new TH1F("TOBL4","Track angle (TOB L4)",120,-60.,60.);
  xaxis = htaTOBL4->GetXaxis();
  xaxis->SetTitle("degree");
  
  if(!(conf_.getParameter<bool>("MTCCtrack"))){  //MTCCtrack FALSE
    htaTOBL5 = new TH1F("TOBL5","Track angle (TOB L5)",120,-60.,60.);
    xaxis = htaTOBL5->GetXaxis();
    xaxis->SetTitle("degree");
    htaTOBL6 = new TH1F("TOBL6","Track angle (TOB L6)",120,-60.,60.);
    xaxis = htaTOBL6->GetXaxis();
    xaxis->SetTitle("degree");
  }
          	   
  edm::ESHandle<MagneticField> esmagfield;
  c.get<IdealMagneticFieldRecord>().get(esmagfield);
  magfield=&(*esmagfield);
    
  edm::ESHandle<TrackerGeometry> estracker;
  c.get<TrackerDigiGeometryRecord>().get(estracker);
  tracker=&(*estracker); 
  
  //Get Ids;
  
  const TrackerGeometry::DetIdContainer& Id = estracker->detIds();
   
  TrackerGeometry::DetIdContainer::const_iterator Iditer;
    
  monodscounter=0;
  monosscounter=0;
  stereocounter=0;
    
for(Iditer=Id.begin();Iditer!=Id.end();Iditer++){
  
  	if((Iditer->subdetId() != PixelSubdetector::PixelBarrel) && (Iditer->subdetId() != PixelSubdetector::PixelEndcap)){
	   
		StripSubdetector subid(*Iditer);
		
		//Mono single sided detectors
		
		if(subid.glued() == 0){
		
		        if(conf_.getParameter<bool>("MTCCtrack")){  //MTCCtrack TRUE
			
			if(subid.subdetId() == int (StripSubdetector::TIB)){ 
			TIBDetId TIBid=TIBDetId(*Iditer);
			if(TIBid.layer()!=1){		
			monosscounter++;
    			Detvector.push_back(*Iditer);
    			histos[Iditer->rawId()] = new TProfile(makename(*Iditer),makedescription(*Iditer),30,-30.,30.);
    			xaxis = histos[Iditer->rawId()]->GetXaxis();
    			xaxis->SetTitle("degree");
    			yaxis = histos[Iditer->rawId()]->GetYaxis();
    			yaxis->SetTitle("number of strips");
			}
			}
			
			}
		
		        if(!(conf_.getParameter<bool>("MTCCtrack"))){  //MTCCtrack FALSE
			
			if(subid.subdetId() == int (StripSubdetector::TIB)){
			TIBDetId TIBid=TIBDetId(*Iditer);
			if((TIBid.layer()!=1) && (TIBid.layer()!=2)){			
		   	monosscounter++;
    			Detvector.push_back(*Iditer);
    			histos[Iditer->rawId()] = new TProfile(makename(*Iditer),makedescription(*Iditer),30,-30.,30.);
    			xaxis = histos[Iditer->rawId()]->GetXaxis();
    			xaxis->SetTitle("degree");
    			yaxis = histos[Iditer->rawId()]->GetYaxis();
    			yaxis->SetTitle("number of strips");
			}
			}
			
			}      
			
			if(conf_.getParameter<bool>("MTCCtrack")){  //MTCCtrack TRUE
			
			if(subid.subdetId() == int (StripSubdetector::TOB)){  
			monosscounter++;
    			Detvector.push_back(*Iditer);
    			histos[Iditer->rawId()] = new TProfile(makename(*Iditer),makedescription(*Iditer),30,-30.,30.);
    			xaxis = histos[Iditer->rawId()]->GetXaxis();
    			xaxis->SetTitle("degree");
    			yaxis = histos[Iditer->rawId()]->GetYaxis();
    			yaxis->SetTitle("number of strips");
			}
			
			}
			
			if(!(conf_.getParameter<bool>("MTCCtrack"))){  //MTCCtrack FALSE
			
			if(subid.subdetId() == int (StripSubdetector::TOB)){
			TOBDetId TOBid=TOBDetId(*Iditer);
			if((TOBid.layer()!=1) && (TOBid.layer()!=2)){
			monosscounter++;
    			Detvector.push_back(*Iditer);
    			histos[Iditer->rawId()] = new TProfile(makename(*Iditer),makedescription(*Iditer),30,-30.,30.);
    			xaxis = histos[Iditer->rawId()]->GetXaxis();
    			xaxis->SetTitle("degree");
    			yaxis = histos[Iditer->rawId()]->GetYaxis();
    			yaxis->SetTitle("number of strips");
			}
			}
			
			}
			
			if(subid.subdetId() == int (StripSubdetector::TID)){
			TIDDetId TIDid=TIDDetId(*Iditer);
			if((TIDid.ring()!=1) && (TIDid.ring()!=2)){
			monosscounter++;
    			Detvector.push_back(*Iditer);
    			histos[Iditer->rawId()] = new TProfile(makename(*Iditer),makedescription(*Iditer),30,-30.,30.);
    			xaxis = histos[Iditer->rawId()]->GetXaxis();
    			xaxis->SetTitle("degree");
    			yaxis = histos[Iditer->rawId()]->GetYaxis();
    			yaxis->SetTitle("number of strips");
			}
			}
			
			if(subid.subdetId() == int (StripSubdetector::TEC)){
			TECDetId TECid=TECDetId(*Iditer);
			if((TECid.ring()!=1) && (TECid.ring()!=2) && (TECid.ring()!=5)){
			monosscounter++;
    			Detvector.push_back(*Iditer);
    			histos[Iditer->rawId()] = new TProfile(makename(*Iditer),makedescription(*Iditer),30,-30.,30.);
    			xaxis = histos[Iditer->rawId()]->GetXaxis();
    			xaxis->SetTitle("degree");
    			yaxis = histos[Iditer->rawId()]->GetYaxis();
    			yaxis->SetTitle("number of strips");
			}
			}
		}
		
		//Mono double sided detectors
		
		if((subid.glued() != 0) && (subid.stereo() == 0)){
		        monodscounter++;
    			Detvector.push_back(*Iditer);
    			histos[Iditer->rawId()] = new TProfile(makename(*Iditer),makedescription(*Iditer),30,-30.,30.);
    			xaxis = histos[Iditer->rawId()]->GetXaxis();
    			xaxis->SetTitle("degree");
    			yaxis = histos[Iditer->rawId()]->GetYaxis();
    			yaxis->SetTitle("number of strips");
			}
		
		//Stereo detectors
				
		if((subid.glued() != 0) && (subid.stereo() == 1)){
		        stereocounter++;
    			Detvector.push_back(*Iditer);
    			histos[Iditer->rawId()] = new TProfile(makename(*Iditer),makedescription(*Iditer),30,-30.,30.);
    			xaxis = histos[Iditer->rawId()]->GetXaxis();
    			xaxis->SetTitle("degree");
    			yaxis = histos[Iditer->rawId()]->GetYaxis();
    			yaxis->SetTitle("number of strips");
			} 
	}
  } 
  
  //Summary histograms
  
  histos[1] = new TProfile("TIBL2_widthvsangle", "Cluster width vs track angle: TIB layer 2",30,-30.,30.);
  xaxis = histos[1]->GetXaxis();
  xaxis->SetTitle("degree");
  yaxis = histos[1]->GetYaxis();
  yaxis->SetTitle("number of strips");
  histos[2] = new TProfile("TIBL2_widthvsangle_int", "Cluster width vs track angle: TIB layer 2 INT",30,-30.,30.);
  xaxis = histos[2]->GetXaxis();
  xaxis->SetTitle("degree");
  yaxis = histos[2]->GetYaxis();
  yaxis->SetTitle("number of strips");
  histos[3] = new TProfile("TIBL2_widthvsangle_ext", "Cluster width vs track angle: TIB layer 2 EXT",30,-30.,30.);
  xaxis = histos[3]->GetXaxis();
  xaxis->SetTitle("degree");
  yaxis = histos[3]->GetYaxis();
  yaxis->SetTitle("number of strips");
  histos[4] = new TProfile("TIBL3_widthvsangle", "Cluster width vs track angle: TIB layer 3",30,-30.,30.);
  xaxis = histos[4]->GetXaxis();
  xaxis->SetTitle("degree");
  yaxis = histos[4]->GetYaxis();
  yaxis->SetTitle("number of strips");  
  histos[5] = new TProfile("TIBL3_widthvsangle_int", "Cluster width vs track angle: TIB layer 3 INT",30,-30.,30.);
  xaxis = histos[5]->GetXaxis();
  xaxis->SetTitle("degree");
  yaxis = histos[5]->GetYaxis();
  yaxis->SetTitle("number of strips");
  histos[6] = new TProfile("TIBL3_widthvsangle_ext", "Cluster width vs track angle: TIB layer 3 EXT",30,-30.,30.);
  xaxis = histos[6]->GetXaxis();
  xaxis->SetTitle("degree");
  yaxis = histos[6]->GetYaxis();
  yaxis->SetTitle("number of strips");
  
  histos[7] = new TProfile("TOBL3_widthvsangle", "Cluster width vs track angle: TOB layer 3",30,-30.,30.);
  xaxis = histos[7]->GetXaxis();
  xaxis->SetTitle("degree");
  yaxis = histos[7]->GetYaxis();
  yaxis->SetTitle("number of strips");
  histos[8] = new TProfile("TOBL4_widthvsangle", "Cluster width vs track angle: TOB layer 4",30,-30.,30.);
  xaxis = histos[8]->GetXaxis();
  xaxis->SetTitle("degree");
  yaxis = histos[8]->GetYaxis();
  yaxis->SetTitle("number of strips");
    
  if(!(conf_.getParameter<bool>("MTCCtrack"))){  //MTCCtrack FALSE
    histos[9] = new TProfile("TIBL1_widthvsangle", "Cluster width vs track angle: TIB layer 1",30,-30.,30.);
    xaxis = histos[9]->GetXaxis();
    xaxis->SetTitle("degree");
    yaxis = histos[9]->GetYaxis();
    yaxis->SetTitle("number of strips");
    histos[10] = new TProfile("TIBL1_widthvsangle_int", "Cluster width vs track angle: TIB layer 1 INT",30,-30.,30.);
    xaxis = histos[10]->GetXaxis();
    xaxis->SetTitle("degree");
    yaxis = histos[10]->GetYaxis();
    yaxis->SetTitle("number of strips");
    histos[11] = new TProfile("TIBL1_widthvsangle_ext", "Cluster width vs track angle: TIB layer 1 EXT",30,-30.,30.);
    xaxis = histos[11]->GetXaxis();
    xaxis->SetTitle("degree");
    yaxis = histos[11]->GetYaxis();
    yaxis->SetTitle("number of strips");
    histos[12] = new TProfile("TIBL4_widthvsangle", "Cluster width vs track angle: TIB layer 4",30,-30.,30.);
    xaxis = histos[12]->GetXaxis();
    xaxis->SetTitle("degree");
    yaxis = histos[12]->GetYaxis();
    yaxis->SetTitle("number of strips");
    histos[13] = new TProfile("TIBL4_widthvsangle_int", "Cluster width vs track angle: TIB layer 4 INT",30,-30.,30.);
    xaxis = histos[13]->GetXaxis();
    xaxis->SetTitle("degree");
    yaxis = histos[13]->GetYaxis();
    yaxis->SetTitle("number of strips");
    histos[14] = new TProfile("TIBL4_widthvsangle_ext", "Cluster width vs track angle: TIB layer 4 EXT",30,-30.,30.);
    xaxis = histos[14]->GetXaxis();
    xaxis->SetTitle("degree");
    yaxis = histos[14]->GetYaxis();
    yaxis->SetTitle("number of strips");
    
    histos[15] = new TProfile("TOBL1_widthvsangle", "Cluster width vs track angle: TOB layer 1",30,-30.,30.);
    xaxis = histos[15]->GetXaxis();
    xaxis->SetTitle("degree");
    yaxis = histos[15]->GetYaxis();
    yaxis->SetTitle("number of strips");
    histos[16] = new TProfile("TOBL2_widthvsangle", "Cluster width vs track angle: TOB layer 2",30,-30.,30.);
    xaxis = histos[16]->GetXaxis();
    xaxis->SetTitle("degree");
    yaxis = histos[16]->GetYaxis();
    yaxis->SetTitle("number of strips");
    histos[17] = new TProfile("TOBL5_widthvsangle", "Cluster width vs track angle: TOB layer 5",30,-30.,30.);
    xaxis = histos[17]->GetXaxis();
    xaxis->SetTitle("degree");
    yaxis = histos[17]->GetYaxis();
    yaxis->SetTitle("number of strips");
    histos[18] = new TProfile("TOBL6_widthvsangle", "Cluster width vs track angle: TOB layer 6",30,-30.,30.);
    xaxis = histos[18]->GetXaxis();
    xaxis->SetTitle("degree");
    yaxis = histos[18]->GetYaxis();
    yaxis->SetTitle("number of strips");
  }


  histos[19] = new TProfile("Charge_vs_Angle", "Cluster charge vs track angle: TOB layer 6",30,-30.,30.);
  xaxis = histos[19]->GetXaxis();
  xaxis->SetTitle("degree");
  yaxis = histos[19]->GetYaxis();
  yaxis->SetTitle("charge");

  poClusterChargeTH1F = new TH1F( "Cluster_Charge", "Cluster Charge",600,0.,300.);
  xaxis = poClusterChargeTH1F->GetXaxis();
  xaxis->SetTitle("Charge");

  {
    TH1D *poNewPlot;
    // Global
    poNewPlot = new TH1D( "NumberOfTrackClusterDivTotNumOfClusters", "Number of Track Clusters / Total number of Clusters",150,0.,1.5);
    poNewPlot->GetXaxis()->SetTitle( "TrackClusters / AllClusters");
    oGlobalPlots.push_back( poNewPlot);

    // TIB
    poNewPlot = new TH1D( "NumberOfClustersInTIBL2", "TIB L2: Number of clusters per layer",10,0.,10.);
    poNewPlot->GetXaxis()->SetTitle( "Clusters");
    oDetPlots[StripSubdetector::TIB].push_back( poNewPlot);

    poNewPlot = new TH1D( "NumberOfClustersInTIBL3", "TIB L3: Number of clusters per layer",10,0.,10.);
    poNewPlot->GetXaxis()->SetTitle( "Clusters");
    oDetPlots[StripSubdetector::TIB].push_back( poNewPlot);

    // TOB
    poNewPlot = new TH1D( "NumberOfClustersInTOBL3", "TOB L3: Number of clusters per layer",10,0.,10.);
    poNewPlot->GetXaxis()->SetTitle( "Clusters");
    oDetPlots[StripSubdetector::TOB].push_back( poNewPlot);

    poNewPlot = new TH1D( "NumberOfClustersInTOBL4", "TOB L4: Number of clusters per layer",10,0.,10.);
    poNewPlot->GetXaxis()->SetTitle( "Clusters");
    oDetPlots[StripSubdetector::TOB].push_back( poNewPlot);
  }
  
    
  //Directory hierarchy  
  histograms = new TDirectory("Histograms", "Histograms", "");
  summary = new TDirectory("Summary", "Summary", "");  
  
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
  
    
} 

// Virtual destructor needed.

MTCCNtupleMaker::~MTCCNtupleMaker() {  
  // delete poTrackTree;
}  

// Analyzer: Functions that gets called by framework every event

void MTCCNtupleMaker::analyze(const edm::Event& e, const edm::EventSetup& es)
{
  //m_oSiStripNoiseService.setESObjects( es);

  run       = e.id().run();
  event     = e.id().event();
  
  eventcounter+=1;
   
  using namespace edm;
  
  // Step A: Get Inputs 
  
  anglefinder_->init(e,es);
  
  trackhitmap trackhits;
  trackhitmap trackhitsXZ;
  trackhitmap trackhitsYZ;

  trklcldirmap oLclDirs;
  trkglbdirmap oGlbDirs;
  
  if(conf_.getParameter<bool>("MTCCtrack")){  //MTCCtrack TRUE
    
  //LogDebug("MTCCNtupleMaker::analyze")<<"MTCC - Getting tracks";
  
  edm::Handle<reco::TrackCollection> trackCollection;
  e.getByLabel( oTracksLabel_,
                oTracksProdInstName_,
                trackCollection);
  //e.getByType(trackCollection);

  if( bUseLTCDigis_) {
    // Extract Trigger Bits
    edm::Handle<LTCDigiCollection> oLTCDigis;
    e.getByType( oLTCDigis);

    // Now loop over all 6 triggers and save their values in corresponding
    // boolean variables :)
    if( 1 > oLTCDigis->size()) {
      LogDebug( "MTCCNtupleMaker::analyze")
	<< "[warning] More than one LTCDigis object stored in LTCDigiCollection";
    }

    for( LTCDigiCollection::const_iterator oITER = oLTCDigis->begin();
	 oITER != oLTCDigis->end();
	 ++oITER) {

      bTriggerDT   |= oITER->HasTriggered( 0);
      bTriggerCSC  |= oITER->HasTriggered( 1);
      bTriggerRBC1 |= oITER->HasTriggered( 2);
      bTriggerRBC2 |= oITER->HasTriggered( 3);
      bTriggerRPC  |= oITER->HasTriggered( 4);
    }
  }
    
  const reco::TrackCollection *tracks=trackCollection.product();
 
    //LogDebug("MTCCNtupleMaker::analyze")<<"MTCC - Getting seed";
    
    edm::Handle<TrajectorySeedCollection> seedcoll;
    e.getByLabel( oSeedsLabel_,
                  oSeedsProdInstName_,
                  seedcoll);
    //e.getByType(seedcoll);
    
    //LogDebug("MTCCNtupleMaker::analyze")<<"MTCC - Getting used rechit";
    
    if((*seedcoll).size()>0){
       if (tracks->size()>0){
       	            
        trackcounter+=tracks->size();
      
	reco::TrackCollection::const_iterator ibeg=trackCollection.product()->begin();
			
	hphi->Fill((*ibeg).outerPhi());
	hnhit->Fill((*ibeg).recHitsSize());
			
	std::vector<std::pair<const TrackingRecHit *,float> > tmphitangle=anglefinder_->findtrackangle((*(*seedcoll).begin()),tracks->front());
	std::vector<std::pair<const TrackingRecHit *,float> >::iterator tmpiter;
	
	trackhits[&(*ibeg)] = tmphitangle;
	trackhitsXZ[&(*ibeg)] = anglefinder_->getXZHitAngle();
	trackhitsYZ[&(*ibeg)] = anglefinder_->getYZHitAngle();
	oLclDirs[&(*ibeg)] = anglefinder_->getLocalDir();
	oGlbDirs[&(*ibeg)] = anglefinder_->getGlobalDir();
	
	for(tmpiter=tmphitangle.begin();tmpiter!=tmphitangle.end();tmpiter++){
	 hitcounter+=1;
	}
	
      }
    }
  }
  else{                                         //MTCCtrack FALSE
  
  //LogDebug("MTCCNtupleMaker::analyze")<<"Getting tracks";
  
  std::string src=conf_.getParameter<std::string>( "src" );
  
  edm::Handle<reco::TrackCollection> trackCollection;
  e.getByLabel(src, trackCollection);
  
  const reco::TrackCollection *tracks=trackCollection.product();
  reco::TrackCollection::const_iterator tciter;
  
    if(tracks->size()>0){
    
      trackcounter+=tracks->size();
          
      for(tciter=tracks->begin();tciter!=tracks->end();tciter++){
           
        std::vector<std::pair<const TrackingRecHit *,float> > tmphitangle=anglefinder_->findtrackangle(*tciter);
	std::vector<std::pair<const TrackingRecHit *,float> >::iterator tmpiter;
	
	hphi->Fill((*tciter).outerPhi());
	hnhit->Fill((*tciter).recHitsSize());

	trackhits[&(*tciter)] = tmphitangle;
	trackhitsXZ[&(*tciter)] = anglefinder_->getXZHitAngle();
	trackhitsYZ[&(*tciter)] = anglefinder_->getYZHitAngle();
	oLclDirs[&(*tciter)] = anglefinder_->getLocalDir();
	oGlbDirs[&(*tciter)] = anglefinder_->getGlobalDir();
				
	for(tmpiter=tmphitangle.begin();tmpiter!=tmphitangle.end();tmpiter++){
	  hitcounter+=1;
	}
      }
    }
  }

  // Get SiStripClusterInfos
  edm::Handle<edm::DetSetVector<SiStripClusterInfo> > oDSVClusterInfos;
  e.getByLabel( "siStripClusterInfoProducer", oDSVClusterInfos);

  // Get SiStripDigis
  edm::Handle<edm::DetSetVector<SiStripDigi> > oDSVDigis;
  if( oSiStripDigisProdInstName_.size()) {
    e.getByLabel( oSiStripDigisLabel_.c_str(), oSiStripDigisProdInstName_.c_str(), oDSVDigis);
  } else {
    e.getByLabel( oSiStripDigisLabel_.c_str(), oDSVDigis);
  }
      
  std::map<uint32_t, int> oProcessedClusters;

  int nTrackClusters = 0;
  
  if(trackhits.size()!=0){
       
    trackhitmap::iterator mapiter;
    
    for(mapiter = trackhits.begin(); mapiter != trackhits.end(); mapiter++){
          
      momentum=-99;
      pt=-99;
      charge=-99;
           
      momentum = (*mapiter).first->p();
      pt = (*mapiter).first->pt();
      eta      = mapiter->first->eta();
      phi      = mapiter->first->phi();
      chi2     = mapiter->first->chi2();
      ndof     = mapiter->first->ndof();
      normchi2 = mapiter->first->normalizedChi2();
      hitspertrack = mapiter->first->recHitsSize();
      charge = (*mapiter).first->charge();
     
      poTrackTree->Fill();
                
      std::vector<std::pair<const TrackingRecHit *,float> > hitangle = (*mapiter).second;
      std::vector<std::pair<const TrackingRecHit *,float> >::iterator hitsiter;

    
    if(hitangle.size()!=0){
      // create reference to YZAngles
      TrackLocalAngleNew::HitAngleAssociation &roHitAngleAssocXZ = trackhitsXZ[mapiter->first];
      TrackLocalAngleNew::HitAngleAssociation &roHitAngleAssocYZ = trackhitsYZ[mapiter->first];
      TrackLocalAngleNew::HitLclDirAssociation &roLclDirAssoc    = oLclDirs[mapiter->first];
      TrackLocalAngleNew::HitGlbDirAssociation &roGlbDirAssoc    = oGlbDirs[mapiter->first];
      
    
    int nHitNum = 0;
    for(hitsiter=hitangle.begin();hitsiter!=hitangle.end();hitsiter++){

      TrackLocalAngleNew::HitAngleAssociation::reference hitsrefXZ = roHitAngleAssocXZ[nHitNum];
      TrackLocalAngleNew::HitAngleAssociation::reference hitsrefYZ = roHitAngleAssocYZ[nHitNum];
      TrackLocalAngleNew::HitLclDirAssociation::reference roLclDir = roLclDirAssoc[nHitNum];
      TrackLocalAngleNew::HitGlbDirAssociation::reference roGlbDir = roGlbDirAssoc[nHitNum];
      ++nHitNum;
        
      module=-99;
      type=-99;
      layer=-99;
      wheel=-99;
      string=-99;
      size=-99;
      extint=-99;
      bwfw=-99;
      rod=-99;
      angle=-9999;
      stereocorrection=-9999;
      localmagfield=-99;
      sign = -99;
      monostereo=-99;
      clusternoise=-99;
      clusterbarycenter=-99;
      clusterseednoise=-99;
    
      const SiStripRecHit2D* hit=dynamic_cast<const SiStripRecHit2D*>(hitsiter->first);
      dLclX = hit->localPosition().x(); 
      dLclY = hit->localPosition().y(); 
      dLclZ = hit->localPosition().z(); 

      GlobalPoint oRecHitGlobalPos = tracker->idToDet( hit->geographicalId())->toGlobal( hit->localPosition());
      dGlbX = oRecHitGlobalPos.x();
      dGlbY = oRecHitGlobalPos.y();
      dGlbZ = oRecHitGlobalPos.z();

      const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > cluster=hit->cluster();

      std::vector<SiStripDigi> oDigis = 
	oDSVDigis->operator[]( cluster->geographicalId()).data;

      size=(cluster->amplitudes()).size();

      clustereta = getClusterEta( cluster->amplitudes(),
				  cluster->firstStrip(),
				  oDigis);
      clustercrosstalk = getClusterCrossTalk( cluster->amplitudes(),
				              cluster->firstStrip(),
				              oDigis);

      std::vector<SiStripClusterInfo> oClusterInfos = 
	oDSVClusterInfos->operator[]( cluster->geographicalId()).data;

      for( std::vector<SiStripClusterInfo>::iterator oIter = 
	     oClusterInfos.begin();
	   oIter != oClusterInfos.end();
	   ++oIter) {
	
	if( oIter->firstStrip() == cluster->firstStrip()) {
	  // ClusterInfo matched given cluster
	  clusterpos    = oIter->position();
	  clusterchg	= oIter->charge();
	  clusterchgl   = oIter->chargeL();
	  clusterchgr   = oIter->chargeR();
	  clusternoise  = oIter->noise();
	  clusterbarycenter = cluster->barycenter();
	  clusterseednoise = oIter->stripNoises().operator[]( ( int) cluster->barycenter() - cluster->firstStrip());
	  clustermaxchg = oIter->maxCharge();
	  bTrack = true;

	  poClusterChargeTH1F->Fill( clusterchg);

	  // Mark Current ClusterInfo as processed
	  // [Note: One Module might hold several clusters (!)]
	  oProcessedClusters[cluster->geographicalId()] = oIter->firstStrip();

	  break;
	} else {
	  // ClusterInfo was found but didn't match given cluster
	}
      }


      StripSubdetector detid=(StripSubdetector)hit->geographicalId();
      
      type = detid.subdetId();
      
      module = (hit->geographicalId()).rawId();
      
      angle=hitsiter->second;
      float angleXZ=hitsrefXZ.second;
      float angleYZ=hitsrefYZ.second;
      
      monostereo=detid.stereo();
      
      //Local Magnetic Field
      
      const GeomDet *geomdet = tracker->idToDet(hit->geographicalId());
      LocalPoint localp(0,0,0);
      const GlobalPoint globalp = (geomdet->surface()).toGlobal(localp);
      GlobalVector globalmagdir = magfield->inTesla(globalp);
      localmagdir = (geomdet->surface()).toLocal(globalmagdir);
      localmagfield = localmagdir.mag();
      
      if(localmagfield != 0.){
      
      //Sign correction for TIB and TOB
      
      if((detid.subdetId() == int (StripSubdetector::TIB)) || (detid.subdetId() == int (StripSubdetector::TOB))){
      
      LocalVector ylocal(0,1,0);
      
      float normprojection = (localmagdir * ylocal)/(localmagfield);
      
      if(normprojection>0){sign = 1;}
      if(normprojection<0){sign = -1;}
                  
      //Stereocorrection applied in TrackLocalAngleNew
      
      if((detid.stereo()==1) && (normprojection == 0.)){
      LogDebug("MTCCNtupleMaker::analyze")<<"Error: TIB|TOB YBprojection = 0";
      }
      
      if((detid.stereo()==1) && (normprojection != 0.)){
      stereocorrection = 1/normprojection;
      stereocorrection*=sign;
      
      float tg = tan((angle*TMath::Pi())/180);
      tg*=stereocorrection;
      angle = atan(tg)*180/TMath::Pi();
      
      tg = tan( angleXZ);
      tg*=stereocorrection;
      angleXZ = atan( tg);

      tg = tan( angleYZ);
      tg*=stereocorrection;
      angleYZ = atan( tg);
           
      }
      
      angle   *= sign;
      angleXZ *= sign;
      angleYZ *= sign;
       
      }
      }


      //Filling histograms
            
      histos[module]->Fill(angle,size);
      
      LogDebug("MTCCNtupleMaker::analyze")<<"Module histogram filled";
                      
      //Summary histograms
	
	if(detid.subdetId() == int (StripSubdetector::TIB)){
		TIBDetId TIBid=TIBDetId(hit->geographicalId());
		
		extint = TIBid.string()[1];
		string = TIBid.string()[2];
		bwfw= TIBid.string()[0];
		layer = TIBid.layer() + mtcctibcorr;
		
		if(layer == 1){
		histos[9]->Fill(angle,size);
		if(TIBid.stereo()==0){
		htaTIBL1mono->Fill(angle);}
		if(TIBid.stereo()==1){
		htaTIBL1stereo->Fill(angle);}
		
		if(TIBid.string()[1]==0){//int
		histos[10]->Fill(angle,size);
		}
		if(TIBid.string()[1]==1){//ext
		histos[11]->Fill(angle,size);
		}
		}
		
		if(layer == 2){
		histos[1]->Fill(angle,size);
		if(TIBid.stereo()==0){
		htaTIBL2mono->Fill(angle);}
		if(TIBid.stereo()==1){
		htaTIBL2stereo->Fill(angle);}
		
		if(TIBid.string()[1]==0){//int
		histos[2]->Fill(angle,size);
		}
		if(TIBid.string()[1]==1){//ext
		histos[3]->Fill(angle,size);
		}
		}
		
		if(layer == 3){
		histos[4]->Fill(angle,size);
		htaTIBL3->Fill(angle);
		if(TIBid.string()[1]==0){//int
		histos[5]->Fill(angle,size);
		}
		if(TIBid.string()[1]==1){//ext
		histos[6]->Fill(angle,size);
		}
		}
		
		if(layer == 4){
		histos[12]->Fill(angle,size);
		htaTIBL4->Fill(angle);
		if(TIBid.string()[1]==0){//int
		histos[13]->Fill(angle,size);
		}
		if(TIBid.string()[1]==1){//ext
		histos[14]->Fill(angle,size);
		}
		}
				
		}
		
	if(detid.subdetId() == int (StripSubdetector::TOB)){
		TOBDetId TOBid=TOBDetId(hit->geographicalId());
		
		layer = TOBid.layer() + mtcctobcorr;
		rod = TOBid.rod()[1];
		bwfw = TOBid.rod()[0];
		
		if(layer == 1){
		histos[15]->Fill(angle,size);		
		if(TOBid.stereo()==0){//mono
		htaTOBL1mono->Fill(angle);
		}
		if(TOBid.stereo()==1){//stereo
		htaTOBL1stereo->Fill(angle);
		}
		}
		
		if(layer == 2){
		histos[16]->Fill(angle,size);		
		if(TOBid.stereo()==0){//mono
		htaTOBL2mono->Fill(angle);
		}
		if(TOBid.stereo()==1){//stereo
		htaTOBL2stereo->Fill(angle);
		}
		}
		
		if(layer == 3){
		histos[7]->Fill(angle,size);
		htaTOBL3->Fill(angle);}
		
		if(layer == 4){
		histos[8]->Fill(angle,size);
		htaTOBL4->Fill(angle);}
		
		if(layer == 5){
		histos[17]->Fill(angle,size);
		htaTOBL5->Fill(angle);}
		
		if(layer == 6){
		histos[18]->Fill(angle,size);
		htaTOBL6->Fill(angle);}		
				
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
		}
      
	const GeomDetUnit * stripdet=(const GeomDetUnit*)tracker->idToDetUnit(detid);
	
	const StripTopology& topol=(StripTopology&)stripdet->topology();
	
	float thickness=stripdet->specificSurface().bounds().thickness();
	
	float proj=tan(angle)*thickness/topol.pitch();
	
	//Filling WidthvsTrackProjection histogram
	
	hwvst->Fill(proj,size); 
	
	//Filling Tree
	
        MTCCNtupleMakerTree->Fill();
	
	LogDebug("MTCCNtupleMaker::analyze")<<"Tree Filled";
	
	}
	nTrackClusters += hitangle.size();
      }	

    }
  }

  std::map<SiSubDet, std::map<unsigned char, int> > oClustersPerLayer;
  int nTotClusters = nTrackClusters;

  // Now work out all Cluster that were left unprocessed thus do not belong to
  // tracks

  // Loop over modules
  for( edm::DetSetVector<SiStripClusterInfo>::const_iterator oDSVIter = oDSVClusterInfos->begin();
       oDSVIter != oDSVClusterInfos->end();
       ++oDSVIter) {

    StripSubdetector oStripSubdet( oDSVIter->id);
    unsigned char ucLayer;
    switch( oStripSubdet.subdetId()) {
      case StripSubdetector::TIB:
	{
	  TIBDetId oTIBDetId( oStripSubdet);

	  ucLayer  = oTIBDetId.layer() + mtcctibcorr;
	  break;
	}
      case StripSubdetector::TID:
	break;
      case StripSubdetector::TOB:
	{
	  TOBDetId oTOBDetId( oStripSubdet);

	  ucLayer = oTOBDetId.layer() + mtcctobcorr;
	  break;
	}
      case StripSubdetector::TEC:
	break;
    }
    oClustersPerLayer[SiSubDet( oStripSubdet.subdetId())][ucLayer] += oDSVIter->data.size();

    if( oProcessedClusters.end() ==
	  oProcessedClusters.find( oDSVIter->id)) {

      // Extract ClusterInfos collection for given module
      std::vector<SiStripClusterInfo> oClusterInfos = oDSVIter->data;
      std::vector<SiStripDigi> oDigis = 
	oDSVDigis->operator[]( oDSVIter->id).data;

      nTotClusters += oClusterInfos.size();

      // Loop over ClusterInfos collection
      for( std::vector<SiStripClusterInfo>::iterator oIter = 
	     oClusterInfos.begin();
	   oIter != oClusterInfos.end();
	   ++oIter) {
	
	// Check if given ClusterInfo was already processed
	if( oProcessedClusters[oDSVIter->id] !=
	      oIter->firstStrip()) {

	  // Set default values to the rest of the variables
	  pt	       = -99;
	  eta	       = -99;
	  phi	       = -99;
	  charge       = -99;
	  momentum     = -99;
	  normchi2     = -99;
	  hitspertrack = -99;
	  dLclX	       = -99;
	  dLclY	       = -99;
	  dLclZ	       = -99;
	  dGlbX	       = -99;
	  dGlbY	       = -99;
	  dGlbZ	       = -99;

	  rod		     = -99;	  // Only for TOB
	  sign	     = -99;	  // Not needed
	  wheel	     = -99;	  // Only for TID, TEC
	  angle	     = -9999; // Only for track
	  layer	     = -99;	  // There is no layer for TID and TEC
	  string	     = -99;	  // String exists only for TIB
	  extint	     = -99;	  // Olny for TIB
	  localmagfield    = -99;	  // Not needed
	  stereocorrection = -9999; // Only for track
	  clusterbarycenter = -99;
	  clusterseednoise = -99;

	  StripSubdetector oStripSubdet( oIter->geographicalId());
	  type	   = oStripSubdet.subdetId();
	  module	   = oStripSubdet.rawId();
	  monostereo = oStripSubdet.stereo();

	  switch( type) {
	    case StripSubdetector::TIB:
	      {
		TIBDetId oTIBDetId( oStripSubdet);

		bwfw   = oTIBDetId.string()[0];
		layer  = oTIBDetId.layer() + mtcctibcorr;
		extint = oTIBDetId.string()[1];
		string = oTIBDetId.string()[2];
		break;
	      }
	    case StripSubdetector::TID:
	      {
		TIDDetId oTIDDetId( oStripSubdet);

		bwfw  = oTIDDetId.module()[0];
		wheel = oTIDDetId.wheel();
		break;
	      }
	    case StripSubdetector::TOB:
	      {
		TOBDetId oTOBDetId( oStripSubdet);

		rod   = oTOBDetId.rod()[1];
		bwfw  = oTOBDetId.rod()[0];
		layer = oTOBDetId.layer() + mtcctobcorr;
		break;
	      }
	    case StripSubdetector::TEC:
	      {
		TECDetId oTECDetId( oStripSubdet);

		bwfw  = oTECDetId.petal()[0];
		wheel = oTECDetId.wheel();
		break;
	      }
	  }

	  size	      = (int) oIter->width(); // cluster width

	  // ClusterInfo was not processed yet
	  clusterpos    = oIter->position();
	  clusterchg    = oIter->charge();
	  clusterchgl   = oIter->chargeL();
	  clusterchgr   = oIter->chargeR();
	  clusternoise  = oIter->noise();
	  clustermaxchg = oIter->maxCharge();
	  clustereta    = getClusterEta( oIter->stripAmplitudes(), 
					 oIter->firstStrip(),
					 oDigis);

          clustercrosstalk = getClusterCrossTalk( oIter->stripAmplitudes(),
				                  oIter->firstStrip(),
				                  oDigis);

	  bTrack = false;

	  MTCCNtupleMakerTree->Fill();
	}
      }
    }
  }

  oGlobalPlots[0]->Fill( ( 1.0 * nTrackClusters) / nTotClusters);

  if( 0 < oClustersPerLayer[StripSubdetector::TIB][2])
    oDetPlots[StripSubdetector::TIB][0]->Fill( oClustersPerLayer[StripSubdetector::TIB][2]);

  if( 0 < oClustersPerLayer[StripSubdetector::TIB][3])
    oDetPlots[StripSubdetector::TIB][1]->Fill( oClustersPerLayer[StripSubdetector::TIB][3]);

  if( 0 < oClustersPerLayer[StripSubdetector::TOB][3])
    oDetPlots[StripSubdetector::TOB][0]->Fill( oClustersPerLayer[StripSubdetector::TOB][3]);

  if( 0 < oClustersPerLayer[StripSubdetector::TOB][4])
    oDetPlots[StripSubdetector::TOB][1]->Fill( oClustersPerLayer[StripSubdetector::TOB][4]);
}

//Makename function

const char* MTCCNtupleMaker::makename(DetId detid){
  
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
    int layer = TOBid.layer() + mtcctobcorr; 
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

const char* MTCCNtupleMaker::makedescription(DetId detid){
  
  std::string name;
  
  name="Cluster width vs track angle (";
  
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
    int layer = TOBid.layer() + mtcctobcorr; 
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

void MTCCNtupleMaker::endJob(){


  std::vector<DetId>::iterator Iditer;
  
  //Histograms fit
  
  int histonum = Detvector.size();

  for(Iditer=Detvector.begin(); Iditer!=Detvector.end(); Iditer++){
    
    fitfunc->SetParameters(0, 0, 1);
    histos[Iditer->rawId()]->Fit("fitfunc","E","",-20, 20);
    
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
    fits[Iditer->rawId()]->min     = fitfunction->Eval(fits[Iditer->rawId()]->p0);
    
  }
  
  int n;
  int nmax = 19;
  
  if(conf_.getParameter<bool>("MTCCtrack")){  //MTCCtrack TRUE
  nmax=9;}
  
  for(n=1; n<nmax; n++){
    
    fitfunc->SetParameters(0, 0, 1);
    histos[n]->Fit("fitfunc","E","",-20, 20);
    
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
    fits[n]->min     = fitfunction->Eval(fits[n]->p0);
    
  }
  
  //File with fit parameters  
  
  ofstream fit;
  fit.open("fit.txt");
  
  if(conf_.getParameter<bool>("MTCCtrack")){  //MTCCtrack TRUE
  fit<<endl<<">>> MTCCtrack = TRUE"<<endl<<endl;
  }else{
  fit<<endl<<">>> MTCCtrack = FALSE"<<endl<<endl;
  }
  fit<<">>> TOTAL EVENT = "<<eventcounter<<endl;
  fit<<">>> NUMBER OF RECHITS = "<<hitcounter<<endl;
  fit<<">>> NUMBER OF TRACKS = "<<trackcounter<<endl<<endl;
  fit<<">>> NUMBER OF DETECTOR HISTOGRAMS = "<<histonum<<endl;
  fit<<">>> NUMBER OF MONO SINGLE SIDED DETECTORS = "<<monosscounter<<endl;
  fit<<">>> NUMBER OF MONO DOUBLE SIDED DETECTORS = "<<monodscounter<<endl;
  fit<<">>> NUMBER OF STEREO DETECTORS = "<<stereocounter<<endl<<endl;
    
    if(!(conf_.getParameter<bool>("MTCCtrack"))){  //MTCCtrack FALSE
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TIB LAYER 1 -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[9]->chi2)/(fits[9]->ndf)<<endl;
    fit<<"NdF        = "<<fits[9]->ndf<<endl;
    fit<<"p0 = "<<fits[9]->p0<<"     err p0 = "<<fits[9]->errp0<<endl;
    fit<<"p1 = "<<fits[9]->p1<<"     err p1 = "<<fits[9]->errp1<<endl;
    fit<<"p2 = "<<fits[9]->p2<<"     err p2 = "<<fits[9]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[9]->p0<<"  +-  "<<fits[9]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[9]->min<<endl<<endl;
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TIB LAYER 1 INT -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[10]->chi2)/(fits[10]->ndf)<<endl;
    fit<<"NdF        = "<<fits[10]->ndf<<endl;
    fit<<"p0 = "<<fits[10]->p0<<"     err p0 = "<<fits[10]->errp0<<endl;
    fit<<"p1 = "<<fits[10]->p1<<"     err p1 = "<<fits[10]->errp1<<endl;
    fit<<"p2 = "<<fits[10]->p2<<"     err p2 = "<<fits[10]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[10]->p0<<"  +-  "<<fits[10]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[10]->min<<endl<<endl;
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TIB LAYER 1 EXT -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[11]->chi2)/(fits[11]->ndf)<<endl;
    fit<<"NdF        = "<<fits[11]->ndf<<endl;
    fit<<"p0 = "<<fits[11]->p0<<"     err p0 = "<<fits[11]->errp0<<endl;
    fit<<"p1 = "<<fits[11]->p1<<"     err p1 = "<<fits[11]->errp1<<endl;
    fit<<"p2 = "<<fits[11]->p2<<"     err p2 = "<<fits[11]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[11]->p0<<"  +-  "<<fits[11]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[11]->min<<endl<<endl;
    }
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TIB LAYER 2 -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[1]->chi2)/(fits[1]->ndf)<<endl;
    fit<<"NdF        = "<<fits[1]->ndf<<endl;
    fit<<"p0 = "<<fits[1]->p0<<"     err p0 = "<<fits[1]->errp0<<endl;
    fit<<"p1 = "<<fits[1]->p1<<"     err p1 = "<<fits[1]->errp1<<endl;
    fit<<"p2 = "<<fits[1]->p2<<"     err p2 = "<<fits[1]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[1]->p0<<"  +-  "<<fits[1]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[1]->min<<endl<<endl;
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TIB LAYER 2 INT -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[2]->chi2)/(fits[2]->ndf)<<endl;
    fit<<"NdF        = "<<fits[2]->ndf<<endl;
    fit<<"p0 = "<<fits[2]->p0<<"     err p0 = "<<fits[2]->errp0<<endl;
    fit<<"p1 = "<<fits[2]->p1<<"     err p1 = "<<fits[2]->errp1<<endl;
    fit<<"p2 = "<<fits[2]->p2<<"     err p2 = "<<fits[2]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[2]->p0<<"  +-  "<<fits[2]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[2]->min<<endl<<endl;
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TIB LAYER 2 EXT -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[3]->chi2)/(fits[3]->ndf)<<endl;
    fit<<"NdF        = "<<fits[3]->ndf<<endl;
    fit<<"p0 = "<<fits[3]->p0<<"     err p0 = "<<fits[3]->errp0<<endl;
    fit<<"p1 = "<<fits[3]->p1<<"     err p1 = "<<fits[3]->errp1<<endl;
    fit<<"p2 = "<<fits[3]->p2<<"     err p2 = "<<fits[3]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[3]->p0<<"  +-  "<<fits[3]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[3]->min<<endl<<endl;
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TIB LAYER 3 -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[4]->chi2)/(fits[4]->ndf)<<endl;
    fit<<"NdF        = "<<fits[4]->ndf<<endl;
    fit<<"p0 = "<<fits[4]->p0<<"     err p0 = "<<fits[4]->errp0<<endl;
    fit<<"p1 = "<<fits[4]->p1<<"     err p1 = "<<fits[4]->errp1<<endl;
    fit<<"p2 = "<<fits[4]->p2<<"     err p2 = "<<fits[4]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[4]->p0<<"  +-  "<<fits[4]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[4]->min<<endl<<endl;
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TIB LAYER 3 INT -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[5]->chi2)/(fits[5]->ndf)<<endl;
    fit<<"NdF        = "<<fits[5]->ndf<<endl;
    fit<<"p0 = "<<fits[5]->p0<<"     err p0 = "<<fits[5]->errp0<<endl;
    fit<<"p1 = "<<fits[5]->p1<<"     err p1 = "<<fits[5]->errp1<<endl;
    fit<<"p2 = "<<fits[5]->p2<<"     err p2 = "<<fits[5]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[5]->p0<<"  +-  "<<fits[5]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[5]->min<<endl<<endl;
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TIB LAYER 3 EXT -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[6]->chi2)/(fits[6]->ndf)<<endl;
    fit<<"NdF        = "<<fits[6]->ndf<<endl;
    fit<<"p0 = "<<fits[6]->p0<<"     err p0 = "<<fits[6]->errp0<<endl;
    fit<<"p1 = "<<fits[6]->p1<<"     err p1 = "<<fits[6]->errp1<<endl;
    fit<<"p2 = "<<fits[6]->p2<<"     err p2 = "<<fits[6]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[6]->p0<<"  +-  "<<fits[6]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[6]->min<<endl<<endl;
    
    if(!(conf_.getParameter<bool>("MTCCtrack"))){  //MTCCtrack FALSE 
    fit<<endl<<"--------------------------- SUMMARY FIT: TIB LAYER 4 -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[12]->chi2)/(fits[12]->ndf)<<endl;
    fit<<"NdF        = "<<fits[12]->ndf<<endl;
    fit<<"p0 = "<<fits[12]->p0<<"     err p0 = "<<fits[12]->errp0<<endl;
    fit<<"p1 = "<<fits[12]->p1<<"     err p1 = "<<fits[12]->errp1<<endl;
    fit<<"p2 = "<<fits[12]->p2<<"     err p2 = "<<fits[12]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[12]->p0<<"  +-  "<<fits[12]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[12]->min<<endl<<endl;
      
    fit<<endl<<"--------------------------- SUMMARY FIT: TIB LAYER 4 INT -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[13]->chi2)/(fits[13]->ndf)<<endl;
    fit<<"NdF        = "<<fits[13]->ndf<<endl;
    fit<<"p0 = "<<fits[13]->p0<<"     err p0 = "<<fits[13]->errp0<<endl;
    fit<<"p1 = "<<fits[13]->p1<<"     err p1 = "<<fits[13]->errp1<<endl;
    fit<<"p2 = "<<fits[13]->p2<<"     err p2 = "<<fits[13]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[13]->p0<<"  +-  "<<fits[13]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[13]->min<<endl<<endl;
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TIB LAYER 4 EXT -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[14]->chi2)/(fits[14]->ndf)<<endl;
    fit<<"NdF        = "<<fits[14]->ndf<<endl;
    fit<<"p0 = "<<fits[14]->p0<<"     err p0 = "<<fits[14]->errp0<<endl;
    fit<<"p1 = "<<fits[14]->p1<<"     err p1 = "<<fits[14]->errp1<<endl;
    fit<<"p2 = "<<fits[14]->p2<<"     err p2 = "<<fits[14]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[14]->p0<<"  +-  "<<fits[14]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[14]->min<<endl<<endl;
    
             
    fit<<endl<<"--------------------------- SUMMARY FIT: TOB LAYER 1 -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[15]->chi2)/(fits[15]->ndf)<<endl;
    fit<<"NdF        = "<<fits[15]->ndf<<endl;
    fit<<"p0 = "<<fits[15]->p0<<"     err p0 = "<<fits[15]->errp0<<endl;
    fit<<"p1 = "<<fits[15]->p1<<"     err p1 = "<<fits[15]->errp1<<endl;
    fit<<"p2 = "<<fits[15]->p2<<"     err p2 = "<<fits[15]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[15]->p0<<"  +-  "<<fits[15]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[15]->min<<endl<<endl;
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TOB LAYER 2 -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[16]->chi2)/(fits[16]->ndf)<<endl;
    fit<<"NdF        = "<<fits[16]->ndf<<endl;
    fit<<"p0 = "<<fits[16]->p0<<"     err p0 = "<<fits[16]->errp0<<endl;
    fit<<"p1 = "<<fits[16]->p1<<"     err p1 = "<<fits[16]->errp1<<endl;
    fit<<"p2 = "<<fits[16]->p2<<"     err p2 = "<<fits[16]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[16]->p0<<"  +-  "<<fits[16]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[16]->min<<endl<<endl;
    }
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TOB LAYER 3 -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[7]->chi2)/(fits[7]->ndf)<<endl;
    fit<<"NdF        = "<<fits[7]->ndf<<endl;
    fit<<"p0 = "<<fits[7]->p0<<"     err p0 = "<<fits[7]->errp0<<endl;
    fit<<"p1 = "<<fits[7]->p1<<"     err p1 = "<<fits[7]->errp1<<endl;
    fit<<"p2 = "<<fits[7]->p2<<"     err p2 = "<<fits[7]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[7]->p0<<"  +-  "<<fits[7]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[7]->min<<endl<<endl;
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TOB LAYER 4 -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[8]->chi2)/(fits[8]->ndf)<<endl;
    fit<<"NdF        = "<<fits[8]->ndf<<endl;
    fit<<"p0 = "<<fits[8]->p0<<"     err p0 = "<<fits[8]->errp0<<endl;
    fit<<"p1 = "<<fits[8]->p1<<"     err p1 = "<<fits[8]->errp1<<endl;
    fit<<"p2 = "<<fits[8]->p2<<"     err p2 = "<<fits[8]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[8]->p0<<"  +-  "<<fits[8]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[8]->min<<endl<<endl;
    
    if(!(conf_.getParameter<bool>("MTCCtrack"))){  //MTCCtrack FALSE
    fit<<endl<<"--------------------------- SUMMARY FIT: TOB LAYER 5 -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[17]->chi2)/(fits[17]->ndf)<<endl;
    fit<<"NdF        = "<<fits[17]->ndf<<endl;
    fit<<"p0 = "<<fits[17]->p0<<"     err p0 = "<<fits[17]->errp0<<endl;
    fit<<"p1 = "<<fits[17]->p1<<"     err p1 = "<<fits[17]->errp1<<endl;
    fit<<"p2 = "<<fits[17]->p2<<"     err p2 = "<<fits[17]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[17]->p0<<"  +-  "<<fits[17]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[17]->min<<endl<<endl;
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TOB LAYER 6 -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[18]->chi2)/(fits[18]->ndf)<<endl;
    fit<<"NdF        = "<<fits[18]->ndf<<endl;
    fit<<"p0 = "<<fits[18]->p0<<"     err p0 = "<<fits[18]->errp0<<endl;
    fit<<"p1 = "<<fits[18]->p1<<"     err p1 = "<<fits[18]->errp1<<endl;
    fit<<"p2 = "<<fits[18]->p2<<"     err p2 = "<<fits[18]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[18]->p0<<"  +-  "<<fits[18]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[18]->min<<endl<<endl;
    }
    
  for(Iditer=Detvector.begin(); Iditer!=Detvector.end(); Iditer++){
    
    fit<<endl<<"-------------------------- MODULE HISTOGRAM FIT ------------------------"<<endl<<endl;
    fit<<makedescription(*Iditer)<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[Iditer->rawId()]->chi2)/(fits[Iditer->rawId()]->ndf)<<endl;
    fit<<"NdF        = "<<fits[Iditer->rawId()]->ndf<<endl;
    fit<<"p0 = "<<fits[Iditer->rawId()]->p0<<"     err p0 = "<<fits[Iditer->rawId()]->errp0<<endl;
    fit<<"p1 = "<<fits[Iditer->rawId()]->p1<<"     err p1 = "<<fits[Iditer->rawId()]->errp1<<endl;
    fit<<"p2 = "<<fits[Iditer->rawId()]->p2<<"     err p2 = "<<fits[Iditer->rawId()]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[Iditer->rawId()]->p0<<"  +-  "<<fits[Iditer->rawId()]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[Iditer->rawId()]->min<<endl<<endl;
    
  }
  
  fit.close();
    
  //Set directories
  
  for(n=1;n<nmax;n++){
  histos[n]->SetDirectory(summary);}
  
  histos[19]->SetDirectory( summary);

  poClusterChargeTH1F->SetDirectory( summary); 
  hphi->SetDirectory(summary);
  hnhit->SetDirectory(summary);

  oGlobalPlots[0]->SetDirectory( summary);
  for( std::vector<TH1D *>::iterator oIter = oDetPlots[StripSubdetector::TIB].begin();
       oIter != oDetPlots[StripSubdetector::TIB].end();
       ++oIter) {
    ( *oIter)->SetDirectory( summary);
  }

  for( std::vector<TH1D *>::iterator oIter = oDetPlots[StripSubdetector::TOB].begin();
       oIter != oDetPlots[StripSubdetector::TOB].end();
       ++oIter) {
    ( *oIter)->SetDirectory( summary);
  }
  
  if(!(conf_.getParameter<bool>("MTCCtrack"))){  //MTCCtrack FALSE 
  htaTIBL1mono->SetDirectory(summary);
  htaTIBL1stereo->SetDirectory(summary);
  }
  
  htaTIBL2mono->SetDirectory(summary);
  htaTIBL2stereo->SetDirectory(summary);
  htaTIBL3->SetDirectory(summary);
  
  if(!(conf_.getParameter<bool>("MTCCtrack"))){  //MTCCtrack FALSE 
  htaTIBL4->SetDirectory(summary);
  htaTOBL1mono->SetDirectory(summary);
  htaTOBL1stereo->SetDirectory(summary);
  htaTOBL2mono->SetDirectory(summary);
  htaTOBL2stereo->SetDirectory(summary);
  }
  
  htaTOBL3->SetDirectory(summary);
  htaTOBL4->SetDirectory(summary);
  
  if(!(conf_.getParameter<bool>("MTCCtrack"))){  //MTCCtrack FALSE 
  htaTOBL5->SetDirectory(summary);
  htaTOBL6->SetDirectory(summary);
  }
  
  hwvst->SetDirectory(summary);  
  
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
    
    int correctedlayer = TOBid.layer() + mtcctobcorr;
    
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

// ClusterEta = SignalL / ( SignalL + SignalR)
// where:
//   SignalL and SignalR are two strips with the highest amplitudes. 
//   SignalL - is the strip with the smaller strip number
//   SignalR - with the highest strip number accordingly
// @arguments
//   roSTRIP_AMPLITUDES	 vector of strips ADC counts in cluster
//   rnFIRST_STRIP	 cluster first strip shift whithin module
//   roDIGIS		 vector of digis within current module
// @return
//   int  ClusterEta or -99 on error
double 
  MTCCNtupleMaker::getClusterEta( const std::vector<uint16_t> &roSTRIP_AMPLITUDES,
				      const int			  &rnFIRST_STRIP,
				      const DigisVector		  &roDIGIS) const {
  // Given value is used to separate non-physical values
  // [Example: cluster with empty amplitudes vector]
  double dClusterEta = -99;

  // Cluster eta calculation
  int anMaxSignal[2][2];

  // Null array before using it
  for( int i = 0; 2 > i; ++i) {
    for( int j = 0; 2 > j; ++j) {
      anMaxSignal[i][j] = 0;
    }
  }
	
  // Find two strips with highest amplitudes
  // i is a stip number within module
  for( int i = 0, nSize = roSTRIP_AMPLITUDES.size(); nSize > i; ++i) {
    int nCurCharge = roSTRIP_AMPLITUDES[i];

    if( nCurCharge > anMaxSignal[1][1]) {
      anMaxSignal[0][0] = anMaxSignal[1][0];
      anMaxSignal[0][1] = anMaxSignal[1][1];
      // Convert to global strip number within module
      anMaxSignal[1][0] = i + rnFIRST_STRIP; 
      anMaxSignal[1][1] = nCurCharge;
    } else if( nCurCharge > anMaxSignal[0][1]) {
      // Convert to global strip number within module
      anMaxSignal[0][0] = i + rnFIRST_STRIP;
      anMaxSignal[0][1] = nCurCharge;
    }
  }
  
  if( ( anMaxSignal[1][1] + anMaxSignal[0][1]) != 0) {
    if( anMaxSignal[0][0] > anMaxSignal[1][0]) {
      // anMaxSignal[1] is Left one
      dClusterEta = ( 1.0 * anMaxSignal[1][1]) / ( anMaxSignal[1][1] + 
						   anMaxSignal[0][1]);
    } else if( 0 == anMaxSignal[0][0] && 
	       0 == anMaxSignal[0][1]) {

      // One Strip cluster: check for Digis
      DigisVector::const_iterator oITER( roDIGIS.begin());
      for( ;
	   oITER != roDIGIS.end() && oITER->strip() != anMaxSignal[1][0];
	   ++oITER) {}

      // Check if Digi for given cluster strip was found
      if( oITER != roDIGIS.end()) {

	// Check if previous neighbouring strip exists
	DigisVector::const_iterator oITER_PREV( roDIGIS.end());
	if( oITER != roDIGIS.begin() &&
	    ( oITER->strip() - 1) == ( oITER - 1)->strip()) {
	  // There is previous strip specified :)
	  oITER_PREV = oITER - 1;
	}

	// Check if next neighbouring strip exists
	DigisVector::const_iterator oITER_NEXT( roDIGIS.end());
	if( oITER != roDIGIS.end() &&
	    oITER != ( roDIGIS.end() - 1) &&
	    ( oITER->strip() + 1) == ( oITER + 1)->strip()) {
	  // There is previous strip specified :)
	  oITER_NEXT = oITER + 1;
	}

	if( oITER_PREV != oITER_NEXT) {
	  if( oITER_PREV != roDIGIS.end() && oITER_NEXT != roDIGIS.end()) {
	    // Both Digis are specified
	    // Now Pick the one with max amplitude
	    if( oITER_PREV->adc() > oITER_NEXT->adc()) {
	      dClusterEta = ( 1.0 * oITER_PREV->adc()) / ( oITER_PREV->adc() + 
							   anMaxSignal[1][1]);
	    } else {
	      dClusterEta = ( 1.0 * anMaxSignal[1][1]) / ( oITER_NEXT->adc() + 
							   anMaxSignal[1][1]);
	    }
	  } else if( oITER_PREV != roDIGIS.end()) {
	    // only Prev digi is specified
	    dClusterEta = ( 1.0 * oITER_PREV->adc()) / ( oITER_PREV->adc() + 
							 anMaxSignal[1][1]);
	  } else {
	    // only Next digi is specified
	    dClusterEta = ( 1.0 * anMaxSignal[1][1]) / ( oITER_NEXT->adc() + 
							 anMaxSignal[1][1]);
	  }
	} else {
	  // PREV and NEXT iterators point to the end of DIGIs vector. 
	  // Consequently it is assumed there are no neighbouring digis at all
	  // for given cluster. It is obvious why ClusterEta should be Zero.
	  // [Hint: take a look at the case [0][0] < [1][0] ]
	  dClusterEta = 0;
	} // End check if any neighbouring digi is specified
      } else {
	// Digi for given Clusters strip was not found
	dClusterEta = 0;
      } // end check if Digi for given cluster strip was found
    } else {
      // anMaxSignal[0] is Left one
      dClusterEta = ( 1.0 * anMaxSignal[0][1]) / ( anMaxSignal[1][1] + 
						   anMaxSignal[0][1]);
    }
  } 

  return dClusterEta;
}

// ClusterCrosstalk = ( SignalL + SignalR) / SignalSeed
//   Extremely useful for croostalk determination that should be used in
// MonteCarlo clusters simulation. In MonteCarlo each cluster affects 
// neighbouring ones, the charge is divided:
//
//  Simulated             Digitized
//    Signal                Signal
//       +
//       +
//       +                    +
//       +                    +
//       +          =>        +
//       +                    +
//       +                    +
//       +                +   +   +
//       +                +   +   +
//  N-1  N  N+1          N-1  N  N+1
//
//  Strip   Crosstalk
//   N-1     x * SignalN
//   N       ( 1 - 2 * x) * SignalN
//   N+1     x * SignalN
//
// @arguments
//   roSTRIP_AMPLITUDES	 vector of strips ADC counts in cluster
//   rnFIRST_STRIP	 cluster first strip shift whithin module
//   roDIGIS		 vector of digis within current module
// @return
//   int  ClusterCrosstalk or -99 on error
double 
  MTCCNtupleMaker::getClusterCrossTalk( const std::vector<uint16_t> 
                                          &roSTRIP_AMPLITUDES,
				        const int         &rnFIRST_STRIP,
				        const DigisVector &roDIGIS) const {
  // Given value is used to separate non-physical values
  // [Example: cluster with empty amplitudes vector]
  double dClusterCrossTalk = -99;

  switch( roSTRIP_AMPLITUDES.size()) {
    case 1: {
      // One Strip cluster: try to find corresponding Digi
      DigisVector::const_iterator oITER( roDIGIS.begin());
      for( ;
	   oITER != roDIGIS.end() && oITER->strip() != roSTRIP_AMPLITUDES[0];
	   ++oITER) {}

      // Check if Digi for given cluster strip was found
      if( oITER != roDIGIS.end()) {

	// Check if previous neighbouring strip exists
	DigisVector::const_iterator oITER_PREV( roDIGIS.end());
	if( oITER != roDIGIS.begin() &&
	    ( oITER->strip() - 1) == ( oITER - 1)->strip()) {
	  // There is previous strip specified :)
	  oITER_PREV = oITER - 1;
	}

	// Check if next neighbouring strip exists
	DigisVector::const_iterator oITER_NEXT( roDIGIS.end());
	if( oITER != roDIGIS.end() &&
	    oITER != ( roDIGIS.end() - 1) &&
	    ( oITER->strip() + 1) == ( oITER + 1)->strip()) {
	  // There is previous strip specified :)
	  oITER_NEXT = oITER + 1;
	}

        // Now check if both neighbouring digis exist
	if( oITER_PREV != roDIGIS.end() && oITER_NEXT != roDIGIS.end()) {
	  // Both Digis are specified
	  // Now Pick the one with max amplitude
	  dClusterCrossTalk = 
	    calculateClusterCrossTalk( oITER_PREV->adc(),
	                               roSTRIP_AMPLITUDES[0],
				       oITER_NEXT->adc());
	} // End check if both neighbouring digis exist
      } // end check if Digi for given cluster strip was found
    }
    case 3: {
      dClusterCrossTalk = calculateClusterCrossTalk( roSTRIP_AMPLITUDES[0],
                                                     roSTRIP_AMPLITUDES[1],
						     roSTRIP_AMPLITUDES[2]);
    }
    default:
      break;
  }



  return dClusterCrossTalk;
}

// Calculate Cluster CrossTalk:
// ClusterCrosstalk = ( SignalL + SignalR) / SignalSeed
// @arguments
//   rdADC_STRIPL  ADC in left strip
//   rnADC_STRIP   ADC in Center strip
//   rnADC_STRIPR  ADC in right strip
// @return
//   double  Calculated crosstalk or -99 on error 
double 
  MTCCNtupleMaker::calculateClusterCrossTalk( const double &rdADC_STRIPL,
                                              const int    &rnADC_STRIP,
					      const int    &rnADC_STRIPR) const 
{
  // Check if neigbouring strips have signals amplitudes within some
  // error
  return ( abs( rdADC_STRIPL - rnADC_STRIPR) < 
             dCROSS_TALK_ERR * ( rdADC_STRIPL + rnADC_STRIPR) / 2 &&
             0 < rnADC_STRIP ? 
	   ( rdADC_STRIPL + rnADC_STRIPR) / rnADC_STRIP : 
	   -99);
}

/*
double 
  MTCCNtupleMaker::getClusterCrossTalk( const std::vector<uint16_t> 
                                          &roSTRIP_AMPLITUDES,
				        const int         &rnFIRST_STRIP,
				        const DigisVector &roDIGIS) const {
  // Given value is used to separate non-physical values
  // [Example: cluster with empty amplitudes vector]
  double dClusterCrossTalk = -99;

  switch( roSTRIP_AMPLITUDES.size()) {
    case 1: {
      // One Strip cluster: try to find corresponding Digi
      DigisVector::const_iterator oITER( roDIGIS.begin());
      for( ;
	   oITER != roDIGIS.end() && oITER->strip() != roSTRIP_AMPLITUDES[0];
	   ++oITER) {}

      // Check if Digi for given cluster strip was found
      if( oITER != roDIGIS.end()) {

	// Check if previous neighbouring strip exists
	DigisVector::const_iterator oITER_PREV( roDIGIS.end());
	if( oITER != roDIGIS.begin() &&
	    ( oITER->strip() - 1) == ( oITER - 1)->strip()) {
	  // There is previous strip specified :)
	  oITER_PREV = oITER - 1;
	}

	// Check if next neighbouring strip exists
	DigisVector::const_iterator oITER_NEXT( roDIGIS.end());
	if( oITER != roDIGIS.end() &&
	    oITER != ( roDIGIS.end() - 1) &&
	    ( oITER->strip() + 1) == ( oITER + 1)->strip()) {
	  // There is previous strip specified :)
	  oITER_NEXT = oITER + 1;
	}

        // Now check if both neighbouring digis exist and there is no
	// anything in N-2 and N+2 to make sure neighbouring digis were
	// created from central one
	if( oITER_PREV != roDIGIS.end() && 
	    oITER_NEXT != roDIGIS.end() &&
	    !( oITER_PREV != roDIGIS.begin() && 
	       ( oITER_PREV - 1)->strip() == oITER_PREV->strip() - 1) && 
	    !( oITER_NEXT != roDIGIS.end() - 1 &&
	       ( oITER_NEXT + 1)->strip() == oITER_NEXT->strip() + 1)) {

	  // Both Digis are specified
	  // Now Pick the one with max amplitude
	  dClusterCrossTalk = 
	    calculateClusterCrossTalk( oITER_PREV->adc(),
	                               roSTRIP_AMPLITUDES[0],
				       oITER_NEXT->adc());
	} // End check if both neighbouring digis exist
      } // end check if Digi for given cluster strip was found
    }
    case 3: {
      // Try to find Digi that corresponds to central strip
      DigisVector::const_iterator oITER( roDIGIS.begin());
      for( ;
	   oITER != roDIGIS.end() && oITER->strip() != roSTRIP_AMPLITUDES[1];
	   ++oITER) {}

      // Check if Digi for given cluster strip was found
      if( oITER != roDIGIS.end()) {
	DigisVector::const_iterator oITER_N_MINUS_2( oITER);
	for( int i = 2; i > 0; --i) {
	  if( oITER_N_MINUS_2 == roDIGIS.begin()) {
	    // There is no N-2 Digi
	    oITER_N_MINUS_2 = roDIGIS.end();
	    break;
	  }
	  --oITER_N_MINUS_2;
	}

	DigisVector::const_iterator oITER_N_PLUS_2( oITER);
	for( int i = 2; oITER_N_PLUS_2 != roDIGIS.end() && i > 0; --i) {
	  ++oITER_N_PLUS_2;
	}

	// Check if there is no N-2/N+2 Digi specified
	if( ( ( oITER_N_MINUS_2 == roDIGIS.end() ||
	        oITER_N_MINUS_2->strip() + 1 != ( oITER_N_MINUS_2 + 1)->strip()) &&
	      ( oITER_N_PLUS_2 == roDIGIS.end() ||
	        oITER_N_PLUS_2->strip() - 1 != ( oITER_N_PLUS_2 - 1)->strip())) ) {
	  dClusterCrossTalk = calculateClusterCrossTalk( roSTRIP_AMPLITUDES[0],
							 roSTRIP_AMPLITUDES[1],
							 roSTRIP_AMPLITUDES[2]);
	}
      }
    }
    default:
      break;
  }



  return dClusterCrossTalk;
}
*/
