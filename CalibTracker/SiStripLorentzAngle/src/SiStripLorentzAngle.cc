
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <TMath.h>
#include "CalibTracker/SiStripLorentzAngle/interface/SiStripLorentzAngle.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/GlobalVector.h"
#include "Geometry/Vector/interface/LocalVector.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DMatchedLocalPos.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"


#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"

using namespace std;
SiStripLorentzAngle::SiStripLorentzAngle(edm::ParameterSet const& conf) : 
  conf_(conf), filename_(conf.getParameter<std::string>("fileName"))
{
  anglefinder_=new  TrackLocalAngle(conf);
}

void SiStripLorentzAngle::beginJob(const edm::EventSetup& c){

  hFile = new TFile (filename_.c_str(), "RECREATE" );
  hphi = new TH1F("hphi","Phi distribution",20,-3.14,3.14);
  hnhit = new TH1F("hnhit","Number of Hits per Track ",5,2.5,7.5);
  htaTIBL2 = new TH1F("htal2","Track angle (TIB L2)",30,-60.,60.);
  htaTIBL3 = new TH1F("htal3","Track angle (TIB L3)",30,-60.,60.);
  htaTOB1 = new TH1F("htar1","Track angle (TOB 1)",30,-60.,60.);
  htaTOB2 = new TH1F("htar2","Track angle (TOB 2)",30,-60.,60.);
  hwvsaTIBL2 = new TProfile("hwvsatibl2","Cluster width vs track angle (TIB L2)",30,-60.,60.);
  hwvsaTIBL2intstr1 = new TProfile("hwvsatibl2intstr1","Cluster width vs track angle (TIB L2 INT STRING 1)",30,-60.,60.);
  hwvsaTIBL2intstr2 = new TProfile("hwvsatibl2intstr2","Cluster width vs track angle (TIB L2 INT STRING 2)",30,-60.,60.);
  hwvsaTIBL2extstr1 = new TProfile("hwvsatibl2extstr1","Cluster width vs track angle (TIB L2 EXT STRING 1)",30,-60.,60.);
  hwvsaTIBL2extstr2 = new TProfile("hwvsatibl2extstr2","Cluster width vs track angle (TIB L2 EXT STRING 2)",30,-60.,60.);
  hwvsaTIBL2extstr3 = new TProfile("hwvsatibl2extstr3","Cluster width vs track angle (TIB L2 EXT STRING 3)",30,-60.,60.);
  hwvsaTIBL3 = new TProfile("hwvsatibl3","Cluster width vs track angle (TIB L3)",30,-60.,60.);
  hwvsaTIBL3intstr1 = new TProfile("hwvsatibl3intstr1","Cluster width vs track angle (TIB L3 INT STRING 1)",30,-60.,60.);
  hwvsaTIBL3intstr2 = new TProfile("hwvsatibl3intstr2","Cluster width vs track angle (TIB L3 INT STRING 2)",30,-60.,60.);
  hwvsaTIBL3intstr3 = new TProfile("hwvsatibl3intstr3","Cluster width vs track angle (TIB L3 INT STRING 3)",30,-60.,60.);
  hwvsaTIBL3intstr4 = new TProfile("hwvsatibl3intstr4","Cluster width vs track angle (TIB L3 INT STRING 4)",30,-60.,60.);
  hwvsaTIBL3intstr5 = new TProfile("hwvsatibl3intstr5","Cluster width vs track angle (TIB L3 INT STRING 5)",30,-60.,60.);
  hwvsaTIBL3intstr6 = new TProfile("hwvsatibl3intstr6","Cluster width vs track angle (TIB L3 INT STRING 6)",30,-60.,60.);
  hwvsaTIBL3intstr7 = new TProfile("hwvsatibl3intstr7","Cluster width vs track angle (TIB L3 INT STRING 7)",30,-60.,60.);
  hwvsaTIBL3intstr8 = new TProfile("hwvsatibl3intstr8","Cluster width vs track angle (TIB L3 INT STRING 8)",30,-60.,60.);
  hwvsaTIBL3extstr1 = new TProfile("hwvsatibl3extstr1","Cluster width vs track angle (TIB L3 EXT STRING 1)",30,-60.,60.);
  hwvsaTIBL3extstr2 = new TProfile("hwvsatibl3extstr2","Cluster width vs track angle (TIB L3 EXT STRING 2)",30,-60.,60.);
  hwvsaTIBL3extstr3 = new TProfile("hwvsatibl3extstr3","Cluster width vs track angle (TIB L3 EXT STRING 3)",30,-60.,60.);
  hwvsaTIBL3extstr4 = new TProfile("hwvsatibl3extstr4","Cluster width vs track angle (TIB L3 EXT STRING 4)",30,-60.,60.);
  hwvsaTIBL3extstr5 = new TProfile("hwvsatibl3extstr5","Cluster width vs track angle (TIB L3 EXT STRING 5)",30,-60.,60.);
  hwvsaTIBL3extstr6 = new TProfile("hwvsatibl3extstr6","Cluster width vs track angle (TIB L3 EXT STRING 6)",30,-60.,60.);
  hwvsaTIBL3extstr7 = new TProfile("hwvsatibl3extstr7","Cluster width vs track angle (TIB L3 EXT STRING 7)",30,-60.,60.);
  hwvsaTOB = new TProfile("hwvsatob","Cluster width vs track angle (TOB)",30,-60.,60.);
  hwvsaTOBL1 = new TProfile("hwvsatobl1","Cluster width vs track angle (TOB L1)",30,-60.,60.);
  hwvsaTOBL1rod1 = new TProfile("hwvsatobl1rod1","Cluster width vs track angle (TOB L1 ROD 1)",30,-60.,60.);
  hwvsaTOBL1rod2 = new TProfile("hwvsatobl1rod2","Cluster width vs track angle (TOB L1 ROD 2)",30,-60.,60.);
  hwvsaTOBL2 = new TProfile("hwvsatobl2","Cluster width vs track angle (TOB L2)",30,-60.,60.);
  hwvsaTOBL2rod1 = new TProfile("hwvsatobl2rod1","Cluster width vs track angle (TOB L2 ROD 1)",30,-60.,60.);
  hwvsaTOBL2rod2 = new TProfile("hwvsatobl2rod2","Cluster width vs track angle (TOB L2 ROD 2)",30,-60.,60.);
  hwvst = new TProfile("hwvst","Cluster width vs track projection ",30,-60.,60.);
  fitfunc = new TF1("fitfunc","[1]*((x-[0])^2)+[2]",-60,60);
  
  
  SiStripLorentzAngleTree = new TTree("SiStripLorentzAngleTree","SiStrip LorentzAngle tree");
  SiStripLorentzAngleTree->Branch("run", &run, "run/I");
  SiStripLorentzAngleTree->Branch("event", &event, "event/I");
  SiStripLorentzAngleTree->Branch("module", &module, "module/I");
  SiStripLorentzAngleTree->Branch("type", &type, "type/I");
  SiStripLorentzAngleTree->Branch("layer", &layer, "layer/I");
  SiStripLorentzAngleTree->Branch("string", &string, "string/I");
  SiStripLorentzAngleTree->Branch("extint", &extint, "extint/I");
  SiStripLorentzAngleTree->Branch("size", &size, "size/I");
  SiStripLorentzAngleTree->Branch("angle", &angle, "angle/F");
  
  eventcounter = 0;
  eventnumber = -1;
  trackcounter = 0;
  
  edm::ESHandle<TrackerGeometry> estracker;
  c.get<TrackerDigiGeometryRecord>().get(estracker);
  tracker=&(* estracker);
  //edm::ESHandle<MagneticField> esmagfield;
  //c.get<IdealMagneticFieldRecord>().get(esmagfield);
  //magfield=&(*esmagfield);
}
// Virtual destructor needed.
SiStripLorentzAngle::~SiStripLorentzAngle() {  }  

// Functions that gets called by framework every event
void SiStripLorentzAngle::analyze(const edm::Event& e, const edm::EventSetup& es)
{
  module=-1;
  type=-1;
  layer=-1;
  string=-1;
  size=-1;
  extint=-1;
  angle=-9999;
  run       = e.id().run();
  event     = e.id().event();
  
  if(event != eventnumber){
  eventcounter+=1;
  eventnumber = event;
  }
  
  using namespace edm;
  // Step A: Get Inputs 
  anglefinder_->init(e,es);
  LogDebug("SiStripLorentzAngle::analyze")<<"Getting tracks";
  
  edm::Handle<reco::TrackCollection> trackCollection;
    //  //    event.getByLabel("trackp", trackCollection);
  e.getByType(trackCollection);
  //edm::Handle<reco::TrackCollection> trackCollection;
  //e.getByLabel("cosmictrackfinder",trackCollection);

  LogDebug("SiStripLorentzAngle::analyze")<<"Getting seed";

 edm::Handle<TrajectorySeedCollection> seedcoll;
  e.getByType(seedcoll);
  LogDebug("SiStripLorentzAngle::analyze")<<"Getting used rechit";

  edm::Handle<TrackingRecHitCollection> trackrechitCollection;
  e.getByType(trackrechitCollection);
  std::vector<std::pair<const TrackingRecHit * ,float> > hitangle;
  const reco::TrackCollection *tracks=trackCollection.product();
  if((*seedcoll).size()>0){
    if (tracks->size()>0){
    trackcounter+=1;
    reco::TrackCollection::const_iterator ibeg=trackCollection.product()->begin();
      LogDebug("SiStripLorentzAngle::analyze")<<"Filling histograms";

      hphi->Fill((*ibeg).outerPhi());
      hnhit->Fill((*ibeg).recHitsSize() );
      LogDebug("SiStripLorentzAngle::analyze")<<"Finding TSOS";
      hitangle =anglefinder_->findtrackangle((*(*seedcoll).begin()),*trackrechitCollection);
      std::vector<std::pair<const TrackingRecHit * ,float> >::iterator iter;
      for(iter=hitangle.begin();iter!=hitangle.end();iter++){
        const SiStripRecHit2DLocalPos* hit=dynamic_cast<const SiStripRecHit2DLocalPos*>(iter->first);
	const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > cluster=hit->cluster();
	size=(cluster->amplitudes()).size();
	StripSubdetector detid=(StripSubdetector)hit->geographicalId();
	type = detid.subdetId();
	module = (hit->geographicalId()).rawId();
	angle=iter->second;
	if(detid.subdetId() == int (StripSubdetector::TIB)){
	TIBDetId id=TIBDetId(detid);
	layer=id.layer();
	string=id.string()[2];
	extint=id.string()[1];
	
	if(id.layer()==1){
	  htaTIBL2->Fill(angle);
	  hwvsaTIBL2->Fill(angle,size);
	  if(extint==0){
	  if(string==1){
	  hwvsaTIBL2intstr1->Fill(angle,size);}
	  if(string==2){
	  hwvsaTIBL2intstr2->Fill(angle,size);}
	  }
	  if(extint==1){
	  if(string==1){
	  hwvsaTIBL2extstr1->Fill(angle,size);}
	  if(string==2){
	  hwvsaTIBL2extstr2->Fill(angle,size);}
	  if(string==3){
	  hwvsaTIBL2extstr3->Fill(angle,size);}
	  }
	}
	else if(id.layer()==2){
	  htaTIBL3->Fill(angle);
	  hwvsaTIBL3->Fill(angle,size);
	  if(extint==0){
	  if(string==1){
	  hwvsaTIBL3intstr1->Fill(angle,size);}
	  if(string==2){
	  hwvsaTIBL3intstr2->Fill(angle,size);}
	  if(string==3){
	  hwvsaTIBL3intstr3->Fill(angle,size);}
	  if(string==4){
	  hwvsaTIBL3intstr4->Fill(angle,size);}
	  if(string==5){
	  hwvsaTIBL3intstr5->Fill(angle,size);}
	  if(string==6){
	  hwvsaTIBL3intstr6->Fill(angle,size);}
	  if(string==7){
	  hwvsaTIBL3intstr7->Fill(angle,size);}
	  if(string==8){
	  hwvsaTIBL3intstr8->Fill(angle,size);}
	  }
	  if(extint==1){
	  if(string==1){
	  hwvsaTIBL3extstr1->Fill(angle,size);}
	  if(string==2){
	  hwvsaTIBL3extstr2->Fill(angle,size);}
	  if(string==3){
	  hwvsaTIBL3extstr3->Fill(angle,size);}
	  if(string==4){
	  hwvsaTIBL3extstr4->Fill(angle,size);}
	  if(string==5){
	  hwvsaTIBL3extstr5->Fill(angle,size);}
	  if(string==6){
	  hwvsaTIBL3extstr6->Fill(angle,size);}
	  if(string==7){
	  hwvsaTIBL3extstr7->Fill(angle,size);}
	  }
	}
      }
      else if(detid.subdetId() == int (StripSubdetector::TOB)){
      	TOBDetId id=TOBDetId(detid);
	layer=id.layer();
	string=id.rod()[1];
	extint=-1;
	
	if(id.layer()==1){
	  htaTOB1->Fill(angle);
	  hwvsaTOBL1->Fill(angle,size);
	  if (string == 1){
	  hwvsaTOBL1rod1->Fill(angle,size);}
	  if (string == 2){
	  hwvsaTOBL1rod2->Fill(angle,size);}
	}
	else if(id.layer()==2){
	  htaTOB2->Fill(angle);
	  hwvsaTOBL2->Fill(angle,size);
	  if (string == 1){
	  hwvsaTOBL2rod1->Fill(angle,size);}
	  if (string == 2){
	  hwvsaTOBL2rod2->Fill(angle,size);}
       }
       hwvsaTOB->Fill(angle,size);
     }
     
     const GeomDetUnit * stripdet=(const GeomDetUnit*)tracker->idToDetUnit(detid);
     const StripTopology& topol=(StripTopology&)stripdet->topology();
     float thickness=stripdet->specificSurface().bounds().thickness();
     float proj=tan(angle)*thickness/topol.pitch();
         
     //	hwvsa->Fill(angle,size);
     hwvst->Fill(proj,size);
     SiStripLorentzAngleTree->Fill();
      
      }
      
    }
  }
}

void SiStripLorentzAngle::endJob(){

//Fit histogram TIB Layer 2
  fitfunc->SetParameters(0,0,1);
  hwvsaTIBL2->Fit("fitfunc","E","",-20,20);
  TF1 *fitTIB2 = hwvsaTIBL2->GetFunction("fitfunc");
  chi2TIB2 = fitTIB2->GetChisquare();
  p0TIB2 = fitTIB2->GetParameter(0);
  err0TIB2 = fitTIB2->GetParError(0);
  p1TIB2 = fitTIB2->GetParameter(1);
  err1TIB2 = fitTIB2->GetParError(1);
  p2TIB2 = fitTIB2->GetParameter(2);
  err2TIB2 = fitTIB2->GetParError(2);
  minTIB2 = fitTIB2->Eval(p0TIB2);
  
//Fit histogram TIB Layer 2 String int 1
  fitfunc->SetParameters(0,0,1);
  hwvsaTIBL2intstr1->Fit("fitfunc","E","",-20,20);
  TF1 *fitTIB2intstr1 = hwvsaTIBL2intstr1->GetFunction("fitfunc");
  chi2TIB2intstr1 = fitTIB2intstr1->GetChisquare();
  p0TIB2intstr1 = fitTIB2intstr1->GetParameter(0);
  err0TIB2intstr1 = fitTIB2intstr1->GetParError(0);
  p1TIB2intstr1 = fitTIB2intstr1->GetParameter(1);
  err1TIB2intstr1 = fitTIB2intstr1->GetParError(1);
  p2TIB2intstr1 = fitTIB2intstr1->GetParameter(2);
  err2TIB2intstr1 = fitTIB2intstr1->GetParError(2);
  minTIB2intstr1 = fitTIB2intstr1->Eval(p0TIB2);
  
//Fit histogram TIB Layer 2 String int 2
  fitfunc->SetParameters(0,0,1);
  hwvsaTIBL2intstr2->Fit("fitfunc","E","",-20,20);
  TF1 *fitTIB2intstr2 = hwvsaTIBL2intstr2->GetFunction("fitfunc");
  chi2TIB2intstr2 = fitTIB2intstr2->GetChisquare();
  p0TIB2intstr2 = fitTIB2intstr2->GetParameter(0);
  err0TIB2intstr2 = fitTIB2intstr2->GetParError(0);
  p1TIB2intstr2 = fitTIB2intstr2->GetParameter(1);
  err1TIB2intstr2 = fitTIB2intstr2->GetParError(1);
  p2TIB2intstr2 = fitTIB2intstr2->GetParameter(2);
  err2TIB2intstr2 = fitTIB2intstr2->GetParError(2);
  minTIB2intstr2 = fitTIB2intstr2->Eval(p0TIB2);
  
//Fit histogram TIB Layer 2 String ext 1
  fitfunc->SetParameters(0,0,1);
  hwvsaTIBL2extstr1->Fit("fitfunc","E","",-20,20);
  TF1 *fitTIB2extstr1 = hwvsaTIBL2extstr1->GetFunction("fitfunc");
  chi2TIB2extstr1 = fitTIB2extstr1->GetChisquare();
  p0TIB2extstr1 = fitTIB2extstr1->GetParameter(0);
  err0TIB2extstr1 = fitTIB2extstr1->GetParError(0);
  p1TIB2extstr1 = fitTIB2extstr1->GetParameter(1);
  err1TIB2extstr1 = fitTIB2extstr1->GetParError(1);
  p2TIB2extstr1 = fitTIB2extstr1->GetParameter(2);
  err2TIB2extstr1 = fitTIB2extstr1->GetParError(2);
  minTIB2extstr1 = fitTIB2extstr1->Eval(p0TIB2);
  
//Fit histogram TIB Layer 2 String ext 2
  fitfunc->SetParameters(0,0,1);
  hwvsaTIBL2extstr2->Fit("fitfunc","E","",-20,20);
  TF1 *fitTIB2extstr2 = hwvsaTIBL2extstr2->GetFunction("fitfunc");
  chi2TIB2extstr2 = fitTIB2extstr2->GetChisquare();
  p0TIB2extstr2 = fitTIB2extstr2->GetParameter(0);
  err0TIB2extstr2 = fitTIB2extstr2->GetParError(0);
  p1TIB2extstr2 = fitTIB2extstr2->GetParameter(1);
  err1TIB2extstr2 = fitTIB2extstr2->GetParError(1);
  p2TIB2extstr2 = fitTIB2extstr2->GetParameter(2);
  err2TIB2extstr2 = fitTIB2extstr2->GetParError(2);
  minTIB2extstr2 = fitTIB2extstr2->Eval(p0TIB2);
  
//Fit histogram TIB Layer 2 String ext 3
  fitfunc->SetParameters(0,0,1);
  hwvsaTIBL2extstr3->Fit("fitfunc","E","",-20,20);
  TF1 *fitTIB2extstr3 = hwvsaTIBL2extstr3->GetFunction("fitfunc");
  chi2TIB2extstr3 = fitTIB2extstr3->GetChisquare();
  p0TIB2extstr3 = fitTIB2extstr3->GetParameter(0);
  err0TIB2extstr3 = fitTIB2extstr3->GetParError(0);
  p1TIB2extstr3 = fitTIB2extstr3->GetParameter(1);
  err1TIB2extstr3 = fitTIB2extstr3->GetParError(1);
  p2TIB2extstr3 = fitTIB2extstr3->GetParameter(2);
  err2TIB2extstr3 = fitTIB2extstr3->GetParError(2);
  minTIB2extstr3 = fitTIB2extstr3->Eval(p0TIB2);
  
//Fit histogram TIB Layer 3
  fitfunc->SetParameters(0,0,1);
  hwvsaTIBL3->Fit("fitfunc","E","",-22,22);
  TF1 *fitTIB3 = hwvsaTIBL3->GetFunction("fitfunc");
  chi2TIB3 = fitTIB3->GetChisquare();
  p0TIB3 = fitTIB3->GetParameter(0);
  err0TIB3 = fitTIB3->GetParError(0);
  p1TIB3 = fitTIB3->GetParameter(1);
  err1TIB3 = fitTIB3->GetParError(1);
  p2TIB3 = fitTIB3->GetParameter(2);
  err2TIB3 = fitTIB3->GetParError(2);
  minTIB3 = fitTIB3->Eval(p0TIB3);
  
//Fit histogram TIB Layer 3 String int 1
  fitfunc->SetParameters(0,0,1);
  hwvsaTIBL3intstr1->Fit("fitfunc","E","",-22,22);
  TF1 *fitTIB3intstr1 = hwvsaTIBL3intstr1->GetFunction("fitfunc");
  chi2TIB3intstr1 = fitTIB3intstr1->GetChisquare();
  p0TIB3intstr1 = fitTIB3intstr1->GetParameter(0);
  err0TIB3intstr1 = fitTIB3intstr1->GetParError(0);
  p1TIB3intstr1 = fitTIB3intstr1->GetParameter(1);
  err1TIB3intstr1 = fitTIB3intstr1->GetParError(1);
  p2TIB3intstr1 = fitTIB3intstr1->GetParameter(2);
  err2TIB3intstr1 = fitTIB3intstr1->GetParError(2);
  minTIB3intstr1 = fitTIB3intstr1->Eval(p0TIB3);
  
//Fit histogram TIB Layer 3 String int 2
  fitfunc->SetParameters(0,0,1);
  hwvsaTIBL3intstr2->Fit("fitfunc","E","",-22,22);
  TF1 *fitTIB3intstr2 = hwvsaTIBL3intstr2->GetFunction("fitfunc");
  chi2TIB3intstr2 = fitTIB3intstr2->GetChisquare();
  p0TIB3intstr2 = fitTIB3intstr2->GetParameter(0);
  err0TIB3intstr2 = fitTIB3intstr2->GetParError(0);
  p1TIB3intstr2 = fitTIB3intstr2->GetParameter(1);
  err1TIB3intstr2 = fitTIB3intstr2->GetParError(1);
  p2TIB3intstr2 = fitTIB3intstr2->GetParameter(2);
  err2TIB3intstr2 = fitTIB3intstr2->GetParError(2);
  minTIB3intstr2 = fitTIB3intstr2->Eval(p0TIB3);
  
//Fit histogram TIB Layer 3 String int 3
  fitfunc->SetParameters(0,0,1);
  hwvsaTIBL3intstr3->Fit("fitfunc","E","",-22,22);
  TF1 *fitTIB3intstr3 = hwvsaTIBL3intstr3->GetFunction("fitfunc");
  chi2TIB3intstr3 = fitTIB3intstr3->GetChisquare();
  p0TIB3intstr3 = fitTIB3intstr3->GetParameter(0);
  err0TIB3intstr3 = fitTIB3intstr3->GetParError(0);
  p1TIB3intstr3 = fitTIB3intstr3->GetParameter(1);
  err1TIB3intstr3 = fitTIB3intstr3->GetParError(1);
  p2TIB3intstr3 = fitTIB3intstr3->GetParameter(2);
  err2TIB3intstr3 = fitTIB3intstr3->GetParError(2);
  minTIB3intstr3 = fitTIB3intstr3->Eval(p0TIB3);
  
  //Fit histogram TIB Layer 3 String int 4
  fitfunc->SetParameters(0,0,1);
  hwvsaTIBL3intstr4->Fit("fitfunc","E","",-22,22);
  TF1 *fitTIB3intstr4 = hwvsaTIBL3intstr4->GetFunction("fitfunc");
  chi2TIB3intstr4 = fitTIB3intstr4->GetChisquare();
  p0TIB3intstr4 = fitTIB3intstr4->GetParameter(0);
  err0TIB3intstr4 = fitTIB3intstr4->GetParError(0);
  p1TIB3intstr4 = fitTIB3intstr4->GetParameter(1);
  err1TIB3intstr4 = fitTIB3intstr4->GetParError(1);
  p2TIB3intstr4 = fitTIB3intstr4->GetParameter(2);
  err2TIB3intstr4 = fitTIB3intstr4->GetParError(2);
  minTIB3intstr4 = fitTIB3intstr4->Eval(p0TIB3);
  
//Fit histogram TIB Layer 3 String int 5
  fitfunc->SetParameters(0,0,1);
  hwvsaTIBL3intstr5->Fit("fitfunc","E","",-22,22);
  TF1 *fitTIB3intstr5 = hwvsaTIBL3intstr5->GetFunction("fitfunc");
  chi2TIB3intstr5 = fitTIB3intstr5->GetChisquare();
  p0TIB3intstr5 = fitTIB3intstr5->GetParameter(0);
  err0TIB3intstr5 = fitTIB3intstr5->GetParError(0);
  p1TIB3intstr5 = fitTIB3intstr5->GetParameter(1);
  err1TIB3intstr5 = fitTIB3intstr5->GetParError(1);
  p2TIB3intstr5 = fitTIB3intstr5->GetParameter(2);
  err2TIB3intstr5 = fitTIB3intstr5->GetParError(2);
  minTIB3intstr5 = fitTIB3intstr5->Eval(p0TIB3);
  
//Fit histogram TIB Layer 3 String int 6
  fitfunc->SetParameters(0,0,1);
  hwvsaTIBL3intstr6->Fit("fitfunc","E","",-22,22);
  TF1 *fitTIB3intstr6 = hwvsaTIBL3intstr6->GetFunction("fitfunc");
  chi2TIB3intstr6 = fitTIB3intstr6->GetChisquare();
  p0TIB3intstr6 = fitTIB3intstr6->GetParameter(0);
  err0TIB3intstr6 = fitTIB3intstr6->GetParError(0);
  p1TIB3intstr6 = fitTIB3intstr6->GetParameter(1);
  err1TIB3intstr6 = fitTIB3intstr6->GetParError(1);
  p2TIB3intstr6 = fitTIB3intstr6->GetParameter(2);
  err2TIB3intstr6 = fitTIB3intstr6->GetParError(2);
  minTIB3intstr6 = fitTIB3intstr6->Eval(p0TIB3);
  
//Fit histogram TIB Layer 3 String int 7
  fitfunc->SetParameters(0,0,1);
  hwvsaTIBL3intstr7->Fit("fitfunc","E","",-22,22);
  TF1 *fitTIB3intstr7 = hwvsaTIBL3intstr7->GetFunction("fitfunc");
  chi2TIB3intstr7 = fitTIB3intstr7->GetChisquare();
  p0TIB3intstr7 = fitTIB3intstr7->GetParameter(0);
  err0TIB3intstr7 = fitTIB3intstr7->GetParError(0);
  p1TIB3intstr7 = fitTIB3intstr7->GetParameter(1);
  err1TIB3intstr7 = fitTIB3intstr7->GetParError(1);
  p2TIB3intstr7 = fitTIB3intstr7->GetParameter(2);
  err2TIB3intstr7 = fitTIB3intstr7->GetParError(2);
  minTIB3intstr7 = fitTIB3intstr7->Eval(p0TIB3);
  
//Fit histogram TIB Layer 3 String int 8
  fitfunc->SetParameters(0,0,1);
  hwvsaTIBL3intstr8->Fit("fitfunc","E","",-22,22);
  TF1 *fitTIB3intstr8 = hwvsaTIBL3intstr8->GetFunction("fitfunc");
  chi2TIB3intstr8 = fitTIB3intstr8->GetChisquare();
  p0TIB3intstr8 = fitTIB3intstr8->GetParameter(0);
  err0TIB3intstr8 = fitTIB3intstr8->GetParError(0);
  p1TIB3intstr8 = fitTIB3intstr8->GetParameter(1);
  err1TIB3intstr8 = fitTIB3intstr8->GetParError(1);
  p2TIB3intstr8 = fitTIB3intstr8->GetParameter(2);
  err2TIB3intstr8 = fitTIB3intstr8->GetParError(2);
  minTIB3intstr8 = fitTIB3intstr8->Eval(p0TIB3);
  
//Fit histogram TIB Layer 3 String ext 1
  fitfunc->SetParameters(0,0,1);
  hwvsaTIBL3extstr1->Fit("fitfunc","E","",-22,22);
  TF1 *fitTIB3extstr1 = hwvsaTIBL3extstr1->GetFunction("fitfunc");
  chi2TIB3extstr1 = fitTIB3extstr1->GetChisquare();
  p0TIB3extstr1 = fitTIB3extstr1->GetParameter(0);
  err0TIB3extstr1 = fitTIB3extstr1->GetParError(0);
  p1TIB3extstr1 = fitTIB3extstr1->GetParameter(1);
  err1TIB3extstr1 = fitTIB3extstr1->GetParError(1);
  p2TIB3extstr1 = fitTIB3extstr1->GetParameter(2);
  err2TIB3extstr1 = fitTIB3extstr1->GetParError(2);
  minTIB3extstr1 = fitTIB3extstr1->Eval(p0TIB3);
  
//Fit histogram TIB Layer 3 String ext 2
  fitfunc->SetParameters(0,0,1);
  hwvsaTIBL3extstr2->Fit("fitfunc","E","",-22,22);
  TF1 *fitTIB3extstr2 = hwvsaTIBL3extstr2->GetFunction("fitfunc");
  chi2TIB3extstr2 = fitTIB3extstr2->GetChisquare();
  p0TIB3extstr2 = fitTIB3extstr2->GetParameter(0);
  err0TIB3extstr2 = fitTIB3extstr2->GetParError(0);
  p1TIB3extstr2 = fitTIB3extstr2->GetParameter(1);
  err1TIB3extstr2 = fitTIB3extstr2->GetParError(1);
  p2TIB3extstr2 = fitTIB3extstr2->GetParameter(2);
  err2TIB3extstr2 = fitTIB3extstr2->GetParError(2);
  minTIB3extstr2 = fitTIB3extstr2->Eval(p0TIB3);
  
//Fit histogram TIB Layer 3 String ext 3
  fitfunc->SetParameters(0,0,1);
  hwvsaTIBL3extstr3->Fit("fitfunc","E","",-22,22);
  TF1 *fitTIB3extstr3 = hwvsaTIBL3extstr3->GetFunction("fitfunc");
  chi2TIB3extstr3 = fitTIB3extstr3->GetChisquare();
  p0TIB3extstr3 = fitTIB3extstr3->GetParameter(0);
  err0TIB3extstr3 = fitTIB3extstr3->GetParError(0);
  p1TIB3extstr3 = fitTIB3extstr3->GetParameter(1);
  err1TIB3extstr3 = fitTIB3extstr3->GetParError(1);
  p2TIB3extstr3 = fitTIB3extstr3->GetParameter(2);
  err2TIB3extstr3 = fitTIB3extstr3->GetParError(2);
  minTIB3extstr3 = fitTIB3extstr3->Eval(p0TIB3);
  
//Fit histogram TIB Layer 3 String ext 4
  fitfunc->SetParameters(0,0,1);
  hwvsaTIBL3extstr4->Fit("fitfunc","E","",-22,22);
  TF1 *fitTIB3extstr4 = hwvsaTIBL3extstr4->GetFunction("fitfunc");
  chi2TIB3extstr4 = fitTIB3extstr4->GetChisquare();
  p0TIB3extstr4 = fitTIB3extstr4->GetParameter(0);
  err0TIB3extstr4 = fitTIB3extstr4->GetParError(0);
  p1TIB3extstr4 = fitTIB3extstr4->GetParameter(1);
  err1TIB3extstr4 = fitTIB3extstr4->GetParError(1);
  p2TIB3extstr4 = fitTIB3extstr4->GetParameter(2);
  err2TIB3extstr4 = fitTIB3extstr4->GetParError(2);
  minTIB3extstr4 = fitTIB3extstr4->Eval(p0TIB3);
  
//Fit histogram TIB Layer 3 String ext 5
  fitfunc->SetParameters(0,0,1);
  hwvsaTIBL3extstr5->Fit("fitfunc","E","",-22,22);
  TF1 *fitTIB3extstr5 = hwvsaTIBL3extstr5->GetFunction("fitfunc");
  chi2TIB3extstr5 = fitTIB3extstr5->GetChisquare();
  p0TIB3extstr5 = fitTIB3extstr5->GetParameter(0);
  err0TIB3extstr5 = fitTIB3extstr5->GetParError(0);
  p1TIB3extstr5 = fitTIB3extstr5->GetParameter(1);
  err1TIB3extstr5 = fitTIB3extstr5->GetParError(1);
  p2TIB3extstr5 = fitTIB3extstr5->GetParameter(2);
  err2TIB3extstr5 = fitTIB3extstr5->GetParError(2);
  minTIB3extstr5 = fitTIB3extstr5->Eval(p0TIB3);
  
//Fit histogram TIB Layer 3 String ext 6
  fitfunc->SetParameters(0,0,1);
  hwvsaTIBL3extstr6->Fit("fitfunc","E","",-22,22);
  TF1 *fitTIB3extstr6 = hwvsaTIBL3extstr6->GetFunction("fitfunc");
  chi2TIB3extstr6 = fitTIB3extstr6->GetChisquare();
  p0TIB3extstr6 = fitTIB3extstr6->GetParameter(0);
  err0TIB3extstr6 = fitTIB3extstr6->GetParError(0);
  p1TIB3extstr6 = fitTIB3extstr6->GetParameter(1);
  err1TIB3extstr6 = fitTIB3extstr6->GetParError(1);
  p2TIB3extstr6 = fitTIB3extstr6->GetParameter(2);
  err2TIB3extstr6 = fitTIB3extstr6->GetParError(2);
  minTIB3extstr6 = fitTIB3extstr6->Eval(p0TIB3);
  
//Fit histogram TIB Layer 3 String ext 7
  fitfunc->SetParameters(0,0,1);
  hwvsaTIBL3extstr7->Fit("fitfunc","E","",-22,22);
  TF1 *fitTIB3extstr7 = hwvsaTIBL3extstr7->GetFunction("fitfunc");
  chi2TIB3extstr7 = fitTIB3extstr7->GetChisquare();
  p0TIB3extstr7 = fitTIB3extstr7->GetParameter(0);
  err0TIB3extstr7 = fitTIB3extstr7->GetParError(0);
  p1TIB3extstr7 = fitTIB3extstr7->GetParameter(1);
  err1TIB3extstr7 = fitTIB3extstr7->GetParError(1);
  p2TIB3extstr7 = fitTIB3extstr7->GetParameter(2);
  err2TIB3extstr7 = fitTIB3extstr7->GetParError(2);
  minTIB3extstr7 = fitTIB3extstr7->Eval(p0TIB3);
  
//Fit histogram TOB Layer 1
  fitfunc->SetParameters(0,0,1);
  hwvsaTOBL1->Fit("fitfunc","E","",-14,14);
  TF1 *fitTOB1 = hwvsaTOBL1->GetFunction("fitfunc");
  chi2TOB1 = fitTOB1->GetChisquare();
  p0TOB1 = fitTOB1->GetParameter(0);
  err0TOB1 = fitTOB1->GetParError(0);
  p1TOB1 = fitTOB1->GetParameter(1);
  err1TOB1 = fitTOB1->GetParError(1);
  p2TOB1 = fitTOB1->GetParameter(2);
  err2TOB1 = fitTOB1->GetParError(2);
  minTOB1 = fitTOB1->Eval(p0TOB1);
  
//Fit histogram TOB Layer 1 rod 1
  fitfunc->SetParameters(0,0,1);
  hwvsaTOBL1rod1->Fit("fitfunc","E","",-14,14);
  TF1 *fitTOB1rod1 = hwvsaTOBL1rod1->GetFunction("fitfunc");
  chi2TOB1rod1 = fitTOB1rod1->GetChisquare();
  p0TOB1rod1 = fitTOB1rod1->GetParameter(0);
  err0TOB1rod1 = fitTOB1rod1->GetParError(0);
  p1TOB1rod1 = fitTOB1rod1->GetParameter(1);
  err1TOB1rod1 = fitTOB1rod1->GetParError(1);
  p2TOB1rod1 = fitTOB1rod1->GetParameter(2);
  err2TOB1rod1 = fitTOB1rod1->GetParError(2);
  minTOB1rod1 = fitTOB1rod1->Eval(p0TOB1);
  
//Fit histogram TOB Layer 1 rod 2
  fitfunc->SetParameters(0,0,1);
  hwvsaTOBL1rod2->Fit("fitfunc","E","",-14,14);
  TF1 *fitTOB1rod2 = hwvsaTOBL1rod2->GetFunction("fitfunc");
  chi2TOB1rod2 = fitTOB1rod2->GetChisquare();
  p0TOB1rod2 = fitTOB1rod2->GetParameter(0);
  err0TOB1rod2 = fitTOB1rod2->GetParError(0);
  p1TOB1rod2 = fitTOB1rod2->GetParameter(1);
  err1TOB1rod2 = fitTOB1rod2->GetParError(1);
  p2TOB1rod2 = fitTOB1rod2->GetParameter(2);
  err2TOB1rod2 = fitTOB1rod2->GetParError(2);
  minTOB1rod2 = fitTOB1rod2->Eval(p0TOB1);
  
//Fit histogram TOB Layer 2
  fitfunc->SetParameters(0,0,1);
  hwvsaTOBL2->Fit("fitfunc","E","",-8,8);
  TF1 *fitTOB2 = hwvsaTOBL2->GetFunction("fitfunc");
  chi2TOB2 = fitTOB2->GetChisquare();
  p0TOB2 = fitTOB2->GetParameter(0);
  err0TOB2 = fitTOB2->GetParError(0);
  p1TOB2 = fitTOB2->GetParameter(1);
  err1TOB2 = fitTOB2->GetParError(1);
  p2TOB2 = fitTOB2->GetParameter(2);
  err2TOB2 = fitTOB2->GetParError(2);
  minTOB2 = fitTOB2->Eval(p0TOB2);
  
//Fit histogram TOB Layer 2 rod 1
  fitfunc->SetParameters(0,0,1);
  hwvsaTOBL2rod1->Fit("fitfunc","E","",-14,14);
  TF1 *fitTOB2rod1 = hwvsaTOBL2rod1->GetFunction("fitfunc");
  chi2TOB2rod1 = fitTOB2rod1->GetChisquare();
  p0TOB2rod1 = fitTOB2rod1->GetParameter(0);
  err0TOB2rod1 = fitTOB2rod1->GetParError(0);
  p1TOB2rod1 = fitTOB2rod1->GetParameter(1);
  err1TOB2rod1 = fitTOB2rod1->GetParError(1);
  p2TOB2rod1 = fitTOB2rod1->GetParameter(2);
  err2TOB2rod1 = fitTOB2rod1->GetParError(2);
  minTOB2rod1 = fitTOB2rod1->Eval(p0TOB1);
  
//Fit histogram TOB Layer 2 rod 2
  fitfunc->SetParameters(0,0,1);
  hwvsaTOBL2rod2->Fit("fitfunc","E","",-14,14);
  TF1 *fitTOB2rod2 = hwvsaTOBL2rod2->GetFunction("fitfunc");
  chi2TOB2rod2 = fitTOB2rod2->GetChisquare();
  p0TOB2rod2 = fitTOB2rod2->GetParameter(0);
  err0TOB2rod2 = fitTOB2rod2->GetParError(0);
  p1TOB2rod2 = fitTOB2rod2->GetParameter(1);
  err1TOB2rod2 = fitTOB2rod2->GetParError(1);
  p2TOB2rod2 = fitTOB2rod2->GetParameter(2);
  err2TOB2rod2 = fitTOB2rod2->GetParError(2);
  minTOB2rod2 = fitTOB2rod2->Eval(p0TOB1);
  
//Fit histogram TOB L1+L2
  fitfunc->SetParameters(0,0,1);
  hwvsaTOB->Fit("fitfunc","E","",-14,14);
  TF1 *fitTOB = hwvsaTOB->GetFunction("fitfunc");
  chi2TOB = fitTOB->GetChisquare();
  p0TOB = fitTOB->GetParameter(0);
  err0TOB = fitTOB->GetParError(0);
  p1TOB = fitTOB->GetParameter(1);
  err1TOB = fitTOB->GetParError(1);
  p2TOB = fitTOB->GetParameter(2);
  err2TOB = fitTOB->GetParError(2);
  minTOB = fitTOB->Eval(p0TOB);
  
 
  ofstream fit;
  fit.open("fit.txt");
  
  fit<<endl<<">>>>>>>> Histogram fit: Cluster width vs track angle TIB Layer 2 <<<<<<<<"<<endl<<endl; 
  fit<<"Chi Square/ndf = "<<chi2TIB2<<endl;
  fit<<"p0 = "<<p0TIB2<<"     err p0 = "<<err0TIB2<<endl;
  fit<<"p1 = "<<p1TIB2<<"     err p1 = "<<err1TIB2<<endl;
  fit<<"p2 = "<<p2TIB2<<"     err p2 = "<<err2TIB2<<endl<<endl;
  fit<<"Minimum at angle = "<<p0TIB2<<"  +-  "<<err0TIB2<<endl;
  fit<<"Cluster size at the minimum = "<<minTIB2<<endl<<endl;
  
  fit<<endl<<"Histogram fit: Cluster width vs track angle TIB Layer 2 String int 1"<<endl<<endl; 
  fit<<"Chi Square/ndf = "<<chi2TIB2intstr1<<endl;
  fit<<"p0 = "<<p0TIB2intstr1<<"     err p0 = "<<err0TIB2intstr1<<endl;
  fit<<"p1 = "<<p1TIB2intstr1<<"     err p1 = "<<err1TIB2intstr1<<endl;
  fit<<"p2 = "<<p2TIB2intstr1<<"     err p2 = "<<err2TIB2intstr1<<endl<<endl;
  fit<<"Minimum at angle = "<<p0TIB2intstr1<<"  +-  "<<err0TIB2intstr1<<endl;
  fit<<"Cluster size at the minimum = "<<minTIB2intstr1<<endl<<endl;
  
  fit<<endl<<"Histogram fit: Cluster width vs track angle TIB Layer 2 String int 2"<<endl<<endl; 
  fit<<"Chi Square/ndf = "<<chi2TIB2intstr2<<endl;
  fit<<"p0 = "<<p0TIB2intstr2<<"     err p0 = "<<err0TIB2intstr2<<endl;
  fit<<"p1 = "<<p1TIB2intstr2<<"     err p1 = "<<err1TIB2intstr2<<endl;
  fit<<"p2 = "<<p2TIB2intstr2<<"     err p2 = "<<err2TIB2intstr2<<endl<<endl;
  fit<<"Minimum at angle = "<<p0TIB2intstr2<<"  +-  "<<err0TIB2intstr2<<endl;
  fit<<"Cluster size at the minimum = "<<minTIB2intstr2<<endl<<endl;
  
  fit<<endl<<"Histogram fit: Cluster width vs track angle TIB Layer 2 String ext 1"<<endl<<endl; 
  fit<<"Chi Square/ndf = "<<chi2TIB2extstr1<<endl;
  fit<<"p0 = "<<p0TIB2extstr1<<"     err p0 = "<<err0TIB2extstr1<<endl;
  fit<<"p1 = "<<p1TIB2extstr1<<"     err p1 = "<<err1TIB2extstr1<<endl;
  fit<<"p2 = "<<p2TIB2extstr1<<"     err p2 = "<<err2TIB2extstr1<<endl<<endl;
  fit<<"Minimum at angle = "<<p0TIB2extstr1<<"  +-  "<<err0TIB2extstr1<<endl;
  fit<<"Cluster size at the minimum = "<<minTIB2extstr1<<endl<<endl;
  
  fit<<endl<<"Histogram fit: Cluster width vs track angle TIB Layer 2 String ext 2"<<endl<<endl; 
  fit<<"Chi Square/ndf = "<<chi2TIB2extstr2<<endl;
  fit<<"p0 = "<<p0TIB2extstr2<<"     err p0 = "<<err0TIB2extstr2<<endl;
  fit<<"p1 = "<<p1TIB2extstr2<<"     err p1 = "<<err1TIB2extstr2<<endl;
  fit<<"p2 = "<<p2TIB2extstr2<<"     err p2 = "<<err2TIB2extstr2<<endl<<endl;
  fit<<"Minimum at angle = "<<p0TIB2extstr2<<"  +-  "<<err0TIB2extstr2<<endl;
  fit<<"Cluster size at the minimum = "<<minTIB2extstr2<<endl<<endl;
  
  fit<<endl<<"Histogram fit: Cluster width vs track angle TIB Layer 2 String ext 3"<<endl<<endl; 
  fit<<"Chi Square/ndf = "<<chi2TIB2extstr3<<endl;
  fit<<"p0 = "<<p0TIB2extstr3<<"     err p0 = "<<err0TIB2extstr3<<endl;
  fit<<"p1 = "<<p1TIB2extstr3<<"     err p1 = "<<err1TIB2extstr3<<endl;
  fit<<"p2 = "<<p2TIB2extstr3<<"     err p2 = "<<err2TIB2extstr3<<endl<<endl;
  fit<<"Minimum at angle = "<<p0TIB2extstr3<<"  +-  "<<err0TIB2extstr3<<endl;
  fit<<"Cluster size at the minimum = "<<minTIB2extstr3<<endl<<endl;
  
  fit<<">>>>>>>> Histogram fit: Cluster width vs track angle TIB Layer 3 <<<<<<<<"<<endl<<endl;
  fit<<"Chi Square/ndf = "<<chi2TIB3<<endl;
  fit<<"p0 = "<<p0TIB3<<"     err p0 = "<<err0TIB3<<endl;
  fit<<"p1 = "<<p1TIB3<<"     err p1 = "<<err1TIB3<<endl;
  fit<<"p2 = "<<p2TIB3<<"     err p2 = "<<err2TIB3<<endl<<endl;
  fit<<"Minimum at angle = "<<p0TIB3<<"  +-  "<<err0TIB3<<endl;
  fit<<"Cluster size at the minimum = "<<minTIB3<<endl<<endl;
  
  fit<<"Histogram fit: Cluster width vs track angle TIB Layer 3 String int 1"<<endl<<endl;
  fit<<"Chi Square/ndf = "<<chi2TIB3intstr1<<endl;
  fit<<"p0 = "<<p0TIB3intstr1<<"     err p0 = "<<err0TIB3intstr1<<endl;
  fit<<"p1 = "<<p1TIB3intstr1<<"     err p1 = "<<err1TIB3intstr1<<endl;
  fit<<"p2 = "<<p2TIB3intstr1<<"     err p2 = "<<err2TIB3intstr1<<endl<<endl;
  fit<<"Minimum at angle = "<<p0TIB3intstr1<<"  +-  "<<err0TIB3intstr1<<endl;
  fit<<"Cluster size at the minimum = "<<minTIB3intstr1<<endl<<endl;
  
  fit<<"Histogram fit: Cluster width vs track angle TIB Layer 3 String int 2"<<endl<<endl;
  fit<<"Chi Square/ndf = "<<chi2TIB3intstr2<<endl;
  fit<<"p0 = "<<p0TIB3intstr2<<"     err p0 = "<<err0TIB3intstr2<<endl;
  fit<<"p1 = "<<p1TIB3intstr2<<"     err p1 = "<<err1TIB3intstr2<<endl;
  fit<<"p2 = "<<p2TIB3intstr2<<"     err p2 = "<<err2TIB3intstr2<<endl<<endl;
  fit<<"Minimum at angle = "<<p0TIB3intstr2<<"  +-  "<<err0TIB3intstr2<<endl;
  fit<<"Cluster size at the minimum = "<<minTIB3intstr2<<endl<<endl;
  
  fit<<"Histogram fit: Cluster width vs track angle TIB Layer 3 String int 3"<<endl<<endl;
  fit<<"Chi Square/ndf = "<<chi2TIB3intstr3<<endl;
  fit<<"p0 = "<<p0TIB3intstr3<<"     err p0 = "<<err0TIB3intstr3<<endl;
  fit<<"p1 = "<<p1TIB3intstr3<<"     err p1 = "<<err1TIB3intstr3<<endl;
  fit<<"p2 = "<<p2TIB3intstr3<<"     err p2 = "<<err2TIB3intstr3<<endl<<endl;
  fit<<"Minimum at angle = "<<p0TIB3intstr3<<"  +-  "<<err0TIB3intstr3<<endl;
  fit<<"Cluster size at the minimum = "<<minTIB3intstr3<<endl<<endl;
  
  fit<<"Histogram fit: Cluster width vs track angle TIB Layer 3 String int 4"<<endl<<endl;
  fit<<"Chi Square/ndf = "<<chi2TIB3intstr4<<endl;
  fit<<"p0 = "<<p0TIB3intstr4<<"     err p0 = "<<err0TIB3intstr4<<endl;
  fit<<"p1 = "<<p1TIB3intstr4<<"     err p1 = "<<err1TIB3intstr4<<endl;
  fit<<"p2 = "<<p2TIB3intstr4<<"     err p2 = "<<err2TIB3intstr4<<endl<<endl;
  fit<<"Minimum at angle = "<<p0TIB3intstr4<<"  +-  "<<err0TIB3intstr4<<endl;
  fit<<"Cluster size at the minimum = "<<minTIB3intstr4<<endl<<endl;
  
  fit<<"Histogram fit: Cluster width vs track angle TIB Layer 3 String int 5"<<endl<<endl;
  fit<<"Chi Square/ndf = "<<chi2TIB3intstr5<<endl;
  fit<<"p0 = "<<p0TIB3intstr5<<"     err p0 = "<<err0TIB3intstr5<<endl;
  fit<<"p1 = "<<p1TIB3intstr5<<"     err p1 = "<<err1TIB3intstr5<<endl;
  fit<<"p2 = "<<p2TIB3intstr5<<"     err p2 = "<<err2TIB3intstr5<<endl<<endl;
  fit<<"Minimum at angle = "<<p0TIB3intstr5<<"  +-  "<<err0TIB3intstr5<<endl;
  fit<<"Cluster size at the minimum = "<<minTIB3intstr5<<endl<<endl;
  
  fit<<"Histogram fit: Cluster width vs track angle TIB Layer 3 String int 6"<<endl<<endl;
  fit<<"Chi Square/ndf = "<<chi2TIB3intstr6<<endl;
  fit<<"p0 = "<<p0TIB3intstr6<<"     err p0 = "<<err0TIB3intstr6<<endl;
  fit<<"p1 = "<<p1TIB3intstr6<<"     err p1 = "<<err1TIB3intstr6<<endl;
  fit<<"p2 = "<<p2TIB3intstr6<<"     err p2 = "<<err2TIB3intstr6<<endl<<endl;
  fit<<"Minimum at angle = "<<p0TIB3intstr6<<"  +-  "<<err0TIB3intstr6<<endl;
  fit<<"Cluster size at the minimum = "<<minTIB3intstr6<<endl<<endl;
  
  fit<<"Histogram fit: Cluster width vs track angle TIB Layer 3 String int 7"<<endl<<endl;
  fit<<"Chi Square/ndf = "<<chi2TIB3intstr7<<endl;
  fit<<"p0 = "<<p0TIB3intstr7<<"     err p0 = "<<err0TIB3intstr7<<endl;
  fit<<"p1 = "<<p1TIB3intstr7<<"     err p1 = "<<err1TIB3intstr7<<endl;
  fit<<"p2 = "<<p2TIB3intstr7<<"     err p2 = "<<err2TIB3intstr7<<endl<<endl;
  fit<<"Minimum at angle = "<<p0TIB3intstr7<<"  +-  "<<err0TIB3intstr7<<endl;
  fit<<"Cluster size at the minimum = "<<minTIB3intstr7<<endl<<endl;
  
  fit<<"Histogram fit: Cluster width vs track angle TIB Layer 3 String int 8"<<endl<<endl;
  fit<<"Chi Square/ndf = "<<chi2TIB3intstr8<<endl;
  fit<<"p0 = "<<p0TIB3intstr8<<"     err p0 = "<<err0TIB3intstr8<<endl;
  fit<<"p1 = "<<p1TIB3intstr8<<"     err p1 = "<<err1TIB3intstr8<<endl;
  fit<<"p2 = "<<p2TIB3intstr8<<"     err p2 = "<<err2TIB3intstr8<<endl<<endl;
  fit<<"Minimum at angle = "<<p0TIB3intstr8<<"  +-  "<<err0TIB3intstr8<<endl;
  fit<<"Cluster size at the minimum = "<<minTIB3intstr8<<endl<<endl;
  
  fit<<"Histogram fit: Cluster width vs track angle TIB Layer 3 String ext 1"<<endl<<endl;
  fit<<"Chi Square/ndf = "<<chi2TIB3extstr1<<endl;
  fit<<"p0 = "<<p0TIB3extstr1<<"     err p0 = "<<err0TIB3extstr1<<endl;
  fit<<"p1 = "<<p1TIB3extstr1<<"     err p1 = "<<err1TIB3extstr1<<endl;
  fit<<"p2 = "<<p2TIB3extstr1<<"     err p2 = "<<err2TIB3extstr1<<endl<<endl;
  fit<<"Minimum at angle = "<<p0TIB3extstr1<<"  +-  "<<err0TIB3extstr1<<endl;
  fit<<"Cluster size at the minimum = "<<minTIB3extstr1<<endl<<endl;
  
  fit<<"Histogram fit: Cluster width vs track angle TIB Layer 3 String ext 2"<<endl<<endl;
  fit<<"Chi Square/ndf = "<<chi2TIB3extstr2<<endl;
  fit<<"p0 = "<<p0TIB3extstr2<<"     err p0 = "<<err0TIB3extstr2<<endl;
  fit<<"p1 = "<<p1TIB3extstr2<<"     err p1 = "<<err1TIB3extstr2<<endl;
  fit<<"p2 = "<<p2TIB3extstr2<<"     err p2 = "<<err2TIB3extstr2<<endl<<endl;
  fit<<"Minimum at angle = "<<p0TIB3extstr2<<"  +-  "<<err0TIB3extstr2<<endl;
  fit<<"Cluster size at the minimum = "<<minTIB3extstr2<<endl<<endl;
  
   fit<<"Histogram fit: Cluster width vs track angle TIB Layer 3 String ext 3"<<endl<<endl;
  fit<<"Chi Square/ndf = "<<chi2TIB3extstr3<<endl;
  fit<<"p0 = "<<p0TIB3extstr3<<"     err p0 = "<<err0TIB3extstr3<<endl;
  fit<<"p1 = "<<p1TIB3extstr3<<"     err p1 = "<<err1TIB3extstr3<<endl;
  fit<<"p2 = "<<p2TIB3extstr3<<"     err p2 = "<<err2TIB3extstr3<<endl<<endl;
  fit<<"Minimum at angle = "<<p0TIB3extstr3<<"  +-  "<<err0TIB3extstr3<<endl;
  fit<<"Cluster size at the minimum = "<<minTIB3extstr3<<endl<<endl;
  
  fit<<"Histogram fit: Cluster width vs track angle TIB Layer 3 String ext 4"<<endl<<endl;
  fit<<"Chi Square/ndf = "<<chi2TIB3extstr4<<endl;
  fit<<"p0 = "<<p0TIB3extstr4<<"     err p0 = "<<err0TIB3extstr4<<endl;
  fit<<"p1 = "<<p1TIB3extstr4<<"     err p1 = "<<err1TIB3extstr4<<endl;
  fit<<"p2 = "<<p2TIB3extstr4<<"     err p2 = "<<err2TIB3extstr4<<endl<<endl;
  fit<<"Minimum at angle = "<<p0TIB3extstr4<<"  +-  "<<err0TIB3extstr4<<endl;
  fit<<"Cluster size at the minimum = "<<minTIB3extstr4<<endl<<endl;
  
  fit<<"Histogram fit: Cluster width vs track angle TIB Layer 3 String ext 5"<<endl<<endl;
  fit<<"Chi Square/ndf = "<<chi2TIB3extstr5<<endl;
  fit<<"p0 = "<<p0TIB3extstr5<<"     err p0 = "<<err0TIB3extstr5<<endl;
  fit<<"p1 = "<<p1TIB3extstr5<<"     err p1 = "<<err1TIB3extstr5<<endl;
  fit<<"p2 = "<<p2TIB3extstr5<<"     err p2 = "<<err2TIB3extstr5<<endl<<endl;
  fit<<"Minimum at angle = "<<p0TIB3extstr5<<"  +-  "<<err0TIB3extstr5<<endl;
  fit<<"Cluster size at the minimum = "<<minTIB3extstr5<<endl<<endl;
  
  fit<<"Histogram fit: Cluster width vs track angle TIB Layer 3 String ext 6"<<endl<<endl;
  fit<<"Chi Square/ndf = "<<chi2TIB3extstr6<<endl;
  fit<<"p0 = "<<p0TIB3extstr6<<"     err p0 = "<<err0TIB3extstr6<<endl;
  fit<<"p1 = "<<p1TIB3extstr6<<"     err p1 = "<<err1TIB3extstr6<<endl;
  fit<<"p2 = "<<p2TIB3extstr6<<"     err p2 = "<<err2TIB3extstr6<<endl<<endl;
  fit<<"Minimum at angle = "<<p0TIB3extstr6<<"  +-  "<<err0TIB3extstr6<<endl;
  fit<<"Cluster size at the minimum = "<<minTIB3extstr6<<endl<<endl;
  
  fit<<"Histogram fit: Cluster width vs track angle TIB Layer 3 String ext 7"<<endl<<endl;
  fit<<"Chi Square/ndf = "<<chi2TIB3extstr7<<endl;
  fit<<"p0 = "<<p0TIB3extstr7<<"     err p0 = "<<err0TIB3extstr7<<endl;
  fit<<"p1 = "<<p1TIB3extstr7<<"     err p1 = "<<err1TIB3extstr7<<endl;
  fit<<"p2 = "<<p2TIB3extstr7<<"     err p2 = "<<err2TIB3extstr7<<endl<<endl;
  fit<<"Minimum at angle = "<<p0TIB3extstr7<<"  +-  "<<err0TIB3extstr7<<endl;
  fit<<"Cluster size at the minimum = "<<minTIB3extstr7<<endl<<endl;
  
  fit<<">>>>>>>> Histogram fit: Cluster width vs track angle TOB Layer 1 <<<<<<<<"<<endl<<endl;
  fit<<"Chi Square/ndf = "<<chi2TOB1<<endl;
  fit<<"p0 = "<<p0TOB1<<"     err p0 = "<<err0TOB1<<endl;
  fit<<"p1 = "<<p1TOB1<<"     err p1 = "<<err1TOB1<<endl;
  fit<<"p2 = "<<p2TOB1<<"     err p2 = "<<err2TOB1<<endl<<endl;
  fit<<"Minimum at angle = "<<p0TOB1<<"  +-  "<<err0TOB1<<endl;
  fit<<"Cluster size at the minimum = "<<minTOB1<<endl<<endl;
  
  fit<<"Histogram fit: Cluster width vs track angle TOB Layer 1 Rod 1"<<endl<<endl;
  fit<<"Chi Square/ndf = "<<chi2TOB1rod1<<endl;
  fit<<"p0 = "<<p0TOB1rod1<<"     err p0 = "<<err0TOB1rod1<<endl;
  fit<<"p1 = "<<p1TOB1rod1<<"     err p1 = "<<err1TOB1rod1<<endl;
  fit<<"p2 = "<<p2TOB1rod1<<"     err p2 = "<<err2TOB1rod1<<endl<<endl;
  fit<<"Minimum at angle = "<<p0TOB1rod1<<"  +-  "<<err0TOB1rod1<<endl;
  fit<<"Cluster size at the minimum = "<<minTOB1rod1<<endl<<endl;
  
  fit<<"Histogram fit: Cluster width vs track angle TOB Layer 1 Rod 2"<<endl<<endl;
  fit<<"Chi Square/ndf = "<<chi2TOB1rod2<<endl;
  fit<<"p0 = "<<p0TOB1rod2<<"     err p0 = "<<err0TOB1rod2<<endl;
  fit<<"p1 = "<<p1TOB1rod2<<"     err p1 = "<<err1TOB1rod2<<endl;
  fit<<"p2 = "<<p2TOB1rod2<<"     err p2 = "<<err2TOB1rod2<<endl<<endl;
  fit<<"Minimum at angle = "<<p0TOB1rod2<<"  +-  "<<err0TOB1rod2<<endl;
  fit<<"Cluster size at the minimum = "<<minTOB1rod2<<endl<<endl;
  
  fit<<">>>>>>>> Histogram fit: Cluster width vs track angle TOB Layer 2 <<<<<<<<"<<endl<<endl;
  fit<<"Chi Square/ndf = "<<chi2TOB2<<endl;
  fit<<"p0 = "<<p0TOB2<<"     err p0 = "<<err0TOB2<<endl;
  fit<<"p1 = "<<p1TOB2<<"     err p1 = "<<err1TOB2<<endl;
  fit<<"p2 = "<<p2TOB2<<"     err p2 = "<<err2TOB2<<endl<<endl;
  fit<<"Minimum at angle = "<<p0TOB2<<"  +-  "<<err0TOB2<<endl;
  fit<<"Cluster size at the minimum = "<<minTOB2<<endl<<endl;
  
  fit<<"Histogram fit: Cluster width vs track angle TOB Layer 2 Rod 1"<<endl<<endl;
  fit<<"Chi Square/ndf = "<<chi2TOB2rod1<<endl;
  fit<<"p0 = "<<p0TOB2rod1<<"     err p0 = "<<err0TOB2rod1<<endl;
  fit<<"p1 = "<<p1TOB2rod1<<"     err p1 = "<<err1TOB2rod1<<endl;
  fit<<"p2 = "<<p2TOB2rod1<<"     err p2 = "<<err2TOB2rod1<<endl<<endl;
  fit<<"Minimum at angle = "<<p0TOB2rod1<<"  +-  "<<err0TOB2rod1<<endl;
  fit<<"Cluster size at the minimum = "<<minTOB2rod1<<endl<<endl;
   
  fit<<"Histogram fit: Cluster width vs track angle TOB Layer 2 Rod 2"<<endl<<endl;
  fit<<"Chi Square/ndf = "<<chi2TOB2rod2<<endl;
  fit<<"p0 = "<<p0TOB2rod2<<"     err p0 = "<<err0TOB2rod2<<endl;
  fit<<"p1 = "<<p1TOB2rod2<<"     err p1 = "<<err1TOB2rod2<<endl;
  fit<<"p2 = "<<p2TOB2rod2<<"     err p2 = "<<err2TOB2rod2<<endl<<endl;
  fit<<"Minimum at angle = "<<p0TOB2rod2<<"  +-  "<<err0TOB2rod2<<endl;
  fit<<"Cluster size at the minimum = "<<minTOB2rod2<<endl<<endl;
  
  fit<<">>>>> Histogram fit: Cluster width vs track angle TOB L1 + L2 <<<<<"<<endl<<endl;
  fit<<"Chi Square/ndf = "<<chi2TOB<<endl;
  fit<<"p0 = "<<p0TOB<<"     err p0 = "<<err0TOB<<endl;
  fit<<"p1 = "<<p1TOB<<"     err p1 = "<<err1TOB<<endl;
  fit<<"p2 = "<<p2TOB<<"     err p2 = "<<err2TOB<<endl<<endl;
  fit<<"Minimum at angle = "<<p0TOB<<"  +-  "<<err0TOB<<endl;
  fit<<"Cluster size at the minimum = "<<minTOB<<endl<<endl;
  
  fit<<">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>    Total event = "<<eventcounter<<endl;
  fit<<">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>    Total reconstructed tracks = "<<trackcounter<<endl;
  
  fit.close();
  
  hFile->Write();
  hFile->Close();
}
