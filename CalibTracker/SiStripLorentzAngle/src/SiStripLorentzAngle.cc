
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
  hwvsaTIBL3 = new TProfile("hwvsatibl3","Cluster width vs track angle (TIB L3)",30,-60.,60.);
  hwvsaTOB = new TProfile("hwvsatob","Cluster width vs track angle (TOB)",30,-60.,60.);
  hwvsaTOBL1 = new TProfile("hwvsatobl1","Cluster width vs track angle (TOB L1)",30,-60.,60.);
  hwvsaTOBL2 = new TProfile("hwvsatobl2","Cluster width vs track angle (TOB L2)",30,-60.,60.);
  hwvst = new TProfile("hwvst","Cluster width vs track projection ",30,-60.,60.);
  fitfunc = new TF1("fitfunc","[1]*((x-[0])^2)+[2]",-60,60);
  fitTIB2 = new TF1;
  fitTIB3 = new TF1;
  fitTOB1 = new TF1;
  fitTOB2 = new TF1;
  fitTOB = new TF1;
  
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
	}
	else if(id.layer()==2){
	  htaTIBL3->Fill(angle);
	  hwvsaTIBL3->Fill(angle,size);
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
	}
	else if(id.layer()==2){
	  htaTOB2->Fill(angle);
	  hwvsaTOBL2->Fill(angle,size);
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
  
  fit<<endl<<">>>>> Histogram fit: Cluster width vs track angle TIB Layer 2"<<endl<<endl; 
  fit<<"Chi Square/ndf = "<<chi2TIB2<<endl;
  fit<<"p0 = "<<p0TIB2<<"     err p0 = "<<err0TIB2<<endl;
  fit<<"p1 = "<<p1TIB2<<"     err p1 = "<<err1TIB2<<endl;
  fit<<"p2 = "<<p2TIB2<<"     err p2 = "<<err2TIB2<<endl<<endl;
  fit<<"Minimum at angle = "<<p0TIB2<<"  +-  "<<err0TIB2<<endl;
  fit<<"Cluster size at the minimum = "<<minTIB2<<endl<<endl;
  
  fit<<">>>>> Histogram fit: Cluster width vs track angle TIB Layer 3"<<endl<<endl;
  fit<<"Chi Square/ndf = "<<chi2TIB3<<endl;
  fit<<"p0 = "<<p0TIB3<<"     err p0 = "<<err0TIB3<<endl;
  fit<<"p1 = "<<p1TIB3<<"     err p1 = "<<err1TIB3<<endl;
  fit<<"p2 = "<<p2TIB3<<"     err p2 = "<<err2TIB3<<endl<<endl;
  fit<<"Minimum at angle = "<<p0TIB3<<"  +-  "<<err0TIB3<<endl;
  fit<<"Cluster size at the minimum = "<<minTIB3<<endl<<endl;
  
  fit<<">>>>> Histogram fit: Cluster width vs track angle TOB Layer 1"<<endl<<endl;
  fit<<"Chi Square/ndf = "<<chi2TOB1<<endl;
  fit<<"p0 = "<<p0TOB1<<"     err p0 = "<<err0TOB1<<endl;
  fit<<"p1 = "<<p1TOB1<<"     err p1 = "<<err1TOB1<<endl;
  fit<<"p2 = "<<p2TOB1<<"     err p2 = "<<err2TOB1<<endl<<endl;
  fit<<"Minimum at angle = "<<p0TOB1<<"  +-  "<<err0TOB1<<endl;
  fit<<"Cluster size at the minimum = "<<minTOB1<<endl<<endl;
  
  fit<<">>>>> Histogram fit: Cluster width vs track angle TOB Layer 2"<<endl<<endl;
  fit<<"Chi Square/ndf = "<<chi2TOB2<<endl;
  fit<<"p0 = "<<p0TOB2<<"     err p0 = "<<err0TOB2<<endl;
  fit<<"p1 = "<<p1TOB2<<"     err p1 = "<<err1TOB2<<endl;
  fit<<"p2 = "<<p2TOB2<<"     err p2 = "<<err2TOB2<<endl<<endl;
  fit<<"Minimum at angle = "<<p0TOB2<<"  +-  "<<err0TOB2<<endl;
  fit<<"Cluster size at the minimum = "<<minTOB2<<endl<<endl;
  
  fit<<">>>>> Histogram fit: Cluster width vs track angle TOB L1 + L2"<<endl<<endl;
  fit<<"Chi Square/ndf = "<<chi2TOB<<endl;
  fit<<"p0 = "<<p0TOB<<"     err p0 = "<<err0TOB<<endl;
  fit<<"p1 = "<<p1TOB<<"     err p1 = "<<err1TOB<<endl;
  fit<<"p2 = "<<p2TOB<<"     err p2 = "<<err2TOB<<endl<<endl;
  fit<<"Minimum at angle = "<<p0TOB<<"  +-  "<<err0TOB<<endl;
  fit<<"Cluster size at the minimum = "<<minTOB<<endl<<endl;
  
  fit<<"Total event = "<<eventcounter<<endl;
  fit<<"Total reconstructed tracks = "<<trackcounter<<endl;
  
  fit.close();
  
  hFile->Write();
  hFile->Close();
}
