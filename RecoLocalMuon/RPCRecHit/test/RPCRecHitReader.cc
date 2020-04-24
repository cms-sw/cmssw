/*
 *  Program to calculate the global and local efficiency of
 *  RPC chambers using a software autotrigger and 
 *  a linear reconstruction of tracks.  
 *
 *  Author: R. Trentadue - Bari University
 */

#include "RPCRecHitReader.h"

#include <memory>
#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <cmath>
#include <vector>
#include <iomanip>
#include <set>
#include <stdio.h>

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHit.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
 
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "TFile.h"
#include "TVector3.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TF1.h"
#include "TF2.h"
#include "TVectorT.h"
#include "TGraph.h"

#include <map>


Double_t linearF(Double_t *x, Double_t *par)
{
  Double_t y=0.;
  y = par[0]*(*x) + par[1];
  return y;
}
  
// Constructor
RPCRecHitReader::RPCRecHitReader(const edm::ParameterSet& pset):_phi(0)
{

  // Get the various input parameters
  fOutputFileName = pset.getUntrackedParameter<std::string>("HistOutFile");
  recHitLabel1    = pset.getUntrackedParameter<std::string>("recHitLabel1");
  recHitLabel2    = pset.getUntrackedParameter<std::string>("recHitLabel2");

  region = pset.getUntrackedParameter<int>("region",0);
  wheel = pset.getUntrackedParameter<int>("wheel",1);
  sector = pset.getUntrackedParameter<int>("sector",10);
  station = pset.getUntrackedParameter<int>("station");
  layer = pset.getUntrackedParameter<int>("layer");
  subsector = pset.getUntrackedParameter<int>("subsector");

  _trigConfig.clear();

  _trigRPC1 = pset.getUntrackedParameter<bool>("trigRPC1");
  _trigRPC2 = pset.getUntrackedParameter<bool>("trigRPC2");
  _trigRPC3 = pset.getUntrackedParameter<bool>("trigRPC3");
  _trigRPC4 = pset.getUntrackedParameter<bool>("trigRPC4");
  _trigRPC5 = pset.getUntrackedParameter<bool>("trigRPC5");
  _trigRPC6 = pset.getUntrackedParameter<bool>("trigRPC6");

  _trigConfig.push_back(_trigRPC1);
  _trigConfig.push_back(_trigRPC2);
  _trigConfig.push_back(_trigRPC3);
  _trigConfig.push_back(_trigRPC4);
  _trigConfig.push_back(_trigRPC5);
  _trigConfig.push_back(_trigRPC6);

}

void RPCRecHitReader::beginRun(const edm::Run&, const edm::EventSetup& iSetup)
{

  edm::ESHandle<RPCGeometry> rpcGeo;
  iSetup.get<MuonGeometryRecord>().get(rpcGeo);

  RPCDetId id(region,wheel,station,sector,layer,subsector,3);
  const RPCRoll* roll = dynamic_cast<const RPCRoll* >( rpcGeo->roll(id));
  _rollEff = roll;

  fOutputFile  = new TFile( fOutputFileName.c_str(), "RECREATE" );
  fout = new std::fstream("RecHitOut.dat", std::ios::out);

  _mapLayer[0] = -413.675;
  _mapLayer[1] = -448.675;
  _mapLayer[2] = -494.975;
  _mapLayer[3] = -529.975;
  _mapLayer[4] = -602.150;
  _mapLayer[5] = -704.550;

  unsigned int layer=0;

  for(unsigned int i = 0; i < _trigConfig.size(); ++i){
    if(_trigConfig[i] == false) layer = i;
  }

  yLayer = _mapLayer[layer];

  if(sector != 10){
    GlobalPoint cntr10, cntr11;
    for (RPCGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();
	 it<rpcGeo->dets().end();it++){
      RPCRoll const* ir = dynamic_cast<const RPCRoll*>(*it);
      RPCDetId id = ir->id();
      
      const Surface& bSurface = ir->surface();
      
      if(id.region() == region && id.ring() == wheel && id.sector() == 10 && id.station() == 1 && id.layer() == 1){
	LocalPoint orgn(0,0,0);
	cntr10 = bSurface.toGlobal(orgn);
      }
      if(id.region() == region && id.ring() == wheel && id.sector() == sector && id.station() == 1 && id.layer() == 1){
	LocalPoint orng(0,0,0);
	cntr11 = bSurface.toGlobal(orng);
      }
    }

    float radius = 413.675;
    float crd = sqrt(std::pow((cntr10.x()-cntr11.x()),2)+std::pow((cntr10.y()-cntr11.y()),2));
    _phi = 2*asin(crd/(2*radius));
  }

  *fout<<"Angolo di rotazione = "<<_phi<<std::endl;

  _trigger = 0;
  _triggerGOOD = 0;
  _spurious = 0;
  _spuriousPeak = 0;
  _efficiencyBAD = 0;
  _efficiencyGOOD = 0;
  _efficiencyBEST = 0;

  histoXY = new TH2F("HistoXY","Histoxy",300,-300,300,300,-900,-300);
  histoSlope = new TH1F("Slope","Slope",100, -100,100);
  histoChi2 = new TH1F("Chi2","Chi2",100,0,10000);
  histoRes = new TH1F("Residui","Residui",500,-100,100);
  histoRes1 = new TH1F("Single Cluster Residuals","Single Cluster Residuals",500,-100,100);
  histoRes2 = new TH1F("Double Cluster Residuals","Double Cluster Residuals",500,-100,100);
  histoRes3 = new TH1F("Triple Cluster Residuals","Triple Cluster Residuals",500,-100,100);
  histoPool1 = new TH1F("Single Cluster Size Pools","Single Cluster Size Pools",500,-100,100);
  histoPool2 = new TH1F("Double Cluster Size Pools","Double Cluster Size Pools",500,-100,100);
  histoPool3 = new TH1F("Triple Cluster Size Pools","Triple Cluster Size Pools",500,-100,100);

  histoExpectedOcc = new TH1F("ExpectedOcc","ExpectedOcc",100,-0.5,99.5);
  histoRealOcc = new TH1F("RealOcc","RealOcc",100,-0.5,99.5);
  histoLocalEff =  new TH1F("LocalEff","LocalEff",100,-0.5,99.5);

  return ;

}

// The Analysis  (the main)
void RPCRecHitReader::analyze(const edm::Event & event, const edm::EventSetup& eventSetup)
{
  
  if (event.id().event()%100 == 0) std::cout << " Event analysed #Run: " << event.id().run() 
					<< " #Event: " << event.id().event() << std::endl;  
  
  // Get the RPC Geometry :
  edm::ESHandle<RPCGeometry> rpcGeom;
  eventSetup.get<MuonGeometryRecord>().get(rpcGeom);
  
  // Get the RecHits collection :
  edm::Handle<RPCRecHitCollection> recHits; 
  event.getByLabel(recHitLabel1,recHitLabel2,recHits);  
  
  //----------------------------------------------------------------------
  //---------------  Loop over rechits -----------------------------------
  
  // Build iterator for rechits and loop :
  RPCRecHitCollection::const_iterator recIt;

  std::vector<float> globalX,globalY,globalZ;
  std::vector<float> effX,effY,effZ;
  std::vector<float> errX,clus,rescut;

  std::map<int,bool> _mapTrig;

  TF1 *func = new TF1("linearF",linearF,-300,300,2);
  func->SetParameters(0.,0.);
  func->SetParNames("angCoef","interc");

  for (recIt = recHits->begin(); recIt != recHits->end(); recIt++) {
	
    // Find chamber with rechits in RPC 
    RPCDetId id = (RPCDetId)(*recIt).rpcId();
    const RPCRoll* roll = dynamic_cast<const RPCRoll* >( rpcGeom->roll(id));
    const Surface& bSurface = roll->surface();	

    if((roll->isForward())) return;
   
    LocalPoint rhitlocal = (*recIt).localPosition();
    LocalError locerr = (*recIt).localPositionError(); 
    GlobalPoint rhitglob = bSurface.toGlobal(rhitlocal);

    float x = 0, y = 0, z = 0;

    if(id.sector() > 10 || (1 <= id.sector() && id.sector() <= 4)){ 
      x = rhitglob.x()*cos(-_phi)-rhitglob.y()*sin(-_phi);
      y = rhitglob.y()*cos(-_phi)+rhitglob.x()*sin(-_phi);
      z = rhitglob.z();
    }
    else if(5 <= id.sector() && id.sector() <= 9){ 
      x = rhitglob.x()*cos(_phi)-rhitglob.y()*sin(_phi);
      y = rhitglob.y()*cos(_phi)+rhitglob.x()*sin(_phi);
      z = rhitglob.z();
    }
    else if(id.sector() == 10){
      x = rhitglob.x();
      y = rhitglob.y();
      z = rhitglob.z();
    }

    if(_trigConfig[layerRecHit(*recIt)] == true && id.sector() == sector){
      
      _mapTrig[layerRecHit(*recIt)] = true;

      globalX.push_back(x);
      globalY.push_back(y);
      globalZ.push_back(z);
      
    }
    else if(_trigConfig[layerRecHit(*recIt)] == false  && id.sector() == sector){
      
      effX.push_back(rhitglob.x());
      effY.push_back(rhitglob.y());
      effZ.push_back(rhitglob.z());
      errX.push_back(locerr.xx());
      clus.push_back((*recIt).clusterSize());

    }
    else{
      continue;
    }
  }
  
  if(_mapTrig.size() == 0) return;

  char folder[128];
  sprintf(folder,"HistoXYFit_%llu",event.id().event());
  TH1F* histoXYFit = new TH1F(folder,folder,300,-300,300);

  for(unsigned int i = 0; i < globalX.size(); ++i){
    histoXY->Fill(globalX[i],globalY[i]);
    histoXYFit->Fill(globalX[i],globalY[i]);
  }
 
  //-------- PRIMA STIMA EFFICIENZA ----------------------

  if(_mapTrig.size() > 4){
    
    _trigger++;
    if(effX.size() > 0) _efficiencyBAD++;
    
    //-----------------------------------------------------
    //--------------- FIT SULLE TRACCE---------------------
    
    histoXYFit->Fit("linearF","r");
    histoSlope->Fill(func->GetParameter(0));
    histoChi2->Fill(func->GetChisquare());

    if(func->GetChisquare() < 0.5){
      _triggerGOOD++;
      
      float prepoint = 
	((yLayer) - func->GetParameter(1))/func->GetParameter(0);

      
      LocalPoint expPoint(prepoint,0,0);      
      float expStrip = _rollEff->strip(expPoint);

      histoExpectedOcc->Fill(expStrip);
      
      if(effX.size() > 0){
	_efficiencyGOOD++;    
	
	unsigned int k = 0;
	for(unsigned int i = 0; i < effX.size(); ++i){
	  
	  float res = (effX[i] - prepoint);
          float errcl =  errX[i]*clus[i];
	  float pools = res/errX[i];

	  histoRes->Fill(res);
	  if(clus[i] == 1){
	    histoRes1->Fill(res);
	    histoPool1->Fill(pools);
	  }
	  else if(clus[i] == 2){
	    histoRes2->Fill(res);
	    histoPool2->Fill(pools);
	  }
	  else if(clus[i] == 3){
	    histoRes3->Fill(res);
	    histoPool3->Fill(pools);
	  }

	  if(fabs(res) > errcl){
	    _spurious++;
 	    *fout<<"Il residuo è maggiore di "<<errcl<<" ";
	    *fout<<" #Event: "<<event.id().event()<<std::endl;
	  }
	  else if(fabs(res) <= errcl){
	    k++;
	    rescut.push_back(res);
	    if (k == 1) histoRealOcc->Fill(expStrip);
	  }
	}

	if(rescut.size() > 1){
	  _spuriousPeak += rescut.size() - 1;
	  *fout<<"Ci sono più RecHit = "<<rescut.size()<<"  ";
	  *fout<<" #Event: "<<event.id().event()<<std::endl;
	}
      }
      else {
 	*fout<<"Camera non efficiente!"<<"  "<< " #Event: " << event.id().event()<<std::endl;
      }
    }
  }
  else{
    *fout<<"Non esiste RecHit appartenente a piani di interesse!"<<"   ";
    *fout<< " #Event: " << event.id().event()<<std::endl;
  }
}

unsigned int RPCRecHitReader::layerRecHit(RPCRecHit rechit){

  unsigned int layer=0;
  RPCDetId id = (RPCDetId)(rechit).rpcId();
  
  if(id.station() == 1 && id.layer() == 1) layer = 0;
  if(id.station() == 1 && id.layer() == 2) layer = 1;
  if(id.station() == 2 && id.layer() == 1) layer = 2;
  if(id.station() == 2 && id.layer() == 2) layer = 3;
  if(id.station() == 3 && id.layer() == 1) layer = 4;
  if(id.station() == 4 && id.layer() == 1) layer = 5;
  
  return layer;
}

void RPCRecHitReader::endJob(){

  TF1* bell = new TF1("Gaussiana","gaus",-50,50);
  histoRes->Fit("Gaussiana","r");
  float sgmp = bell->GetParameter(1)+3*(bell->GetParameter(2));
  float sgmm = bell->GetParameter(1)-3*(bell->GetParameter(2));

  int binmax = histoRes->GetXaxis()->FindBin(sgmp);
  int binmin = histoRes->GetXaxis()->FindBin(sgmm);
  _efficiencyBEST = histoRes->Integral(binmin,binmax);

  _efficiencyBEST -= _spuriousPeak;

  *fout<<"Media = "<<bell->GetParameter(1)<<" Deviazione standard = "<<bell->GetParameter(2)<<std::endl;
  *fout<<"Taglio a 3 sigma"<<std::endl;

  for(unsigned int i = 1; i <= 100; ++i ){
      
    if( histoExpectedOcc->GetBinContent(i) != 0){
      float eff =  histoRealOcc->GetBinContent(i)/histoExpectedOcc->GetBinContent(i);
      float erreff = sqrt(eff*(1-eff)/histoExpectedOcc->GetBinContent(i));
      histoLocalEff->SetBinContent(i,eff);
      histoLocalEff->SetBinError(i,erreff);
    }
  }

  histoRes1->Fit("Gaussiana","r");
  histoRes2->Fit("Gaussiana","r");
  histoRes3->Fit("Gaussiana","r");

  histoPool1->Fit("Gaussiana","r");
  histoPool2->Fit("Gaussiana","r");
  histoPool3->Fit("Gaussiana","r");

  fOutputFile->Write() ;
  fOutputFile->Close() ;
  return ;
}

// Destructor
RPCRecHitReader::~RPCRecHitReader(){
  
  *fout<<"Trigger Complessivi = "<<_trigger<<std::endl;
  *fout<<"Trigger GOOD = "<<_triggerGOOD<<std::endl;
  *fout<<"Efficiency BAD Counter = "<<_efficiencyBAD<<std::endl;
  *fout<<"Efficiency GOOD Counter = "<<_efficiencyGOOD<<std::endl;
  *fout<<"Efficiency BEST Counter = "<<_efficiencyBEST<<std::endl;
  *fout<<"Spurious counter = "<<_spurious<<std::endl;

  *fout<<"Efficienza BAD = "<<_efficiencyBAD/_trigger<<std::endl;
  *fout<<"Efficienza GOOD = "<<_efficiencyGOOD/_triggerGOOD<<std::endl;
  *fout<<"Efficienza BEST = "<<_efficiencyBEST/_triggerGOOD<<std::endl;

}


DEFINE_FWK_MODULE(RPCRecHitReader);
  
