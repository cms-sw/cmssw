//
// Original Author:  Cesare Calabria,161 R-006,
//         Created:  Fri May 13 12:58:43 CEST 2011


// user include files
#include "DQM/RPCMonitorClient/interface/RPCEfficiencyPerRingLayer.h"
#include <sstream>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

RPCEfficiencyPerRingLayer::RPCEfficiencyPerRingLayer(const edm::ParameterSet& ps) {

  globalFolder_ = ps.getUntrackedParameter<std::string>("GlobalFolder", "RPC/RPCEfficiency/");
  SaveFile  = ps.getUntrackedParameter<bool>("SaveFile", false);
  NameFile  = ps.getUntrackedParameter<std::string>("NameFile","RPCEfficiency.root");

  numberOfDisks_ = ps.getUntrackedParameter<int>("NumberOfEndcapDisks", 4);
  innermostRings_ = ps.getUntrackedParameter<int>("NumberOfInnermostEndcapRings", 2);

}


RPCEfficiencyPerRingLayer::~RPCEfficiencyPerRingLayer(){ }

void RPCEfficiencyPerRingLayer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){}

void RPCEfficiencyPerRingLayer::beginJob(){

  dbe_ = edm::Service<DQMStore>().operator->();
}

void RPCEfficiencyPerRingLayer::beginRun(const edm::Run& r, const edm::EventSetup& c){

  if(dbe_ == 0) return;

  dbe_->setCurrentFolder(globalFolder_);
  EfficiencyPerRing = dbe_->book1D("EfficiencyPerRing","Efficiency per Ring",12,0.5,12.5);
  EfficiencyPerLayer = dbe_->book1D("EfficiencyPerLayer","Efficiency per Layer",6,0.5,6.5);


}

void RPCEfficiencyPerRingLayer::endRun(const edm::Run& r, const edm::EventSetup& c){

   std::stringstream meName1;
   MonitorElement * myMe1;

   std::stringstream meName2;
   MonitorElement * myMe2;

   std::stringstream binName;
   binName.str("");

   meName1.str("");
   meName1<<globalFolder_;

   meName2.str("");
   meName2<<globalFolder_;

   int k = 0;

   //EfficiencyPerRing

   for(int i=(-1*numberOfDisks_ ); i<=numberOfDisks_ ; i++){
	if(i == 0) continue;
	for(int j=innermostRings_; j<=3; j++){
		meName1.str("");
		meName2.str("");
		if(i < 0){
	       		meName1<<globalFolder_<<"Azimutal/ExGregDistroDm"<<abs(i)<<"R"<<j;
	       		meName2<<globalFolder_<<"Azimutal/OcGregDistroDm"<<abs(i)<<"R"<<j;
			//std::cout<<meName1.str()<<std::endl;
			//std::cout<<meName2.str()<<std::endl;
	   		binName.str("");
			binName<<"RE-"<<i<<"/"<<j;
			k++;}
		else if(i >0){
	       		meName1<<globalFolder_<<"Azimutal/ExGregDistroD"<<abs(i)<<"R"<<j;
	       		meName2<<globalFolder_<<"Azimutal/OcGregDistroD"<<abs(i)<<"R"<<j;
			//std::cout<<meName1.str()<<std::endl;
			//std::cout<<meName2.str()<<std::endl;
	   		binName.str("");
			binName<<"RE+"<<abs(i)<<"/"<<j;
			k++;}

		myMe1 = NULL; myMe1 = NULL;
       		myMe1 = dbe_->get(meName1.str());
       		myMe2 = dbe_->get(meName2.str());

		if(myMe1 && myMe2){
		  TH1 * histo1 = myMe1->getTH1();
		  TH1 * histo2 = myMe2->getTH1();
		  
		  int exg = histo1->Integral(); 
		  int obg = histo2->Integral();
		  double eff = 0;
		  double err = 0;
		  if(exg != 0){ 
		    eff = (double)obg/(double)exg; 
		    err = sqrt(eff*(1-eff)/(double)exg);}
		  
		  if(k%2 != 0 && k < 7){ 
		    EfficiencyPerRing->setBinContent(k+1, eff);
		    EfficiencyPerRing->setBinError(k+1, err);
		    EfficiencyPerRing->setBinLabel(k+1, binName.str());}
		  else if(k%2 == 0 && k < 7){
		    EfficiencyPerRing->setBinContent(k-1, eff);
		    EfficiencyPerRing->setBinError(k-1, err);
		    EfficiencyPerRing->setBinLabel(k-1, binName.str());}
		  else if(k >= 7){
		    EfficiencyPerRing->setBinContent(k, eff);
		    EfficiencyPerRing->setBinError(k, err);
		    EfficiencyPerRing->setBinLabel(k, binName.str());}
		}
	}
   }

   //EfficiencyPerLayer
   
   for(int i=1; i<=6; i++){
     
	int Exp = 0;
	int Obs = 0;
	double eff = 0;
	double err = 0;

	for(int j=-2; j<=2; j++){
		meName1.str("");
		meName2.str("");
	       	if(j < 0){ 
			meName1<<globalFolder_<<"BarrelPerLayer/ExpLayerWm"<<abs(j);
	       		meName2<<globalFolder_<<"BarrelPerLayer/ObsLayerWm"<<abs(j);}
		else{
			meName1<<globalFolder_<<"BarrelPerLayer/ExpLayerW"<<j;
	       		meName2<<globalFolder_<<"BarrelPerLayer/ObsLayerW"<<j;}
		myMe1 = NULL; myMe1 = NULL;
		myMe1 = dbe_->get(meName1.str());
	       	myMe2 = dbe_->get(meName2.str());
		if(myMe1 && myMe2){
			Exp += myMe1->getBinContent(i);
			Obs += myMe2->getBinContent(i);
		}
	}

	if(Exp != 0){
		eff = (double)Obs/(double)Exp;
		err = sqrt(eff*(1-eff)/(double)Exp);}
	EfficiencyPerLayer->setBinContent(i, eff);
 	EfficiencyPerLayer->setBinError(i, err);
   	binName.str("");
	binName<<"Layer "<<i;
	EfficiencyPerLayer->setBinLabel(i, binName.str());

  }

   if(SaveFile) dbe_->save(NameFile);

}

