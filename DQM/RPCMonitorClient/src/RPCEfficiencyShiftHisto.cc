//
// Original Author:  Cesare Calabria,161 R-006,
//         Created:  Tue Jul 20 12:58:43 CEST 2010

// Reviewed : Anna Cimmino  Tue Aug 16 10:20  2010


// user include files
#include "DQM/RPCMonitorClient/interface/RPCEfficiencyShiftHisto.h"
#include <sstream>

//CondFormats
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//DQM Services
#include "DQMServices/Core/interface/DQMStore.h"

RPCEfficiencyShiftHisto::RPCEfficiencyShiftHisto(const edm::ParameterSet& ps) {

  globalFolder_ = ps.getUntrackedParameter<std::string>("GlobalFolder", "RPC/RPCEfficiency/");
  effCut_= ps.getUntrackedParameter<int>("EffCut", 90);
  numberOfDisks_ = ps.getUntrackedParameter<int>("NumberOfEndcapDisks", 4);
}


RPCEfficiencyShiftHisto::~RPCEfficiencyShiftHisto(){ }

void RPCEfficiencyShiftHisto::beginJob(){}

void RPCEfficiencyShiftHisto::dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const&){}

void RPCEfficiencyShiftHisto::dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter) {

  ibooker.setCurrentFolder(globalFolder_);
  EffBarrelRoll = ibooker.book1D("EffBarrelRoll", "Barrel Efficiency",101,-0.5, 100.5);
  EffEndcapPlusRoll = ibooker.book1D("EffEndcapPlusRoll", "Endcap + Efficiency",101,-0.5, 100.5);
  EffEndcapMinusRoll = ibooker.book1D("EffEndcapMinusRoll", "Endcap - Efficiency",101,-0.5, 100.5);
  RollPercentage = ibooker.book2D("RollPercentage", "RollPercentage",1,0.,1.,3,0.,3.);

  RollPercentage->setBinLabel(1,"%",1);
  RollPercentage->setBinLabel(1,"E+",2);
  RollPercentage->setBinLabel(2,"B",2);
  RollPercentage->setBinLabel(3,"E-",2);


   std::stringstream meName;
   MonitorElement * myMe;
   
   meName.str("");
   meName<<globalFolder_;

   //Barrel

   int entriesBarrel = 0;
   int entriesBarrelBeyondEff = 0;
   float percBarrel = 0;

   for(int w = -2 ; w<3; w++){
     
       meName.str("");
       meName<<globalFolder_<<"Efficiency_Roll_vs_Sector_Wheel_"<<w;

       myMe = igetter.get(meName.str());
	 
       if(myMe){
	 for(int s = 1; s <= myMe->getNbinsX(); s++){
	   for(int r = 1;r <= myMe->getNbinsY(); r++){
	   
	   	double effBarrel = myMe->getBinContent(s,r);

		if(effBarrel >= 0){//5

		  if (EffBarrelRoll) EffBarrelRoll->Fill(effBarrel);
                	entriesBarrel++;

			if(effBarrel >= effCut_) entriesBarrelBeyondEff++;
		}
	   }
         }
       }
     }

     if(entriesBarrel != 0){
	percBarrel = 100*entriesBarrelBeyondEff/entriesBarrel;
	if(RollPercentage)RollPercentage->setBinContent(1,2,percBarrel);}


   //EndcapMinus

   int entriesEndcapMinus = 0;
   int entriesEndcapMinusBeyondEff = 0;
   float percEndcapMinus = 0;
   int entriesEndcapPlus = 0;
   int entriesEndcapPlusBeyondEff = 0;
   float percEndcapPlus = 0;
     for(int d = -numberOfDisks_ ; d <= numberOfDisks_; d++){
	 
       if(d == 0) continue;

	 meName.str("");
         meName<<globalFolder_<<"Efficiency_Roll_vs_Segment_Disk_"<<d;
         myMe = igetter.get(meName.str());

       
       if(myMe){    

	 for (int x = 1 ;x <= myMe->getNbinsX();x++){
	   for(int y = 1;y<=myMe->getNbinsY(); y++){

	       	double effEndcap = myMe->getBinContent(x,y);
	
		if(d<0 ){
		  entriesEndcapMinus++;	
		  if(EffEndcapMinusRoll)EffEndcapMinusRoll->Fill(effEndcap);
      		  if(effEndcap >= effCut_) entriesEndcapMinusBeyondEff++;
		}else {
		  entriesEndcapPlus++;
		  if(EffEndcapPlusRoll)EffEndcapPlusRoll->Fill(effEndcap);
		  if(effEndcap >= effCut_) entriesEndcapPlusBeyondEff++;
		}
	   }
	 }
       }
     }


     if(entriesEndcapMinus != 0){
       percEndcapMinus = 100*entriesEndcapMinusBeyondEff/entriesEndcapMinus;
       if( RollPercentage) RollPercentage->setBinContent(1,3,percEndcapMinus);}


     if(entriesEndcapPlus != 0){
	percEndcapPlus = 100*entriesEndcapPlusBeyondEff/entriesEndcapPlus;
	if(RollPercentage)	RollPercentage->setBinContent(1,1,percEndcapPlus);
     }

   
}


