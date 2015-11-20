////Function to calculte pT for a given InternalTrack
////
////2985826856 old checksum11 in DataFormats/L1TMuon/src/classes_def.xml
////1494215132 12

#include "EmulatorClasses.h"
#include "L1TMuonEndCapTrackProducer.h"
#include "Forest.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float CalculatePt_FullPrecision(L1TMuon::InternalTrack track){

	bool verbose = false;

	int dphi[6] = {-999,-999,-999,-999,-999,-999}, deta[6] = {-999,-999,-999,-999,-999,-999};
	int clct[4] = {-999,-999,-999,-999}, cscid[4] = {-999,-999,-999,-999};
	int phis[4] = {-999,-999,-999,-999}, etas[4] = {-999,-999,-999,-999}, mode = 0;;
	
	float theta_angle = ((track.theta)*0.2874016 + 8.5)*(3.14159265359/180);
	float eta = (-1)*log(tan(theta_angle/2));

	const TriggerPrimitiveStationMap stubs = track.getStubs();
		
	if(verbose) std::cout<<"Track eta = "<<eta<<" and has hits in stations ";//
	
	int x=0;
	for(unsigned int s=8;s<12;s++){
		if((stubs.find(s)->second).size() == 1){
			
			if(verbose) std::cout<<(stubs.find(s)->second)[0]->detId<CSCDetId>().station()<<" ";
			
			etas[s-8] = (fabs((stubs.find(s)->second)[0]->getCMSGlobalEta()) + 0.9)/(0.0125);
			phis[s-8] = track.phis[x];//(stubs.find(s)->second)[0]->getCMSGlobalPhi();//
			clct[s-8] = (stubs.find(s)->second)[0]->getPattern();
			cscid[s-8] = (stubs.find(s)->second)[0]->Id();
			
			switch(s-7){
				case 1: mode |= 1;break;
				case 2: mode |= 2;break;
				case 3: mode |= 4;break;
				case 4: mode |= 8;break;
				default: mode |= 0;
			}
			x++;
		}
	}
	
	if(verbose) std::cout<<"\nMode = "<<mode<<std::endl; 
	
	//////////////////////////////////////////////////
	//// Calculate Delta Phi and Eta Combinations ////
	//////////////////////////////////////////////////
	
    if(phis[0] > 0 && phis[1] > 0){
		dphi[0] = phis[1] - phis[0];
		deta[0] = etas[1] - etas[0];
	}
	if(phis[0] > 0 && phis[2] > 0){
		dphi[1] = phis[2] - phis[0];
		deta[1] = etas[2] - etas[0];
	}
	if(phis[0] > 0 && phis[3] > 0){
		dphi[2] = phis[3] - phis[0];
		deta[2] = etas[3] - etas[0];
	}
	if(phis[1] > 0 && phis[2] > 0){
		dphi[3] = phis[2] - phis[1];
		deta[3] = etas[2] - etas[1];
	}
	if(phis[1] > 0 && phis[3] > 0){
		dphi[4] = phis[3] - phis[1];
		deta[4] = etas[3] - etas[1];
	}
	if(phis[2] > 0 && phis[3] > 0){
		dphi[5] = phis[3] - phis[2];
		deta[5] = etas[3] - etas[2];
	}
	
	
	if(verbose){
		for(int f=0;f<4;f++){
			std::cout<<"\nphis["<<f<<"] = "<<phis[f]<<" and etas = "<<etas[f]<<std::endl;
			std::cout<<"\nclct["<<f<<"] = "<<clct[f]<<" and cscid = "<<cscid[f]<<std::endl;
		}
	
		for(int u=0;u<6;u++)
			std::cout<<"\ndphi["<<u<<"] = "<<dphi[u]<<" and deta = "<<deta[u]<<std::endl;
	}
	
	float MpT = -1;//final pT to return
	
	
	///////////////////////////////////////////////////////////////////////////////
	//// Variables is a array of all possible variables used in pT calculation ////
	///////////////////////////////////////////////////////////////////////////////
	
	int size[13] = {4,0,4,4,5,0,4,4,5,4,5,5,6};
	int Variables[20] = {dphi[0], dphi[1], dphi[2], dphi[3], dphi[4], dphi[5], deta[0], deta[1], deta[2], deta[3], deta[4], deta[5],
								clct[0], clct[1], clct[2], clct[3], cscid[0], cscid[1], cscid[2], cscid[3]};
	
	
	///////////////////////
	/// Mode Variables ////
	///////////////////////
	
	//ModeVariables is a 2D arrary indexed by [TrackMode(13 Total Listed Below)][VariableNumber(20 Total Constructed Above)]
	
	//3:TrackEta:dPhi12:dEta12:CLCT1:cscid1
	//4:Single Station Track Not Possible
	//5:TrackEta:dPhi13:dEta13:CLCT1:cscid1
	//6:TrackEta:dPhi23:dEta23:CLCT2:cscid2
	//7:TrackEta:dPhi12:dPhi23:dEta13:CLCT1:cscid1
	//8:Single Station Track Not Possible
	//9:TrackEta:dPhi14:dEta14:CLCT1:cscid1
	//10:TrackEta:dPhi24:dEta24:CLCT2:cscid2 
	//11:TrackEta:dPhi12:dPhi24:dEta14:CLCT1:cscid1
	//12:TrackEta:dPhi34:dEta34:CLCT3:cscid3
	//13:TrackEta:dPhi13:dPhi34:dEta14:CLCT1:cscid1
	//14:TrackEta:dPhi23:dPhi34:dEta24:CLCT2:cscid2
	//15:TrackEta:dPhi12:dPhi23:dPhi34:dEta14:CLCT1:cscid1 
	
	int ModeVariables[13][6] = {{0,6,12,16,-999,-999},{-999,-999,-999,-999,-999,-999},{1,7,12,16,-999,-999},{3,9,13,17,-999,-999},{0,3,7,12,16,-999},
								  {-999,-999,-999,-999,-999,-999},{2,8,12,16,-999,-999},{4,10,13,17,-999,-999},{0,4,8,12,16,-999},{5,11,14,18,-999,-999},
								  {1,5,8,12,16,-999},{3,5,10,13,17,-999},{0,3,5,8,12,16}};
								
	
	////////////////////////
	//// pT Calculation ////
	////////////////////////
	//float gpt = -1;
	for(int i=3;i<16;i++){
	
		if(i != mode)
			continue;
			
		if(verbose) std::cout<<"\nMode = "<<mode<<"\n\n";
		
		Forest *forest = new Forest();
		const char *dir = "L1Trigger/L1TMuon/data/emtf_luts/ModeVariables/trees";
		std::stringstream ss;
        ss << dir << "/" << mode;//
		
		forest-> loadForestFromXML(ss.str().c_str(),64);
		
		std::vector<Double_t> Data;
		Data.push_back(1.0);
		Data.push_back(eta);
		for(int y=0;y<size[mode-3];y++){
			
			Data.push_back(Variables[ModeVariables[mode-3][y]]);
			if(verbose) std::cout<<"Generalized Variables "<<y<<" "<<Variables[ModeVariables[mode-3][y]]<<"\n";
		}
		
		if(verbose){
		std::cout<<"Data.size() = "<<Data.size()<<"\n";
		for(int i=0;i<5;i++)  
		  std::cout<<"Data["<<i<<"] = "<<Data[i]<<"\n";
		}
		
		Event *event = new Event();
		event->data = Data;
		
		std::vector<Event*> vevent;
		vevent.push_back(event);
		
		forest->predictEvents(vevent,64);
		
		float OpT = vevent[0]->predictedValue;
		MpT = 1/OpT;
	}

	return MpT;
}
