////Trigger Primitive Converter
////
////Takes in raw information from the TriggerPrimitive class(part of L1TMuon software package);
////and outputs vector of 'ConvertedHits'
////
////
////
////

#include <cassert>
#include <iostream>
#include <fstream>

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "L1Trigger/L1TMuonEndCap/interface/PrimitiveConverter.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCPatternLUT.h"
#include "L1Trigger/CSCTrackFinder/test/src/RefTrack.h"

#include "L1Trigger/L1TMuonEndCap/interface/EmulatorClasses.h"

using namespace std;

int ph_offsets_neighbor[5][10][3] = {{{39,39,-99}  ,{57,57,-99}   ,{76,76,-99}   ,{39,-99,-99} ,{58,-99,-99}  ,{76,-99,-99}  ,{41,-99,-99} ,{60,-99,-99}   ,{79,-99,-99}     ,{21,21,23}  },//not sure if 23 is done right yet
			   						 {{95,95,-99}  ,{114,114,-99} ,{132,132,-99} ,{95,-99,-99} ,{114,-99,-99} ,{133,-99,-99} ,{98,-99,-99} ,{116,-99,-99}  ,{135,-99,-99}    ,{21,21,23}  },//not sure if 23 is done right yet
			   						 {{38,38,-99}  ,{76,76,-99}   ,{113,113,-99} ,{39,39,-99}  ,{58,58,-99}   ,{76,76,-99}   ,{95,95,-99}  ,{114,114,-99}  ,{132,132,-99}    ,{1,21,21}  },
			   						 {{38,-99,-99} ,{76,-99,-99}  ,{113,-99,-99} ,{39,39,39}   ,{58,58,58}    ,{76,76,76}    ,{95,95,95}   ,{114,114,114}  ,{132,132,132}    ,{1,21,21}  },
			   						 {{38,-99,-99} ,{76,-99,-99}  ,{113,-99,-99} ,{38,38,-99}  ,{57,57,-99}   ,{76,76,-99}   ,{95,95,-99}  ,{113,113,-99}  ,{132,132,-99}    ,{1,20,20}  }};//[station][id][phzvl look up #(-99 indicates invaled entry)]




bool neighbor(int endcap,int sector,int SectIndex,int id,int sub,int station){

	bool neighbor = false;
	
	int CompIndex = (endcap - 1)*6 + sector - 1;
	bool AdSector = false;
	if((SectIndex == 0 && CompIndex == 5) || 
	   (SectIndex == 1 && CompIndex == 0) ||
	   (SectIndex == 2 && CompIndex == 1) ||
	   (SectIndex == 3 && CompIndex == 2) ||
	   (SectIndex == 4 && CompIndex == 3) ||
	   (SectIndex == 5 && CompIndex == 4) ||
	   (SectIndex == 6 && CompIndex == 11) || 
	   (SectIndex == 7 && CompIndex == 6) ||
	   (SectIndex == 8 && CompIndex == 7) ||
	   (SectIndex == 9 && CompIndex == 8) ||
	   (SectIndex == 10 && CompIndex == 9) ||
	   (SectIndex == 11 && CompIndex == 10) ){AdSector = true;}
	
	if(AdSector && sub == 2 && station == 1 && (id == 3 || id == 6 || id == 9) )
		neighbor = true;
		
	if(AdSector && station > 1 && (id == 3 || id == 9) )
		neighbor = true;
	
	return neighbor;

}

template <typename T>
static void read_buf(const char * name, T * buf, unsigned size){
  ifstream input(edm::FileInPath(name).fullPath().c_str());
  for (unsigned i=0; i<size; i++){
    input >> buf[i];
  }
}


PrimitiveConverter::PrimitiveConverter(){
  read_buf("L1Trigger/L1TMuon/data/emtf_luts/ph_lut_v1/ph_disp_neighbor.txt",      &Ph_Disp_Neighbor_[0][0], 12*61); 
  read_buf("L1Trigger/L1TMuon/data/emtf_luts/ph_lut_v1/ph_init_neighbor.txt",      &Ph_Init_Neighbor_[0][0][0], 12*5*16); 
  read_buf("L1Trigger/L1TMuon/data/emtf_luts/ph_lut_v1/th_corr_neighbor.txt",      &Th_Corr_Neighbor_[0][0][0][0], 2*12*4*96); 
  read_buf("L1Trigger/L1TMuon/data/emtf_luts/ph_lut_v1/th_init_neighbor.txt",      &Th_Init_Neighbor_[0][0], 12*61); 
  read_buf("L1Trigger/L1TMuon/data/emtf_luts/ph_lut_v1/th_lut_st1_neighbor.txt",   &Th_LUT_St1_Neighbor_[0][0][0][0], 2*12*16*64); 
  read_buf("L1Trigger/L1TMuon/data/emtf_luts/ph_lut_v1/th_lut_st234_neighbor.txt", &Th_LUT_St234_Neighbor_[0][0][0][0], 3*12*11*112);   


}

std::vector<ConvertedHit> PrimitiveConverter::convert(std::vector<TriggerPrimitive> TriggPrim, int SectIndex){

	//bool verbose = false;

	std::vector<ConvertedHit> ConvHits;
	for(std::vector<TriggerPrimitive>::iterator C1 = TriggPrim.begin();C1 != TriggPrim.end();C1++){
	
  	/////////////////////////////////////
  	///// get all input variables ///////
	/////////////////////////////////////
	TriggerPrimitive C3 = *C1;
	CSCDetId Det = C3.detId<CSCDetId>();
	
	int station = Det.station(), chamber = Det.chamber(), ring = Det.ring(), wire = C3.getCSCData().keywire, sector = Det.triggerSector(), strip = C3.getCSCData().strip; 
	int pattern = C3.getPattern(), Id = C3.Id(), quality = C3.getCSCData().quality, BX = C3.getCSCData().bx, endcap = Det.endcap();
	
	if(station == 1 && ring == 1 && strip > 127){
	  ring = 4;
	}
	
	int sub = 0;
	
	////////////////////////
	/// Define Subsector ///
	////// ME1 only ////////
	////////////////////////
	
	if(station == 1)//is this correct? I know it starts from 1 so not quite what I intended I think.
	{
	
		if(chamber%6 > 2)
			sub = 1;
		else
			sub = 2;
		
	}
	
	bool IsNeighbor = neighbor(endcap,sector,SectIndex,Id,sub,station);
	
		
	if(ring == 4){Id += 9;}

	//if(endcap == 1 && sector == 1)//
	if( (SectIndex ==  (endcap - 1)*6 + sector - 1 )  || IsNeighbor )
	{
	
		
	//if(verbose){
	// 	std::cout<<"\n\nSECTOR "<<SectIndex<<"\n\n";
	// 	std::cout<<"\n\nRING = "<<ring<<"\n\n";
	//}
	
	/////////////////////////////////////
	//////// define/set variables////////
	/////////////////////////////////////
	
	int ph_tmp = -999, th_tmp = -999;
	int clctpatcor = -999, clctpatsign = -999;
	int eightstrip = -999;
	
	int factor = (station == 1 && ring == 4) ? 1707://ME1/1a1707?
		     (station == 1 && ring == 3) ? 947: //ME1/3//changed Id > 6 to ring == 3.
		     (station == 1 && ring == 1) ? 1301://ME1/1b
			1024;//all other chambers
			
	//std::cout<<"factor = "<<factor<<std::endl;
		
	bool ph_reverse = (endcap == 1 && station >= 3) ? 1:
			  (endcap == 2 && station < 3) ? 1: 0;
		
	int ph_coverage  = (station <= 1 && Id > 6 && Id < 10) ? 15 : //30 :
		(station >= 2 && Id <= 3) ? 40 : 20; //80 : 40;
		
	int th_coverage =  (station == 1 && Id <= 3) ? 45 :
	 		   (station == 1 && Id > 6 && Id < 10) ? 27 :
			   (station == 1 && Id > 3) ? 39 :
			   (station == 2 && Id <= 3) ? 43 :
			   (station == 2 && Id > 3) ? 56 :
			   (station == 3 && Id <= 3) ? 34 :
			   (station == 3 && Id > 3) ? 52 :
			   (station == 4 && Id <= 3) ? 28 :
			   (station == 4 && Id > 3) ? 50 : 0;
		
	int ph_zone_bnd1 = (station <= 1 && (Id <= 3 || Id > 9)) ? 41 :
			   (station == 2 && Id <= 3) ? 41 :
			   (station == 2 && Id >  3) ? 87 :
			   (station == 3 && Id >  3) ? 49 :
			   (station == 4 && Id >  3) ? 49 : 127;
												   
	int ph_zone_bnd2 = (station == 3 && Id >  3) ? 87 : 127;
						
	int zone_overlap = 2;
		
	int fph = -999, th = -999, ph_hit = -999, phzvl = -999;// th_hit = -999, ////
	
	
	
	//if(sub && verbose)
	//	std::cout<<"\nsub = "<<sub<<"\n";

	////////////////////////////
	/// Define look-up index ///
	////////////////////////////
	
	int LUTi = -999;
	int nId = Id;
	if(IsNeighbor){
		
		if(station < 2){
		
			nId = 12 + Id/3;
			if(ring == 4)
				nId ++;
		
		}
		else{
			int xx = Id;
			if(xx > 6)
				xx = 6;
				
			nId =  9 + xx/3;
			
		}
		
	}
	switch(station)
	{
		case 1: 
			switch(sub)
			{
				case 1: LUTi = nId - 1;break;
				case 2: LUTi = 15 + nId;break;
				default:;//std::cout<<"Sub is out of range"<<std::endl;
			}
			break;
		case 2: LUTi = 27 + nId;break;
		case 3: LUTi = 38 + nId;break;
		case 4: LUTi = 49 + nId;break;
		default:;//std::cout<<"station is out of range"<<std::endl;
	}
	
	if(IsNeighbor && station == 1){
	
		switch(sub)
		{
			case 1: LUTi = 15 + nId;break;
			case 2: LUTi = nId - 1;break;
			default:;//std::cout<<"Sub is out of range"<<std::endl;
		}
	
	}
	
	
	
	/////////////////////////////////////
	//////// CLCT Pattern Correc ////////
	/////////////////////////////////////
	
	clctpatcor = 0;
	clctpatsign = 0;
	
	if(pattern > 0 && pattern < 11){
		
		clctpatsign = ((pattern%2) == 0);
		
		if(pattern >= 2) {clctpatcor = 5;}
		if(pattern >= 6) {clctpatcor = 2;}
		if(pattern == 10) {clctpatcor = 0;clctpatsign = 0;}
	}
	
	

	//////////////////////////////////////
	///////// chamber phi offset /////////
	//////////////////////////////////////
	
	eightstrip = 8*strip;
	int patcor = clctpatcor;
	
	//std::cout<<"8strip = "<<eightstrip<<"\n";
	
	if(station == 1 || Id > 3){//10 Degree Chambers
	
		eightstrip = (eightstrip>>1);
		patcor = (patcor>>1);
		if(ring == 4 && strip > 127) eightstrip -= 512;
	}
	
	//std::cout<<"8strip = "<<eightstrip<<"\n";	
	
	if(clctpatsign) patcor = -patcor;
	eightstrip += patcor;
	
	//////////////////////
	////Phi Conversion////
	//////////////////////
	
	ph_tmp = ((eightstrip*factor)>>10);
	int phShift = (ph_tmp>>5);
	int phLow = 0;

	if(ph_reverse){
	
		ph_tmp = -ph_tmp;
		phShift = -phShift;
		phLow = ph_coverage;
	}
	
	int phInitIndex = Id;
	//std::cout<<"phInitIndex = "<<phInitIndex<<" and ph_tmp = "<<ph_tmp<<"\n";
	
	if(station == 1){
	
		int neighborId = C3.Id()/3;
		int subId = sub;
		if(IsNeighbor ){
	
			subId = 1;
			phInitIndex = 12 + neighborId;
			if(ring == 4)
				phInitIndex = 16;//phInitIndex++;
	
		
		}
		
		fph = Ph_Init_Neighbor_[SectIndex][subId-1][phInitIndex - 1] + ph_tmp;
		//std::cout<<"ph init = "<<Ph_Init_Neighbor_[SectIndex][subId-1][phInitIndex - 1]<<", index = "<<phInitIndex<<", neighborId = "<<neighborId<<", Id = "<<Id<<"\n";
	}
	else{
	
		int neighborId = Id/3;
		if(neighborId > 2)
			neighborId = 2;
		
		if(IsNeighbor)
			phInitIndex = 9 + neighborId;

		fph = Ph_Init_Neighbor_[SectIndex][station][phInitIndex - 1] + ph_tmp;
	}
	
	//std::cout<<"pl = "<<phLow<<", ps = "<<phShift<<", ph disp = "<<Ph_Disp_Neighbor_[SectIndex][LUTi]<<", >>1 = "<<(Ph_Disp_Neighbor_[SectIndex][LUTi]>>1)<<", LUTi = "<<LUTi<<"\n";
	ph_hit = phLow + phShift + (Ph_Disp_Neighbor_[SectIndex][LUTi]>>1);
	
	////////////////////////
	////Theta Conversion////
	////////////////////////
		
	int index = -999;
	int th_corr = -999;	
	
	int idl = Id;
	
	if(station == 1){
	
		
		int neighborId = C3.Id()/3;
		int subId = sub;
		if(IsNeighbor){
	
			subId = 1;
			idl = 12 + neighborId;
			if(ring == 4)
				idl = 16;
	
		}
		
		//std::cout<<"nid = "<<neighborId<<", idl = "<<idl<<"\n";
		
		th_tmp = Th_LUT_St1_Neighbor_[subId-1][SectIndex][idl -1][wire];
		//std::cout<<"th_tmpr = "<<th_tmp<<"\n";
	}
	else{
		
		
		int neighborId = Id/3;
		if(neighborId > 2)
			neighborId = 2;
		
		if(IsNeighbor)
			idl = 9 + neighborId;
		
		th_tmp = Th_LUT_St234_Neighbor_[station-2][SectIndex][idl-1][wire];
		//if(verbose) std::cout<<"\n\nth_tmpr = "<<th_tmp<<"\n\n";
	}
	
		
	th = th_tmp + Th_Init_Neighbor_[SectIndex][LUTi];
	int rth = th;
	//std::cout<<"Th_Init_Neighbor_["<<SectIndex<<"]["<<LUTi<<"] = "<<Th_Init_Neighbor_[SectIndex][LUTi]<<"\n";
	
	if(station == 1 && (ring == 1 || ring == 4) /*&& endcap == 1*/){
	
		index = (wire>>4)*32 + (eightstrip>>4);
		
		int corrIndex = Id;
		int subId = sub;
		//std::cout<<"corrIndex = "<<corrIndex<<"\n";
		if(corrIndex > 3)
			corrIndex -= 9;
			
		//std::cout<<"corrIndex = "<<corrIndex<<"\n";
			
		if(IsNeighbor && ring == 4){
			subId = 1;
			corrIndex++;
		}
			
		//std::cout<<"corrIndex = "<<corrIndex<<"\n";
		
		//if(Id > 3){
		//	th_corr = Th_Corr_Neighbor_[sub-1][SectIndex][Id-10][index];
			//if(verbose) std::cout<<"\n\nth_corr = "<<th_corr<<"\n\n";
		//}
		//else{
			th_corr = Th_Corr_Neighbor_[subId-1][SectIndex][corrIndex-1][index];
			//std::cout<<"th_corr["<<subId-1<<"]["<<SectIndex<<"]["<<corrIndex-1<<"] = "<<th_corr<<"\n";
		//}
		
		
		if(ph_reverse) th_corr = -th_corr;
		
		//std::cout<<"th_tmp = "<<th_tmp<<"\n";
		
		th_tmp += th_corr;                  //add correction to th_tmp
		//std::cout<<"th_tmp = "<<th_tmp<<"\n";
		if(th_tmp < 0)
			th_tmp = 0;
		th_tmp &= 0x3f;                     //keep only lowest 6 bits
		//std::cout<<"th_tmp = "<<th_tmp<<"\n";
		//std::cout<<"coverage = "<<th_coverage<<"\n";
		
		if(th_tmp < th_coverage){
		
			//if(ring == 1){LUTi += 9;}  //change because new Verilog3 sp_tf treats ME11b with LUT's of ME11a
			
			th = th_tmp + Th_Init_Neighbor_[SectIndex][LUTi];
			//std::cout<<"th_init["<<SectIndex<<"]["<<LUTi<<"] = "<<Th_Init_Neighbor_[SectIndex][LUTi]<<"\n";
		}
		else{th = rth;}//was -999
 
	}
	
	///////////////////////////
	//// Zones for ph_hits ////
	///////////////////////////
	
	if(th != -999){
	
		phzvl = 0;
		
		if(th <= (ph_zone_bnd1 + zone_overlap)) 
			{phzvl |= 1;}
			
		if(th > (ph_zone_bnd2 - zone_overlap)) 
			{phzvl |= 4;}
			
		if((th > (ph_zone_bnd1 - zone_overlap)) && (th <= (ph_zone_bnd2 + zone_overlap)))
			{phzvl |= 2;}
	}
	
	
	
	////////////////////////////////////////////////////
	/////   Calculate Zhit and ZoneContribution    /////
	/////    Before the zone creation so it can    /////
	/////  Be an artifact of Converted Hit Class   /////
	////////////////////////////////////////////////////
	
	
	
	int zhit = -99, pz = -99;
	std::vector<int> zonecontribution; //Each hit could go in more than one zone so we make a vector which stores all the zones for which this hit will contribute

	if(ring == 4){
		Id -= 9;
		
		if(strip < 128)
			strip += 128;
	}
	
	
	//determination of zone contribution
	if((phzvl & 1) && (Id < 4 || Id > 9)){pz=0;zonecontribution.push_back(0);}
	if((phzvl & 2) && (Id < 4)){pz=1;zonecontribution.push_back(1);}
	if((phzvl & 1) && (Id > 3) && (station > 2)){pz=0;zonecontribution.push_back(1);}
	if((phzvl & 1) && (Id > 3) && (Id < 7) && (station == 1)){pz=0;zonecontribution.push_back(2);}
	if((phzvl & 1) && (Id > 3) && (station == 2)){pz=0;zonecontribution.push_back(2);}
	if((phzvl & 2) && (Id > 3) && (station > 2)){pz=1;zonecontribution.push_back(2);}
	if((phzvl & 1) && (Id > 4) && (station < 2)){pz=0;zonecontribution.push_back(3);}
	if(phzvl & 4){pz=2;zonecontribution.push_back(3);}
	if((phzvl & 2) && (Id > 3) && (station < 3)){pz=1;zonecontribution.push_back(3);}
	
	
	int phOffIndex = Id;
	if(IsNeighbor)
		phOffIndex = 10;
	
	//applying ph_offsets
	if(sub == 1){
		zhit = ph_hit + ph_offsets_neighbor[station-1][phOffIndex-1][pz];
		//std::cout<<"\nph_hit = "<<ph_hit<<" and ph_offsets_neighbor["<<station-1<<"]["<<phOffIndex-1<<"]["<<pz<<"] = "<<ph_offsets_neighbor[station-1][phOffIndex-1][pz]<<"\n";
	}
	else{
			
		zhit = ph_hit + ph_offsets_neighbor[station][phOffIndex-1][pz];
		//std::cout<<"ph_hit = "<<ph_hit<<" and ph_offsets_neighbor["<<station<<"]["<<phOffIndex-1<<"]["<<pz<<"] = "<<ph_offsets_neighbor[station][phOffIndex-1][pz]<<"\n";
	}
	
	
		
	
		
	///////////////////////////////////////////////////////
	//////// set vector of ConvertedHits to move //////////
	/////////   Converted TP's around code   //////////////
	///////////////////////////////////////////////////////
	
	
	//if(SectIndex == 8){
		//std::cout<<"phi = "<<fph<<", theta = "<<th<<", ph_hit = "<<ph_hit<<",zhit = "<<zhit<<", station = "<<station<<", ring = "<<ring<<", id = "<<Id<<", sector "<<SectIndex<<",sub = "<<sub<<", strip = "<<strip<<", wire = "<<wire<<", IsNeighbor = "<<IsNeighbor<<"\n";
	
		//std::cout<<BX-3<<" "<<endcap<<" "<<sector<<" "<<sub<<" "<<station<<" 1 "<<quality<<" "<<pattern<<" "<<wire<<" "<<C3.Id()<<" 0 "<<strip<<"\n";
	//}
	
	/* if(station != 1) */
	/* 	sub = 1; */
	/* std::cout<<"proper FR[0] = "<<FRLUT[endcap-1][sector-1][station-1][sub-1][Id-1]<<"\n"; */
	/* if(station != 1) */
	/* 	sub = 0; */

	
	ConvertedHit Hit;
	
	int in = 0;
	if(IsNeighbor)
		in = 1;

	Hit.SetValues(fph,th,ph_hit,phzvl,station,sub,Id,quality,pattern,wire,strip,BX);
	Hit.SetTP(C3);
	Hit.SetZhit(zhit);
	Hit.SetZoneContribution(zonecontribution);
	Hit.SetSectorIndex(SectIndex);
	Hit.SetNeighbor(in);

	if(Hit.Theta() != -999 && Hit.Phi() > 0){//if theta is valid
		ConvHits.push_back(Hit);
		/*if(verbose){	
			std::cout<<"Phzvl() = "<<Hit.Phzvl()<<", ph_hit = "<<Hit.Ph_hit()<<", station = "<<Hit.Station()<<" and id = "<<Hit.Id()<<std::endl;
			std::cout<<"strip = "<<strip<<", wire = "<<wire<<" and zhit = "<<zhit<<std::endl;
			std::cout<<"\n\nIn Zones: ";
			for(std::vector<int>::iterator in = zonecontribution.begin();in!=zonecontribution.end();in++){
				std::cout<<" "<<*in<<" ";
			}
		}*/
	}	
	
	
	}//if sector == sectIndex
	
    }
    return ConvHits;
    
}

