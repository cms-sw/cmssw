////Trigger Primitive Converter
////
////Takes in raw information from the TriggerPrimitive class(part of L1TMuon software package);
////and outputs vector of 'ConvertedHits'
////

#include <cassert>
#include <iostream>
#include <fstream>

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "L1Trigger/L1TMuonEndCap/interface/PrimitiveConverter.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCPatternLUT.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include "L1Trigger/L1TMuonEndCap/interface/EmulatorClasses.h"
#include "L1Trigger/L1TMuonEndCap/interface/MakeRegionalCand.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

int ph_offsets[6][9] = { {39,  57,  76, 39,  58,  76, 41,  60,  79},
			 {95, 114, 132, 95, 114, 133, 98, 116, 135},
			 {38,  76, 113, 39,  58,  76, 95, 114, 132},
			 {38,  76, 113, 39,  58,  76, 95, 114, 132},
			 {38,  76, 113, 38,  57,  76, 95, 113, 132},
			 {21,  21,  23,  1,  21,   1, 21,   1,  20} };

bool neighbor(int endcap,int sector,int SectIndex,int id,int sub,int station){
  
  bool neighbor = false;
  
  int CompIndex = (endcap - 1)*6 + sector - 1;
  bool AdSector = false;
  if((SectIndex ==  0 && CompIndex ==  5) || 
     (SectIndex ==  1 && CompIndex ==  0) ||
     (SectIndex ==  2 && CompIndex ==  1) ||
     (SectIndex ==  3 && CompIndex ==  2) ||
     (SectIndex ==  4 && CompIndex ==  3) ||
     (SectIndex ==  5 && CompIndex ==  4) ||
     (SectIndex ==  6 && CompIndex == 11) || 
     (SectIndex ==  7 && CompIndex ==  6) ||
     (SectIndex ==  8 && CompIndex ==  7) ||
     (SectIndex ==  9 && CompIndex ==  8) ||
     (SectIndex == 10 && CompIndex ==  9) ||
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

std::vector<ConvertedHit> PrimitiveConverter::convert(std::vector<L1TMuon::TriggerPrimitive> TriggPrim, int SectIndex){
  
  //bool verbose = false;

  std::vector<ConvertedHit> ConvHits;
  for(std::vector<L1TMuon::TriggerPrimitive>::iterator C1 = TriggPrim.begin();C1 != TriggPrim.end();C1++){
    
    /////////////////////////////////////
    ///// get all input variables ///////
    /////////////////////////////////////
    L1TMuon::TriggerPrimitive C3 = *C1;
    CSCDetId Det = C3.detId<CSCDetId>();
    
    int station = Det.station(), chamber = Det.chamber(), ring = Det.ring(), wire = C3.getCSCData().keywire;
    int sector = Det.triggerSector(), strip = C3.getCSCData().strip, pattern = C3.getPattern(), Id = C3.Id();
    int quality = C3.getCSCData().quality, BX = C3.getCSCData().bx, endcap = Det.endcap();
    
    if (station == 1 && ring == 1 && strip > 127) ring = 4;
    int sub = 0;
	
    ////////////////////////
    /// Define Subsector ///
    ////// ME1 only ////////
    ////////////////////////
    
    if (station == 1)//is this correct? I know it starts from 1 so not quite what I intended I think.
      {
	if(chamber%6 > 2)
	  sub = 1;
	else
	  sub = 2;
      }
    
    bool IsNeighbor = neighbor(endcap,sector,SectIndex,Id,sub,station);
    
    if(ring == 4){Id += 9;}
    
    if( (SectIndex ==  (endcap - 1)*6 + sector - 1 ) || IsNeighbor ) {
	
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

      ////////////////////////////
      /// Define look-up index ///
      ////////////////////////////
      
      int LUTi = -999;
      int nId = Id;
      if(IsNeighbor){
	
	if(station < 2){
	  nId = 12 + Id/3;
	}
	else{
	  int xx = Id;
	  if(xx > 6) xx = 6;
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
      
      if(station == 1 || Id > 3){//10 Degree Chambers
	eightstrip = (eightstrip>>1);
	patcor = (patcor>>1);
	if(ring == 4 && strip > 127) eightstrip -= 512;
      }
      
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
      }
      else{
	int neighborId = Id/3;
	if(neighborId > 2) neighborId = 2;
	
	if(IsNeighbor) phInitIndex = 9 + neighborId;
	
	fph = Ph_Init_Neighbor_[SectIndex][station][phInitIndex - 1] + ph_tmp;
      }

      if (station == 0 || nId == -1 || SectIndex < 0 || SectIndex > 11 || LUTi < 0 || LUTi > 60) {
	LogDebug("L1TMuonEndCap") << "\n*********************************************************************" << std::endl;
	LogDebug("L1TMuonEndCap") << "EMTF malformed LCT: BX " << C3.getCSCData().bx << ", endcap " << Det.endcap() 
		  << ", station " << Det.station() << ", sector " << Det.triggerSector()
		  << ", ring " << Det.ring() << ", ID " << C3.Id() << ", chamber " << Det.chamber() 
		  << ", strip " << C3.getCSCData().strip << ", wire " << C3.getCSCData().keywire
		  << ", pattern " << C3.getPattern() << ", quality " << C3.getCSCData().quality << std::endl;
	LogDebug("L1TMuonEndCap") << "Produces: station " << station << ", nId " << nId
		  << ", SectIndex " << SectIndex << ", LUTi " << LUTi << std::endl;
	continue;
      }
      
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
	  if(ring == 4) idl = 16;
	}
	th_tmp = Th_LUT_St1_Neighbor_[subId-1][SectIndex][idl -1][wire];
      }
      else{
	int neighborId = Id/3;
	if(neighborId > 2) neighborId = 2;
	if(IsNeighbor) idl = 9 + neighborId;
	th_tmp = Th_LUT_St234_Neighbor_[station-2][SectIndex][idl-1][wire];
      }
		
      th = th_tmp + Th_Init_Neighbor_[SectIndex][LUTi];
      int rth = th;
	
      if(station == 1 && (ring == 1 || ring == 4) /*&& endcap == 1*/){
	
	index = (wire>>4)*32 + (eightstrip>>4);
	
	int corrIndex = Id;
	int subId = sub;
	if(corrIndex > 3) corrIndex -= 9;
			
	if(IsNeighbor && ring == 4){
	  subId = 1;
	  corrIndex++;
	}
	
	th_corr = Th_Corr_Neighbor_[subId-1][SectIndex][corrIndex-1][index];
		
	if(ph_reverse) th_corr = -th_corr;
	
	th_tmp += th_corr; // add correction to th_tmp
	if(th_tmp < 0 || wire == 0) th_tmp = 0;
			
	if(th_tmp > th_coverage)//this is one change that I'm not sure if it does anything good or not
	  th_tmp = th_coverage;	
		
	th_tmp &= 0x3f; //keep only lowest 6 bits
		
	if (th_tmp <= th_coverage) th = th_tmp + Th_Init_Neighbor_[SectIndex][LUTi];
	else th = rth; //was -999
      }
	
      ///////////////////////////
      //// Zones for ph_hits ////
      ///////////////////////////
      
      if(th != -999){
	phzvl = 0;
	if (th <= (ph_zone_bnd1 + zone_overlap)) phzvl |= 1;
	if (th > (ph_zone_bnd2 - zone_overlap)) phzvl |= 4;
	if ((th > (ph_zone_bnd1 - zone_overlap)) && (th <= (ph_zone_bnd2 + zone_overlap))) phzvl |= 2;
      }
	
      ////////////////////////////////////////////////////
      /////   Calculate Zhit and ZoneContribution    /////
      /////    Before the zone creation so it can    /////
      /////  Be an artifact of Converted Hit Class   /////
      ////////////////////////////////////////////////////
      
      if(ring == 4){
	Id -= 9;
	if(strip < 128) strip += 128;
      }
      
      //determination of zone contribution
      int zoneword = 0, zhit = -99, zmask[4] = {1,2,4,8};
      bool zoneConditions[4] {((phzvl & 1) && (Id < 4)),
	  (((phzvl & 2) && (Id < 4) && (station < 3)) || ((phzvl & 1) && (Id > 3) && (station > 2))),
	  (((phzvl & 1) && (Id > 3) && (Id < 7) && (station == 1)) || ((phzvl & 1) && (Id > 3) && (station == 2)) || ((phzvl & 2) && (Id > 3) && (station > 2))),
	  ( ((station == 1) && (Id > 6)) || ((phzvl & 2) && (Id > 3) && (station == 2)) || ((phzvl & 4) && (station == 3) && (Id > 3)) )};
      
      for(int z=0;z<4;z++){
	if(zoneConditions[z]) zoneword |= zmask[z];
      }
      
      int cindex = Id - 1;
      int sindex = station;
      if(sub == 1) sindex--;
      
      if(IsNeighbor){
	sindex = 5;
	if(station == 1) cindex = Id/3 - 1;
	else cindex = (station - 1)*2 + ((Id > 6) ? 2:1);
      }
      
      zhit = ph_hit + ph_offsets[sindex][cindex];
      
      ///////////////////////////////////////////////////////
      //////// set vector of ConvertedHits to move //////////
      /////////   Converted TP's around code   //////////////
      ///////////////////////////////////////////////////////
	
      ConvertedHit Hit;
      
      int in = 0;
      if(IsNeighbor) in = 1;
      
      Hit.SetValues(fph,th,ph_hit,phzvl,station,sub,Id,quality,pattern,wire,strip,BX);
      Hit.AddTheta(th);
      Hit.SetTP(C3);
      Hit.SetZhit(zhit);
      Hit.SetSectorIndex(SectIndex);
      Hit.SetNeighbor(in);
      Hit.SetZoneWord(zoneword);
      
      if(Hit.Theta() != -999 && Hit.Phi() > 0 ){//if theta is valid
	ConvHits.push_back(Hit);
      }
      else {
	LogDebug("L1TMuonEndCap") << "\n#####################################################################" << std::endl;
	LogDebug("L1TMuonEndCap") << "LCT w/o theta/phi: BX " << C3.getCSCData().bx << ", endcap " << Det.endcap() 
		  << ", station " << Det.station() << ", sector " << Det.triggerSector()
		  << ", ring " << Det.ring() << ", ID " << C3.Id() << ", chamber " << Det.chamber() 
		  << ", strip " << C3.getCSCData().strip << ", wire " << C3.getCSCData().keywire
		  << ", pattern " << C3.getPattern() << ", quality " << C3.getCSCData().quality << std::endl;
	LogDebug("L1TMuonEndCap") << "Has fph " << fph << ", th " << th << ", ph_hit " << ph_hit 
		  << ", phzvl " << phzvl << ", station " << station << ", sub " << sub
		  << ", Id " << Id << ", quality " << quality << ", pattern " << pattern
		  << ", wire " << wire << ", strip " << strip << ", BX " << BX << std::endl;
      }
    
    } //if sector == sectIndex
  }
  return ConvHits;  
}
