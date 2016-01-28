////Trigger Primitive Converter
////
////Takes in raw information from the TriggerPrimitive class(part of L1TMuon software package);
////and outputs vector of 'ConvertedHits'
////
////
////
////

#include "L1Trigger/L1TMuonEndCap/interface/EmulatorClasses.h"
#include "L1Trigger/L1TMuonEndCap/interface/PhThLUTs.h"

		     
int ph_offsetss[5][9][3] = {{{2,2,-99},{20,20,-99},{39,39,-99},{2,-99,-99},{21,-99,-99},{39,-99,-99},{4,-99,-99},{23,-99,-99},{42,-99,-99}},
			   {{58,58,-99},{77,77,-99},{95,95,-99},{58,-99,-99},{77,-99,-99},{96,-99,-99},{61,-99,-99},{79,-99,-99},{98,-99,-99}},
			   {{1,1,-99},{39,39,-99},{76,76,-99},{2,2,-99},{21,21,-99},{39,39,-99},{58,58,-99},{77,77,-99},{95,95,-99}},
			   {{1,-99,-99},{39,-99,-99},{76,-99,-99},{2,2,2},{21,21,21},{39,39,39},{58,58,58},{77,77,77},{95,95,95}},
			   {{1,-99,-99},{39,-99,-99},{76,-99,-99},{2,1,-99},{21,20,-99},{39,39,-99},{58,58,-99},{77,76,-99},{95,95,-99}}};//[station][id][phzvl look up #(-99 indicates invaled entry)]


std::vector<ConvertedHit> PrimConv(std::vector<TriggerPrimitive> TriggPrim, int SectIndex){

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
	
	if(ring == 4){Id += 9;}

	//if(endcap == 1 && sector == 1)//
	if(SectIndex ==  (endcap - 1)*6 + sector - 1)
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
	
	int sub = 0;
	
	////////////////////////
	/// Define Subsector ///
	////// ME1 only ////////
	////////////////////////
	
	if(station == 1)
	{
	
		if(chamber%6 > 2)
			sub = 1;
		else
			sub = 2;
		
	}
	
	//if(sub && verbose)
	//	std::cout<<"\nsub = "<<sub<<"\n";

	////////////////////////////
	/// Define look-up index ///
	////////////////////////////
	
	int LUTi = -999;
	switch(station)
	{
		case 1: 
			switch(sub)
			{
				case 1: LUTi = Id - 1;break;
				case 2: LUTi = 11 + Id;break;
				default:;//std::cout<<"Sub is out of range"<<std::endl;
			}
			break;
		case 2: LUTi = 23 + Id;break;
		case 3: LUTi = 32 + Id;break;
		case 4: LUTi = 41 + Id;break;
		default:;//std::cout<<"station is out of range"<<std::endl;
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
		//if(ring == 4) eightstrip -= 512;
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
	
	
	if(station == 1){
		
		fph = PhInit[SectIndex][sub-1][Id - 1] + ph_tmp;
	}
	else{
		
		fph = PhInit[SectIndex][station][Id - 1] + ph_tmp;
	}
	
	
	ph_hit = phLow + phShift + (PhDisp[SectIndex][LUTi]>>1);
	
	////////////////////////
	////Theta Conversion////
	////////////////////////
		
	int index = -999;
	int th_corr = -999;	
	
	if(station == 1){
	
		int idl = Id;
		if(Id < 4)//
			idl += 9;
		
		th_tmp = St1ThLUT[sub-1][SectIndex][idl -1][wire];
		//if(verbose) std::cout<<"\n\nth_tmpr = "<<th_tmp<<"\n\n";
	}
	else{
		th_tmp = ThLUT[station-2][SectIndex][Id-1][wire];
		//if(verbose) std::cout<<"\n\nth_tmpr = "<<th_tmp<<"\n\n";
	}
	
	
	th = th_tmp + ThInit[SectIndex][LUTi];
	//if(verbose) std::cout<<"ThInit = "<<ThInit[SectIndex][LUTi]<<"\n";
	
	if(station == 1 && (ring == 1 || ring == 4) /*&& endcap == 1*/){
	
		index = (wire>>4)*32 + (eightstrip>>4);
		
		if(Id > 3){
			th_corr = THCORR[sub-1][SectIndex][Id-10][index];
			//if(verbose) std::cout<<"\n\nth_corr = "<<th_corr<<"\n\n";
		}
		else{
			th_corr = THCORR[sub-1][SectIndex][Id-1][index];
			//if(verbose) std::cout<<"\n\nth_corr = "<<th_corr<<"\n\n";
		}
		
		
		if(ph_reverse) th_corr = -th_corr;
		
		
		th_tmp += th_corr;                  //add correction to th_tmp
		th_tmp &= 0x3f;                     //keep only lowest 6 bits
		
		
		if(th_tmp < th_coverage){
		
			if(ring == 1){LUTi += 9;}  //change because new Verilog3 sp_tf treats ME11b with LUT's of ME11a
			
			th = th_tmp + ThInit[SectIndex][LUTi];
		}
		else{th = -999;}
 
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
	
	
	if(Id > 9){
		Id -= 9;strip += 128;
	}
	
	
	int zhit = -99, pz = -99;
	std::vector<int> zonecontribution; //Each hit could go in more than one zone so we make a vector which stores all the zones for which this hit will contribute

	
	//determination of zone contribution
	if((phzvl & 1) && (Id < 4)){pz=0;zonecontribution.push_back(0);}
	if((phzvl & 2) && (Id < 4)){pz=1;zonecontribution.push_back(1);}
	if((phzvl & 1) && (Id > 3) && (station > 2)){pz=0;zonecontribution.push_back(1);}
	if((phzvl & 1) && (Id > 3) && (Id < 7) && (station == 1)){pz=0;zonecontribution.push_back(2);}
	if((phzvl & 1) && (Id > 3) && (station == 2)){pz=0;zonecontribution.push_back(2);}
	if((phzvl & 2) && (Id > 3) && (station > 2)){pz=1;zonecontribution.push_back(2);}
	if((phzvl & 1) && (Id > 4) && (station < 2)){pz=0;zonecontribution.push_back(3);}
	if(phzvl & 4){pz=2;zonecontribution.push_back(3);}
	if((phzvl & 2) && (Id > 3) && (station < 3)){pz=1;zonecontribution.push_back(3);}
	
	
	//applying ph_offsets
	if(sub == 1)
		zhit = ph_hit + ph_offsetss[station-1][Id-1][pz];
	else
		zhit = ph_hit + ph_offsetss[station][Id-1][pz];
		
		
		
		
	///////////////////////////////////////////////////////
	//////// set vector of ConvertedHits to move //////////
	/////////   Converted TP's around code   //////////////
	///////////////////////////////////////////////////////
	
	//if(verbose) std::cout<<"Phi = "<<fph<<" and Theta = "<<th<<std::endl;
	
	ConvertedHit Hit;

	Hit.SetValues(fph,th,ph_hit,phzvl,station,sub,Id,quality,pattern,wire,strip,BX);
	Hit.SetTP(C3);
	Hit.SetZhit(zhit);
	Hit.SetZoneContribution(zonecontribution);

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

