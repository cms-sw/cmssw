//////Takes in output of Matching.h which are the matched converted hit values 
//////to the patterns
//////
//////
//////
//////

#ifndef ADD_DELTAS
#define ADD_DELTAS


#include "L1Trigger/L1TMuonEndCap/interface/EmulatorClasses.h"

DeltaOutput Deltas(MatchingOutput Mout, int zone, int winner){
  
  //bool verbose = false;
	
  PhOutput phmatch = Mout.PhiMatch();
  ThOutput thmatch = Mout.ThetaMatch();
  // ThOutput2 t2 = Mout.TMatch2(); // Unused ... also we later declare an "unsigned int t2" - confusing! - AWB 29.08.16
  
  /////////////////////////////////////
  ///Set Null dphi and dtheta arrays///
  /////////////////////////////////////
  int dphi[6] = {-999,-999,-999,-999,-999,-999};
  int dtmp2[6] = {-999,-999,-999,-999,-999,-999};
  int dtmp2_ths[6][2] = {{-999,-999},{-999,-999},{-999,-999},{-999,-999},{-999,-999},{-999,-999}};
  
  
  for(int s1=0;s1<3;s1++){
    
    for(int s2=s1+1;s2<4;s2++){
      
      ///////////////////////// dphi index order runs like (dphi12,dphi13,dphi14,dphi23,dphi24,dphi34) hence the
      ///  calc delta phis  /// indexing procedure dphi[s2-1] for the first 3 and dphi[s1+s2] for the rest 
      /////////////////////////
      
      if((s1 == 0) && (phmatch.x[zone][winner][s1].Phi() != -999) && (phmatch.x[zone][winner][s2].Phi() != -999)){ //if using station one and both hits are valid
	
	dphi[s2-1] = phmatch.x[zone][winner][s1].Phi() - phmatch.x[zone][winner][s2].Phi();
      }
      else if((s1 != 0) && (phmatch.x[zone][winner][s1].Phi() != -999) && (phmatch.x[zone][winner][s2].Phi() != -999)){//if not using station one and both hits are valid
					
	dphi[s1+s2] = phmatch.x[zone][winner][s1].Phi() - phmatch.x[zone][winner][s2].Phi();	
      }
			
      
      ///////////////////////// There is a further index on dTh because there are 4 dth combinations 
      /// calc delta theta  /// possible if there are two theta segments for both stations. 
      ///////////////////////// EXPLAIN ABOUT [I+J] AND [I+J+1] 
      
      for(unsigned int t1=0;t1<phmatch.x[zone][winner][s1].AllThetas().size();t1++){
	for(unsigned int t2=0;t2<phmatch.x[zone][winner][s2].AllThetas().size();t2++){
	  
	  int dth_tmp = phmatch.x[zone][winner][s2].AllThetas()[t2] - phmatch.x[zone][winner][s1].AllThetas()[t1];
	  
	  if(s1 == 0){
	    
	    if(dtmp2[s2-1] == -999){
	      dtmp2[s2-1] = dth_tmp;
	      dtmp2_ths[s2-1][0] = phmatch.x[zone][winner][s1].AllThetas()[t1];
	      dtmp2_ths[s2-1][1] = phmatch.x[zone][winner][s2].AllThetas()[t2];
	    }
	    else if(abs(dth_tmp) < abs(dtmp2[s2-1])){
	      dtmp2[s2-1] = dth_tmp;
	      dtmp2_ths[s2-1][0] = phmatch.x[zone][winner][s1].AllThetas()[t1];
	      dtmp2_ths[s2-1][1] = phmatch.x[zone][winner][s2].AllThetas()[t2];
	    }
	    
	  }
	  else{
	    
	    if(dtmp2[s2+s1] == -999){
	      dtmp2[s2+s1] = dth_tmp;
	      dtmp2_ths[s1+s2][0] = phmatch.x[zone][winner][s1].AllThetas()[t1];
	      dtmp2_ths[s2+s1][1] = phmatch.x[zone][winner][s2].AllThetas()[t2];
	    }
	    else if(abs(dth_tmp) < abs(dtmp2[s2+s1])){
	      dtmp2[s2+s1] = dth_tmp;
	      dtmp2_ths[s1+s2][0] = phmatch.x[zone][winner][s1].AllThetas()[t1];
	      dtmp2_ths[s2+s1][1] = phmatch.x[zone][winner][s2].AllThetas()[t2];
	    }
	  }
	  
	}
      }
      
    }
  } 
	
  int vmask[3] = {0,0,0};
  const unsigned int mask[6] = {0x3,0x5,0x9,0x6,0xa,0xc};
  //////////////////////////////////////////////////////////////////////////////////////////
  /// the mask[6] is a way to indicate which stations are present and valid
  ///
  /// it is clearer to see in binary below
  ///
  /// stations 1 and 2 present and valid --> 0x3 --> 0011
  /// stations 1 and 3 present and valid --> 0x5 --> 0101
  /// stations 1 and 4 present and valid --> 0x9 --> 1001
  /// stations 2 and 3 present and valid --> 0x6 --> 0110
  /// stations 2 and 4 present and valid --> 0xa --> 1010
  /// stations 3 and 4 present and valid --> 0xc --> 1100
  ////////////////////////////////////////////////////////////////////////////////////////////
  const unsigned int mindex[6] = {0,0,0,1,1,2};//vmasindex-->says that the first three entries of mask belong to vmask[0] etc...
  
  for(int p=0;p<6;p++){
    
    if((abs(dtmp2[p]) <= 4) && (dtmp2[p] != -999)){//if dtheta is small enought and valid
      vmask[mindex[p]] |= mask[p];	
    } 
  }
  
  unsigned int vstat = vmask[0];
  
  if( !vstat || (vstat & vmask[1])){vstat |= vmask[1];}
  if( !vstat || (vstat & vmask[2])){vstat |= vmask[2];}
  
  //if(vstat ){std::cout<<"vstat = "<<vstat<<std::endl;}//
  /*
  //const unsigned int vstatindex[11] = {0xc,0xa,0x6,0xe,0x9,0x5,0xd,0x3,0xb,0x7,0xf};
  /////////////////////////////////////////////////////////////////////////////////////////////
  /// vstatindex[11] is a list of possible combinations of valid and present stations
  /// in order of increasing quality(i.e. if all stations are present and valid it's 
  /// better to take delta 12 and 23 as opposed to delta 14 and 34 . 
  /// Binary representation is below
  /// stations 3 and 4 present   --> 0xc --> 1100
  /// stations 2 and 4 present   --> 0xa --> 1010
  /// stations 2 and 3 present   --> 0x6 --> 0110
  /// stations 2,3 and 4 present --> 0xe --> 1110
  /// stations 1 and 4 present   --> 0x9 --> 1001
  /// stations 1 and 3 present   --> 0x5 --> 0101
  /// stations 1,3 and 4 present --> 0xd --> 1101
  /// stations 1 and 2 present   --> 0x3 --> 0011
  /// stations 1,2 and 4 present --> 0xb --> 1011
  /// stations 1,2 and 3 present --> 0x7 --> 0111
  /// all stations present       --> 0xf --> 1111
  /////////////////////////////////////////////////////////////////////////////////////////////
  //const unsigned int viindex[2][11] = {{5,4,3,3,2,1,1,0,0,0,0},{5,4,3,5,2,1,5,0,4,3,3}};///index on which entry of vstatindex[11] to choose for both dphi and dtheta
  */

  std::vector<int> d (6,-999);
  std::vector<std::vector<int>> deltas (2,d);//deltas[0]->dPhi & deltas[1]->dTheta
  
  for(int i=0;i<6;i++){
    
    deltas[0][i] = dphi[i];
    deltas[1][i] = dtmp2[i];
  }
  
  ///////////Set Precise Phi&Theta//////////
  int phi = 0, theta = 0;//, id = 0;
  if(vstat & 2){//ME2 Present
    
    //phi is simple, we have only one per station to choose from
    phi = phmatch.x[zone][winner][1].Phi();
    
    //for theta, select delta to best station, use dtmpi as index
    if(dtmp2[0] != -999){
      theta = dtmp2_ths[0][1];//t2[zone][winner][1][id];
    }
    else if(dtmp2[3] != -999){
      theta = dtmp2_ths[3][0];//t2[zone][winner][1][id];
    }
    else if(dtmp2[4] != -999){
      theta = dtmp2_ths[4][0];//t2[zone][winner][1][id];
    }
    
  }
  else if(vstat & 4){//ME3 Present, but not ME2
	
    phi = phmatch.x[zone][winner][2].Phi();
    if(dtmp2[1] != -999){
      theta = dtmp2_ths[1][1];//t2[zone][winner][2][id];
    }
    else if(dtmp2[5] != -999){
      theta = dtmp2_ths[5][0];//t2[zone][winner][2][id];
    }
  }
  else if(vstat & 8){//ME4 Present but not ME2 or ME3
    phi = phmatch.x[zone][winner][3].Phi();
    if(dtmp2[2] != -999){
      theta = dtmp2_ths[2][1];//t2[zone][winner][3][id];
    }
  }
	
  //////////////////////////////////////////
  
  int rank = (Mout.Winners()[zone][winner].Rank()<<1);
  
  //if(rank) std::cout<<"rank = "<<rank<<"\n";
  ///here we separate stations 3 and 4////
  if(vstat & 8){rank |= 1;}else{rank &= 0x7e;}//if station 4 add to rank, else keep everythign and zero the bit which indicates station 4 is present
  if(vstat & 4){rank |= 2;}else{rank &= 0x7d;}//if station 3 add to rank, else keep everythign and zero the bit which indicates station 3 is present
  if(vstat & 2){rank |= 8;}else{rank &= 0x77;}//if station 2 add to rank, else keep everythign and zero the bit which indicates station 2 is present
  if(vstat & 1){rank |= 32;}else{rank &= 0x5f;}//if station 1 add to rank, else keep everythign and zero the bit which indicates station 1 is present
  if(vstat == 0 || vstat == 1 || vstat == 2 || vstat == 4 || vstat == 8){rank = 0;}//removes single, and ME3--ME4 hit combinations
  DeltaOutput output;
  Mout.SetPhOut(phmatch);
  Winner win = Mout.Winners()[zone][winner];
  win.SetRank(rank);
  
  output.SetValues(Mout,deltas,phi,theta,win);
  
  return output;
  
}

std::vector<std::vector<DeltaOutput>> CalcDeltas(MatchingOutput Mout){

  DeltaOutput output;output.SetNull();
  
  std::vector<DeltaOutput> o (3,output);
  std::vector<std::vector<DeltaOutput>> out (4,o);
  
  for(int zone=0;zone<4;zone++){
    for(int winner=0;winner<3;winner++){
      out[zone][winner] = Deltas(Mout, zone, winner);
    }
  }	
	
  return out;
}

DeltaOutArr3 CalcDeltas_Hold(const std::vector<MatchingOutput> & Mout){
  
  DeltaOutput output;
  output.SetNull();
  DeltaOutArr3 Output = { {{{output}}} };
  
  for(int bx=0;bx<3;bx++){
    for(int zone=0;zone<4;zone++){
      for(int winner=0;winner<3;winner++){
	Output.x[bx][zone][winner] = Deltas(Mout[bx], zone, winner);
      }
    }
  }
  
  return Output;
  
}

#endif
