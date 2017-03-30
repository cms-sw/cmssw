//////Takes in Pattern Output and delivers three best tracks per zone
//////
//////
//////
//////
//////

#ifndef ADD_SortSector
#define ADD_SortSector


#include "L1Trigger/L1TMuonEndCap/interface/PhiMemoryImage.h"
#include "L1Trigger/L1TMuonEndCap/interface/EmulatorClasses.h"

SortingOutput  SortSect(PatternOutput Pout){
	
  ////variable declaration ///////////////////////////
  QualityOutput Detected = Pout.detected;
  Winner tmp;tmp.SetValues(0,0);
  std::vector<Winner> tmmp (3,tmp);
  std::vector<std::vector<Winner>> Winners (4,tmmp);
  ////////////////////////////////////////////////////
  
  for(int i=0;i<3;i++){//loop to get three best
    
    Winner temp[4];
    for(int zone=0;zone<4;zone++){
      
      temp[zone].SetValues(0,0);
      /*//for(int strip=0;strip<192;strip++){//was 128*/
      for(int strip=191;strip>-1;strip--){//was 128
	
	if(Detected.rank[zone][strip] > temp[zone].Rank()){temp[zone].SetValues(Detected.rank[zone][strip], strip);}
      }
      
      if(temp[zone].Rank()){
	
	//removes rank as not to count twice/////////////////
	Detected.rank[zone][temp[zone].Strip()] = 0;
	Detected.layer[zone][temp[zone].Strip()] = 0;
	Detected.straightness[zone][temp[zone].Strip()] = 0;
	/////////////////////////////////////////////////////
	Winners[zone][i] = temp[zone];
	Winners[zone][i].SetBXGroup(Detected.bxgroup[zone][temp[zone].Strip()]);
      }
    }
  }
  
  
  /////////////////////////////////
  // Printing for Comparison///////
  /////////////////////////////////
  /*
    for(int l=0;l<4;l++){//zone loop
			
    for(int m=0;m<3;m++){
    
    if(Winners[l][m].Rank())
    std::cout<<"Winner["<<l<<"]["<<m<<"]!-----Q:"<<Winners[l][m].Rank()<<"  S:"<<Winners[l][m].Strip()<<"\n";
    
    }
    }
  */	
  /////////////////////////////////
  /////////////////////////////////
  /////////////////////////////////
  
  SortingOutput output;
  
  output.SetValues(Winners,Pout.hits);
  
  return output;
  
}

SortingOutput  SortSector(PatternOutput Pout, int bxgroup){
  
  ////variable declaration ///////////////////////////
  QualityOutput Detected = Pout.detected;
  Winner tmp;tmp.SetValues(0,0);
  std::vector<Winner> tmmp (3,tmp);
  std::vector<std::vector<Winner>> Winners (4,tmmp);
  ////////////////////////////////////////////////////
  
  for(int i=0;i<3;i++){//loop to get three best
    
    Winner temp[4];
    for(int zone=0;zone<4;zone++){
			
      temp[zone].SetValues(0,0);
      /*//for(int strip=0;strip<192;strip++){//was 128*/
      for(int strip=191;strip>-1;strip--){//was 128
	
	if(Detected.rank[zone][strip] > temp[zone].Rank()){temp[zone].SetValues(Detected.rank[zone][strip], strip);}
	
      }
      
      if(temp[zone].Rank()){
				
	//removes rank as not to count twice/////////////////
	Detected.rank[zone][temp[zone].Strip()] = 0;
	Detected.layer[zone][temp[zone].Strip()] = 0;
	Detected.straightness[zone][temp[zone].Strip()] = 0;
	/////////////////////////////////////////////////////
	Winners[zone][i] = temp[zone];
	Winners[zone][i].SetBXGroup(bxgroup);
      }
    }
  }
  
  
  /////////////////////////////////
  // Printing for Comparison///////
  /////////////////////////////////
  /*
    for(int l=0;l<4;l++){//zone loop
    
    for(int m=0;m<3;m++){
    
    if(Winners[l][m].Rank())
    std::cout<<"Winner["<<l<<"]["<<m<<"]!-----Q:"<<Winners[l][m].Rank()<<"  S:"<<Winners[l][m].Strip()<<"\n";
    
    }
    }
  */	
  /////////////////////////////////
  /////////////////////////////////
  /////////////////////////////////
  
  SortingOutput output;
  
  output.SetValues(Winners,Pout.hits);
  
  return output;
  
}


std::vector<SortingOutput> SortSect_Hold(std::vector<PatternOutput> Pout){
  
  SortingOutput tmp;
  std::vector<SortingOutput> output (3,tmp);
  
  if(Pout.size() != 3)
    std::cout<<"Incorrect number of BX windows! Please check to see why.\n";
  
  for(int i=0;i<3;i++){
    //std::cout<<"start sorting on BX group "<<i<<"\n";
    output[i] = SortSector(Pout[i], i+1);
  }
  
  return output;
}


#endif
