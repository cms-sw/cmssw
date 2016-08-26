//////Takes in output from SortSect and returns a matching output which 
//////are the winning patterns and associated converted hits
//////
//////
//////
//////

#include "L1Trigger/L1TMuonEndCap/interface/EmulatorClasses.h"
#include "L1Trigger/L1TMuonEndCap/interface/PhiMemoryImage.h"


MatchingOutput PhiMatching(SortingOutput Sout){

  bool verbose = false;
  
  std::vector<ConvertedHit> Thits = Sout.Hits();
  std::vector<std::vector<Winner>> Winners = Sout.Winners();
  std::vector<int> segment (4,0);
  int phdiff[4] = {15,7,7,7};
  
  /////////////////////////////////////////
  //// Set Null Ph and Th outputs /////////
  /////////////////////////////////////////
  ConvertedHit tt; tt.SetNull();
  std::vector<ConvertedHit> p (4,tt);std::vector<std::vector<ConvertedHit>> pp (3,p);
  PhOutput ph_output (4,pp);
  
  int t2 = -999;
  std::vector<int> q2 (2,t2);
  std::vector<std::vector<int>> qq2 (4,q2);std::vector<std::vector<std::vector<int>>> qqq2 (3,qq2);
  ThOutput2 th_output2 (4,qqq2);
  
  
  std::vector<ConvertedHit> q (2,tt);
  std::vector<std::vector<ConvertedHit>> qq (4,q);std::vector<std::vector<std::vector<ConvertedHit>>> qqq (3,qq);
  ThOutput th_output (4,qqq);
  /////////////////////////////////////////
  /////////////////////////////////////////
  /////////////////////////////////////////
	
  for(int z=0;z<4;z++){//zone loop
    
    for(int w=0;w<3;w++){//winner loop
			
      if(Winners[z][w].Rank()){//is there a winner present?	
	
	if(verbose) std::cout<<"\n\nWinner position-"<<Winners[z][w].Strip()<<". Zone = "<<z<<std::endl;
	if(verbose) std::cout<<"Number of possible hits to match = "<<Thits.size()<<"\n";			
	
	//for(std::vector<ConvertedHit>::iterator i = Thits.begin();i != Thits.end();i++){//Possible associated hits
	for(int i = Thits.size() - 1;i > -1;i--){//Possible associated hits
	  //for(unsigned int i=0;i<Thits.size();i++){//Possible associated hits
	  
	  //int id = Thits[i].Id();
	  
	  if(verbose) std::cout<<"station = "<<Thits[i].Station()<<", strip = "<<Thits[i].Strip()<<", keywire = "<<Thits[i].Wire()<<" and zhit-"<<Thits[i].Zhit()<<", bx = "<<Thits[i].BX()<<", phi>>5 = "<<(Thits[i].Phi()>>5)<<"\n";
	  
	  // Unused variable
	  /* bool inzone = 0;///Is the converted hit in the zone we're looking at now? 
	     for(std::vector<int>::iterator znc = Thits[i].ZoneContribution().begin();znc != Thits[i].ZoneContribution().end();znc++){ 
	     if((*znc) == z) 
	     inzone = 1;//yes 
	     } */
	  
	  bool inBXgroup = false;
	  
	  switch(Winners[z][w].BXGroup()){
	    
	  case 1: if(Thits[i].BX() > 3 && Thits[i].BX() < 7) inBXgroup = true;break;
	  case 2: if(Thits[i].BX() > 4 && Thits[i].BX() < 8) inBXgroup = true;break;
	  case 3: if(Thits[i].BX() > 5 && Thits[i].BX() < 9) inBXgroup = true;break;
	  default: inBXgroup = false;
	    
	  }
	  
	  ////////////////////////////////////////////////////////////////////////////////////////////
	  /////////////////// Setting the matched hits based on phi //////////////////////////////////
	  ////////////////////////////////////////////////////////////////////////////////////////////
	  int setstation = Thits[i].Station() - 1;
	  bool setphi = 0;
	  
	  if(verbose)
	    std::cout<<"setstation = "<<setstation<<std::endl;
	  
	  //if(verbose){
	  
	  //	std::cout<<"Winners[z][w].Strip(): "<<Winners[z][w].Strip()<<" + 1 - Thits[i].Zhit():"<<Thits[i].Zhit()<<" = "<<(Winners[z][w].Strip() + 1) - Thits[i].Zhit()<<". Thits[i].Phi()>>5 = "<<(Thits[i].Phi() >> 5)<<"\n";
	  //}
	  
	  if((fabs(Winners[z][w].Strip() - (Thits[i].Phi()>>5)) <= phdiff[setstation]) && inBXgroup /*&& inzone*/){//is close to winner keystrip and in same zone?
	    
	    if(ph_output[z][w][setstation].Phi() == -999){//has this already been set? no
	      
	      if(verbose) std::cout<<"hasn't been set"<<std::endl;
	      
	      ph_output[z][w][setstation] = (Thits[i]);
	      
	      if(verbose) std::cout<<"set with strip-"<<Thits[i].Strip()<<", and wire-"<<Thits[i].Wire()<<std::endl;
	      setphi = true;
	    }
	    else{//if yes, find absolute difference between zhit of each hit and keystrip
	      
	      if(verbose) std::cout<<"has already been set"<<std::endl;
	      
	      int d1 = fabs((ph_output[z][w][setstation].Phi()>>5) - Winners[z][w].Strip());
	      int d2 = fabs((Thits[i].Phi()>>5) - Winners[z][w].Strip());
	      
	      if(verbose) std::cout<<"d1 = "<<d1<<" and d2 = "<<d2<<"\n";
	      
	      if(d2 < d1){//if new hit is closer then replace phi
		
		if(verbose) std::cout<<"this is closer strip-"<<Thits[i].Strip()<<", and wire-"<<Thits[i].Wire()<<std::endl;
		
		ph_output[z][w][setstation] = (Thits[i]);
		
		setphi = true;
		
	      }
	      
	    }
	    
	    /////////////////////////////////////////////////////////////////////////////////////
	    /////////////  Setting matched theta values; Take both of two from same chamber /////
	    /////////////////////////////////////////////////////////////////////////////////////
	    
	    //if((th_output[z][w][setstation][0].Theta() != -999) && (th_output[z][w][setstation][0].Id() == Thits[i].Id())){//if same chamber take as well
	    //		th_output[z][w][setstation][1] = (Thits[i]);
	    //		if(verbose) std::cout<<"in here with set th = "<<th_output[z][w][setstation][0].Theta()<<" and new th = "<<Thits[i].Theta()<<"\n";
	    //	}
	    
	    if(setphi){//only set if phi was also set
	      
	      th_output2[z][w][setstation][0] = Thits[i].Theta();
	      
	      /*if(th_output[z][w][setstation][0].Theta() == -999){
		th_output[z][w][setstation][0] = (Thits[i]);
		}
		else if((th_output[z][w][setstation][0].Theta() != -999) && (th_output[z][w][setstation][0].Id() == Thits[i].Id())){//if same chamber take as well
		th_output[z][w][setstation][1] = (Thits[i]);
		if(verbose) std::cout<<"in here with set th = "<<th_output[z][w][setstation][0].Theta()<<" and new th = "<<Thits[i].Theta()<<"\n";
		}*/
	      
	      //if((th_output[z][w][setstation][0].Theta() != -999) && (th_output[z][w][setstation][0].Id() == Thits[i].Id())){//if same chamber take as well
	      //	th_output[z][w][setstation][1] = (Thits[i]);
	      //	if(verbose) std::cout<<"in here with set th = "<<th_output[z][w][setstation][0].Theta()<<" and new th = "<<Thits[i].Theta()<<"\n";
	      //}
	      //else{
	      //	th_output[z][w][setstation][0] = (Thits[i]);
	      
	      if(Thits[i].Theta2() != -999)
		th_output2[z][w][setstation][1] = Thits[i].Theta2();
	      //}
	      
	    }
	    
	  }
	}
	
      }
      
    }
  }
  
  MatchingOutput output;
  output.SetValues(th_output,ph_output,Thits,Winners,segment);
  output.setM2(th_output2);
  
  return output;
}

std::vector<MatchingOutput> PhiMatching_Hold(std::vector<SortingOutput> Sout){
  
  MatchingOutput tmp;
  std::vector<MatchingOutput> output (3,tmp);
  
  if(Sout.size() != 3)
    std::cout<<"Incorrect BX window size for sorting output. Please check to see why.\n";
  
  for(int i=0;i<3;i++){
    output[i] = PhiMatching(Sout[i]);
  }
  
  return output;
}

