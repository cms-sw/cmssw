/*
Takes in the output of Deltas module and sorts to find 
three best tracks per given sector.


Author: Matthew R. Carver
Date: 7/29/13
*/

#ifndef ADD_BESTTRACK
#define ADD_BESTTRACK


#include "L1Trigger/L1TMuonEndCap/interface/EmulatorClasses.h"


std::vector<BTrack> BestTracks(DeltaOutArr2 Dout){

  bool verbose = false;
  
  int larger[12][12] = {{0},{0}}, kill[12] = {0};
  int exists[12] = {0};
  int winner[3][12] = {{0,0,0,0,0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0,0,0,0,0}};
  int phi[4][3][4], id[4][3][4];
  
  for(int r=0;r<4;r++){
    for(int t=0;t<3;t++){
      for(int d=0;d<4;d++){
	
	phi[r][t][d] = Dout.x[r][t].GetMatchOut().PhiMatch().x[r][t][d].Phi();
	//if(phi[r][t][d] != -999) std::cout<<"phi = "<<phi[r][t][d]<<"\n";
	id[r][t][d] = Dout.x[r][t].GetMatchOut().PhiMatch().x[r][t][d].Id();
      }
    }
  }
  
  BTrack tmp;
  std::vector<BTrack> output (3,tmp);
	
  for(int zone=0;zone<4;zone++){
    for(int winner=0;winner<3;winner++){
      for(int station=0;station<4;station++){
	
	int cham = Dout.x[zone][winner].GetMatchOut().PhiMatch().x[zone][winner][station].Id();
	//int relst = 0;
	int relch = 0;
	
	if(station == 0){
	  
	  //relst = (cham < 3) ? 0 : 1;
	  relch = cham%3;
	  if(zone == 2)
	    relch += 3;
	  if(zone == 3)
	    relch += 6;
	}
	else if(station == 1){
	  
	  //relst = station + 1;
	  relch = cham;
	  if(zone > 1)
	    relch += 3;
	}
	else{
	  
	  //relst = station + 1;
	  relch = cham;
	  if(zone > 0)
	    relch += 3;
	}
	
	//phi[zone][winner][relst] = phi[zone][winner][station];
	//id[zone][winner][relst] = id[zone][winner][station];
	
	//if(phi[zone][winner][relst] != -999 && verbose)
	//	std::cout<<"st:"<<relst<<"::real phi = "<<phi[zone][winner][relst]<<" and id = "<<id[zone][winner][relst]<<std::endl;
      }
    }
  }
  
  ///Here we compare the best three ranks from each zone with each other
  for(int i=0;i<12;i++){
    
    larger[i][i] = 1;//result of comparison with itself
    int ranki = Dout.x[i%4][i/4].GetWinner().Rank();//the zone loops faster such that each is given equal priority
    
    for(int j=0;j<12;j++){
      
      int rankj = Dout.x[j%4][j/4].GetWinner().Rank();
      bool greater = (ranki > rankj);
      bool equal = (ranki == rankj);
      
      if(((i<j) && (greater || equal)) || ((i>j) && greater))
	larger[i][j] = 1;
    }
    
    exists[i] = (ranki != 0);
  }
  
  // ghost cancelltion. only in current BX so far(as in firmware as well)
  int vmask[4] = {32,8,2,1};
  for(int k=0;k<12;k++){
    for(int l=0;l<12;l++){
      int sh_seg = 0;
      for(int s=0;s<4;s++){
	
	//if(id[k%4][k/4][s] && (k != l) && ((phi[k%4][k/4][s] != -999) && (phi[l%4][l/4][s] != -999)) && verbose)
	//	std::cout<<"id1 = "<<id[k%4][k/4][s]<<", id2 = "<<id[l%4][l/4][s]<<"\nphi1 = "<<phi[k%4][k/4][s]<<", phi1 = "<<phi[l%4][l/4][s]<<".\n";
	
	if((id[k%4][k/4][s] == id[l%4][l/4][s])
	   && ((phi[k%4][k/4][s] != -999) || (phi[l%4][l/4][s] != -999))
	   && (phi[k%4][k/4][s] == phi[l%4][l/4][s])
	   && (k != l)
	   && (Dout.x[k%4][k/4].GetWinner().Rank() & vmask[s]) //station from track one is valid after deltas
	   && (Dout.x[l%4][l/4].GetWinner().Rank() & vmask[s]) //station from track two is valid after deltas
	   ){
	  
	  sh_seg++;
	}
      }
      
      if(sh_seg){
	//kill candidate that has lower rank
	if(larger[k][l]){kill[l] = 1;}
	else{kill[k] = 1;}
      }
    }
  }
  
  //remove ghosts according to kill array
  for(int q=0;q<12;q++){
    if(kill[q]){exists[q] = 0;}
  }
  
  for(int p=0;p<12;p++){
    if(exists[p]){
      for(int x=0;x<12;x++){
	if(!exists[x]){larger[p][x] = 1;}
      }
    }
    else{
      for(int x=0;x<12;x++){
	larger[p][x] = 0;
      }
    }
    
    int sum = 0;
    for(int j=0;j<12;j++){
      if(!larger[p][j]){sum++;}
    }
    
    if(sum < 3){winner[sum][p] = 1;}
  }
  
  for(int n=0;n<3;n++){
    for(int i=0;i<12;i++){
      if(winner[n][i]){
	
	BTrack bests;
	int mode = 0;
	if(Dout.x[i%4][i/4].GetWinner().Rank() & 32)
	  mode |= 8;
	if(Dout.x[i%4][i/4].GetWinner().Rank() & 8)
	  mode |= 4;
	if(Dout.x[i%4][i/4].GetWinner().Rank() & 2)
	  mode |= 2;
	if(Dout.x[i%4][i/4].GetWinner().Rank() & 1)
	  mode |= 1;
	
	if(verbose) std::cout<<"Best Rank "<<n<<" = "<<Dout.x[i%4][i/4].GetWinner().Rank()<<" and mode = "<<mode<<"\n\n";
	if(verbose) std::cout<<"Phi = "<<Dout.x[i%4][i/4].Phi()<<" and Theta = "<<Dout.x[i%4][i/4].Theta()<<"\n\n";
	if(verbose) std::cout<<"Ph Deltas: "<<Dout.x[i%4][i/4].Deltas()[0][0]<<" "<<Dout.x[i%4][i/4].Deltas()[0][1]<<" "<<Dout.x[i%4][i/4].Deltas()[0][2]<<" "<<Dout.x[i%4][i/4].Deltas()[0][3]
			     <<" "<<Dout.x[i%4][i/4].Deltas()[0][4]<<" "<<Dout.x[i%4][i/4].Deltas()[0][5]<<"   \nTh Deltas: "<<Dout.x[i%4][i/4].Deltas()[1][0]
			     <<" "<<Dout.x[i%4][i/4].Deltas()[1][1]<<" "<<Dout.x[i%4][i/4].Deltas()[1][2]<<" "<<Dout.x[i%4][i/4].Deltas()[1][3]
			     <<" "<<Dout.x[i%4][i/4].Deltas()[1][4]<<" "<<Dout.x[i%4][i/4].Deltas()[1][5]<<"\n\n";
	
	bests.winner = Dout.x[i%4][i/4].GetWinner();
	bests.phi = Dout.x[i%4][i/4].Phi();
	bests.theta = Dout.x[i%4][i/4].Theta();
	bests.deltas = Dout.x[i%4][i/4].Deltas();
	bests.clctpattern = Dout.x[i%4][i/4].GetMatchOut().PhiMatch().x[i%4][i/4][0].Pattern();
	bests.AHits.clear();
	for (int iPh = 0; iPh < 4; iPh++) {
	  if ( Dout.x[i%4][i/4].GetMatchOut().PhiMatch().x[i%4][i/4][iPh].Theta() != -999 &&
	       Dout.x[i%4][i/4].GetMatchOut().PhiMatch().x[i%4][i/4][iPh].Phi() > 0 )
	    bests.AHits.push_back( Dout.x[i%4][i/4].GetMatchOut().PhiMatch().x[i%4][i/4][iPh] );
	}

	if (bests.phi != 0 && bests.theta == 0)
	  std::cout << "In BestTracks.h, phi = " << bests.phi << " and theta = " << bests.theta << std::endl;
	
	output[n] = bests;
	
      }
    }
  }

  return output;
  
}

std::vector<std::vector<BTrack>> BestTracks_Hold(DeltaOutArr3 Dout){
	
  BTrack tmp;
  std::vector<BTrack> output (3,tmp);
  std::vector<std::vector<BTrack>> full_output (3,output);
  
  for(int bx=0;bx<3;bx++){
    DeltaOutArr2 Dout2;
    for(int zone=0;zone<4;zone++){
      for(int winner=0;winner<3;winner++){
	Dout2.x[zone][winner] = Dout.x[bx][zone][winner];
      }
    }
    full_output[bx] = BestTracks(Dout2);
  }

  return full_output;
  
}


#endif
