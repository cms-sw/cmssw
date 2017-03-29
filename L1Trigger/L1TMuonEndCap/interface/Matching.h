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
	int phdiff[4] = {15,15,8,8};
	
	/////////////////////////////////////////
	//// Set Null Ph and Th outputs /////////
	/////////////////////////////////////////
	ConvertedHit tt; tt.SetNull();
	std::vector<ConvertedHit> p (4,tt);std::vector<std::vector<ConvertedHit>> pp (3,p);
	PhOutput ph_output (4,pp);
					
	
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
				
				for(std::vector<ConvertedHit>::iterator i = Thits.begin();i != Thits.end();i++){//Possible associated hits
				
					//int id = i->Id();
					
					if(verbose) std::cout<<"strip = "<<i->Strip()<<", keywire = "<<i->Wire()<<" and zhit-"<<i->Zhit()<<std:: endl;

					// Unused variable
					/* bool inzone = 0;///Is the converted hit in the zone we're looking at now? */
					/* for(std::vector<int>::iterator znc = i->ZoneContribution().begin();znc != i->ZoneContribution().end();znc++){ */
					/* 	if((*znc) == z) */
					/* 		inzone = 1;//yes */
					/* } */
					
					////////////////////////////////////////////////////////////////////////////////////////////
					/////////////////// Setting the matched hits based on phi //////////////////////////////////
					////////////////////////////////////////////////////////////////////////////////////////////
					int setstation = i->Station() - 1;
					//bool one = ((z == 3) && (i->Station() > 1));                //Zone 3 is handled differently so we
					//bool two = ((z == 3) && (i->Station() == 1) && (id > 3));   //have this conditions here
					bool setphi = 0;
					//if(one || two)
					//	setstation++;
					
					if(verbose)
						std::cout<<"setstation = "<<setstation<<std::endl;
					
					if((fabs((Winners[z][w].Strip()) - i->Zhit()) <= phdiff[setstation]) ){//is close to winner keystrip and in same zone?
					
						if(ph_output[z][w][setstation].Phi() == -999){//has this already been set? no
						
							if(verbose) std::cout<<"hasn't been set"<<std::endl;
							
							ph_output[z][w][setstation] = (*i);
							
							if(verbose) std::cout<<"set with strip-"<<i->Strip()<<", and wire-"<<i->Wire()<<std::endl;
							setphi = true;
						}
						else{//if yes, find absolute difference between zhit of each hit and keystrip
						
							if(verbose) std::cout<<"has already been set"<<std::endl;
						
							int d1 = fabs(ph_output[z][w][setstation].Zhit() - Winners[z][w].Strip());
							int d2 = fabs(i->Zhit() - Winners[z][w].Strip());
							
							if(d2 < d1){//if new hit is closer then replace phi
							
								if(verbose) std::cout<<"this is closer strip-"<<i->Strip()<<", and wire-"<<i->Wire()<<std::endl;
								
								ph_output[z][w][setstation] = (*i);
								
								setphi = true;
							
							}
							
						}
						
						
						/////////////////////////////////////////////////////////////////////////////////////
						/////////////  Setting matched theta values; Take both of two from same chamber /////
						/////////////////////////////////////////////////////////////////////////////////////
					
						if(setphi)//only set if phi was also set
							th_output[z][w][setstation][0] = (*i);
							
						if((th_output[z][w][setstation][0].Theta() != -999) && (th_output[z][w][setstation][0].Id() == i->Id()))//if same chamber take as well
							th_output[z][w][setstation][1] = (*i);
						
					
					}
				}
				
				
			}

		}
	}

	MatchingOutput output;
	output.SetValues(th_output,ph_output,Thits,Winners,segment);
	
	return output;
}

