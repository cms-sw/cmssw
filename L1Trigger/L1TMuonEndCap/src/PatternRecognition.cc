///// Takes in a std::vector<PhiMemoryImage> Merged which is the merged image of the BXs per zone and outputs a vector of Integers containing strip,rank,layer and straightness
/////
/////
/////
/////
/////

#include "L1Trigger/L1TMuonEndCap/interface/PatternRecognition.h"

//hardcoded pattern definitions
#define PATTERN_SIZE 9
PhiMemoryImage pattern1 (0x8000,0,0x8000,0,0x8000,0,0x8000,0);
PhiMemoryImage pattern2 (0x10000,0,0x8000,0,0xc000,0,0xc000,0);
PhiMemoryImage pattern3 (0x4000,0,0x8000,0,0x18000,0,0x18000,0);
PhiMemoryImage pattern4 (0x60000,0,0x8000,0,0xe000,0,0xe000,0);
PhiMemoryImage pattern5 (0x3000,0,0x8000,0,0x78000,0,0x78000,0);
PhiMemoryImage pattern6 (0x780000,0,0x8000,0,0xff00,0,0xff00,0);
PhiMemoryImage pattern7 (0xf00,0,0x8000,0,0x7f8000,0,0x7f8000,0);
PhiMemoryImage pattern8 (0x7f800000,0,0x8000,0,0xff00,0,0xff00,0);//
PhiMemoryImage pattern9 (0xff,0,0x8000,0,0x7f8000,0,0x7f8000,0);
PhiMemoryImage patterns[PATTERN_SIZE] = {pattern8, pattern9, pattern6, pattern7, pattern4, pattern5, pattern2, pattern3, pattern1};
 
 
 PatternOutput DetectPatterns(ZonesOutput Eout){
 
 	///variable declarations///
	//bool verbose = false;
 	std::vector<int> tmp (128, 0);
	std::vector<std::vector<int>> lya (4, tmp), stra (4, tmp), ranka_t (4, tmp), ranka (4, tmp);
  	std::vector<PhiMemoryImage> Merged = Eout.zone;
	////////////////////////////
	
  	for(int zone=0;zone<4;zone++){
	
		
  
  		for(int b=0;b<128;b++){//loop over stips of detector zones
  	
  			int ly[PATTERN_SIZE] = {0}, srt[PATTERN_SIZE] = {0}, qu[PATTERN_SIZE] = {0};
			PhiMemoryImage patt[PATTERN_SIZE];
			for (int i=0; i<PATTERN_SIZE; i++) patt[i] = patterns[i];
  			for(int y=0;y != PATTERN_SIZE;y++){//loop over patterns
	
				bool zona[8] = {false}; //Clear out station presence
		
				if((b-15) < 63){ 					//////Due to bug in BitShift function. 
					patt[y].BitShift(b-15);			 	//////Can try and fix later before uploading to CMSSW.
				}							//////
				else{							//////
					patt[y].BitShift(63);patt[y].BitShift(b-78);	//////
				}							//////
			
			
				for(int yy=0;yy != 8;yy++){//loop over 8 long integers of each pattern
				
					zona[yy] = patt[y][yy] & Merged[zone][yy];
				}
			
			
		
				if(zona[0] || zona[1]){ly[y] += 4;}//if station 1 is present
				if(zona[2] || zona[3]){ly[y] += 2;}//if station 2 is present
				if(zona[4] || zona[5] || zona[6] || zona[7]){ly[y] += 1;}//if station 3 or 4 is present
				srt[y] = y/2;//straightness code::pattern specific
				 
				if(  (ly[y] != 0) && (ly[y] != 1) && (ly[y] != 2) && (ly[y] != 4)  ){//Removes single and ME34 hit combinations//

						
						//creates quality code by interleving straightness and layer bit words///
						if((srt[y] & 1) != 0){qu[y] += 2;}
						if((srt[y] & 2) != 0){qu[y] += 8;}
						if((srt[y] & 4) != 0){qu[y] += 32;}
						if((ly[y] & 1) != 0){qu[y] += 1;}
						if((ly[y] & 2) != 0){qu[y] += 4;}
						if((ly[y] & 4) != 0){qu[y] += 16;}
						//////////////////////////////////////////////////////////////////////////
					
				}
				 
			}//pattern loop
		
			for(int a=0;a<PATTERN_SIZE;a++)
			{
				if(qu[a] > ranka_t[zone][b]){//take highest quality from all patterns at given key strip
					
					ranka_t[zone][b] = qu[a];
					lya[zone][b] = ly[a];
					stra[zone][b] = srt[a];
				}
			}//pattern loop to assign quality
	
  	
  	
  		
	
  		}//strip loop
	}//zone loop
	

 	///////////////////////////////
 	///// Ghost Cancellation //////
 	///////////////////////////////
 
 
 	for(int zone=0;zone<4;zone++){
 
 		for(int k=0;k<128;k++){
	
		  int qr = (k==0) ? qr=0 : ranka_t[zone][k-1]; 
		  int ql = (k==127) ? ql=0 :ranka_t[zone][k+1];
		  int qc = ranka_t[zone][k];
		
			if((qc <= ql) || (qc < qr)){qc = 0;}
		
			ranka[zone][k] = qc;
		}
 	}
 
 
 
 	
 	QualityOutput qout;
	qout.rank = ranka;
	qout.layer = lya;
	qout.straightness = stra;
	
	PatternOutput output;
	output.detected = qout;
	output.hits = Eout.convertedhits;
	
		
	return output;
 
 }
 
 
 std::vector<PatternOutput> Patterns(std::vector<ZonesOutput> Zones){
 
 	PatternOutput tmp;
	std::vector<PatternOutput> output (3,tmp);
	
	for(int i=0;i<3;i++)
		output[i] = DetectPatterns(Zones[i]);
		
		
 	return output;
 
 }


 void PrintQuality (QualityOutput out){
 
 
 	 ////////////////////////////////////
	 /// Print Quality for comparison ///
	 ////////////////////////////////////
	 for(int zz=0;zz<4;zz++){
 	   for(int z = 0;z<128;z++){
	
		//if((out.rank)[zz][z]){std::cout<<"Zone: "<<zz<<"::new "<<(z+1)<<": "<<(out.layer)[zz][z]<<", "<<(out.straightness)[zz][z]<<", "<<(out.rank)[zz][z]<<" ";std::cout<<"\n\n";}//changing zones to Merged
  
  	   }
	 }
 }
