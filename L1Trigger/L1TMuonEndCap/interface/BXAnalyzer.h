//////Groups the output of TriggerPrimitive Converter into
//////groups of hits separated by a number of BX's. For the 
//////current trackfinder which looks either before or ahead
//////at most 2 BX's this requires 3 groups of 3 BX's, including
//////the central BX.
//////
//////Author M. Carver, UF
//////

#include "L1Trigger/L1TMuonEndCap/interface/EmulatorClasses.h"
#include "L1Trigger/L1TMuonEndCap/interface/PhiMemoryImage.h"


std::vector<std::vector<ConvertedHit>> GroupBX(std::vector<ConvertedHit> ConvHits){

	std::vector<ConvertedHit> tmp;
	std::vector<std::vector<ConvertedHit>> output (3,tmp);

	const int CentralBX = 6;

	for(std::vector<ConvertedHit>::iterator i = ConvHits.begin();i != ConvHits.end();i++){
	
		int diff = i->BX() - CentralBX;
		
		if((diff > -3) && (diff < 1))
			output[0].push_back(*i);
		
		if((diff < 3) && (diff > -1))
			output[2].push_back(*i);
		
		if(fabs(diff) < 2)
			output[1].push_back(*i);
	}
	
	return output;

}


PatternOutput DeleteDuplicatePatterns(std::vector<PatternOutput> Pout){

	std::vector<int> tmp (192,0);//was 128
	std::vector<std::vector<int>> rank (4,tmp), layer(4,tmp),straightness(4,tmp);
	std::vector<ConvertedHit> Hits;
	
	for(int i=0;i<3;i++){
		
		bool set = 0;
		
		for(int zone=0;zone<4;zone++){
			for(int strip=0;strip<192;strip++){//was 128
				
				if(Pout[i].detected.rank[zone][strip] >= rank[zone][strip]){
					
					rank[zone][strip] = Pout[i].detected.rank[zone][strip];
					layer[zone][strip] = Pout[i].detected.layer[zone][strip];
					straightness[zone][strip] = Pout[i].detected.straightness[zone][strip];
					set = 1;
				}
			}
		}
		
		if(set && (Pout[i].hits.size() > Hits.size())){
			
			std::vector<ConvertedHit> test = Pout[i].hits;
			
			for(std::vector<ConvertedHit>::iterator it = test.begin();it != test.end();it++){
				Hits.push_back(*it);
			}
		}
		
			
	}

	QualityOutput qout;
	
	qout.rank = rank;
	qout.layer = layer;
	qout.straightness = straightness;
	
	PatternOutput output;
	
	output.detected = qout;
	output.hits = Hits;
	
	return output;

}
