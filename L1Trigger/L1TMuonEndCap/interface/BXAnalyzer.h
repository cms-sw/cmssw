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
    
    if(std::abs(diff) < 2)
      output[1].push_back(*i);
  }

  for (int i = 1; i < 3; i++) {
    unsigned int HasAllInNext = 0;
    for (unsigned int it = 0; it != output[i-1].size(); it++) {
      bool InNext = false;
      for (unsigned int it2 = 0; it2 != output[i].size(); it2++) {
	if( output[i-1][it].Phi() == output[i][it2].Phi() &&
	    output[i-1][it].Theta() == output[i][it2].Theta() &&
	    output[i-1][it].Ph_hit() == output[i][it2].Ph_hit() &&
	    output[i-1][it].Phzvl() == output[i][it2].Phzvl() &&
	    output[i-1][it].Station() == output[i][it2].Station() &&
	    output[i-1][it].Sub() == output[i][it2].Sub() &&
	    output[i-1][it].Id() == output[i][it2].Id() &&
	    output[i-1][it].Quality() == output[i][it2].Quality() &&
	    output[i-1][it].Pattern() == output[i][it2].Pattern() &&
	    output[i-1][it].Wire() == output[i][it2].Wire() &&
	    output[i-1][it].Strip() == output[i][it2].Strip() &&
	    output[i-1][it].BX() == output[i][it2].BX() ) InNext = true;
      }
      if (InNext) HasAllInNext++;
    }
    if (HasAllInNext == output[i-1].size()) output[i-1].clear();
  }
  
  for(int i=0;i<3;i++){
		
    for(std::vector<ConvertedHit>::iterator it = output[i].begin();it != output[i].end();it++){
      for(std::vector<ConvertedHit>::iterator it2 = it;it2 != output[i].end();it2++){
	
	if(it == it2) continue;
	
	//add that phis have to be equal if assuming that a phi position can have only 2 possible thetas
	if(it->Station() == it2->Station() && it->Id() == it2->Id() && it->IsNeighbor() == it2->IsNeighbor()){
	  
	  it->SetTheta2(it2->Theta());
	  it2->SetTheta2(it->Theta());
	  
	  it->AddTheta(it2->Theta());
	  it2->AddTheta(it->Theta());
	}
	
      }
    }
    
  }

  return output;
}

PatternOutput DeleteDuplicatePatterns(std::vector<PatternOutput> Pout){

  std::vector<int> tmp (192,0);//was 128
  std::vector<std::vector<int>> rank (4,tmp), layer(4,tmp),straightness(4,tmp),bxgroup(4,tmp);
  std::vector<ConvertedHit> Hits;
  
  for(int i=0;i<3;i++){
    
    bool set = 0;
    
    for(int zone=0;zone<4;zone++){
      for(int strip=0;strip<192;strip++){//was 128
	
	if(Pout[i].detected.rank[zone][strip] > rank[zone][strip]){
	  
	  rank[zone][strip] = Pout[i].detected.rank[zone][strip];
	  layer[zone][strip] = Pout[i].detected.layer[zone][strip];
	  straightness[zone][strip] = Pout[i].detected.straightness[zone][strip];
	  bxgroup[zone][strip] = i+1;
	  set = 1;
	}
      }
    }
    
    if(set ){/*//&& (Pout[i].hits.size() >= Hits.size())){*/
      
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
  qout.bxgroup = bxgroup;
  
  PatternOutput output;
  output.detected = qout;
  output.hits = Hits;
	
  return output;
  
}
