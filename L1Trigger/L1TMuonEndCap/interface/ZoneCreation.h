/////// Creation of Phi Zones used in Pattern Recognition
///////
///////Takes in vector of Converted Hits from Primitive Converter function and creates a set of 4 zones 
///////which are PhiMemoryImages containing all of the hits
///////

#ifndef ADD_ZoneCreation
#define ADD_ZoneCreation


#include "L1Trigger/L1TMuonEndCap/interface/PhiMemoryImage.h"
#include "L1Trigger/L1TMuonEndCap/interface/EmulatorClasses.h"

ZonesOutput zonemaker(std::vector<ConvertedHit> ConvHits){
  
  //bool verbose = false;
  PhiMemoryImage image0;
  std::vector<PhiMemoryImage> zones (4,image0);
  
  for (std::vector<ConvertedHit>::iterator h = ConvHits.begin(); h != ConvHits.end(); h++){ 

    int zmask[4] = {1,2,4,8};
    for(int zone=0;zone<4;zone++){
      if(h->ZoneWord() & zmask[zone]){
	zones[zone].SetBit(h->Station(),h->Zhit()+1);
      }
    }
  }
  
  ZonesOutput output;
  output.zone = zones;
  output.convertedhits = ConvHits;

  return output;
}

std::vector<ZonesOutput> Zones(std::vector<std::vector<ConvertedHit>> Hits){
  
  ZonesOutput tmp;
  std::vector<ZonesOutput> output (3,tmp);
  
  for(int i=0;i<3;i++)
    output[i] = zonemaker(Hits[i]);
  
  return output;
  
}

#endif
