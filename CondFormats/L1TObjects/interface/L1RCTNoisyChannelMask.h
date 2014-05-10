#ifndef L1TObjects_L1RCTNoisyChannelMask_h
#define L1TObjects_L1RCTNoisyChannelMask_h
#include "CondFormats/Serialization/interface/Serializable.h"

#include <ostream>


struct L1RCTNoisyChannelMask {

  bool ecalMask[18][2][28];
  bool hcalMask[18][2][28];
  bool hfMask[18][2][4];

  float ecalThreshold;
  float hcalThreshold;
  float hfThreshold;


  void print(std::ostream& s) const{
    s << "Printing record L1RCTNoisyChannelMaskRcd " << std::endl;

     s << "ECAL noise mask threshold: ecalThreshold" << ecalThreshold << std::endl ;
     s << "HCAL noise mask threshold: hcalThreshold" << hcalThreshold << std::endl ;
     s << "HF noise mask threshold: hfThreshold" << hfThreshold << std::endl ;
    s << "Noisy Masked channels in L1RCTNoisyChannelMask" <<std::endl;
     for(int i = 0; i< 18; i++)
       for(int j =0; j< 2; j++){
         for(int k =0; k<28; k++){
           if(ecalMask[i][j][k])
             s << "ECAL masked noisy channel: RCT crate " << i << " iphi " << j <<" ieta " <<k <<std::endl; 
           if(hcalMask[i][j][k])
             s << "HCAL masked noisy channel: RCT crate " << i << " iphi " << j <<" ieta " <<k <<std::endl; 
         }
         for(int k =0; k<4;k++)
           if(hfMask[i][j][k])
             s << "HF masked noisy channel: RCT crate " << i << " iphi " << j <<" ieta " <<k <<std::endl; 
       }

  }

  COND_SERIALIZABLE;
};

#endif
