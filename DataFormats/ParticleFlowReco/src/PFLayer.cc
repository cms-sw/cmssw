#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"

#include <assert.h>
#include <iostream>

using namespace reco;
using namespace std;

CaloID  PFLayer::toCaloID( Layer layer) {
  
  switch(layer) {
  case PS2           : return  CaloID(CaloID::DET_PS2);      
  case PS1           : return  CaloID(CaloID::DET_PS1); 
  case ECAL_ENDCAP   : return  CaloID(CaloID::DET_ECAL_ENDCAP); 
  case ECAL_BARREL   : return  CaloID(CaloID::DET_ECAL_BARREL);  
  case HCAL_BARREL1  : return  CaloID(CaloID::DET_HCAL_BARREL); 
  case HCAL_BARREL2  : return  CaloID(CaloID::DET_HO); 
  case HCAL_ENDCAP   : return  CaloID(CaloID::DET_HCAL_ENDCAP); 
  case VFCAL         : return  CaloID(CaloID::DET_HF); 
  default            : return  CaloID();
  }
}


PFLayer::Layer   PFLayer::fromCaloID( const CaloID& id) {

  //  cout<<"PFLayer::fromCaloID "<<id<<" "<<id.detector()<<endl;
  if( !id.isSingleDetector() ) {
    assert(0); 
  }

  switch( id.detector() ) {
  case CaloID::DET_ECAL_BARREL   : return  ECAL_BARREL;  
  case CaloID::DET_ECAL_ENDCAP   : return  ECAL_ENDCAP; 
  case CaloID::DET_PS1	         : return  PS1;
  case CaloID::DET_PS2	         : return  PS2;
  case CaloID::DET_HCAL_BARREL   : return  HCAL_BARREL1;
  case CaloID::DET_HCAL_ENDCAP   : return  HCAL_ENDCAP;
  case CaloID::DET_HF 	         : return  VFCAL;
  case CaloID::DET_HO            : return  HCAL_BARREL2; 
  default                        : return  NONE;
  }
}
