#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cassert>
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
  case HF_EM         : return  CaloID(CaloID::DET_HF_EM); 
  case HF_HAD        : return  CaloID(CaloID::DET_HF_HAD); 
  case HGCAL         : return  CaloID(CaloID::DET_HGCAL_ENDCAP);
  default            : return  CaloID();
  }
}


PFLayer::Layer   PFLayer::fromCaloID( const CaloID& id) {

  //  cout<<"PFLayer::fromCaloID "<<id<<" "<<id.detector()<<endl;
  if( !id.isSingleDetector() ) {
    edm::LogError("PFLayer")<<"cannot convert "<<id<<" to a layer, as this CaloID does not correspond to a single detector"; 
  }

  switch( id.detector() ) {
  case CaloID::DET_ECAL_BARREL   : return  ECAL_BARREL;  
  case CaloID::DET_ECAL_ENDCAP   : return  ECAL_ENDCAP; 
  case CaloID::DET_PS1	         : return  PS1;
  case CaloID::DET_PS2	         : return  PS2;
  case CaloID::DET_HCAL_BARREL   : return  HCAL_BARREL1;
  case CaloID::DET_HCAL_ENDCAP   : return  HCAL_ENDCAP;
  case CaloID::DET_HF_EM         : return  HF_EM;
  case CaloID::DET_HF_HAD        : return  HF_HAD;
  case CaloID::DET_HO            : return  HCAL_BARREL2; 
  case CaloID::DET_HGCAL_ENDCAP  : return  HGCAL;
  default                        : return  NONE;
  }
}
