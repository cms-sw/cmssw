#include "CondFormats/L1TObjects/interface/L1RPCHsbConfig.h"
#include "FWCore/Utilities/interface/Exception.h"

L1RPCHsbConfig::L1RPCHsbConfig() 
{

}


L1RPCHsbConfig::~L1RPCHsbConfig(){



}


void L1RPCHsbConfig::setHsbMask(int hsb, const std::vector<int>& mask ) {

  if (getMaskSize() != int(mask.size()) )  {
     throw cms::Exception("L1RPCHsbConfig::setHsbMask") 
       << "Wrong size of hsb mask "
       << mask.size() << " " << getMaskSize()
       << "\n";
  
  }


  if (hsb != 0 && hsb != 1) {
     throw cms::Exception("L1RPCHsbConfig::getHsbMask") 
       << "Wrong hsb index: " << hsb << "\n";
  }
  if ( hsb == 0 ){
     for (int i = 0 ; i < getMaskSize() ; ++i ) {m_hsb0[i] = mask[i];}
  }

  if ( hsb == 1 ){
     for (int i = 0 ; i < getMaskSize() ; ++i ) {m_hsb1[i] = mask[i];}
  }
}


int  L1RPCHsbConfig::getHsbMask(int hsb, int input) const {

   if (input < 0 || input >= int(sizeof(m_hsb0)/(sizeof( m_hsb0[0] )) ) )
   {
     throw cms::Exception("L1RPCHsbConfig::getHsbMask") 
       << "Wrong hsb input index: " << input << "\n";
   }
   if (hsb==0) {
     return m_hsb0[input];
   } else if (hsb==1) {
     return m_hsb1[input];
   } else {
     throw cms::Exception("L1RPCHsbConfig::getHsbMask") 
       << "Wrong hsb index: " << hsb << "\n";
   }

   return -1;
}
