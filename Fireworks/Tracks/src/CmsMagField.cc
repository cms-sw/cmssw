#include "Fireworks/Tracks/interface/CmsMagField.h"
#include "TMath.h"

TEveVector CmsMagField::GetField(Float_t x, Float_t y, Float_t z) const
{
  double R = sqrt(x*x+y*y);
  double field = m_reverse?-GetMaxFieldMag():GetMaxFieldMag();
  //barrel
  if ( TMath::Abs(z)<724 ){
    
    //inside solenoid
    if ( R < 300) return TEveVector(0,0,field);
    // outside solinoid
    if ( m_simpleModel ||
	 ( R>461.0 && R<490.5 ) ||
	 ( R>534.5 && R<597.5 ) ||
	 ( R>637.0 && R<700.0 ) )
      return TEveVector(0,0,-field/3.8*1.2);
  
  } else {
    // endcaps
    if (m_simpleModel){
      if ( R < 50 ) return TEveVector(0,0,field);
      if ( z > 0 )
	return TEveVector(x/R*field/3.8*2.0, y/R*field/3.8*2.0, 0);
      else
	return TEveVector(-x/R*field/3.8*2.0, -y/R*field/3.8*2.0, 0);
    }
    // proper model
    if ( ( ( TMath::Abs(z)>724 ) && ( TMath::Abs(z)<786 ) ) ||
	 ( ( TMath::Abs(z)>850 ) && ( TMath::Abs(z)<910 ) ) ||
	 ( ( TMath::Abs(z)>975 ) && ( TMath::Abs(z)<1003 ) ) )
      {
	if ( z > 0 )
	  return TEveVector(x/R*field/3.8*2.0, y/R*field/3.8*2.0, 0);
	else
	  return TEveVector(-x/R*field/3.8*2.0, -y/R*field/3.8*2.0, 0);
      }
  }
  return TEveVector(0,0,0);
}

