#include "L1Trigger/L1TCalorimeter/interface/Cordic.h"

#include <stdint.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <iomanip> 

Cordic::Cordic( const uint32_t& aPhiScale , const uint32_t& aMagnitudeBits , const uint32_t& aSteps ) : mPhiScale( aPhiScale ) , mMagnitudeScale( 1 << aMagnitudeBits ) , mMagnitudeBits( aMagnitudeBits ) , mSteps( aSteps ) , mPi( 3.1415926535897932384626433832795 )
{
  mRotations.reserve( mSteps );

  double lValue( 1.0 );

  for( uint32_t lStep( 0 ); lStep!=mSteps ; ++lStep ){
    lValue /= sqrt( 1.0 + pow( 4.0 , -double(lStep) ) );
    mRotations.push_back( tower( atan( pow( 2.0 , -double(lStep) ) ) ) );
  }
  mMagnitudeRenormalization = uint32_t( round( mMagnitudeScale * lValue ) );
}

Cordic::~Cordic(){}

double Cordic::NormalizePhi( const uint32_t& aPhi)
{
  return double( aPhi ) / double( mPhiScale );
}

double Cordic::NormalizeMagnitude( const uint32_t& aMagnitude )
{
  return double( aMagnitude ) / double( mMagnitudeScale );
}

int32_t Cordic::IntegerizeMagnitude( const double& aMagnitude )
{
  return int32_t( aMagnitude * mMagnitudeScale );
}

uint32_t Cordic::tower( const double& aRadians )
{
  return uint32_t( round( mPhiScale * 36.0 * aRadians / mPi ) );
}

void Cordic::operator() ( int32_t aX , int32_t aY , int32_t& aPhi , uint32_t& aMagnitude )
{
  bool lSign(true);

  switch( ((aY>=0)?0x0:0x2) | ((aX>=0)?0x0:0x1) ){
  case 0:
    aPhi = 0;
    break;
  case 1:
    aPhi = tower( mPi );
    lSign = false;
    aX = -aX;
    break;
  case 2:
    aPhi = tower( 2 * mPi );
    lSign = false;
    aY = -aY;    
    break;
  case 3:
    aPhi = tower( mPi );
    aX = -aX;
    aY = -aY;   
    break;
  default:
    throw 0;
  }

  for( uint32_t lStep( 0 ); lStep!=mSteps ; ++lStep ){
    if ( (aY < 0) == lSign ){
      aPhi -= mRotations[ lStep ];
    }else{    
      aPhi += mRotations[ lStep ];
    }

    int32_t lX(aX), lY(aY);
    if( lY < 0 ){
      aX = lX - (lY >> lStep);
      aY = lY + (lX >> lStep);
    }else{    
      aX = lX + (lY >> lStep);
      aY = lY - (lX >> lStep);
    }
  }

  aMagnitude = (aX * mMagnitudeRenormalization) >> mMagnitudeBits;
}
