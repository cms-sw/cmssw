#include <iostream>
#include <stdlib.h>
#include <string>
#include <assert.h>
using namespace std;

#include "CalibCalorimetry/EcalLaserAnalyzer/interface/ME.h"
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/MEEBGeom.h"

#include <TGraph.h>

//GHM ClassImp(MEEBGeom)

int 
MEEBGeom::barrel( EBGlobalCoord ieta, EBGlobalCoord iphi )
{
  int iz=1;
  if( ieta<0 ) iz=-1;
  ieta *= iz;
  assert( ieta>0 && ieta<=85 );
  if( iphi<0 ) iphi+=360;
  assert( iphi>0 && iphi<=360 );
  return iz;
}

int 
MEEBGeom::sm( EBGlobalCoord ieta, EBGlobalCoord iphi )
{
  int iz=1;
  if( ieta<0 ) iz=-1;
  ieta *= iz;
  assert( ieta>0 && ieta<=85 );

  //  int iphi_ = iphi+10;
  int iphi_ = iphi;
  if( iphi_>360 ) iphi_-=360;
  assert( iphi_>0 && iphi_<=360 );

  int ism = (iphi_-1)/20 + 1;
  assert( ism>=1 && ism<=18 );
  //  if( iz==1 ) ism += 18;
  if( iz==-1 ) ism += 18;

  return ism;
}

int 
MEEBGeom::dcc( EBGlobalCoord ieta, EBGlobalCoord iphi )
{
  int ism = sm( ieta, iphi );
  return dccFromSm( ism );
}

int
MEEBGeom::dccFromSm( int ism )
{
  assert( ism>=1 && ism<=36 );
  int iz=1;
  if( ism>18 ) iz=-1;
  if( iz==-1  ) ism-=18;
  assert( ism>=1 && ism<=18 );
  int idcc = 9+ism;
  if( iz==+1  ) idcc+=18;
  return idcc;
}

int
MEEBGeom::smFromDcc( int idcc )
{
  if( idcc>600 ) idcc-=600;  // also works with FEDids
  assert( idcc>=10 && idcc<=45 );
  int ism=idcc-9;
  if( ism>18 ) ism-=18;
  else         ism+=18;
  return ism;
}

TString
MEEBGeom::smName( int ism )
{
  assert( ism>=1 && ism<=36 );
  TString out = "EB+";
  if( ism>18 )
    {
      out = "EB-";
      ism -= 18;
    }
  out += ism;
  return out;
}

int 
MEEBGeom::lmmod( EBGlobalCoord ieta, EBGlobalCoord iphi )
{
  std::pair<EBLocalCoord,EBLocalCoord> ixy = localCoord( ieta, iphi );
  return lm_channel( ixy.first/5, ixy.second/5 );
}

int 
MEEBGeom::tt( EBGlobalCoord ieta, EBGlobalCoord iphi )
{
  std::pair<EBLocalCoord,EBLocalCoord> ixy = localCoord( ieta, iphi );
  return tt_channel( ixy.first/5, ixy.second/5 );
}

int 
MEEBGeom::crystal( EBGlobalCoord ieta, EBGlobalCoord iphi )
{
  std::pair<EBLocalCoord,EBLocalCoord> ixy = localCoord( ieta, iphi );
  return crystal_channel( ixy.first, ixy.second );
}

int 
MEEBGeom::side( EBGlobalCoord ieta, EBGlobalCoord iphi )
{
  int ilmmod = lmmod( ieta, iphi );
  return (ilmmod%2==0)?1:0;
}

int 
MEEBGeom::lmr( EBGlobalCoord ieta, EBGlobalCoord iphi )
{
  int idcc  = dcc( ieta, iphi );
  int ism   = idcc-9;
  int iside = side( ieta, iphi );
  int ilmr = 1 + 2*(ism-1) + iside;
  return ilmr;
}

std::pair< MEEBGeom::EBLocalCoord, MEEBGeom::EBLocalCoord >    
MEEBGeom::localCoord( MEEBGeom::EBGlobalCoord ieta, MEEBGeom::EBGlobalCoord iphi )
{
  int iz=1;
  if( ieta<0 ) iz=-1;
  ieta *= iz;
  assert( ieta>0 && ieta<=85 );

  //  int iphi_ = iphi+10;
  int iphi_ = iphi;
  if( iphi_>360 ) iphi_-=360;
  assert( iphi_>0 && iphi_<=360 );

  int ix = ieta-1;
  
  int iy = (iphi_-1)%20;
  if( iz==-1 ) iy = 19-iy;
  // if( iz==1 ) iy = 19-iy;

  return std::pair< EBLocalCoord, EBLocalCoord >(ix,iy);  
}

std::pair< MEEBGeom::EBLocalCoord, MEEBGeom::EBLocalCoord >    
MEEBGeom::localCoord( int icr )
{
  assert( icr>=1 && icr<=1700 );
  int ix = (icr-1)/20;
  int iy = 19 - (icr-1)%20;
  return std::pair< EBLocalCoord, EBLocalCoord >(ix,iy);  
}

std::pair< MEEBGeom::EBGlobalCoord, MEEBGeom::EBGlobalCoord >    
MEEBGeom::globalCoord( int ism, MEEBGeom::EBLocalCoord ix, MEEBGeom::EBLocalCoord iy )
{
  assert( ism>=1 && ism<=36 );
  assert( ix>=0 && ix<85 );
  assert( iy>=0 && iy<20 );
  //  int iz=-1;
  int iz=1;
  if( ism>18 ) 
    {
      iz=-1;
      ism -= 18;
    }
  if( iz==-1 ) iy = 19-iy;
  // if( iz==1 ) iy = 19-iy;

  int ieta = ix+1;
  ieta *= iz;
  //  int iphi = -9 + iy + 20*(ism-1);
  int iphi = 1 + iy + 20*(ism-1);

  return std::pair< EBGlobalCoord, EBGlobalCoord >(ieta,iphi);  
}

std::pair< float, float >    
MEEBGeom::globalCoord( int ism, float x, float y )
{
  assert( ism>=1 && ism<=36 );
  int iz=1;
  if( ism>18 ) 
    {
      iz=-1;
      ism -= 18;
    }
  if( iz==-1 ) y = 19-y;

  float eta = x+1;
  eta *= iz;
  //  float phi = -9 + y + 20*(ism-1);
  float phi = 1 + y + 20*(ism-1);

  return std::pair< float, float >(eta,phi);  
}

std::pair< MEEBGeom::EBGlobalCoord, MEEBGeom::EBGlobalCoord >    
MEEBGeom::globalCoord( int ism, int icr )
{
  assert( ism>=1 && ism<=36 );
  assert( icr>=1 && icr<=1700 );

  int ix = (icr-1)/20;
  int iy = 19 - (icr-1)%20;

  return globalCoord( ism, ix, iy );  
}

int 
MEEBGeom::lm_channel( EBTTLocalCoord iX, EBTTLocalCoord iY )
{
  static const int 
    idx_[] = 
    {
     // 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
        1, 2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6, 8, 8, 8, 8, // 3
        1, 2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6, 8, 8, 8, 8, // 2
        1, 3, 3, 3, 3, 5, 5, 5, 5, 7, 7, 7, 7, 9, 9, 9, 9, // 1
        1, 3, 3, 3, 3, 5, 5, 5, 5, 7, 7, 7, 7, 9, 9, 9, 9  // 0
     // 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
     };

  int iym, ixm, il, ic, ii;
  iym=4;
  ixm=17;
  int iX_ = iX+1;
  int iY_ = iY+1;
  il=iym-iY_;
  ic=iX_-1;
  ii=il*ixm+ic;
  if(ii < 0 || ii > (int)(sizeof(idx_)/sizeof(int))) { return -1; };
  return idx_[ii];
}

int 
MEEBGeom::tt_type( EBTTLocalCoord iX, EBTTLocalCoord iY )
{
  static const int 
    idx_[] =
    {
     // 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
        1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, // 3
        1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, // 2
        1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, // 1
        1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2  // 0
     // 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
     };

  int iym, ixm, il, ic, ii;
  iym=4;
  ixm=17;
  int iX_ = iX+1;
  int iY_ = iY+1;
  il=iym-iY_;
  ic=iX_-1;
  ii=il*ixm+ic;
  if(ii < 0 || ii > (int)(sizeof(idx_)/sizeof(int))) { return -1; };
  return idx_[ii];
}

int
MEEBGeom::hv_channel( EBTTLocalCoord iX, EBTTLocalCoord iY )
{
  static const int 
    idx_[] = 
    {
     // 0  1  2  3   4   5   6   7   8   9  10  11  12  13  14  15  16
        1, 3, 5, 7,  9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, // 3
        1, 3, 5, 7,  9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, // 2
        2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, // 1
        2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34  // 0
     // 0  1  2  3   4   5   6   7   8   9  10  11  12  13  14  15  16
     };

  int iym, ixm, il, ic, ii;
  iym=4;
  ixm=17;
  int iX_ = iX+1;
  int iY_ = iY+1;
  il=iym-iY_;
  ic=iX_-1;
  ii=il*ixm+ic;
  if(ii < 0 || ii > (int)(sizeof(idx_)/sizeof(int))) { return -1; };
  return idx_[ii];
}

int
MEEBGeom::lv_channel( EBTTLocalCoord iX, EBTTLocalCoord iY )
{
  static const int
    idx_[] =
    {
     // 0  1  2  3  4  5  6  7  8   9  10  11  12  13  14  15  16
        1, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14, 16, 16, // 3
        1, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14, 16, 16, // 2
        1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15, 17, 17, // 1
        1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15, 17, 17  // 0
     // 0  1  2  3  4  5  6  7  8   9  10  11  12  13  14  15  16
     };

  int iym, ixm, il, ic, ii;
  iym=4;
  ixm=17;
  int iX_ = iX+1;
  int iY_ = iY+1;
  il=iym-iY_;
  ic=iX_-1;
  ii=il*ixm+ic;
  if(ii < 0 || ii > (int)(sizeof(idx_)/sizeof(int))) { return -1; };
  return idx_[ii];
}

int 
MEEBGeom::tt_channel( EBTTLocalCoord iX, EBTTLocalCoord iY )
{

  int itt =  4*iX+4-iY;  // :)

  return itt;

}

int 
MEEBGeom::crystal_channel( EBLocalCoord ix, EBLocalCoord iy )
{
  // "Test beam numbering"
  int icr =  20*ix + 19-iy + 1;  // :)

  return icr;
}

int 
MEEBGeom::electronic_channel( EBLocalCoord ix, EBLocalCoord iy )
{
  int iX = ix/5;
  int iY = iy/5;
  int itt  = tt_channel( iX, iY );
  int type = tt_type( iX, iY );

  int iVFE = ix%5+1;
  int islot = iy%5+1;
  if( iVFE%2==1 ) islot = 6-islot;
  int icr = 5*(iVFE-1)+(islot-1);
  // rotate for type-1 towers
  if( type==1 )
    {
      icr = 24-icr;
    }
  icr += 25*(itt-1);

  return icr;
}

TGraph*
MEEBGeom::getGraphBoundary(  int type, int num, bool global )
{
  int ism_=0;
  if( type==iSuperModule )
    {
      ism_ = num; 
    }
  else if( type==iLMRegion )
    {
      ism_ = (num-1)/2+1;
      if( ism_>18 ) 
	ism_-=18;
      else
	ism_+=18;
    }
  else
      abort();

  int ism=1;
  if( global ) ism = ism_;

  //  std::list< std::pair< float, float > > l;
  // getBoundary( l, type, num, global, ism );
  //  int n = l.size();
  //  if( n==0 ) return 0;
  
  float ix[100];
  float iy[100];
  int ixmin =  0;
  int ixmax = 84;
  int iymin =  0;
  int iymax = 19;

  int n(0);
  if( type==iSuperModule )
    {
      n=5;
      ix[0] = ixmin-0.5;
      iy[0] = iymin-0.5;
      ix[1] = ixmin-0.5;
      iy[1] = iymax+0.5;
      ix[2] = ixmax+0.5;
      iy[2] = iymax+0.5;
      ix[3] = ixmax+0.5;
      iy[3] = iymin-0.5;
      ix[4] = ix[0];
      iy[4] = iy[0];
    }
  else if( type==iLMRegion )
    {
      int iside = num;
      if( global )
	{
	  iside = (num-1)%2;
	}
      
      if( iside==1 )
	{
	  n=5;
	  ix[0] = ixmin+5-0.5;
	  iy[0] = iymin+10-0.5;
	  ix[1] = ixmin+5-0.5;
	  iy[1] = iymax+0.5;
	  ix[2] = ixmax+0.5;
	  iy[2] = iymax+0.5;
	  ix[3] = ixmax+0.5;
	  iy[3] = iymin+10-0.5;
	  ix[4] = ix[0];
	  iy[4] = iy[0];
	}
      else
	{
	  n=7;
	  ix[0] = ixmin-0.5;
	  iy[0] = iymin-0.5;
	  ix[1] = ixmin-0.5;
	  iy[1] = iymax+0.5;
	  ix[2] = ixmin+5-0.5;
	  iy[2] = iymax+0.5;
	  ix[3] = ixmin+5-0.5;
	  iy[3] = iymax-10+0.5;
	  ix[4] = ixmax+0.5;
	  iy[4] = iymax-10+0.5;
	  ix[5] = ixmax+0.5;
	  iy[5] = iymin-0.5;
	  ix[6] = ix[0];
	  iy[6] = iy[0];
	}
    }

  if( global )
    {
      for( int ii=0; ii<n; ii ++ )
	{
	  std::pair<float,float> xy = globalCoord( ism, ix[ii], iy[ii] ); 
	  ix[ii] = xy.first;
	  iy[ii] = xy.second;
	}
    }

//   int ii=0;
//   std::list< std::pair< float, float > >::const_iterator l_it;      
//   for( l_it=l.begin(); l_it!=l.end(); l_it++ )
//     {
//       //      std::cout << "[" << l_it->first << "," << l_it->second << "]" << std::endl;
//       ix[ii] = l_it->first;
//       iy[ii] = l_it->second;
//       ii++;
//     }


//  assert( ii==n );
  return new TGraph( n, ix, iy );
}

std::pair< int, int >
MEEBGeom::pn( int ilmmod )
{
  switch( ilmmod )
    {
    case   1: return std::pair<int,int>(  0,  5 );
    case   2: return std::pair<int,int>(  1,  6 );
    case   3: return std::pair<int,int>(  1,  6 );
    case   4: return std::pair<int,int>(  2,  7 );
    case   5: return std::pair<int,int>(  2,  7 );
    case   6: return std::pair<int,int>(  3,  8 );
    case   7: return std::pair<int,int>(  3,  8 );
    case   8: return std::pair<int,int>(  4,  9 );
    case   9: return std::pair<int,int>(  4,  9 );
    default:
      abort();
    }
  return std::pair<int,int>(-1,-1);
}

std::pair<int,int> 
MEEBGeom::memFromLmr( int ilmr )
{
  std::pair< int, int > dccAndSide_ = ME::dccAndSide( ilmr );
  int idcc  = dccAndSide_.first;
  return std::pair<int,int>( idcc, idcc );
}

std::vector<int> 
MEEBGeom::lmmodFromLmr( int ilmr )
{
  std::pair< int, int > dccAndSide_ = ME::dccAndSide( ilmr );
  int iside = dccAndSide_.second;
  std::vector< int > vec;
  for( int ilmmod=1; ilmmod<=9; ilmmod++ )
    {
      if( (ilmmod+iside)%2==1 )  vec.push_back(ilmmod);
    }
  return vec;
}

int
MEEBGeom::apdRefTower( int ilmmod )
{
  switch( ilmmod )
    {
      /* case   1: return 1;
    case   2: return 5; 
    case   3: return 7;
    case   4: return 21; 
    case   5: return 23;
    case   6: return 37;
    case   7: return 39;
    case   8: return 53;
    case   9: return 55;
      */
    case   1: return 2;
    case   2: return 6; 
    case   3: return 8;
    case   4: return 22; 
    case   5: return 24;
    case   6: return 38;
    case   7: return 40;
    case   8: return 54;
    case   9: return 56;
    default:
      abort();
    }
  return 0;
}


std::vector< int >
MEEBGeom::apdRefChannels( int ilmmod )
{
  
  std::vector< int > vec;
  switch( ilmmod )
    {
    case   1:  vec.push_back(0); vec.push_back(24); break;
    case   2:  vec.push_back(0); vec.push_back(24); break;
    case   3:  vec.push_back(0); vec.push_back(24); break;
    case   4:  vec.push_back(0); vec.push_back(24); break;
    case   5:  vec.push_back(0); vec.push_back(24); break;
    case   6:  vec.push_back(0); vec.push_back(24); break;
    case   7:  vec.push_back(0); vec.push_back(24); break;
    case   8:  vec.push_back(0); vec.push_back(24); break;
    case   9:  vec.push_back(0); vec.push_back(24); break;
    default:
      abort();
    }
  return vec;
}

