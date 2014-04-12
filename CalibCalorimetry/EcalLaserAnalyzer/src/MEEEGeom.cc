#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <string>
using namespace std;

#include "CalibCalorimetry/EcalLaserAnalyzer/interface/ME.h"
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/MEEEGeom.h"

//GHM ClassImp(MEEEGeom)

bool 
MEEEGeom::pnTheory = true;

int 
MEEEGeom::quadrant( SuperCrysCoord iX, SuperCrysCoord iY )
{
  bool near = iX>=11;  bool far  = !near;
  bool top  = iY>=11;  bool bot  = !top;
  
  int iquad=0;
  if( near && top ) iquad=1;
  if( far  && top ) iquad=2;
  if( far  && bot ) iquad=3;
  if( near && bot ) iquad=4;
  
  return iquad;
}

int 
MEEEGeom::sector( SuperCrysCoord iX, SuperCrysCoord iY )
{
 
  //  Y (towards the surface)
  //  T
  //  |
  //  |
  //  |
  //  o---------| X  (towards center of LHC)
  //
  static const int 
    idx_[] =
    {
     // 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
     0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0, // 20
     0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, // 19
     0, 0, 0, 2, 1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9, 8, 0, 0, 0, // 18
     0, 0, 2, 2, 2, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 8, 8, 8, 0, 0, // 17
     0, 2, 2, 2, 2, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 8, 8, 8, 8, 0, // 16
     0, 2, 2, 2, 2, 2, 1, 1, 1, 1, 9, 9, 9, 9, 8, 8, 8, 8, 8, 0, // 15
     0, 2, 2, 2, 2, 2, 2, 1, 1, 1, 9, 9, 9, 8, 8, 8, 8, 8, 8, 0, // 14
     2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, // 13
     3, 3, 2, 2, 2, 2, 2, 2, 2, 0, 0, 8, 8, 8, 8, 8, 8, 8, 7, 7, // 12
     3, 3, 3, 3, 3, 3, 3, 2, 0, 0, 0, 0, 8, 7, 7, 7, 7, 7, 7, 7, // 11
     3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7, // 10
     3, 3, 3, 3, 3, 3, 3, 4, 4, 0, 0, 6, 6, 7, 7, 7, 7, 7, 7, 7, // 9
     3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 7, 7, 7, 7, 7, 7, // 8
     0, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 0, // 7
     0, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 0, // 6
     0, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 0, // 5
     0, 0, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 0, 0, // 4
     0, 0, 0, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 0, 0, 0, // 3
     0, 0, 0, 0, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 0, 0, 0, 0, // 2
     0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0  // 1
     // 1  2  3  4  5  6  7  8  9 10   11 12 13 14 15 16 17 18 19 20 
     };


  int iym, ixm, il, ic, ii;
  iym=20;
  ixm=20;
  int iX_ = iX;
  int iY_ = iY;
  il=iym-iY_;
  ic=iX_-1;
  ii=il*ixm+ic;

  if(ii < 0 || ii > (int)(sizeof(idx_)/sizeof(int)) || idx_[ii] == 0) { return -1; };
  return idx_[ii];
}

int 
MEEEGeom::sm( SuperCrysCoord iX, SuperCrysCoord iY, int iz )
{
  // this is *my* convention. To be consistent with the barrel
  // sm goes from 1 to 9 for iz+ and from 10 to 18 for iz-
  int ism_ = sector( iX, iY );
  if( ism_<0 ) return ism_;
  if( iz<0 ) ism_+=9;
  return ism_;
}

int 
MEEEGeom::lmmod( SuperCrysCoord iX, SuperCrysCoord iY )
{
  //
  // laser monitoring modules for EE-F and EE+F
  // for EE-N and EE+N :  iX ---> 20-iX+1
  //  

  static const int 
    idx_[] =
    {
     // 1  2  3   4   5   6   7   8   9   A 
     //------------------------------
      0,  0,  0,  0,  0,  0,  0,  1,  1,  1,
      0,  0,  0,  0,  2,  1,  1,  1,  1,  1,
      0,  0,  0,  5,  2,  2,  2,  2,  2,  1,
      0,  0,  5,  5,  5,  3,  3,  2,  2,  2,
      0,  8,  5,  5,  5,  3,  3,  3,  3,  3,
      0,  8,  8,  5,  6,  6,  4,  4,  4,  3,
      0,  8,  8,  5,  6,  6,  7,  4,  4,  4,
      8,  8,  8,  6,  6,  7,  7,  7,  4,  4,
      9,  9,  8,  6,  6,  7,  7,  7,  7,  0,
      9,  9,  9, 10, 10, 11, 11,  7,  0,  0,
     12,  9,  9, 10, 10, 11, 11, 11,  0,  0,
     12, 12, 13, 10, 10, 11, 11, 17, 17,  0,
     12, 12, 13, 13, 13, 11, 17, 17, 17, 19,
      0, 12, 13, 13, 14, 15, 17, 17, 17, 19,
      0, 12, 14, 14, 14, 15, 16, 17, 19, 19,
      0, 14, 14, 14, 14, 15, 16, 16, 19, 19,
      0,  0, 14, 15, 15, 15, 16, 16, 19, 19,
      0,  0,  0, 15, 15, 15, 16, 18, 18, 18,
      0,  0,  0,  0, 16, 16, 16, 18, 18, 18,
      0,  0,  0,  0,  0,  0,  0, 18, 18, 18
     };

  int iym, ixm, il, ic, ii;
  iym=20;
  ixm=10;
  int iX_ = iX;
  if( iX>=11 ) iX_ = 20-iX+1;
  int iY_ = iY;
  il=iym-iY_;
  ic=iX_-1;
  ii=il*ixm+ic;
  if(ii < 0 || ii > (int)(sizeof(idx_)/sizeof(int)) || idx_[ii] == 0) { return -1; };
  return idx_[ii];
}

int 
MEEEGeom::sc_in_quad( SuperCrysCoord iX, SuperCrysCoord iY )
{
  static const int 
    idx_[] =
    {
     // 1  2  3  4  5  6  7  8  9  A 
     //-----------------------------------
       77, 71, 63, 55, 46, 37, 28, 18,  0,  0, // A
       78, 72, 64, 56, 47, 38, 29, 19,  9,  0, // 9
       79, 73, 65, 57, 48, 39, 30, 20, 10,  1, // 8
        0, 74, 66, 58, 49, 40, 31, 21, 11,  2, // 7
        0, 75, 67, 59, 50, 41, 32, 22, 12,  3, // 6
        0, 76, 68, 60, 51, 42, 33, 23, 13,  4, // 5
        0,  0, 69, 61, 52, 43, 34, 24, 14,  5, // 4
        0,  0, 70, 62, 53, 44, 35, 25, 15,  6, // 3
        0,  0,  0,  0, 54, 45, 36, 26, 16,  7, // 2
        0,  0,  0,  0,  0,  0,  0, 27, 17,  8, // 1
     //-----------------------------------
     };
  int iym, ixm, il, ic, ii;
  iym=10;
  ixm=10;
  int iX_ = iX;
  if( iX>=11 ) iX_ = 20-iX+1;
  int iY_ = iY;
  if( iY>=11 ) iY_ = 20-iY+1;
  il=iym-iY_;
  ic=iX_-1;
  ii=il*ixm+ic;
  if(ii < 0 || ii > (int)(sizeof(idx_)/sizeof(int)) || idx_[ii] == 0) { return -1; };
  return idx_[ii];
}

int 
MEEEGeom::sc_type( SuperCrysCoord iX, SuperCrysCoord iY )
{
  static const int 
    idx_[] =
    {
     // there are seven types of super-crystals
     // 1  2  3  4  5  6  7  8  9 10 
     //-----------------------------------
        0,  0,  0,  0,  0,  0,  0,  3, -1, -1, // 10
        0,  0,  0,  0,  0,  0,  0,  0,  2, -1, //  9
        6,  0,  0,  0,  0,  0,  0,  0,  0,  1, //  8
       -1,  0,  0,  0,  0,  0,  0,  0,  0,  0, //  7
       -1,  0,  0,  0,  0,  0,  0,  0,  0,  0, //  6
       -1,  6,  0,  0,  0,  0,  0,  0,  0,  0, //  5
       -1, -1,  6,  0,  0,  0,  0,  0,  0,  0, //  4
       -1, -1,  5,  4,  0,  0,  0,  0,  0,  0, //  3
       -1, -1, -1, -1,  4,  0,  0,  0,  0,  0, //  2
       -1, -1, -1, -1, -1, -1, -1,  4,  0,  0, //  1
     //-----------------------------------
     };
  int iym, ixm, il, ic, ii;
  iym=10;
  ixm=10;
  int iX_ = iX;
  if( iX>=11 ) iX_ = 20-iX+1;
  int iY_ = iY;
  if( iY>=11 ) iY_ = 20-iY+1;
  il=iym-iY_;
  ic=iX_-1;
  ii=il*ixm+ic;
  if(ii < 0 || ii > (int)(sizeof(idx_)/sizeof(int)) || idx_[ii] == -1) { return -1; };
  return idx_[ii];
}

int 
MEEEGeom::sc( SuperCrysCoord iX, SuperCrysCoord iY )
{
  int isc_in_quad = sc_in_quad( iX, iY );
  assert( isc_in_quad!=0 );
  if( isc_in_quad<0 ) return -1;

  int iquad = quadrant( iX, iY );
  return 79*(iquad-1)+isc_in_quad;
}

int 
MEEEGeom::dcc( SuperCrysCoord iX, SuperCrysCoord iY, int iz )
{
  // end-cap minus:
  // S7  --> DCC-1 (N)
  // S8  --> DCC-2 (N)
  // S9  --> DCC-3 (N)
  // S1  --> DCC-4 (F) 
  // S2  --> DCC-5 (F)
  // S3  --> DCC-6 (F)
  // S4  --> DCC-7 (F)
  // S5  --> DCC-8 (F and N)
  // S6  --> DCC-9 (N)
  // for the end-cap plus, add 45 to the DCC number
  int isect = sector( iX, iY );
  assert( isect!=0 );
  assert( abs(iz)==1 );
  if( isect<0 ) return -1;

  int idcc=0;
  
  idcc = isect-6;
  if( idcc<=0 ) idcc+=9;
  if( iz==+1 )  idcc+=45;

  return idcc;
}

int 
MEEEGeom::lmr( SuperCrysCoord iX, SuperCrysCoord iY, int iz )
{
  // laser monitoring regions
  // end-cap minus:
  // S7  --> LM-1
  // S8  --> LM-2
  // S9  --> LM-3
  // S1  --> LM-4 
  // S2  --> LM-5 
  // S3  --> LM-6 
  // S4  --> LM-7 
  // S5  --> LM-8 (F) and LM-9 (N)
  // S6  --> LM-10
  // for the end-cap plus,  add 72 to the LM number
  // for the end-cap minus, add 82 to the LM number
  
  int iquad = quadrant( iX, iY );
  int isect = sector( iX, iY );
  assert( isect!=0 );
  assert( abs(iz)==1 );
  if( isect<0 ) return -1;

  int ilmr=0;
  ilmr = isect-6;
  if( ilmr<=0 ) ilmr+=9;
  if( ilmr==9 ) ilmr++;
  if( ilmr==8 && iquad==4 ) ilmr++;
  if( iz==+1 )  ilmr+=72;
  else          ilmr+=82;

  return ilmr;
}

int 
MEEEGeom::dee( SuperCrysCoord iX, SuperCrysCoord iY, int iz )
{
  int iquad = quadrant( iX, iY );
  int idee=0;
  bool far = ( iquad==2 || iquad==3 );  bool near = !far;
  bool plus = (iz>0); bool minus = !plus;
  if( far  && plus  ) idee=1;
  if( near && plus  ) idee=2;
  if( near && minus ) idee=3;
  if( far  && minus ) idee=4; 

  return idee;
}

int 
MEEEGeom::crystal_in_sc( CrysCoord ix, CrysCoord iy )
{
  // the seven types of super-crystals
  static const int 
    idx_[7][25] =
    {
            {
      21, 16, 11,  6,  1,
      22, 17, 12,  7,  2,
      23, 18, 13,  8,  3,
      24, 19, 14,  9,  4,
      25, 20, 15, 10,  5
            }, { 
      -1, -1, -1, -1, -1,
      22, 17, 12,  7,  2,
      23, 18, 13,  8,  3,
      24, 19, 14,  9,  4,
      25, 20, 15, 10,  5,
            }, {
      -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      23, -1, -1, -1, -1,          
      24, 19, -1, -1, -1,
      25, 20, 15, -1, -1
            }, {
      21, 16, 11,  6, -1,
      22, 17, 12,  7, -1,
      23, 18, 13,  8, -1,
      24, 19, 14,  9, -1,
      25, 20, 15, 10, -1
            }, {
      21, 16, 11,  6,  1,
      22, 17, 12,  7,  2,
      -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
            }, {
      -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
            }, {
      -1, -1, -1,  6,  1,
      -1, -1, -1,  7,  2,
      -1, -1, -1,  8,  3,
      -1, -1, -1,  9,  4,
      -1, -1, -1, 10,  5
            }
    };
  
  int iX, iY, jx, jy;
  int iX_ = (ix-1)/5+1;
  int iY_ = (iy-1)/5+1;
  int jx_ = ix - 5*(iX_-1);
  int jy_ = iy - 5*(iY_-1);

  int iquad = quadrant( iX_, iY_ );
  if( iX_>=11 ) 
    {
      iX_ = 20-iX_+1;
      jx_ =  5-jx_+1;
    }
  if( iY_>=11 ) 
    {
      iY_ = 20-iY_+1;
      jy_ =  5-jy_+1;
    }

  // FIXME : this is stupid !!!
  if( iquad==1 || iquad==3 )
    {
      iX = iX_;
      iY = iY_;
      jx = jx_;
      jy = jy_;
    }
  else
    {
      iX = iY_;
      iY = iX_;
      jx = jy_;
      jy = jx_;
    }

  int isc_type = sc_type( iX, iY );
  assert( isc_type>=0 && isc_type<7 );
  
  int iym, ixm, il, ic, ii;
  iym=5;
  ixm=5;
  il=iym-jy;
  ic=jx-1;
  ii=il*ixm+ic;
  if(ii < 0 || ii > (int)(sizeof(idx_)/sizeof(int)) || idx_[isc_type][ii] == -1) { return -1; };
  return idx_[isc_type][ii];
}

int 
MEEEGeom::crystal( CrysCoord ix, CrysCoord iy )
{

  int iX = (ix-1)/5+1;
  int iY = (iy-1)/5+1;
  int isc = sc( iX, iY );
  assert( isc!=0 );
  if( isc<0 ) return -2;

  int icr_in_sc = crystal_in_sc( ix, iy );
  assert( icr_in_sc!=0 );
  if( icr_in_sc<0 ) return -1;
  
  return 25*(isc-1) + icr_in_sc;
}

TString
MEEEGeom::smName( int ism )
{
  assert( ism>=1 && ism<=18 );
  TString out = "EE+";
  if( ism>9 )
    {
      out = "EE-";
      ism -= 9;
    }
  out += ism;
  return out;
}

int
MEEEGeom::dccFromSm( int ism )
{
  assert( ism>=1 && ism<=18 );
  int dcc_[18] = 
    { 49, 50, 51, 52, 53, 54, 46, 47, 48, 
       4,  5,  6,  7,  8,  9,  1,  2,  3 };
  return dcc_[ism-1];
}

int
MEEEGeom::smFromDcc( int idcc )
{
  if( idcc>600 ) idcc-=600;  // also works with FEDids
  int ism(0);
  if( idcc>=1 && idcc<=9 )
    {
      ism = 6+idcc;
      if( ism>9 ) ism-=9;
      ism+=9;
    }
  else if( idcc>=46 && idcc<=54 )
    {
      ism = idcc-46+7;
      if( ism>9 ) ism-=9;
    }
  else
    abort();
  return ism;
}

TGraph*
MEEEGeom::getGraphBoundary(  int type, int num, int iz, int xside )
{
  std::list< std::pair< float, float > > l;
  getBoundary( l, type, num, iz, xside );
  int n = l.size();
  if( n==0 ) return 0;
  
  // GHM : changed to comply to CMSSW compilator options
  float ix[1000];
  float iy[1000];
  assert( n<1000 );

  int ii=0;
  std::list< std::pair< float, float > >::const_iterator l_it;      
  for( l_it=l.begin(); l_it!=l.end(); l_it++ )
    {
      //      std::cout << "[" << l_it->first << "," << l_it->second << "]" << std::endl;
      ix[ii] = l_it->first;
      iy[ii] = l_it->second;
      ii++;
    }
  assert( ii==n );
  return new TGraph( n, ix, iy );
}

void
MEEEGeom::getBoundary(  std::list< std::pair< float, float > >& l, int type, int num, int iz, int xside )
{
  // for each iy, get first and last ix for <whatever>
  std::multimap< int, std::pair< int, int > > map_;

  int iymin=1;
  int iymax=100;
  int ixmin=1;
  int ixmax=100;
  if( xside==1 ) 
    {
      ixmin=1; ixmax=50;
    } 
  if( xside==2 ) 
    {
      ixmin=51; ixmax=100;
    } 

  for( int iy=iymin; iy<=iymax; iy++ )
    {
      bool in=false;
      int firstix(0);
      int lastix(0);
      for( int ix=ixmin; ix<=ixmax; ix++ )
	{
	  
	  int icr = crystal( ix, iy );
	  bool ok = icr>0;

	  int iX = (ix-1)/5+1;
	  int iY = (iy-1)/5+1;

	  int num_(0);
	  switch( type )
	    {
	    case iDee:
	      num_ = dee( iX, iY, iz );   break;
	    case iQuadrant:
	      num_ = quadrant( iX, iY );  break;
	    case iSector: 
	      num_ = sector( iX, iY );    break;
	    case iLMRegion:
	      num_ = lmr( iX, iY, iz );   break;
	    case iLMModule:
	      num_ = lmmod( iX, iY );     break;
	    case iDCC:
	      num_ = dcc( iX, iY, iz );   break;
	    case iSuperCrystal:
	      num_ = sc( iX, iY );        break;
	    case iCrystal:
	      num_ = crystal( ix, iy );   break;
	    default:
	      abort();
	    };
	  ok &= num_==num;

	  if( !in && !ok ) continue;
	  if(  in &&  ok ) 
	    {
	      lastix = ix;
	      continue;
	    }
	  if( !in &&  ok )
	    {
	      in = true;
	      firstix = ix;
	    }
	  else if(  in && !ok )
	    {
	      in = false;
	      map_.insert( std::pair< int, std::pair<int, int> >( iy, std::pair<int,int>( firstix, lastix ) ) );
	    }
	}
      if( in ) map_.insert( std::pair< int, std::pair<int, int> >( iy, std::pair<int,int>( firstix, lastix ) ) );
	
    }

  // clean the list
  l.clear();

  std::multimap< int, std::pair< int, int > >::const_iterator it;
  std::multimap< int, std::pair< int, int > >::const_iterator lastelement;
  std::list< std::pair< float, float > > rightl;
  for( int iy=1; iy<=100; iy++ )
    {
      it = map_.find(iy);
      if( it==map_.end() ) continue;
      int n_ =  map_.count( iy );
      //      std::cout << "n[" << iy << "]=" << n_ << std::endl;
      assert( n_==1 );  // fixme !
      
      lastelement = map_.upper_bound(iy);
      for( ; it!=lastelement; ++it )
	{
	  std::pair<float,float> p_ = it->second;
	  l.push_back(   std::pair<float,float>(   p_.first-0.5, iy-0.5 ) );
	  l.push_back(   std::pair<float,float>(   p_.first-0.5, iy+0.5 )   );
	  rightl.push_back( std::pair<float,float>(   p_.second+0.5, iy-0.5 )   );
	  rightl.push_back( std::pair<float,float>(   p_.second+0.5, iy+0.5 )     );
	  
	}
    }
  l.unique();
  rightl.unique();
  rightl.reverse();

  std::list< std::pair< float, float > >::const_iterator rightl_it;  
  for( rightl_it=rightl.begin(); rightl_it!=rightl.end(); rightl_it++ )
    {
      l.push_back( std::pair<float,float>( rightl_it->first, rightl_it->second ) );
    }
  l.push_back( *l.begin() );

}

int
MEEEGeom::deeFromMem( int imem )
{
  int imem_ = imem%600;
  int dee_(0);
  if( imem_==50 || imem_==51 ) dee_=1;
  else if( imem_==47 || imem_==46 ) dee_=2;
  else if( imem_==1  || imem_==2  ) dee_=3;
  else if( imem_==5  || imem_==6  ) dee_=4;
  else abort();
  return dee_;
}

std::pair< int, int > 
MEEEGeom::pn( int dee, int ilmmod )
{
  // table below
  // convention to be consistent with Marc's numbering
  // PNA = 100 + pn + 1
  // PNB = 200 + pn + 1

  // LMR=73 EE+7(646/0)
  // -- module 9     PNA=(647)106    PNB=(646)201
  // -- module 10    PNA=(647)103    PNB=(646)208
  // -- module 11    PNA=(647)103    PNB=(646)203
  // -- module 12    PNA=(647)101    PNB=(646)206
  // -- module 13    PNA=(647)104    PNB=(646)209
  // LMR=74 EE+8(647/0)
  // -- module 5     PNA=(647)105    PNB=(646)210
  // -- module 6     PNA=(647)110    PNB=(646)205
  // -- module 7     PNA=(647)107    PNB=(646)202
  // -- module 8     PNA=(647)105    PNB=(646)210
  // LMR=75 EE+9(648/0)
  // -- module 1     PNA=(647)110    PNB=(646)205
  // -- module 2     PNA=(647)107    PNB=(646)202
  // -- module 3     PNA=(647)105    PNB=(646)210
  // -- module 4     PNA=(647)105    PNB=(646)210
  // LMR=76 EE+1(649/0)
  // -- module 1     PNA=(650)102    PNB=(651)206
  // -- module 2     PNA=(650)102    PNB=(651)207
  // -- module 3     PNA=(650)103    PNB=(651)208
  // -- module 4     PNA=(650)104    PNB=(651)209
  // LMR=77 EE+2(650/0)
  // -- module 5     PNA=(650)103    PNB=(651)208
  // -- module 6     PNA=(650)102    PNB=(651)206
  // -- module 7     PNA=(650)102    PNB=(651)207
  // -- module 8     PNA=(650)104    PNB=(651)209
  // LMR=78 EE+3(651/0)
  // -- module 9     PNA=(650)106    PNB=(651)201
  // -- module 10    PNA=(650)107    PNB=(651)202
  // -- module 11    PNA=(650)108    PNB=(651)203
  // -- module 12    PNA=(650)109    PNB=(651)204
  // -- module 13    PNA=(650)110    PNB=(651)205
  // LMR=79 EE+4(652/0)
  // -- module 14    PNA=(650)108    PNB=(651)203
  // -- module 15    PNA=(650)106    PNB=(651)201
  // -- module 16    PNA=(650)107    PNB=(651)202
  // -- module 17    PNA=(650)110    PNB=(651)205
  // LMR=80 EE+5(653/0)
  // -- module 18    PNA=(650)105    PNB=(651)210
  // -- module 19    PNA=(650)109    PNB=(651)204
  // LMR=81 EE+5(653/1)
  // -- module 18    PNA=(647)101    PNB=(646)206
  // -- module 19    PNA=(647)101    PNB=(646)206
  // LMR=82 EE+6(654/0)
  // -- module 14    PNA=(647)108    PNB=(646)203
  // -- module 15    PNA=(647)106    PNB=(646)201
  // -- module 16    PNA=(647)103    PNB=(646)208
  // -- module 17    PNA=(647)104    PNB=(646)209
  // LMR=83 EE-7(601/0)
  // -- module 9     PNA=(601)108    PNB=(602)203
  // -- module 10    PNA=(601)105    PNB=(602)210
  // -- module 11    PNA=(601)106    PNB=(602)201
  // -- module 12    PNA=(601)110    PNB=(602)205
  // -- module 13    PNA=(601)110    PNB=(602)205
  // LMR=84 EE-8(602/0)
  // -- module 5     PNA=(601)103    PNB=(602)208
  // -- module 6     PNA=(601)101    PNB=(602)206
  // -- module 7     PNA=(601)101    PNB=(602)206
  // -- module 8     PNA=(601)103    PNB=(602)209
  // LMR=85 EE-9(603/0)
  // -- module 1     PNA=(601)101    PNB=(602)206
  // -- module 2     PNA=(601)101    PNB=(602)206
  // -- module 3     PNA=(601)103    PNB=(602)208
  // -- module 4     PNA=(601)103    PNB=(602)209
  // LMR=86 EE-1(604/0)
  // -- module 1     PNA=(605)105    PNB=(606)210
  // -- module 2     PNA=(605)102    PNB=(606)207
  // -- module 3     PNA=(605)102    PNB=(606)207
  // -- module 4     PNA=(605)110    PNB=(606)205
  // LMR=87 EE-2(605/0)
  // -- module 5     PNA=(605)105    PNB=(606)210
  // -- module 6     PNA=(605)105    PNB=(606)210
  // -- module 7     PNA=(605)102    PNB=(606)207
  // -- module 8     PNA=(605)110    PNB=(606)205
  // LMR=88 EE-3(606/0)
  // -- module 9     PNA=(605)101    PNB=(606)206
  // -- module 10    PNA=(605)108    PNB=(606)203
  // -- module 11    PNA=(605)103    PNB=(606)208
  // -- module 12    PNA=(605)106    PNB=(606)201
  // -- module 13    PNA=(605)109    PNB=(606)204
  // LMR=89 EE-4(607/0)
  // -- module 14    PNA=(605)103    PNB=(606)208
  // -- module 15    PNA=(605)101    PNB=(606)206
  // -- module 16    PNA=(605)108    PNB=(606)203
  // -- module 17    PNA=(605)109    PNB=(606)204
  // LMR=90 EE-5(608/0)
  // -- module 18    PNA=(605)106    PNB=(606)201
  // -- module 19    PNA=(605)106    PNB=(606)201
  // LMR=91 EE-5(608/1)
  // -- module 18    PNA=(601)107    PNB=(602)202
  // -- module 19    PNA=(601)110    PNB=(602)205
  // LMR=92 EE-6(609/0)
  // -- module 14    PNA=(601)106    PNB=(602)201
  // -- module 15    PNA=(601)108    PNB=(602)203
  // -- module 16    PNA=(601)105    PNB=(602)210
  // -- module 17    PNA=(601)105    PNB=(602)210

  if( ilmmod==20 ) ilmmod=18;
  if( ilmmod==21 ) ilmmod=19;

  std::pair<int,int> pns(0,0);

  if( pnTheory )
    {
      switch( ilmmod )
	{
	case   1: pns.first=0; pns.second=5; break;
	case   2: pns.first=1; pns.second=6; break;
	case   3: pns.first=2; pns.second=7; break;
	case   4: pns.first=3; pns.second=8; break;
	case   5: pns.first=2; pns.second=7; break;
	case   6: pns.first=0; pns.second=5; break;
	case   7: pns.first=1; pns.second=6; break;
	case   8: pns.first=3; pns.second=8; break;
	case   9: pns.first=5; pns.second=0; break;
	case  10: pns.first=6; pns.second=1; break;
	case  11: pns.first=7; pns.second=2; break;
	case  12: pns.first=8; pns.second=3; break;
	case  13: pns.first=9; pns.second=4; break;
	case  14: pns.first=7; pns.second=2; break;
	case  15: pns.first=5; pns.second=0; break;
	case  16: pns.first=6; pns.second=1; break;
	case  17: pns.first=9; pns.second=4; break; 
	case  18: pns.first=4; pns.second=9; break;
	case  19: pns.first=8; pns.second=3; break;
	default:
	  abort();
	};
    }
  else
    {
      // theoretical ~ dee 1
      if( dee==1 )
	{
	  switch( ilmmod )
	    {
	      //	case   1: pns.first=0; pns.second=5; break;
	    case   1: pns.first=1; pns.second=5; break; // missing PNA
	    case   2: pns.first=1; pns.second=6; break;
	    case   3: pns.first=2; pns.second=7; break;
	    case   4: pns.first=3; pns.second=8; break;
	    case   5: pns.first=2; pns.second=7; break;
	      //	case   6: pns.first=0; pns.second=5; break;
	    case   6: pns.first=1; pns.second=5; break; // missing PNA
	    case   7: pns.first=1; pns.second=6; break;
	    case   8: pns.first=3; pns.second=8; break;
	    case   9: pns.first=5; pns.second=0; break;
	    case  10: pns.first=6; pns.second=1; break;
	    case  11: pns.first=7; pns.second=2; break;
	    case  12: pns.first=8; pns.second=3; break;
	    case  13: pns.first=9; pns.second=4; break;
	    case  14: pns.first=7; pns.second=2; break;
	    case  15: pns.first=5; pns.second=0; break;
	    case  16: pns.first=6; pns.second=1; break;
	    case  17: pns.first=9; pns.second=4; break; 
	    case  18: pns.first=4; pns.second=9; break;
	    case  19: pns.first=8; pns.second=3; break;
	    default:
	      abort();
	    };
	}
      else if( dee==2 )
	{
	  switch( ilmmod )
	    {
	    case   1: pns.first=9; pns.second=4; break;
	    case   2: pns.first=6; pns.second=1; break;
	      //	case   3: pns.first=2; pns.second=7; break;
	    case   3: pns.first=4; pns.second=9; break;  // missing PNA & PNB
	    case   4: pns.first=4; pns.second=9; break;
	      //	case   5: pns.first=2; pns.second=7; break;
	    case   5: pns.first=4; pns.second=9; break;  // missing PNA & PNB
	    case   6: pns.first=9; pns.second=4; break;
	    case   7: pns.first=6; pns.second=1; break;
	    case   8: pns.first=4; pns.second=9; break;
	    case   9: pns.first=5; pns.second=0; break;
	    case  10: pns.first=2; pns.second=7; break;
	      //	case  11: pns.first=7; pns2second=2; break;
	    case  11: pns.first=2; pns.second=2; break;  // PNA - fibre cassee
	    case  12: pns.first=0; pns.second=5; break;
	    case  13: pns.first=3; pns.second=8; break;
	    case  14: pns.first=7; pns.second=2; break;
	    case  15: pns.first=5; pns.second=0; break;
	    case  16: pns.first=2; pns.second=7; break;
	    case  17: pns.first=3; pns.second=8; break; 
	      //	case  18: pns.first=4; pns.second=9; break;
	    case  18: pns.first=0; pns.second=5; break; // missing PNA & PNB
	    case  19: pns.first=0; pns.second=5; break;
	    default:
	      abort();
	    };
	}
      else if( dee==3 )
	{
	  switch( ilmmod )
	    {
	    case   1: pns.first=0; pns.second=5; break;
	      //	case   2: pns.first=0; pns.second=1; break;
	    case   2: pns.first=0; pns.second=5; break;  // missing PNA & PNB
	    case   3: pns.first=2; pns.second=7; break;
	      //	case   4: pns.first=4; pns.second=9; break;
	    case   4: pns.first=2; pns.second=8; break; // missing PNA
	    case   5: pns.first=2; pns.second=7; break;
	    case   6: pns.first=0; pns.second=5; break;
	      //	case   7: pns.first=6; pns.second=1; break;
	    case   7: pns.first=0; pns.second=5; break;  // missing PNA & PNB 
	      //	case   8: pns.first=4; pns.second=9; break;
	    case   8: pns.first=2; pns.second=8; break;  // missing PNA
	    case   9: pns.first=7; pns.second=2; break;
	    case  10: pns.first=4; pns.second=9; break;
	    case  11: pns.first=5; pns.second=0; break;
	    case  12: pns.first=9; pns.second=4; break;
	      //	case  13: pns.first=3; pns.second=8; break;
	    case  13: pns.first=9; pns.second=4; break;  // missing PNA & PNB 
	    case  14: pns.first=5; pns.second=0; break;
	    case  15: pns.first=7; pns.second=2; break;
	    case  16: pns.first=4; pns.second=9; break;
	      //	case  17: pns.first=3; pns.second=8; break; 
	    case  17: pns.first=4; pns.second=9; break;  // missing PNA & PNB 
	    case  18: pns.first=6; pns.second=1; break;
	    case  19: pns.first=9; pns.second=4; break;
	    default:
	      abort();
	    };
	}
      else if( dee==4 )
	{
	  switch( ilmmod )
	    {
	    case   1: pns.first=4; pns.second=9; break;
	    case   2: pns.first=1; pns.second=6; break;
	      //	case   3: pns.first=6; pns.second=1; break;
	    case   3: pns.first=1; pns.second=6; break;  // missing PNA & PNB
	    case   4: pns.first=9; pns.second=4; break;
	      //	case   5: pns.first=4; pns.second=9; break;
	    case   5: pns.first=4; pns.second=9; break;  // missing PNA & PNB
	    case   6: pns.first=4; pns.second=9; break;
	    case   7: pns.first=1; pns.second=6; break;
	    case   8: pns.first=9; pns.second=4; break;
	    case   9: pns.first=0; pns.second=5; break;
	    case  10: pns.first=7; pns.second=2; break;
	    case  11: pns.first=2; pns.second=7; break;
	    case  12: pns.first=5; pns.second=0; break;
	    case  13: pns.first=8; pns.second=3; break;
	    case  14: pns.first=2; pns.second=7; break;
	    case  15: pns.first=0; pns.second=5; break;
	    case  16: pns.first=7; pns.second=2; break;
	    case  17: pns.first=8; pns.second=3; break; 
	      //	case  18: pns.first=9; pns.second=4; break;
	    case  18: pns.first=5; pns.second=0; break; // missing PNA & PNB
	    case  19: pns.first=5; pns.second=0; break;
	    default:
	      abort();
	    };
	}
    }  
  return pns;
}

int
MEEEGeom::dee( int ilmr )
{ 
  int dee_(0);
  int i_[7] = { 73 , 76 , 81 , 83 , 86 , 91 , 93 };
  int d_[6] = {    2 ,  1 ,  2 ,  3 ,  4 ,  3    };  
  for( int ii=0; ii<6; ii ++ )
  {
    if( ilmr>=i_[ii] && ilmr <i_[ii+1] ) 
      {
	dee_=d_[ii];
	break;
      }
  }
  if( dee_==0 )
    {
      std::cout << "ilmr=" << ilmr << std::endl;
    }
  assert( dee_!=0 );
  return dee_;
}

std::pair< int, int >
MEEEGeom::memFromLmr( int ilmr )
{
  std::pair< int, int > out_;
  int dee_ = dee( ilmr );
  if( dee_==1 )    // EE+F
    {
      out_.first  = 50;
      out_.second = 51;
    }
  else if( dee_==2 )  // EE+N
    {
      out_.first  = 47;  // JM: warning -- inverted !!
      out_.second = 46;
    }
  else if( dee_==3 )  // EE-N
    {
      out_.first   = 1;
      out_.second  = 2;
    }
  else if( dee_==4 )
    {
      out_.first   = 5;
      out_.second  = 6;
    }
  return out_;
}

bool
MEEEGeom::near( int ilmr )
{
  int idee  = dee( ilmr );
  return ( idee==2 || idee==3 );  
}

int 
MEEEGeom::side( SuperCrysCoord iX, SuperCrysCoord iY, int iz )
{
  int side = 0;
  int ilmr = lmr( iX, iY, iz );
  if ( ilmr == 81 || ilmr == 91 ) side=1;

  return side;
}
std::vector<int>
MEEEGeom::lmmodFromLmr( int ilmr )
{
 
  std::vector<int> vec;
 
  std::pair<int, int> dccAndSide_ = ME::dccAndSide( ilmr );
  int idcc  = dccAndSide_.first;
  int iside = dccAndSide_.second;
  bool near_ = near(ilmr);
  int ism   = smFromDcc( idcc );
  if( ism>9 ) ism-=9;
  assert( iside==0 || ism==5 );
  if ( ism==5 || (ism<5 && !near_) || (ism>5 && near_) )
    {}
  else
    {
      std::cout << "ism/near " <<  ism << "/" << near_ << std::endl;
    }

  if( ism==1 || ism==9 )
    {
      vec.push_back(1);
      vec.push_back(2);
      vec.push_back(3);
      vec.push_back(4);
    }
  else if( ism==2 || ism==8 )
    {
      vec.push_back(5);
      vec.push_back(6);
      vec.push_back(7);
      vec.push_back(8);
    }
  else if( ism==3 || ism==7 )
    {
      vec.push_back(9);
      vec.push_back(10);
      vec.push_back(11);
      vec.push_back(12);
      vec.push_back(13);
    }
  else if( ism==4 || ism==6 )
    {
      vec.push_back(14);
      vec.push_back(15);
      vec.push_back(16);
      vec.push_back(17);
    }
  else
    {
      assert( ism==5 );
      vec.push_back(18);
      vec.push_back(19);
    }
  return vec;
}

int
MEEEGeom::apdRefTower( int ilmr, int ilmmod )
{
  int ilmr10=ilmr%10;
  int tower=0;

  if( ilmr10==3 ){      
    switch( ilmmod ){
    case   9:  tower=77; break;
    case   10: tower=55; break;
    case   11: tower=37; break;
    case   12: tower=310; break;
    case   13: tower=294; break;
    default:
      abort();
    }
  }else if( ilmr10==4 ){
    switch( ilmmod ){
    case   5: tower=52; break;
    case   6: tower=41; break;
    case   7: tower=18; break;
    case   8: tower=65; break;
    default:
      abort();
    }
  }else if( ilmr10==5 ){
    
    switch( ilmmod ){
    case   1: tower=45; break;
    case   2: tower=53; break; 
    case   3: tower=42; break;
    case   4: tower=32; break; 
    default:
      abort();
    }
  }else if( ilmr10==6 ){ 
    switch( ilmmod ){
    case   1: tower=124; break;
    case   2: tower=132; break;
    case   3: tower=121; break;
    case   4: tower=111; break;
    default:
      abort();
    }
  }else if( ilmr10==7 ){ 
    switch( ilmmod ){
    case   5: tower=147; break;
    case   6: tower=135; break;
    case   7: tower=117; break;
    case   8: tower=158; break; 
    default:
      abort();
    }
    
  }else if( ilmr10==8 ){ 
    switch( ilmmod ){
    case   9: tower=156; break;
    case  10: tower=214; break;
    case  11: tower=197; break;
    case  12: tower=237; break; 
    case  13: tower=224; break; 
    default:
      abort();
    }  
  }else if( ilmr10==9 ){ 
    switch( ilmmod ){
    case  14: tower=234; break; 
    case  15: tower=220; break;
    case  16: tower=212; break; 
    case  17: tower=189; break;
    default:
      abort();
    } 
  }else if( ilmr10==0 ){ 
    switch( ilmmod ){
    case  18: tower=185; break; 
    case  19: tower=172; break;
    default:
      abort();
    }
  }else if( ilmr10==1 ){ 
    switch( ilmmod ){
    case  18: tower=264; break; 
    case  19: tower=251; break;
    default:
      abort();
    }
  }
  return tower; 
}

std::vector< int>
MEEEGeom::apdRefChannels( int ilmmod )
{

  std::vector< int> vec;
  switch( ilmmod )
    {
    case   1: vec.push_back(0); vec.push_back(1); break;
    case   2: vec.push_back(0); vec.push_back(1); break;
    case   3: vec.push_back(0); vec.push_back(1); break;
    case   4: vec.push_back(0); vec.push_back(1); break;
    case   5: vec.push_back(0); vec.push_back(1); break;
    case   6: vec.push_back(0); vec.push_back(1); break;
    case   7: vec.push_back(0); vec.push_back(1); break;
    case   8: vec.push_back(0); vec.push_back(1); break;
    case   9: vec.push_back(0); vec.push_back(1); break;
    case  10: vec.push_back(0); vec.push_back(1); break;
    case  11: vec.push_back(0); vec.push_back(1); break;
    case  12: vec.push_back(0); vec.push_back(1); break;
    case  13: vec.push_back(0); vec.push_back(1); break;
    case  14: vec.push_back(0); vec.push_back(1); break;
    case  15: vec.push_back(0); vec.push_back(1); break;
    case  16: vec.push_back(0); vec.push_back(1); break;
    case  17: vec.push_back(0); vec.push_back(1); break;
    case  18: vec.push_back(0); vec.push_back(1); break;
    case  19: vec.push_back(0); vec.push_back(1); break;
    case  20: vec.push_back(0); vec.push_back(1); break;
    case  21: vec.push_back(0); vec.push_back(1); break;

    default:
      abort();
    }
  return vec;
}

