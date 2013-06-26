//-------------------------------------------------
//
//   Class: DTBtiChip
//
//   Description: Implementation of DTBtiChip 
//                trigger algorithm
//                (Equations' implementation)
//
//
//   Author List:
//   C. Grandi
//   Modifications: 
//   22/X/02 S. Vanini: redundant patterns added 
//   9/XII/02 SV : equation in manual form
//   13/I/2003 SV equations in manual order  
//   22/VI/04 SV: last trigger code update
//--------------------------------------------------

//#include "Utilities/Configuration/interface/Architecture.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTBti/interface/DTBtiChip.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "L1Trigger/DTBti/interface/DTBtiHit.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>

using namespace std;

void
DTBtiChip::computeSums(){
  //
  // compute all sums and diffs
  // nothing more than a table...
  // numbering convention here is the same as the fortran version:
  //   sum number in the range [1,25]
  //   cell numbers in the range [1,9]
  //     --> sum(int,int,int) decreases the indices by 1
  // sum (sum_number, first_cell, second_cell)
  //
  
  if(config()->debug()>3){
    cout << "DTBtiChip::computeSums called" << endl; 
  }

  sum( 1,2,1);
  sum( 2,3,1);
  sum( 3,4,1);
  sum( 4,6,1);
  sum( 5,8,1);
  sum( 6,3,2);
  sum( 7,4,2);
  sum( 8,5,2);
  sum( 9,4,3);
  sum(10,5,3);
  sum(11,6,3);
  sum(12,8,3);
  sum(13,5,4);
  sum(14,6,4);
  sum(15,7,4);
  sum(16,9,4);
  sum(17,6,5);
  sum(18,7,5);
  sum(19,8,5);
  sum(20,7,6);
  sum(21,8,6);
  sum(22,9,6);
  sum(23,8,7);
  sum(24,9,7);
  sum(25,9,8);
}

void
DTBtiChip::sum(const int s, const int a, const int b) {
  //
  // fill the sums and difs arrays
  // a and b are the cell numbers (in the range [1,9])
  // s is the sum number (in the range [1,25])
  //

  if( _thisStepUsedHit[a-1]!=0 && _thisStepUsedHit[b-1]!=0 ){
    _sums[s-1] = (float)(_thisStepUsedHit[a-1]->jtrig() +
                         _thisStepUsedHit[b-1]->jtrig()  );
    _difs[s-1] = (float)(_thisStepUsedHit[a-1]->jtrig() -
                         _thisStepUsedHit[b-1]->jtrig()  );
  } else {
    _sums[s-1] = 1000;
    _difs[s-1] = 1000;
  }

}


void
DTBtiChip::reSumSet(){

  reSumAr[2][ 2 +2]=0;
  reSumAr[2][ 1 +2]=0;
  reSumAr[2][ 0 +2]=0;
  reSumAr[2][-1 +2]=-1;
  reSumAr[2][-2 +2]=-1;

  reSumAr[1][ 2 +2]=1;
  reSumAr[1][ 1 +2]=1;
  reSumAr[1][ 0 +2]=0;
  reSumAr[1][-1 +2]=0;
  reSumAr[1][-2 +2]=0;

  reSumAr[0][ 2 +2]=1;
  reSumAr[0][ 1 +2]=0;
  reSumAr[0][ 0 +2]=0;
  reSumAr[0][-1 +2]=0;
  reSumAr[0][-2 +2]=-1;

  reSumAr23[2][ 2 +2]=1;
  reSumAr23[2][ 1 +2]=1;
  reSumAr23[2][ 0 +2]=1;
  reSumAr23[2][-1 +2]=0;
  reSumAr23[2][-2 +2]=0;

  reSumAr23[1][ 2 +2]=1;
  reSumAr23[1][ 1 +2]=1;
  reSumAr23[1][ 0 +2]=0;
  reSumAr23[1][-1 +2]=0;
  reSumAr23[1][-2 +2]=0;

  reSumAr23[0][ 2 +2]=1;
  reSumAr23[0][ 1 +2]=0;
  reSumAr23[0][ 0 +2]=0;
  reSumAr23[0][-1 +2]=0;
  reSumAr23[0][-2 +2]=-1;
 }



void 
DTBtiChip::computeEqs(){
  //
  // Compute all K and X equations of DTBtiChip algorithm
  // NB now Keq=Kman

  float K0 = config()->ST();
  //cout <<"K0="<<K0<<endl;

  //enabled patterns....
  int PTMS[32];
  for(int i=0; i<32; i++){
    PTMS[i] = config()->PTMSflag(i);
  }

    int i;
  // redundant patterns added by Sara Vanini
  i=0;  //  1324A  --> 1L3L2R4L
  if(PTMS[i] ){
  _Keq[i][0] =  _difs[1]          + 2.*K0;                  //eq. AB
  _Keq[i][1] = -_sums[5]          + 2.*K0;                  //eq. BC
  _Keq[i][2] =  _sums[6];	  	                    //eq. CD
  _Keq[i][3] = -(_sums[0]/2.)  + 2.*K0 + 0.01;              //eq. AC
  _Keq[i][4] =  (_difs[8]/2.)  +    K0 + 0.01 ;             //eq. BD
  //_Keq[i][5] =  (_difs[2]/3.) + 4.*K0/3. + 0.51;                 //eq. AD
  _Keq[i][5] =  ST43 + 
                reSum( RE43 , int(fmod(double(_difs[2]),3.)) ) + 
                int( (double(_difs[2])/3.) );                        //eq. AD

  //patt 0 always uses Xbc, because Xad could be negative...
  //when wire B is missing,  TshiftB=0
  //when hit B is gone out of shift register, Tshift=K0+1
  float _difs_p0 = _difs[5]; 

  float TshiftB = 0;
  float TshiftC = 0;
  if(_thisStepUsedHit[3-1]==0){
    if(_hits[3-1].size()==0 )
      TshiftB = 0;
    if(_hits[3-1].size()>0 && (*(_hits[3-1].begin()))->clockTime()<=-K0 )
      TshiftB = K0+1; 
  }
  else
   TshiftB = _thisStepUsedHit[3-1]->jtrig();  

  if(_thisStepUsedHit[2-1]==0){
    if(_hits[2-1].size()==0 )
      TshiftC = 0;
    if(_hits[2-1].size()>0 && (*(_hits[2-1].begin()))->clockTime()<=-K0 )
      TshiftC = K0+1; 
  }
  else
    TshiftC = _thisStepUsedHit[2-1]->jtrig();  

  _difs_p0 = (float)(TshiftB - TshiftC);

// Zotto's
  _XeqAB_patt0 = (_sums[1] - K0) / 4.;                       //eq. AB
  _Xeq[i][0]   = (_difs_p0 + K0) / 4.;                       //eq. BC
  _XeqCD_patt0 = (_difs[6] + K0) / 4.;                        //eq. CD
  _XeqAC_patt0 = -(_difs[0])   / 4.;                         //eq. AC
  _XeqBD_patt0 =  (_sums[8])   / 4.;                         //eq. BD
  _Xeq[i][1] = (_sums[2] - K0) / 4.;                         //eq. AD
  
/*
// my eq
  _XeqAB_patt0 = (_sums[1] - K0) / 4.;                       //eq. AB
  _Xeq[i][0]   = (_difs[5] + K0) / 4.;                       //eq. BC
  _XeqCD_patt0 = (_difs[6] + K0) / 4.;                        //eq. CD
  _XeqAC_patt0 = -(_difs[0])   / 4.;                         //eq. AC
  _XeqBD_patt0 =  (_sums[8])   / 4.;                         //eq. BD
  _Xeq[i][1] = (_sums[2] - K0) / 4.;                         //eq. AD
*/
   
  }
 

  i=1;  //  1324B  --> 1L3L2R4R
  if(PTMS[i] ){
  _Keq[i][0] =  _difs[1]        + 2.*K0;
  _Keq[i][1] = -(_sums[5])  + 2.*K0;
  _Keq[i][2] = -_difs[6]        + 2.*K0;
  _Keq[i][3] = -(_sums[0]/2.) + 2.*K0 + 0.01;
  _Keq[i][4] = -(_sums[8]/2.) + 2.*K0 + 0.01;
  //_Keq[i][5] = -(_sums[2]/3.)     + 2.*K0 + 0.51;
  _Keq[i][5] = ST2 +
               reSum23( 0 , int(fmod(double(-_sums[2]),3.)) ) +
               int( (double(-_sums[2])/3.) );   

  _Xeq[i][0] = ( _difs[5] + K0) / 4.;
  _Xeq[i][1] = (-_difs[2] + K0) / 4.;
  }
 
  i=2;  //  1324C  --> 1R3L2R4L
  if(PTMS[i] ){
  _Keq[i][0] =  _sums[1];
  _Keq[i][1] = -_sums[5]       + 2.*K0;
  _Keq[i][2] =  _sums[6];
  _Keq[i][3] = -(_difs[0]/2.) +   K0 + 0.01;
  _Keq[i][4] =  (_difs[8]/2.) +   K0 + 0.01;
  //_Keq[i][5] =  (_sums[2]/3.)    + 2.*K0/3. + 0.51;
  _Keq[i][5] =  ST23 + 
                reSum23( RE23 , int(fmod(double(_sums[2]),3.)) ) + 
                int( (double(_sums[2])/3.) );                        //eq. AD
/*  
cout << "ST23 =" << ST23 << endl;
cout << "RE23 =" << RE23 << endl;
cout << "fmod(double(_sums[2]),3.) =" << fmod(double(_sums[2]),3.) << endl;
cout << "reSum23 = "<< reSum23( RE23 , fmod(double(_sums[2]),3.) ) << endl;
cout << "double(_sums[2])/3.="<<double(_sums[2])/3.<< endl;
cout << "int('') = " << int( (double(_sums[2])/3.) ) << endl;
*/
  _Xeq[i][0] = (_difs[5] + K0) / 4.;
  _Xeq[i][1] = (_difs[2] + K0) / 4.;
  }
 
  i=3;  //  1324D  --> 1R3L2R4R
  if(PTMS[i] ){
  _Keq[i][0] =  _sums[1];
  _Keq[i][1] = -_sums[5]      + 2.*K0;
  _Keq[i][2] = -_difs[6]      + 2.*K0;
  _Keq[i][3] = -(_difs[0]/2.) +    K0 + 0.01;
  _Keq[i][4] = -(_sums[8]/2.) + 2.*K0 + 0.01;
  //_Keq[i][5] = -(_difs[2]/3.) + 4.*K0/3.;
  _Keq[i][5] =  ST43 + 
                reSum( RE43 , int(fmod(double(-_difs[2]),3.)) ) + 
                int( (double(-_difs[2])/3.) );                        //eq. AD


  _Xeq[i][0] = ( _difs[5] +   K0) / 4.;
  _Xeq[i][1] = (-_sums[2] + 3.*K0) / 4.;
  }
 

  i=4;  //  i = DTBtiChipEQMAP->index("1364A");  --> 1L3L6L4R
  if(PTMS[i] ){
  _Keq[i][0] =  _difs[1]        + 2.*K0;
  _Keq[i][1] =  _difs[10]       + 2.*K0;
  _Keq[i][2] = -(_sums[13]) + 2.*K0;
  _Keq[i][3] =  (_difs[3]/2.)   + 2.*K0 + 0.01;
  _Keq[i][4] = -(_sums[8]/2.)   + 2.*K0 + 0.01;
  //_Keq[i][5] = -(_sums[2]/3.)   + 2.*K0 + 0.51;
  _Keq[i][5] = ST2 +
               reSum23( 0 , int(fmod(double(-_sums[2]),3.)) ) +
               int( (double(-_sums[2])/3.) );   

  _Xeq[i][0] = ( _sums[10] + K0) / 4.;
  _Xeq[i][1] = (-_difs[2]  + K0) / 4.;
  }
 
  i=5;  //  i = DTBtiChipEQMAP->index("1364B");  --> 1R3L6L4R
  if(PTMS[i] ){
  _Keq[i][0] =  _sums[1];
  _Keq[i][1] =  _difs[10]      + 2.*K0;
  _Keq[i][2] = -_sums[13]      + 2.*K0;
  _Keq[i][3] =  (_sums[3]/2.)  +   K0 + 0.01;
  _Keq[i][4] = -(_sums[8]/2.)  + 2.*K0 + 0.01;
  //_Keq[i][5] = -(_difs[2]/3.)  + 4.*K0/3.;
  _Keq[i][5] =  ST43 + 
                reSum( RE43 , int(fmod(double(-_difs[2]),3.)) ) + 
                int( (double(-_difs[2])/3.) );                        //eq. AD

  _Xeq[i][0] = ( _sums[10]+   K0) / 4.;
  _Xeq[i][1] = (-_sums[2] + 3.*K0) / 4.;
  }

  i=6;  //  i = DTBtiChipEQMAP->index("1364C");  --> 1R3R6L4R
  if(PTMS[i] ){
  _Keq[i][0] = -_difs[1]       + 2.*K0;
  _Keq[i][1] =  _sums[10];
  _Keq[i][2] = -_sums[13]      + 2.*K0;
  _Keq[i][3] =  (_sums[3]/2.)  +    K0 + 0.01;
  _Keq[i][4] = -(_difs[8]/2.)  +    K0 + 0.01;
  //_Keq[i][5] = -(_difs[2]/3.)  + 4.*K0/3.;
  _Keq[i][5] =  ST43 + 
                reSum( RE43 , int(fmod(double(-_difs[2]),3.)) ) + 
                int( (double(-_difs[2])/3.) );                        //eq. AD

  _Xeq[i][0] = ( _difs[10]+ 3.*K0) / 4.;
  _Xeq[i][1] = (-_sums[2] + 3.*K0) / 4.;
  }
 
  i=7;  //  i = DTBtiChipEQMAP->index("1368A");  --> 1R3R6L8L
  if(PTMS[i] ){
  _Keq[i][0] = -_difs[1]        + 2.*K0;
  _Keq[i][1] =  (_sums[10]);
  _Keq[i][2] =  _difs[20]       + 2.*K0;
  _Keq[i][3] =  (_sums[3]/2.)   +    K0 + 0.01;
  _Keq[i][4] =  (_sums[11]/2.)  +    K0 + 0.01;
  //_Keq[i][5] =  (_sums[4]/3.)   + 4.*K0/3. + 0.51;
  _Keq[i][5] =  ST43 + 
                reSum( RE43 , int(fmod(double(_sums[4]),3.)) ) + 
                int( (double(_sums[4])/3.) );                        //eq. AD

  _Xeq[i][0] =  (_difs[10]+ 3.*K0) / 4.;
  _Xeq[i][1] =  (_difs[4] + 3.*K0) / 4.;
  }
 
  i=8;  //  i = DTBtiChipEQMAP->index("1368B");  --> 1R3R6R8L
  if(PTMS[i] ){
  _Keq[i][0] = -_difs[1]          + 2.*K0;
  _Keq[i][1] = -_difs[10]         + 2.*K0;
  _Keq[i][2] =  (_sums[20]);
  _Keq[i][3] = -(_difs[3]/2.)     + 2.*K0 + 0.01;
  _Keq[i][4] =  (_sums[11]/2.)    +    K0 + 0.01;
  //_Keq[i][5] =  (_sums[4]/3.)     + 4.*K0/3.;
  _Keq[i][5] =  ST43 + 
                reSum( RE43 , int(fmod(double(_sums[4]),3.)) ) + 
                int( (double(_sums[4])/3.) );                        //eq. AD


  _Xeq[i][0] = (-_sums[10]+ 5.*K0) / 4.;
  _Xeq[i][1] = ( _difs[4] + 3.*K0) / 4.;
  }

  i=9;  //  i = DTBtiChipEQMAP->index("1368C");  --> 1R3L6L8L
  if(PTMS[i] ){
  _Keq[i][0] =  (_sums[1]);
  _Keq[i][1] =  _difs[10]        + 2.*K0;
  _Keq[i][2] =  _difs[20]        + 2.*K0;
  _Keq[i][3] =  (_sums[3]/2.)    +    K0 + 0.01;
  _Keq[i][4] =  (_difs[11]/2.)   + 2.*K0 + 0.01;
  //_Keq[i][5] =  (_sums[4]/3.)    + 4.*K0/3.;
  _Keq[i][5] =  ST43 + 
                reSum( RE43 , int(fmod(double(_sums[4]),3.)) ) + 
                int( (double(_sums[4])/3.) );                        //eq. AD


  _Xeq[i][0] =  (_sums[10]+  K0) / 4.;
  _Xeq[i][1] =  (_difs[4] + 3.*K0) / 4.;
  }
  
  i=10;  //  i = DTBtiChipEQMAP->index("5324A");  --> 5L3L2R4L
  if(PTMS[i] ){
  _Keq[i][0] = -_difs[9];
  _Keq[i][1] = -_sums[5]       + 2.*K0;
  _Keq[i][2] =  _sums[6];
  _Keq[i][3] = -(_sums[7]/2.)  +    K0 + 0.01;
  _Keq[i][4] =  (_difs[8]/2.)  +    K0 + 0.01;
  //_Keq[i][5] = -(_difs[12]/3.) + 2.*K0/3.;
  _Keq[i][5] =  ST23 + 
                reSum23( RE23 , int(fmod(double(-_difs[12]),3.)) ) + 
                int( (double(-_difs[12])/3.) );                        //eq. AD

  _Xeq[i][0] =  (_difs[5] + K0) / 4.;
  _Xeq[i][1] =  (_sums[12]+ K0) / 4.;
  }

  i=11;  //  i = DTBtiChipEQMAP->index("5324B");  --> 5L3R2R4L
  if(PTMS[i] ){
  _Keq[i][0] = -_sums[9]       + 2.*K0;
  _Keq[i][1] =  _difs[5];
  _Keq[i][2] =  _sums[6];
  _Keq[i][3] = -(_sums[7]/2.)  +    K0 + 0.01;
  _Keq[i][4] =  (_sums[8]/2.   + 0.01);
  //_Keq[i][5] = -(_difs[12]/3.) + 2.*K0/3.;
  _Keq[i][5] =  ST23 + 
                reSum23( RE23 , int(fmod(double(-_difs[12]),3.)) ) + 
                int( (double(-_difs[12])/3.) );                        //eq. AD

  _Xeq[i][0] = (-_sums[5] + 3.*K0) / 4.;
  _Xeq[i][1] = ( _sums[12]+   K0) / 4.;
  }

  i=12;  //  i = DTBtiChipEQMAP->index("5324C");  --> 5R3R2R4L
  if(PTMS[i] ){
  _Keq[i][0] =  _difs[9];
  _Keq[i][1] =  _difs[5];
  _Keq[i][2] =  (_sums[6]);
  _Keq[i][3] =  (_difs[7]/2.) + 0.01;
  _Keq[i][4] =  (_sums[8]/2.)  + 0.01;
  _Keq[i][5] =  (_sums[12]/3.) + 0.51;

  _Xeq[i][0] = (-_sums[5] + 3. * K0) / 4.;
  _Xeq[i][1] = (-_difs[12]+ 3. * K0) / 4.;
  }

  i=13;  //  i = DTBtiChipEQMAP->index("5364A");  --> 5L3R6L4L
  if(PTMS[i] ){
  _Keq[i][0] = -_sums[9]        + 2.*K0;
  _Keq[i][1] =  _sums[10];
  _Keq[i][2] = -_difs[13];
  _Keq[i][3] =  (_difs[16]/2.)  +    K0 + 0.01;
  _Keq[i][4] =  (_sums[8]/2.    + 0.01);
  //_Keq[i][5] = -(_difs[12]/3.)  +  2.*K0/3.;
  _Keq[i][5] =  ST23 + 
                reSum23( RE23 , int(fmod(double(-_difs[12]),3.)) ) + 
                int( (double(-_difs[12])/3.) );                        //eq. AD

  _Xeq[i][0] =  (_difs[10]+ 3.*K0) / 4.;
  _Xeq[i][1] =  (_sums[12]+   K0) / 4.;
  }
 
  i=14;  //  i = DTBtiChipEQMAP->index("5364B");  --> 5L3R6L4R
  if(PTMS[i] ){
  _Keq[i][0] = -_sums[9]        + 2.*K0;
  _Keq[i][1] =  _sums[10];
  _Keq[i][2] = -_sums[13]       + 2.*K0;
  _Keq[i][3] =  (_difs[16]/2.)  +    K0 + 0.01;
  _Keq[i][4] = -(_difs[8]/2.)   +    K0 + 0.01;
  //_Keq[i][5] = -(_sums[12]/3.)  + 4.*K0/3.;
  _Keq[i][5] =  ST43 + 
                reSum( RE43 , int(fmod(double(-_sums[12]),3.)) ) + 
                int( (double(-_sums[12])/3.) ); 

  _Xeq[i][0] =  (_difs[10]+ 3.*K0) / 4.;
  _Xeq[i][1] =  (_difs[12]+ 3.*K0) / 4.;
  }
 
  i=15;  //  i = DTBtiChipEQMAP->index("5364C");  --> 5R3R6L4L
  if(PTMS[i] ){
  _Keq[i][0] =  _difs[9];
  _Keq[i][1] =  (_sums[10]);
  _Keq[i][2] = -_difs[13];
  _Keq[i][3] =  (_sums[16]/2.  + 0.01);
  _Keq[i][4] =  (_sums[8]/2.   + 0.01);
  _Keq[i][5] =  (_sums[12]/3.) + 0.51;

  _Xeq[i][0] = ( _difs[10]+ 3.*K0) / 4.;
  _Xeq[i][1] = (-_difs[12]+ 3.*K0) / 4.;
  }
 
  i=16;  //  i = DTBtiChipEQMAP->index("5364D");  --> 5R3R6L4R
  if(PTMS[i] ){
  _Keq[i][0] =  _difs[9];
  _Keq[i][1] =  _sums[10];
  _Keq[i][2] = -_sums[13]      + 2.*K0;
  _Keq[i][3] =  (_sums[16]/2.  + 0.01);
  _Keq[i][4] = -(_difs[8]/2.)  +    K0 + 0.01;
  //_Keq[i][5] =  (_difs[12]/3.) + 2.*K0/3.;
  _Keq[i][5] =  ST23 + 
                reSum23( RE23 , int(fmod(double(_difs[12]),3.)) ) + 
                int( (double(_difs[12])/3.) );                        //eq. AD

  _Xeq[i][0] = ( _difs[10]+ 3.*K0) / 4.;
  _Xeq[i][1] = (-_sums[12]+ 5.*K0) / 4.;
  }
 
  i=17;  //  i = DTBtiChipEQMAP->index("5368A");  --> 5L3R6L8L
  if(PTMS[i] ){
  _Keq[i][0] = -_sums[9]        + 2.*K0;
  _Keq[i][1] =  _sums[10];
  _Keq[i][2] =  _difs[20]       + 2.*K0;
  _Keq[i][3] =  (_difs[16]/2.)  +    K0 + 0.01;
  _Keq[i][4] =  (_sums[11]/2.)  +    K0 + 0.01;
  //_Keq[i][5] =  (_difs[18]/3.)  + 4.*K0/3.;
  _Keq[i][5] =  ST43 + 
                reSum( RE43 , int(fmod(double(_difs[18]),3.)) ) + 
                int( (double(_difs[18])/3.) );                        //eq. AD

  _Xeq[i][0] =  (_difs[10]+ 3.*K0) / 4.;
  _Xeq[i][1] =  (_sums[18]+ 3.*K0) / 4.;
  }
 
  i=18;  //  i = DTBtiChipEQMAP->index("5368B");  --> 5L3R6R8L
  if(PTMS[i] ){
  _Keq[i][0] = -_sums[9]        + 2.*K0;
  _Keq[i][1] = -_difs[10]       + 2.*K0;
  _Keq[i][2] =  _sums[20];
  _Keq[i][3] = -(_sums[16]/2.)  + 2.*K0 + 0.01;
  _Keq[i][4] =  (_sums[11]/2.)  +    K0 + 0.01;
  //_Keq[i][5] =  (_difs[18]/3.)  + 4.*K0/3.;
  _Keq[i][5] =  ST43 + 
                reSum( RE43 , int(fmod(double(_difs[18]),3.)) ) + 
                int( (double(_difs[18])/3.) );                        //eq. AD

  _Xeq[i][0] = (-_sums[10]+ 5.*K0) / 4.;
  _Xeq[i][1] = ( _sums[18]+ 3.*K0) / 4.;
  }
 
  i=19;  //  i = DTBtiChipEQMAP->index("5368C");  --> 5L3R6R8R
  if(PTMS[i] ){
  _Keq[i][0] = -(_sums[9])  + 2.*K0;
  _Keq[i][1] = -_difs[10]       + 2.*K0;
  _Keq[i][2] = -_difs[20]       + 2.*K0;
  _Keq[i][3] = -(_sums[16]/2.)  + 2.*K0 + 0.01;
  _Keq[i][4] = -(_difs[11]/2.)  + 2.*K0 + 0.01;
  //_Keq[i][5] = -(_sums[18]/3.)  + 2.*K0 + 0.51;
  _Keq[i][5] = ST2 +
               reSum23( 0 , int(fmod(double(-_sums[18]),3.)) ) +
               int( (double(-_sums[18])/3.) );   

  _Xeq[i][0] = (-_sums[10]+ 5.*K0) / 4.;
  _Xeq[i][1] = (-_difs[18]+ 5.*K0) / 4.;
  }
 
  i=20;  //  i = DTBtiChipEQMAP->index("5764A");  --> 5R7L6L4R
  if(PTMS[i] ){
  _Keq[i][0] =  _sums[17];
  _Keq[i][1] = -_difs[19];
  _Keq[i][2] = -_sums[13]       + 2.*K0;
  _Keq[i][3] =  (_sums[16]/2.   + 0.01);
  _Keq[i][4] = -(_sums[14]/2.)  +    K0 + 0.01;
  //_Keq[i][5] =  (_difs[12]/3.)  + 2.*K0/3.;
  _Keq[i][5] =  ST23 + 
                reSum23( RE23 , int(fmod(double(_difs[12]),3.)) ) + 
                int( (double(_difs[12])/3.) );                        //eq. AD

  _Xeq[i][0] = ( _sums[19]+ 3.*K0) / 4.;
  _Xeq[i][1] = (-_sums[12]+ 5.*K0) / 4.;
  }
 
  i=21;  //  i = DTBtiChipEQMAP->index("5764B");  --> 5R7L6R4R
  if(PTMS[i] ){
  _Keq[i][0] =  _sums[17];
  _Keq[i][1] = -_sums[19]      + 2.*K0;
  _Keq[i][2] =  _difs[13];
  _Keq[i][3] = -(_difs[16]/2.) +    K0 + 0.01;
  _Keq[i][4] = -(_sums[14]/2.) +    K0 + 0.01;
  //_Keq[i][5] =  (_difs[12]/3.) + 2.*K0/3.;
  _Keq[i][5] =  ST23 + 
                reSum23( RE23 , int(fmod(double(_difs[12]),3.)) ) + 
                int( (double(_difs[12])/3.) );                        //eq. AD

  _Xeq[i][0] = ( _difs[19]+ 5.*K0) / 4.;
  _Xeq[i][1] = (-_sums[12]+ 5.*K0) / 4.;
  }
 
  i=22;  //  i = DTBtiChipEQMAP->index("5764C");  --> 5R7L6L4L
  if(PTMS[i] ){
  _Keq[i][0] =  (_sums[17]);
  _Keq[i][1] = -_difs[19];
  _Keq[i][2] = -_difs[13];
  _Keq[i][3] =  (_sums[16]/2.  + 0.01);
  _Keq[i][4] = -(_difs[14]/2.) + 0.01;
  _Keq[i][5] =  (_sums[12]/3.) + 0.51;

  _Xeq[i][0] = ( _sums[19]+ 3.*K0) / 4.;
  _Xeq[i][1] = (-_difs[12]+ 3.*K0) / 4.;
  }
 
  i=23;  //  i = DTBtiChipEQMAP->index("9764A");  --> 9L7L6L4R
  if(PTMS[i] ){
  _Keq[i][0] = -_difs[23];
  _Keq[i][1] = -_difs[19];
  _Keq[i][2] = -(_sums[13]) + 2.*K0;
  _Keq[i][3] = -(_difs[21]/2.) + 0.01;
  _Keq[i][4] = -(_sums[14]/2.)  +    K0 + 0.01;
  //_Keq[i][5] = -(_sums[15]/3.)  + 2.*K0/3.;
  _Keq[i][5] =  ST23 + 
                reSum23( RE23 , int(fmod(double(-_sums[15]),3.)) ) + 
                int( (double(-_sums[15])/3.) );                        //eq. AD

  _Xeq[i][0] =  (_sums[19]+ 3.*K0) / 4.;
  _Xeq[i][1] =  (_difs[15]+ 5.*K0) / 4.;
  }
 
  i=24;  //  i = DTBtiChipEQMAP->index("9764B");  --> 9L7L6R4R
  if(PTMS[i] ){
  _Keq[i][0] = -_difs[23];
  _Keq[i][1] = -(_sums[19])   + 2.*K0;
  _Keq[i][2] =  _difs[13];
  _Keq[i][3] = -(_sums[21]/2.)    +    K0 + 0.01;
  _Keq[i][4] = -(_sums[14]/2.)    +    K0 + 0.01;
  //_Keq[i][5] = -(_sums[15]/3.)    + 2.*K0/3.;
  _Keq[i][5] =  ST23 + 
                reSum23( RE23 , int(fmod(double(-_sums[15]),3.)) ) + 
                int( (double(-_sums[15])/3.) );                        //eq. AD

  _Xeq[i][0] =  (_difs[19]+ 5.*K0) / 4.;
  _Xeq[i][1] =  (_difs[15]+ 5.*K0) / 4.;
  }
 
  i=25;  //  i = DTBtiChipEQMAP->index("9764C");  --> 9L7R6R4R
  if(PTMS[i] ){
  _Keq[i][0] = -(_sums[23])   + 2.*K0;
  _Keq[i][1] =  _difs[19];
  _Keq[i][2] =  _difs[13];
  _Keq[i][3] = -(_sums[21]/2.)    +   K0 + 0.01;
  _Keq[i][4] =  (_difs[14]/2.) + 0.01;
  //_Keq[i][5] = -(_sums[15]/3.)    + 2.*K0/3.;
  _Keq[i][5] =  ST23 + 
                reSum23( RE23 , int(fmod(double(-_sums[15]),3.)) ) + 
                int( (double(-_sums[15])/3.) );                        //eq. AD

  _Xeq[i][0] = (-_sums[19]+ 7.*K0) / 4.;
  _Xeq[i][1] = ( _difs[15]+ 5.*K0) / 4.;
  }

  i=26;  //  int i = DTBtiChipEQMAP->index("5768A") --> 5L7L6R8L
  if(PTMS[i] ){
  _Keq[i][0] =  _difs[17]       + 2.*K0;
  _Keq[i][1] = -_sums[19]       + 2.*K0;
  _Keq[i][2] =  _sums[20];
  _Keq[i][3] = -(_sums[16]/2.)  + 2.*K0 + 0.01;
  _Keq[i][4] =  (_difs[22]/2.)  +    K0 + 0.01;
  //_Keq[i][5] =  (_difs[18]/3.)  + 4.*K0/3.;
  _Keq[i][5] =  ST43 + 
                reSum( RE43 , int(fmod(double(_difs[18]),3.)) ) + 
                int( (double(_difs[18])/3.) );                        //eq. AD

  _Xeq[i][0] =  (_difs[19] + 5.*K0) / 4.;
  _Xeq[i][1] =  (_sums[18] + 3.*K0) / 4.;
  }
 
  i=27;  //  i = DTBtiChipEQMAP->index("5768B");  --> 5L7L6R8R
  if(PTMS[i] ){
  _Keq[i][0] =  _difs[17]       + 2.*K0;
  _Keq[i][1] = -(_sums[19]) + 2.*K0;
  _Keq[i][2] = -_difs[20]       + 2.*K0;
  _Keq[i][3] = -(_sums[16]/2.)  + 2.*K0 + 0.01;
  _Keq[i][4] = -(_sums[22]/2.)  + 2.*K0 + 0.01;
  //_Keq[i][5] = -(_sums[18]/3.)  + 2.*K0 + 0.51;
  _Keq[i][5] = ST2 +
               reSum23( 0 , int(fmod(double(-_sums[18]),3.)) ) +
               int( (double(-_sums[18])/3.) );   
  
  _Xeq[i][0] = ( _difs[19] + 5.*K0) / 4.;
  _Xeq[i][1] = (-_difs[18] + 5.*K0) / 4.;
  }
 
  i=28;  //  i = DTBtiChipEQMAP->index("5768C");  --> 5R7L6R8L
  if(PTMS[i] ){
  _Keq[i][0] =  _sums[17];
  _Keq[i][1] = -_sums[19]       + 2.*K0;
  _Keq[i][2] =  _sums[20];
  _Keq[i][3] = -(_difs[16]/2.)  +   K0 + 0.01;
  _Keq[i][4] =  (_difs[22]/2.)  +   K0 + 0.01;
  //_Keq[i][5] =  (_sums[18]/3.)  + 2.*K0/3.;
  _Keq[i][5] =  ST23 + 
                reSum23( RE23 , int(fmod(double(_sums[18]),3.)) ) + 
                int( (double(_sums[18])/3.) );                        //eq. AD

  _Xeq[i][0] =  (_difs[19] + 5.*K0) / 4.;
  _Xeq[i][1] =  (_difs[18] + 5.*K0) / 4.;
  }

  i=29;  //  i = DTBtiChipEQMAP->index("5768D");  --> 5R7L6R8R
  if(PTMS[i] ){
  _Keq[i][0] =  _sums[17];
  _Keq[i][1] = -_sums[19]       + 2.*K0;
  _Keq[i][2] = -_difs[20]       + 2.*K0;
  _Keq[i][3] = -(_difs[16]/2.)  +    K0 + 0.01;
  _Keq[i][4] = -(_sums[22]/2.)  + 2.*K0 + 0.01;
  //_Keq[i][5] = -(_difs[18]/3.)  + 4.*K0/3. ;
  _Keq[i][5] =  ST43 + 
                reSum( RE43 , int(fmod(double(-_difs[18]),3.)) ) + 
                int( (double(-_difs[18])/3.) );                        //eq. AD

  _Xeq[i][0] = ( _difs[19] + 5.*K0) / 4.;
  _Xeq[i][1] = (-_sums[18] + 7.*K0) / 4.;
  }

  i=30;  //  9768A  --> 9L7L6R8L
  if(PTMS[i] ){
  _Keq[i][0] = -_difs[23];
  _Keq[i][1] = -_sums[19]       + 2.*K0;
  _Keq[i][2] =  _sums[20];
  _Keq[i][3] = -(_sums[21]/2.)  +    K0 + 0.01;
  _Keq[i][4] =  (_difs[22]/2.)  +    K0 + 0.01;
  //_Keq[i][5] = -(_difs[24]/3.)  + 2.*K0/3. ;
  _Keq[i][5] =  ST23 + 
                reSum23( RE23 , int(fmod(double(-_difs[24]),3.)) ) + 
                int( (double(-_difs[24])/3.) );                        //eq. AD

  _Xeq[i][0] = (_difs[19] + 5.*K0) / 4.;
  _Xeq[i][1] = (_sums[24] + 5.*K0) / 4.;
  }
 
  i=31;  //  9768B  --> 9L7R6R8L
  if(PTMS[i] ){
  _Keq[i][0] = -_sums[23]       + 2.*K0;
  _Keq[i][1] =  _difs[19];
  _Keq[i][2] =  _sums[20];
  _Keq[i][3] = -(_sums[21]/2.)  +   K0 + 0.01;
  _Keq[i][4] =  (_sums[22]/2.) + 0.01;
  //_Keq[i][5] = -(_difs[24]/3.)  + 2.*K0/3.;
  _Keq[i][5] =  ST23 + 
                reSum23( RE23 , int(fmod(double(-_difs[24]),3.)) ) + 
                int( (double(-_difs[24])/3.) );                        //eq. AD

  _Xeq[i][0] = (-_sums[19] + 7.*K0) / 4.;
  _Xeq[i][1] = ( _sums[24] + 5.*K0) / 4.;
  }
 
  // debugging
  if(config()->debug()>3){
    cout << endl << " Step: " << currentStep() << endl;
    for(i=0;i<32;i++){
      if(PTMS[i] ){
        cout << "K Equation " << i << " --> ";
        int j=0;
        for(j=0;j<6;j++){
	  cout << _Keq[i][j] << " ";
        }
        cout << endl;
        cout << "X Equation " << i << " --> ";
        for(j=0;j<2;j++){
	  cout << _Xeq[i][j] << " ";
        }
        if( i==0 ){
          cout << _XeqAB_patt0 << " " << _XeqCD_patt0 << " ";
          cout << _XeqAC_patt0 << " " << _XeqBD_patt0 << " ";
        }
        cout << endl;
      }
    }
  }
  // end debugging
  
}
