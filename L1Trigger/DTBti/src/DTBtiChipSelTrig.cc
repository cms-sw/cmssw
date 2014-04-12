//-----------------------------------------------------------------
//
//   Class: DTBtiChip
//
//   Description: Implementation of DTBtiChip 
//                trigger algorithm
//                (Trigger selection implementation)
//
//
//   Author List:
//   C. Grandi
//   Modifications: 
//   23/X/02 Sara Vanini : asymmetric acceptance added for H trig
//   07/I/03 SV: sort patterns for equivalence with bti manual
//   13/I/03 SV: fixed for Keq in manual form
//   30/IX/03 SV: redundancies for xA xB xC xD low trigs included
//   22/VI/04 SV: last trigger code update
//----------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTBti/interface/DTBtiChip.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "L1TriggerConfig/DTTPGConfig/interface/BitArray.h"

//---------------
// C++ Headers --
//---------------
#include<iostream>
#include<cmath>
#include<iomanip>

using namespace std;

void 
DTBtiChip::findTrig(){

  if(config()->debug()>3){
    cout << "DTBtiChip::findTrig called" << endl; 
  }

  //pattern type:       1 = normal
  //			2 = 1L1
  //			3 = 11L
  //			4 = L11
  //			5 = 1H1
  //			6 = 11H
  //			7 = H11

//  int accpatB = config()->AccPattB(); //default +-1
//  int accpatA = config()->AccPattA(); //default +-2
//  int tiKes = config()->XON();
  int RON = config()->RONflag();  //default 1, redundant patterns enabled
  int PTMS[32];
  for(int i=0; i<32; i++){
    PTMS[i] = config()->PTMSflag(i);
  }

  //Triggers (same order as manual):
  for(int hl=0;hl<2;hl++){  //hl=0 high   hl=1 low
  if( RON==1 ){
    if( keepTrigPatt(PTMS[0],0,1,hl) ) return;   // 1324A --> 0 - 1L3L2R4L: nor
  }
  if( keepTrigPatt(PTMS[1],1,2,hl) ) return;   // 1324B --> 1 - 1L3L2R4R: 1L1
  if( keepTrigPatt(PTMS[2],2,1,hl) ) return;   // 1324C --> 2 - 1R3L2R4L: nor
  if( keepTrigPatt(PTMS[3],3,1,hl) ) return;   // 1324D --> 3 - 1R3L2R4R: nor
  if( keepTrigPatt(PTMS[4],4,3,hl) ) return;   // 1364A -->  4 - 1L3L6L4R: 11L
  if( keepTrigPatt(PTMS[5],5,1,hl) ) return;   // 1364B -->  5 - 1R3L6L4R: nor
  if( keepTrigPatt(PTMS[6],6,1,hl) ) return;   // 1364C -->  6 - 1R3R6L4R: nor
  if( keepTrigPatt(PTMS[7],7,5,hl) ) return;   // 1368A -->  7 - 1R3R6L8L: 1H1
  if( keepTrigPatt(PTMS[8],8,6,hl) ) return;   // 1368B -->  8 - 1R3R6R8L: 11H
  if( keepTrigPatt(PTMS[9],9,7,hl) ) return;   // 1368C -->  9 - 1R3L6L8L: H11
  if( keepTrigPatt(PTMS[10],10,1,hl) ) return;   // 5324A --> 10 - 5L3L2R4L: nor
  if( keepTrigPatt(PTMS[11],11,1,hl) ) return;   // 5324B --> 11 - 5L3R2R4L: nor
  if( keepTrigPatt(PTMS[12],12,6,hl) ) return;   // 5324C --> 12 - 5R3R2R4L: 11H
  if( keepTrigPatt(PTMS[13],13,1,hl) ) return;   // 5364A --> 13 - 5L3R6L4L: nor
  if( keepTrigPatt(PTMS[14],14,1,hl) ) return;   // 5364B --> 14 - 5L3R6L4R: nor
  if( keepTrigPatt(PTMS[15],15,5,hl) ) return;   // 5364C --> 15 - 5R3R6L4L: 1H1
  if( keepTrigPatt(PTMS[16],16,1,hl) ) return;   // 5364D --> 16 - 5R3R6L4R: nor
  if( keepTrigPatt(PTMS[17],17,1,hl) ) return;   // 5368A --> 17 - 5L3R6L8L: nor
  if( keepTrigPatt(PTMS[18],18,1,hl) ) return;   // 5368B --> 18 - 5L3R6R8L: nor
  if( keepTrigPatt(PTMS[19],19,4,hl) ) return;   // 5368C --> 19 - 5L3R6R8R: L11
  if( keepTrigPatt(PTMS[20],20,1,hl) ) return;   // 5764A --> 20 - 5R7L6L4R: nor
  if( keepTrigPatt(PTMS[21],21,1,hl) ) return;   // 5764B --> 21 - 5R7L6R4R: nor
  if( keepTrigPatt(PTMS[22],22,7,hl) ) return;   // 5764C --> 22 - 5R7L6L4L: H11
  if( keepTrigPatt(PTMS[23],23,3,hl) ) return;   // 9764A --> 23 - 9L7L6L4R: 11L
  if( keepTrigPatt(PTMS[24],24,2,hl) ) return;   // 9764B --> 24 - 9L7L6R4R: 1L1
  if( keepTrigPatt(PTMS[25],25,4,hl) ) return;   // 9764C --> 25 - 9L7R6R4R: L11
  if( keepTrigPatt(PTMS[26],26,1,hl) ) return;   // 5768A -->  26 - 5L7L6R8L: nor   
  if( RON==1 ){
    if( keepTrigPatt(PTMS[27],27,2,hl) ) return;   // 5768B --> 27 - 5L7L6R8R: 1L1
    if( keepTrigPatt(PTMS[28],28,1,hl) ) return;   // 5768C --> 28 - 5R7L6R8L: nor
    if( keepTrigPatt(PTMS[29],29,1,hl) ) return;   // 5768D --> 29 - 5R7L6R8R: nor
    if( keepTrigPatt(PTMS[30],30,1,hl) ) return;   // 9768A --> 30 - 9L7L6R8L: nor
    if( keepTrigPatt(PTMS[31],31,1,hl) ) return;   // 9768B --> 31 - 9L7R6R8L: nor 
  }
 
  }//end h/l loop



/*
  for(int hl=0;hl<2;hl++){  //hl=0 high   hl=1 low
  if( keepTrigPatt(PTMS[0],0,1,hl) ) return;   // 5768A -->  0 - 5L7L6R8L: nor   
  if( RON==1 ){
    if( keepTrigPatt(PTMS[1],1,2,hl) ) return;   // 5768B -->  1 - 5L7L6R8R: 1L1
    if( keepTrigPatt(PTMS[2],2,1,hl) ) return;   // 5768C -->  2 - 5R7L6R8L: nor
    if( keepTrigPatt(PTMS[3],3,1,hl) ) return;   // 5768D -->  3 - 5R7L6R8R: nor
  }
  if( keepTrigPatt(PTMS[4],4,3,hl) ) return;   // 1364A -->  4 - 1L3L6L4R: 11L
  if( keepTrigPatt(PTMS[5],5,1,hl) ) return;   // 1364B -->  5 - 1R3L6L4R: nor
  if( keepTrigPatt(PTMS[6],6,1,hl) ) return;   // 1364C -->  6 - 1R3R6L4R: nor
  if( keepTrigPatt(PTMS[7],7,5,hl) ) return;   // 1368A -->  7 - 1R3R6L8L: 1H1
  if( keepTrigPatt(PTMS[8],8,6,hl) ) return;   // 1368B -->  8 - 1R3R6R8L: 11H
  if( keepTrigPatt(PTMS[9],9,7,hl) ) return;   // 1368C -->  9 - 1R3L6L8L: H11
  if( keepTrigPatt(PTMS[10],10,1,hl) ) return;   // 5324A --> 10 - 5L3L2R4L: nor
  if( keepTrigPatt(PTMS[11],11,1,hl) ) return;   // 5324B --> 11 - 5L3R2R4L: nor
  if( keepTrigPatt(PTMS[12],12,6,hl) ) return;   // 5324C --> 12 - 5R3R2R4L: 11H
  if( keepTrigPatt(PTMS[13],13,1,hl) ) return;   // 5364A --> 13 - 5L3R6L4L: nor
  if( keepTrigPatt(PTMS[14],14,1,hl) ) return;   // 5364B --> 14 - 5L3R6L4R: nor
  if( keepTrigPatt(PTMS[15],15,5,hl) ) return;   // 5364C --> 15 - 5R3R6L4L: 1H1
  if( keepTrigPatt(PTMS[16],16,1,hl) ) return;   // 5364D --> 16 - 5R3R6L4R: nor
  if( keepTrigPatt(PTMS[17],17,1,hl) ) return;   // 5368A --> 17 - 5L3R6L8L: nor
  if( keepTrigPatt(PTMS[18],18,1,hl) ) return;   // 5368B --> 18 - 5L3R6R8L: nor
  if( keepTrigPatt(PTMS[19],19,4,hl) ) return;   // 5368C --> 19 - 5L3R6R8R: L11
  if( keepTrigPatt(PTMS[20],20,1,hl) ) return;   // 5764A --> 20 - 5R7L6L4R: nor
  if( keepTrigPatt(PTMS[21],21,1,hl) ) return;   // 5764B --> 21 - 5R7L6R4R: nor
  if( keepTrigPatt(PTMS[22],22,7,hl) ) return;   // 5764C --> 22 - 5R7L6L4L: H11
  if( keepTrigPatt(PTMS[23],23,3,hl) ) return;   // 9764A --> 23 - 9L7L6L4R: 11L
  if( keepTrigPatt(PTMS[24],24,2,hl) ) return;   // 9764B --> 24 - 9L7L6R4R: 1L1
  if( keepTrigPatt(PTMS[25],25,4,hl) ) return;   // 9764C --> 25 - 9L7R6R4R: L11
  if( RON==1 ){
    if( keepTrigPatt(PTMS[26],26,1,hl) ) return;   // 1324A --> 26 - 1L3L2R4L: nor
  }
  if( keepTrigPatt(PTMS[27],27,2,hl) ) return;   // 1324B --> 27 - 1L3L2R4R: 1L1
  if( keepTrigPatt(PTMS[28],28,1,hl) ) return;   // 1324C --> 28 - 1R3L2R4L: nor
  if( keepTrigPatt(PTMS[29],29,1,hl) ) return;   // 1324D --> 29 - 1R3L2R4R: nor
  if( RON==1 ){
    if( keepTrigPatt(PTMS[30],30,1,hl) ) return;   // 9768A --> 30 - 9L7L6R8L: nor
  }
  if( keepTrigPatt(PTMS[31],31,1,hl) ) return;   // 9768B --> 31 - 9L7R6R8L: nor 

  }//end h/l loop

*/
  
/* 
  // High level triggers:
  if( keepTrig( 1,accpatB,8) ) return;   // 5768B -->  1 - acc. patt. B
  if( keepTrig( 2,accpatA,8) ) return;   // 5768C -->  2 - acc. patt. A
  if( keepTrig( 3,accpatA,8) ) return;   // 5768D -->  3 - acc. patt. A
  if( keepTrig( 4,accpatB,8) ) return;   // 1364A -->  4 - acc. patt. B
  if( keepTrig( 5,accpatA,8) ) return;   // 1364B -->  5 - acc. patt. A
  if( keepTrig( 6,accpatA,8) ) return;   // 1364C -->  6 - acc. patt. A
  if( keepTrig( 7,accpatB,8) ) return;   // 1368A -->  7 - acc. patt. B
  if( keepTrig( 8,accpatB,8) ) return;   // 1368B -->  8 - acc. patt. B
  if( keepTrig( 9,accpatB,8) ) return;   // 1368C -->  9 - acc. patt. B
  if( keepTrig(10,accpatA,8) ) return;   // 5324A --> 10 - acc. patt. A
  if( keepTrig(11,accpatA,8) ) return;   // 5324B --> 11 - acc. patt. A
  if( keepTrig(12,accpatB,8) ) return;   // 5324C --> 12 - acc. patt. B
  if( keepTrig(13,accpatA,8) ) return;   // 5364A --> 13 - acc. patt. A
  if( keepTrig(14,accpatA,8) ) return;   // 5364B --> 14 - acc. patt. A
  if( keepTrig(15,accpatB,8) ) return;   // 5364C --> 15 - acc. patt. B
  if( keepTrig(16,accpatA,8) ) return;   // 5364D --> 16 - acc. patt. A
  if( keepTrig(17,accpatA,8) ) return;   // 5368A --> 17 - acc. patt. A
  if( keepTrig(18,accpatA,8) ) return;   // 5368B --> 18 - acc. patt. A
  if( keepTrig(19,accpatB,8) ) return;   // 5368C --> 19 - acc. patt. B
  if( keepTrig(20,accpatA,8) ) return;   // 5764A --> 20 - acc. patt. A
  if( keepTrig(21,accpatA,8) ) return;   // 5764B --> 21 - acc. patt. A
  if( keepTrig(22,accpatB,8) ) return;   // 5764C --> 22 - acc. patt. B
  if( keepTrig(23,accpatB,8) ) return;   // 9764A --> 23 - acc. patt. B
  if( keepTrig(24,accpatB,8) ) return;   // 9764B --> 24 - acc. patt. B
  if( keepTrig(25,accpatB,8) ) return;   // 9764C --> 25 - acc. patt. B
  if( keepTrig( 0,accpatA,8) ) return;   // 5768A -->  0 - acc. patt. A
  */ 
 /* 
  // Low level triggers -B
  if( keepTrig( 1,accpatB,2) ) return;   // 5768B -->  1 - acc. patt. B
  if( keepTrig( 2,accpatA,2) ) return;   // 5768C -->  2 - acc. patt. A
  if(tiKes==1) {
    if( keepTrig( 3,accpatA,2) ) return; // 5768D -->  3 - acc. patt. A
  }
  if( keepTrig( 4,accpatB,2) ) return;   // 1364A -->  4 - acc. patt. B
  if( keepTrig( 5,accpatA,2) ) return;   // 1364B -->  5 - acc. patt. A
  if( keepTrig( 6,accpatA,2) ) return;   // 1364C -->  6 - acc. patt. A
  if( keepTrig( 7,accpatB,2) ) return;   // 1368A -->  7 - acc. patt. B
  if( keepTrig( 8,accpatB,2) ) return;   // 1368B -->  8 - acc. patt. B
  if( keepTrig( 9,accpatB,2) ) return;   // 1368C -->  9 - acc. patt. B
  if( keepTrig(10,accpatA,2) ) return;   // 5324A --> 10 - acc. patt. A
  if( keepTrig(11,accpatA,2) ) return;   // 5324B --> 11 - acc. patt. A
  if( keepTrig(12,accpatB,2) ) return;   // 5324C --> 12 - acc. patt. B
  if(tiKes==1) {
    if( keepTrig(13,accpatA,2) ) return; // 5364A --> 13 - acc. patt. A
  }
  if( keepTrig(14,accpatA,2) ) return;   // 5364B --> 14 - acc. patt. A
  if( keepTrig(15,accpatB,2) ) return;   // 5364C --> 15 - acc. patt. B
  if( keepTrig(16,accpatA,2) ) return;   // 5364D --> 16 - acc. patt. A
  if(tiKes==1) {
    if( keepTrig(17,accpatA,2) ) return; // 5368A --> 17 - acc. patt. A
  }
  if( keepTrig(18,accpatA,2) ) return;   // 5368B --> 18 - acc. patt. A
  if( keepTrig(19,accpatB,2) ) return;   // 5368C --> 19 - acc. patt. B
  if( keepTrig(20,accpatA,2) ) return;   // 5764A --> 20 - acc. patt. A
  if(tiKes==1) {
    if( keepTrig(21,accpatA,2) ) return; // 5764B --> 21 - acc. patt. A
  }
  if( keepTrig(22,accpatB,2) ) return;   // 5764C --> 22 - acc. patt. B
  if( keepTrig(23,accpatB,2) ) return;   // 9764A --> 23 - acc. patt. B
  if( keepTrig(24,accpatB,2) ) return;   // 9764B --> 24 - acc. patt. B
  if( keepTrig(25,accpatB,2) ) return;   // 9764C --> 25 - acc. patt. B
  if( keepTrig( 0,accpatA,2) ) return;   // 5768A -->  0 - acc. patt. A

  // Low level triggers -C
  if( keepTrig( 1,accpatB,3) ) return;   // 5768B -->  1 - acc. patt. B
  if( keepTrig( 2,accpatA,3) ) return;   // 5768C -->  2 - acc. patt. A
  if( keepTrig( 3,accpatA,3) ) return;   // 5768D -->  3 - acc. patt. A
  if( keepTrig( 4,accpatB,3) ) return;   // 1364A -->  4 - acc. patt. B
  if( keepTrig( 5,accpatA,3) ) return;   // 1364B -->  5 - acc. patt. A
  if(tiKes==1) {
    if( keepTrig( 6,accpatA,3) ) return; // 1364C -->  6 - acc. patt. A
  }
  if( keepTrig( 7,accpatB,3) ) return;   // 1368A -->  7 - acc. patt. B
  if( keepTrig( 8,accpatB,3) ) return;   // 1368B -->  8 - acc. patt. B
  if( keepTrig( 9,accpatB,3) ) return;   // 1368C -->  9 - acc. patt. B
  if(tiKes==1) {
    if( keepTrig(10,accpatA,3) ) return; // 5324A --> 10 - acc. patt. A
  }
  if( keepTrig(11,accpatA,3) ) return;   // 5324B --> 11 - acc. patt. A
  if( keepTrig(12,accpatB,3) ) return;   // 5324C --> 12 - acc. patt. B
  if( keepTrig(13,accpatA,3) ) return;   // 5364A --> 13 - acc. patt. A
  if( keepTrig(14,accpatA,3) ) return;   // 5364B --> 14 - acc. patt. A
  if( keepTrig(15,accpatB,3) ) return;   // 5364C --> 15 - acc. patt. B
  if(tiKes==1) {
    if( keepTrig(16,accpatA,3) ) return; // 5364D --> 16 - acc. patt. A
  }
  if( keepTrig(17,accpatA,3) ) return;   // 5368A --> 17 - acc. patt. A
  if( keepTrig(18,accpatA,3) ) return;   // 5368B --> 18 - acc. patt. A
  if( keepTrig(19,accpatB,3) ) return;   // 5368C --> 19 - acc. patt. B
  if( keepTrig(20,accpatA,3) ) return;   // 5764A --> 20 - acc. patt. A
  if( keepTrig(21,accpatA,3) ) return;   // 5764B --> 21 - acc. patt. A
  if( keepTrig(22,accpatB,3) ) return;   // 5764C --> 22 - acc. patt. B
  if( keepTrig(23,accpatB,3) ) return;   // 9764A --> 23 - acc. patt. B
  if( keepTrig(24,accpatB,3) ) return;   // 9764B --> 24 - acc. patt. B
  if( keepTrig(25,accpatB,3) ) return;   // 9764C --> 25 - acc. patt. B
  if(tiKes==1) {
    if( keepTrig( 0,accpatA,3) ) return; // 5768A -->  0 - acc. patt. A
  }

  // Low level triggers -A
  if( keepTrig( 1,accpatB,1) ) return;   // 5768B -->  1 - acc. patt. B
  if( keepTrig( 2,accpatA,1) ) return;   // 5768C -->  2 - acc. patt. A
  if( keepTrig( 3,accpatA,1) ) return;   // 5768D -->  3 - acc. patt. A
  if( keepTrig( 4,accpatB,1) ) return;   // 1364A -->  4 - acc. patt. B
  if( keepTrig( 5,accpatA,1) ) return;   // 1364B -->  5 - acc. patt. A
  if( keepTrig( 6,accpatA,1) ) return;   // 1364C -->  6 - acc. patt. A
  if( keepTrig( 7,accpatB,1) ) return;   // 1368A -->  7 - acc. patt. B
  if( keepTrig( 8,accpatB,1) ) return;   // 1368B -->  8 - acc. patt. B
  if(tiKes==1) {
    if( keepTrig( 9,accpatB,1) ) return; // 1368C -->  9 - acc. patt. B
  }
  if( keepTrig(10,accpatA,1) ) return;   // 5324A --> 10 - acc. patt. A
  if( keepTrig(11,accpatA,1) ) return;   // 5324B --> 11 - acc. patt. A
  if( keepTrig(12,accpatB,1) ) return;   // 5324C --> 12 - acc. patt. B
  if( keepTrig(13,accpatA,1) ) return;   // 5364A --> 13 - acc. patt. A
  if( keepTrig(14,accpatA,1) ) return;   // 5364B --> 14 - acc. patt. A
  if( keepTrig(15,accpatB,1) ) return;   // 5364C --> 15 - acc. patt. B
  if( keepTrig(16,accpatA,1) ) return;   // 5364D --> 16 - acc. patt. A
  if( keepTrig(17,accpatA,1) ) return;   // 5368A --> 17 - acc. patt. A
  if( keepTrig(18,accpatA,1) ) return;   // 5368B --> 18 - acc. patt. A
  if(tiKes==1) {
    if( keepTrig(19,accpatB,1) ) return; // 5368C --> 19 - acc. patt. B
  }
  if( keepTrig(20,accpatA,1) ) return;   // 5764A --> 20 - acc. patt. A
  if( keepTrig(21,accpatA,1) ) return;   // 5764B --> 21 - acc. patt. A
  if(tiKes==1) {
    if( keepTrig(22,accpatB,1) ) return; // 5764C --> 22 - acc. patt. B
  }
  if( keepTrig(23,accpatB,1) ) return;   // 9764A --> 23 - acc. patt. B
  if( keepTrig(24,accpatB,1) ) return;   // 9764B --> 24 - acc. patt. B
  if(tiKes==1) {
    if( keepTrig(25,accpatB,1) ) return; // 9764C --> 25 - acc. patt. B
  }
  if( keepTrig( 0,accpatA,1) ) return;   // 5768A -->  0 - acc. patt. A

  // Low level triggers -D
  if( keepTrig( 0,accpatA,4) ) return;   // 5768A -->  0 - acc. patt. A
  if( keepTrig( 1,accpatB,4) ) return;   // 5768B -->  1 - acc. patt. B
  if( keepTrig( 2,accpatA,4) ) return;   // 5768C -->  2 - acc. patt. A
  if( keepTrig( 3,accpatA,4) ) return;   // 5768D -->  3 - acc. patt. A
  if(tiKes==1) {
    if( keepTrig( 4,accpatB,4) ) return; // 1364A -->  4 - acc. patt. B
  }
  if( keepTrig( 5,accpatA,4) ) return;   // 1364B -->  5 - acc. patt. A
  if( keepTrig( 6,accpatA,4) ) return;   // 1364C -->  6 - acc. patt. A
  if( keepTrig( 7,accpatB,4) ) return;   // 1368A -->  7 - acc. patt. B
  if(tiKes==1) {
    if( keepTrig( 8,accpatB,4) ) return; // 1368B -->  8 - acc. patt. B
  }
  if( keepTrig( 9,accpatB,4) ) return;   // 1368C -->  9 - acc. patt. B
  if( keepTrig(10,accpatA,4) ) return;   // 5324A --> 10 - acc. patt. A
  if( keepTrig(11,accpatA,4) ) return;   // 5324B --> 11 - acc. patt. A
  if(tiKes==1) {
    if( keepTrig(12,accpatB,4) ) return; // 5324C --> 12 - acc. patt. B
  }
  if( keepTrig(13,accpatA,4) ) return;   // 5364A --> 13 - acc. patt. A
  if( keepTrig(14,accpatA,4) ) return;   // 5364B --> 14 - acc. patt. A
  if( keepTrig(15,accpatB,4) ) return;   // 5364C --> 15 - acc. patt. B
  if( keepTrig(16,accpatA,4) ) return;   // 5364D --> 16 - acc. patt. A
  if( keepTrig(17,accpatA,4) ) return;   // 5368A --> 17 - acc. patt. A
  if( keepTrig(18,accpatA,4) ) return;   // 5368B --> 18 - acc. patt. A
  if( keepTrig(19,accpatB,4) ) return;   // 5368C --> 19 - acc. patt. B
  if( keepTrig(20,accpatA,4) ) return;   // 5764A --> 20 - acc. patt. A
  if( keepTrig(21,accpatA,4) ) return;   // 5764B --> 21 - acc. patt. A
  if( keepTrig(22,accpatB,4) ) return;   // 5764C --> 22 - acc. patt. B
  if(tiKes==1) {
    if( keepTrig(23,accpatB,4) ) return; // 9764A --> 23 - acc. patt. B
  }
  if( keepTrig(24,accpatB,4) ) return;   // 9764B --> 24 - acc. patt. B
  if( keepTrig(25,accpatB,4) ) return;   // 9764C --> 25 - acc. patt. B
*/
}

int DTBtiChip::keepTrigPatt(const int flag,const int eq,const int pattType, int hlflag) {
  //if pattern is not enabled, return
  if(flag==0)
    return  0;

  int AC1 = config()->AccPattAC1(); //default 0
  int AC2 = config()->AccPattAC2(); //default 3
  int ACH = config()->AccPattACH(); //default 1
  int ACL = config()->AccPattACL(); //default 2
  int tiKes = config()->XON();

  if(config()->debug()>4){
    cout << "DTBtiChip::keepTrigPatt called with arguments: ";
    cout << eq << ", " << pattType << ", " << hlflag << endl; 
    cout<<"AC1,AC2,ACH,ACL="<<AC1<<" "<<AC2<<" "<<ACH<<" "<<ACL<<endl;
  }

  BitArray<80> val0, val1, val2, val3, val4, val5;
  int mm[6];

  //NB _Keq = (Kman - K0)/2   where K0=tmax*2  --->OBSOLETE, now Keq=Kman
  //int K0 = 2 * (config()->ST()/2.); 
  int K0 = int (config()->ST());

  int i=0;
  float Keqfloat[6] = {0,0,0,0,0,0};
  for(i=0;i<6;i++){
    mm[i] = -1;
    int mk = (int)(_Keq[eq][i] - K0);
    //if(abs(mk) > config()->KCut(_id.superlayer()-1))continue;
    if(abs(mk) > 2*K0)  continue;
    Keqfloat[i]=_Keq[eq][i];
    mm[i]=(int)(_Keq[eq][i]);
    //    if(_Keq[eq][i]<0){
    //      mm[i]=_Keq[eq][i]*2-0.5+KCen;
    //    } else {
    //      mm[i]=_Keq[eq][i]*2+KCen;
    //    }
  }

  switch(pattType)
    {
      case 1: //normal pattern
	//if(hlflag==1 && (eq!=2 && eq!=14 && eq!=28) ){  //test for L
/*	if(hlflag==1){  //test for L
          acceptMask(&val0,mm[0],AC1); //eqAB
          acceptMask(&val1,mm[1],AC1); //eqBC
          acceptMask(&val2,mm[2],AC1); //eqCD
        }
        else
*/
        {
          acceptMask(&val0,mm[0],AC2); //eqAB
          acceptMask(&val1,mm[1],AC2); //eqBC
          acceptMask(&val2,mm[2],AC2); //eqCD
        }
	break;

      case 2: //1L1 pattern
	acceptMask(&val0,mm[0],AC1); //eqAB
        acceptMask(&val1,mm[1],ACL); //eqBC
        acceptMask(&val2,mm[2],AC1); //eqCD
	break;

      case 3: //11L pattern
	acceptMask(&val0,mm[0],AC1); //eqAB
        acceptMask(&val1,mm[1],AC1); //eqBC
        acceptMask(&val2,mm[2],ACL); //eqCD
	break;

      case 4: //L11 pattern
	acceptMask(&val0,mm[0],ACL); //eqAB
        acceptMask(&val1,mm[1],AC1); //eqBC
        acceptMask(&val2,mm[2],AC1); //eqCD
	break;

      case 5: //1H1 pattern
	acceptMask(&val0,mm[0],AC1); //eqAB
        acceptMask(&val1,mm[1],ACH); //eqBC
        acceptMask(&val2,mm[2],AC1); //eqCD
	break;

      case 6: //11H pattern
	acceptMask(&val0,mm[0],AC1); //eqAB
        acceptMask(&val1,mm[1],AC1); //eqBC
        acceptMask(&val2,mm[2],ACH); //eqCD
	break;

      case 7: //H11 pattern
	acceptMask(&val0,mm[0],ACH); //eqAB
        acceptMask(&val1,mm[1],AC1); //eqBC
        acceptMask(&val2,mm[2],AC1); //eqCD
	break;
 
      default:
	acceptMask(&val0,mm[0],AC2); //eqAB
        acceptMask(&val1,mm[1],AC2); //eqBC
        acceptMask(&val2,mm[2],AC2); //eqCD
	break;
    } 	

  //eq. AC and BD acceptance are always +-1 ->code 00
  int acc = 0;
  acceptMask(&val3,mm[3],acc); //eq.AC
  acceptMask(&val4,mm[4],acc); //eq.BD

  //eq. AD is the reference value!
  if(mm[5]>0){
    val5.set(mm[5]);
  }

  // debugging: print() method prints from last to first bit!
  if(config()->debug()>4){
    cout << " dump of val arrays: " << endl;
    //    cout << val0.to_string() << endl;
    //    cout << val1.to_string() << endl;
    //    cout << val2.to_string() << endl;
    //    cout << val3.to_string() << endl;
    //    cout << val4.to_string() << endl;
    //    cout << val5.to_string() << endl;
    val0.print(); 
    cout << endl;
    val1.print();
    cout << endl;
    val2.print();
    cout << endl;
    val3.print();
    cout << endl;
    val4.print();
    cout << endl;
    val5.print();
    cout << endl;
   }
  // end debugging

  //search for High trigger:
  if(hlflag==0){
    int code = 0;
    int KMax = 0;
    int LKMax = -1;
    for(i=0;i<80;i++){
      int val = val0.element(i)+val1.element(i)+val2.element(i)+
                val3.element(i)+val4.element(i)+val5.element(i);
      if(val>KMax) {
        KMax=val;
        LKMax=i;
      }
    }
  
    //SV: K value is stored in 6 bits, so K=64->0, etc
    if(LKMax>63)
      LKMax-=64;

    if(KMax==6) {
      code=8;
      int X;
      if( eq==0 )  //store Xbc only for patt 0, else store Xad
        X=int(_Xeq[eq][0]);
      else
        X=int(_Xeq[eq][1]);
      store(eq,code,LKMax,X,Keqfloat[0],Keqfloat[1],
           Keqfloat[2],Keqfloat[3],Keqfloat[4],Keqfloat[5]);
      return 1;
    }
    return 0;
  } //end H 

  //search for Low triggers:
  if(hlflag==1){
    int code = 0;
    int RON = config()->RONflag();


    //hit in B is missing
    if(config()->debug()>4)
      cout << "SelTrig: searching low-B" << endl;
    int LKMax = -1;
    for(i=0;i<80;i++){
      int val = val2.element(i)+val3.element(i)+val5.element(i);
      if(val==3){ //Ref. is eqAD
        code=2;
        LKMax=i;
        int storefg = 1; 

        //SV - XON time-ind.Keq suppr. XON=0 do not abilitate patterns 
        if(tiKes==0) {
          if(eq==3 || eq==13 || eq==17 || eq==21 || eq==29){  
            if(config()->debug()>3)
              cout << "SelTrig: doing XON suppression!"<<endl;
            storefg = 0;
          }
        }

        //SV - RON suppression for low triggers
        if( RON==0 ){
          if( eq==19 ){
            if(config()->debug()>3)
              cout << "SelTrig: doing RON low triggers suppression!"<<endl;
            storefg = 0;
          }
        }

        if(storefg){ 
          //SV: K value is stored in 6 bits, so K=64->0, etc
          if(LKMax>63)
            LKMax-=64;
          int X;
          if( eq==0 )
            X=int(_Xeq[eq][0]);
          else
            X=int(_Xeq[eq][1]);
          store(eq,code,LKMax,X,Keqfloat[0],Keqfloat[1],
           Keqfloat[2],Keqfloat[3],Keqfloat[4],Keqfloat[5]);
          return 1;
        }  
      }
    } //end -B Low

    //hit in C is missing
    if(config()->debug()>3)
      cout << "SelTrig: searching low-C" << endl;
     for(i=0;i<80;i++){
      int val = val0.element(i)+val4.element(i)+val5.element(i);
      if(val==3){ //Ref. is eqAD
        code=3;
        LKMax=i;
        int storefg = 1;
 
        //SV - XON time-ind.Keq suppr.
        if(tiKes==0) {
          if(eq==0 || eq==6 || eq==10 || eq==16 || eq==26 || eq==30){  
            if(config()->debug()>3)
              cout << "SelTrig: doing XON suppression!"<<endl;
            storefg = 0;
          }
        }

        if(storefg){
          //SV: K value is stored in 6 bits, so K=64->0, etc
          if(LKMax>63)
            LKMax-=64;

          int X;
          if( eq==0 )
            X=int(_Xeq[eq][0]);
          else
            X=int(_Xeq[eq][1]);
          store(eq,code,LKMax,X,Keqfloat[0],Keqfloat[1],
           Keqfloat[2],Keqfloat[3],Keqfloat[4],Keqfloat[5]);
          return 1;
        }
      }
    } // end -C Low

    //for -A and -D low acceptance is +-1
//    if(pattType==1){
      val0.reset();
      val1.reset();
      val2.reset();
      acceptMask(&val0,mm[0],AC1); //eqAB
      acceptMask(&val1,mm[1],AC1); //eqBC
      acceptMask(&val2,mm[2],AC1); //eqCD
//    }

    //hit in A is missing
    if(config()->debug()>4)
      cout << "SelTrig: searching low-A" << endl;
    for(i=0;i<80;i++){
      int val = val1.element(i)+val2.element(i)+val4.element(i);
      if(val==3 && i==mm[4]){ //Ref. is eqBD
        code=1;
        LKMax=i;
        int storefg = 1;

        //SV - XON time-ind.Keq suppr.
        if(tiKes==0) {
          if(eq==9 || eq==19 || eq==22 || eq==25 ){  
            if(config()->debug()>3)
              cout << "SelTrig: doing low-A XON suppression!"<<endl;
          storefg = 0;
          }
        }

        if( RON==0 ){ //SV - RON suppression
          if( eq==26 ){
            if(config()->debug()>3)
              cout << "SelTrig: doing RON low triggers suppression!"<<endl;
            storefg = 0;
          }
        }

        if(storefg){
          //SV: K value is stored in 6 bits, so K=64->0, etc
          if(LKMax>63)
            LKMax-=64;

          store(eq,code,LKMax,int(_Xeq[eq][0]),Keqfloat[0],Keqfloat[1],
           Keqfloat[2],Keqfloat[3],Keqfloat[4],Keqfloat[5]);
          return 1;
        }
      }
    } //end -A Low
 

    //hit in D is missing
    if(config()->debug()>4)
      cout << "SelTrig: searching low-D" << endl;
    for(i=0;i<80;i++){
      int val = val0.element(i)+val1.element(i)+val3.element(i);
      if(val==3 && i==mm[3]){ //Ref. is eqAC
        code=4;
        LKMax=i;
        int storefg = 1;

        //SV - XON time-ind.Keq suppr.
        if(tiKes==0){
          if(eq==4 || eq==8 || eq==12 || eq==23){  
            if(config()->debug()>3)
              cout << "SelTrig: doing XON suppression!"<<endl;
            storefg = 0;
          }
        }

        //SV - RON suppression for low triggers
        if( RON==0 ){
          if(eq==1 || eq==2 || eq==3 || eq==24 || eq==25){
            if(config()->debug()>3)
              cout << "SelTrig: doing RON low triggers suppression!"<<endl;
            storefg = 0;
          }
        }

        if(storefg){ // && _Xeq[eq][1] >=0){
          //SV: K value is stored in 6 bits, so K=64->0, etc
          if(LKMax>63)
            LKMax-=64;

          store(eq,code,LKMax,int(_Xeq[eq][0]),Keqfloat[0],Keqfloat[1],
           Keqfloat[2],Keqfloat[3],Keqfloat[4],Keqfloat[5]);
          return 1;
        }
      }
    } //end -D Low
 
    return 0; 
  } //end Low
  return 0;
}

  
void 
DTBtiChip::acceptMask(BitArray<80> * BitArrPtr,int k,int accep)
{
   if(k>=0&&k<78){
     if(config()->debug()>4)
       cout<<"DTBtiChip::acceptMask ->  Setting acceptance for k="<<k<<endl;

     if(accep==0){ //code 00
       if(k>=1) 
         BitArrPtr->set(k-1);
       BitArrPtr->set(k);
       BitArrPtr->set(k+1);
     }
     if(accep==1){ //code 01
       BitArrPtr->set(k);
       BitArrPtr->set(k+1);
       BitArrPtr->set(k+2);
     }
     if(accep==2){ //code 10
       if(k>1)
         BitArrPtr->set(k-2);
       if(k>=1)
         BitArrPtr->set(k-1);
       BitArrPtr->set(k);
     }
     if(accep==3){ //code 11
       if(k>1)
         BitArrPtr->set(k-2);
       if(k>=1)
         BitArrPtr->set(k-1);
       BitArrPtr->set(k);
       BitArrPtr->set(k+1);
       BitArrPtr->set(k+2);
     }
   }


  if(config()->debug()>4)
    cout<<"DTBtiChip::acceptMask ->  END "<<endl;

}



int 
DTBtiChip::keepTrig(const int eq, const int acp, const int code) {

  if(config()->debug()>4){
    cout << "DTBtiChip::keepTrig called with arguments: ";
    cout << eq << ", " << acp << ", " << code << endl; 
  }

  int const KCen = 40; // Arrays will start from 0 --> use 40 instead of 41
  BitArray<80> val0, val1, val2, val3, val4, val5;
  int mm[6];

  int i=0;
  for(i=0;i<6;i++){
    mm[i]=0;
    int mk = (int)(2*_Keq[eq][i]);
    if(abs(mk) > config()->KCut() )
      	continue;
    mm[i]=(int)(_Keq[eq][i]*2)+KCen;
    //    if(_Keq[eq][i]<0){
    //      mm[i]=_Keq[eq][i]*2-0.5+KCen;
    //    } else {
    //      mm[i]=_Keq[eq][i]*2+KCen;
    //    }
  }

  if(mm[0]>0 && (code==8 || code==3 || code==4) ){
    val0.set(mm[0]-1);
    val0.set(mm[0]);
    val0.set(mm[0]+1);
    if(acp==2 && (code==8 || code==3) ) {
      val0.set(mm[0]-2);
      val0.set(mm[0]+2);
    }
  }

  if(mm[1]>0 && (code==8 || code==1 || code==4) ){
    val1.set(mm[1]-1);
    val1.set(mm[1]);
    val1.set(mm[1]+1);
    if(acp==2 && code==8 ) {
      val1.set(mm[1]-2);
      val1.set(mm[1]+2);
    }
  }

  if(mm[2]>0 && (code==8 || code==1 || code==2) ){
    val2.set(mm[2]-1);
    val2.set(mm[2]);
    val2.set(mm[2]+1);
    if(acp==2 && (code==8 || code==2) ) {
      val2.set(mm[2]-2);
      val2.set(mm[2]+2);
    }
  }

  if(mm[3]>0 && (code==8 || code==2 || code==4) ){
    val3.set(mm[3]-1);
    val3.set(mm[3]);
    val3.set(mm[3]+1);
  }

  if(mm[4]>0 && (code==8 || code==1 || code==3) ){
    val4.set(mm[4]-1);
    val4.set(mm[4]);
    val4.set(mm[4]+1);
  }

  if(mm[5]>0 && (code==8 || code==2 || code==3) ){
    val5.set(mm[5]);
  }

  // debugging
  if(config()->debug()>4){
    cout << " dump of val arrays: " << endl;
    //    cout << val0.to_string() << endl;
    //    cout << val1.to_string() << endl;
    //    cout << val2.to_string() << endl;
    //    cout << val3.to_string() << endl;
    //    cout << val4.to_string() << endl;
    //    cout << val5.to_string() << endl;
    val0.print();
    cout << endl;
    val1.print();
    cout << endl;
    val2.print();
    cout << endl;
    val3.print();
    cout << endl;
    val4.print();
    cout << endl;
    val5.print();
    cout << endl;
  }
  // end debugging

  int KMax = 0;
  int LKMax = -1;
  for(i=0;i<80;i++){
    int val = val0.element(i)+val1.element(i)+val2.element(i)+
              val3.element(i)+val4.element(i)+val5.element(i);
    //    int val = val0.test(i)+val1.test(i)+val2.test(i)+
    //              val3.test(i)+val4.test(i)+val5.test(i);
    if(val>KMax) {
      KMax=val;
      LKMax=i;
    }
  }
  
  // Note that all bits in val are shifted by one w.r.t. FORTRAN version
  // The output K will be the same because of the different value of Kcen

  if        (KMax==6 && code==8) {
    store(eq,8,LKMax-KCen,int(_Xeq[eq][1]));
    return 1;
  } else if (KMax==3 && code!=8) {
    if(code==1 || code==4) {
      store(eq,code,LKMax-KCen,int(_Xeq[eq][0]));
    } else {
      store(eq,code,LKMax-KCen,int(_Xeq[eq][1]));
    }
    return 1;
  }
  return 0; 
  // return value is ITFL of FORTRAN version
}
