#include "rctDataBase.h"


l1t::rctDataBase::rctDataBase() {

  length[RCEt]=10;
  length[RCTau]=1;
  length[RCOf]=1;
  length[HFEt]=8;
  length[HFFg]=1;
  length[IEEt]=6;
  length[IEReg]=1;
  length[IECard]=3;
  length[NEEt]=6;
  length[NEReg]=1;
  length[NECard]=3;
  length[RCHad]=1;

  link[RCEt]=0;
  link[RCTau]=0;
  link[RCOf]=0;
  link[HFEt]=1;
  link[HFFg]=1;
  link[IEEt]=1;
  link[IEReg]=1;
  link[IECard]=1;
  link[NEEt]=1;
  link[NEReg]=1;
  link[NECard]=1;
  link[RCHad]=1;
  
  
  indexfromMP7toRCT[0]=0;
  indexfromMP7toRCT[1]=1;
  indexfromMP7toRCT[2]=18;
  indexfromMP7toRCT[3]=19;
  indexfromMP7toRCT[4]=16;
  indexfromMP7toRCT[5]=17;
  indexfromMP7toRCT[6]=34;
  indexfromMP7toRCT[7]=35;
  indexfromMP7toRCT[8]=2;
  indexfromMP7toRCT[9]=3;
  indexfromMP7toRCT[10]=20;
  indexfromMP7toRCT[11]=21;
  indexfromMP7toRCT[12]=14;
  indexfromMP7toRCT[13]=15;
  indexfromMP7toRCT[14]=32;
  indexfromMP7toRCT[15]=33;
  indexfromMP7toRCT[16]=4;
  indexfromMP7toRCT[17]=5;
  indexfromMP7toRCT[18]=22;
  indexfromMP7toRCT[19]=23;
  indexfromMP7toRCT[20]=12;
  indexfromMP7toRCT[21]=13;
  indexfromMP7toRCT[22]=30;
  indexfromMP7toRCT[23]=31;
  indexfromMP7toRCT[24]=6;
  indexfromMP7toRCT[25]=7;
  indexfromMP7toRCT[26]=24;
  indexfromMP7toRCT[27]=25;
  indexfromMP7toRCT[28]=10;
  indexfromMP7toRCT[29]=11;
  indexfromMP7toRCT[30]=28;
  indexfromMP7toRCT[31]=29;
  indexfromMP7toRCT[32]=8;
  indexfromMP7toRCT[33]=9;
  indexfromMP7toRCT[34]=26;
  indexfromMP7toRCT[35]=27;
  
  indexfromoRSCtoMP7[0]=0;
  indexfromoRSCtoMP7[1]=1;
  indexfromoRSCtoMP7[2]=8;
  indexfromoRSCtoMP7[3]=9;
  indexfromoRSCtoMP7[4]=16;
  indexfromoRSCtoMP7[5]=17;
  indexfromoRSCtoMP7[6]=24;
  indexfromoRSCtoMP7[7]=25;
  indexfromoRSCtoMP7[8]=32;
  indexfromoRSCtoMP7[9]=33;
  indexfromoRSCtoMP7[10]=28;
  indexfromoRSCtoMP7[11]=29;
  indexfromoRSCtoMP7[12]=20;
  indexfromoRSCtoMP7[13]=21;
  indexfromoRSCtoMP7[14]=12;
  indexfromoRSCtoMP7[15]=13;
  indexfromoRSCtoMP7[16]=4;
  indexfromoRSCtoMP7[17]=5;
  indexfromoRSCtoMP7[18]=2;
  indexfromoRSCtoMP7[19]=3;
  indexfromoRSCtoMP7[20]=10;
  indexfromoRSCtoMP7[21]=11;
  indexfromoRSCtoMP7[22]=18;
  indexfromoRSCtoMP7[23]=19;
  indexfromoRSCtoMP7[24]=26;
  indexfromoRSCtoMP7[25]=27;
  indexfromoRSCtoMP7[26]=34;
  indexfromoRSCtoMP7[27]=35;
  indexfromoRSCtoMP7[28]=30;
  indexfromoRSCtoMP7[29]=31;
  indexfromoRSCtoMP7[30]=22;
  indexfromoRSCtoMP7[31]=23;
  indexfromoRSCtoMP7[32]=14;
  indexfromoRSCtoMP7[33]=15;
  indexfromoRSCtoMP7[34]=6;
  indexfromoRSCtoMP7[35]=7;
  
  RCEt_start[0][0]=8;
  RCEt_start[0][1]=18;
  RCEt_start[1][0]=28;
  RCEt_start[1][1]=38;
  RCEt_start[2][0]=48;
  RCEt_start[2][1]=58;
  RCEt_start[3][0]=68;
  RCEt_start[3][1]=78;
  RCEt_start[4][0]=88;
  RCEt_start[4][1]=98;
  RCEt_start[5][0]=108;
  RCEt_start[5][1]=118;
  RCEt_start[6][0]=128;
  RCEt_start[6][1]=138;

  RCTau_start[0][0]=148;
  RCTau_start[0][1]=149;
  RCTau_start[1][0]=150;
  RCTau_start[1][1]=151;
  RCTau_start[2][0]=152;
  RCTau_start[2][1]=153;
  RCTau_start[3][0]=154;
  RCTau_start[3][1]=155;
  RCTau_start[4][0]=156;
  RCTau_start[4][1]=157;
  RCTau_start[5][0]=158;
  RCTau_start[5][1]=159;
  RCTau_start[6][0]=160;
  RCTau_start[6][1]=161;

  RCOf_start[0][0]=162;
  RCOf_start[0][1]=163;
  RCOf_start[1][0]=164;
  RCOf_start[1][1]=165;
  RCOf_start[2][0]=166;
  RCOf_start[2][1]=167;
  RCOf_start[3][0]=168;
  RCOf_start[3][1]=169;
  RCOf_start[4][0]=170;
  RCOf_start[4][1]=171;
  RCOf_start[5][0]=172;
  RCOf_start[5][1]=173;
  RCOf_start[6][0]=174;
  RCOf_start[6][1]=175;
  
  
  //calo object index 0= ,order in the cable=  0  ,bits= 8,72 
  //calo object index 1= ,order in the cable=  1  ,bits= 16,73
  //calo object index 2= ,order in the cable=  4  ,bits= 40,76
  //calo object index 3= ,order in the cable=  5  ,bits= 48,77
  //calo object index 4= ,order in the cable=  2  ,bits= 24,74
  //calo object index 5= ,order in the cable=  3  ,bits= 32,75
  //calo object index 6= ,order in the cable=  6  ,bits= 56,78
  //calo object index 7= ,order in the cable=  7  ,bits= 64,79


  HFEt_start[0]=8; 
  HFEt_start[1]=16; 
  HFEt_start[2]=40;
  HFEt_start[3]=48;
  HFEt_start[4]=24;
  HFEt_start[5]=32;
  HFEt_start[6]=56;
  HFEt_start[7]=64;

  HFFg_start[0]=72;
  HFFg_start[1]=73; 
  HFFg_start[2]=76;
  HFFg_start[3]=77;
  HFFg_start[4]=74;
  HFFg_start[5]=75;
  HFFg_start[6]=78;
  HFFg_start[7]=79;

  IEEt_start[0]=80;
  IEEt_start[1]=90;
  IEEt_start[2]=100;
  IEEt_start[3]=110;

  IEReg_start[0]=86;
  IEReg_start[1]=96;
  IEReg_start[2]=106;
  IEReg_start[3]=116;
  
  IECard_start[0]=87;
  IECard_start[1]=97;
  IECard_start[2]=107; 
  IECard_start[3]=117;

  NEEt_start[0]=120;
  NEEt_start[1]=130;
  NEEt_start[2]=140;
  NEEt_start[3]=150;

  NEReg_start[0]=126;
  NEReg_start[1]=136;
  NEReg_start[2]=146;
  NEReg_start[3]=156;

  NECard_start[0]=127;
  NECard_start[1]=137;
  NECard_start[2]=147;
  NECard_start[3]=157;


  RCHad_start[0][0]=160;
  RCHad_start[0][1]=161;
  RCHad_start[1][0]=162;
  RCHad_start[1][1]=163;
  RCHad_start[2][0]=164;
  RCHad_start[2][1]=165;
  RCHad_start[3][0]=166;
  RCHad_start[3][1]=167;
  RCHad_start[4][0]=168;
  RCHad_start[4][1]=169;
  RCHad_start[5][0]=170;
  RCHad_start[5][1]=171;
  RCHad_start[6][0]=172;
  RCHad_start[6][1]=173;

}
