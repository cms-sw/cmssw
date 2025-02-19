#include "TMultiLayerPerceptron.h"
#include <TTree.h>
#include <TCanvas.h>
#include <TGraph2D.h>
#include <TSystem.h>
#include <TMath.h>
#include <TGraphErrors.h>
#include <TProfile2D.h>
#include <TProfile.h>
#include <TPostScript.h>
#include <TLegend.h>
#include <TStyle.h>
#include <TRandom.h>
#include "getopt.h"
#include <map>
#include <iostream>
#include "string.h"
#include <sstream>
#include <unistd.h>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <time.h>


#include "Test_Central_1500.cxx"
#include "Test_Side_1500.cxx"
#include "Test_Corner_1500.cxx"
#include "TestPos_100.cxx"

//M5x5 must be ordered as :
  //Using Patrick Matrix:
  //22|10|09|11|17
  //21|04|01|07|16
  //20|03|00|06|15
  //23|05|02|08|18
  //24|13|12|14|19

using namespace std;

double CorrectDeadChannelsNN(double *M5x5Input){


  //From 07 May 07 we have as input a vector of :
  //So now, we have a vector which is ordered around the Maximum containement and which contains a dead channel as:
    //Filling of the vector : NxNaroundDC with N==11 Typo are possible...
    // 060 is Maximum containement which is in +/- 2 from DC
    //
     // 120 119 118 117 116 115 114 113 112 111 110
     // 109 108 107 106 105 104 103 102 101 100 099
     // 098 097 096 095 094 093 092 091 090 089 088
     // 087 086 085 084 083 082 081 080 079 078 077
     // 076 075 074 073 072 071 070 069 068 067 066
     // 065 064 063 062 061 060 059 058 057 056 055
     // 054 053 052 051 050 049 048 047 046 045 044
     // 043 042 041 040 039 038 037 036 035 034 033
     // 032 031 030 029 028 027 026 025 024 023 022
     // 021 020 019 018 017 016 015 014 013 012 011
     // 010 009 008 007 006 005 004 003 002 001 000
     //////////////////////////////////////////////

  //Conversion between input and input need by NN:
  double M5x5[25];
  M5x5[0]=M5x5Input[60];
  M5x5[1]=M5x5Input[71];
  M5x5[2]=M5x5Input[49];
  M5x5[3]=M5x5Input[61];
  M5x5[4]=M5x5Input[72];
  M5x5[5]=M5x5Input[50];
  M5x5[6]=M5x5Input[59];
  M5x5[7]=M5x5Input[70];
  M5x5[8]=M5x5Input[48];
  M5x5[9]=M5x5Input[82];
  M5x5[10]=M5x5Input[83];
  M5x5[11]=M5x5Input[81];
  M5x5[12]=M5x5Input[38];
  M5x5[13]=M5x5Input[39];
  M5x5[14]=M5x5Input[37];
  M5x5[15]=M5x5Input[58];
  M5x5[16]=M5x5Input[69];
  M5x5[17]=M5x5Input[80];
  M5x5[18]=M5x5Input[47];
  M5x5[19]=M5x5Input[36];
  M5x5[20]=M5x5Input[62];
  M5x5[21]=M5x5Input[73];
  M5x5[22]=M5x5Input[84];
  M5x5[23]=M5x5Input[51];
  M5x5[24]=M5x5Input[40];


  Test_Central_1500 *NNCentral =new Test_Central_1500();
  Test_Side_1500 *NNAdj =new Test_Side_1500();
  Test_Corner_1500 *NNCorner =new Test_Corner_1500();
 
  double in[6]; //Input of neurals networks
  double NN1=-1;//Output Neural Net

  double epsilon = 0.0000001;

  //variables
  bool Adjacent=false;
  bool Corner=false;
  bool Central=false;
  double SUM8=-1;
  double logX8=-1;
  double logY8=-1;
  double SUM24=-1;

  //First Find the position of the Dead Channel in 3x3 matrix
  int IndDeadCha=-1;
  for(int i =0 ; i< 9;i++){
    if(fabs(M5x5[i])<epsilon && IndDeadCha >0){/*cout<<" Problem 2 dead channels in sum9! Can not correct "<<std::endl;*/ return 0.0;}
    if(fabs(M5x5[i])<epsilon && IndDeadCha==-1)IndDeadCha=i;
  }



//   std::cout<<" THE DEAD CHANNEL IS : " << IndDeadCha <<std::endl;

//   std::cout<<"========================="<<std::endl;
//   std::cout<<M5x5[22]<<" "<<M5x5[10]<<" "<<M5x5[9] <<" "<<M5x5[11]<<" "<<M5x5[17]<<std::endl;
//   std::cout<<M5x5[21]<<" "<<M5x5[4] <<" "<<M5x5[1] <<" "<<M5x5[7] <<" "<<M5x5[16]<<std::endl;
//   std::cout<<M5x5[20]<<" "<<M5x5[3] <<" "<<M5x5[0] <<" "<<M5x5[6] <<" "<<M5x5[15]<<std::endl;
//   std::cout<<M5x5[23]<<" "<<M5x5[5] <<" "<<M5x5[2] <<" "<<M5x5[8] <<" "<<M5x5[18]<<std::endl;
//   std::cout<<M5x5[24]<<" "<<M5x5[13]<<" "<<M5x5[12]<<" "<<M5x5[14]<<" "<<M5x5[19]<<std::endl;
//   std::cout<<"========================="<<std::endl;


//  int isOK; isOK=0;
//  if(IndDeadCha == 0)isOK=1;
//  if(isOK == 0 )return 1000.0;



  int lineX1[3];
  int lineX2[3];
  int lineY1[3];
  int lineY2[3];
  int voisin1,voisin2,voisin3,voisin4;
  
  switch (IndDeadCha){
  case 0:
    lineX1[0]=3;lineX1[1]=4;lineX1[2]=5; 
    lineX2[0]=6;lineX2[1]=7;lineX2[2]=8; 
    lineY1[0]=1;lineY1[1]=4;lineY1[2]=7; 
    lineY2[0]=2;lineY2[1]=8;lineY2[2]=5;
    voisin1=1;voisin2=2;voisin3=3;voisin4=6;
    break;
  case 1:
    lineY1[0]=3;lineY1[1]=4;lineY1[2]=5; 
    lineY2[0]=6;lineY2[1]=7;lineY2[2]=8; 
    lineX2[0]=3;lineX2[1]=0;lineX2[2]=6; 
    lineX1[0]=2;lineX1[1]=8;lineX1[2]=5; 
    voisin1=0;voisin2=4;voisin3=7;voisin4=9;
    break;
  case 2:
    lineY2[0]=3;lineY2[1]=4;lineY2[2]=5; 
    lineY1[0]=6;lineY1[1]=7;lineY1[2]=8; 
    lineX1[0]=1;lineX1[1]=4;lineX1[2]=7; 
    lineX2[0]=3;lineX2[1]=0;lineX2[2]=6; 
    voisin1=0;voisin2=5;voisin3=8;voisin4=12;
    break;
  case 3:
    lineX2[0]=0;lineX2[1]=1;lineX2[2]=2; 
    lineX1[0]=6;lineX1[1]=7;lineX1[2]=8; 
    lineY1[0]=1;lineY1[1]=4;lineY1[2]=7; 
    lineY2[0]=2;lineY2[1]=8;lineY2[2]=5; 
    voisin1=0;voisin2=20;voisin3=4;voisin4=5;
    break;
  case 4:
    lineX1[0]=0;lineX1[1]=1;lineX1[2]=2; 
    lineX2[0]=6;lineX2[1]=7;lineX2[2]=8; 
    lineY1[0]=0;lineY1[1]=3;lineY1[2]=6; 
    lineY2[0]=2;lineY2[1]=8;lineY2[2]=5; 
    voisin1=1;voisin2=10;voisin3=3;voisin4=21;
    break;
  case 5:
    lineY1[0]=0;lineY1[1]=1;lineY1[2]=2; 
    lineY2[0]=6;lineY2[1]=7;lineY2[2]=8; 
    lineX2[0]=4;lineX2[1]=1;lineX2[2]=7; 
    lineX1[0]=0;lineX1[1]=3;lineX1[2]=6; 
    voisin1=13;voisin2=2;voisin3=3;voisin4=23;
    break;
  case 6:
    lineX1[0]=3;lineX1[1]=4;lineX1[2]=5; 
    lineX2[0]=0;lineX2[1]=1;lineX2[2]=2; 
    lineY1[0]=1;lineY1[1]=4;lineY1[2]=7; 
    lineY2[0]=5;lineY2[1]=2;lineY2[2]=8; 
    voisin1=15;voisin2=0;voisin3=7;voisin4=8;
    break;
  case 7:
    lineY2[0]=3;lineY2[1]=4;lineY2[2]=5; 
    lineY1[0]=0;lineY1[1]=1;lineY1[2]=2; 
    lineX1[0]=0;lineX1[1]=3;lineX1[2]=6; 
    lineX2[0]=2;lineX2[1]=8;lineX2[2]=5; 
    voisin1=11;voisin2=1;voisin3=6;voisin4=16;
    break;
  case 8:
    lineX2[0]=3;lineX2[1]=4;lineX2[2]=5; 
    lineX1[0]=0;lineX1[1]=1;lineX1[2]=2; 
    lineY2[0]=1;lineY2[1]=4;lineY2[2]=7; 
    lineY1[0]=0;lineY1[1]=3;lineY1[2]=6; 
    voisin1=6;voisin2=2;voisin3=18;voisin4=14;
    break;
  default:
    //    std::cout<<" Error, Dead Channel to far from main containement one, no correction is applied"<<std::endl;
    return 0.0;
    break;
  }//end switch

  float XL8=-50;
  float XR8=-50;
  float YL8=-50;
  float YR8=-50;

  XL8=0;
  XR8=0;
  for(int j=0;j<3;j++){XL8+=M5x5[lineX1[j]];}
  for(int j=0;j<3;j++){XR8+=M5x5[lineX2[j]];}
  YL8=0;
  YR8=0;
  for(int j=0;j<3;j++)YL8+=M5x5[lineY1[j]];
  for(int j=0;j<3;j++)YR8+=M5x5[lineY2[j]];

  SUM8 =0;
  for(int j=0;j<9;j++)if(j!=IndDeadCha)SUM8+=M5x5[j];

  float XR24=XR8+M5x5[11]+M5x5[14]+M5x5[15]+M5x5[16]+M5x5[17]+M5x5[18]+M5x5[19];
  float YR24=YR8+M5x5[18]+M5x5[23]+M5x5[24]+M5x5[13]+M5x5[12]+M5x5[14]+M5x5[19];

  float sum24 = 0.;
  for(int j=0;j<25;j++)if(j!=IndDeadCha)sum24+=M5x5[j];

  if(XR8 > 0 && XL8>0 && YL8 >0 && YR8>0 && SUM8>0 && XR24>0 &&YR24>0){
    logX8=TMath::Log(XL8/XR8);
    logY8=TMath::Log(YL8/YR8);
    SUM24=sum24;
    //Added 15 Janvier 2007 to get ride of energy dependance!
    SUM24 = SUM8/SUM24;
  
    Adjacent=false;
    Corner=false; 
    Central=false;

    //Il faut trouver le canal d'energie maximal, et par rapport a ce canal regarde l'energie dans les adjacent pour voir si nous sommes sur le central.
    float maxi;
    int IndMax;
    if(IndDeadCha!=0){
      maxi=M5x5[0];
      IndMax=0;
    }else{
      maxi=M5x5[1];
      IndMax=1;
    }
    for(int j=IndMax;j<9;j++){
      if(j!=IndDeadCha){
	if(M5x5[j] > maxi){IndMax=j;maxi=M5x5[j];}
      }
    }
    float Secmaxi=0;
    int IndSecMax=0;
    for(int j=IndSecMax;j<9;j++){
      if(j!=IndDeadCha && j!=IndMax){
	if(M5x5[j] > Secmaxi){IndSecMax=j;Secmaxi=M5x5[j];}
      }
    }

    in[0]=logX8;
    in[1]=logY8;
    in[2]=SUM24;

    //Define Adjacent/Central/Side with a Neural Net
    TestPos_100 *position=new TestPos_100();
    float indpos = position->value(0,in[0],in[1],in[2],(SUM8-maxi)/sum24,(maxi-Secmaxi)/maxi,M5x5[voisin1]/SUM8,M5x5[voisin2]/SUM8,M5x5[voisin3]/SUM8,M5x5[voisin4]/SUM8);
    if( indpos < 1.5){
      Central=true;
      Adjacent=false;
      Corner=false; 
    }else{
      if(indpos <2.5){
	Central=false;
	Adjacent=true;
	Corner=false; 
      }else{
	Central=false;
	Adjacent=false;
	Corner=true; 
      }
    }
    delete position;


    if(logX8!=-50 && logY8!=-50 && SUM8>0 && XR24>0 &&YR24>0){
  	if(Adjacent)NN1 =NNAdj->value(0,in[0],in[1],in[2]);
	if(Central)NN1 =NNCentral->value(0,in[0],in[1],in[2]);
	if(Corner)NN1 =NNCorner->value(0,in[0],in[1],in[2]);
    }else{
      NN1=0;
    }
  }

  return NN1*SUM8;
}
