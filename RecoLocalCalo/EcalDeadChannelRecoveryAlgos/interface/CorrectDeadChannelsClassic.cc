//
// Original Author:  Georgios Daskalakis , Georgios.Daskalakis@cern.ch
//         Created:  Fri Mar 30 18:15:12 CET 2007
//         
//
//
//
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

#include "PositionCorrector.cxx"
#include "SplineCorrector.cxx"
#include "ExponCorrector.cxx"



using namespace std;

double CorrectDeadChannelsClassic(double *M11x11Input, const int DCeta){

  //std::cout<<"Inside Correction Function ...."<< std::endl;

  PositionCorrector *PosCorr = new PositionCorrector();
  SplineCorrector *SplCorr   = new SplineCorrector();
  ExponCorrector *ExpCorr    = new ExponCorrector();




  double epsilon = 0.0000001;
  float NEWx,NEWy;
  float estimX,estimY,SUMlogFR;
  float SUM24;
  float crE[25];
  
   

  crE[0]=M11x11Input[40];
  crE[1]=M11x11Input[51];
  crE[2]=M11x11Input[62];
  crE[3]=M11x11Input[73];
  crE[4]=M11x11Input[84];

  crE[5]=M11x11Input[39];
  crE[6]=M11x11Input[50];
  crE[7]=M11x11Input[61];
  crE[8]=M11x11Input[72];
  crE[9]=M11x11Input[83];

  crE[10]=M11x11Input[38];
  crE[11]=M11x11Input[49];
  crE[12]=M11x11Input[60];
  crE[13]=M11x11Input[71];
  crE[14]=M11x11Input[82];

  crE[15]=M11x11Input[37];
  crE[16]=M11x11Input[48];
  crE[17]=M11x11Input[59];
  crE[18]=M11x11Input[70];
  crE[19]=M11x11Input[81];

  crE[20]=M11x11Input[36];
  crE[21]=M11x11Input[47];
  crE[22]=M11x11Input[58];
  crE[23]=M11x11Input[69];
  crE[24]=M11x11Input[80];


  std::cout<<"Inside Correction Function : Dead Channel ETA" << DCeta << std::endl;
  //revert crystal matrix because of negative eta
  if(DCeta<0){

  crE[0]=M11x11Input[36];
  crE[1]=M11x11Input[47];
  crE[2]=M11x11Input[58];
  crE[3]=M11x11Input[69];
  crE[4]=M11x11Input[80];

  crE[5]=M11x11Input[37];
  crE[6]=M11x11Input[48];
  crE[7]=M11x11Input[59];
  crE[8]=M11x11Input[70];
  crE[9]=M11x11Input[81];

  crE[10]=M11x11Input[38];
  crE[11]=M11x11Input[49];
  crE[12]=M11x11Input[60];
  crE[13]=M11x11Input[71];
  crE[14]=M11x11Input[82];

  crE[15]=M11x11Input[39];
  crE[16]=M11x11Input[50];
  crE[17]=M11x11Input[61];
  crE[18]=M11x11Input[72];
  crE[19]=M11x11Input[83];

  crE[20]=M11x11Input[40];
  crE[21]=M11x11Input[51];
  crE[22]=M11x11Input[62];
  crE[23]=M11x11Input[73];
  crE[24]=M11x11Input[84];


  }
  




  float SUMuu = 0.0;
  float SUMu  = 0.0;
  float SUMdd = 0.0;
  float SUMd  = 0.0;
  float SUMll = 0.0;
  float SUMl  = 0.0;  
  float SUMrr = 0.0;
  float SUMr  = 0.0;

  SUMuu  = crE[4]  + crE[9]  + crE[14] + crE[19] + crE[24];
  SUMu   = crE[3]  + crE[8]  + crE[13] + crE[18] + crE[23];
  SUMd   = crE[1]  + crE[6]  + crE[11] + crE[16] + crE[21];
  SUMdd  = crE[0]  + crE[5]  + crE[10] + crE[15] + crE[20];
  
  SUMll  = crE[0]  + crE[1]  + crE[2]  + crE[3]  + crE[4];
  SUMl   = crE[5]  + crE[6]  + crE[7]  + crE[8]  + crE[9];
  SUMr   = crE[15] + crE[16] + crE[17] + crE[18] + crE[19];
  SUMrr  = crE[20] + crE[21] + crE[22] + crE[23] + crE[24];

  /*
  std::cout<<"================================================================="<<std::endl;
  std::cout<<crE[4]<<setw(12)<<crE[9]<<setw(12)<<crE[14]<<setw(12)<<crE[19]<<setw(12)<<crE[24]<<std::endl;
  std::cout<<crE[3]<<setw(12)<<crE[8]<<setw(12)<<crE[13]<<setw(12)<<crE[18]<<setw(12)<<crE[23]<<std::endl;
  std::cout<<crE[2]<<setw(12)<<crE[7]<<setw(12)<<crE[12]<<setw(12)<<crE[17]<<setw(12)<<crE[22]<<std::endl;
  std::cout<<crE[1]<<setw(12)<<crE[6]<<setw(12)<<crE[11]<<setw(12)<<crE[16]<<setw(12)<<crE[21]<<std::endl;
  std::cout<<crE[0]<<setw(12)<<crE[5]<<setw(12)<<crE[10]<<setw(12)<<crE[15]<<setw(12)<<crE[20]<<std::endl;
  std::cout<<"================================================================="<<std::endl;
  */


////////////////////////////////////////////////////////////////////

  float XLOW[50],XHIG[50],YLOW[50],YHIG[50];
  int NSUBS = 25;
  for(int ix=0; ix<5 ; ix++){
    for(int iy=0; iy<5 ; iy++){
      int isub= ix*5+iy ; 
      
      XLOW[isub]= -10.0 +float(ix)*4.0;
      XHIG[isub]= XLOW[isub] + 4.0;
      YLOW[isub]= -10.0 +float(iy)*4.0;;
      YHIG[isub]= YLOW[isub] + 4.0; 
      
    }
  }




////////////////////////////////////////////////////////////////////



  //First Find the position of the Dead Channel in 3x3 matrix
  int DeadCR = -1;
  for(int ix=1; ix<4 ; ix++){
    for(int iy=1; iy<4 ; iy++){
      int idx = ix*5+iy; 
      if(fabs(crE[idx])<epsilon && DeadCR >0){std::cout<<" Problem 2 dead channels in sum9! Can not correct ... I return 0.0"<<std::endl; return 0.0;}
      if(fabs(crE[idx])<epsilon && DeadCR==-1)DeadCR=idx;
    }
  }


  //std::cout<<" THE DEAD CHANNEL IS : "<< DeadCR <<std::endl;
  SUM24=0.0;
  for(int j=0;j<25;j++)SUM24+=crE[j];
  if(DeadCR == -1){std::cout<<" No Dead Channel in 3x3 !   I don't correct ... I return 0.0 ....look S25 = "<<SUM24<<std::endl; return 0.0;}



  // CR=12 must be the maximum in 5x5 !!!
  int CisMax; CisMax=-1;
  float MaxEin5x5; MaxEin5x5=0.0;
  for(int ic=0;ic<24;ic++){
    if(crE[ic]>MaxEin5x5){
      MaxEin5x5 = crE[ic];
      CisMax = ic ;
    }
  }
  if(CisMax != 12){
    std::cout<<"ERROR ----> Central has NOT the MAX energy in 5x5 ... I return 0.0"<<std::endl;
    return 0.0;
  }
  


  //AVOID BREM or OTHER DEPOSITIONS IN 5x5
  //std::cout<<" CHECK FOR BREM or UNUSUAL PATTERNS  ... " <<std::endl;
  //std::cout<<" SUMuu="<<SUMuu<<",SUMu="<<SUMu<<",SUMd="<<SUMd <<",SUMdd="<<SUMdd <<",SUMll="<<SUMll <<",SUMl="<<SUMl <<",SUMrr="<<SUMrr <<",SUMr="<<SUMr <<std::endl;
  if(DeadCR==6  && (SUMuu>SUMu || SUMrr>SUMr || SUMll>3.0*SUMr || SUMdd>3.0*SUMu)){std::cout<<"Unusual Pattern in 6 I return 0.0"<<std::endl; return 0.0;}
  if(DeadCR==8  && (SUMdd>SUMd || SUMrr>SUMr || SUMll>3.0*SUMr || SUMuu>3.0*SUMd)){std::cout<<"Unusual Pattern in 8 I return 0.0"<<std::endl; return 0.0;}
  if(DeadCR==16 && (SUMuu>SUMu || SUMll>SUMl || SUMrr>3.0*SUMl || SUMdd>3.0*SUMu)){std::cout<<"Unusual Pattern in 16 I return 0.0"<<std::endl; return 0.0;}
  if(DeadCR==18 && (SUMdd>SUMd || SUMll>SUMl || SUMrr>3.0*SUMl || SUMuu>3.0*SUMd)){std::cout<<"Unusual Pattern in 18 I return 0.0"<<std::endl; return 0.0;}

  if(DeadCR==7  && (SUMuu>SUMu || SUMdd>SUMd || SUMrr>SUMr)){std::cout<<"Unusual Pattern in 7 I return 0.0"<<std::endl; return 0.0;}
  if(DeadCR==17 && (SUMuu>SUMu || SUMdd>SUMd || SUMll>SUMl)){std::cout<<"Unusual Pattern in 17 I return 0.0"<<std::endl; return 0.0;}
  if(DeadCR==11 && (SUMll>SUMl || SUMrr>SUMr || SUMuu>SUMu)){std::cout<<"Unusual Pattern in 11 I return 0.0"<<std::endl; return 0.0;}
  if(DeadCR==13 && (SUMll>SUMl || SUMrr>SUMr || SUMdd>SUMd)){std::cout<<"Unusual Pattern in 13 I return 0.0"<<std::endl; return 0.0;}


  //std::cout<<" NO BREM OR OTHER DEPOSITIONS IN THIS PATTERN --- I continue ... " <<std::endl;

  SUM24=0.0;
  for(int j=0;j<25;j++)if(j!=DeadCR)SUM24+=crE[j];
  // std::cout<<" THE Enorm is : "<< SUM24 <<std::endl;


  //CHECK IF IT IS REALLY THE CORRECT DEAD CHANNEL
  // This will be included in the next version



   ////////////////////////////////////////////////    
   ////////////////////////////////////////////////
   //    	        Estimate X,Y
   ////////////////////////////////////////////////
   ////////////////////////////////////////////////

    NEWx=0.0; NEWy=0.0;			    
    
    
    estimX=0.0;
    estimY=0.0;
    SUMlogFR=0.0;
    for(int ix=0; ix<5 ; ix++){
      for(int iy=0; iy<5 ; iy++){
	int idx = ix*5+iy; 
	
	float xpos = 20.0*(float(ix)-2.0);
	float ypos = 20.0*(float(iy)-2.0);
	
	if(idx != DeadCR){
	  
	  // weight definition
	  float w = crE[idx]/SUM24;
	  
	  if(w<0.0)w=0.0;
	  SUMlogFR = SUMlogFR + w;
	  
	  estimX = estimX + xpos*w;
	  estimY = estimY + ypos*w;
	}
	
      } // iy loop
    }// ix loop
    
    estimX = estimX/SUMlogFR;	    	    
    estimY = estimY/SUMlogFR;	 
    

       NEWx = PosCorr->CORRX(DeadCR,0,50,estimX);
       NEWy = PosCorr->CORRY(DeadCR,0,50,estimY);  


       //std::cout<<" FINAL X,Y calculation: DC , estimX, estimY, NEWx, NEWy : "<<DeadCR<<" "<<estimX<<" "<<estimY<<" "<< NEWx<<" "<<NEWy<<std::endl;


    if(DeadCR==7  && (estimX>7.0 || estimX< -2.0 || estimY>10.0 || estimY<-9.0) ){std::cout<<"DC=7  Position OUT of LIMIT I return 0.0"<<std::endl; return 0.0;}
    if(DeadCR==17 && (estimX>2.5 || estimX< -8.0 || estimY>10.0 || estimY<-8.0) ){std::cout<<"DC=17 Position OUT of LIMIT I return 0.0"<<std::endl; return 0.0;}
    if(DeadCR==11 && (estimX>8.0 || estimX<-10.0 || estimY> 9.0 || estimY<-4.0) ){std::cout<<"DC=11 Position OUT of LIMIT I return 0.0"<<std::endl; return 0.0;}
    if(DeadCR==13 && (estimX>8.0 || estimX<-10.0 || estimY> 4.5 || estimY<-8.0) ){std::cout<<"DC=13 Position OUT of LIMIT I return 0.0"<<std::endl; return 0.0;}

    if(DeadCR==12 && (estimX>18.0 || estimX<-18.0 || estimY> 17.0 || estimY<-17.0) ){std::cout<<"DC=12 Position OUT of LIMIT I return 0.0"<<std::endl; return 0.0;}

    if(DeadCR==6  && (estimX>8.0 || estimX< -9.0 || estimY>10.0 || estimY<-8.0) ){std::cout<<"DC=6  Position OUT of LIMIT I return 0.0"<<std::endl; return 0.0;}
    if(DeadCR==8  && (estimX>8.0 || estimX< -9.0 || estimY> 9.0 || estimY<-8.5) ){std::cout<<"DC=8  Position OUT of LIMIT I return 0.0"<<std::endl; return 0.0;}
    if(DeadCR==16 && (estimX>7.0 || estimX<-10.0 || estimY> 9.0 || estimY<-8.0) ){std::cout<<"DC=16 Position OUT of LIMIT I return 0.0"<<std::endl; return 0.0;}
    if(DeadCR==18 && (estimX>8.0 || estimX< -9.0 || estimY> 9.0 || estimY<-8.0) ){std::cout<<"DC=18 Position OUT of LIMIT I return 0.0"<<std::endl; return 0.0;}

   ///////////////////////////////////////////////////
   ///////////////////////////////////////////////////
   ///////////////////////////////////////////////////



  





     // INFO from SPLINE
     float RECOfrDcr = 0.0;
     RECOfrDcr = SplCorr->value(DeadCR,0,50,NEWx,NEWy);



    // RESOLUTIONS PER AREA
    for (int isub=0 ; isub<NSUBS ; isub++){
      if( NEWx>XLOW[isub] && NEWx<XHIG[isub] && NEWy>YLOW[isub] && NEWy<YHIG[isub] ){
	
	
	//TEST THE IDEA OF EXPs
	if(DeadCR==6  && (isub==0  || isub ==1  || isub==5)  ){
	  RECOfrDcr = ExpCorr->value(DeadCR,0,50,isub,NEWx,NEWy);
	}
	if(DeadCR==8  && (isub==4  || isub ==3  || isub==9)  ){
	  RECOfrDcr = ExpCorr->value(DeadCR,0,50,isub,NEWx,NEWy);
	}
	if(DeadCR==16 && (isub==15 || isub ==21 || isub==20) ){
	  RECOfrDcr = ExpCorr->value(DeadCR,0,50,isub,NEWx,NEWy);
	}
	if(DeadCR==18 && (isub==19 || isub ==23 || isub==24) ){
	  RECOfrDcr = ExpCorr->value(DeadCR,0,50,isub,NEWx,NEWy);
	}
	if(DeadCR==7 ||  DeadCR==11 || DeadCR==12 || DeadCR==13 || DeadCR==17 ){
	  RECOfrDcr = ExpCorr->value(DeadCR,0,50,isub,NEWx,NEWy);
	}
	
	
      }
    }
    
    
    double ESTIMATED_ENERGY = RECOfrDcr*SUM24;
    std::cout<<" THE ESTIMATED RECOfrDcr is : "<< RECOfrDcr <<std::endl;
    std::cout<<" THE ESTIMATED ENERGY is : "<<  ESTIMATED_ENERGY <<std::endl;

    if(RECOfrDcr<0.0){std::cout<<"NEGATIVE RECOfrDr I Return 0.0"<<std::endl; RECOfrDcr=0.0;}
    if( (DeadCR==6 || DeadCR==8  || DeadCR==16 || DeadCR==18) && RECOfrDcr>0.20){std::cout<<"Fraction OUT of LIMIT I return 0.0"<<std::endl; return 0.0;}
    if( (DeadCR==7 || DeadCR==17 || DeadCR==11 || DeadCR==13) && RECOfrDcr>1.00){std::cout<<"Fraction OUT of LIMIT I return 0.0"<<std::endl; return 0.0;}
    if( DeadCR==12 && RECOfrDcr>5.00){std::cout<<"Fraction OUT of LIMIT I return 0.0"<<std::endl; return 0.0;}


////////////////////////////////////////////////////////////////////
// Using the findings above I build the position and the energy again!
////////////////////////////////////////////////////////////////////



  // THIS IS THE FINAL ANSWER
  return ESTIMATED_ENERGY;
}



