//
// Original Author:  Georgios Daskalakis , Georgios.Daskalakis@cern.ch
//         Created:  Fri Mar 30 18:15:12 CET 2007
//         
//
//
//


#include "PositionCorrector.h"
#include <cmath>



//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



double PositionCorrector::CORRX(int DeadCrystal, int DeadCrystalEta, int estimE, double estimX) {

  float pos=0.0;
  index=-1;

  int CR; CR = DeadCrystal;

  //===================================================================================
  if(CR<5 || CR>19 || CR==5 || CR==10 || CR==15 || CR==9 || CR==14 || CR==19 || CR==11 || CR==13){
    if(estimX<=1.0)index=0;
    if(estimX> 1.0)index=1;
  }
  //==================================================================================
  if( CR==6 || CR==8 ){
    if(estimX<=1.0)index=2;
    if(estimX> 1.0)index=3;
  }
  //==================================================================================
  if( CR==16 || CR==18 ){
    if(estimX<=0.5)index=4;
    if(estimX >0.5)index=5;
  }
  //==================================================================================
  if( CR==7 ){
    if(estimX<=0.1)index=6;
    if(estimX >0.1)index=7;
  }
  //==================================================================================
  if( CR==17 ){
    if(estimX<=0.5)index=8;
    if(estimX >0.5)index=9;
  }
  //==================================================================================
  if( CR==12 ){
    if(estimX<=0.0)index=10;
    if(estimX >0.0)index=11;
  }
  //==================================================================================


   switch(estimE) {
     case 50:
         correction_50(); break;
     default:
         correction_50(); break;
   }//end switch


   pos = estimX - ( a0+a1*estimX+a2*pow(estimX,2)+a3*pow(estimX,3) );    
   return pos;
    
}



//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


double PositionCorrector::CORRY(int DeadCrystal, int DeadCrystalEta, int estimE, double estimY) {

  float pos=0.0;
  index=-1;

  int CR; CR = DeadCrystal;

  //===================================================================================
  if(CR<5 || CR>19 || CR==5 || CR==10 || CR==15 || CR==9 || CR==14 || CR==19 || CR==7 || CR==17){
    if(estimY<=1.0)index=12;
    if(estimY>1.0)index=13;
  }
  //==================================================================================
  if( CR==6 || CR==16 ){
    if(estimY<=1.0)index=14;
    if(estimY>1.0)index=15;
  }
  //==================================================================================
  if( CR==8 || CR==18 ){
    if(estimY<=1.0)index=16;
    if(estimY>1.0)index=17;
  }
  //==================================================================================
  if( CR==11 ){
    if(estimY<=0.1)index=18;
    if(estimY>0.1)index=19;
  }
  //==================================================================================
  if( CR==13 ){
    if(estimY<=0.5)index=20;
    if(estimY>0.5)index=21;
  }
  //==================================================================================
  if( CR==12 ){
    if(estimY<=0.0)index=22;
    if(estimY>0.0)index=23;
  }
  //==================================================================================
  
  

   switch(estimE) {
     case 50:
         correction_50(); break;
     default:
         correction_50(); break;
   }//end switch

   pos = estimY - ( a0+a1*estimY+a2*pow(estimY,2)+a3*pow(estimY,3) );    
   return pos;
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


void PositionCorrector::correction_50(){
   switch(index) {
case 0:
a0=0.742593 ; a1=-2.443654 ; a2=-0.465629 ; a3=-0.021076; break;
case 1:
a0=0.723622 ; a1=-2.985130 ; a2=0.499481  ; a3=-0.019927; break;
case 2:
a0=1.523233 ; a1=-2.403963 ; a2=-0.509061 ; a3=-0.025730; break;
case 3:
a0=1.664726 ; a1=-3.356694 ; a2=0.549796 ;  a3=-0.022267; break;
case 4:
a0=-0.159428; a1=-2.913839 ; a2=-0.553717 ; a3=-0.026586; break;
case 5:
a0=-0.243322; a1=-2.982453;  a2=0.562717 ;  a3=-0.025937; break;
case 6:
a0=4.041620 ; a1=-4.430074;  a2=-1.810173;  a3=-0.195484; break;
case 7:
a0=3.865189 ; a1=-4.704274;  a2=0.821256 ;  a3=-0.039980; break;
case 8:
a0=-2.149538; a1=-4.287159;  a2=-0.860018;  a3=-0.049002; break;
case 9:
a0=-1.212509; a1=-7.920839;  a2=3.424293 ;  a3=-0.450545; break;
case 10:
a0=0.613761 ; a1=0.135842 ;  a2=-0.013459;  a3=0.000387 ; break;
case 11:
a0=0.583244 ; a1=0.118751 ;  a2=0.002993 ;  a3=0.000765 ; break;
case 12:
a0=-0.667742; a1=-2.345453;  a2=-0.207485;  a3=0.015167 ; break;
case 13:
a0=-0.802945; a1=-2.473333;  a2=0.509882 ;  a3=-0.025977; break;
case 14:
a0=0.176678 ; a1=-2.515074;  a2=-0.316820;  a3=0.006614 ; break;
case 15:
a0=0.053470 ; a1=-2.847765;  a2=0.569733 ;  a3=-0.029271; break;
case 16:
a0=-1.523717; a1=-2.677881;  a2=-0.263621;  a3=0.010810 ; break;
case 17:
a0=-1.728052; a1=-2.304793;  a2=0.518835 ;  a3=-0.028141; break;
case 18:
a0=1.726925 ; a1=-5.905270;  a2=-2.038082;  a3=-0.151892; break;
case 19:
a0=1.886734 ; a1=-3.997667;  a2=0.822161 ;  a3=-0.046878; break;
case 20:
a0=-3.924581; a1=-4.341018;  a2=-0.671026;  a3=-0.021481; break;
case 21:
a0=-4.464453; a1=-4.115324;  a2=1.873425 ;  a3=-0.152978; break;
case 22:
a0=-0.806418; a1=0.143467 ;  a2=-0.001304;  a3=0.000772 ; break;
case 23:
a0=-0.835848; a1=0.154018 ;  a2=0.017631 ;  a3=0.000075 ; break;
default:
  std::cout<<" Error, not valid Dead Channel Number, Abort"<<std::endl;
break;
}//end switch

}

