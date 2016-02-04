 /***************************************************************************** 
  * Project: RooFit                                                           * 
  *                                                                           * 
  * Copyright (c) 2000-2005, Regents of the University of California          * 
  *                          and Stanford University. All rights reserved.    * 
  *                                                                           * 
  * Redistribution and use in source and binary forms,                        * 
  * with or without modification, are permitted according to the terms        * 
  * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             * 
  *****************************************************************************/ 

 // -- CLASS DESCRIPTION [PDF] -- 
 // Your description goes here... 

 #include "Riostream.h" 

 #include "RooCBExGaussShape.h" 
 #include "RooAbsReal.h" 
 #include "RooAbsCategory.h" 

 ClassImp(RooCBExGaussShape) 

 RooCBExGaussShape::RooCBExGaussShape(const char *name, const char *title, 
				      RooAbsReal& _m, //x
				      RooAbsReal& _m0, //p1
				      RooAbsReal& _sigma, //p2
				      RooAbsReal& _alpha, //p3
				      RooAbsReal& _n, //p4
				      RooAbsReal& _sigma_2, //sigma
				      RooAbsReal& _frac //frac
) :
   RooAbsPdf(name,title), 
   m("m","m",this,_m),
    m0(" m0"," m0",this,_m0),
    sigma(" sigma"," sigma",this,_sigma),
    alpha(" alpha"," alpha",this,_alpha),
    n(" n"," n",this,_n),
    sigma_2(" sigma_2"," sigma_2",this,_sigma_2),
    frac(" frac"," frac",this,_frac) 
{ 
 } 


 RooCBExGaussShape::RooCBExGaussShape(const RooCBExGaussShape& other, const char* name) :  
   RooAbsPdf(other,name), 
    m("m",this,other.m),
    m0(" m0",this,other. m0),
    sigma(" sigma",this,other. sigma),
    alpha(" alpha",this,other. alpha),
    n(" n",this,other. n),
    sigma_2(" sigma_2",this,other. sigma_2),
    frac(" frac",this,other. frac)
 { 
 } 



 Double_t RooCBExGaussShape::evaluate() const 
 { 
   // ENTER EXPRESSION IN TERMS OF VARIABLE ARGUMENTS HERE 


   Double_t t = (m-m0)/sigma;
   Double_t t0 = (m-m0)/sigma_2;
   if (alpha < 0){ 
     t = -t;
     t0 = -t0;
   }

   Double_t absAlpha = fabs((Double_t)alpha);

   if (t >= -absAlpha) {
   //if ((t+t0)/2.0 >= -absAlpha) {
     return frac*exp(-0.5*t*t) + (1.0-frac)*exp(-0.5*t0*t0);
     //return 1.0*(exp(-0.5*t*t));// + exp(-0.5*t0*t0));
   }
   else {
     Double_t a =  TMath::Power(n/absAlpha,n)*exp(-0.5*absAlpha*absAlpha);
     Double_t b= n/absAlpha - absAlpha; 

     return a/TMath::Power(b - t, n);
     //return a/TMath::Power(b - (t+t0)/2.0, n);
   }




   //return 1.0 ; 
 } 



