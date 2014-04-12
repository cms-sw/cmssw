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

 #include "RooPowLaw.h" 
 #include "RooAbsReal.h" 
 #include "RooAbsCategory.h" 

 ClassImp(RooPowLaw) 

 RooPowLaw::RooPowLaw(const char *name, const char *title, 
				      RooAbsReal& _m,
				      RooAbsReal& _alpha
) :
   RooAbsPdf(name,title), 
   m("m","m",this,_m),
    alpha(" alpha"," alpha",this,_alpha)
{ 
 } 


 RooPowLaw::RooPowLaw(const RooPowLaw& other, const char* name) :  
   RooAbsPdf(other,name), 
    m("m",this,other.m),
    alpha(" alpha",this,other. alpha)
 { 
 } 



 Double_t RooPowLaw::evaluate() const 
 { 
   // ENTER EXPRESSION IN TERMS OF VARIABLE ARGUMENTS HERE 


   if(m <= 0.0 ){ return 1000.0; }
   else{ return TMath::Power(m, alpha); }
 } 



