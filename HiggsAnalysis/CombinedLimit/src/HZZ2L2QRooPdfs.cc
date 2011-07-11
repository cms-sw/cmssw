#include <iostream>
#include <math.h>

//#include "../interface/RooDoubleCB.h"
//#include "../interface/RooFermi.h"
//#include "../interface/RooRelBW.h"
#include "../interface/HZZ2L2QRooPdfs.h"
#include "RooRealVar.h"
#include "RooRealConstant.h"

using namespace RooFit;

 ClassImp(RooDoubleCB) 

 RooDoubleCB::RooDoubleCB(){}

 RooDoubleCB::RooDoubleCB(const char *name, const char *title, 
		    RooAbsReal& _x,
		    RooAbsReal& _mean,
		    RooAbsReal& _width,
		    RooAbsReal& _alpha1,
		    RooAbsReal& _n1,
		    RooAbsReal& _alpha2,
		    RooAbsReal& _n2
		    ) :
   RooAbsPdf(name,title), 
   x("x","x",this,_x),
   mean("mean","mean",this,_mean),
   width("width","width",this,_width),
   alpha1("alpha1","alpha1",this,_alpha1),
   n1("n1","n1",this,_n1),
   alpha2("alpha2","alpha2",this,_alpha2),
   n2("n2","n2",this,_n2)
 { 
 } 


 RooDoubleCB::RooDoubleCB(const RooDoubleCB& other, const char* name) :  
   RooAbsPdf(other,name), 
   x("x",this,other.x),
   mean("mean",this,other.mean),
   width("width",this,other.width),
   alpha1("alpha1",this,other.alpha1),
   n1("n1",this,other.n1),
   alpha2("alpha2",this,other.alpha2),
   n2("n2",this,other.n2)

 { 
 } 

 double RooDoubleCB::evaluate() const 
 { 
   double A1 = pow(n1/fabs(alpha1),n1)*exp(-alpha1*alpha1/2);
   double A2 = pow(n2/fabs(alpha2),n2)*exp(-alpha2*alpha2/2);
   double B1 = n1/fabs(alpha1)-fabs(alpha1);
   double B2 = n2/fabs(alpha2)-fabs(alpha2);

   if((x-mean)/width>-alpha1 && (x-mean)/width<alpha2){
     return exp(-(x-mean)*(x-mean)/(2*width*width));
   }else if((x-mean)/width<-alpha1){
     return A1*pow(B1-(x-mean)/width,-n1);
   }else if((x-mean)/width>alpha2){
     return A2*pow(B2+(x-mean)/width,-n2);
   }else{
     cout << "ERROR evaluating range..." << endl;
     return 99;
   }
   
 } 

 ClassImp(RooFermi) 

 RooFermi::RooFermi(){}

 RooFermi::RooFermi(const char *name, const char *title, 
		      RooAbsReal& _x,
		      RooAbsReal& _cutOff,
		    RooAbsReal& _beta
		    ) :
   RooAbsPdf(name,title), 
   x("x","x",this,_x),
   cutOff("cutOff","cutOff",this,_cutOff),
   beta("beta","beta",this,_beta)
 { 
 } 


 RooFermi::RooFermi(const RooFermi& other, const char* name) :  
   RooAbsPdf(other,name), 
   x("x",this,other.x),
   cutOff("cutOff",this,other.cutOff),
   beta("beta",this,other.beta)

 { 
 } 



 double RooFermi::evaluate() const 
 { 
   return 1.0/(exp((cutOff-x)/beta)+1);
 } 

 ClassImp(RooRelBW) 

 RooRelBW::RooRelBW(){}

 RooRelBW::RooRelBW(const char *name, const char *title, 
		    RooAbsReal& _x,
		    RooAbsReal& _mean,
		    RooAbsReal& _width,
		    RooAbsReal& _n
		    ) :
   RooAbsPdf(name,title), 
   x("x","x",this,_x),
   mean("mean","mean",this,_mean),
   width("width","width",this,_width),
   n("n","n",this,_n)
 { 
 } 


 RooRelBW::RooRelBW(const RooRelBW& other, const char* name) :  
   RooAbsPdf(other,name), 
   x("x",this,other.x),
   mean("mean",this,other.mean),
   width("width",this,other.width),
   n("n",this,other.n)

 { 
 } 



 double RooRelBW::evaluate() const 
 { 
   return pow(x*x,n)/((x*x-mean*mean)*(x*x-mean*mean)+pow(x*x/(mean*mean),2*n)*mean*mean*width*width);
 } 

