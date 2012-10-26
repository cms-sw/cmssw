#include <iostream>
#include <math.h>
#include "TMath.h"

//#include "../interface/RooDoubleCB.h"
//#include "../interface/RooFermi.h"
//#include "../interface/RooRelBW.h"
#include "../interface/HZZ2L2QRooPdfs.h"
#include "RooRealVar.h"
#include "RooRealConstant.h"

using namespace RooFit;
using namespace std; 

ClassImp(RooCB)

  RooCB::RooCB(){}

RooCB::RooCB(const char *name, const char *title,
	     RooAbsReal& _x,
	     RooAbsReal& _mean,
	     RooAbsReal& _width,
	     RooAbsReal& _alpha,
	     RooAbsReal& _n,
              RooAbsReal& _theta
	     ) :
  RooAbsPdf(name,title),
  x("x","x",this,_x),
  mean("mean","mean",this,_mean),
  width("width","width",this,_width),
  alpha("alpha","alpha",this,_alpha),
  n("n","n",this,_n),
  theta("theta","theta",this,_theta)
{
}

RooCB::RooCB(const RooCB& other, const char* name) :
  RooAbsPdf(other,name),
  x("x",this,other.x),
  mean("mean",this,other.mean),
  width("width",this,other.width),
  alpha("alpha",this,other.alpha),
  n("n",this,other.n),
  theta("theta",this,other.theta)
{
}

double RooCB::evaluate() const
{
  double a = cos(theta)*alpha - sin(theta)*width;
  double w = sin(theta)*alpha + cos(theta)*width;

  double t = (x-mean)/w;
  if(a<0) t = -t;

  double absa = fabs((double)a);

  double A = TMath::Power(n/absa,n)*exp(-0.5*absa*absa);
  double B = n/absa-absa;

  if(t >= -absa){
    return exp(-0.5*t*t);
  }else{
    return A/TMath::Power(B-t,n);
  }
}


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
   double t = (x-mean)/width;
   if(t>-alpha1 && t<alpha2){
     return exp(-0.5*t*t);
   }else if(t<-alpha1){
     double A1 = pow(n1/fabs(alpha1),n1)*exp(-alpha1*alpha1/2);
     double B1 = n1/fabs(alpha1)-fabs(alpha1);
     return A1*pow(B1-t,-n1);
   }else if(t>alpha2){
     double A2 = pow(n2/fabs(alpha2),n2)*exp(-alpha2*alpha2/2);
     double B2 = n2/fabs(alpha2)-fabs(alpha2);
     return A2*pow(B2+t,-n2);
   }else{
     cout << "ERROR evaluating range..." << endl;
     return 99;
   }
   
 } 

 Int_t RooDoubleCB::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* range) const 
 {
   if (matchArgs(allVars,analVars,x)) return 1;
   return 0;
 }

 Double_t RooDoubleCB::analyticalIntegral(Int_t code, const char* rangeName) const 
 {
   assert(code==1) ;
 
   double central=0;
   double left=0;
   double right=0;
 
   static const Double_t root2 = sqrt(2) ;
   static const Double_t rootPiBy2 = sqrt(atan2(0.0,-1.0)/2.0);
   Double_t xscale = root2*width;
 
   //compute gaussian contribution
   double central_low =max(x.min(rangeName),mean - alpha1*width );
   double central_high=min(x.max(rangeName),mean + alpha2*width );
   if(central_low < central_high) // is the gaussian part in range?
     central = rootPiBy2*width*(TMath::Erf((central_high-mean)/xscale)-TMath::Erf((central_low-mean)/xscale));
 
   //compute left tail;
   double A1 = pow(n1/fabs(alpha1),n1)*exp(-alpha1*alpha1/2);
   double B1 = n1/fabs(alpha1)-fabs(alpha1);
 
   double left_low=x.min(rangeName);
   double left_high=min(x.max(rangeName),mean - alpha1*width);
   if(left_low < left_high){ //is the left tail in range?
     if(fabs(n1-1.0)>1.e-5)
       left = A1/(-n1+1.0)*width*(pow(B1-(left_low-mean)/width,-n1+1.)-pow(B1-(left_high-mean)/width,-n1+1.));
     else
       left = A1*width*(log(B1-(left_low-mean)/width) - log(B1-(left_high-mean)/width) );
   }
 
   //compute right tail;
   double A2 = pow(n2/fabs(alpha2),n2)*exp(-alpha2*alpha2/2);
   double B2 = n2/fabs(alpha2)-fabs(alpha2);
 
   double right_low=max(x.min(rangeName),mean + alpha2*width);
   double right_high=x.max(rangeName);
   if(right_low < right_high){ //is the right tail in range?
     if(fabs(n2-1.0)>1.e-5)
       right = A2/(-n2+1.0)*width*(pow(B2+(right_high-mean)/width,-n2+1.)-pow(B2+(right_low-mean)/width,-n2+1.));
     else
       right = A2*width*(log(B2+(right_high-mean)/width) - log(B2+(right_low-mean)/width) );
   }
     
   return left+central+right;
 
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


ClassImp(Triangle)

  Triangle::Triangle(){}

Triangle::Triangle(const char *name, const char *title,                
		   RooAbsReal& _m,
		   RooAbsReal& _start,
		   RooAbsReal& _turn,
		   RooAbsReal& _stop
		   ):
  RooAbsPdf(name, title),
  m("m", "Dependent", this, _m),
  start("start","start",this,_start),
  turn("turn","turn",this,_turn),
  stop("stop","stop",this,_stop)
{
}

Triangle::Triangle(const Triangle& other, const char* name) :
  RooAbsPdf(other, name), m("m", this, other.m),start("start", this, other.start), turn("turn", this, other.turn), stop("stop", this, other.stop)
{
}

Double_t Triangle::evaluate() const 
{
  //std::cout << m << " "<<1.+(start-m)/turn << " " << 1+(turn-m)/stop << std::endl;
  if(m<turn  && m > turn+start)
    return 1.+(turn-m)/start;
  if(m>=turn && m < turn+stop)
    return 1.+(turn-m)/stop;
  
  return 0;
}


Int_t Triangle::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* range) const 
{
  if (matchArgs(allVars,analVars,m)) return 1;
  return 0;
}

Double_t Triangle::analyticalIntegral(Int_t code, const char* rangeName) const 
{

  // WARNING, ASSSUMES TURN TO BE IN INTERVAL
  assert(code==1) ;
  //whole triangle
  Double_t sumleft = sqrt(1+ (turn+start)*(turn+start) ) ;
  Double_t sumright= sqrt(1+ (turn+stop)*(turn+stop) );


  if(m.min() < turn+start)// correct for left missing bit
    sumleft -= sumleft*(m.min()-(turn+start))/fabs(start);


  if(m.max() > turn+stop)// correct for right missing bit
    sumright -= sumright*(turn+stop -m.max())/fabs(stop);

  

  return sumleft+sumright;    
}




ClassImp(RooLevelledExp)

  RooLevelledExp::RooLevelledExp(){}

RooLevelledExp::RooLevelledExp(const char *name, const char *title,
			       RooAbsReal& _x,
			       RooAbsReal& _sigma, 
			       RooAbsReal& _alpha,
			       RooAbsReal& _m,
			       RooAbsReal& _theta):
  RooAbsPdf(name,title),
  x("x","x",this,_x),
  sigma("sigma","sigma",this,_sigma),
  alpha("alpha","alpha",this,_alpha),
  m("m","m",this,_m),
  //  k("k","k",this,_k),
  theta("theta","theta",this,_theta)
{
}

RooLevelledExp::RooLevelledExp(const RooLevelledExp& other, const char* name) :
  RooAbsPdf(other,name),
  x("x",this,other.x),
  sigma("sigma",this,other.sigma),
  alpha("alpha",this,other.alpha),
  m("m",this,other.m),
  theta("theta",this,other.theta)
{
}

double RooLevelledExp::evaluate() const
{
  double res=0.0;
  double s = cos(theta)*sigma - sin(theta)*alpha;
  double a = sin(theta)*sigma + cos(theta)*alpha;
    
  //original
  double t = fabs(x-m);
  double den = (s + a*t);
  res=exp(-1.0*t/den);
  

  return res;
}
