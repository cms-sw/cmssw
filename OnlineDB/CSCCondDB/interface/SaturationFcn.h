#ifndef SaturationFcn_h
#define SaturationFcn_h
                                                               
#include "Minuit2/FCNBase.h"  
#include <vector>

using namespace ROOT::Minuit2;

class SaturationFcn : public FCNBase{ 

 public:

 SaturationFcn(){}

 ~SaturationFcn(){}


 void set_data(int N,float *charge_ptr,float *adc_ptr){
   N=24;
   float x[24],y[24];

   for(int i=0;i<N;i++){
     x[i]=charge_ptr[i];
     y[i]=adc_ptr[i];
     //std::cout<<"This is charge: "<<x[i]<<" and adc: "<<y[i]<<std::endl;
     datx[i]=x[i];
     daty[i]=y[i];
     printf("%d  datx daty %f %f \n",i,datx[i],daty[i]);
   }
   x3start=(y[4]*x[1]-y[1]*x[4])/(x[1]-x[4]);
   x0start=daty[13]-x3start;
   x1start=(y[4]-y[1])/(x[4]-x[1])/x0start;
   x2start=20.;
   printf(" x0-2start %f %f %f %f\n",x0start,x1start,x2start,x3start);
 }
 
 virtual double Up() const {return 1.;}
 
 virtual double operator()(const std::vector<double>& x) const {
   double chisq = 0.0;  
  for(int i=0;i<N;i++){
    double val=1.0+pow(x[1]*datx[i],x[2]);
    double val2=1.0/x[2];
    val=x[0]*x[1]*datx[i]/pow(val,val2);
    double tmp=(daty[i]-x[3]-val);
    printf(" dat: %d %f %f %f %f \n",i,datx[i],daty[i]-x[3],val,tmp);
    chisq=chisq+tmp*tmp;
  }
  printf("x0-3 %f %f %f %fchisq %f \n",x[0],x[1],x[2],x[3],chisq);
  return chisq; 
 }

 double x0start;
 double x1start;
 double x2start;
 double x3start;
 
 private:
 
 double datx[24],daty[24];
 int N;
}; 

#endif
