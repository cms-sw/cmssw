#ifndef SaturationFcn_h
#define SaturationFcn_h
                                                               
#include "Minuit2/FCNBase.h"                                                  
#include <vector>

using namespace ROOT::Minuit2;

class SaturationFcn : public FCNBase{ 

 public:

 SaturationFcn(){}

 ~SaturationFcn(){}


 void set_data(){
   N=25;
   
   //   float x[14]={22.4, 44.8, 67.2, 89.6, 112, 134.4, 156.8, 179.2,201.6, 224.0, 246.4, 268.8, 291.2, 313.6};
   //   float y[14]={1974.28, 2155.27, 2335.6, 2513.77, 2686.75, 2854.5,3016.83, 3173.9, 3244.83, 3245.4, 3245.02, 3245.78, 3245.08, 3245.53};

   float x[25],y[25];

   for(int i=0;i<N;i++){
     x[i]=(*charge_ptr)[i];
     y[i]=(*adc_ptr)[i];
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
 
 double datx[14],daty[14];
 int N;
 
}; 

#endif
