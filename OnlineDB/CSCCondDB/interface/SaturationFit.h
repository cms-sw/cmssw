#include "Minuit2/VariableMetricMinimizer.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnPrint.h"

#include <vector>
#include <cmath>
#include "OnlineDB/CSCCondDB/interface/SaturationFcn.h"

class SaturationFit{ 

 public:

  SaturationFit(int N,float *charge_ptr,float *adc_ptr, float *u0_ptr, float *u1_ptr, float *u2_ptr, float*u3_ptr){
   double u[3],sigma[3],chisq;
   VariableMetricMinimizer* pMinimizer=new VariableMetricMinimizer() ; 
   SaturationFcn* pFcn=new SaturationFcn();
   pFcn->set_data(N,charge_ptr,adc_ptr);
   std::vector<double> par(4,0);
   std::vector<double> err(4,0);
   //printf(" about to fill pars \n");
   par[0]=pFcn->x0start;
   par[1]=pFcn->x1start;
   par[2]=pFcn->x2start;
   par[3]=pFcn->x3start;
   //printf(" pars: %f %f %f %f \n",par[0],par[1],par[2],par[3]);
   err[0]=20.0;
   err[1]=0.0001;
   err[2]=1.0;
   err[3]=20.0;
   FunctionMinimum fmin = pMinimizer->Minimize(*pFcn, par, err, 1, 5000, 0.01);
   if( !fmin.IsValid()){
     printf(" minuit did not converge \n");
   } else {
     std::cout<<"fit succeeded... results: "<<fmin<<std::endl;
    chisq = fmin.Fval();
    u[0]     = fmin.UserParameters().Value( static_cast<unsigned int>(0) );
    *u0_ptr=u[0];
    sigma[0] = fmin.UserParameters().Error( static_cast<unsigned int>(0) );
    u[1]     = fmin.UserParameters().Value( static_cast<unsigned int>(1) );
    *u1_ptr=u[1];
    sigma[1] = fmin.UserParameters().Error( static_cast<unsigned int>(1) );
    u[2]     = fmin.UserParameters().Value( static_cast<unsigned int>(2) );
    *u2_ptr=u[2];
    sigma[2] = fmin.UserParameters().Error( static_cast<unsigned int>(2) );
    u[3]     = fmin.UserParameters().Value( static_cast<unsigned int>(3) );
    *u3_ptr=u[3];
    sigma[3] = fmin.UserParameters().Error( static_cast<unsigned int>(3) );
    //std::cout<<"fitresults: "<<u[0]<<" "<<u[1]<<" "<<u[2]<<" "<<u[3]<<std::endl;
    
   }
 }

 ~SaturationFit(){}

 private:
 
}; 
