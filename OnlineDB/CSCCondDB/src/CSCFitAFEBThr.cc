#include "Minuit2/VariableMetricMinimizer.h"
#include "Minuit2/FunctionMinimum.h"

#include <OnlineDB/CSCCondDB/interface/CSCFitAFEBThr.h>
#include <OnlineDB/CSCCondDB/interface/CSCThrTurnOnFcn.h>

#include <cmath>
#include <vector>
#include <iostream>

using namespace ROOT::Minuit2;
using namespace std;

CSCFitAFEBThr::CSCFitAFEBThr() {
  theOBJfun = new CSCThrTurnOnFcn();
  theFitter = new VariableMetricMinimizer();
}

CSCFitAFEBThr::~CSCFitAFEBThr() {
  delete theFitter;
  delete theOBJfun;
}

bool CSCFitAFEBThr::ThresholdNoise(const std::vector<float> & inputx, 
                                   const std::vector<float> & inputy, 
                                   const int                & npulses,
                                   std::vector<int>         & dacoccup,
                                   std::vector<float>       & mypar,
                                   std::vector<float>       & ermypar,
                                   float                    & ercorr,
                                   float                    & chisq, 
                                   int                      & ndf,
                                   int                      & niter,
                                   float                    & edm
                                  ) const {
  bool status = false;			 
 
  std::vector<double> parinit(2,0);
  std::vector<double> erparinit(2,0);

  /// initial parameters, parinit[0]-threshold,  parinit[1]-noise  
  parinit[0] = 30.0;
  parinit[1] = 2.0;
  
  erparinit[0] = 20;
  erparinit[1] = 0.5;

  /// do not fit input[y]==max and input[y]==0.0; calculate binom. error;
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> ynorm;
  std::vector<float> ery;
  x.clear();
  y.clear();
  ynorm.clear();
  ery.clear();

  /// ndf > 0 if there is input data,number of points to fit > 2 and
  ///         fit did not fail.
  /// ndf = 0 if number of points to fit = 2
  /// ndf =-1 .......................... = 1
  /// ndf =-2 .......................... = 0
  /// ndf =-3 fit failed (number of points to fit was > 2)
  /// ndf =-4 no input data
  
  int sum=0;
  float r;
  for(size_t i=0;i<inputx.size();i++) {
     if(inputy[i]>0.0) sum++;
     r=inputy[i]/(float)dacoccup[i];
     ynorm.push_back(r);
//     std::cout<<" "<<ynorm[i];
  }
//  std::cout<<std::endl;
  if(sum==0) {
    ndf=-4;
    return status;
  }
   

  int nbeg=inputx.size();
  // for(size_t i=inputx.size()-1;i>=0;i--) // Wrong.
  // Because i is unsigned, i>=0 is always true, 
  // and the loop termination condition  is never reached.
  // We offset by 1.
  for(size_t i=inputx.size();i>0;i--) {
    if(ynorm[i-1]<1.0) nbeg--;
    if(ynorm[i-1]==1.0) break;
  }

  for(size_t i=nbeg;i<inputx.size();i++) {
    if(ynorm[i]>0.0) {
      x.push_back(inputx[i]); 
      y.push_back(ynorm[i]);

      float p=inputy[i]/(float)dacoccup[i];
      float s=(float)dacoccup[i] * p * (1.0-p);
      s=sqrt(s)/(float)dacoccup[i];
      ery.push_back(s);
    }			       
  }

  /// do not fit data with less than 3 points
  ndf=x.size()-2; 
  if(ndf <=0) return status;

  /// Calculate approximate initial threshold par[0]
  float half=0.5;
  float dmin=999999.0;
  float diff;
  for(size_t i=0;i<x.size();i++) {
    diff=y[i]-half; if(diff<0.0) diff=-diff;
    if(diff<dmin) {dmin=diff; parinit[0]=x[i];}   // par[0] from data    
    //std::cout<<i+1<<" "<<x[i]<<" "<<y[i]<<" "<<ery[i]<<std::endl;
  }

  /// store data, errors and npulses for fit
  theOBJfun->setData(x,y); 
  theOBJfun->setErrors(ery);   
  theOBJfun->setNorm(1.0);

 // for(size_t int i=0;i<x.size();i++) std::cout<<" "<<x[i]<<" "<<y[i]
 //                                               <<" "<<ery[i]<<std::endl; 

  /// Fit  as 1D, <=500 iterations, edm=10**-5 (->0.1)
  FunctionMinimum fmin=theFitter->Minimize(*theOBJfun,parinit,erparinit,1,500,0.1);

  status = fmin.IsValid();

  if(status) { 
    mypar[0]=(float)fmin.Parameters().Vec()(0);
    mypar[1]=(float)fmin.Parameters().Vec()(1);
    ermypar[0]=(float)sqrt( fmin.Error().Matrix()(0,0) );
    ermypar[1]=(float)sqrt( fmin.Error().Matrix()(1,1) );
    ercorr=0;
    if(ermypar[0] !=0.0 && ermypar[1]!=0.0)
    ercorr=(float)fmin.Error().Matrix()(0,1)/(ermypar[0]*ermypar[1]);

    chisq  = fmin.Fval();
    ndf=y.size()-mypar.size();
    niter=fmin.NFcn();         
    edm=fmin.Edm();
  }
  else ndf=-3;
  return status;
}
