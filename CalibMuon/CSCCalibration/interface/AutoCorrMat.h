/*
 Author: Stan Durkin
 
*/

#include <stdio.h>
#include <math.h>

class AutoCorrMat{

 public:

  AutoCorrMat(){
    for(int i=0;i<12;i++){
      Mat[i]=0.0;
      N[i]=0.0;
    }
  }

  ~AutoCorrMat(){}

  void zero(){
   for(int i=0;i<12;i++){
     Mat[i]=0.0;
     N[i]=0.0;
    }
  }

  void add(int *adc){
    int pairs[12][2]={{3,3},{3,4},{4,4},{3,5},{4,5},{5,5},{4,6},{5,6},{6,6},{5,7},{6,7},{7,7}};
    double ped=(adc[0]+adc[1])/2.;
    for(int i=0;i<12;i++){
      N[i]=N[i]+1;
      Mat[i]=Mat[i]+(adc[pairs[i][0]]-ped)*(adc[pairs[i][1]]-ped);
      
    }
  }

  double *mat(){
    double *tmp;
    for(int i=0;i<12;i++)tMat[i]=Mat[i]/N[i];
    tmp=tMat;
    return tmp; 
  }

 private:

  double Mat[12];
  double N[12];
  double tMat[12];

};

class Chamber_AutoCorrMat{
 public:

  Chamber_AutoCorrMat(){}
  ~Chamber_AutoCorrMat(){}

  void zero(){
    for(int lay=0;lay<6;lay++){
      for(int strip=0;strip<80;strip++){
        CMat[lay][strip].zero();
      }
    }
  }

  void add(int lay,int strip,int *adc){
    CMat[lay][strip].add(adc);
  } 

  double *autocorrmat(int lay,int strip){
    double *tmp;
    tmp=m;
    tmp=CMat[lay][strip].mat();
    return tmp;
  }

 private:

  AutoCorrMat CMat[6][80];
  double m[12];

};
