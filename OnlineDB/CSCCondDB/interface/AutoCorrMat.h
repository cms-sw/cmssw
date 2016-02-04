/*
 Author: Stan Durkin
 
*/

#include <stdio.h>
#include <math.h> 
#define LAYERS_ma 6
#define STRIPS_ma 80

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
   for (int i=3;i<12;i++){
     evar[i]=1; 
     variance[i]=10.0;
     mymean[i]=3.2;
   }
   evt=0;
  }

  void add(int *adc){
    int pairs[12][2]={{3,3},{3,4},{4,4},{3,5},{4,5},{5,5},{4,6},{5,6},{6,6},{5,7},{6,7},{7,7}};
    double ped=(adc[0]+adc[1])/2.;
    evt++;      

    for(int i=3;i<8;i++){
      if(fabs(adc[i]-ped)<25.){
        evar[i]++;
        mymean[i]   += adc[i]-ped;
        variance[i] += (adc[i]-ped)*(adc[i]-ped);
      }
    }
    
    for(int i=0;i<12;i++){
      
      //Add values within 3 sigma of mean only
      float threeSigma0 = 3. * sqrt(variance[pairs[i][0]]/evar[pairs[i][0]]);
      float threeSigma1 = 3. * sqrt(variance[pairs[i][1]]/evar[pairs[i][1]]);
      
     if (fabs(adc[pairs[i][0]]-ped)<threeSigma0 && fabs(adc[pairs[i][1]]-ped)<threeSigma1){
  	N[i]=N[i]+1;
	Mat[i]=Mat[i]+(adc[pairs[i][0]]-ped)*(adc[pairs[i][1]]-ped);
      }
      //end 3 sigma 
    }
  }
  
  float *mat(){
    float *tmp;
    for(int i=0;i<12;i++)tMat[i]=Mat[i]/N[i];
    tmp=tMat;
    return tmp; 
  }

 private:

  float Mat[12];
  float N[12];
  float tMat[12];
  float mymean[12];
  float variance[12];
  int evt;
  int evar[12];
};

class Chamber_AutoCorrMat{
 public:

  Chamber_AutoCorrMat(){}
  ~Chamber_AutoCorrMat(){}

  void zero(){
    for(int lay=0;lay<6;lay++){
      for(int strip=0;strip<STRIPS_ma;strip++){
        CMat[lay][strip].zero();
      }
    }
  }

  void add(int lay,int strip,int *adc){
    CMat[lay][strip].add(adc);
  } 

  float *autocorrmat(int lay,int strip){
    float *tmp;
    tmp=m;
    tmp=CMat[lay][strip].mat();
    return tmp;
  }

 private:

  AutoCorrMat CMat[LAYERS_ma][STRIPS_ma];
  float m[12];

};
