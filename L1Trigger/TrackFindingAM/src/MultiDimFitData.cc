#include "../interface/MultiDimFitData.h"

MultiDimFitData::~MultiDimFitData(){
  //cout<<"destruction MultiDimFitData"<<endl;
  if(gXMin!=NULL)
    delete [] gXMin;
  if(gXMax!=NULL)
    delete [] gXMax;
  if(gCoefficient!=NULL)
    delete [] gCoefficient;
  if(gPowerIndex!=NULL)
    delete [] gPowerIndex;
  if(gPower!=NULL)
    delete [] gPower;
  if(m_final_coeffs!=NULL)
    delete [] m_final_coeffs;
  //cout<<"destruction MultiDimFitData ok"<<endl;
}

MultiDimFitData::MultiDimFitData(){
  gNVariables=-1;
  gNCoefficients=-1;
  gDMean=-1;
  gNMaxTerms = -1;
  gNMaxFunctions = -1;

  nb_layers=-1;
  
  gXMin=NULL;
  gXMax=NULL;
  gCoefficient=NULL;
  gPowerIndex=NULL;
  gPower=NULL;
  m_final_coeffs=NULL;
}

MultiDimFitData::MultiDimFitData(const MultiDimFitData& ref){
  gNVariables=ref.gNVariables;
  gNCoefficients=ref.gNCoefficients;
  gDMean=ref.gDMean;
  gNMaxTerms =ref.gNMaxTerms;
  gNMaxFunctions =ref.gNMaxFunctions;
  
  nb_layers=ref.nb_layers;
  
  gXMin = new double[gNVariables];
  for(int i=0;i<gNVariables;i++){
    gXMin[i]=ref.gXMin[i];
  }

  gXMax = new double[gNVariables];
  for(int i=0;i<gNVariables;i++){
    gXMax[i]=ref.gXMax[i];
  }

  gCoefficient = new double[gNCoefficients];
  for(int i=0;i<gNCoefficients;i++){
    gCoefficient[i]=ref.gCoefficient[i];
  }
  
  gPowerIndex = new int[gNMaxTerms];
  for(int i=0;i<gNMaxTerms;i++){
    gPowerIndex[i]=ref.gPowerIndex[i];
  }

  gPower = new int[gNVariables*gNMaxFunctions];
  for(int i=0;i<gNVariables*gNMaxFunctions;i++){
    gPower[i]=ref.gPower[i];
  }

  int size = ((nb_layers-1)*3)+1;
  m_final_coeffs = new double[size];
  for(int i=0;i<size;i++){
    m_final_coeffs[i]=ref.m_final_coeffs[i];
  }
}

MultiDimFitData::MultiDimFitData(TMultiDimFit* m, int nb){
  cout<<"Construction multidimfitdata"<<endl;
  gNVariables=m->GetNVariables();
  gNCoefficients=m->GetNCoefficients();
  gNMaxTerms = m->GetMaxTerms();
  gNMaxFunctions = m->GetMaxFunctions();
  gDMean = m->GetMeanQuantity();

  nb_layers = nb;

  gXMin = new double[gNVariables];
  for(int i=0;i<gNVariables;i++){
    gXMin[i]=(*m->GetMinVariables())[i];
  }

  gXMax = new double[gNVariables];
  for(int i=0;i<gNVariables;i++){
    gXMax[i]=(*m->GetMaxVariables())[i];
  }

  gCoefficient = new double[gNCoefficients];
  for(int i=0;i<gNCoefficients;i++){
    gCoefficient[i]=(*m->GetCoefficients())[i];
  }
  
  gPowerIndex = new int[gNMaxTerms];
  for(int i=0;i<gNMaxTerms;i++){
    gPowerIndex[i]=m->GetPowerIndex()[i];
  }

  gPower = new int[gNVariables*gNMaxFunctions];
  for(int i=0;i<gNVariables*gNMaxFunctions;i++){
    gPower[i]=m->GetPowers()[i];
  }

  int size = ((nb_layers-1)*3)+1;
  m_final_coeffs = new double[size];
  double returnValue = gDMean;
  for (int i=0; i<gNCoefficients; i++){
    double term  = gCoefficient[i];
    for (int j=0; j<gNVariables; j++){
      double r=0;
      
      int power = gPower[gNVariables*gPowerIndex[i]+j];
      double v  =  1+2.*(-gXMax[j])/(gXMax[j]-gXMin[j]);

      if (power!=1) m_final_coeffs[j+1] = 2.*term/(gXMax[j]-gXMin[j]);

      switch(power){
      case 1: r = 1; break;
      case 2: r = v; break;
        //case 3: r = 2*v*v-1; break; 
      }
      // multiply this term by the poly in the jth var
      term *= r;
    }
    returnValue += term;
  }
  m_final_coeffs[0] = returnValue;
  
}

double MultiDimFitData::getVal(double *x)
{
  double returnValue = m_final_coeffs[0];
  int size = ((nb_layers-1)*3)+1;

  for (int i=1; i<size; ++i) 
    returnValue += m_final_coeffs[i]*x[i-1];

  return returnValue;
}
