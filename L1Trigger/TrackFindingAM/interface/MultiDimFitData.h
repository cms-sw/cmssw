#ifndef _MULTIDIMFITDATA_H_
#define _MULTIDIMFITDATA_H_

#include <iostream>
#include "TMultiDimFit.h"

#include <boost/serialization/split_member.hpp>

using namespace std;

/**
   \brief Parameters needed for the MultiDimFit in a sub-sector
**/

class MultiDimFitData{
 private:
  int    gNVariables;
  int    gNCoefficients;
  int    gNMaxTerms;
  int    gNMaxFunctions;
  double gDMean;

  int nb_layers;
  
  double *gXMin;
  double *gXMax;
  double *gCoefficient;
  int   *gPowerIndex;
  int   *gPower;
  double *m_final_coeffs;

 public:
  MultiDimFitData(TMultiDimFit* m, int nb);
  MultiDimFitData();
  MultiDimFitData(const MultiDimFitData&);
  ~MultiDimFitData();

  double getVal(double *x);

 friend class boost::serialization::access;
  
  template<class Archive> void save(Archive & ar, const unsigned int version) const{

    ar << gNVariables;
    ar << gNCoefficients;
    ar << gDMean;
    ar << gNMaxTerms;
    ar << gNMaxFunctions;
    ar << nb_layers;
  
    for(int i=0;i<gNVariables;i++){
      ar << gXMin[i];
    }
    
    for(int i=0;i<gNVariables;i++){
      ar << gXMax[i];
    }
    
    for(int i=0;i<gNCoefficients;i++){
      ar << gCoefficient[i];
    }

    for(int i=0;i<gNMaxTerms;i++){
      ar << gPowerIndex[i];
    }

    for(int i=0;i<gNVariables*gNMaxFunctions;i++){
      ar << gPower[i];
    }

    int size = ((nb_layers-1)*3)+1;
    for(int i=0;i<size;i++){
      ar << m_final_coeffs[i];
    }

  }
  
  template<class Archive> void load(Archive & ar, const unsigned int version){
    ar >> gNVariables;
    ar >> gNCoefficients;
    ar >> gDMean;
    ar >> gNMaxTerms;
    ar >> gNMaxFunctions;
    ar >> nb_layers;
  
    gXMin = new double[gNVariables];
    for(int i=0;i<gNVariables;i++){
      ar >> gXMin[i];
    }
    
    gXMax = new double[gNVariables];
    for(int i=0;i<gNVariables;i++){
      ar >> gXMax[i];
    }

    gCoefficient = new double[gNCoefficients];
    for(int i=0;i<gNCoefficients;i++){
      ar >> gCoefficient[i];
    }

    gPowerIndex = new int[gNMaxTerms];
    for(int i=0;i<gNMaxTerms;i++){
      ar >> gPowerIndex[i];
    }

    gPower = new int[gNVariables*gNMaxFunctions];
    for(int i=0;i<gNVariables*gNMaxFunctions;i++){
      ar >> gPower[i];
    }

    int size = ((nb_layers-1)*3)+1;
    m_final_coeffs = new double[size];
    for(int i=0;i<size;i++){
      ar >> m_final_coeffs[i];
    }
  }
  
  BOOST_SERIALIZATION_SPLIT_MEMBER()

};

#endif
