#ifndef _FITPARAMS_H_
#define _FITPARAMS_H_

#include <iostream>
#include <cmath>
#include "TPrincipal.h"
#include "TMultiDimFit.h"
#include "MultiDimFitData.h"
#include "Track.h"

#include <boost/serialization/split_member.hpp>
#include <boost/serialization/vector.hpp>
//#include <boost/serialization/base_object.hpp>
//#include <boost/serialization/export.hpp>

using namespace std;

/**
   \brief Parameters needed for Principal Components Analysis fit in a sub-sector (PCA+MultiDimFit params)
**/

class FitParams{

 private:
 
  int nb_layers;
  int threshold;
  int nb_principal;
  int nb_multidimfit;

  TPrincipal* principal;
  vector<double> eigen;
  vector<double> mean;
  vector<double> sig;
  double** transform;

  TMultiDimFit* pt_fit;
  MultiDimFitData* pt_fit_data;
  TMultiDimFit* phi0_fit;
  MultiDimFitData* phi0_fit_data;
  TMultiDimFit* d0_fit;
  MultiDimFitData* d0_fit_data;
  TMultiDimFit* eta0_fit;
  MultiDimFitData* eta0_fit_data;
  TMultiDimFit* z0_fit;
  MultiDimFitData* z0_fit_data;

  void computePrincipalParams();
  void computeMultiDimFitParams();
  void initializeMultiDimFit(TMultiDimFit* f);
  void init();
  double getPTFitValue(double* val);
  double getPhi0FitValue(double* val);
  double getD0FitValue(double* val);
  double getEta0FitValue(double* val);
  double getZ0FitValue(double* val);

 public:
  FitParams();
  FitParams(const FitParams& ref);
  FitParams(int n_layers, int thresh);
  ~FitParams();
  void addDataForPrincipal(double* d);
  bool hasPrincipalParams();
  void forcePrincipalParamsComputing();
  void addDataForMultiDimFit(double* d, double* val);
  bool hasMultiDimFitParams();
  void forceMultiDimFitParamsComputing();
  void x2p(double *x, double *p);
  double get_chi_square(double *x, double p);
  Track* getTrack(double* val);
  int getNbPrincipalTracks();
  int getNbMultiDimFitTracks();

  friend class boost::serialization::access;
  
  template<class Archive> void save(Archive & ar, const unsigned int version) const{
    ar << nb_layers;
    ar << threshold;
    if(nb_principal>threshold)
      ar << nb_principal;
    else{ // if the parametrization is not done we will have to start again from scratch
      int tmp = 0;
      ar << tmp;
    }
    ar << eigen;
    ar << sig;
    ar << mean;
    for(int i=0;i<nb_layers*3;i++){
      for(int j=0;j<nb_layers*3;j++){
	ar << transform[i][j];
      }
    }
    if(nb_multidimfit>threshold){
      ar << nb_multidimfit;
      ar << pt_fit_data;
      ar << phi0_fit_data;
      ar << d0_fit_data;
      ar << eta0_fit_data;
      ar << z0_fit_data;
    } 
    else{ // if the fit parameters computing is not done we will have to start again from scratch
      int tmp = 0;
      ar << tmp;
    }
  }
  
  template<class Archive> void load(Archive & ar, const unsigned int version){
    ar >> nb_layers;
    //init();
    ar >> threshold;
    ar >> nb_principal;
    ar >> eigen;
    ar >> sig;
    ar >> mean;
    for(int i=0;i<nb_layers*3;i++){
      for(int j=0;j<nb_layers*3;j++){
	ar >> transform[i][j];
      }
    }
    principal = NULL;
    ar >> nb_multidimfit;
    if(nb_multidimfit>threshold){ // restauration des parametres de FIT
      ar >> pt_fit_data;
      ar >> phi0_fit_data;
      ar >> d0_fit_data;
      ar >> eta0_fit_data;
      ar >> z0_fit_data;
    }
    else{
      pt_fit_data=NULL;
      phi0_fit_data=NULL;
      d0_fit_data=NULL;
      eta0_fit_data=NULL;
      z0_fit_data=NULL;
    }
  }
  
  BOOST_SERIALIZATION_SPLIT_MEMBER()

};
#endif
