#ifndef TSFit_H
#define TSFit_H

#define  SDIM     14 /* default number of samples for cristal */
#define  PLSHDIM 650  /* default size of the pulse shape array */
//these 2 last parameters are overwritten in constructor


#define  matdim     5 /* parameters fit max matrice size */    
#define  diminpar  10   
#define  dimoutpar 10   
#define  npar_moni  4

#include "TObject.h"

class TSFit : public TObject {
private :
  /*
    nbs = nb of samples in sample data array[sdim]   nbs<=sdim  
    nmxu  number of samples to be used to fit the apd pulse shape
    nmxu<=nbs
  */
  int nbs;
  int n_presamples;
  int iinf, isup;//limits indices for max search using pshape
  double avtm;// mean time in clock unit of the maximums
  int n_samples_bef_max;//number of samples before and after sample max
  int n_samples_aft_max;//to be used in pulse fit
                        //(  theoritical and experimental )
                        //the total number of samples used is 
                        //n_samples_bef_max+n_samples_aft_max+1
  double xki2_max, norme;
  //  int *sample_flag;
  int nmxu_sto ;
  double alpha_th, beta_th;
  int nbr_iter_fit;
  double cov[matdim][matdim], invcov[matdim][matdim];
  double al[matdim][matdim], be[matdim][matdim];//intern to inverms
  //double *t, *z, *f, *acc, *adfmx, *maskp3, *adcp, *corel, *nbcor; //[sdim]
  //double **ff; //[sdim][4]
  //double **der=new double[SDIM];//[sdim][5]
  double parfp3[dimoutpar];

  //double  *tb, *fb, *accb;//[plshdim]
  //int *sample_flag_call;//[plshdim]
  //double **derb; //[plshdim][5]
  //double **coeff;//[plshdim][3]


  double errmat[SDIM][SDIM];//inverse of error matrix
  int sample_flag[SDIM];
  double t[SDIM];
  double z[SDIM];
  double f[SDIM];
  double acc[SDIM];
  double adfmx[SDIM];
  double adcp[SDIM];
  double maskp3[SDIM];
  double corel[SDIM];
  double nbcor[SDIM];
 
  double ff[SDIM][4];
  double der[SDIM][5];
 

public :
  int sdim;
  int plshdim;

  TSFit( int size = SDIM, int size_sh = PLSHDIM );

  virtual ~TSFit() {}

  void set_params( int, int, int, int, int, double, double, int, int );

  void init_errmat(double);

  double fit_third_degree_polynomial( double *,
                                      double * );


  double fpol3dg (  int,
		    double *,
		    double *,
		    double *);

  double inverms  ( int, double xx[matdim][matdim], double yy[matdim][matdim] );

  //  ClassDef( TSFit, 1 )
};

#endif


