#ifndef TFParams_h
#define TFParams_h
#include "TROOT.h"
#include "TObject.h"
#include "TArrayI.h"
#include "TArrayD.h"
#include "TArrayC.h"
#include "TCanvas.h"
#include "TDirectory.h"
#include "TPaveLabel.h"
#include "TF1.h"
#include "time.h"
#include "TGraph.h"
#include <stdio.h>
#include <math.h>
#include "TH2.h"
#include "TH1.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TPaveText.h"
#include "TPaveLabel.h"
#include "TProfile.h"
#include "TVirtualX.h"
#include "TObject.h"
//#include "TMatrixD.h"
#define  SDIM2     10 /* number of samples for cristal */
#define  PLSHDIM 650  /* size of the pulse shape array */
//double pulseShape( Double_t x[1], Double_t par[4] ) ;
//
  struct matrice
{
	    int nb_lignes ;
	    int nb_colonnes ;
	    double **coeff ;
};
 typedef struct matrice matrice ;
 matrice cree_mat(int,int) ;
 matrice cree_mat_prod(matrice,matrice) ;
 void fill_mat(matrice,matrice) ;
 matrice fill_mat_int(matrice,matrice,matrice) ;
 
#define dimmat  30
#define dimout 10
#define nbmax_cell 1000


 class TFParams : public TObject {

 private:

 int ns ; // number of samples
 int nsmin ; // beginning of fit
 int nsmax ; // end of fit
 int nevtmax ; // number of events to fit
 double a1ini ; // value of alpha at starting point
 double a2ini ; // value of alpha_prim/beta at starting point
 double a3ini ; // value of beta/alpha_prim at starting point
 double step_shape ;
 double adclu[26] ;
 double weight_matrix[10][10];
 int METHODE ;

 public:

 TFParams( int size = SDIM2, int size_sh = PLSHDIM );
~TFParams(){};
double fitpj(double **, double *,double ** , double noise_val, int debug) ;
 void set_const( int ,int ,int ,double ,double ,int);
 void produit_mat(matrice,matrice,matrice) ;
 void produit_mat_int(matrice,matrice,matrice) ;
 void diff_mat(matrice,matrice,matrice) ;
 void somme_mat_int(matrice,matrice) ;
 void somme_mat_int_scale(matrice,matrice,double) ;
 void print_mat_nk(matrice,int) ;
 void print_mat(matrice) ;
 void transpose_mat(matrice,matrice) ;
 void inverse_mat(matrice,matrice) ;
 void copie_colonne_mat(matrice,matrice,int) ;
 char name_mat[10] ;
 void zero_mat(matrice) ;
 void zero_mat_nk(matrice,int) ;
 double f3deg(int ,  double parom[dimout] , double mask[dimmat] ,
  double adcpj[dimmat] , double errpj[dimmat][dimmat]) ;
 double parab(double *,Int_t,Int_t,double *) ;
 Double_t polfit(Int_t ns ,Int_t imax , Double_t par3d[dimout] , 
 Double_t errpj[dimmat][dimmat] ,double * ) ; 
 double inverpj(int,double g[dimmat][dimmat],double ginv[dimmat][dimmat]);
 double inv3x3(double a[3][3] , double b[3][3] ) ;
 double pulseShapepj( Double_t *, Double_t * ) ;
 double pulseShapepj2( Double_t *, Double_t * ) ;
 double lastShape( Double_t *, Double_t * ) ;
 double lastShape2( Double_t *, Double_t * ) ;
 double mixShape( Double_t *, Double_t * ) ;
 double computePulseWidth( int, double, double) ;

 //  ClassDef( TFParams, 1 ) 
};
#endif
