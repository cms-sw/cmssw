/** This routine is used to fit the signal line
      shape of CMS barrel calorimeter

  The method used is the one described in note LPC 84-30 (Billoir 1984) :
    "Methode d'ajustement dans un probleme a parametrisation hierarchisee"
  First we read the data file which contains a least 1000 events with sample
   data , then we adjust the best function shape to the data in order to get
   the general parameters which will be used later to get the maximum and the
   time of any signal
  This function is only used to get the 2 parameters of the general function */


#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TFParams.h>
#include "TMatrixD.h"
#include "TMath.h"

#include <iostream>
#include <time.h>


//ClassImp(TFParams)

using namespace std;

TFParams::TFParams( int size, int size_sh ) {

  //int  sdim = size;
  //int plshdim = size_sh;

for (int i=0 ; i<10 ; i++) {
  for (int j=0 ; j<10 ; j++) {
      weight_matrix[i][j] = 8.; 
    }
  } 

}

double TFParams::fitpj(double **adcval , double *parout , double **db_i, double noise_val, int debug)
{
  
#define dimn 10
#define dimin 10
#define plshdim 300
#define nsamp 10                                            
#define ntrack 500       
  //#define debug debug1

  // ******************************************************************
  // *  Definitions of variables used in the routine                    
  // ******************************************************************
  
  
  double a1,a2,a3,b1,b2;
  int iter,nevt;
  //double errpj[dimmat][dimmat]  ;
  double bi[ntrack][2],dbi[ntrack][2];
  double zi[ntrack][2] ;
  double par3degre[3] ;
  int    ioktk[ntrack],nk,nborn_min=0,nborn_max=0;
  double cti[ntrack][6],dm1i[ntrack][4]; 
  double par[4],tsig[1];
  double amp,delta[nsamp],delta2,fun;
  double num_fit_min[ntrack],num_fit_max[ntrack] ;
  int i,j,k,imax[ntrack];

  double ampmax[ntrack],dt,t;
  double chi2, chi2s, da1[nsamp], da2[nsamp], db1[nsamp], db2[nsamp] ;
  double chi2tot;
  double fact2;
  double albet,dtsbeta,variab,alpha,beta;
  double  unsurs1 /*,unsurs2*/ ;
//  double fit3 ;
  int numb_a,numb_b,numb_x;

  fun=0; chi2s=0; chi2tot=0;
  matrice DA,DAT,BK,DB,DBT,C,CT,D,DM1,CDM1,CDM1CT,Z,CDM1Z,YK,Y,B,X,XINV,RES2 ;
  matrice A_CROISS,ZMCT ;

  double *amplu ;
  amplu = new double[nsamp] ;
  
  parout[0] = 0. ;
  parout[1] = 0. ;
  parout[2] = 0. ;

  //  
  //  Initialisation of fit parameters 
  //  

  a1 = a1ini ;
  a2 = a2ini ;
  a3 = a3ini ;
  if( METHODE==2) {
    a2 = a3ini ;   // for lastshape BETA is the third parameter ( ... ! )
  }
  if (debug==1){
    printf(" ------> __> valeurs de a1 %f a2 %f a3 %f\n",a1,a2,a3) ;
  }
  for (i=0 ; i<ntrack ; i++) {
    for (j=0 ; j<2 ; j++ ) {
      bi[i][j] = (double)0. ;
      dbi[i][j] = (double)0. ;
      zi[i][j]=(double)0. ;
      cti[i][j]=(double)0. ;
      dm1i[i][j]=(double)0. ;
    }
  }

  numb_a = 2 ;


  //
  //  Matrices initialisation
  //

  numb_x = 1 ;
  numb_b = 2 ;
  DA = cree_mat(numb_a,numb_x) ;
  DAT = cree_mat(numb_x,numb_a) ;
  BK = cree_mat_prod(DA,DAT) ;
  DB = cree_mat(numb_b,numb_x) ;
  DBT = cree_mat(numb_x,numb_b) ;
  C = cree_mat(numb_a,numb_b) ;
  CT = cree_mat(numb_b,numb_a) ;
  D = cree_mat_prod(DB,DBT) ;
  DM1 = cree_mat_prod(DB,DBT) ;
  CDM1 = cree_mat_prod(C,DM1) ;
  CDM1CT = cree_mat_prod(CDM1,CT) ;
  Z = cree_mat(numb_b,numb_x) ;
  CDM1Z =cree_mat_prod(CDM1,Z) ;
  YK =cree_mat(numb_a,numb_x) ;
  Y =cree_mat(numb_a,numb_x) ;
  B = cree_mat_prod(DA,DAT) ;
  X = cree_mat_prod(DA,DAT) ;
  XINV = cree_mat_prod(DA,DAT) ;
  RES2=cree_mat(numb_a,numb_x) ;
  A_CROISS = cree_mat(numb_a,numb_x) ;
  ZMCT = cree_mat(numb_b,numb_x) ;


  //////////////////////////////                                   
  // First Loop on iterations //	                    
  /////////////////////////////    
 
  for (iter=0 ; iter < 6 ; iter++) {

    chi2tot=0;

    //    
    //    Set zeros for general matrices
    //                                                                      
    
    if (debug==1){
      printf(" Debut de l'iteration numero %d \n",iter) ;
    }
    zero_mat(CDM1Z) ;
    zero_mat(Y) ;
    zero_mat(CDM1CT) ;
    zero_mat(B) ;
    zero_mat(X) ;
    zero_mat(CDM1) ;


    nk = -1 ;	
    if (debug==1){
      printf( " resultats injectes a iterations %d \n",iter) ;
      printf( " parametre a1 = %f \n",a1) ;
      printf( " parametre a2 = %f \n",a2) ;
      printf( " chi2 du fit chi2s = %f \n",chi2s) ;
      
      printf(" value de nevtmax _______________ %d \n",nevtmax) ;
    }
    
    
    /////////////////////						            
    //  Loop on events //
    /////////////////////          
    
    for (nevt=0 ; nevt < nevtmax ; nevt++) {
                 
      //       B1 = BI[nk,1] est la normalisation du signal                    
      //       B2 = BI[nk,2] ewst le dephasage par rapport a une            
      //                 fonction centree en zero                                 
      //      	Nous choisissons ici de demarrer avec les resultats           
      //            de l'ajustement parabolique mais il faudra bien             
      //            entendu verifier que cela ne biaise pas le resultat !      
      //            mise a zero des matrices utilisees dans la boucle             
      
      
      zero_mat(Z) ;
      zero_mat(YK) ;
      zero_mat(BK) ;
      zero_mat(C) ;
      zero_mat(D) ;
      
      nk=nevt ;  
      ampmax[nk] = 0. ;
      imax[nk] = 0 ;
      for ( k = 0 ; k < 10 ; k++) {     
	amplu[k]=adcval[nevt][k] ; 
	if (amplu[k] > ampmax[nk] ) {
	  ampmax[nk] = amplu[k] ;
	  imax[nk] = k ; 
	}
      }
      
      
      if( iter == 0 ) {

        // start with degree 3 polynomial .... 
	//fit3 =polfit(ns ,imax[nk] ,par3degre ,errpj ,amplu ) ;
	//	std::cout << "Poly Fit Param  :"<< par3degre[0] <<" "<< par3degre[1]<< std::endl; 
        
	// start with parabol
	//fit3 = parab(amplu,4,12,par3degre) ;
	/*fit3 =*/ parab(amplu,2,9,par3degre) ;
	//std::cout << "Parab Fit Param :"<< par3degre[0] <<" "<< par3degre[1]<< std::endl; 


	// start with basic initial values
	//par3degre[0]= ampmax+ampmax/10. ;
	//par3degre[1]= (double)imax[nk]+0.1 ;
	//bi[nk][0] = ampmax[nk] ;
	//bi[nk][1] = (double)imax[nk] ;

	num_fit_min[nevt] = (double)imax[nk] - (double)nsmin ;
        num_fit_max[nevt] = (double)imax[nk] + (double)nsmax ;
	
	
	bi[nk][0] = par3degre[0] ;
	bi[nk][1] = par3degre[1] ;
	
	
	if (debug==1){
	  printf("---------> depart ampmax[%d]=%f   maximum %f tim %f \n",
		 nk,ampmax[nk],bi[nk][0],bi[nk][1]);
	}
	
      } else {
	
	
	//  in other iterations  :                                              
	//   increment bi[][] parameters with bdi[][]                         
	//   calculated in previous                                           
	//   iteration 	    			                    			    
	

	bi[nk][0] +=  dbi[nk][0] ;
	bi[nk][1] +=  dbi[nk][1] ;
	
	if (debug==1){
	  printf("iter %d valeur de max %f et norma %f poly 3 \n",iter,bi[nk][1],bi[nk][0]) ;
	}	
      }
    
      b1 = bi[nk][0] ;
      b2 = bi[nk][1] ;


      /////////////////////                                   
      // Loop on samples //	                    
      /////////////////////    
      
      chi2 = 0. ;
      ioktk[nk] = 0 ;
      ns =nborn_max-nborn_min+1 ;
       
      for (k=0 ; k < 10 ; k++){
	
	//
	//	calculation of fonction used to fit
	//	 
	
	dt =(double)k - b2 ;
	t = (double)k ;
	amp = amplu[k] ;
	if (debug==1){
	  printf(" CHECK sample %f ampli %f \n",t,amp) ;
	}
        //unsurs1 = 1./sig_err ;
	//unsurs2 = 1./(sig_err*sig_err) ;
	//unsurs1 = 0.1 ;
	//unsurs2 = 0.01 ;
	
	
	unsurs1=1./noise_val;
	//unsurs2=(1./noise_val)*(1./noise_val);

	//	 					     
	// Pulse shape function used: pulseShapepj
	//
	
	nborn_min = (int)num_fit_min[nevt] ;
	nborn_max = (int)num_fit_max[nevt] ;
        if(k < nborn_min || k > nborn_max ) continue ;
        tsig[0] =(double)k  ;

	
        if(METHODE==2) {
	par[0]=  b1 ;
	par[1] = b2 ;
	par[2] = a1 ;
	par[3] = a2 ;
	fun = pulseShapepj( tsig , par) ;
        }
	if (debug==1){
	  printf(" valeur ampli %f et function %f min %d max %d \n",amp,fun,nsmin,nsmax) ;
	  printf("min %f max %f \n",num_fit_min[nevt],num_fit_max[nevt]) ;
	}
	
	//	 we need to determine a1,a2 which are global parameters         
	//	  and  b1, b2 which are parameters for each individual signal: 
	//        b1, b2 = amplitude and time event by event
	//        a1, a2 = alpha and beta global      
	//	  we first begin to calculate the derivatives used in the following calculation 					     
	
	if(METHODE==2){
	  alpha = a1 ;
  	  beta  = a2 ;
	  albet=alpha*beta;
	  if(dt > -albet)  { 
	    variab = (double)1. + dt/albet ;
	    dtsbeta = dt/beta ;
	    db1[k] = unsurs1*fun/b1 ;
	    fact2 =  fun ;
	    db2[k] = unsurs1*fact2*dtsbeta/(albet*variab) ;
	    da1[k] = unsurs1*fact2*(log(variab)-dtsbeta/(alpha*variab)) ;
	    da2[k] = unsurs1*fact2*dtsbeta*dtsbeta/(albet*variab) ;
	  }
	}
        delta[k] = (amp - fun)*unsurs1 ;
	if (debug==1){
	  printf(" ------->iter %d valeur de k %d amp %f fun %f delta %f \n",
		 iter,k,amp,fun,delta[k]) ;
	  printf(" -----> valeur de k %d delta %f da1 %f da2 %f  \n",
		 k,delta[k],da1[k],da2[k]) ;
	}

	chi2 = chi2 + delta[k]*delta[k]     ;
	
	if (debug==1){
	  printf(" CHECK chi2 %f deltachi2 %f sample %d iter %d \n",chi2,delta[k]*delta[k],k, iter) ;
	}

      }
    
      
      /////////////////////////
      // End Loop on samples //	                    
      /////////////////////////
      
      
      double wk1wk2 ;

      ///////////////////////////
      // Start Loop on samples //	                    
      ///////////////////////////

      for(int k1=nborn_min ; k1<nborn_max+1 ; k1++) {
	wk1wk2 = 1. ;
	int k2 = k1 ;
	
	DA.coeff[0][0] = da1[k1]*wk1wk2 ;
	DA.coeff[1][0] = da2[k1]*wk1wk2 ;
	DAT.coeff[0][0]= da1[k2] ;
	DAT.coeff[0][1]= da2[k2] ;
	DB.coeff[0][0] = db1[k1]*wk1wk2 ;
	DB.coeff[1][0] = db2[k1]*wk1wk2 ;
	DBT.coeff[0][0]= db1[k2] ;
	DBT.coeff[0][1]= db2[k2] ;
	
	//  Compute derivative matrix : matrix b[2][2]  
	
       	produit_mat_int(DA,DAT,BK) ;
	
	//  Compute matrix c[2][2]	  	                    
	
    	produit_mat_int(DA,DBT,C) ;
	
	//  Compute matrix d[2][2]	                             
	
	produit_mat_int(DB,DBT,D) ;
	
	//  Compute matrix y[3] and z[2] depending of delta (amp-fun)        
	
	delta2 = delta[k2] ;
	
	somme_mat_int_scale(DA,YK,delta2) ;			
        somme_mat_int_scale(DB,Z,delta2) ;
	
	ioktk[nk]++ ;
	
      }
      
      /////////////////////////
      // End Loop on samples //	                    
      /////////////////////////
      
      
      //  Remove events with a bad shape 
      
      if(ioktk[nk] < 4 ) {
	printf(" event rejected because npamp_used = %d \n",ioktk[nk]);
	continue ;
      }
      chi2s = chi2/(2. + (double)ns + 2.)  ;      
      chi2tot+=chi2s;

      if (debug==1){
	if (nevt==198 || nevt==199){
	  std::cout << "adc123 pour l'evt " << nevt <<" = "<<adcval[nevt][nborn_min]<<" = "<<adcval[nevt][imax[nevt]]<<" = "<<adcval[nevt][nborn_max]<<std::endl;
	  std::cout << "chi2s  pour l'evt " << nevt <<" = "<< chi2s<<" "<< chi2<<" "<< ns<<"  "<<iter<<std::endl;
	  std::cout << "chi2tot           " << nevt <<" = "<< chi2tot<<"  "<<iter<<std::endl;
	}
      }
      
      //  Transpose matrix C ---> CT                        
     
      transpose_mat(C,CT) ;
      
      //  Calculate DM1 (inverse of D matrix 2x2)  	          
      
      inverse_mat(D,DM1) ;

      
      //  Set matrix product c*d in memory in order to compute variations    
      //   of parameters B at the end of the iteration loop                   
      //   the variations of parameters b are dependant of the variations of  
      //   parameters da[1],da[2]                                            
         
      cti[nk][0] = CT.coeff[0][0]  ;
      cti[nk][1] = CT.coeff[0][1]  ;
      cti[nk][2] = CT.coeff[1][0]  ;
      cti[nk][3] = CT.coeff[1][1]  ;
      
      
       dm1i[nk][0] = DM1.coeff[0][0] ;
       dm1i[nk][1] = DM1.coeff[0][1] ;
       dm1i[nk][2] = DM1.coeff[1][0] ;
       dm1i[nk][3] = DM1.coeff[1][1] ; 

       zi[nk][0] = Z.coeff[0][0]  ;
       zi[nk][1] = Z.coeff[1][0]  ;

       //   Sum the matrix b and y after every event            
      
       for( k=0 ; k< numb_a ; k++) {
	 Y.coeff[k][0] += YK.coeff[k][0] ;
       }
       somme_mat_int(BK,B) ;
       

       //   Calculate c(d-1)                                     

      produit_mat(C,DM1,CDM1) ;

      // Compute c(d-1)ct                                         

      produit_mat_int(CDM1,CT,CDM1CT);
                                                                 
      // Compute c(d-1)z                                          
      
      produit_mat_int(CDM1,Z,CDM1Z) ;


    }
      /////////////////////////
      // End Loop on events //	                    
      /////////////////////////
    
    
    //  Compute b-cdm1ct
       
    diff_mat(B,CDM1CT,X) ;
    inverse_mat(X,XINV) ;
    diff_mat(Y,CDM1Z,RES2) ;

                                                                  
    // Calculation is now easy for da[0] da[1]                         
                                                                  
    produit_mat(XINV,RES2,A_CROISS) ;

    
    //  A la fin, on peut iterer en mesurant l'accroissement a apporter
    //    des parametres globaux par la formule db[i] = dm1(z-ct*da[i])  
         
    for( k=0 ; k< nk+1 ; k++) {
      
      if(METHODE ==2 ) {
	ZMCT.coeff[0][0] = zi[k][0] - (cti[k][0]*A_CROISS.coeff[0][0]+
				       cti[k][1]*A_CROISS.coeff[1][0]) ;
	ZMCT.coeff[1][0] = zi[k][1] - (cti[k][2]*A_CROISS.coeff[0][0]+
				       cti[k][3]*A_CROISS.coeff[1][0]) ;
      }
      
      dbi[k][0] = dm1i[k][0]*ZMCT.coeff[0][0] + dm1i[k][1]*ZMCT.coeff[1][0] ;
      dbi[k][1] = dm1i[k][2]*ZMCT.coeff[0][0] + dm1i[k][3]*ZMCT.coeff[1][0] ;
      if (debug==1){
	if( k < 100 ){
	  printf(" variations de b1= %f et b2= %f  \n",dbi[k][0],dbi[k][1]) ;
	} 
      }
      db_i[k][0] = bi[k][0]+ dbi[k][0]   ;
      db_i[k][1] = bi[k][1]+ dbi[k][1]   ;
    }
    
    
    //   dbi[0] et dbi[1] mesurent les variations a apporter aux       
    //   parametres du signal                                          
                                                                     
    a1 += A_CROISS.coeff[0][0] ;
    a2 += A_CROISS.coeff[1][0] ;
    

    if (debug==1){
      printf(" CHECK croiss coef0: %f  croiss coef1: %f iter %d \n",fabs(A_CROISS.coeff[0][0]),fabs(A_CROISS.coeff[1][0]), iter);
    }
    if(fabs(A_CROISS.coeff[0][0]) < 0.001 && fabs(A_CROISS.coeff[1][0]) < 0.001)
      break;
    
  }
  
  /////////////////////////////
  //  End Loop on iterations //
  /////////////////////////////

  parout[0] = a1 ;
  parout[1] = a2 ;
  parout[2] = a3 ;
  if (debug==1){
    printf( " resultats trouves au bout de %d iterations \n",iter) ;
    printf( " parametre a1 = %f \n",a1) ;
    printf( " parametre a2 = %f \n",a2) ;
  }

  if (debug==1){
    std::cout << " Final chi2 / NDOF  :  "<< chi2tot/nevtmax << std::endl;
    std::cout << " Final (alpha,beta) : ("<< a1<<","<<a2<<")"<< std::endl;
  }

  return chi2tot/nevtmax ;

}

////////////////
// End  Fitpj //
////////////////



/**************************************************************************/
void TFParams::set_const( int n_samples, int sample_min, int sample_max ,
         double alpha ,double beta ,int nevtmaximum) {
/*------------------------------------------------------------------------*/
  ns      = n_samples;
  nsmin   = sample_min ;
  nsmax   = sample_max ;
  nevtmax = nevtmaximum ;
  a1ini = alpha ;
  a2ini = 0.0 ;
  a3ini = beta ;
  step_shape = .04;
  METHODE = 2;
  if(ns > SDIM2) printf("warning: NbOfsamples exceed maximum\n");
} 
void TFParams::produit_mat(matrice A , matrice B , matrice M)
{
  int i,j,k ;
//  resultat du produit A*B = M 
  if(A.nb_colonnes != B.nb_lignes) {
    printf( " Erreur : produit de matrices de tailles incompatibles \n ");
    M.coeff = NULL ;
    return ;
  }
  M.nb_lignes = A.nb_lignes ;
  M.nb_colonnes = B.nb_colonnes ;
  zero_mat(M) ;
  for(i=0 ; i< M.nb_lignes; i++) {
    for(j=0 ; j< M.nb_colonnes ; j++) {
      for(k=0 ; k< A.nb_colonnes; k++){
	M.coeff[i][j] += A.coeff[i][k]*B.coeff[k][j] ;
      }
    }
  }
  return  ;
}

void TFParams::produit_mat_int(matrice A , matrice B, matrice M)
{
  int i,j,k ;
  if(A.nb_colonnes != B.nb_lignes) {
    printf( " Erreur : produit de matrices de tailles incompatibles \n ");
    M.coeff = NULL ;
    return ;
  }
  M.nb_lignes = A.nb_lignes ;
  M.nb_colonnes = B.nb_colonnes ;
  for(i=0 ; i< M.nb_lignes; i++) {
    for(j=0 ; j< M.nb_colonnes ; j++) {
      for(k=0 ; k< A.nb_colonnes; k++){
	M.coeff[i][j] += A.coeff[i][k]*B.coeff[k][j] ;
      }
    }
  }
  return  ;
}
void TFParams::diff_mat(matrice A , matrice B , matrice M)
{
  int i,j ;
//resultat de la difference A-B = M 
  if(A.nb_lignes != B.nb_lignes) {
    printf( " Erreur : difference de matrices de tailles incompatibles \n ");
    M.coeff = NULL ;
    return ;
  }
  M.nb_lignes = A.nb_lignes ;
  M.nb_colonnes = A.nb_colonnes ;
  for(i=0 ; i< M.nb_lignes; i++) {
    for(j=0 ; j < M.nb_colonnes ; j++) {
      M.coeff[i][j] = A.coeff[i][j] - B.coeff[i][j] ;
    }
  }
  return  ;
  
}
void TFParams::copie_colonne_mat(matrice A , matrice M , int nk)
{
  int i,j ;
  int k ;
 /* resultat de la copie de A dans un vecteur colonne M */
  k = 0 ;
  for(i=0 ; i< A.nb_lignes; i++) {
    for(j=0 ; j < A.nb_colonnes ; j++) {
      M.coeff[nk][k] = A.coeff[i][j] ;
   printf(" copie nk %d  i %d j %d k %d A %e M %e \n ",nk,i,j,k,A.coeff[i][j],
	  M.coeff[nk][k]);      
      k++ ;
    }
  }
  return  ;
}

void TFParams::somme_mat_int(matrice A , matrice M)
{
  int i,j;
 /* resultat de la somme integree M += A */
  if(A.nb_lignes != M.nb_lignes) {
    printf( " Erreur : somme de matrices de tailles incompatibles \n ");
    M.coeff = NULL ;
    return ;
  }
  M.nb_lignes = A.nb_lignes ;
  M.nb_colonnes = A.nb_colonnes ;
  for(i=0 ; i< M.nb_lignes; i++) {
    for(j=0 ; j< M.nb_colonnes ; j++) 
      M.coeff[i][j] += A.coeff[i][j] ;
  }
  return  ;
}
void TFParams::somme_mat_int_scale(matrice A , matrice M , double delta)
{
  int i,j ;
  M.nb_lignes = A.nb_lignes ;
  M.nb_colonnes = A.nb_colonnes ;
  for(i=0 ; i< M.nb_lignes; i++) {
    for(j=0 ; j< M.nb_colonnes ; j++) M.coeff[i][j] += A.coeff[i][j]*delta ;
    }
  return  ;
}
void TFParams::transpose_mat(matrice A , matrice M)
{
  int i,j;
// resultat de la transposition = matrice M 
  for(i=0 ; i< A.nb_lignes; i++) {
    for(j=0 ; j< A.nb_colonnes ; j++) {
  	M.coeff[j][i] = A.coeff[i][j]  ;
    }
  }
  return  ;
}
matrice cree_mat_prod(matrice A , matrice B)
{
  int i,j;
  matrice M ; /* resultat de la creation */
 
  M.nb_lignes = A.nb_lignes ;
  M.nb_colonnes = B.nb_colonnes ;
  M.coeff = (double**)malloc(M.nb_lignes*sizeof(double*)) ;
  for(i=0 ; i< M.nb_lignes; i++) 
    M.coeff[i]=(double*)calloc(M.nb_colonnes,sizeof(double));
  for(i=0 ; i< M.nb_lignes; i++) {

    for(j=0 ; j< M.nb_colonnes ; j++) {
  	M.coeff[i][j] = 0.  ;
    }
  }
  //printf(" creation de matrice ---->  nlignes %d ncolonnes %d  \n",
//	 M.nb_lignes,M.nb_colonnes) ;
  return (M) ;
}
matrice cree_mat(int n_lignes,int n_colonnes)
{
  int i,j;
  matrice M ; /* resultat de la creation */
 
  M.nb_lignes = n_lignes ;
  M.nb_colonnes = n_colonnes ;
  M.coeff = (double**)malloc(M.nb_lignes*sizeof(double*)) ;
  for(i=0 ; i< M.nb_lignes; i++) 
    M.coeff[i]=(double*)calloc(M.nb_colonnes,sizeof(double));
  for(i=0 ; i< M.nb_lignes; i++) {
    for(j=0 ; j< M.nb_colonnes ; j++) {
  	M.coeff[i][j] = 0.  ;
    }
  }
  //printf(" creation de matrice --->  nlignes %d ncolonnes %d  \n",
	// M.nb_lignes,M.nb_colonnes) ;
  return (M) ;
}

void fill_mat( matrice A , matrice M)
{
  int i,j;
  /* on remplit la matrice M avec la matrice A */
 
  M.nb_lignes = A.nb_lignes ;
  M.nb_colonnes = A.nb_colonnes ;
  for(i=0 ; i< M.nb_lignes; i++) {
    for(j=0 ; j< M.nb_colonnes ; j++) {
	M.coeff[i][j] = A.coeff[i][j]  ;
	printf("matrice remplie %e \n",M.coeff[i][j]) ;
    }
  } 
  return ;
}
void TFParams::print_mat(matrice M)
{
  int i,j ;
  if( M.coeff == NULL) 
  {
    printf(" erreur : affichage d'une matrice vide \n") ;
    return;
  }
  printf(" m_nli %d M_ncol %d \n",M.nb_lignes,M.nb_colonnes) ;
  for(i=0 ; i< M.nb_lignes; i++) {
    for(j=0 ; j< M.nb_colonnes ; j++) 
      printf(" MATRICE i= %d j= %d ---> %e \n",i,j,M.coeff[i][j]) ;
  }
  //printf(" apres passage d'impression \n") ;
  return ;
}
void TFParams::zero_mat(matrice M)
{
  int i,j ;
  for(i=0 ; i< M.nb_lignes; i++) {
    for(j=0 ; j< M.nb_colonnes ; j++) M.coeff[i][j]=0. ; 
  }
  return ;
}
void TFParams::zero_mat_nk(matrice M,int nk)
{
  int j ;
    for(j=0 ; j< M.nb_colonnes ; j++) M.coeff[nk][j]=0. ;
  return ;
}
void TFParams::print_mat_nk(matrice M,int nk)
{
  int j ;
  if( M.coeff == NULL)
    printf(" erreur : affichage d'une matrice vide \n") ;
  printf(" nk = %d m_nli %d M_ncol %d \n",nk,M.nb_lignes,M.nb_colonnes) ;
    for(j=0 ; j< M.nb_colonnes ; j++) 
      printf(" MATRICE nk= %d j= %d  ---> %e \n",nk,j,M.coeff[nk][j]) ;    
  printf(" apres passage d'impression \n") ;
  return ;
}
void TFParams::inverse_mat( matrice A , matrice M )
{
/*   A[ligne][colonne]   B[ligne][colonne]   */
int i , j   ;
double  deter=0.  ;
/*  M est la matrice inverse de A */
 
 if(A.nb_lignes != A.nb_colonnes) {
   printf( " attention matrice non inversible !!!! %d lignes %d colonnes \n",
	   A.nb_lignes,A.nb_colonnes) ;
   return ;
 }
  zero_mat(M) ;
  if(A.nb_lignes == 2) {
    deter = A.coeff[0][0]*A.coeff[1][1] - A.coeff[0][1]*A.coeff[1][0] ;
    M.coeff[0][0] = A.coeff[1][1]/deter ;
    M.coeff[0][1] = -A.coeff[0][1]/deter ;
    M.coeff[1][0] = -A.coeff[1][0]/deter ;
    M.coeff[1][1] = A.coeff[0][0]/deter ;
  }
 else if(A.nb_lignes == 3) {
      M.coeff[0][0]=A.coeff[1][1]*A.coeff[2][2]-A.coeff[2][1]*A.coeff[1][2] ;
      M.coeff[1][1]=A.coeff[0][0]*A.coeff[2][2]-A.coeff[2][0]*A.coeff[0][2] ;

      M.coeff[2][2]=A.coeff[0][0]*A.coeff[1][1]-A.coeff[0][1]*A.coeff[1][0] ;
      M.coeff[0][1]=A.coeff[2][1]*A.coeff[0][2]-A.coeff[0][1]*A.coeff[2][2] ;
      M.coeff[0][2]=A.coeff[0][1]*A.coeff[1][2]-A.coeff[1][1]*A.coeff[0][2] ;
      M.coeff[1][0]=A.coeff[1][2]*A.coeff[2][0]-A.coeff[1][0]*A.coeff[2][2] ;
      M.coeff[1][2]=A.coeff[1][0]*A.coeff[0][2]-A.coeff[0][0]*A.coeff[1][2] ;
      M.coeff[2][0]=A.coeff[1][0]*A.coeff[2][1]-A.coeff[1][1]*A.coeff[2][0] ;
      M.coeff[2][1]=A.coeff[0][1]*A.coeff[2][0]-A.coeff[0][0]*A.coeff[2][1] ;
      deter=A.coeff[0][0]*M.coeff[0][0]+A.coeff[1][0]*M.coeff[0][1]
	+A.coeff[2][0]*M.coeff[0][2] ;
      for ( i=0 ; i<3 ; i++ ) {
	for ( j=0 ; j<3 ; j++ ) M.coeff[i][j] = M.coeff[i][j]/deter  ;
      }
 }
 else {
 printf(" Attention , on ne peut inverser la MATRICE %d \n",A.nb_lignes) ;
 return ;
 }
  
 return ;
}
Double_t TFParams::polfit(Int_t ns ,Int_t imax , Double_t par3d[dimout] , 
		 Double_t errpj[dimmat][dimmat] ,double *adcpj )
 {
  double val , val2 , val3 , adfmx[dimmat] , parfp3[dimout]  ;
  double ius[dimmat], maskp3[dimmat] ;
  double deglib,fit3,tm,h,xki2 ;
  int i ,nus ,ilow,isup ;
  val=adcpj[imax] ;
  val2=val/2.  ;
  val3=val/3.  ;
  ilow=0       ;
  isup=ns    ;
  deglib=-4.  ;
  for (i=0 ; i<ns ; i++ ){
    deglib=deglib+1.  ;
    ius[i] = 1. ;
    if((adcpj[i] < val3) && (i<imax) ){
      ilow=i  ;
    }
    if(adcpj[i] > val2 ){
      isup=i  ;
    }
  }
  ilow=ilow+1   ;
  if(ilow == imax )ilow=ilow-1 ;
  if(isup-ilow < 3) isup=ilow+3 ;
  nus=0  ;
  for(i=ilow ; i<=isup ; i++){
    
    adfmx[nus]=adcpj[i]  ;
    maskp3[nus] =0. ;
    if(ius[i] == 1) {
      maskp3[nus]=1. ;
      nus=nus+1    ;
    }
  }
  if(nus < 4) return 10000. ;
  xki2 =  f3deg (  nus , parfp3 ,  maskp3 , adfmx ,  errpj ) ;
  tm= parfp3[4]  ;
  h=parfp3[5] ;
  tm=tm+(double)ilow  ;
  par3d[0] = h ;
  par3d[1] = tm ;
  fit3 = xki2 ;
  return fit3 ;
}
double TFParams::f3deg (  int nmxu ,  double parom[dimout] , double mask[dimmat] , double adcpj[dimmat] , double errpj[dimmat][dimmat] ) {
/*                                                                   */
/*  fit   3rd degree polynomial                                      */
/*  nmxu = nb of samples in sample data array adcpj[]
    parom   values of parameters
    errpj  inverse of the error matrix
    fplo3dg uses only the diagonal terms of errpj[][]
*/
  int i , k , l  /*,iworst*/ ;
  double  h , t2 , tm , delta , tmp ;
  double xki2 , dif , difmx , deglib   ;
  double t[dimmat] ,  f[dimmat][4]   ;
  double cov[dimmat][dimmat] , bv[4] , invcov[dimmat][dimmat] , s /*, deter*/  ;
  
  deglib=(double)nmxu - 4.  ;
  for ( i=0 ; i<nmxu ; i++ ) {
    t[i]=i ;
    f[i][0]=1. ;
    f[i][1]=t[i]  ;
    f[i][2]=t[i]*t[i]  ;
    f[i][3]=f[i][2]*t[i] ;
  }
/*   computation of covariance matrix     */
  for ( k=0 ; k<4 ; k++ ) {
    for ( l=0 ; l<4 ; l++ ) {
      s=0.   ;
      for (i=0 ; i<nmxu ; i++ ) {
        s=s+f[i][k]*f[i][l]*errpj[i][i]*mask[i]   ;
      }
      cov[k][l]=s  ;
    }
    s=0.    ;
    for (i=0 ; i<nmxu ; i++ ) {
        s=s+f[i][k]*adcpj[i]*errpj[i][i]*mask[i]   ;
    }
      bv[k]=s  ;
  }
/*     parameters                          */
  /*deter =*/ inverpj ( 4 , cov , invcov );
  for ( k=0 ; k<4 ; k++ ) {
    s=0.  ;
    for ( l=0 ; l<4 ; l++ ) {
      s=s+bv[l]*invcov[l][k]   ;
    }
    parom[k]=s  ;
  }

  if( parom[3] == 0. ){
    parom[4] = -1000.;
    parom[5] = -1000.;
    parom[6] = -1000.;
    return 1000000.;
  }
/*    worst hit and ki2                    */
  xki2=0.    ;
  difmx=0.   ;
    for (i=0 ; i<nmxu ; i++ ){
      t2=t[i]*t[i]  ;
      h= parom[0]+parom[1]*t[i]+parom[2]*t2+parom[3]*t2*t[i] ;
      dif=(adcpj[i]-h)*mask[i]     ;
        if(dif > difmx) {
	  // iworst=i  ;
	  difmx=dif ;
	}
    }
    if(deglib > 0.5) xki2=xki2/deglib ;
/*     amplitude and maximum position                    */
  delta=parom[2]*parom[2]-3.*parom[3]*parom[1]  ;
  if(delta > 0.){
    delta=sqrt(delta)  ;
    tm=-(delta+parom[2])/(3.*parom[3])  ;
    tmp=(delta-parom[2])/(3.*parom[3])  ;
  }
  else{
    parom[4] = -1000.;
    parom[5] = -1000.;
    parom[6] = -1000.;
    return xki2  ;
  }
  parom[4]= tm  ;
  parom[5]= parom[0]+parom[1]*tm+parom[2]*tm*tm+parom[3]*tm*tm*tm ;
  parom[6]= tmp ;
  // printf("par --------> %f %f %f %f \n",parom[3],parom[2],parom[1],parom[0]);
  
   return xki2  ;
}
/*------------------------------------------------------------------*/

double TFParams::inverpj(int n,double g[dimmat][dimmat],double ginv[dimmat][dimmat] )
{
/*                                                                   */
/*  inversion d une matrice symetrique definie positive de taille n  */
/*  J.P. Pansart   Novembre 99                                       */
/*                                                                   */
int i , j , k , jj  ;
double r ,  s  ;
double deter=0  ;
double al[dimmat][dimmat] , be[dimmat][dimmat]  ;
/*   initialisation                                                  */
 for( i=0 ; i<n ; i++ ) {
   for ( j=0 ; j<n ; j++ ) {
    al[i][j] = 0.  ;
    be[i][j] = 0.  ;
   }
 }
/*  decomposition en vecteurs sur une base orthonormee               */
 al[0][0] =  sqrt( g[0][0] )  ;
 for ( i=1 ; i<n ; i++ ) {
 al[i][0] = g[0][i] / al[0][0]  ;
   for ( j=1 ; j<=i ; j++ ) {
    s=0.   ;
    for ( k=0 ; k<=j-1 ; k++ ) {
     s=s+ al[i][k] * al[j][k]  ;
    }
    r= g[i][j] - s   ;
   if( j < i ) al[i][j] = r/al[j][j]  ;
   if( j == i ) al[i][j] =  sqrt ( r)  ;
   }
 }
/*  inversion de la matrice al                                       */
 be[0][0] = 1./al[0][0]  ;
 for ( i=1 ; i<n ; i++ ) {
 be[i][i] = 1. / al[i][i]  ;
   for ( j=0 ; j<i ; j++ ) {
    jj=i-j-1  ;
    s=0.   ;
    for ( k=jj+1 ; k<=i ; k++ ) {
     s=s+ be[i][k] * al[k][jj]  ;
    }
    be[i][jj]=-s/al[jj][jj]  ;
   }
 }
/*   calcul de la matrice ginv                                       */
 for ( i=0 ; i<n ; i++ ) {
   for ( j=0 ; j<n ; j++ ) {
    s=0.   ;
    for ( k=0 ; k<n ; k++ ) {
     s=s+ be[k][i] * be[k][j]  ;
    }
    ginv[i][j]=s  ;
    //    if (debug==1){
    //printf("valeur de la matrice %d %d %f \n",i,j,ginv[i][j]) ;
    //}
   }
 }
  return deter ;
}
/*                                                                   */
/*  inversion d une matrice 3x3                                      */
/*                                                                   */
double TFParams::inv3x3(double a[3][3] , double b[3][3] )
{
/*   a[ligne][colonne]   b[ligne][colonne]   */
int i , j   ;
double  deter=0.  ;
      b[0][0]=a[1][1]*a[2][2]-a[2][1]*a[1][2] ;
      b[1][1]=a[0][0]*a[2][2]-a[2][0]*a[0][2] ;
      b[2][2]=a[0][0]*a[1][1]-a[0][1]*a[1][0] ;
      printf("a[x][x] %e %e %e %e %e %e %e \n",a[0][0],a[1][1],a[0][1],a[1][0],
	     a[0][0]*a[1][1],a[0][1]*a[1][0],b[2][2]);
      b[0][1]=a[2][1]*a[0][2]-a[0][1]*a[2][2] ;
      b[0][2]=a[0][1]*a[1][2]-a[1][1]*a[0][2] ;
      b[1][0]=a[1][2]*a[2][0]-a[1][0]*a[2][2] ;
      b[1][2]=a[1][0]*a[0][2]-a[0][0]*a[1][2] ;
      b[2][0]=a[1][0]*a[2][1]-a[1][1]*a[2][0] ;
      b[2][1]=a[0][1]*a[2][0]-a[0][0]*a[2][1] ;
      deter=a[0][0]*b[0][0] + a[1][0]*b[0][1] + a[2][0]*b[0][2] ;
      printf(" deter = %e \n",deter) ;
 for ( i=0 ; i<3 ; i++ ) {
   for ( j=0 ; j<3 ; j++ ) {
     printf(" avant division a[3][3] %d %d  %e \n",i,j,a[i][j]) ;
     printf(" avant division b[3][3] %d %d  %e %e \n",i,j,b[i][j],deter) ;
     b[i][j] = b[i][j]/deter  ;
     printf(" valeur de b[3][3] apres division %d %d  %e %e \n",i,j,b[i][j],deter) ;
   }
 }
  return deter ;
}

double TFParams::pulseShapepj( Double_t *x, Double_t *par )
{

  Double_t fitfun;
  Double_t ped, h, tm, alpha, beta;
  Double_t  dt, dtsbeta, albet, variab, puiss;
  Double_t b1,b2,a1,a2 ;
  b1 = par[0] ;
  b2 = par[1] ;
  a1 = par[2] ;
  a2 = par[3] ;

  ped   =  0. ;
  h     =  b1 ;
  tm    =  b2 ;
  alpha =  a1 ;
  beta  =  a2 ;
  dt= x[0] - tm  ;
  //printf(" par %f %f %f %f dt = %f albet = %f",b1,b2,a1,a2,dt,albet) ;
  albet = alpha*beta ;
  if( albet <= 0 )return( (Double_t)0. );

  if(dt > -albet)  {
    dtsbeta=dt/beta ;
    variab=1.+dt/albet ;
    puiss=pow(variab,alpha);
    fitfun=h*puiss*exp(-dtsbeta) + ped;
    //printf(" dt = %f h = %f puiss = %f exp(-dtsbeta) %f \n",dt,h,puiss,
    // exp(-dtsbeta)) ;
     }
  else {
      fitfun = ped;
     }

  return fitfun;
}

double TFParams::lastShape( Double_t *x, Double_t *par )
{

  Double_t fitfun;
  Double_t alpha, beta;
  Double_t dt,alphadt,exponent ;
  Double_t b1,b2 ;
  b1 = par[0] ;
  b2 = par[1] ;
  alpha = par[2] ;
  beta  = par[3] ;
  dt= x[0] - b2  ;
  alphadt = alpha*dt ;
  exponent = -(alphadt+(exp(-alphadt)-1.))/beta ; 
  fitfun = b1*exp(exponent) ; 
  return fitfun;
}
double TFParams::lastShape2( Double_t *x, Double_t *par )
{

  Double_t fitfun;
  Double_t alpha, beta;
  Double_t dt,expo1,dt2,exponent ;
  Double_t b1,b2 ;
  b1 = par[0] ;
  b2 = par[1] ;
  alpha = par[2] ;
  beta  = par[3] ;
  dt= x[0] - b2  ;
  expo1 = exp(-beta*dt) ;
  dt2 = dt*dt ;
  exponent = -(alpha*dt2+(expo1-1.)) ;
  fitfun = b1*exp(exponent) ; 
  return fitfun;
}

Double_t TFParams::pulseShapepj2( Double_t *x, Double_t *par )
{

  Double_t fitfun;
  Double_t ped, h, /*tm,*/ alpha, beta;
  Double_t  dt, dtsbeta, albet, variab, puiss;
  Double_t b1,/*b2,*/a1,a2 ;
  b1 = par[0] ;
  //b2 = par[1] ;
  a1 = par[2] ;
  a2 = par[3] ;
  ped   =  0. ;
  h     =  b1 ;
  //tm    =  b2 ;
  alpha =  a1 ;
  beta  =  a2 ;
  dt= x[0]  ;
  albet = alpha*beta ;
  if( albet <= 0 )return( (Double_t)0. );

  if(dt > -albet)  {
    dtsbeta=dt/beta ;
    variab=1.+dt/albet ;
    puiss=pow(variab,alpha);
    fitfun=h*puiss*exp(-dtsbeta) + ped;
  }
  else {
    fitfun = ped;
  }

  /*  printf( "fitfun %f %f %f %f, %f %f %f\n", ped, h, tm, alpha, beta, *x, fitfun );  */

  return fitfun;
}

double TFParams::parab(Double_t ampl[nsamp],Int_t nmin,Int_t nmax,Double_t parout[3])
{
/* Now we calculate the parabolic adjustement in order to get        */
/*    maximum and time max                                           */
  
  double denom,dt,amp1,amp2,amp3 ; 
  double ampmax = 0. ;				
  int imax = 0 ;
  int k ;
/*
                                                                   */	  
  for ( k = nmin ; k < nmax ; k++) {
    if (ampl[k] > ampmax ) {
      ampmax = ampl[k] ;
      imax = k ;
    }
  }
	amp1 = ampl[imax-1] ;
	amp2 = ampl[imax] ;
	amp3 = ampl[imax+1] ;
	denom=2.*amp2-amp1-amp3  ;
/* 							             */	      
	if(denom>0.){
	  dt =0.5*(amp3-amp1)/denom  ;
	}
	else {
	  //printf("denom =%f\n",denom)  ;
	  dt=0.5  ;
	}
/* 						                     */	       
/* ampmax correspond au maximum d'amplitude parabolique et dt        */
/* decalage en temps par rapport au sample maximum soit k + dt       */
		
	parout[0] =amp2+(amp3-amp1)*dt*0.25 ;
	parout[1] = (double)imax + dt ;
	parout[2] = (double)imax ;
return denom ;
}

double TFParams::mixShape( Double_t *x, Double_t *par )
{
  Double_t fitval0,fitval;
  Double_t alpha,beta,fact,puiss;
  Double_t dt,alpha2dt,exponent ;
  Double_t b1,b2,alpha2,t ;
  b1 = par[0] ;
  b2 = par[1] ;
  alpha  = par[2] ;
  alpha2 = par[3] ;
  beta   = par[4] ;
//
  t = x[0] ;
  dt= x[0]-b2  ;
//
  if(t>0.) {
  fact = t/b2 ;
  puiss = pow(fact,alpha) ;
  fitval0 = puiss*exp(-alpha*dt/b2) ;
  }
  else
  {
  fitval0=1. ;
  }
  dt = x[0] - b2 ;
  alpha2dt = dt*alpha2 ;
  exponent = -(alpha2dt+(exp(-alpha2dt)-1.))/beta ;
  fitval = b1*fitval0*exp(exponent) ;  
  return fitval;
}
//=========================
// Method computePulseWidth
//=========================
double TFParams::computePulseWidth( int methode , double alpha_here , double beta_here){ 

// level of amplitude where we calculate the width ( level = 0.5 if at 50 % )
//   (level = 0.3 if at 30 % )
  double level = 0.30 ;
// fixed parameters
  double amplitude   = 1.00 ;
  double offset      = 7.00;  
  double amp_max     = amplitude;

// steps in time
  double t_min       =  offset-4.50;
  double t_max       =  offset+12.50;

  int    t_step_max  = 3000;
  double delta_t     =  (double)((t_max-t_min)/t_step_max);
      
// Loop over time ( Loop 2  --> get width )
  int    t_amp_half_flag =    0;
  double t_amp_half_min  =  999.; 
  double t_amp_half_max  = -999.; 

  for (int t_step=0; t_step<t_step_max; t_step++){

       double t_val = t_min + (double)t_step*delta_t;
       double albet = alpha_here*beta_here ;
       double dt = t_val-offset ;
       double amp =0;

       if( methode == 2 ) { // electronic function
	  if( (t_val-offset) > -albet) {

            amp =  amplitude*TMath::Power( ( 1 + ( dt / (alpha_here*beta_here) ) ) , alpha_here ) * TMath::Exp(-1.0*(dt/beta_here));
	  } else {
	    
	    amp = 1. ;
	  }
       } 

       if( amp > (amp_max*level) && t_amp_half_flag == 0) {
           t_amp_half_flag = 1;
           t_amp_half_min = t_val;
       }

       if( amp < (amp_max*level) && t_amp_half_flag == 1) {
           t_amp_half_flag = 2;
           t_amp_half_max = t_val;
       }          

   }
    
// Compute Width
  double width = (t_amp_half_max - t_amp_half_min);

  return width;
}


