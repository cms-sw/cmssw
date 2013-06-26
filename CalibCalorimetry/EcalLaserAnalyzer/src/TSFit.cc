/* 
 *  \class TSFit
 *
 *  $Date: 2012/02/09 10:08:10 $
 *  \author: Jean-Pierre Pansart - CEA/Saclay
 */

#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TSFit.h>

#include <cstdio>
#include <cmath>
#include <cstring>

//ClassImp(TSFit)

//------------------------------------------------------------------------
// TSFit provide fitting methods of pulses with third degree polynomial
//

TSFit::TSFit( int size, int size_sh ) {
  sdim = size;
  plshdim = size_sh;
  // sample_flag = new int[size];
//   t = new double[sdim];
//   z = new double[sdim];
//   f = new double[sdim];
//   acc = new double[sdim];
//   adfmx = new double[sdim];
//   adcp = new double[sdim];
//   maskp3 = new double[sdim];
//   corel = new double[sdim];
//   nbcor = new double[sdim];
//   int k;
//   ff = new double *[sdim];
//   for(k=0;k<sdim;k++)ff[k] = new double[4];
//   der = new double *[sdim];
//   for(k=0;k<sdim;k++)der[k] = new double[5];
   

}

void TSFit::set_params( int n_samples, int niter, int n_presmpl,
		   int sample_min, int sample_max,
		   double time_of_max, double chi2_max,
		   int nsbm, int nsam ){
  // parameters initialisation of the fit package
  // nsbm  n_samples_bef_max, nsam n_samples_aft_max

  nbs          = n_samples;
  nbr_iter_fit = niter;
  n_presamples = n_presmpl;
  iinf         = sample_min;
  isup         = sample_max;
  avtm         = time_of_max;
  xki2_max     = chi2_max;
  n_samples_bef_max = nsbm;
  n_samples_aft_max = nsam;

  norme        = 0.;
  alpha_th     = 2.20;
  beta_th      = 1.11;

  int k;

  for(k=0;k<=nbs;k++){
    sample_flag[k] = 0;
  }

  for(k=sample_min;k<=sample_max;k++){
    sample_flag[k] = 1;
  }
  /*
  int lim1 = ( iinf > n_presamples ) ? n_presamples : iinf;
  for(int k=lim1;k<=sample_max;k++){
    sample_flag[k] = 2;
  }
  */
 //  printf( "sample_fag : " );
//   for(k=0;k<=nbs;k++){
//     printf( "%1d ", sample_flag[k] );
//   }
//   printf( "\n" );

}

void TSFit::init_errmat( double noise_initialvalue){
  //  Input:  noise_initial value  noise (in adc channels) as read in the 
  //  data base.
/*------------------------------------------------------------------------*/

  int i, j;
  //double one_over_noisesq;

  //one_over_noisesq = 1. / ( noise_initialvalue * noise_initialvalue );
  for(i=0;i<sdim;i++ ){
    for(j=0;j<sdim;j++){
      errmat[i][j] = noise_initialvalue;
      //errmat[i][j] = 0.;
    }
    //errmat[i][i] = one_over_noisesq;
  }
}

double TSFit::fpol3dg ( int nmxul,
                        double *parom,
			double *mask,
			double *adc){
  // fit third degree polynomial
  // nmxul   array adc[] length
  // parom   return parameters (a0,a1,a2,a3,pos max,height)
  // fplo3dg uses only the diagonal terms of errmat[][]
  // errmat  inverse of the error matrix

  int i, k, l;
  double  h, t2, tm, delta, tmp;
  double xki2, dif, difmx, deglib;
  double bv[4], s;

  deglib=(double)nmxul - 4.;
  for(i=0;i<nmxul;i++){
    t[i]=i;
    ff[i][0]=1.;
    ff[i][1]=t[i];
    ff[i][2]=t[i]*t[i];
    ff[i][3]=ff[i][2]*t[i];
  }
  /*   computation of covariance matrix     */
  for(k=0;k<4;k++){
    for(l=0;l<4;l++){
      s=0.;
      for(i=0;i<nmxul;i++){
        s=s+ff[i][k]*ff[i][l]*errmat[i][i]*mask[i];
      }
      cov[k][l]=s;
    }
    s=0.;
    for(i=0;i<nmxul;i++){
      s=s+ff[i][k]*adc[i]*errmat[i][i]*mask[i];
    }
    bv[k]=s;
  }
  /*     parameters                          */
  inverms( 4, cov, invcov );
  for(k=0;k<4;k++){
    s=0.;
    for(l=0;l<4;l++){
      s=s+bv[l]*invcov[l][k];
    }
    parom[k]=s;
  }

  if( parom[3] == 0. ){
    parom[4] = -1000.;
    parom[5] = -1000.;
    parom[6] = -1000.;
    return 1000000.;
  }
  /*    worst hit and ki2                    */
  xki2=0.;
  difmx=0.;
  for(i=0;i<nmxul;i++ ){
    t2=t[i]*t[i];
    h= parom[0]+parom[1]*t[i]+parom[2]*t2+parom[3]*t2*t[i];
    dif=(adc[i]-h)*mask[i];
    xki2=xki2+dif*dif*errmat[i][i];
    if(dif > difmx) {
      difmx=dif;
    }
  }
  if(deglib > 0.5) xki2=xki2/deglib;
  /*     amplitude and maximum position                    */
  delta=parom[2]*parom[2]-3.*parom[3]*parom[1];
  if(delta > 0.){
    delta=sqrt(delta);
    tm=-(delta+parom[2])/(3.*parom[3]);
    tmp=(delta-parom[2])/(3.*parom[3]);
  }
  else{
    parom[4] = -1000.;
    parom[5] = -1000.;
    parom[6] = -1000.;
    return xki2;
  }
  parom[4]=tm;
  parom[5]= parom[0]+parom[1]*tm+parom[2]*tm*tm+parom[3]*tm*tm*tm;
  parom[6]=tmp;
  return xki2;
}
double TSFit::inverms( int n, double g[matdim][matdim], double ginv[matdim][matdim] ){
  // inversion of a positive definite symetric matrix of size n

  int i, j, k, jj;
  double r,  s;
  double deter=0;

  /*   initialisation  */

  if( n > matdim ){
    printf(
    "ERROR : trying to use TSFit::inverms with size %d( max allowed %d\n",
    n, matdim );
    return -999.;
  }

  int zero = 0;
  memset( (char *)al, zero, 8*n*n );
  memset( (char *)be, zero, 8*n*n );
  /*
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      al[i][j] = 0.;
      be[i][j] = 0.;
    }
  }
  */
  /*  decomposition en vecteurs sur une base orthonormee  */
  al[0][0] =  sqrt( g[0][0] );
  for(i=1;i<n;i++){
    al[i][0] = g[0][i] / al[0][0];
    for(j=1;j<=i;j++){
      s=0.;
      for(k=0;k<=j-1;k++){
	s = s + al[i][k] * al[j][k];
      }
      r= g[i][j] - s;
      if( j < i )  al[i][j] = r / al[j][j];
      if( j == i ) al[i][j] = sqrt( r );
    }
  }
  /*  inversion de la matrice al   */
  be[0][0] = 1./al[0][0];
  for(i=1;i<n;i++){
    be[i][i] = 1. / al[i][i];
    for(j=0;j<i;j++){
      jj=i-j-1;
      s=0.;
      for(k=jj+1;k<=i;k++){
	s=s+ be[i][k] * al[k][jj];
      }
      be[i][jj]=-s/al[jj][jj];
    }
  }
  /*   calcul de la matrice ginv   */
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      s=0.;
      for(k=0;k<n;k++){
	s=s+ be[k][i] * be[k][j];
      }
      ginv[i][j]=s;
    }
  }

  return deter;
}

double TSFit::fit_third_degree_polynomial(
		       double *bdc,
		       double *ret_dat ){
  //  third degree polynomial fit of the pulse summit. 
  //  samples are contained in array bdc and must be pedestal
  //  substracted.
  //  only samples having sample_flag >= 1 are used.
  //  the unit of time is one clock unit, that is to say 25 ns.
  //  output: ret_dat[0] = pulse height
  //          ret_dat[1]   position of maximum in the sample frame in clock units
  //          ret_dat[2]   adc value of the highest sample
  //          ret_dat[3]   number of the highest sample
  //          ret_dat[4]   lower sample number used for fitting
  //          ret_dat[5]   upper sample number used for fitting
  // errmat  inverse of the error matrix

  int i;
  int nus;
  double xki2;
  double tm, tmp, amp;

  static double nevt;

  ret_dat[0] = -999.;
  ret_dat[1] = -999.;

  //    search the maximum
  double val_max = 0.;
  int imax = 0;
  for(i=0;i<nbs;i++){
    if( sample_flag[i] == 0 )continue;
    if( bdc[i] > val_max ){
      val_max = bdc[i];
      imax = i;
    }
  }

  if( (val_max*val_max) * errmat[imax][imax] < 16. )return -118;

  //  if( imax != 9 )printf( "imax : %d !!!!!!!!!!!!!!!!!!!!!!!!!!!\n", imax );

  if( norme == 0. )norme = val_max;

  // look for samples above 1/3 of maximum before and 1/2 after
  double val2 = val_max / 2.;
  double val3 = val_max / 2.;
  int ilow = iinf;
  int ihig = 0;

  for(i=iinf;i<=isup;i++){
    if( sample_flag[i] >= 1 ){
      if( ( bdc[i] < val3 ) && ( i < imax ) )ilow = i;
      if( bdc[i] > val2 )ihig = i;
    }
  }

  ilow++;
  
  //ilow = imax - 1;

  /*  le test suivant, apparemment idiot, est mis a cause des sequences 0. 2048. qui apparaissent dans certains mauvais evts     JPP 11/09/00 */
 
  if( ihig == ilow)return -105;
  if( ilow == imax )ilow = ilow-1;
  //if( ihig - ilow < 3 )ihig = ilow + 3;
  ihig = ilow + 3;

 /*   printf("  third degree:   ilow %d ihig %d \n",ilow,ihig);  */
  nus=0;
  int number_of_good_samples = 0;
  for(i=ilow;i<=ihig;i++){
    maskp3[nus] = 0;
    adfmx[nus]  = 0.;
/*    printf(" adc %f sample_flag %d number_of good_samples %d \n",bdc[i],sample_flag[i],number_of_good_samples);  */
    if( sample_flag[i] >= 1 ){
      adfmx[nus] = bdc[i];
      maskp3[nus] = 1.;
      number_of_good_samples++;
    }
    nus++;
  }

  if( number_of_good_samples < 4 ){
    return( -106 );
  }

  xki2 = fpol3dg( nus, &parfp3[0], &maskp3[0], &adfmx[0]);
  
  /* printf( "fpol3dg-----------------------------------> %f %f %f %f %f\n",
	  parfp3[0], parfp3[1], parfp3[2], parfp3[3], parfp3[4] );  */

  tm = parfp3[4] + (float)ilow;
  amp = parfp3[5];

  if( amp * amp * errmat[0][0] < 2. )return -101.;
  tmp = parfp3[6] + (float)ilow;

  /*
    validation of fit quality.  Most of the time the fit is done with
    four samples, therefore there is no possible ki2 check. When more than
    4 samples are used the ki2 is often bad. So, in order to suppress some 
    events with bad samples, a consistency check on the position of the
    maximum and minimum of the 3rd degree polynomial is used.
  */

  if( xki2 > xki2_max ){
      return -102.;
  }
  if( (tm < (double)ilow ) || (tm > (double)ihig)){
    return  -103.;
  }

  if( (tmp > (double)ilow ) && (tmp < (double)ihig - 1.) ){
    return -104.;
  }

  nevt += 1.;

  ret_dat[0] = amp;
  ret_dat[1] = tm;
  ret_dat[2] = val_max;
  ret_dat[3] = (double)imax;
  ret_dat[4] = (double)ilow;
  ret_dat[5] = (double)ihig;
  ret_dat[6] = (double)tmp;
  int k;
  for(i=0;i<4;i++){
    k=i+7;
    ret_dat[k] = parfp3[i];
  }

  return xki2;
}













