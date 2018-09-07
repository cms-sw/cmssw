#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>

#include "mlp_gen.h"
#include "mlp_sigmoide.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NPMAX 100000
#define NLMAX 1000

struct net_ net_ MLP_HIDDEN;
struct learn_ learn_ MLP_HIDDEN;
struct pat_ pat_ MLP_HIDDEN;
struct divers_ divers_ MLP_HIDDEN;
struct stat_ stat_ MLP_HIDDEN;

int MessLang = 0;
int OutputWeights = 100;
int ExamplesMemory = 0;
int WeightsMemory = 0;
int PatMemory[2] = {0,0};
int BFGSMemory = 0;
int JacobianMemory = 0;
int LearnMemory = 0;
int NetMemory = 0;
float MLPfitVersion = (float) 1.40;
dbl LastAlpha = 0;
int NLineSearchFail = 0;

dbl ***dir;
dbl *delta;
dbl **BFGSH;
dbl *Gamma;
dbl **JacobianMatrix;
int *ExamplesIndex;
dbl **Hessian;

/* The following lines are needed to use the dgels routine from the LAPACK
   library in Reslin() 							    */

/* Subroutine */ int dgels_(char *trans, int *m, int *n, int *
	nrhs, double *a, int *lda, double *b, int *ldb, 
	double *work, int *lwork, int *info);
    
/***********************************************************/
/* MLP_Out                                                 */
/*                                                         */
/* computes the output of the Neural Network               */
/* inputs:     double *rrin = inputs of the MLP            */    
/* outputs:    double *rrout = outputs of the MLP          */    
/*                                                         */
/* Author: J.Schwindling   29-Mar-99                       */
/* Modified: J.Schwindling 16-Jul-99 unrolled loops        */
/* Modified: J.Schwindling 20-Jul-99 fast sigmoid          */
/***********************************************************/
   
/* extern "C"Dllexport */void MLP_Out(type_pat *rrin, dbl *rrout)
{
        int i, il, in, j, m, mp1;
	dbl **deriv1;

/* input layer */  

	deriv1 = NET.Deriv1;
	m = NET.Nneur[0]%4;
	if(m==0) goto L10;
	for(j=0;j<m;j++) NET.Outn[0][j] = rrin[j];
L10:	
	mp1 = m+1;
  	for(i=mp1; i<=NET.Nneur[0]; i+=4)
		{
    		NET.Outn[0][i-1] = rrin[i-1];
    		NET.Outn[0][i] = rrin[i];
    		NET.Outn[0][i+1] = rrin[i+1];
    		NET.Outn[0][i+2] = rrin[i+2];
		}
  
/* hidden and output layers */

	MLP_MatrixVectorBias(NET.vWeights[1],NET.Outn[0],
			NET.Outn[1],NET.Nneur[1],NET.Nneur[0]);
			
  	for(il=2; il<NET.Nlayer; il++)
		{
		MLP_vSigmoideDeriv(NET.Outn[il-1],
				deriv1[il-1],NET.Nneur[il-1]);
		MLP_MatrixVectorBias(NET.vWeights[il],NET.Outn[il-1],
				NET.Outn[il],NET.Nneur[il],
				NET.Nneur[il-1]);			
 		}	 
	for(in=0; in<NET.Nneur[NET.Nlayer-1]; in++)
		{								 
		deriv1[NET.Nlayer-1][in] = 1;
		}		
}


/***********************************************************/
/* MLP_Out_T                                               */
/*                                                         */
/* computes the output of the Neural Network when called   */
/* from MLP_Test					   */
/* inputs:     double *rrin = inputs of the MLP            */    
/*                                                         */
/* Author: J.Schwindling   23-Jul-1999                     */
/***********************************************************/
   
/* extern "C"Dllexport */void MLP_Out_T(type_pat *rrin)
{
        int i, il, in, j, ilm1, m, mp1;
	dbl a;

/* input layer */  

	m = NET.Nneur[0]%4;
	if(m==0) goto L10;
	for(j=0;j<m;j++) NET.Outn[0][j] = rrin[j];
L10:	
	mp1 = m+1;
  	for(i=mp1; i<=NET.Nneur[0]; i+=4)
		{
    		NET.Outn[0][i-1] = rrin[i-1];
    		NET.Outn[0][i] = rrin[i];
    		NET.Outn[0][i+1] = rrin[i+1];
    		NET.Outn[0][i+2] = rrin[i+2];
		}
  
/* hidden and output layers */

/*	for(in=0;in<NET.Nneur[0]; in++) printf("%e %e\n",
		NET.Outn[0][in],NET.Weights[1][0][in]);
		printf("\n");  */
  	for(il=1; il<NET.Nlayer; il++)
		{
		ilm1 = il-1;
 		m = NET.Nneur[ilm1]%4;
   		for(in=0; in<NET.Nneur[il]; in++)
			{
      			a = NET.Weights[il][in][0];
			if(m==0) goto L20;
			for(j=1;j<=m;j++) a += 
				NET.Weights[il][in][j]*NET.Outn[ilm1][j-1];
L20:			
			mp1 = m+1;
      			for(j=mp1; j<=NET.Nneur[ilm1]; j+=4)
				{
				a += 
				NET.Weights[il][in][j+3]*NET.Outn[ilm1][j+2]+
				NET.Weights[il][in][j+2]*NET.Outn[ilm1][j+1]+
				NET.Weights[il][in][j+1]*NET.Outn[ilm1][j]+
				NET.Weights[il][in][j]*NET.Outn[ilm1][j-1];
				}
		 	switch(NET.T_func[il][in])
	 			{
				case 2: NET.Outn[il][in] = MLP_Sigmoide(a);
					break;					
				case 1: NET.Outn[il][in] = a; 
					break; 
				case 0: NET.Outn[il][in] = 0; 
					break; 
				}	
     			}
		}	 
}


/***********************************************************/
/* MLP_Out2                                                 */
/*                                                         */
/* computes the output of the Neural Network               */
/* inputs:     double *rrin = inputs of the MLP            */    
/* outputs:    double *rrout = outputs of the MLP          */    
/*                                                         */
/* Author: J.Schwindling   29-Mar-99                       */
/* Modified: J.Schwindling 16-Jul-99 unrolled loops        */
/* Modified: J.Schwindling 20-Jul-99 fast sigmoid          */
/***********************************************************/
   
/* extern "C"Dllexport */void MLP_Out2(type_pat *rrin)
{
  	int il, in, m, mp1;
	int i;
	dbl **rrout, **deriv1;
	dbl *prrout;
	type_pat *prrin;
	int nhid = NET.Nneur[1];
	int nin = NET.Nneur[0];

	rrout = NET.Outn;
	deriv1 = NET.Deriv1;
	
	m = NET.Nneur[0]%4;
	if(m==0) goto L10;
	if(m==1) 
		{
		rrout[0][0] = rrin[1];
		goto L10;
		}
	else if(m==2)
		{
		rrout[0][0] = rrin[1];
		rrout[0][1] = rrin[2];
		goto L10;
		}	
	else if(m==3)
		{
		rrout[0][0] = rrin[1];
		rrout[0][1] = rrin[2];
		rrout[0][2] = rrin[3];
		goto L10;
		}	
L10:	
	mp1 = m+1;
	prrout = &(rrout[0][mp1]);
	prrin = &(rrin[mp1+1]);
  	for(i=mp1; i<=NET.Nneur[0]; i+=4, prrout+=4, prrin+=4)
		{
		*(prrout-1) = *(prrin-1);
		*prrout = *prrin;
		*(prrout+1)= *(prrin+1);
		*(prrout+2) = *(prrin+2);
		}
		
/* input layer */ 

  	MLP_MatrixVectorBias(NET.vWeights[1],NET.Outn[0],
		NET.Outn[1],nhid,nin);
	
     	  
/* hidden and output layers */

  	for(il=2; il<NET.Nlayer; il++)
		{
		MLP_vSigmoideDeriv(NET.Outn[il-1],deriv1[il-1],NET.Nneur[il-1]);
  		MLP_MatrixVectorBias(NET.vWeights[il],NET.Outn[il-1],
			NET.Outn[il],NET.Nneur[il],NET.Nneur[il-1]);
		}	
	for(in=0; in<NET.Nneur[NET.Nlayer-1]; in++)
		deriv1[NET.Nlayer-1][in] = 1;	 
}


/***********************************************************/
/* MLP_Test_MM                                             */
/*                                                         */
/* computes the MLP error function using matrix-matrix     */
/* multiplications		                	   */
/* inputs:     int ifile = file number: 0=learn, 1=test    */
/*	       dbl *tmp = a pointer to an array of size    */
/*                        2 x number of neurons in first   */
/*			  hidden layer			   */ 	
/*                                                         */
/* return value (dbl) = error value                        */ 
/*                                                         */
/* Author: J.Schwindling   25-Jan-2000                     */
/***********************************************************/
   
dbl MLP_Test_MM(int ifile, dbl *tmp)
{
	int ipat;
	int npat = PAT.Npat[ifile];
	int nhid = NET.Nneur[1];
	int nin = NET.Nneur[0];
	int jpat, j, il, ilm1, m, in, mp1;
	dbl err, a, rrans;
	dbl *pweights, *ptmp;

	err = 0;
	for(ipat=0; ipat<npat-1; ipat+=2)
		{
		MLP_MM2rows(tmp, &(PAT.vRin[ifile][ipat*(nin+1)]), 
		        NET.vWeights[1], 2, nhid, nin+1, 
			nin+1, nin+1);
	
		switch(NET.T_func[1][0])
	 		{
			case 2: 
			ptmp = &(tmp[0]);
			MLP_vSigmoide(ptmp,2*nhid);	
			break;
					
			case 1: 
			break;
					 
			case 0: 
			for(jpat=0; jpat<2; jpat++)
				{
				for(j=0; j<nhid; j++)
					{	
					tmp[j+jpat*nhid] = 0;
					}
				}
			break; 
			}
		
		for(jpat=0; jpat<2; jpat++)
		{
		for(in=0; in<nhid; in++) 
			{
			NET.Outn[1][in] = tmp[jpat*nhid+in];
			}
  		for(il=2; il<NET.Nlayer; il++)
			{
			ilm1 = il-1;
 			m = NET.Nneur[ilm1]%4;
   			for(in=0; in<NET.Nneur[il]; in++)
				{
				pweights = &(NET.Weights[il][in][0]);
      				a = *pweights;
				pweights++;
				if(m==0) goto L20;
				for(j=1;j<=m;j++,pweights++) a += 
				(*pweights)*NET.Outn[ilm1][j-1];
L20:			
				mp1 = m+1;
      				for(j=mp1; j<=NET.Nneur[ilm1]; 
					j+=4, pweights+=4)
					{
					a += 
				*(pweights+3)*NET.Outn[ilm1][j+2]+
				*(pweights+2)*NET.Outn[ilm1][j+1]+
				*(pweights+1)*NET.Outn[ilm1][j]+
				*(pweights  )*NET.Outn[ilm1][j-1];
					}
		 		switch(NET.T_func[il][in])
	 			{
				case 2: NET.Outn[il][in] = MLP_Sigmoide(a);
					break;					
				case 1: NET.Outn[il][in] = a; 
					break; 
				case 0: NET.Outn[il][in] = 0; 
					break; 
				}	
     				}
			if(il == NET.Nlayer-1)
			{
			for(in=0; in<NET.Nneur[NET.Nlayer-1]; in++) 
			{
			rrans = (dbl) PAT.Rans[ifile][ipat+jpat][in];
			err  += (rrans-NET.Outn[NET.Nlayer-1][in])*
			        (rrans-NET.Outn[NET.Nlayer-1][in])*
				PAT.Pond[ifile][ipat+jpat];
			} 
			}
			}
		}		
	}
	
/* cas npat impair */	
	for(/*'ipat' set above*/; ipat<npat; ipat++)
		{
		MLP_MatrixVector(NET.vWeights[1],
				&(PAT.vRin[ifile][ipat*(nin+1)]),tmp,
				nhid,nin+1);
	
		switch(NET.T_func[1][0])
	 		{
			case 2: 
			ptmp = &(tmp[0]);
			MLP_vSigmoide(ptmp,2*nhid);	
			break;
					
			case 1: 
			break;
					 
			case 0: 
			for(j=0; j<nhid; j++)
				{	
				tmp[j] = 0;
				}
			break; 
			}
		
		for(in=0; in<nhid; in++) 
			{
			NET.Outn[1][in] = tmp[in];
			}
  		for(il=2; il<NET.Nlayer; il++)
			{
			ilm1 = il-1;
 			m = NET.Nneur[ilm1]%4;
   			for(in=0; in<NET.Nneur[il]; in++)
				{
				pweights = &(NET.Weights[il][in][0]);
      				a = *pweights;
				pweights++;
				if(m==0) goto L25;
				for(j=1;j<=m;j++,pweights++) a += 
				(*pweights)*NET.Outn[ilm1][j-1];
L25:			
				mp1 = m+1;
      				for(j=mp1; j<=NET.Nneur[ilm1]; 
					j+=4, pweights+=4)
					{
					a += 
				*(pweights+3)*NET.Outn[ilm1][j+2]+
				*(pweights+2)*NET.Outn[ilm1][j+1]+
				*(pweights+1)*NET.Outn[ilm1][j]+
				*(pweights  )*NET.Outn[ilm1][j-1];
					}
		 		switch(NET.T_func[il][in])
	 			{
				case 2: NET.Outn[il][in] = MLP_Sigmoide(a);
					break;					
				case 1: NET.Outn[il][in] = a; 
					break; 
				case 0: NET.Outn[il][in] = 0; 
					break; 
				}	
     				}
			if(il == NET.Nlayer-1)
			{
			for(in=0; in<NET.Nneur[NET.Nlayer-1]; in++) 
			{
			rrans = (dbl) PAT.Rans[ifile][ipat][in];
			err  += (rrans-NET.Outn[NET.Nlayer-1][in])*
			        (rrans-NET.Outn[NET.Nlayer-1][in])*
				PAT.Pond[ifile][ipat];
			} 
			}
		}		
	}
	return(err);	
}


/***********************************************************/
/* MLP_Test                                                */
/*                                                         */
/* computes the MLP error function                         */
/* inputs:     int ifile = file number: 0=learn, 1=test    */
/*             int regul = 0: no regularisation term       */
/*                         1: regularisation term          */
/*                            (for hybrid learning method) */     
/*                                                         */
/* return value (dbl) = error value                        */ 
/*                                                         */
/* Author: J.Schwindling   31-Mar-99                       */
/***********************************************************/
   
/* extern "C"Dllexport */dbl MLP_Test(int ifile,int regul)
{
	dbl err, rrans;
	int in,jn,ipat,ipati;
	
	dbl *tmp;
	
	tmp = (dbl *) malloc(2 * NET.Nneur[1] * sizeof(dbl)); 
	if(tmp == nullptr)	/* not enough memory */
	{
	printf("not enough memory in MLP_Test\n");				
	err = 0;
	for(ipat=0; ipat<PAT.Npat[ifile]; ipat++)
		{
		if(ifile==0)
			{
			ipati = ExamplesIndex[ipat];
			}
		else
			{
			ipati = ipat;
			}	
		MLP_Out_T(PAT.Rin[ifile][ipati]);
		for(in=0; in<NET.Nneur[NET.Nlayer-1]; in++) 
			{
			rrans = (dbl) PAT.Rans[ifile][ipati][in];
			err  += (rrans-NET.Outn[NET.Nlayer-1][in])*
			        (rrans-NET.Outn[NET.Nlayer-1][in])*
				PAT.Pond[ifile][ipati];
			}
		}

	if(regul>=1) 
		{
		for(in=0; in<NET.Nneur[NET.Nlayer-1]; in++)
			for(jn=0; jn<=NET.Nneur[NET.Nlayer-2]; jn++)
			{
			err += LEARN.Alambda*NET.Weights[NET.Nlayer-1][in][jn]*
				NET.Weights[NET.Nlayer-1][in][jn];
			}
		}
	free(tmp);	
	return(err);
	}
	else 	/* computation using matrix - matrix multiply */
	{
	err = MLP_Test_MM(ifile, tmp);
	if(regul>=1) 
		{
		for(in=0; in<NET.Nneur[NET.Nlayer-1]; in++)
			for(jn=0; jn<=NET.Nneur[NET.Nlayer-2]; jn++)
			{
			err += LEARN.Alambda*NET.Weights[NET.Nlayer-1][in][jn]*
				NET.Weights[NET.Nlayer-1][in][jn];
			}
		}
	free(tmp);
	return(err);
	}		
}

   
/***********************************************************/
/* MLP_Stochastic                                          */
/*                                                         */
/* one epoch of MLP stochastic training                    */
/* (optimized for speed)				   */
/*                                                         */
/* Author: J.Schwindling   08-Jul-99                       */
/* Modified: J.Schwindling 20-Jul-99 remove unused cases   */
/***********************************************************/
   
/* extern "C"Dllexport */dbl MLP_Stochastic()
{
	int ipat, ii, inm1;
	dbl err = 0;
	int il, in1, in, itest2;
	dbl deriv, deriv1, deriv2, deriv3, deriv4, pond;
	dbl eta, eps;
	dbl a, b, dd, a1, a2, a3, a4;
	dbl *pout, *pdelta, *pw1, *pw2, *pw3, *pw4;
	dbl ***weights;
    
	if(NET.Debug>=5) printf(" Entry MLP_Stochastic\n");		
	weights = NET.Weights;
/* shuffle patterns */		
	ShuffleExamples(PAT.Npat[0],ExamplesIndex); 
		
/* reduce learning parameter */
	if(LEARN.Decay<1) EtaDecay();
		
	eta = -LEARN.eta;
	eps = LEARN.epsilon;
	
/* loop on the examples */		
	for(ipat=0;ipat<PAT.Npat[0];ipat++)
		{
		ii = ExamplesIndex[ipat];
		pond = PAT.Pond[0][ii];
		
   		MLP_Out2(&(PAT.vRin[0][ii*(NET.Nneur[0]+1)])); 
			
/* next lines are equivalent to DeDwSum */			    
 		for(in=0; in<NET.Nneur[NET.Nlayer-1]; in++) 
			{
			deriv = NET.Deriv1[NET.Nlayer-1][in]; 
			a = (dbl) PAT.Rans[0][ii][in];
			b = NET.Outn[NET.Nlayer-1][in]-a;
			err  += b*b*pond;
			NET.Delta[NET.Nlayer-1][in] = b*deriv*pond*eta;
			}
			
   		for(il=NET.Nlayer-2; il>0; il--) 
			{
			dd = NET.Delta[il+1][0];
			for(in=0; in<NET.Nneur[il]-3; in+=4) 
				{
				deriv1 = NET.Deriv1[il][in]; 
				deriv2 = NET.Deriv1[il][in+1]; 
				deriv3 = NET.Deriv1[il][in+2]; 
				deriv4 = NET.Deriv1[il][in+3]; 
				itest2 = (NET.Nneur[il+1]==1);
				a1 = dd*weights[il+1][0][in+1];
				a2 = dd*weights[il+1][0][in+2];
				a3 = dd*weights[il+1][0][in+3];
				a4 = dd*weights[il+1][0][in+4];
				if(itest2) goto L1;
				pdelta = &(NET.Delta[il+1][1]);
				for(in1=1; in1<NET.Nneur[il+1];
					in1++, pdelta++) 
					{
					a1 += *pdelta * weights[il+1][in1][in+1];
					a2 += *pdelta * weights[il+1][in1][in+2];
					a3 += *pdelta * weights[il+1][in1][in+3];
					a4 += *pdelta * weights[il+1][in1][in+4];
					}
L1:				NET.Delta[il][in] = a1*deriv1;
				NET.Delta[il][in+1] = a2*deriv2;
				NET.Delta[il][in+2] = a3*deriv3;
				NET.Delta[il][in+3] = a4*deriv4;
				} 
			for(/*'in' set above*/; in<NET.Nneur[il]; in++) 
				{
				deriv = NET.Deriv1[il][in]; 
				itest2 = (NET.Nneur[il+1]==1);
				a = dd*weights[il+1][0][in+1];
				if(itest2) goto L2;
				pdelta = &(NET.Delta[il+1][1]);
				for(in1=1; in1<NET.Nneur[il+1];
					in1++, pdelta++) 
					{
					a += *pdelta * 
					weights[il+1][in1][in+1];
					}
L2:				NET.Delta[il][in] = a*deriv;
				} 
				
			}			/* end of loop on layers */


/* update the weights */
		if(eps==0)
			{
    		for(il=1; il<NET.Nlayer; il++)
			{
			inm1 = NET.Nneur[il-1];
			for(in=0; in<NET.Nneur[il]-3; in+=4)
				{ 
				a1 = NET.Delta[il][in];
				a2 = NET.Delta[il][in+1];
				a3 = NET.Delta[il][in+2];
				a4 = NET.Delta[il][in+3];
				pout = &(NET.Outn[il-1][0]);
				weights[il][in][0] += a1;				
				weights[il][in+1][0] += a2;				
				weights[il][in+2][0] += a3;				
				weights[il][in+3][0] += a4;				
			        weights[il][in][1] += a1* (*pout);
			        weights[il][in+1][1] += a2* (*pout);
			        weights[il][in+2][1] += a3* (*pout);
			        weights[il][in+3][1] += a4* (*pout);
				pout++;
				pw1 = &(weights[il][in][2]);
				pw2 = &(weights[il][in+1][2]);
				pw3 = &(weights[il][in+2][2]);
				pw4 = &(weights[il][in+3][2]);
				for(in1=2; in1<=inm1; 
					++in1, ++pout, ++pw1, ++pw2,
					++pw3, ++pw4)
					{
			        	*pw1 += a1 * *pout;
			        	*pw2 += a2 * *pout;
			        	*pw3 += a3 * *pout;
			        	*pw4 += a4 * *pout;
					}
				}
			for(/*'in' set above*/; in<NET.Nneur[il]; in++)
				{ 
				a1 = NET.Delta[il][in];
				pout = &(NET.Outn[il-1][0]);
				weights[il][in][0] += a1;				
			        weights[il][in][1] += a1* (*pout);
				pout++;
				pw1 = &(weights[il][in][2]);
				for(in1=2; in1<=inm1; 
					++in1, ++pout, ++pw1)
					{
			        	*pw1 += a1 * *pout;
					}
				}
			}
			}
		else
			{			
    		for(il=1; il<NET.Nlayer; il++)
			{
			for(in=0; in<NET.Nneur[il]; in++)
				{ 
				
				a = NET.Delta[il][in];
				LEARN.Odw[il][in][0] = a + eps * LEARN.Odw[il][in][0];
				NET.Weights[il][in][0] += LEARN.Odw[il][in][0];
				
				b = a*NET.Outn[il-1][0];
				LEARN.Odw[il][in][1] = b + eps*LEARN.Odw[il][in][1];
			        NET.Weights[il][in][1] += LEARN.Odw[il][in][1];
				
				for(in1=2; in1<=NET.Nneur[il-1]; in1++)
					{
					b = a*NET.Outn[il-1][in1-1];
					LEARN.Odw[il][in][in1] = b + eps*LEARN.Odw[il][in][in1];
			        	NET.Weights[il][in][in1] += LEARN.Odw[il][in][in1];
					}
				}
			}
			}		
			
		}			/* end of loop on examples */
	return(err);		
}


/***********************************************************/
/* MLP_Epoch                                               */
/*                                                         */
/* one epoch of MLP training                               */
/* inputs:     int iepoch = epoch number                   */
/*             dbl *alpmin = optimal alpha from Line Search*/
/*                                                         */
/* return value (dbl) = error value on learning sample     */
/*                      BEFORE changing the weights        */  
/*                                                         */
/* Author: J.Schwindling   31-Mar-99                       */
/* Modified: J.Schwindling 09-Apr-99                       */
/*                re-organize routine                      */
/*           J.Schwindling 13-Apr-99                       */
/*                remove Quickprop algorithm               */
/*                implement Ribiere-Polak conjugate grad.  */  
/***********************************************************/
   
/* extern "C"Dllexport */dbl MLP_Epoch(int iepoch, dbl *alpmin, int *Ntest)
{
	dbl err, ONorm, beta, prod, ddir;	
/*	int *index;*/
	int Nweights, Nlinear, ipat, ierr;
	int nn;
	
	err = 0;
	*alpmin = 0.;
	
	Nweights = NET.Nweights;
	Nlinear = NET.Nneur[NET.Nlayer-2] + 1;
	
	if(NET.Debug>=5) printf(" Entry MLP_Epoch\n");		
/* stochastic minimization */		
	if(LEARN.Meth==1) 
		{

		err = MLP_Stochastic();
			
		}
	else
		{
		if(iepoch==1 && LEARN.Meth==7)
			{
			SetLambda(10000);
			MLP_ResLin();
			if(NET.Debug>=2) PrintWeights();
			}
			
/* save previous gradient and reset current one */
		DeDwSaveZero();
		if(LEARN.Meth==16) 
			{			
			ShuffleExamples(PAT.Npat[0],ExamplesIndex);
			nn = PAT.Npat[0];
			PAT.Npat[0] = nn/10;
			for(ipat=0;ipat<nn;ipat++)
				{
				ierr = MLP_Train(&ExamplesIndex[ipat],&err);
				if(ierr!=0) printf("Epoch: ierr= %d\n",ierr);
				}
			} 
		else
			{
			for(ipat=0;ipat<PAT.Npat[0];ipat++)
				{
				ierr = MLP_Train(&ipat,&err);
				if(ierr!=0) printf("Epoch: ierr= %d\n",ierr);
				}
			}
		DeDwScale(PAT.Npat[0]);
		if(LEARN.Meth==2) StochStep();
		if(LEARN.Meth==3) 
			{
			SteepestDir(); 
			if(LineSearch(alpmin,Ntest,err)==1) StochStep();
			}

/* Conjugate Gradients Ribiere - Polak */
		if(LEARN.Meth==4) 
			{
			if((iepoch-1)%LEARN.Nreset==0) 
				{
				LEARN.Norm = DeDwNorm(); /* for next epoch */
				SteepestDir();
				}
			else
				{
				ONorm = LEARN.Norm;
				LEARN.Norm = DeDwNorm();
				prod = DeDwProd();
				beta = (LEARN.Norm-prod)/ONorm;
				CGDir(beta);
				}
			if(LineSearch(alpmin,Ntest,err)==1) StochStep();
			}

/* Conjugate Gradients Fletcher - Reeves */
		if(LEARN.Meth==5) 
			{
			if((iepoch-1)%LEARN.Nreset==0) 
				{
				LEARN.Norm = DeDwNorm(); /* for next epoch */
				SteepestDir();
				}
			else
				{
				ONorm = LEARN.Norm;
				LEARN.Norm = DeDwNorm();
				beta = LEARN.Norm/ONorm;
				CGDir(beta);
				}
			if(LineSearch(alpmin,Ntest,err)==1) StochStep();
			}
		if(LEARN.Meth==6)
			{
			if((iepoch-1)%LEARN.Nreset==0)
				{
				SteepestDir();
				InitBFGSH(Nweights);
				}
			else
				{
				GetGammaDelta();
				ierr = GetBFGSH(Nweights);
				if(ierr)
					{
					SteepestDir();
					InitBFGSH(Nweights);
					}
				else
					{	
					BFGSdir(Nweights);
					}
				}
			ddir = DerivDir();
			if(ddir>0)
				{
				SteepestDir();
				InitBFGSH(Nweights);
				ddir = DerivDir();
				}
			if(LineSearch(alpmin,Ntest,err)==1) 
				{
				InitBFGSH(Nweights);
				SteepestDir();
				if(LineSearch(alpmin,Ntest,err)==1) 
					{
					printf("Line search fail \n");
					}
				}
			}
		if(LEARN.Meth==7)
			{
			if((iepoch-1)%LEARN.Nreset==0)
				{
				SteepestDir();
				InitBFGSH(Nweights-Nlinear);
				}
			else
				{
			if(NET.Debug>=5) printf("Before GetGammaDelta \n");
				GetGammaDelta();
			if(NET.Debug>=5) printf("After GetGammaDelta \n");
				ierr = GetBFGSH(Nweights-Nlinear);
			if(NET.Debug>=5) printf("After GetBFGSH \n");
				if(ierr)
					{
					SteepestDir();
					InitBFGSH(Nweights-Nlinear);
					}
				else
					{	
					BFGSdir(Nweights-Nlinear);
					}
			if(NET.Debug>=5) printf("After BFGSdir \n");
				}
		 	SetLambda(10000);
			if(LineSearchHyb(alpmin,Ntest)==1)
				{
				InitBFGSH(Nweights-Nlinear);
				SteepestDir();
				if(LineSearchHyb(alpmin,Ntest)==1) 
					{
					printf("Line search fail \n");
					}
				}
			}
		}
	
	if(NET.Debug>=5) printf(" End MLP_Epoch\n");		
	return(err);
}


/***********************************************************/
/* MLP_Train                                               */
/*                                                         */
/* Train Network: compute output, update DeDw              */
/* inputs:     int *ipat = pointer to pattern number       */
/* input/output:    dbl *err = current error               */
/*                                                         */
/* return value (int) = error value: 1 wrong pattern number*/
/*                                   2 *ipat < 0           */   
/*                                                         */
/* Author: J.Schwindling   09-Apr-99                       */
/***********************************************************/
   
/* extern "C"Dllexport */int MLP_Train(int *ipat, dbl *err)
{
	int in;
    
/*    	if(*ipat>=PAT.Npat[0]) return(1);*/
	if(*ipat<0) return(2);
        
/*    	MLP_Out(PAT.Rin[0][*ipat],NET.Outn[NET.Nlayer-1]);*/
    	MLP_Out2(&(PAT.vRin[0][*ipat*(NET.Nneur[0]+1)]));
	for(in=0; in<NET.Nneur[NET.Nlayer-1]; in++) 
		{
		*err  += ((dbl) PAT.Rans[0][*ipat][in]-NET.Outn[NET.Nlayer-1][in])
			*((dbl) PAT.Rans[0][*ipat][in]-NET.Outn[NET.Nlayer-1][in])*
			PAT.Pond[0][*ipat];
		}
    	DeDwSum(PAT.Rans[0][*ipat],NET.Outn[NET.Nlayer-1],*ipat); 	
    	return(0); 
} 

      	  
/***********************************************************/
/* StochStepHyb                                            */
/*                                                         */
/* Update the weights according to stochastic minimization */
/* formula (for hybrid methods)                            */
/*                                                         */
/* return value (int) = 0				   */
/*                                                         */
/* Author: J.Schwindling   09-Apr-99                       */
/***********************************************************/
   
/* extern "C"Dllexport */int StochStepHyb()
{
	int il, in1, in;
	dbl eta, eps;
    
	eta = LEARN.eta;
	eps = LEARN.epsilon;
    for(il=NET.Nlayer-2; il>0; il--) {

	for(in=0; in<NET.Nneur[il]; in++) {

		/* compute delta weights */
		for(in1=0; in1<=NET.Nneur[il-1]; in1++) {
			LEARN.Odw[il][in][in1] = -eta * LEARN.DeDw[il][in][in1]
			 + eps * LEARN.Odw[il][in][in1];
		}
		
		/* update weights */
		for(in1=0; in1<=NET.Nneur[il-1]; in1++) {
			NET.Weights[il][in][in1] += LEARN.Odw[il][in][in1];
		}
	}	
	}	
    MLP_ResLin();
    return(0); 
} 
      	  

/***********************************************************/
/* StochStep                                               */
/*                                                         */
/* Update the weights according to stochastic minimization */
/*                                            formula      */
/*                                                         */
/* return value (int) = 0				   */
/*                                                         */
/* Author: J.Schwindling   09-Apr-99                       */
/***********************************************************/
   
/* extern "C"Dllexport */int StochStep()
{
	int il, in1, in;
	dbl eta, eps, epseta;
    
		eta = -LEARN.eta;
		eps = LEARN.epsilon;
		epseta = eps/eta;
    for(il=NET.Nlayer-1; il>0; il--) {
		for(in1=0; in1<=NET.Nneur[il-1]; in1++) {

		/* compute delta weights */
	for(in=0; in<NET.Nneur[il]; in++) {
			LEARN.Odw[il][in][in1] = eta * (LEARN.DeDw[il][in][in1]
			 + epseta * LEARN.Odw[il][in][in1]);
			NET.Weights[il][in][in1] += LEARN.Odw[il][in][in1];
		}
		
	}	
	}	

    return(0); 
}       	  


/***********************************************************/
/* DeDwNorm                                                */
/*                                                         */
/* computes the norm of the gradient                       */
/*                                                         */
/* Author: J.Schwindling   31-Mar-99                       */
/***********************************************************/
   
/* extern "C"Dllexport */dbl DeDwNorm()
{
        int il,in,jn;
	dbl dd=0;
	for(il=1; il<NET.Nlayer; il++)
		for(in=0; in<NET.Nneur[il]; in++)
			for(jn=0; jn<=NET.Nneur[il-1]; jn++)
				dd += LEARN.DeDw[il][in][jn]*
					LEARN.DeDw[il][in][jn];
	return(dd);
}


/***********************************************************/
/* DeDwProd                                                */
/*                                                         */
/* scalar product between new gradient and previous one    */
/* (used in Polak-Ribiere Conjugate Gradient method        */
/*                                                         */
/* Author: J.Schwindling   26-Mar-99                       */
/***********************************************************/
   
/* extern "C"Dllexport */dbl DeDwProd()
{
        int il,in,jn;
	dbl dd=0;
	for(il=1; il<NET.Nlayer; il++)
		for(in=0; in<NET.Nneur[il]; in++)
			for(jn=0; jn<=NET.Nneur[il-1]; jn++)
				dd += LEARN.DeDw[il][in][jn]*
					LEARN.ODeDw[il][in][jn];
	return(dd);
}


/***********************************************************/
/* DeDwZero                                                */
/*                                                         */
/* resets to 0 the gradient (should be done before DeDwSum)*/
/*                                                         */
/* Author: J.Schwindling   31-Mar-99                       */
/***********************************************************/   

/* extern "C"Dllexport */void DeDwZero()
{
	int il, in, jn;
	for(il=1; il<NET.Nlayer; il++)
		for(in=0; in<NET.Nneur[il]; in++)
			for(jn=0; jn<=NET.Nneur[il-1]; jn++)
				LEARN.DeDw[il][in][jn] = 0;
}
	

/***********************************************************/
/* DeDwScale                                               */
/*                                                         */
/* divides the gradient by the number of examples          */
/* inputs:     int Nexamples = number of examples          */
/*                                                         */
/* Author: J.Schwindling   31-Mar-99                       */
/***********************************************************/   

/* extern "C"Dllexport */void DeDwScale(int Nexamples)
{
	int il, in, jn;
	for(il=1; il<NET.Nlayer; il++)
		for(in=0; in<NET.Nneur[il]; in++)
			for(jn=0; jn<=NET.Nneur[il-1]; jn++)
				LEARN.DeDw[il][in][jn] /= (dbl) Nexamples;
}	


/***********************************************************/
/* DeDwSave                                                */
/*                                                         */
/* copies the gradient DeDw into ODeDw                     */
/*                                                         */
/* Author: J.Schwindling   31-Mar-99                       */
/***********************************************************/   

/* extern "C"Dllexport */void DeDwSave()
{
	int il, in, jn;
	for(il=1; il<NET.Nlayer; il++)
		for(in=0; in<NET.Nneur[il]; in++)
			for(jn=0; jn<=NET.Nneur[il-1]; jn++)
				LEARN.ODeDw[il][in][jn] = LEARN.DeDw[il][in][jn];
}	


/***********************************************************/
/* DeDwSaveZero                                            */
/*                                                         */
/* copies the gradient DeDw into ODeDw                     */
/* resets DeDw to 0 					   */
/*                                                         */
/* Author: J.Schwindling   23-Apr-99                       */
/***********************************************************/   

/* extern "C"Dllexport */void DeDwSaveZero()
{
	int il, in, jn;
	for(il=1; il<NET.Nlayer; il++)
		for(in=0; in<NET.Nneur[il]; in++)
			for(jn=0; jn<=NET.Nneur[il-1]; jn++)
				{
				LEARN.ODeDw[il][in][jn] = LEARN.DeDw[il][in][jn];
				LEARN.DeDw[il][in][jn] = 0;
				}
}	


/***********************************************************/
/* DeDwSum                                                 */
/*                                                         */
/* adds to the total gradient the gradient in the current  */
/* example                                                 */
/* inputs:     int Nexamples = number of examples          */
/*                                                         */
/* Author: J.Schwindling   31-Mar-99                       */
/* Modified: B.Mansoulie   23-Apr-99                       */
/*           faster and still correct way to compute the   */
/*           sigmoid derivative                            */   
/***********************************************************/   

/* extern "C"Dllexport */int DeDwSum(type_pat *ans, dbl *out, int ipat)
{
	int il, in1, in, ii;
/*	dbl err[NMAX][4]; */
	dbl deriv;
	dbl *pout, *pdedw, *pdelta;
	dbl a, b;
/*	char buf[50];*/

/* output layer */ 
	b =  (dbl) PAT.Pond[0][ipat];  
	for(in=0; in<NET.Nneur[NET.Nlayer-1]; in++) 
	{	           
		deriv = NET.Deriv1[NET.Nlayer-1][in];
		NET.Delta[NET.Nlayer-1][in] = 
			(out[in] - (dbl) ans[in])*deriv*b;
	}
	
    for(il=NET.Nlayer-2; il>0; il--) 
    	{

	for(in=0; in<NET.Nneur[il]; in++) 
	{		           
		deriv = NET.Deriv1[il][in];
		a = NET.Delta[il+1][0] * NET.Weights[il+1][0][in+1];
		pdelta = &(NET.Delta[il+1][1]);
		for(in1=1; in1<NET.Nneur[il+1]; in1++, pdelta++) 
		{
			a += *pdelta * NET.Weights[il+1][in1][in+1];
		}
		NET.Delta[il][in] = a * deriv; 
	}
	}
		
    	for(il=1; il<NET.Nlayer; il++) 
    		{
		ii = NET.Nneur[il-1];
		for(in=0; in<NET.Nneur[il]; in++) 
			{
			a = NET.Delta[il][in];	
			LEARN.DeDw[il][in][0] += a;
			LEARN.DeDw[il][in][1] += a * NET.Outn[il-1][0];
			pout = &(NET.Outn[il-1][1]);
			pdedw = &(LEARN.DeDw[il][in][2]);
			for(in1=1; in1<ii; ++in1, ++pout, ++pdedw) 
				{
				(*pdedw) += a * (*pout);
				}
			}
		}		

    return(0); 
}


/***********************************************************/
/* SetTransFunc                                            */
/*                                                         */
/* to set the transfert function of a neuron               */
/* inputs:     int layer = layer number (1 -> Nlayer)      */
/*	       int neuron = neuron number (1 -> Nneur)	   */
/*             int func = transfert function:              */
/*     				0: y=0			   */ 	
/*     				1: y=x			   */ 	
/*     				2: y=1/(1+exp(-x)) 	   */ 	
/*     				3: y=tanh(x)		   */ 	
/*     				4: y=delta*x+1/(1+exp(-x)) */ 	
/*     				5: y=exp(-x**2)		   */ 	
/*                                                         */
/* return value (int) = error value:   		           */
/*                              0: no error		   */ 
/*                              1: layer > 4		   */ 
/*                              2: neuron > NMAX           */ 
/*                                                         */
/* Author: J.Schwindling   02-Apr-99                       */
/***********************************************************/   

/* extern "C"Dllexport */int SetTransFunc(int layer, int neuron, 
					  int func)
{    
    if(layer>NLMAX) return(1);
/*    if(neuron>NMAX) return(2);*/
         
    NET.T_func[layer-1][neuron-1] = func;

    return(0); 
}       	  


/***********************************************************/
/* SetDefaultFuncs                                         */
/*                                                         */
/* - sets the default transfer functions to sigmoid for    */
/* hidden layers and linear for output layer               */
/* - sets temperatures to 1                                */
/*                                                         */
/*                                                         */
/* Author: J.Schwindling   08-Apr-99                       */
/***********************************************************/   

void SetDefaultFuncs()
{
    int il,in;
    for(il=0; il<NET.Nlayer; il++) {
       for(in=0; in<NET.Nneur[il]; in++) {
          NET.T_func[il][in] = 2; 
	  if(il==NET.Nlayer-1) NET.T_func[il][in] = 1;
	 }
      }

}


/***********************************************************/
/* SteepestDir                                             */
/*                                                         */
/* sets the search direction to steepest descent           */
/*                                                         */
/* Author: J.Schwindling   02-Apr-99                       */
/***********************************************************/   

/* extern "C"Dllexport */void SteepestDir()
{
        int il,in,jn;
	for(il=1; il<NET.Nlayer; il++)
		for(in=0; in<NET.Nneur[il]; in++)
			for(jn=0; jn<=NET.Nneur[il-1]; jn++)
				dir[il][in][jn] = -LEARN.DeDw[il][in][jn];
}


/***********************************************************/
/* CGDir                                                   */
/*                                                         */
/* sets the search direction to conjugate gradient dir     */
/* inputs:     dbl beta : d(t+1) = -g(t+1) + beta d(t)     */
/*                        beta should be:                  */
/*    ||g(t+1)||^2 / ||g(t)||^2 (Fletcher-Reeves)          */
/*    g(t+1) (g(t+1)-g(t)) / ||g(t)||^2 (Polak-Ribiere)    */
/*                                                         */
/* Author: J.Schwindling   02-Apr-99                       */
/***********************************************************/   

/* extern "C"Dllexport */void CGDir(dbl beta)
{
        int il,in,jn;
	for(il=1; il<NET.Nlayer; il++)
		for(in=0; in<NET.Nneur[il]; in++)
			for(jn=0; jn<=NET.Nneur[il-1]; jn++)
				{
				dir[il][in][jn] = -LEARN.DeDw[il][in][jn]+
					beta*dir[il][in][jn];
				}
}


/***********************************************************/
/* DerivDir                                                */
/*                                                         */
/* scalar product between gradient and direction           */
/*     = derivative along direction                        */
/*                                                         */
/* Author: J.Schwindling   01-Jul-99                       */
/***********************************************************/   

/* extern "C"Dllexport */dbl DerivDir()
{
        int il,in,jn;
	dbl ddir = 0;
	
	for(il=1; il<NET.Nlayer; il++)
		for(in=0; in<NET.Nneur[il]; in++)
			for(jn=0; jn<=NET.Nneur[il-1]; jn++)
				{
				ddir += LEARN.DeDw[il][in][jn]*dir[il][in][jn];
				}
	return(ddir);			
}


/***********************************************************/
/* GetGammaDelta                                           */
/*                                                         */
/* sets the vectors Gamma (=g(t+1)-g(t)) 		   */
/*              and delta (=w(t+1)-w(t))	           */
/* (for BFGS and Hybrid learning methods)		   */
/*                                                         */
/* Author: J.Schwindling   02-Apr-99                       */
/***********************************************************/   

/* extern "C"Dllexport */void GetGammaDelta()
{
	int i=0;
        int il,in,jn;
	for(il=1; il<NET.Nlayer; il++)
		for(in=0; in<NET.Nneur[il]; in++)
			for(jn=0; jn<=NET.Nneur[il-1]; jn++)
				{
				Gamma[i] = LEARN.DeDw[il][in][jn]-
					LEARN.ODeDw[il][in][jn];
				delta[i] = LEARN.Odw[il][in][jn];
				i++;
				}
}


/***********************************************************/
/* BFGSDir                                                 */
/*                                                         */
/* sets the search direction to BFGS direction from the    */
/*                              BFGS matrix                */
/*                                                         */
/* inputs:     int Nweights = number of weights            */
/*                                                         */
/* Author: J.Schwindling   02-Apr-99                       */
/***********************************************************/   

/* extern "C"Dllexport */void BFGSdir(int Nweights)
{
	dbl *g, *s;
	int kk=0;
	int il,i,j,in,jn;
	
	g = (dbl*) malloc(NET.Nweights*sizeof(dbl));
	s = (dbl*) malloc(Nweights*sizeof(dbl));
	
	for(il=1; kk<Nweights; il++)
		for(in=0; in<NET.Nneur[il]; in++)
			for(jn=0; jn<=NET.Nneur[il-1]; jn++)
				{
				g[kk] = LEARN.DeDw[il][in][jn];
				kk++;
				}
	for(i=0; i<Nweights; i++)
		{
		s[i] = 0;
		for(j=0; j<Nweights; j++)
			{
			s[i] += BFGSH[i][j] * g[j];
			}
		}
	
	kk = 0;
	for(il=1; kk<Nweights; il++)
		for(in=0; in<NET.Nneur[il]; in++)
			for(jn=0; jn<=NET.Nneur[il-1]; jn++)
				{
				dir[il][in][jn] = -s[kk];
				kk++;
				}
	free(g);
	free(s);
}


/***********************************************************/
/* InitBFGS                                                */
/*                                                         */
/* initializes BFGS matrix to identity                     */
/*                                                         */
/* inputs:     int Nweights = number of weights            */
/*                                                         */
/* Author: J.Schwindling   02-Apr-99                       */
/***********************************************************/   

/* extern "C"Dllexport */void InitBFGSH(int Nweights)
{
	int i,j;
	for(i=0; i<Nweights; i++)
		for(j=0; j<Nweights; j++)
			{
			BFGSH[i][j] = 0;
			if(i==j) BFGSH[i][j] = 1;
			}
}


/***********************************************************/
/* GetBFGSH                                                */
/*                                                         */
/* sets the BFGS matrix 		                   */
/*                                                         */
/* inputs:     int Nweights = number of weights            */
/*                                                         */
/* return value (int) = 0 if no problem			   */
/*			1 is deltaTgamma = 0 -> switch to  */
/*				steepest dir		   */
/*                                                         */
/* Author: J.Schwindling   02-Apr-99                       */
/* Modified: J.Schwindling 04-May-99 			   */
/*           computations as Nw^2 , matrices removed       */
/* Modified: J.Schwindling 11-Jun-99 			   */
/*	     return value				   */
/***********************************************************/   

/* extern "C"Dllexport */int GetBFGSH(int Nweights)
{
        typedef double dble;
	dble deltaTgamma=0;
	dble factor=0; 
	dble *Hgamma;
	dble *tmp;
	dble a, b;
	int i,j;
	
	Hgamma = (dble *) malloc(Nweights*sizeof(dble));
	tmp = (dble *) malloc(Nweights*sizeof(dble));
	
	for(i=0; i<Nweights; i++)
		{
		deltaTgamma += (dble) delta[i] * (dble) Gamma[i];
		a = 0;
		b = 0;
		for(j=0; j<Nweights; j++)
			{
			a += (dble) BFGSH[i][j] * (dble) Gamma[j];
			b += (dble) Gamma[j] * (dble) BFGSH[j][i];
			}
		Hgamma[i] = a;
		tmp[i] = b;	
		factor += (dble) Gamma[i]*Hgamma[i];
		}
	if(deltaTgamma == 0) 
        {
          free(tmp);
          free(Hgamma);
          return 1;
        }
	a = 1 / deltaTgamma;	
	factor = 1 + factor*a;
	
	for(i=0; i<Nweights; i++)
		{
		b = (dble) delta[i];
		for(j=0; j<Nweights; j++)
			BFGSH[i][j] += (dbl) (factor*b* (dble) 
			delta[j]-(tmp[j]*b+Hgamma[i]*(dble)delta[j]))*a;	
		}	
	free(Hgamma);
	free(tmp);
	return 0;
}


/***********************************************************/
/* LineSearch                                              */
/*                                                         */
/* search along the line defined by dir                    */
/*                                                         */
/* outputs:     dbl *alpmin = optimal step length          */
/*                                                         */
/* Author: B.Mansoulie     01-Jul-98                       */
/***********************************************************/   

/* extern "C"Dllexport */
int LineSearch(dbl *alpmin, int *Ntest, dbl Err0)
{
	dbl ***w0;
	dbl alpha1, alpha2, alpha3;
	dbl err1, err2, err3;
	dbl tau;
	int icount, il, in, jn;
	
	tau=LEARN.Tau;

/* store weights before line search */	

	*Ntest = 0;
   	w0 = (dbl ***) malloc(NET.Nlayer*sizeof(dbl**));
	for(il=1; il<NET.Nlayer; il++)
		{
	        w0[il] = (dbl **) malloc(NET.Nneur[il]*sizeof(dbl*));
		for(in=0; in<NET.Nneur[il]; in++)
			{
			w0[il][in] = (dbl *) malloc((NET.Nneur[il-1]+1)*
				sizeof(dbl));
			for(jn=0; jn<=NET.Nneur[il-1]; jn++)
				{
				w0[il][in][jn] = NET.Weights[il][in][jn];
				}
			}
		}
	
/* compute error(w0) */

/*	err1 = MLP_Test(0,0); 
	(*Ntest) ++;*/
	err1 = Err0;
	
	if(NET.Debug>=4) printf("err depart= %f\n",err1);   
	
	*alpmin = 0;
	alpha1 = 0;
/*	alpha2 = 0.05;
	if(LastAlpha != 0) alpha2 = LastAlpha;*/
	alpha2 = LastAlpha;
	if(alpha2 < 0.01) alpha2 = 0.01;
	if(alpha2 > 2.0) alpha2 = 2.0;
	MLP_Line(w0,alpha2);
	err2 = MLP_Test(0,0);
	(*Ntest) ++;
	if(NET.Debug>=4) printf("alpha, err= %e %e\n",alpha2,err2);   
	
	alpha3 = alpha2;
	err3 = err2;
	
/* try to find a triplet (alpha1, alpha2, alpha3) such that 
   Error(alpha1)>Error(alpha2)<Error(alpha3)                 */
   
   	if(err1>err2) 
		{
   		for(icount=1;icount<=100;icount++)
			{
			alpha3 = alpha3*tau;
			MLP_Line(w0,alpha3);
			err3 =MLP_Test(0,0);
	if(NET.Debug>=4) printf("alpha, err= %e %e\n",alpha3,err3);   
			(*Ntest) ++;
			if(err3>err2) break;
			alpha1 = alpha2;
			err1 = err2;
			alpha2 = alpha3;
			err2 = err3;
			}
		if(icount>=100) 		/* line search fails */
			{
			MLP_Line(w0,0);  	/* reset weights */
			free(w0);
			return(1);
			}
		}
	else
		{
   		for(icount=1;icount<=100;icount++)
			{
			alpha2 = alpha2/tau;
			MLP_Line(w0,alpha2);
			err2 = MLP_Test(0,0);
	if(NET.Debug>=4) printf("alpha, err= %e %e\n",alpha2,err2);   
			(*Ntest) ++;
			if(err1>err2) break;
			alpha3 = alpha2;
			err3 = err2;
			}
		if(icount>=100) 		/* line search fails */
			{
			MLP_Line(w0,0);  	/* reset weights */
			free(w0);
			LastAlpha = 0.05;       /* try to be safe */
			return(1);
			}
		}

/* find bottom of parabola */
	
	*alpmin = 0.5*(alpha1+alpha3-(err3-err1)/((err3-err2)/(alpha3-alpha2)
		-(err2-err1)/(alpha2-alpha1)));
	if(*alpmin>10000) *alpmin=10000;

/* set the weights */
        MLP_Line(w0,*alpmin);
	LastAlpha = *alpmin;
	
/* store weight changes */
	for(il=1; il<NET.Nlayer; il++)
		for(in=0; in<NET.Nneur[il]; in++)
			for(jn=0; jn<=NET.Nneur[il-1]; jn++)
				LEARN.Odw[il][in][jn] = NET.Weights[il][in][jn]
				- w0[il][in][jn];

	for(il=1; il<NET.Nlayer; il++)
		for(in=0; in<NET.Nneur[il]; in++)
			free(w0[il][in]);			
	for(il=1; il<NET.Nlayer; il++)
		free(w0[il]);	
	free(w0);
	
	return(0);
}


/***********************************************************/
/* DecreaseSearch                                          */
/*                                                         */
/* search along the line defined by dir for a point where  */
/* error is decreased (faster than full line search)       */
/*                                                         */
/* outputs:     dbl *alpmin = step length  	           */
/*                                                         */
/* Author: J.Schwindling     11-May-99                     */
/***********************************************************/   

/* extern "C"Dllexport */
int DecreaseSearch(dbl *alpmin, int *Ntest, dbl Err0)
{
	dbl ***w0;
	dbl alpha2;
	dbl err1, err2;
	dbl tau;
	int icount, il, in, jn;
	
	tau=LEARN.Tau;

/* store weights before line search */	

	*Ntest = 0;
   	w0 = (dbl ***) malloc(NET.Nlayer*sizeof(dbl**));
	for(il=1; il<NET.Nlayer; il++)
		{
	        w0[il] = (dbl **) malloc(NET.Nneur[il]*sizeof(dbl*));
		for(in=0; in<NET.Nneur[il]; in++)
			{
			w0[il][in] = (dbl *) malloc((NET.Nneur[il-1]+1)*
				sizeof(dbl));
			for(jn=0; jn<=NET.Nneur[il-1]; jn++)
				{
				w0[il][in][jn] = NET.Weights[il][in][jn];
				}
			}
		}
	
/* compute error(w0) */

/*	err1 = MLP_Test(0,0); 
	(*Ntest) ++;*/
	err1 = Err0;
	
	if(NET.Debug>=4) printf("err depart= %f\n",err1);   
	
	*alpmin = 0;
	alpha2 = 0.05;
	MLP_Line(w0,alpha2);
	err2 = MLP_Test(0,0);
	(*Ntest) ++;
	
	if(err2<err1) 
		{
		*alpmin = alpha2;
		}
	else
		{	
	
   
  		for(icount=1;icount<=100;icount++)
			{
			alpha2 = alpha2/tau;
			MLP_Line(w0,alpha2);
			err2 = MLP_Test(0,0);
			(*Ntest) ++;
			if(err1>err2) break;
			}
		if(icount>=100) 		/* line search fails */
			{
			MLP_Line(w0,0);  	/* reset weights */
			free(w0);
			return(1);
			}
		*alpmin = alpha2;	
		}

/* set the weights */
        MLP_Line(w0,*alpmin);
	
/* store weight changes */
	for(il=1; il<NET.Nlayer; il++)
		for(in=0; in<NET.Nneur[il]; in++)
			for(jn=0; jn<=NET.Nneur[il-1]; jn++)
				LEARN.Odw[il][in][jn] = NET.Weights[il][in][jn]
				- w0[il][in][jn];

	for(il=1; il<NET.Nlayer; il++)
		for(in=0; in<NET.Nneur[il]; in++)
			free(w0[il][in]);			
	for(il=1; il<NET.Nlayer; il++)
		free(w0[il]);	
	free(w0);
	
	return(0);
}


/* extern "C"Dllexport */int FixedStep(dbl alpha)
{
	dbl ***w0;
	int il, in, jn;
	
   	w0 = (dbl ***) malloc(NET.Nlayer*sizeof(dbl**));
	for(il=1; il<NET.Nlayer; il++)
		{
	        w0[il] = (dbl **) malloc(NET.Nneur[il]*sizeof(dbl*));
		for(in=0; in<NET.Nneur[il]; in++)
			{
			w0[il][in] = (dbl *) malloc((NET.Nneur[il-1]+1)*
				sizeof(dbl));
			for(jn=0; jn<=NET.Nneur[il-1]; jn++)
				{
				w0[il][in][jn] = NET.Weights[il][in][jn];
				}
			}
		}
	

/* set the weights */
        MLP_Line(w0,alpha);
	
/* store weight changes */
	for(il=1; il<NET.Nlayer; il++)
		for(in=0; in<NET.Nneur[il]; in++)
			for(jn=0; jn<=NET.Nneur[il-1]; jn++)
				LEARN.Odw[il][in][jn] = NET.Weights[il][in][jn]
				- w0[il][in][jn];

	for(il=1; il<NET.Nlayer; il++)
		for(in=0; in<NET.Nneur[il]; in++)
			free(w0[il][in]);			
	for(il=1; il<NET.Nlayer; il++)
		free(w0[il]);	
	free(w0);
	
	return(0);
}

/***********************************************************/
/* MLP_Line                                                */
/*                                                         */
/* sets the weights to a point along a line                */
/*                                                         */
/* inputs:     dbl ***w0 = initial point                   */
/*             dbl alpha = step length                     */ 
/*                                                         */
/* Author: B.Mansoulie     01-Jul-98                       */
/***********************************************************/   

/* extern "C"Dllexport */
void MLP_Line(dbl ***w0, dbl alpha)
{
	int il,in,jn;
	
	for(il=1; il<NET.Nlayer; il++)
		for(in=0; in<NET.Nneur[il]; in++)
			for(jn=0; jn<=NET.Nneur[il-1]; jn++)
				NET.Weights[il][in][jn] = w0[il][in][jn]+
				alpha*dir[il][in][jn];
				
}


/***********************************************************/
/* LineSearchHyb                                           */
/*                                                         */
/* search along the line defined by dir                    */
/*                                                         */
/* outputs:     dbl *alpmin = optimal step length          */
/*                                                         */
/* Author: B.Mansoulie     01-Jul-98                       */
/***********************************************************/   

/* extern "C"Dllexport */
int LineSearchHyb(dbl *alpmin, int *Ntest)
{
	dbl ***w0;
	dbl alpha1, alpha2, alpha3;
	dbl err1, err2, err3;
	dbl tau;
	int icount, il, in, jn;

/*	char buf [50];
  sprintf (buf,"entree linesearchhyb\n");
  MessageBoxA (0,buf,"dans FreePatterns",MB_OK);*/
	
	if(NET.Debug>=4){
		printf(" entry LineSearchHyb \n");
		}
	tau=LEARN.Tau;

/* store weights before line search */	

	*Ntest = 0;
   	w0 = (dbl ***) malloc((NET.Nlayer-1)*sizeof(dbl**));
	for(il=1; il<NET.Nlayer-1; il++)
		{
	        w0[il] = (dbl **) malloc(NET.Nneur[il]*sizeof(dbl*));
		for(in=0; in<NET.Nneur[il]; in++)
			{
			w0[il][in] = (dbl *) malloc((NET.Nneur[il-1]+1)*
				sizeof(dbl));
			for(jn=0; jn<=NET.Nneur[il-1]; jn++)
				{
				w0[il][in][jn] = NET.Weights[il][in][jn];
				}
			}
		}
	
/* compute error(w0) */
	err1 = MLP_Test(0,1);
	(*Ntest) ++; 
	if(NET.Debug>=4) printf("LinesearchHyb err depart= %f\n",err1);   
	
	*alpmin = 0;
	alpha1 = 0;
/*	alpha2 = 0.05;
	if(LastAlpha != 0) alpha2 = LastAlpha;*/
	alpha2 = LastAlpha;
	if(alpha2 < 0.01) alpha2 = 0.01;
	if(alpha2 > 2.0) alpha2 = 2.0;
	MLP_LineHyb(w0,alpha2);
	err2 = MLP_Test(0,1);
	(*Ntest) ++; 
	
	alpha3 = alpha2;
	err3 = err2;
	
/* try to find a triplet (alpha1, alpha2, alpha3) such that 
   Error(alpha1)>Error(alpha2)<Error(alpha3)                 */
   
   	if(err1>err2) 
		{
   		for(icount=1;icount<=100;icount++)
			{
			alpha3 = alpha3*tau;
			MLP_LineHyb(w0,alpha3);
			err3 = MLP_Test(0,1);
			(*Ntest) ++; 
			if(err3>err2) break;
			alpha1 = alpha2;
			err1 = err2;
			alpha2 = alpha3;
			err2 = err3;
			}
		if(icount>=100) 		/* line search fails */
			{
			MLP_LineHyb(w0,0);  	/* reset weights */
			free(w0);
			return(1);
			}
		}
	else
		{
   		for(icount=1;icount<=100;icount++)
			{
			alpha2 = alpha2/tau;
			MLP_LineHyb(w0,alpha2);
			err2 = MLP_Test(0,1);
			(*Ntest) ++; 
			if(err1>err2) break;
			alpha3 = alpha2;
			err3 = err2;
			}
		if(icount>=100) 		/* line search fails */
			{
			MLP_LineHyb(w0,0);  	/* reset weights */
			free(w0);
			return(1);
			}
		}

/* find bottom of parabola */
	
	*alpmin = 0.5*(alpha1+alpha3-(err3-err1)/((err3-err2)/(alpha3-alpha2)
		-(err2-err1)/(alpha2-alpha1)));
	if(*alpmin>10000) *alpmin=10000;

/* set the weights */
        MLP_LineHyb(w0,*alpmin);
	LastAlpha = *alpmin;
	
/* store weight changes */
	for(il=1; il<NET.Nlayer-1; il++)
		for(in=0; in<NET.Nneur[il]; in++)
			for(jn=0; jn<=NET.Nneur[il-1]; jn++)
				LEARN.Odw[il][in][jn] = NET.Weights[il][in][jn]
				- w0[il][in][jn];

	for(il=1; il<NET.Nlayer-1; il++)
		for(in=0; in<NET.Nneur[il]; in++)
			free(w0[il][in]);			
	for(il=1; il<NET.Nlayer-1; il++)
		free(w0[il]);	
	free(w0);
	if(NET.Debug>=4){
		printf(" exit LineSearchHyb \n");
		}

	return(0);
}


/***********************************************************/
/* MLP_LineHyb                                             */
/*                                                         */
/* sets the weights to a point along a line                */
/*                     (for hybrid methods)                */
/*                                                         */
/* inputs:     dbl ***w0 = initial point                   */
/*             dbl alpha = step length                     */ 
/*                                                         */
/* Author: B.Mansoulie     01-Jul-98                       */
/***********************************************************/   

/* extern "C"Dllexport */
void MLP_LineHyb(dbl ***w0, dbl alpha)
{
	int il,in,jn;
	for(il=1; il<NET.Nlayer-1; il++)
		for(in=0; in<NET.Nneur[il]; in++)
			for(jn=0; jn<=NET.Nneur[il-1]; jn++)
			{
				NET.Weights[il][in][jn] = w0[il][in][jn]+
				alpha*dir[il][in][jn];
			}
	MLP_ResLin();
}


/***********************************************************/
/* SetLambda                                               */
/*                                                         */
/* sets the coefficient of the regularisation term for     */
/* hybrid learning method                                  */
/*                                                         */
/* input:     double Wmax = typical maximum weight         */
/*                                                         */
/* Author: J.Schwindling   13-Apr-99                       */
/***********************************************************/   

/* extern "C"Dllexport */
void SetLambda(double Wmax)
{
	dbl err;
	err = MLP_Test(0,0);
	LEARN.Alambda =
	LEARN.Lambda*err/(Wmax*Wmax*(dbl)(NET.Nneur[NET.Nlayer-2]+1));
}

	
/***********************************************************/
/* MLP_ResLin                                              */
/*                                                         */
/* resolution of linear system of equations for hybrid     */
/* training method 		                           */
/*                                                         */
/*                                                         */
/* Author: B.Mansoulie        end-98                       */
/* Modified: J.Schwindling 29-APR-99			   */
/*                         use dgels from LAPACK	   */  
/***********************************************************/   

/* extern "C"Dllexport */
void MLP_ResLin()
{
/*	dbl rrans[NMAX], rrout[NMAX];*/
/*	type_pat rrin[NMAX];*/
	double *HR,*dpat; //,*wlin,*SV;
	double err,lambda,lambda2;
	int Nl,M,Nhr,khr,nrhs,iret,ierr;
	int   il, in, inl, ipat;
	/*register dbl a;*/ //a unused
	char Trans = 'N';

	
/*	int rank; */
//	double rcond = -1;	/* use machine precision */
	
	lambda2 = LEARN.Alambda;
	
/* Nl = number of linear weights
   M = number of terms in linear system = number of examples + regularisation*/
	Nl = NET.Nneur[NET.Nlayer-2] + 1;
	M = PAT.Npat[0]+Nl;

	int Lwork = 5 * M;
        double *Work = (double*) malloc((int) Lwork*sizeof(double));
	
/* memory allocation */
	dpat = (double*) malloc((int) M*sizeof(double));
//	wlin = (double*) malloc((int) Nl*sizeof(double));
//	SV = (double*) malloc((int) Nl*sizeof(double));
	
	Nhr = M * Nl;
	HR = (double*) malloc((int) Nhr*sizeof(double));
	err = 0.;
	for(ipat=0;ipat<PAT.Npat[0];ipat++)
		{
/* *** Filling dpat and HR *** */
/*		for(in=0; in<NET.Nneur[0]; in++)
			{
			rrin[in] = PAT.Rin[0][ipat][in];
			}*/

		MLP_Out(PAT.Rin[0][ipat],NET.Outn[NET.Nlayer-1]);
/*		MLP_Out(rrin,rrout);*/
		/*for(in=0; in<NET.Nneur[NET.Nlayer-1]; in++)
			{ 
			  a = (dbl) PAT.Rans[0][ipat][in]; //a was not used
			} */
		il = NET.Nlayer-2;
		dpat[ipat] = (dbl) PAT.Rans[0][ipat][0]*sqrt(PAT.Pond[0][ipat]);
		khr = ipat;
		HR[khr] = (dbl) sqrt(PAT.Pond[0][ipat]);
		for(in=0;in<NET.Nneur[il];in++)
			{
			khr =  M *(in+1) + ipat;
			HR[khr] = NET.Outn[il][in]* 
				(dbl) sqrt(PAT.Pond[0][ipat]);
			}
		}
	il = NET.Nlayer-2;
	lambda = sqrt(lambda2);
	for(ipat=0;ipat<=NET.Nneur[il];ipat++)
		{
		dpat[ipat+PAT.Npat[0]] = 0;
		for(in=0;in<=NET.Nneur[il];in++)
			{
			khr =  M *in + ipat + PAT.Npat[0];
			HR[khr] = 0;
			if(in==ipat) HR[khr]=lambda;
			}
		}
	if(NET.Debug>=4) 
		{
		err = MLP_Test(0,0);
		printf("entry ResLin, err=MLP_Test(0,0), err= %f\n",err); 
		}		 
/*                                                                */
/*      Trouve les poids lineaires par resolution lineaire        */
/*                                                                */
	nrhs = 1;
	ierr = dgels_(&Trans,&M,&Nl,&nrhs,HR,&M,dpat,&M,Work,
			&Lwork,&iret);
	if(iret != 0) printf("Warning from dgels: iret = %d\n",(int)iret);
	if(ierr != 0) printf("Warning from dgels: ierr = %d\n",(int)ierr);
	
/*	ierr = dgelss_(&M,&Nl,&nrhs,HR,&M,dpat,&M,SV,&rcond,&rank,Work,&Lwork,
		&iret);
	if(iret != 0) printf("Warning from dgelss: iret = %d\n",iret);
	if(ierr != 0) printf("Warning from dgelss: ierr = %d\n",ierr);*/
	
	il = NET.Nlayer-1;
	for (inl=0; inl<=NET.Nneur[il-1];inl++)
		{
		NET.Weights[il][0][inl] = dpat[inl];
		}
	if(NET.Debug>=4) 
		{
		err = MLP_Test(0,0);
		printf("ResLin, apres tlsfor, err= %f\n",err); 
		}		 
	free(Work);
	free(dpat);
//	free(wlin);
	free(HR);
//	free(SV);
}

/***********************************************************/
/* EtaDecay                                                */
/*                                                         */
/* decreases the learning parameter eta by the factor      */
/* LEARN.Decay                                             */
/*                                                         */
/* Author: J.Schwindling   14-Apr-99                       */
/***********************************************************/   

void EtaDecay()
{
	LEARN.eta *= LEARN.Decay;
}

       
/***********************************************************/
/* ShuffleExamples                                         */
/*                                                         */
/* Shuffles the learning examples (for stochastic method)  */
/*                                                         */
/* Author: J.Schwindling   14-Apr-1999                     */
/* Modified: J.Schwindling 21-Jul-1999	inline MLP_Rand    */
/***********************************************************/   

/* extern "C"Dllexport */
int ShuffleExamples(int n, int *index)
{
	int i,ii,itmp;
	dbl a = (dbl) (n-1);
	
	for(i=0;i<n;i++)
		{
		ii = (int) MLP_Rand(0.,a);
		itmp = index[ii];
		index[ii] = index[i];
		index[i] = itmp;
		}
	return 0;
}


/***********************************************************/
/* MLP_Rand                                                */
/*                                                         */
/* Random Numbers (to initialize weights)                  */
/*                                                         */
/* inputs :	dbl min, dbl max = random number between   */
/*                                 min and max             */
/* return value: (double) random number                    */
/*                                                         */
/* Author: J.Schwindling   14-Apr-99                       */
/***********************************************************/   

/* extern "C"Dllexport */
double MLP_Rand(dbl mini, dbl maxi)
{
return mini+(maxi-mini)*random()/RAND_MAX;
}


/***********************************************************/
/* InitWeights                                             */
/*                                                         */
/* initializes the weights to random numbers between       */
/* -0.5 : 0.5                                              */
/*                                                         */
/* Author: J.Schwindling   14-Apr-99                       */
/***********************************************************/   

/* extern "C"Dllexport */
void InitWeights()
{
        int ilayer,ineur,i;
	
    	for(ilayer=1;ilayer<NET.Nlayer;ilayer++)
		for(ineur=0;ineur<NET.Nneur[ilayer];ineur++)
			for(i=0;i<=NET.Nneur[ilayer-1];i++)
				NET.Weights[ilayer][ineur][i]=
					(dbl) MLP_Rand(-0.5, 0.5);
}


/***********************************************************/
/* PrintWeights                                            */
/*                                                         */
/* Print weights on the screen                             */
/*                                                         */
/* Author: J.Schwindling   14-Apr-99                       */
/***********************************************************/   

/* extern "C"Dllexport */
void PrintWeights()
{
        int ilayer,ineur,i;

	for(ilayer=1; ilayer<NET.Nlayer; ilayer++)
		{
		if(MessLang==1) 
			{
			printf("Couche %d\n",ilayer);
			}
		else
			{
			printf("Layer %d\n",ilayer);
			}
		for(ineur=0; ineur<NET.Nneur[ilayer]; ineur++)
			{
			if(MessLang==1) 
				{
				printf("Neurone %d",ineur);
				}
			else
				{
				printf("Neuron %d",ineur);
				}
			for(i=0; i<=NET.Nneur[ilayer-1]; i++)
				{
				printf(" %f",
					(double) NET.Weights[ilayer][ineur][i]);
				}
			printf("\n");
			}
		printf("\n");
		}
}
 

/***********************************************************/
/* ReadPatterns                                            */
/*                                                         */
/* parser for learn.pat or test.pat files                  */
/*                                                         */
/* inputs: char *filename = name of the file to read       */
/*	   int ifile = 0: learning examples                */
/*                     1: test examples                    */
/*                                                         */
/* outputs: int *inet = 1 if a network is defined          */
/*          int *ilearn = 1 if a learning method is defined*/
/*          int *iexamples = 1 if examples are defined     */ 
/*                                                         */
/* return value (int) = 0:   no error			   */
/*                    = -1:  file could not be opened      */
/*                                                         */
/* Author: J.Schwindling   20-Apr-99                       */
/* Modified: J.Schwindling 01-Jun-99 return inet, ilearn   */
/*					    iexamples      */ 
/*           J.Schwindling 21-Sep-99 return value = -1     */
/***********************************************************/   

#define CLEN 1024

/* extern "C"Dllexport */
int ReadPatterns(char *filename, int ifile,
 		 int *inet, int *ilearn, int *iexamples)
{
char s[CLEN], s2[CLEN], cc[6], cc2[6];
char otherfile[CLEN];
double p;
//int line,i,j;
int line,i;
//int l,ll,ipat,nmax,il,in,tf;
int l,ll,ipat,nmax;
int np=0;       /* nombre d'exemples */
int nin=0;      /* nombre d'entrees */
int nout=0;     /* nombre de sorties */
int npon=0;
int ntot, ierr;
//char **ss;
char **ss=nullptr;
FILE *LVQpat;
int nlayer, nneur[NLMAX];

printf("\nLoading file %s\n",filename);
LVQpat=fopen(filename,"r");
if(LVQpat == nullptr) return -1;

line=0;

while(fgets(s,CLEN,LVQpat))
	{
	if(*s=='N')                                 
		{
		if(*(s+1)=='N') 		/* NNEU */
			{
			printf("Number of neurons %s",s);
			*inet = 1;
			sscanf(s,"%s %s",cc,s2);
			ierr = GetNetStructure(s2,&nlayer,nneur);
			if(ierr != 0) return ierr;
			ierr = MLP_SetNet(&nlayer,nneur);
			if(ierr != 0) return ierr;
			}
		else
			{	
			sscanf(s,"%s %d",cc,&l);
			if(*(cc+1)=='P') 	/* NPAT */
				{
				np=l;
				printf("Number of patterns %d\n",np);
				}
			else if(*(cc+1)=='I') 	/* NINP */
				{
				nin=l;
				PAT.Nin = nin;
				printf("Number of inputs %d\n",nin);
				}
			else if(*(cc+1)=='O' && *(cc+2)=='U') 	/* NOUT */
				{
				nout=l;
				PAT.Nout = nout;
				printf("Number of outputs %d\n",nout);
				}
			else if(*(cc+1)=='O' && *(cc+2)=='R') 	/* NORM */
				{
				DIVERS.Norm=l;
				if(l==1) printf("Normalize inputs\n");
				}
/* obsolete datacard 	 NLAY */			
			else if(*(cc+1)=='L')	
				{
				printf("NLAY datacard is no longer needed\n");
				}				
			else if(*(cc+1)=='E')	/* NEPO */ 
				{
				LEARN.Nepoch=l;
				printf("Number of epochs %d\n",l);
				}
			else if(*(cc+1)=='R') 	/* NRES */
				{
				LEARN.Nreset=l;
				printf(
				"Reset to steepest descent every %d epochs\n",
				l);
				}
			}
		}
	else if(*s=='L') 		
		{
		if(*(s+1)=='P')			/* LPAR */
			{
			sscanf(s,"%s %le",cc,&p);
			printf("Learning parameter %f\n",p);
			LEARN.eta = (dbl) p;
			}
		else if(*(s+1)=='M')		/* LMET */
			{
			*ilearn = 1;
			sscanf(s,"%s %d",cc,&(LEARN.Meth));
			printf("Learning method = ");
			switch(LEARN.Meth)
                	{
                case 1: printf("Stochastic Minimization\n");
                        break;
                case 2: printf("Steepest descent with fixed step\n");
                        break;
                case 3: printf("Steepest descent with line search\n"); break;
                case 4: printf("Polak-Ribiere Conjugate Gradients\n"); break;
                case 5: printf("Fletcher-Reeves Conjugate Gradients\n");
                        break;
                case 6: printf("BFGS\n");
                        break;
                case 7: printf("Hybrid BFGS-linear\n");
                        break;
                default: printf("Error: unknown method\n"); break;
                }

			}
		else if(*(s+1)=='T')	/* LTAU */
			{
			sscanf(s,"%s %lf",cc,&p);
			printf("Tau %f\n",p);
			LEARN.Tau = (dbl) p;
			}
		else if(*(s+1)=='A')	/* LAMB */
			{
			sscanf(s,"%s %lf",cc,&p);
			printf("Lambda %f\n",p);
			LEARN.Lambda = (dbl) p;
			}
		}
	else if(*s=='F') 		
		{
		if(*(s+1)=='S') 	/* FSPO */
			{
			sscanf(s,"%s %le",cc,&p);
			printf("Flat spot elimination parameter %f\n",p);
			LEARN.delta = (dbl) p;
			}
		else if(*(s+1)=='I')	/* FILE */
			{
			sscanf(s,"%s %s",cc,otherfile);
			ierr = ReadPatterns(otherfile,ifile, inet, ilearn, iexamples);
			if(ierr != 0) return ierr;
			}	
		}
	else if(*s=='M') 		/* momentum */
		{
		sscanf(s,"%s %le",cc,&p);
		printf("Momentum term %f\n",p);
		LEARN.epsilon = (dbl) p;
		}		
	else if(*s=='O')                /* OUTx */
		{
		if(*(s+3)=='W') 	/* OUTW */
			{
			sscanf(s,"%s %d",cc,&OutputWeights);
			if(OutputWeights == 0)
				{
				printf("Never write file weights.out\n");
				}
			else if(OutputWeights == -1)
				{
				printf("Write weights to output file at the end\n");
				}
			else 
				{
				printf("Write weights to file every %d epochs\n",
					OutputWeights);
				}
			}
		else if(*(s+3)=='F')	/* OUTF */
			{
			sscanf(s,"%s %s",cc,cc2);
			if(*cc2=='F' || *cc2=='C')
				{
				DIVERS.Outf = *cc2;
				}
			else
				{
			printf(" *** Error while loading file %s at line %s :",
				filename,s);
			printf(" unknown language\n");
				}	
			}
		else
			{
			printf(" *** Error while loading file %s at line %s\n",
				filename,s);
			}					
		}		
	else if(*s=='R')                /* RDWT */
		{
		sscanf(s,"%s %d",cc,&(NET.Rdwt));
		if(NET.Rdwt == 0)
			{
			printf("Random weights \n");
			}
		else
			{
			printf("Read weights from file weights.in\n");
			}	
		}		
	else if(*s=='S')                /* STAT */
		{
		sscanf(s,"%s %d",cc,&(DIVERS.Stat));
		}
/*	else if(*s=='T')                 TFUN 
		{
		sscanf(s,"%s %d %d %d",cc,&il,&in,&tf);
		SetTransFunc(il,in,tf);
		} */	
	else if(*s=='H')                /* HESS */
		{
		sscanf(s,"%s %d",cc,&(DIVERS.Ihess));
		}		
	else if(*s=='D')                
		{
		if(*(s+1)=='C')		/* DCAY */
			{	
			sscanf(s,"%s %le",cc,&p);
			LEARN.Decay = p;
			printf("Learning parameter decay %f\n",
						(double) LEARN.Decay);
			}
		if(*(s+1)=='B')		/* DBIN */	
			{	
			sscanf(s,"%s %d",cc,&(DIVERS.Dbin));
			printf("Fill histogram every %d epochs\n",DIVERS.Dbin);
			}
		if(*(s+1)=='E')		/* DEBU */	
			{	
			sscanf(s,"%s %d",cc,&(NET.Debug));
			printf("Debug mode %d\n",NET.Debug);
			}
		}		
	else if(*s=='P')                /* POND */
		{
		npon = CountLexemes(s);
		if(npon==2) 
			{
			sscanf(s,"%s %d",cc,&(PAT.Iponde));
			}
		else
			{
			ss = (char**) malloc((npon+1)*sizeof(char*)); 
			for(i=0;i<=npon;i++)
				ss[i]=(char*) malloc(40*sizeof(char));
			getnLexemes(npon,s,ss);
			sscanf(ss[1],"%d",&(PAT.Iponde));
			for(i=2;i<npon;i++)
			    {	
			    sscanf(ss[i],"%le",&(PAT.Ponds[i-2]));
			    }
			}
		if(PAT.Iponde==0) 
			{
			npon = 0;
			}
		else
			{
			npon = 1;
			}
		}		
	else if(*s=='#')			    /* comments */
		{
		}
	else                                        /* exemple itself */
		{
		if(np==0) return 1;
		if(nin==0) return 2;
		if(nout==0) return 3;
		

/* store number of exemples and allocate memory*/		
		if(line==0)
			{
			PAT.Npat[ifile] = np;
			ierr = AllocPatterns(ifile,np,nin,nout,0);
			if(ierr != 0) return ierr;
			*iexamples = 1;
			}

/* now get exemple */
				
    		line++;
		ll = (line-1)%2;
		ipat = (line-1)/2;
		/*		printf("Loading event \t %d\r",ipat);*/
/*		if(ipat>NPMAX) 
			{
			printf("Too many examples in file\n");
			printf("Loading %d examples\n",NPMAX);
			PAT.Npat[ifile] = NPMAX;
			break;
			}
*/			
		
/* allocate the number of lines */
		
		if(line==1) 
			{
			
			nmax = nin;
			if(nout>nin) nmax=nout;
			ss = (char**) malloc((nmax+1)*sizeof(char*));
			if(ss == nullptr) return -111; 
			for(i=0;i<=nmax;i++)
				{
				ss[i]=(char*) malloc(40*sizeof(char));
				if(ss[i] == nullptr) return -111;
				}
			}
			
		if(ll==0)    /* inputs */
			{
			getnLexemes(nin,s,ss);
			for(i=0;i<nin;i++)
				{
				sscanf(ss[i],"%le",&p);
				PAT.Rin[ifile][ipat][i] = (type_pat) p;
				}
			}
		else         /* answers */
			{
			ntot=nout+npon;
			getnLexemes(ntot,s,ss);
			for(i=0;i<ntot;i++)
				{
				sscanf(ss[i],"%le",&p);
				if(i<nout)
					{
					PAT.Rans[ifile][ipat][i] = (type_pat) p;
					}
				else
					{
					if(PAT.Iponde==1)
						{
						PAT.Pond[ifile][ipat] = 
							(type_pat) p;
						}
					else
						{
						PAT.Pond[ifile][ipat] = 
						(type_pat) PAT.Ponds[(int) p -1];
						}
					}
				}
			} 				
		}
	}
	printf("%d examples loaded    \n\n",PAT.Npat[ifile]);
	fclose(LVQpat);
	return 0;
}


/* CountLexemes returns the number of lexemes in s separated by blanks.*/
/* extern "C"Dllexport */
int CountLexemes(char *s)
{
  char tmp[1024];
  int i=0;
  
  strcpy(tmp,s);
  if (strtok(tmp," "))
    {
      i=1;
      while (strtok(nullptr," ")) i++;
    }
  return i;
}

/* splits s in substrings ss separated by blanks*/
/* extern "C"Dllexport */
void getnLexemes(int n, char *s, char **ss)
{
  char tmp[1024];
  int i;     
  strcpy(tmp,s);
  if (n>0)
    {
      strcpy(ss[0],strtok(tmp," "));
      for (i=1;i<n;i++)
        strcpy(ss[i],strtok(nullptr," "));
    }
}

/* extern "C"Dllexport */
void getLexemes(char *s,char **ss)
{
  char tmp[1024];
  int i,n;   
    
  strcpy(tmp,s);
  n=CountLexemes(tmp);
  if (n>0)
    {
      strcpy(ss[0],strtok(tmp," "));
      for (i=1;i<n;i++)
        strcpy(ss[i],strtok(nullptr," "));
    }
}


/***********************************************************/
/* LearnFree                                               */
/*                                                         */
/* frees memory allocated for learning		           */
/*                                                         */
/* Author: J.Schwindling   28-May-99                       */
/***********************************************************/   

/* extern "C"Dllexport */
void LearnFree()
{
        int il,in; 
	if(LearnMemory==0) return;
	LearnMemory = 0;
	for(il=0; il<NET.Nlayer; il++)
		{
		for(in=0; in<NET.Nneur[il]; in++)
			{
			free(dir[il][in]);
			}
		free(dir[il]);
		}
	free(dir);
	if(BFGSMemory==0) return;
	BFGSMemory = 0;
	for(il=0; il<NET.Nweights; il++)
		{
		free(BFGSH[il]);
		}		
	free(BFGSH);
	free(Gamma);
	free(delta);
	
/*	if(JacobianMemory == 0) return;
	JacobianMemory = 0;
	for(il=0; il<PAT.Npat[0]; il++) free(JacobianMatrix[il]);
	free(JacobianMatrix);	*/
}


/***********************************************************/
/* LearnAlloc                                              */
/*                                                         */
/* memory allocation for vectors and matrices used in      */
/* conjugate gradients or BFGS like methods                */
/*                                                         */
/* return value (int) = error code: 0 no error   	   */
/*				    -111 no memory	   */
/*                                                         */
/* Author: J.Schwindling   20-Apr-99                       */
/* Modified: J.Schwindling 31-Jan-2000	error code	   */
/***********************************************************/   

/* extern "C"Dllexport */
int LearnAlloc()
{
        int il,in,i; 
	int Nweights = 0;
	
	if(LearnMemory != 0) LearnFree();
	LearnMemory = 1;
   	dir = (dbl ***) malloc(NET.Nlayer*sizeof(dbl**));
	if(dir == nullptr) return -111;
	
	for(il=0; il<NET.Nlayer; il++)
		{
	        dir[il] = (dbl **) malloc(NET.Nneur[il]*sizeof(dbl*));
		if(dir[il] == nullptr) return -111;
		for(in=0; in<NET.Nneur[il]; in++)
			{
			if(il==0) 
				{
/* TODO: understand implications of hard-coded 101 */ 				
				dir[0][in] = (dbl *)
				malloc(101*sizeof(dbl));
				if(dir[0][in] == nullptr) return -111;
				}
			else
				{
				dir[il][in] = (dbl *) 
				malloc((NET.Nneur[il-1]+1)*sizeof(dbl));
				if(dir[il][in] == nullptr) return -111;
			        Nweights += NET.Nneur[il-1]+1;
				}
			}
		}
	NET.Nweights = Nweights;
		
	if(BFGSMemory==0 && LEARN.Meth>= 6) 
		{
		BFGSMemory = 1;
		Gamma = (dbl*) malloc(Nweights*sizeof(dbl));
		delta = (dbl*) malloc(Nweights*sizeof(dbl));
		BFGSH = (dbl**) malloc(Nweights*sizeof(dbl*));
		if(Gamma == nullptr || delta == nullptr || BFGSH == nullptr)
		   return -111;
		   
		for(i=0; i<Nweights; i++)
			{
			BFGSH[i] = (dbl*) malloc(Nweights*sizeof(dbl));
			if(BFGSH[i] == nullptr) return -111;
			}
		}
		
/*	if(JacobianMemory==0)
		{
		JacobianMemory = 1;
		printf("JacobianMemory = %d\n",JacobianMemory);
		JacobianMatrix = (dbl **) malloc(PAT.Npat[0]*sizeof(dbl *));
		for(i=0; i<PAT.Npat[0]; i++)
			JacobianMatrix[i] = 
				(dbl*) malloc(Nweights*sizeof(dbl));
		printf("end memory alloc\n");
		}	
	
	if(DIVERS.Ihess==1) HessianAlloc(Nweights);*/
	
	return 0;
}


/***********************************************************/
/* MLP_PrFFun                                              */
/*                                                         */
/* writes the MLP function to file as a fortran function   */
/*                                                         */
/* inputs :	char *filename = name of the output file   */
/*                                                         */
/* return value (int) = 0: no error			   */
/*		       -1: could not open file		   */ 
/*                                                         */
/* Author: J.Schwindling   20-Apr-99                       */
/* Modified: J.Schwindling 05-May-99			   */
/*		add normalization of inputs		   */
/*           J.Schwindling 30-Nov-99 return code	   */
/***********************************************************/   

/* extern "C"Dllexport */
int MLP_PrFFun(char *filename)
{
	int il,in,jn;
	FILE *W;

	W=fopen(filename,"w");
	if(W==nullptr) return -1;
	fprintf(W,"      SUBROUTINE RNNFUN(rin,rout)\n");
	fprintf(W,"      DIMENSION RIN(%d)\n",NET.Nneur[0]);
	fprintf(W,"      DIMENSION ROUT(%d)\n",NET.Nneur[NET.Nlayer-1]);
	fprintf(W,"C\n");
	
	for(in=0; in<NET.Nneur[0]; in++)
		{
		if(DIVERS.Norm==0)
			{ 
			fprintf(W,"      OUT%d = RIN(%d)\n",in+1,in+1);
			}
		else
			{
			fprintf(W,"      OUT%d = (RIN(%d)-%e)/%e\n",in+1,in+1,
					STAT.mean[in],STAT.sigma[in]);
			}	
		}
	for(il=1; il<NET.Nlayer-1; il++)
		{
		fprintf(W,"C\n");
		fprintf(W,"C     layer %d\n",il+1);
		for(in=0; in<NET.Nneur[il]; in++)
			{
			fprintf(W,"      RIN%d = %e\n",in+1,
					(double) NET.Weights[il][in][0]);
			for(jn=1;jn<=NET.Nneur[il-1]; jn++)
				fprintf(W,"     > +(%e) * OUT%d\n",
					(double) NET.Weights[il][in][jn],jn);
			}
		fprintf(W,"C\n");
		for(in=0; in<NET.Nneur[il]; in++)
			{
			if(NET.T_func[il][in]==0) 
				{
				fprintf(W,"      OUT%d = 0\n",in+1);
				}
			else if(NET.T_func[il][in]==1)
				{
				fprintf(W,"      OUT%d = RIN%d\n",in+1,in+1);
				}
			else if(NET.T_func[il][in]==2)
				{
				fprintf(W,"      OUT%d = SIGMOID(RIN%d)\n",
					in+1,in+1);
				}
			}
		}
	il = NET.Nlayer-1;
	fprintf(W,"C\n");
	fprintf(W,"C     layer %d\n",il+1);
	for(in=0; in<NET.Nneur[il]; in++)
		{
		fprintf(W,"      RIN%d = %e\n",in+1,
				(double) NET.Weights[il][in][0]);
		for(jn=1;jn<=NET.Nneur[il-1]; jn++)
			fprintf(W,"     > +(%e) * OUT%d\n",
				(double) NET.Weights[il][in][jn],jn);
		}
	fprintf(W,"C\n");
	for(in=0; in<NET.Nneur[il]; in++)
		{
		if(NET.T_func[il][in]==0) 
			{
			fprintf(W,"      ROUT(%d) = 0\n",in+1);
			}
		else if(NET.T_func[il][in]==1)
			{
			fprintf(W,"      ROUT(%d) = RIN%d\n",in+1,in+1);
			}
		else if(NET.T_func[il][in]==2)
			{
			fprintf(W,"      ROUT(%d) = SIGMOID(RIN%d)\n",
				in+1,in+1);
			}
		}
		
	fprintf(W,"C\n");
	fprintf(W,"      END\n");
	fprintf(W,"      REAL FUNCTION SIGMOID(X)\n");
	fprintf(W,"      SIGMOID = 1./(1.+EXP(-X))\n");
	fprintf(W,"      END\n");
	
	fclose(W);
	return 0;
}


/***********************************************************/
/* MLP_PrCFun                                              */
/*                                                         */
/* writes the MLP function to file as a C function	   */
/*                                                         */
/* inputs :	char *filename = name of the output file   */
/*                                                         */
/* return value (int) = 0: no error			   */
/*			-1: could not open file		   */ 
/*                                                         */
/* Author: J.Schwindling   23-Apr-99                       */
/* Modified: J.Schwindling 30-Nov-99 return code	   */
/***********************************************************/   

/* extern "C"Dllexport */
int MLP_PrCFun(char *filename)
{
	int il,in,jn;
	FILE *W;

	W=fopen(filename,"w");
	if(W==nullptr) return -1;
	
	fprintf(W,"double sigmoid(double x)\n");
	fprintf(W,"{\n");
	fprintf(W,"return 1/(1+exp(-x));\n");
	fprintf(W,"}\n");
	fprintf(W,"void rnnfun(double *rin,double *rout)\n");
	fprintf(W,"{\n");
	fprintf(W,"      double out1[%d];\n",NET.Nneur[0]);
	fprintf(W,"      double out2[%d];\n",NET.Nneur[1]);
	if(NET.Nlayer>=3) fprintf(W,"      double out3[%d];\n",NET.Nneur[2]);
	if(NET.Nlayer>=4) fprintf(W,"      double out4[%d];\n",NET.Nneur[3]);
	fprintf(W,"\n");
	
	for(in=0; in<NET.Nneur[0]; in++)
		{
		if(DIVERS.Norm==0)
			{ 
			fprintf(W,"      out1[%d] = rin[%d];\n",in,in);
			}
		else
			{
			fprintf(W,"      out1[%d] = (rin[%d]-%e)/%e;\n",
					in,in,
					STAT.mean[in],STAT.sigma[in]);
			}	
		}
		
	for(il=1; il<=NET.Nlayer-1; il++)
		{
		fprintf(W,"\n");
		fprintf(W,"/*     layer %d */\n",il+1);
		for(in=0; in<NET.Nneur[il]; in++)
			{
			fprintf(W,"      out%d[%d] = %e\n",il+1,in,
					(double) NET.Weights[il][in][0]);
			for(jn=1;jn<=NET.Nneur[il-1]; jn++) 
				fprintf(W,"      +(%e) * out%d[%d]\n",
					(double) NET.Weights[il][in][jn],il,jn-1);
		   fprintf(W,"      ;\n"); 
			}
		fprintf(W,"\n");
		for(in=0; in<NET.Nneur[il]; in++)
			{
			if(NET.T_func[il][in]==0) 
				{
				fprintf(W,"      out%d[%d] = 0;\n",il+1,in);
				}
			else if(NET.T_func[il][in]==1)
				{
				}
			else if(NET.T_func[il][in]==2)
				{
				fprintf(W,"      out%d[%d] = sigmoid(out%d[%d]);\n",
					il+1,in,il+1,in);
				}
			}
		}
	il = NET.Nlayer-1;
	for(in=0; in<NET.Nneur[il]; in++)
		{
		fprintf(W,"      rout[%d] = out%d[%d];\n",in,il+1,in);
		}
	fprintf(W,"}\n");	
	fclose(W);
	return 0;
}


/***********************************************************/
/* SaveWeights                                             */
/*                                                         */
/* writes the weights to file                              */
/*                                                         */
/* inputs :	char *filename = name of the output file   */
/*              int iepoch = epoch number                  */
/*                                                         */
/* return value (int): 0 if OK				   */
/*		       -1 if file could not be opened      */	 
/*                                                         */
/* Author: J.Schwindling   20-Apr-99                       */
/* Modified: J.Schwindling 11-Jun-99			   */
/*                         print net structure in header   */
/* Modified: J.Schwindling 05-Nov-99			   */
/*                         return error code               */
/***********************************************************/   

/* extern "C"Dllexport */
int SaveWeights(char *filename, int iepoch)
{
	FILE *W;
	int ilayer,ineur,i;

	W=fopen(filename,"w");
	if(W==nullptr) return -1;
	
	fprintf(W,"# network structure ");
	for(ilayer=0; ilayer<NET.Nlayer; ilayer++)
		{
		fprintf(W,"%d ",NET.Nneur[ilayer]);
		}
		
	fprintf(W,"\n %d\n",iepoch);
	for(ilayer=1; ilayer<NET.Nlayer; ilayer++)
		{
		for(ineur=0; ineur<NET.Nneur[ilayer]; ineur++)
			{
			for(i=0; i<=NET.Nneur[ilayer-1]; i++)
				{
				fprintf(W," %1.15e\n",
				(double) NET.Weights[ilayer][ineur][i]);
				}
			}
		}
	fclose(W);
	return 0;
}
	

/***********************************************************/
/* LoadWeights                                             */
/*                                                         */
/* reads the weights from file                             */
/*                                                         */
/* input :	char *filename = name of the input file    */
/* output :     int *iepoch = epoch number                 */
/*                                                         */
/* return value (int): 0 if OK				   */
/*		       -1 if file could not be opened      */	 
/*                                                         */
/* Author: J.Schwindling   20-Apr-99                       */
/* Modified: J.Schwindling 11-Jun-99			   */
/*           lines starting with # are skipped		   */
/* Modified: J.Schwindling 05-Nov-99			   */
/*                         return error code               */
/***********************************************************/   

/* extern "C"Dllexport */
int LoadWeights(char *filename, int *iepoch)
{
	FILE *W;
	int ilayer,ineur,i;
	double p;
	char s[80];

	W=fopen(filename,"r");
	if(W==nullptr) return -1;
	do
		{
		fgets(s,80,W);
		}
	while(*s == '#');	
		sscanf(s," %d",iepoch);
		for(ilayer=1; ilayer<NET.Nlayer; ilayer++)
			{
			for(ineur=0; ineur<NET.Nneur[ilayer]; ineur++)
				{
				for(i=0; i<=NET.Nneur[ilayer-1]; i++)
					{
					fscanf(W," %le",&p);
					NET.Weights[ilayer][ineur][i] = (dbl) p;
					}
				}
			}
		
	fclose(W);
	return 0;
}


/***********************************************************/
/* AllocPatterns                                           */
/*                                                         */
/* allocate memory for the examples                        */
/*                                                         */
/* input :	int ifile = file number (0 or 1)           */
/*              int npat = number of examples              */
/*              int nin  = number of inputs                */
/*              int nout = number of outputs               */
/*              int iadd = 0: new examples                 */
/*                         1: add to existing ones         */   
/*                                                         */
/* return value (int) = error code: 0 = no error	   */
/*                                  1 = wrong file number  */
/*				   -111 = no memory        */
/*                                                         */
/* Author: J.Schwindling   21-Apr-99                       */
/* Modified: J.Schwindling 26-Apr-99			   */
/*            -	frees memory if already booked and iadd=0  */
/*              (should remove need to call mlpfree)       */ 
/*            - implement iadd = 1                         */  
/***********************************************************/   

/* extern "C"Dllexport */int AllocPatterns(int ifile, int npat, int nin, int nout, int iadd)
{
	int j;
	type_pat *tmp, *tmp3;
	type_pat **tmp2;
	int ntot;
	
	if(ifile>1 || ifile<0) return(1);
/*	scanf("%d",&j); */
	if(ExamplesMemory==0) 
		{
		ExamplesMemory=1;
        	PAT.Pond = (type_pat **) malloc(2*sizeof(dbl*));
	        PAT.Rin = (type_pat***) malloc(2*sizeof(type_pat**));
	        PAT.Rans = (type_pat***) malloc(2*sizeof(type_pat**));
		PAT.vRin = (type_pat**) malloc(2*sizeof(type_pat*));
		if(PAT.Pond == nullptr || PAT.Rin == nullptr
		   || PAT.Rans == nullptr || PAT.vRin == nullptr) return -111; 
		} 
	

/* if iadd=0, check that memory not already allocated. Otherwise free it */
	if(iadd==0 && PatMemory[ifile]!=0)
		{
		 FreePatterns(ifile);
		}

/* allocate memory and initialize ponderations */
 	if(iadd==0 || PatMemory[ifile]==0)
	{
	PatMemory[ifile] = 1;		
        PAT.Pond[ifile] = (type_pat*) malloc(npat*sizeof(type_pat));
	if(PAT.Pond[ifile] == nullptr) return -111;
	for(j=0; j<npat; j++)
           	PAT.Pond[ifile][j] = 1;
			
	PAT.Rin[ifile] = (type_pat**) malloc(npat*sizeof(type_pat*));
	if(PAT.Rin[ifile] == nullptr) return -111;
	PAT.Rans[ifile] = (type_pat**) malloc(npat*sizeof(type_pat*));
	if(PAT.Rans[ifile] == nullptr) return -111;

	PAT.vRin[ifile] = (type_pat *) malloc(npat*(nin+1)*
						sizeof(type_pat));
	if(PAT.vRin[ifile] == nullptr) return -111;
						
	for(j=0; j<npat; j++)
		{
		PAT.Rin[ifile][j] = &(PAT.vRin[ifile][j*(nin+1)+1]);
		PAT.vRin[ifile][j*(nin+1)] = 1;
		}
	for(j=0; j<npat; j++)
		{
		PAT.Rans[ifile][j] = (type_pat*) malloc(nout*sizeof(type_pat));
		if(PAT.Rans[ifile][j] == nullptr) return -111;
		}
	PAT.Npat[ifile] = npat;
	
	if(ifile==0)
		{	
		ExamplesIndex = (int *) malloc(npat*sizeof(int));
		if(ExamplesIndex == nullptr) return -111;
		for(j=0; j<npat; j++) ExamplesIndex[j] = j;
		}
	}
	else		/* add examples */
	{
	ntot = PAT.Npat[ifile]+npat;
	
/* event weighting */	
	tmp = (type_pat *) malloc(ntot*sizeof(type_pat));
	if(tmp == nullptr) return -111;
	
	for(j=0; j<PAT.Npat[ifile]; j++) 
		{
		tmp[j] = PAT.Pond[ifile][j];
		}
	for(j=PAT.Npat[ifile];j<ntot;j++)
		{
		tmp[j] = 1;
		}
	if(PatMemory[ifile]==1) free(PAT.Pond[ifile]);
	PAT.Pond[ifile] = tmp;	
	
/* examples */
/*	tmp2 = (type_pat **) malloc(ntot*sizeof(type_pat*));		
	for(j=0; j<PAT.Npat[ifile]; j++) 
		{
		tmp2[j] = PAT.Rin[ifile][j];
		}
	for(j=PAT.Npat[ifile];j<ntot;j++)
		{
		tmp2[j] = (type_pat*) malloc(nin*sizeof(type_pat));
		}
	if(PatMemory[ifile]==1) free(PAT.Rin[ifile]);
	PAT.Rin[ifile] = tmp2;	*/
	
	tmp3 = (type_pat *) malloc(ntot*(nin+1)*sizeof(type_pat));
	if(tmp3 == nullptr) return -111;
	
	for(j=0; j<PAT.Npat[ifile]*(nin+1); j++)
		{
		tmp3[j] = PAT.vRin[ifile][j];
		}
	if(PatMemory[ifile]==1) free(PAT.vRin[ifile]);
	PAT.vRin[ifile] = tmp3;
	for(j=0; j<ntot; j++)
		{
		PAT.Rin[ifile][j] = &(PAT.vRin[ifile][j*(nin+1)+1]);
		PAT.vRin[ifile][j*(nin+1)] = 1;
		}
		
	tmp2 = (type_pat **) malloc(ntot*sizeof(type_pat*));
	if(tmp2 == nullptr) return -111;		
	for(j=0; j<PAT.Npat[ifile]; j++) 
		{
		tmp2[j] = PAT.Rans[ifile][j];
		}
	for(j=PAT.Npat[ifile];j<ntot;j++)
		{
		tmp2[j] = (type_pat*) malloc(nout*sizeof(type_pat));
		if(tmp2[j] == nullptr) return -111;
		}
	if(PatMemory[ifile]==1) free(PAT.Rans[ifile]);
	PAT.Rans[ifile] = tmp2;	
	PAT.Npat[ifile] = ntot;
	PatMemory[ifile] = 1;		
	
/* indices */	
	if(ifile==0)
		{
		free(ExamplesIndex);	
		ExamplesIndex = (int *) malloc(ntot*sizeof(int));
		if(ExamplesIndex == nullptr) return -111;
		for(j=0; j<ntot; j++) ExamplesIndex[j] = j;
		}
	}
		
	return 0;
} 


/***********************************************************/
/* FreePatterns                                            */
/*                                                         */
/* frees memory for the examples                           */
/*                                                         */
/* input :	int ifile = file number (0 or 1)           */
/*                                                         */
/* return value (int) = error code: 0 = no error	   */
/*                                  1 = wrong file number  */
/*                                  2 = no mem allocated   */
/*                                                         */
/* Author: J.Schwindling   26-Apr-99                       */
/***********************************************************/   

/* extern "C"Dllexport */int FreePatterns(int ifile)
{
	int i;

	if(ifile>1 || ifile<0) return 1;
/*	printf("%d %d \n",ifile,PatMemory[ifile]);*/
	if(PatMemory[ifile]==0) return 2;
	
	free(PAT.Pond[ifile]);
	for(i=0; i<PAT.Npat[ifile]; i++)
		{
/*		free(PAT.Rin[ifile][i]); */
		free(PAT.Rans[ifile][i]); 
		}
	free(PAT.Rin[ifile]);
	free(PAT.Rans[ifile]);
	free(PAT.vRin[ifile]);
	PatMemory[ifile] = 0;
	PAT.Npat[ifile] = 0;
	
	return 0;
}		


/***********************************************************/
/* MLP_StatInputs                                          */
/*                                                         */
/* compute statistics about the inputs: mean, RMS, min, max*/
/*                                                         */
/* inputs:	int Nexamples = number of examples	   */
/*		int Niputs = number of quantities	   */
/* 		type_pat **inputs = input values	   */
/*							   */
/* outputs:	dbl *mean = mean value			   */
/*		dbl *sigma = RMS			   */
/* 		dbl *minimum = minimum			   */
/*		dbl *maximum = maximum			   */
/*							   */
/* return value (int): 	always = 0			   */  
/*                                                         */
/* Author: J.Schwindling   11-Oct-99                       */
/***********************************************************/   

int MLP_StatInputs(int Nexamples, int Ninputs, type_pat **inputs, 
		dbl *mean, dbl *sigma, dbl *minimum, dbl *maximum)	
{
	dbl *fmean;
	int j, ipat, nmax;
	
/* first compute a fast rough mean using the first 100 events */
   	fmean = (dbl*) malloc(Ninputs*sizeof(dbl));
	nmax = 100;
	if(Nexamples<100) nmax=Nexamples;
	
	for(j=0;j<Ninputs;j++)
		{
		fmean[j] = 0;
		for(ipat=0;ipat<nmax;ipat++)
			{
			fmean[j] += (dbl) inputs[ipat][j];
			}
		fmean[j] = fmean[j]/(dbl) nmax;
		
/* now compute real mean and sigma, min and max */		
		mean[j] = 0;
		sigma[j] = 0;
		minimum[j] = 99999;
		maximum[j] = -99999;
		for(ipat=0;ipat<Nexamples;ipat++)
			{
			mean[j] += (dbl) inputs[ipat][j];
			sigma[j] += ((dbl) inputs[ipat][j]-fmean[j])*
				    ((dbl) inputs[ipat][j]-fmean[j]);
			if((dbl) inputs[ipat][j] > maximum[j]) 
				maximum[j]=(dbl) inputs[ipat][j];	    
			if((dbl) inputs[ipat][j] < minimum[j]) 
				minimum[j]=(dbl) inputs[ipat][j];	    
			}
		mean[j] = mean[j]/(dbl) Nexamples;
		sigma[j] = sqrt(sigma[j]/ (dbl) Nexamples - 
			        (mean[j]-fmean[j])*
				(mean[j]-fmean[j]));	
		}
	free(fmean);
	return 0;
}

/***********************************************************/
/* MLP_PrintInputStat                                      */
/*                                                         */
/* prints statistics about the inputs: mean, RMS, min, max */
/*                                                         */
/* return value (int) = error code: 0 = OK		   */
/*				    -111 = could not       */
/*					   allocate memory */
/*                                                         */
/* Author: J.Schwindling   11-Oct-99                       */
/* Modified: J.Schwindling 31-Jan-2000: return value       */
/***********************************************************/   

int MLP_PrintInputStat()
{
	int j;
	dbl *mean, *sigma, *minimum, *maximum;

/* allocate memory */
	mean = (dbl *) malloc(NET.Nneur[0]*sizeof(dbl));
	sigma = (dbl *) malloc(NET.Nneur[0]*sizeof(dbl));
	minimum = (dbl *) malloc(NET.Nneur[0]*sizeof(dbl));
	maximum = (dbl *) malloc(NET.Nneur[0]*sizeof(dbl));
    int returnCode = -111; // to return if any malloc failed

    if(mean && sigma && minimum && maximum) {

        MLP_StatInputs(PAT.Npat[0],NET.Nneur[0],PAT.Rin[0],mean,sigma,minimum,maximum);

        printf("\t mean \t\t RMS \t\t min \t\t max\n");
        for(j=0;j<NET.Nneur[0];j++)
        {
            printf("var%d \t %e \t %e \t %e \t %e\n",j+1,
            mean[j],sigma[j],minimum[j],maximum[j]);
        }
        returnCode = 0; // everything went fine
    }

	free(mean);
	free(sigma);
	free(minimum);
	free(maximum);	
	printf("\n");
	return returnCode;
}


/***********************************************************/
/* NormalizeInputs                                         */
/*                                                         */
/* normalize the inputs: I' = (I - <I>) / RMS(I)           */
/*                                                         */
/* return value (int) = error code: 0 = OK		   */
/*				    -111 = could not       */
/*					   allocate memory */
/*                                                         */
/* Author: J.Schwindling   04-May-1999                     */
/* Modified: J.Schwindling 31-Jan-2000: return value       */
/***********************************************************/   

/* extern "C"Dllexport */int NormalizeInputs()
{
	int j, ipat;
	dbl *mean, *sigma, *minimum, *maximum;

/* allocate memory */
	mean = (dbl *) malloc(NET.Nneur[0]*sizeof(dbl));
	sigma = (dbl *) malloc(NET.Nneur[0]*sizeof(dbl));
	STAT.mean = (dbl *) malloc(NET.Nneur[0]*sizeof(dbl));
	STAT.sigma = (dbl *) malloc(NET.Nneur[0]*sizeof(dbl));
	minimum = (dbl *) malloc(NET.Nneur[0]*sizeof(dbl));
	maximum = (dbl *) malloc(NET.Nneur[0]*sizeof(dbl));
    int returnCode = -111; // to return if any malloc failed

    if(mean && sigma && minimum && maximum && STAT.mean && STAT.sigma) {

        MLP_StatInputs(PAT.Npat[0],NET.Nneur[0],PAT.Rin[0],mean,sigma,minimum,maximum);

        if(NET.Debug>=1) printf("\t mean \t\t RMS \t\t min \t\t max\n");
        for(j=0;j<NET.Nneur[0];j++)
        {
            if(NET.Debug>=1)
                printf("var%d \t %e \t %e \t %e \t %e\n",j+1,
                    mean[j],sigma[j],minimum[j],maximum[j]);

            /* store mean and sigma for output function */
            STAT.mean[j] = mean[j];
            STAT.sigma[j] = sigma[j];

            /* finally apply the normalization */
            for(ipat=0;ipat<PAT.Npat[0];ipat++)
            {
                PAT.Rin[0][ipat][j] =
                (PAT.Rin[0][ipat][j]-(float) mean[j])/
                (float) sigma[j];
            }
            for(ipat=0;ipat<PAT.Npat[1];ipat++)
            {
                PAT.Rin[1][ipat][j] =
                (PAT.Rin[1][ipat][j]-(float) mean[j])/
                (float) sigma[j];
            }
        }
        returnCode = 0; // everything went fine
    }
	
	free(mean);
	free(sigma);
	free(minimum);
	free(maximum);	
	if(NET.Debug>=1) printf("\n");
	return returnCode;
}


/***********************************************************/
/* AllocNetwork                                            */
/*                                                         */
/* memory allocation for weights, etc		           */
/*                                                         */
/* inputs:	int Nlayer: number of layers		   */
/*		int *Neurons: nulber of neurons per layer  */
/*                                                         */
/* return value (int): error = 0: no error		   */
/*			     = -111: could not allocate mem*/ 
/*                                                         */
/* Author: J.Schwindling   28-Sep-99                       */
/***********************************************************/   

int AllocNetwork(int Nlayer, int *Neurons)
{
	int i, j, k, l;
	
	if(NetMemory != 0) FreeNetwork();
	NetMemory = 1;
	
	NET.Nneur = (int *) malloc(Nlayer*sizeof(int));
	if(NET.Nneur == nullptr) return -111;
	
	NET.T_func = (int **) malloc(Nlayer*sizeof(int *));
	NET.Deriv1 = (dbl **) malloc(Nlayer*sizeof(dbl *));
	NET.Inn = (dbl **) malloc(Nlayer*sizeof(dbl *));
	NET.Outn = (dbl **) malloc(Nlayer*sizeof(dbl *));
	NET.Delta = (dbl **) malloc(Nlayer*sizeof(dbl *));
	if(NET.T_func == nullptr || NET.Deriv1 == nullptr
		|| NET.Inn == nullptr || NET.Outn == nullptr
		|| NET.Delta == nullptr) return -111;
	
	for(i=0; i<Nlayer; i++)
		{
		NET.T_func[i] = (int *) malloc(Neurons[i]*sizeof(int));
		NET.Deriv1[i] = (dbl *) malloc(Neurons[i]*sizeof(dbl));
		NET.Inn[i] = (dbl *) malloc(Neurons[i]*sizeof(dbl));
		NET.Outn[i] = (dbl *) malloc(Neurons[i]*sizeof(dbl));
		NET.Delta[i] = (dbl *) malloc(Neurons[i]*sizeof(dbl));
		if(NET.T_func[i] == nullptr || NET.Deriv1[i] == nullptr 
			|| NET.Inn[i] == nullptr || NET.Outn[i] == nullptr
			|| NET.Delta[i] ==nullptr ) return -111;
		}
		
	NET.Weights = (dbl ***) malloc(Nlayer*sizeof(dbl **));
	NET.vWeights = (dbl **) malloc(Nlayer*sizeof(dbl *));
	LEARN.Odw = (dbl ***) malloc(Nlayer*sizeof(dbl **));
	LEARN.ODeDw = (dbl ***) malloc(Nlayer*sizeof(dbl **));
	LEARN.DeDw = (dbl ***) malloc(Nlayer*sizeof(dbl **));
	if(NET.Weights == nullptr || NET.vWeights == nullptr 
	  || LEARN.Odw == nullptr || LEARN.ODeDw == nullptr
	  || LEARN.DeDw == nullptr)  return -111;
	  
	for(i=1; i<Nlayer; i++)
		{
		k = Neurons[i-1]+1;
		NET.vWeights[i] = (dbl *) malloc(k * Neurons[i] *
						sizeof(dbl));
		NET.Weights[i] = (dbl **) malloc(Neurons[i]*sizeof(dbl *));
		LEARN.Odw[i] = (dbl **) malloc(Neurons[i]*sizeof(dbl *));
		LEARN.ODeDw[i] = (dbl **) malloc(Neurons[i]*sizeof(dbl *));
		LEARN.DeDw[i] = (dbl **) malloc(Neurons[i]*sizeof(dbl *));
		if(NET.Weights[i] == nullptr || NET.vWeights[i] == nullptr 
		  || LEARN.Odw[i] == nullptr || LEARN.ODeDw[i] == nullptr
		  || LEARN.DeDw[i] == nullptr)  return -111;
		  
		for(j=0; j<Neurons[i]; j++)
			{
			NET.Weights[i][j] = &(NET.vWeights[i][j*k]);
			LEARN.Odw[i][j] = (dbl *) malloc(k*sizeof(dbl));
			LEARN.ODeDw[i][j] = (dbl *) malloc(k*sizeof(dbl));
			LEARN.DeDw[i][j] = (dbl *) malloc(k*sizeof(dbl));
			if(LEARN.Odw[i][j] == nullptr 
			  || LEARN.ODeDw[i][j] == nullptr
			  || LEARN.DeDw[i][j] == nullptr)  return -111;
			
			for(l=0; l<k; l++)
				{
				LEARN.Odw[i][j][l] = 0;
				LEARN.ODeDw[i][j][l] = 0;
				}
			}
		}
	return 0;	
}


/***********************************************************/
/* FreeNetwork                                             */
/*                                                         */
/* frees the memory allocated for the network	           */
/*                                                         */
/* Author: J.Schwindling   06-Oct-99                       */
/***********************************************************/   

void FreeNetwork()
{
	int i, j;
	for(i=1; i<NET.Nlayer; i++)
		{
		for(j=0; j<NET.Nneur[i]; j++)
			{
/*			free(NET.Weights[i][j]); */
			free(LEARN.Odw[i][j]);
			free(LEARN.ODeDw[i][j]);
			free(LEARN.DeDw[i][j]);
			}
		free(NET.vWeights[i]);
		free(NET.Weights[i]);
		free(LEARN.Odw[i]);
		free(LEARN.ODeDw[i]);
		free(LEARN.DeDw[i]);
		}
	free(NET.Weights);
	free(LEARN.Odw);
	free(LEARN.ODeDw);
	free(LEARN.DeDw);	
	
	free(NET.Nneur);
	
	for(i=0; i<NET.Nlayer; i++)
		{
		free(NET.T_func[i]);
		free(NET.Deriv1[i]);
		free(NET.Inn[i]);
		free(NET.Outn[i]);
		free(NET.Delta[i]);
		}
	free(NET.T_func);
	free(NET.Deriv1);
	free(NET.Inn);
	free(NET.Outn);
	free(NET.Delta);
	
	NetMemory = 0;	
}

/***********************************************************/
/* GetNetStructure                                         */
/*                                                         */
/* given a strinng like "3,4,1" returns the network        */
/* structure						   */
/*                                                         */
/* inputs:	char *s: input string 			   */
/*                                                         */
/* outputs:	int *Nlayer: number of layers		   */
/*              int *Nneur: number of neurons per layer    */
/*                                                         */
/* return value (int): error = 0: no error		   */
/*			     = -1: s is empty              */ 
/*			     = -2: s is too long	   */
/*			     = -3: too many layers	   */
/*                                                         */
/* Author: J.Schwindling   04-Oct-99                       */
/***********************************************************/   

int GetNetStructure(char *s, int *Nlayer, int *Nneur)
{
	int i=0;
	char tmp[1024];

	if(strlen(s)==0) return -1;
	if(strlen(s)>1024) return -2;

	strcpy(tmp,s);
	if (strtok(tmp,","))
    		{
      		i=1;
      		while (strtok(nullptr,",")) i++;
    		}
	*Nlayer = i;
	if(i > NLMAX) return -3;

	strcpy(tmp,s);
	if (*Nlayer>0)
    		{
      		sscanf(strtok(tmp,","),"%d",&(Nneur[0]));
      		for (i=1;i<*Nlayer;i++)
        		sscanf(strtok(nullptr,","),"%d",&(Nneur[i]));
    		}

	return 0;
}


/***********************************************************/
/* MLP_SetNet                                              */
/*                                                         */
/* to set the structure of a neural network                */
/* inputs:     int *nl = number of layers                  */
/*	       int *nn = number of neurons         	   */
/*                                                         */
/* return value (int) = error value:   		           */
/*                              0: no error		   */ 
/*                              1: N layers > NLMAX	   */ 
/*                              2: N layers < 2		   */ 
/*                           -111: Error allocating memory */ 
/*                                                         */
/* Author: J.Schwindling   14-Apr-99                       */
/* Modified: J.Schwindling 05-Oct-99 allocate memory       */
/* Modified: J.Schwindling 29-Nov-99 LearnFree, LearnAlloc */
/***********************************************************/   
                                                
int MLP_SetNet(int *nl, int *nn)
{
    int il,ierr;

    if((*nl)>NLMAX) return(1);
    if((*nl)<2) return(2);
    
/*    LearnFree(); */
/* allocate memory */      
    ierr = AllocNetwork(*nl,nn);
    if(ierr != 0) return ierr;
      
/* set number of layers */               
    NET.Nlayer = (int) *nl;

/* set number of neurons */               
    for(il=0; il<NET.Nlayer; il++) {
       NET.Nneur[il] = nn[il];
      }

/* set transfer functions */      
    SetDefaultFuncs();
/*    LearnAlloc(); */
      
    return(0); 
}


/***********************************************************/
/* MLP_MatrixVectorBias                                    */
/*                                                         */
/* computes a Matrix-Vector product                        */
/* r[j] = M[j][0] + Sum_i M[j][i] v[i]			   */ 
/*                                                         */
/* inputs:     dbl *M = matrix (n lines, m+1 columns)      */
/*	       dbl *v = vector (dimension m)       	   */
/*	       dbl *r = resulting vector (dimension n) 	   */
/*	       int n  					   */
/*	       int m  					   */ 
/*                                                         */
/* Author: J.Schwindling   24-Jan-00                       */
/***********************************************************/
   
void MLP_MatrixVectorBias(dbl *M, dbl *v, dbl *r, int n, int m)
{
	int i,j;
	dbl a1, a2, a3, a4, c, d;
	dbl *pM1 = M;
	dbl *pM2 = &(M[m+1]);
	dbl *pM3 = &(M[2*(m+1)]);
	dbl *pM4 = &(M[3*(m+1)]);
	dbl *pr = r;
	int mp1 = m+1;
	
	for(i=0; i<n-3; 
		i+=4, pM1 += 3*mp1, pM2 += 3*mp1, pM3 += 3*mp1, pM4 += 3*mp1, 
		pr+=4)
		{
		a1 = *pM1;
		a2 = *pM2;
		a3 = *pM3;
		a4 = *pM4;
		pM1++; pM2++; pM3++; pM4++;
		for(j=0; j<m-1; j+=2, pM1+=2, pM2+=2, pM3+=2, pM4+=2) 
			{
			c = v[j];
			d = v[j+1];
			a1 = a1 + *pM1 * c + *(pM1+1) * d;
			a2 = a2 + *pM2 * c + *(pM2+1) * d; 
			a3 = a3 + *pM3 * c + *(pM3+1) * d;
			a4 = a4 + *pM4 * c + *(pM4+1) * d; 
			}
		for(/*j set above*/; j<m; j++, pM1++, pM2++, pM3++, pM4++)
			{
			c = v[j];
			a1 = a1 + *pM1 * c;
			a2 = a2 + *pM2 * c; 
			a3 = a3 + *pM3 * c;
			a4 = a4 + *pM4 * c; 
			}	
		*pr = a1; *(pr+1) = a2; *(pr+2) = a3; *(pr+3) = a4;
		}
	for(/*i set above*/; i<n; i++)
		{
		pM1 = &(M[i*(m+1)]);
		a1 = *pM1;
		pM1++;
		for(j=0; j<m; j++, pM1++)
			{
			a1 = a1 + *pM1 * v[j];
			}
		r[i] = a1;	
		}	
}
/***********************************************************/
/* MLP_MatrixVector 	                                   */
/*                                                         */
/* computes a Matrix-Vector product                        */
/* r[j] = Sum_i M[j][i] v[i]				   */ 
/*                                                         */
/* inputs:     dbl *M = matrix (n lines, m+1 columns)      */
/*	       dbl *v = vector (dimension m)       	   */
/*	       dbl *r = resulting vector (dimension n) 	   */
/*	       int n  					   */
/*	       int m  					   */ 
/*                                                         */
/* Author: J.Schwindling   24-Jan-00                       */
/***********************************************************/
   
void MLP_MatrixVector(dbl *M, type_pat *v, dbl *r, int n, int m)
{
	int i,j;
	dbl a1, a2, a3, a4, c, d;
	dbl *pM1 = M;
	dbl *pM2 = &(M[m]);
	dbl *pM3 = &(M[2*m]);
	dbl *pM4 = &(M[3*m]);
	dbl *pr = r;
	int mp1 = m;
	
	for(i=0; i<n-3; 
		i+=4, pM1 += 3*mp1, pM2 += 3*mp1, pM3 += 3*mp1, pM4 += 3*mp1, 
		pr+=4)
		{
		a1 = 0;
		a2 = 0;
		a3 = 0;
		a4 = 0;
		for(j=0; j<m-1; j+=2, pM1+=2, pM2+=2, pM3+=2, pM4+=2) 
			{
			c = v[j];
			d = v[j+1];
			a1 = a1 + *pM1 * c + *(pM1+1) * d;
			a2 = a2 + *pM2 * c + *(pM2+1) * d; 
			a3 = a3 + *pM3 * c + *(pM3+1) * d;
			a4 = a4 + *pM4 * c + *(pM4+1) * d; 
			}
		for(/*j set above*/; j<m; j++, pM1++, pM2++, pM3++, pM4++)
			{
			c = v[j];
			a1 = a1 + *pM1 * c;
			a2 = a2 + *pM2 * c; 
			a3 = a3 + *pM3 * c;
			a4 = a4 + *pM4 * c; 
			}	
		*pr = a1; *(pr+1) = a2; *(pr+2) = a3; *(pr+3) = a4;
		}
	for(/*i set above*/; i<n; i++)
		{
		pM1 = &(M[i*m]);
		a1 = 0;
		for(j=0; j<m; j++, pM1++)
			{
			a1 = a1 + *pM1 * v[j];
			}
		r[i] = a1;	
		}	
}


/***********************************************************/
/* MLP_MM2rows	                                           */
/*                                                         */
/* computes a Matrix-Matrix product, with the first matrix */
/* having 2 rows		                           */
/*                                                         */
/* inputs:     dbl *c = resulting matrix (Nj * Nk)         */
/*	       dbl *a = first matrix (Ni * Nj)       	   */
/*	       dbl *b = second matrix (Nj * Nk) 	   */
/*	       int Ni 					   */
/*	       int Nj 					   */ 
/*	       int Nk 					   */ 
/*	       int NaOffs				   */ 
/*	       int NbOffs				   */ 
/*                                                         */
/* Author: J.Schwindling   24-Jan-00                       */
/***********************************************************/
   
void MLP_MM2rows(dbl* c, type_pat* a, dbl* b,
             int Ni, int Nj, int Nk, int NaOffs, int NbOffs)
{
//int i,j,k;
int j,k;
dbl s00,s01,s10,s11;
type_pat *pa0,*pa1;
dbl *pb0,*pb1,*pc0,*pc1;

  for (j=0; j<=Nj-2; j+=2)
   {
    pc0 = c+j;
    pc1 = c+j+Nj;
    s00 = 0.0; s01 = 0.0; s10 = 0.0; s11 = 0.0;

    for (k=0,pb0=b+k+NbOffs*j,
             pb1=b+k+NbOffs*(j+1),
             pa0=a+k,
             pa1=a+k+NaOffs;
         k<Nk;
         k++,pa0++,
             pa1++,
             pb0++,
             pb1++)
     {
      s00 += (*pa0)*(*pb0);
      s01 += (*pa0)*(*pb1);
      s10 += (*pa1)*(*pb0);
      s11 += (*pa1)*(*pb1);
     }
    *pc0 = s00; *(pc0+1) = s01; *pc1 = s10; *(pc1+1) = s11;
   }
  for (/*j set above*/; j<Nj; j++)
   {
    pc0 = c+j;
    pc1 = c+j+Nj;
    s00 = 0.0; s10 = 0.0;
    for (k=0,pb0=b+k+NbOffs*j,
             pa0=a+k,
             pa1=a+k+NaOffs;
         k<Nk;
         k++,pa0++,
             pa1++,
             pb0++)
     {
      s00 += (*pa0)*(*pb0);
      s10 += (*pa1)*(*pb0);
     }
    *pc0 = s00; *pc1 = s10;
   }
}

#ifdef __cplusplus
} // extern "C"
#endif
