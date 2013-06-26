/* 
 *  \class TMarkov
 *
 *  $Date: 2012/02/09 10:08:10 $
 *  \author: Patrice Verrecchia - CEA/Saclay
 */

#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TMarkov.h>

#include <iostream>
#include "math.h"

//ClassImp(TMarkov)


// Constructor...
TMarkov::TMarkov()
{ 
  // TMarkov
  // ------ calcule les distributions invariantes de la chaine de TMarkov
  //   correspondantes au spectre original et retourne la dimension de u.
  //
 
  fNPeakValues=3;
  fNbinu=101;
  init();
}

// Destructor
TMarkov::~TMarkov()
{ 
}

void TMarkov::init()
{
  int i;
  for(i=0;i<fNPeakValues;i++) peak[i]=0.;
  for(i=0;i<fNbinu;i++) u[i]=0.;
  for(i=0;i<=fNbinu;i++) binu[i]=0.;
  return ;
}

int TMarkov::computeChain(int *bing)
{
  int i;int k;int nuprime;int offset=0;int m;int pass;
  double sumUprime,sumU;
  double jumpToNext,jumpToPrevious;
  double chainToNext,chainToPrevious;
  double aConst[101],uprime[101];  

  pass=0;
  for(m=3,i=1,nuprime=1;i<101;i++)
  {
       uprime[i]=0.;
       for(k=1,jumpToNext=0.,jumpToPrevious=0.;k<=m;k++)
       {
          if(i+k < 101)
	    if(bing[i] > 0 || bing[i+k] > 0)	    
              jumpToNext += exp( (double)(bing[i+k]-bing[i])
	                        /sqrt((double)(bing[i+k]+bing[i])));
          if(i-k > 0)
            if(bing[i] > 0 || bing[i-k] > 0)
              jumpToPrevious += exp( (double)(bing[i-k]-bing[i])
	                            /sqrt((double)(bing[i-k]+bing[i])));
	}
       //printf(" jump %d to %d = %f\n",i,i+1,jumpToNext);
       //printf(" jump %d to %d = %f\n",i,i-1,jumpToPrevious);
	if(jumpToNext > 0. && jumpToPrevious > 0.)
	{
	  aConst[i] = -log(jumpToNext+jumpToPrevious);
	  chainToNext = aConst[i]+log(jumpToNext);
	  chainToPrevious = aConst[i]+log(jumpToPrevious);
	  uprime[i]=chainToNext - chainToPrevious;
	  nuprime++; u[nuprime] = uprime[i];
	  if(pass == 0)
	  {  offset=i-1; pass=1;}
	}			 			     
    }
    
  //for(i=1;i<101;i++)
  //printf(" bin numero %d   uprime = %f\n",i,uprime[i]);
    
    for(k=3,sumUprime=u[2],sumU=u[2];k<nuprime+1;k++)
    {
       sumU += u[k];  u[k] = sumU;
       sumUprime += log(1.+exp(u[k]-u[k-1]));
    }      
        
    u[1] = -sumUprime;

    for(k=2;k<nuprime+1;k++)
        u[k] += u[1];

    for(i=1;i<offset+1;i++)
       binu[i]=0.;

    for(i=1;i<nuprime+1;i++)
    {
       binu[i+offset] = exp(u[i]);
       //printf(" bin numero %d   log(u) = %f\n",i+offset,u[i]); 
       //printf(" bin numero %d   u = %f\n",i+offset,exp(u[i])); 
       
    }
  
    return nuprime+offset;
}

void TMarkov::peakFinder(int *bing)
{
    int firstBin=0;int lastBin=0;
    double barycentre=0.;
    double sum=0.;
    double maximum=0.;

    int nu= computeChain(&bing[0]);

    for(int i=1;i<nu+1;i++)
    {
       sum += binu[i];
       barycentre += (double)i * binu[i];
       if(binu[i] > maximum)
       { maximum=binu[i]; imax=i; }
    }
    
    maximum *= 0.75;
    for(int i=1,pass=0;i<nu+1;i++) {
       if(binu[i] > maximum) {
         if(pass == 0) {
                firstBin=i;
                lastBin=i;
                pass=1;
         } else {
                 lastBin=i;
         }
       }
    }

    peak[0] = (barycentre/sum);
    peak[1]= (double)(lastBin-firstBin+1);
    peak[2]= sum;
}
