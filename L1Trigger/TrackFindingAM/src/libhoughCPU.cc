////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

/* Template project which demonstrates the basics on how to setup a project
 * example application.
 * Host code.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <stdint.h>


#include "../interface/libhoughCPU.h"





void cleanUICPU(int maxblock,int maxthread,unsigned int *d_layer)
{
for (int b_idx=0;b_idx<maxblock;b_idx++)  for (int t_idx=0;t_idx<maxthread;t_idx++) {{ //STARTOFLOOP
  d_layer[b_idx]=0;
    }}
}
void cleanUI1CPU(int maxblock,int maxthread,unsigned int *d_hough,unsigned int *d_hough_layer,unsigned int *d_hough_map,houghLimits hl)
{
  const unsigned nbinrho=hl.nrho;//int(limits[5]);
for (int b_idx=0;b_idx<maxblock;b_idx++)  for (int t_idx=0;t_idx<maxthread;t_idx++) {{ //STARTOFLOOP

  const unsigned int ith=t_idx;
  const unsigned int ir=b_idx;
  //const unsigned int nbintheta=int(limits[4]);

  d_hough_layer[ith*nbinrho+ir]=0;
  d_hough[ith*nbinrho+ir]=0;
    }}
}
void cleanFCPU(int maxblock,int maxthread,float *d_layer)
{
  for (int b_idx=0;b_idx<maxblock;b_idx++)  for (int t_idx=0;t_idx<maxthread;t_idx++) {{ //STARTOFLOOP

  d_layer[b_idx]=0;
      }}
}
void fillHoughCPU(int maxblock,int maxthread,short *d_val,unsigned int* d_layer,unsigned int* d_hough,unsigned int* d_hough_layer,unsigned int* d_hough_map,houghLimits hl)
{
  // shared memory
  // the size is determined by the host application
  //const unsigned int nbintheta=blockDim.x;
  const unsigned int nbinrho=hl.nrho;//int(limits[5]);
  const unsigned int nbintheta=hl.ntheta;//int(limits[4]);

for (int b_idx=0;b_idx<maxblock;b_idx++)  for (int t_idx=0;t_idx<maxthread;t_idx++) {{ //STARTOFLOOP

  const unsigned int is=b_idx;
  const unsigned int ith=t_idx;
  short ir= d_val[is*nbintheta+ith];
  if (ir>=0)
    {
      d_hough[ith*nbinrho+ir]+=1;
      d_hough_layer[ith*nbinrho+ir]|=(1<<(d_layer[is]&0xFFFF));
    }
    }}

}


void localRegressionCPU(float xp,float yp,float* d,float* res)
{

  double z2=d[1],z=d[0],zx=d[2],x=d[3],n=d[4];
  z/=n;
  z2/=n;
  zx/=n;
  x/=n;
  double s2z = z2-z*z;
  double szx = zx-z*x;
  
  double a = szx/s2z;
  double b=x -a*z;
  double phi=atan(a);
  double theta=phi+PI/2.;
  double r=b*sin(theta);
  double R=0.5/fabs(r);
  double pt=0.3*3.8*R/100.;
  double xi=-a/2./b;
  //  double yi=1./2./b;
  if (phi<0) phi+=2*PI;
  if (xp>0 && yp>0 && phi>PI) phi-=PI;
  if (xp<0 && yp>0 && phi>PI) phi-=PI;
  if (xp<0 && yp<0 && phi<PI) phi+=PI;
  if (xp>0 && yp<0 && phi<PI) phi+=PI;
  


  res[0]=a;
  res[1]=b;
  res[2]=phi;
  res[3]=theta;
  res[4]=r;
  res[5]=R;
  res[6]=pt;
  res[7]=xi;
  res[8]=-log(fabs(tan(atan(a)/2)));
  if (z<0) res[8]=-res[8];
  res[9]=n;

 
}


void computeHoughPointCPU(int maxblock,int maxthread,float *d_x, float *d_y,short* d_val,houghLimits hl)
{
  // shared memory
  // the size is determined by the host application
  //const unsigned int nbintheta=blockDim.x;
  const short nbintheta=hl.ntheta;//int(limits[4]);
  const short nbinrho=hl.nrho;//int(limits[5]);
  const float thmin=hl.thetamin;
  const float thmax=hl.thetamax;
  const float rmin=hl.rmin;
  const float rmax=hl.rmax;

for (int b_idx=0;b_idx<maxblock;b_idx++)  for (int t_idx=0;t_idx<maxthread;t_idx++) {{ //STARTOFLOOP

  const unsigned int is=b_idx;
  const unsigned int ith=t_idx;
  
  
  double theta=thmin+(ith+0.5)*(thmax-thmin)/nbintheta;
  double r=d_x[is]*cos(theta)+d_y[is]*sin(theta);
  short ir=int(floor((r-rmin)/((rmax-rmin)/nbinrho)));
  if (ir>=0 && ir<nbinrho) 
    {
      d_val[is*nbintheta+ith]=ir;
    }
  else
    d_val[is*nbintheta+ith]=-1;
  //sdata[tid*nrhoword]=12;
    }}
}




void ListHoughPointCPU(int maxblock,int maxthread,unsigned int* d_hough,unsigned int* d_hough_layer,unsigned int min_val,unsigned int min_layer,unsigned int* d_cand,houghLimits hl,bool endcap)
{

  const unsigned int nbinrho=hl.nrho;//int(limits[5]);
  const unsigned int nbintheta=hl.ntheta;//int(limits[4]);
  

  int pointerIndex=0;

for (int b_idx=0;b_idx<maxblock;b_idx++)  for (int t_idx=0;t_idx<maxthread;t_idx++) {{ //STARTOFLOOP
  const unsigned int ith=t_idx;
  const unsigned int ir= b_idx;
  if (d_hough[ith*nbinrho+ir]>=min_val)
    {
      bool nmax=false;
      float val=d_hough[ith*nbinrho+ir]*1.;
      if (ith>0 && ir>0 && ith<=(nbintheta-1) && ir<(nbinrho-1))
	{
	  if ((val-d_hough[(ith-1)*nbinrho+ir])<0) nmax=true;
	  if ((val-d_hough[(ith+1)*nbinrho+ir])<0) nmax=true;
	  if((val-d_hough[(ith-1)*nbinrho+(ir-1)])<0) nmax=true;
	  if ((val-d_hough[(ith)*nbinrho+ir-1])<0) nmax=true;
	  if((val-d_hough[(ith+1)*nbinrho+ir-1])<0) nmax=true;
	  if((val-d_hough[(ith-1)*nbinrho+(ir+1)])<0) nmax=true;
	  if((val-d_hough[(ith)*nbinrho+ir+1])<0) nmax=true;
	  if((val-d_hough[(ith+1)*nbinrho+ir+1])<0) nmax=true;
	}
      if (!nmax)
	{
	  unsigned int pattern=d_hough_layer[ith*nbinrho+ir];

	  if (ith>0 && ir>0 && ith<=(nbintheta-1) && ir<(nbinrho-1))
	    {
	      pattern |=d_hough_layer[(ith-1)*nbinrho+ir];
	      pattern |=d_hough_layer[(ith+1)*nbinrho+ir];
	      pattern |=d_hough_layer[(ith-1)*nbinrho+ir-1];
	      pattern |=d_hough_layer[ith*nbinrho+ir-1];
	      pattern |=d_hough_layer[(ith+1)*nbinrho+ir-1];
	      pattern |=d_hough_layer[(ith-1)*nbinrho+ir+1];
	      pattern |=d_hough_layer[ith*nbinrho+ir+1];
	      pattern |=d_hough_layer[(ith+1)*nbinrho+ir+1];
	    }
	  pattern=d_hough_layer[ith*nbinrho+ir]; //@essai
	  unsigned int np=0;
	  bool l[24];
	  for (int ip=1;ip<=24;ip++)
	    {
	      l[ip]=((pattern &(1<<ip))!=0);
	      if (l[ip]) np++;
	    }
	  bool bar56=(l[5]&&l[6])||(l[5]&&l[7])||(l[6]&&l[7]) || endcap;
	  if (endcap) 
	    bar56=l[5];
	  // bar56=true;
	  //np=10;
	  if (np>=min_layer && d_hough[ith*nbinrho+ir]>=min_val && bar56)
	    {
	      //unsigned int id=atomicInc(&pointerIndex,GPU_MAX_CAND);
	      unsigned int id=pointerIndex;
	      pointerIndex++;
	      //d_cand[0]+=1;
	      if (id<GPU_MAX_CAND-1)
		d_cand[id+1]=(ith & 0x3FF)+((ir&0x3FF)<<10)+((d_hough[ith*nbinrho+ir]&0x3FF)<<20);
	    }
	}
    }
  //if (ith==10 && ir==10) d_cand[0]=ith*gridDim.x+ir;
    }}
  //if (ith==1 && ir==1)
  d_cand[0]=pointerIndex;

}

void conformalPositionCPU(int maxblock,int maxthread,float* d_xo,float* d_yo,float* d_ro)
{
 for (int b_idx=0;b_idx<maxblock;b_idx++)  for (int t_idx=0;t_idx<maxthread;t_idx++) {{ //STARTOFLOOP

   unsigned int ib=b_idx;

  double r2=d_xo[ib]*d_xo[ib]+d_yo[ib]*d_yo[ib];
  double r=sqrt(r2);
  double x= d_xo[ib]/r2;
  double y= d_yo[ib]/r2;
  d_xo[ib]=x;
  d_yo[ib]=y;
  d_ro[ib]=r;
     }}
}

void copyFromValCPU(int maxblock,int maxthread,unsigned int ith,unsigned int ir,unsigned int nbintheta,short* d_val,float* d_xi,float* d_yi,unsigned int* di_layer,float* d_ri,float* d_zi,float* d_xo,float* d_yo,unsigned int* do_layer,float* d_ro,float* d_zo,float* d_reg,bool regression,unsigned int* d_temp,bool endcap)
{



 for (int b_idx=0;b_idx<maxblock;b_idx++)  for (int t_idx=0;t_idx<maxthread;t_idx++) {{ //STARTOFLOOP
  unsigned int ib=t_idx;

  //__threadfence();
  if (d_val[ib*nbintheta+ith]==(short)ir )
    {
      int iwm=ib/32;
      int ibm=ib%32;
      if (!(d_temp[iwm] & (1<<ibm)))
	{
	  d_temp[iwm]|=(1<<ibm); // no problem done bin/bin so one stub cannot be set in //
	  float fid=d_reg[20];
	  d_reg[20]+=1.;
	  unsigned int id=int(fid);
      //d_cand[0]+=1;
      if (id<GPU_MAX_STUB)
	{
	  float x=d_xi[ib],y=d_yi[ib],r=d_ri[ib],z=d_zi[ib];
	  unsigned int la=di_layer[ib]; 
	  unsigned int zinfo=(di_layer[ib]>>16)&0xF; 
	  d_xo[id]=x;
	  d_yo[id]=y;
	  d_ro[id]=r;
	  d_zo[id]=z;
	  do_layer[id]=la;
	  if (regression)
	    {
	      d_reg[50]+=x;
	      d_reg[51]+=x*x;
	      d_reg[52]+=x*y;
	      d_reg[53]+=y;
	      d_reg[54]+=1.;
	      //if ((l==5) || (l==6) || (l==7) || endcap)
	      if (zinfo!=0)
		{
		  d_reg[55]+=z;
		  d_reg[56]+=z*z;
		  d_reg[57]+=z*r;
		  d_reg[58]+=r;
		  d_reg[59]+=1.;
		}
	    }
	}
	}
    }
  //if (ith==10 && ir==10) d_cand[0]=ith*gridDim.x+ir;
     }} //ENDOLOOP
  //__threadfence();
  //if (ith==1 && ir==1)
  //  d_cand[0]=pointerIndex;
  if (regression)
    {
      localRegressionCPU(d_xo[0],d_yo[0],&d_reg[50],&d_reg[60]);
      localRegressionCPU(d_xo[0],d_yo[0],&d_reg[55],&d_reg[70]);
    }
}


void
clearFloatCPU(int maxblock,int maxthread,float* d_float)
{
 for (int b_idx=0;b_idx<maxblock;b_idx++)  for (int t_idx=0;t_idx<maxthread;t_idx++) {{ //STARTOFLOOP
  d_float[t_idx]=0;
     }}
}
void
clearUICPU(int maxblock,int maxthread,unsigned int* d_float)
{
 for (int b_idx=0;b_idx<maxblock;b_idx++)  for (int t_idx=0;t_idx<maxthread;t_idx++) {{ //STARTOFLOOP
  d_float[t_idx]=0;
     }}

}

void copyPositionCPU(int maxblock,int maxthread,unsigned int* d_map,float* d_xi,float* d_yi,unsigned int* di_layer,float* d_ri,float* d_zi,float* d_xo,float* d_yo,unsigned int* do_layer,float* d_ro,float* d_zo,float* d_reg,bool regression,bool endcap)
{

  int pointerIndex=0;
  for (int i=50;i<60;i++)
    d_reg[i]=0;



  for (int b_idx=0;b_idx<maxblock;b_idx++)  for (int t_idx=0;t_idx<maxthread;t_idx++) {{ //STARTOFLOOP

	unsigned int ib=t_idx;

	int iwm=ib/32;
	int ibm=ib%32;
    if ((d_map[iwm]&(1<<ibm)) )
    {
      unsigned int id=pointerIndex++;
      //d_cand[0]+=1;
      if (id<512)
	{
	  float x=d_xi[ib],y=d_yi[ib],r=d_ri[ib],z=d_zi[ib];
	  unsigned int la=di_layer[ib]; 
	  //unsigned int l=di_layer[ib]&0xFFFF; 
	  unsigned int zinfo=(di_layer[ib]>>16)&0xF; 
	  d_xo[id]=x;
	  d_yo[id]=y;
	  d_ro[id]=r;
	  d_zo[id]=z;
	  do_layer[id]=la;
	  if (regression)
	    {
	      d_reg[50]+=x;
	      d_reg[51]+=x*x;
	      d_reg[52]+=x*y;
	      d_reg[53]+=y;
	      d_reg[54]+=1.;
	      //if ((l==5) || (l==6) || (l==7) || endcap)
		if (zinfo!=0)
		{
		  d_reg[55]+=z;
		  d_reg[56]+=z*z;
		  d_reg[57]+=z*r;
		  d_reg[58]+=r;
		  d_reg[59]+=1.;
		}
	    }
	}
    }
  //if (ith==10 && ir==10) d_cand[0]=ith*gridDim.x+ir;
      }} //ENDOFLOOP
  if (regression)
    {
      localRegressionCPU(d_xo[0],d_yo[0],&d_reg[50],&d_reg[60]);
      localRegressionCPU(d_xo[0],d_yo[0],&d_reg[55],&d_reg[70]);
    }
}


void createHoughCPU(houghParam* p,uint32_t max_stub,uint32_t max_theta,uint32_t max_rho)
{
  p->max_stub=max_stub;
  p->max_theta=max_theta;
  p->max_rho=max_rho;
   p->h_cand = (unsigned int*) malloc(GPU_MAX_CAND*sizeof(unsigned int));
   p->h_temp = (unsigned int*) malloc(512*sizeof(unsigned int));
   p->h_reg = (float*) malloc(GPU_MAX_REG*sizeof(float));
   p->h_val = (short*) malloc(max_stub*max_theta*sizeof(short));
   
   p->d_reg=(float*)malloc(GPU_MAX_REG *sizeof(float));
   p->d_temp=(unsigned int*)malloc(512 *sizeof(unsigned int));
   
   p->d_val=(short*) malloc(max_stub*max_theta*sizeof(short));
   p->d_x=(float*)malloc(max_stub *sizeof(float));
   p->d_y=(float*)malloc( max_stub*sizeof(float));
   p->d_r=(float*)malloc( max_stub*sizeof(float));
   p->d_z=(float*)malloc( max_stub*sizeof(float));
   p->d_layer=(unsigned int*)malloc( max_stub*sizeof(unsigned int));
   p->d_cand=(unsigned int*)malloc( GPU_MAX_CAND*sizeof(unsigned int));
   p->d_hough=(unsigned int*)malloc(max_theta*max_rho*sizeof(unsigned int) );
   p->d_hough_layer=(unsigned int*)malloc(max_theta*max_rho*sizeof(unsigned int) );

}
void clearHoughCPU(houghParam* p)
{
  //cleanUI1Kernel<<<p->nrho,p->ntheta>>>(p->d_hough,p->d_hough_layer,p->d_hough_map);
  //cleanUI1Kernel<<<GPU_MAX_RHO,GPU_MAX_THETA>>>(p->d_hough_layer);
  clearFloatCPU(1,GPU_MAX_REG,p->d_reg);
  clearUICPU(1,512,p->d_temp);
  clearUICPU(1,GPU_MAX_CAND,p->d_cand);
}


void initialiseHoughCPU(houghParam* p,int nstub,int ntheta,int nrho,float thetamin,float thetamax,float rmin,float rmax)
{
  p->nstub=nstub;
  p->ntheta=ntheta;
  p->nrho=nrho;
  p->thetamin=thetamin;
  p->thetamax=thetamax;
  p->thetabin=(thetamax-thetamin)/ntheta;
  p->rmin=rmin;
  p->rmax=rmax;
  p->rbin=(rmax-rmin)/nrho;


}
void deleteHoughCPU(houghParam* p)
{
  free(p->h_cand);
  free(p->h_reg);
  free(p->h_temp);
  free(p->h_val);
  free(p->d_val);
  free(p->d_x);
  free(p->d_y);
  free(p->d_r);
  free(p->d_z);
  free(p->d_layer);
  free(p->d_hough_layer);
  free(p->d_hough);
  free(p->d_cand);
  free(p->d_reg);
  free(p->d_temp);

}

void fillPositionHoughCPU(houghParam* p,float* h_x,float* h_y,float* h_z)
{
  memcpy(p->d_x, h_x,p->nstub*sizeof(float));

  memcpy(p->d_y, h_y,p->nstub*sizeof(float));

  memcpy(p->d_z, h_z,p->nstub*sizeof(float));

}

void fillLayerHoughCPU(houghParam* p,unsigned int* h_layer)
{
  memcpy(p->d_layer, h_layer,p->nstub*sizeof(unsigned int));

}

void copyHoughImageCPU(houghParam* p,unsigned int* h_hough)
{
  memcpy(h_hough, p->d_hough,p->ntheta*p->nrho*sizeof(unsigned int));

 
}
void copyHoughLayerCPU(houghParam* p,unsigned int* h_hough)
{
  memcpy(h_hough, p->d_hough_layer,p->ntheta*p->nrho*sizeof(unsigned int));

 
}

void fillConformalHoughCPU(houghParam* p,float* h_x,float* h_y,float* h_z)
{
  memcpy(p->d_x, h_x,p->nstub*sizeof(float));

  memcpy(p->d_y, h_y,p->nstub*sizeof(float));

  memcpy(p->d_z, h_z,p->nstub*sizeof(float));

  int grid1=p->nstub;
  conformalPositionCPU(grid1,1,p->d_x,p->d_y,p->d_r);



}
void dumpCPU(houghParam* p)
{

  printf("NHits %f \n",p->h_reg[20]);
  for (int i=50;i<=60;i++)
    printf("%f ",p->h_reg[i]);
  printf("\n");
  for (int i=60;i<70;i++)
    printf("%f ",p->h_reg[i]);
  printf("\n");
  for (int i=70;i<80;i++)
    printf("%f ",p->h_reg[i]);
  printf("\n");

  for (int i=0;i<p->nstub;i++)
    printf("\t %d: (%f,%f,%f) r %f Layer %x \n",i,p->d_x[i],p->d_y[i],p->d_z[i],p->d_r[i],p->d_layer[i]);
}

void copyPositionHoughCPU(houghParam* pi,int icand,houghParam* po,unsigned int mode,bool regression,bool endcap)
{
   int ith=icand&0X3FF;
   int ir=(icand>>10)&0x3FF;

   //printf("Stream %d %x \n",streamid,(unsigned long) stream);
   clearFloatCPU(1,GPU_MAX_REG,po->d_reg);
   clearUICPU(1,512,po->d_temp);

   if (mode == 1)
     {
       copyFromValCPU(1,pi->nstub,ith,ir,pi->ntheta,pi->d_val,pi->d_x,pi->d_y,pi->d_layer,pi->d_r,pi->d_z,po->d_x,po->d_y,po->d_layer,po->d_r,po->d_z,po->d_reg,regression,po->d_temp,endcap);
     }
   else
     {
       copyFromValCPU(1,pi->nstub,ith,ir,pi->ntheta,pi->d_val,pi->d_x,pi->d_y,pi->d_layer,pi->d_r,pi->d_z,po->d_x,po->d_y,po->d_layer,po->d_r,po->d_z,po->d_reg,regression,po->d_temp,endcap);

       if (ith>0)
	 copyFromValCPU(1,pi->nstub,ith-1,ir,pi->ntheta,pi->d_val,pi->d_x,pi->d_y,pi->d_layer,pi->d_r,pi->d_z,po->d_x,po->d_y,po->d_layer,po->d_r,po->d_z,po->d_reg,regression,po->d_temp,endcap);
       


       if (ith<pi->ntheta-1)
	 copyFromValCPU(1,pi->nstub,ith+1,ir,pi->ntheta,pi->d_val,pi->d_x,pi->d_y,pi->d_layer,pi->d_r,pi->d_z,po->d_x,po->d_y,po->d_layer,po->d_r,po->d_z,po->d_reg,regression,po->d_temp,endcap);
       


       if (ith>0 && ir>0)
	 copyFromValCPU(1,pi->nstub,ith-1,ir-1,pi->ntheta,pi->d_val,pi->d_x,pi->d_y,pi->d_layer,pi->d_r,pi->d_z,po->d_x,po->d_y,po->d_layer,po->d_r,po->d_z,po->d_reg,regression,po->d_temp,endcap);
       


       if (ir>0)
	 copyFromValCPU(1,pi->nstub,ith,ir-1,pi->ntheta,pi->d_val,pi->d_x,pi->d_y,pi->d_layer,pi->d_r,pi->d_z,po->d_x,po->d_y,po->d_layer,po->d_r,po->d_z,po->d_reg,regression,po->d_temp,endcap);
       


       if (ir>0 && ith<pi->ntheta-1)
	 copyFromValCPU(1,pi->nstub,ith+1,ir-1,pi->ntheta,pi->d_val,pi->d_x,pi->d_y,pi->d_layer,pi->d_r,pi->d_z,po->d_x,po->d_y,po->d_layer,po->d_r,po->d_z,po->d_reg,regression,po->d_temp,endcap);
       


       if (ith>0 && ir<pi->nrho-1)
	 copyFromValCPU(1,pi->nstub,ith-1,ir+1,pi->ntheta,pi->d_val,pi->d_x,pi->d_y,pi->d_layer,pi->d_r,pi->d_z,po->d_x,po->d_y,po->d_layer,po->d_r,po->d_z,po->d_reg,regression,po->d_temp,endcap);
       


       if (ir<pi->nrho-1)
	 copyFromValCPU(1,pi->nstub,ith,ir+1,pi->ntheta,pi->d_val,pi->d_x,pi->d_y,pi->d_layer,pi->d_r,pi->d_z,po->d_x,po->d_y,po->d_layer,po->d_r,po->d_z,po->d_reg,regression,po->d_temp,endcap);
       


       if (ir<pi->nrho-1 &&  ith<pi->ntheta-1)
	 copyFromValCPU(1,pi->nstub,ith+1,ir+1,pi->ntheta,pi->d_val,pi->d_x,pi->d_y,pi->d_layer,pi->d_r,pi->d_z,po->d_x,po->d_y,po->d_layer,po->d_r,po->d_z,po->d_reg,regression,po->d_temp,endcap);
       


     }
   
   
   memcpy(po->h_reg,po->d_reg,GPU_MAX_REG*sizeof(float));
  
   po->nstub=int(po->h_reg[20]);
  

 }

void processHoughCPU(houghParam* p,unsigned int min_cut,unsigned int min_layer,unsigned int mode,bool endcap)
{
 houghLimits hl;
  hl.thetamin=p->thetamin;
  hl.thetamax=p->thetamax;
  hl.rmin=p->rmin;
  hl.rmax=p->rmax;
  hl.ntheta=p->ntheta;
  hl.nrho=p->nrho;
  
  // setup execution parameters
  int grid1=p->nstub;
  int threads1=p->ntheta;
  int grid2=p->nrho;
  //printf("%d %d %d === %d \n",p->nstub,p->ntheta,p->nrho,mode);
  memset(p->d_val,0,p->max_theta*p->max_stub*sizeof(short));
  memset(p->d_hough,0,p->max_theta*p->max_rho*sizeof(int));
  memset(p->d_hough_layer,0,p->max_theta*p->max_rho*sizeof(int));
  //getchar();
  if (mode==0)
    {
      computeHoughPointCPU(grid1, threads1,p->d_x,p->d_y,p->d_val,hl);
    }
  else
    if (mode==1)
      computeHoughPointCPU(grid1,threads1,p->d_z,p->d_r,p->d_val,hl);
  cleanUI1CPU(p->nrho,p->ntheta,p->d_hough,p->d_hough_layer,p->d_hough_map,hl);

  //if (min_layer==4)
  fillHoughCPU(grid1, threads1,p->d_val,p->d_layer,p->d_hough,p->d_hough_layer,p->d_hough_map,hl);



  p->max_val=0;
  unsigned int threshold=4;
  //int(floor(m+3*rms));
  if (min_cut!=0 ) threshold=min_cut;
  //if (threshold<int(floor(p->max_val*0.5))) threshold=int(floor(p->max_val*0.5));
  //threshold=int(floor(m+3*rms));
   //printf("Max val %d Threshold %d \n",p->max_val,threshold);
  ListHoughPointCPU(grid2,threads1,p->d_hough,p->d_hough_layer,threshold,min_layer,p->d_cand,hl,endcap);

  memcpy(p->h_cand, p->d_cand, GPU_MAX_CAND*sizeof(unsigned int));

  
  }



