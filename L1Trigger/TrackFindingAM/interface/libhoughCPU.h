#ifndef _LIBHOUGHCPU_H_
#define _LIBHOUGHCPU_H_
#include "libhoughStruct.h"
extern "C" void cleanUICPU(int maxblock,int maxthread,unsigned int *d_layer);
extern "C" void cleanUI1CPU(int maxblock,int maxthread,unsigned int *d_hough,unsigned int *d_hough_layer,unsigned int *d_hough_map,houghLimits hl);
extern "C" void cleanFCPU(int maxblock,int maxthread,float *d_layer);
extern "C" void fillHoughCPU(int maxblock,int maxthread,short *d_val,unsigned int* d_layer,unsigned int* d_hough,unsigned int* d_hough_layer,unsigned int* d_hough_map,houghLimits hl);
extern "C" void localRegressionCPU(float xp,float yp,float* d,float* res);
extern "C" void computeHoughPointCPU(int maxblock,int maxthread,float *d_x, float *d_y,short* d_val,houghLimits hl);
extern "C" void ListHoughPointCPU(int maxblock,int maxthread,unsigned int* d_hough,unsigned int* d_hough_layer,unsigned int min_val,unsigned int min_layer,unsigned int* d_cand,houghLimits hl,bool endcap);
extern "C" void conformalPositionCPU(int maxblock,int maxthread,float* d_xo,float* d_yo,float* d_ro);
extern "C" void copyFromValCPU(int maxblock,int maxthread,unsigned int ith,unsigned int ir,unsigned int nbintheta,short* d_val,float* d_xi,float* d_yi,unsigned int* di_layer,float* d_ri,float* d_zi,float* d_xo,float* d_yo,unsigned int* do_layer,float* d_ro,float* d_zo,float* d_reg,bool regression,unsigned int* d_temp,bool endcap);
extern "C" void createHoughCPU(houghParam* p,uint32_t max_stub=GPU_MAX_STUB,uint32_t max_theta=GPU_MAX_THETA,uint32_t max_rho=GPU_MAX_RHO);
extern "C" void clearHoughCPU(houghParam* p);
extern "C" void initialiseHoughCPU(houghParam* p,int nstub,int ntheta,int nrho,float thetamin,float thetamax,float rmin,float rmax);
extern "C" void deleteHoughCPU(houghParam* p);
extern "C" void fillPositionHoughCPU(houghParam* p,float* h_x,float* h_y,float* h_z);
extern "C" void fillLayerHoughCPU(houghParam* p,unsigned int* h_layer);
extern "C" void copyHoughImageCPU(houghParam* p,unsigned int* h_hough);
extern "C" void copyHoughLayerCPU(houghParam* p,unsigned int* h_hough);
extern "C" void fillConformalHoughCPU(houghParam* p,float* h_x,float* h_y,float* h_z);
extern "C" void dumpCPU(houghParam* p);
extern "C" void copyPositionHoughCPU(houghParam* pi,int icand,houghParam* po,unsigned int mode,bool regression,bool endcap);
extern "C" void processHoughCPU(houghParam* p,unsigned int min_cut,unsigned int min_layer,unsigned int mode,bool endcap);
#endif
