#ifndef _LIBHOUGHSTRUCT_H
#define _LIBHOUGHSTRUCT_H
#define GPU_MAX_STUB 512
#define GPU_MAX_THETA 256
#define GPU_MAX_RHO 256
#define GPU_MAX_RHO_WORD 16
#define GPU_MAX_STUB_BIN 16
#define GPU_MAX_CAND (GPU_MAX_THETA*GPU_MAX_RHO)
#define GPU_MAX_REG 100 
#define PI 3.14159265358979323846
#define RMIN -21.05
#define RMAX  21.05

#define HOUGH_LAYER_START_BIT 0
#define HOUGH_ZINFO_START_BIT 6
#define HOUGH_ID_START_BIT 8

#define  HOUGH_LAYER_MASK 0x3F
#define  HOUGH_ZINFO_MASK 0x3
#define  HOUGH_ID_MASK 0xFFFFFF

typedef struct {
  //Parameter
  int nstub;
  int ntheta;
  int nrho;
  float thetamin;
  float thetamax;
  float thetabin;
  float rmin;
  float rmax;
  float rbin;
  // Device position
  float* d_x;
  float* d_y;
  float* d_z;
  float* d_r;
  unsigned int* d_layer;
  // Device image
  unsigned int* d_images;
  short* d_val;
  short* h_val;
  // Device hough
  unsigned int* d_hough;
  unsigned int* d_hough_map;
  unsigned int* d_hough_layer;

  unsigned int* d_temp;
  unsigned int* h_temp;
  // device points
  unsigned int* d_cand;
  // Host points
  unsigned int* h_cand;
  
  // Max value
  unsigned int max_val;

  // Rgression
  float* d_reg;
  float* h_reg;


  // limits
  unsigned int max_stub,max_theta,max_rho;
} houghParam;


#define GET_R_VALUE(p,ir) (p.rmin+(ir+0.5)*p.rbin)
#define GET_THETA_VALUE(p,ith) (p.thetamin+(ith+0.5)*p.thetabin)

typedef struct {
  unsigned int ntheta,nrho;
  float thetamin,thetamax;
  float rmin,rmax;
  unsigned int mode;
} houghLimits;


#endif
