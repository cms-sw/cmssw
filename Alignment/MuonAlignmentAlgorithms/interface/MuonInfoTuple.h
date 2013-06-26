#ifndef Alignment_MuonAlignmentAlgorithms_MuonInfoTuple_H
#define Alignment_MuonAlignmentAlgorithms_MuonInfoTuple_H

/*  $Date: 2010/02/25 11:33:32 $
 *  $Revision: 1.2 $
 *  \author Luca Scodellaro <Luca.Scodellaro@cern.ch>
 */

#define MAX_HIT 60
#define MAX_HIT_CHAM 14
#define MAX_SEGMENT 5


typedef struct {
  int nhits;
  float xc[MAX_HIT]; float yc[MAX_HIT]; float zc[MAX_HIT];
  float erx[MAX_HIT];
  int wh[MAX_HIT]; int st[MAX_HIT]; int sr[MAX_HIT];
  int sl[MAX_HIT]; int la[MAX_HIT];
} Info1D;
  

typedef struct{
  float p, pt, eta, phi, charge;
  int nhits[MAX_SEGMENT];
  int nseg;
  float xSl[MAX_SEGMENT]; 
  float dxdzSl[MAX_SEGMENT];
  float exSl[MAX_SEGMENT];
  float edxdzSl[MAX_SEGMENT];
  float exdxdzSl[MAX_SEGMENT];
  float ySl[MAX_SEGMENT];
  float dydzSl[MAX_SEGMENT];
  float eySl[MAX_SEGMENT];
  float edydzSl[MAX_SEGMENT];
  float eydydzSl[MAX_SEGMENT];
  float xSlSL1[MAX_SEGMENT]; 
  float dxdzSlSL1[MAX_SEGMENT];
  float exSlSL1[MAX_SEGMENT];
  float edxdzSlSL1[MAX_SEGMENT];
  float exdxdzSlSL1[MAX_SEGMENT];
  float xSL1SL3[MAX_SEGMENT];
  float xSlSL3[MAX_SEGMENT];
  float dxdzSlSL3[MAX_SEGMENT];
  float exSlSL3[MAX_SEGMENT];
  float edxdzSlSL3[MAX_SEGMENT]; 
  float exdxdzSlSL3[MAX_SEGMENT];
  float xSL3SL1[MAX_SEGMENT];
  float xc[MAX_SEGMENT][MAX_HIT_CHAM];
  float yc[MAX_SEGMENT][MAX_HIT_CHAM];
  float zc[MAX_SEGMENT][MAX_HIT_CHAM];
  float xcp[MAX_SEGMENT][MAX_HIT_CHAM];
  float ycp[MAX_SEGMENT][MAX_HIT_CHAM];
  float zcp[MAX_SEGMENT][MAX_HIT_CHAM];
  float ex[MAX_SEGMENT][MAX_HIT_CHAM];
  int wh[MAX_SEGMENT]; int st[MAX_SEGMENT]; int sr[MAX_SEGMENT];
  int sl[MAX_SEGMENT][MAX_HIT_CHAM];
  int la[MAX_SEGMENT][MAX_HIT_CHAM];
} Residual1DHit;




typedef struct {
  int wh, st, se;
  float dx, dz, alpha, beta, gamma, dy;
  float ex, ez, ealpha, ebeta, egamma, ey;
  float corr_xz, corr_xalpha, corr_xbeta, corr_xgamma, corr_xy;
  float corr_zalpha, corr_zbeta, corr_zgamma, corr_zy;
  float corr_alphabeta, corr_alphagamma, corr_alphay;
  float corr_betagamma, corr_betay;
  float corr_gammay;
} DTSegmentResult;


typedef struct {
  int wh, st, se;
  float cov[60][60];
  int sl[12], la[12];
  float dx[12], dy[12], dz[12], alpha[12], beta[12], gamma[12];
} DTHitResult;






