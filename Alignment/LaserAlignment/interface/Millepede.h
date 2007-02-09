#ifndef LaserAlignment_millepede1_h
#define LaserAlignment_millepede1_h

/**********************************************************************
 * Wrapper to call MILLEPEDE (FORTRAN) routines in C++ programms      *
 *                                                                    *
 * all comments and descriptions are copied from the MILLEPEDE code
 * written by Volker Blobel:
 *                                                                    
 *                 Millepede - Linear Least Squares
 *                 ================================
 *     A Least Squares Method for Detector Alignment - Fortran code
 *
 *     TESTMP      short test program for detector aligment
 *                 with GENER (generator) + ZRAND, ZNORM (random gen.)
 *
 *     The execution of the test program needs a MAIN program:
 *                 CALL TESTMP(0)
 *                 CALL TESTMP(1)
 *                 END
 *
 *     INITGL      initialization
 *     PARGLO       optional: initialize parameters with nonzero values
 *     PARSIG       optional: define sigma for single parameter
 *     INITUN       optional: unit for iterations
 *     CONSTF       optional: constraints
 *     EQULOC      equations for local fit
 *     ZERLOC
 *     FITLOC      local parameter fit (+entry KILLOC)
 *     FITGLO      final global parameter fit
 *     ERRPAR      parameter errors
 *     CORPAR      parameter correlations
 *     PRTGLO      print result on file
 *
 *     Special matrix subprograms (all in Double Precision):
 *        SPMINV   matrix inversion + solution
 *        SPAVAT   special double matrix product
 *        SPAX     product matrix times vector
 *
 *     PXHIST      histogram printing
 *     CHFMT       formatting of real numbers
 *     CHINDL      limit of chi**2/nd
 *     FITMUT/FITMIN vector input/ouput for parallel processing
 */

extern "C"
{
  //___subroutines_____________________________________________________
  /*test of millepede
   *     IARG = 0   test with constraint
   *     IARG = 1   test without constraint */
  void testmp_(int*); 
  // generates straight line parameters
  void gener_(float y[], float *a, float *b, float x[], float sigma[], float *bias,
			  float heff[]);

  /* Initialization of package
   *     NAGB = number of global parameters
   *     DERGB(1) ... DERGB(NAGB) = derivatives w.r.t. global parameters
   *     NALC = number of local parameters (maximum)
   *     DERLC(1) ... DERLC(NALC) = derivatives w.r.t. local parameters */
  void initgl_(int *nagbar, int *nalcar, int *nstd, int *iprlim);

  /* optional: initialize parameters with nonzero values */
  void parglo_(float par[]);

  /* optional: define sigma for single parameter */
  void parsig_(int *index, float *sigma);

  /* optional: set nonlinear flag for single parameter */
  void nonlin_(int *index);

  /* optional: unit for iterations */
  void initun_(int *lun, float *cutfac);

  /* optional: constraints */
  void constf_(float dercs[], float *rhs);

  /* a single equation with its derivatives
   *     DERGB(1) ... DERGB(NAGB) = derivatives w.r.t. global parameters
   *     DERLC(1) ... DERLC(NALC) = derivatives w.r.t. local parameters
   *     RMEAS       = measured value
   *     SIGMA       = standard deviation
   *    (WGHT       = weight = 1/SIGMA**2) */
  void equloc_(float dergb[], float derlc[], float *rrmeas, 
			   float *sigma);

  /* reset derivatives
   *     DERGB(1) ... DERGB(NAGB) = derivatives w.r.t. global parameters
   *     DERLC(1) ... DERLC(NALC) = derivatives w.r.t. local parameters */
  void zerloc_(float dergb[], float derlc[]);

  /* fit after end of local block - faster(?) version */
  void fitloc_(void);

  /* final global fit */
  void fitglo_(float par[]);

  /* */
  void prtglo_(int *lun);

  /* obtain solution of a system of linear equations with symmetric
   *     matrix and the inverse.
   *
   *                    - - -
   *        CALL SPMINV(V,B,N,NRANK,...,...)      solve  V * X = B
   *                    - -   -----
   *
   *           V = symmetric N-by-N matrix in symmetric storage mode
   *               V(1) = V11, V(2) = V12, V(3) = V22, V(4) = V13, . . .
   *               replaced by inverse matrix
   *           B = N-vector, replaced by solution vector
   *
   *     DIAG(N) =  double precision scratch array
   *     FLAG(N) =  logical scratch array
   *
   *     Method of solution is by elimination selecting the  pivot  on  the
   *     diagonal each stage. The rank of the matrix is returned in  NRANK.
   *     For NRANK ne N, all remaining  rows  and  cols  of  the  resulting
   *     matrix V and the corresponding elements of  B  are  set  to  zero. */
  void spminv_(double v[], double b[], int *n, int *nrank, 
			   double diag[], bool flag[]); 

  /*  multiply symmetric N-by-N matrix from the left with general M-by-N
   *     matrix and from the right with the transposed of the same  general
   *     matrix  to  form  symmetric  M-by-M   matrix   (used   for   error
   *     propagation).
   *
   *                    - -   - -                                   T
   *        CALL SPAVAT(V,A,W,N,M)         W   =   A   *   V   *   A
   *                        -             M*M     M*N     N*N     N*M
   *
   *        where V = symmetric N-by-N matrix
   *              A = general N-by-M matrix
   *              W = symmetric M-by-M matrix */
  void spavat_(double v[], double a[], double w[],
			   int *n, int *m);

  /* multiply general M-by-N matrix A and N-vector X
   *
   *                   - -   - -
   *        CALL  SPAX(A,X,Y,M,N)          Y   :=   A   *    X
   *                       -               M       M*N       N
   *
   *        where A = general M-by-N matrix (A11 A12 ... A1N  A21 A22 ...)
   *              X = N vector
   *              Y = M vector */
  void spax_(double a[], double x[], double y[],
			 int *n, int *m);

  /* print X histogram */
  void pxhist_(int inc[], int *n, float *xa, float *xb);

  /* prepare printout of array of real numbers as character strings
   *
   *                  - -
   *        CALL CHFMT(X,N,XCHAR,ECHAR)
   *                      ----- -----
   *     where X( )     = array of n real values
   *           XCHAR( ) = array of n character*8 variables
   *           ECHAR    = character*4 variable
   *
   *     CHFMT converts an array of  n  real  values  into n character*  8
   *     variables (containing the values as text) and  a  common  exponent
   *     for printing. unneccessary zeros are suppressed.
   *
   *
   *     example: x(1)=1200.0, x(2)=1700.0 with n=2 are converted to
   *               xchar(1)='  1.2   ', xchar(2)='  1.7   ', echar='e 03' */
  void chfmt_(float x[], int *n, const char xchar[], 
			  const char echar[], int, int);

  /* get matrix information out */
  void fitmut_(int *nvec, float vec[]);

  //___functions_______________________________________________________

  /* return random number U(0,1)
   *     (simple generator, showing principle) */
  float zrand_(void);

  /* return random number U(0,1)
   *     (simple generator, showing principle) */
  float znorm_(void);

  /* return error for parameter I */
  double errpar_(int *i);

  /* return correlation between parameters I and J */
  double corpar_(int *i, int *j);

  /* return limit in chi^2/ND for N sigmas (N=1, 2 or 3) */
  float chindl_(int *n, int *nd);

  //___common blocks___________________________________________________

  // basic parameters
  const int mglobl = 1400;
  const int mlocal = 10;
  const int nstore = 10000;
  const int mcs= 10;
  // derived parameters
  const int mgl = mglobl + mcs;
  const int msymgb = (mglobl * mglobl + mglobl)/2;
  const int msym = (mgl * mgl + mgl)/2;
  const int msymlc = (mlocal * mlocal + mlocal)/2;
  const int mrecta = mglobl * mlocal;
  const int mglocs = mglobl * mcs;
  const int msymcs = (mcs * mcs + mcs)/2;

  // common block /LSQRED/ (no idea what all these variables are for)
extern struct
{
  double cgmat[msym], clmat[msymlc], clcmat[mrecta],
	diag[mgl], bgvec[mgl], blvec[mlocal],
	corrm[msymgb], corrv[mglobl],psigm[mglobl],
	pparm[mglobl],adercs[mglocs],arhs[mcs],
	dparm[mglobl],scdiag[mglobl];
  double summ;
	bool scflag[mglobl];
  int	indgb[mglobl],indlc[mlocal];
  int loctot,locrej,
	nagb,nalc,nsum,nhist,mhist[51],khist[51],lhist[51],
	nst,nfl,indst[nstore];
  float arest[nstore];
  int itert,lunit,ncs,
	nlnpa[mglobl],nstdev;
  float cfactr;
  int icnpr,icnlim,
	indnz[mglobl],indbk[mglobl];
}lsqred_; // extern struct
} 

#endif // millepede1_H
