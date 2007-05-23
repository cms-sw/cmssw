#ifndef _SprChiCdf_HH
#define _SprChiCdf_HH

struct SprChiCdf
{

static void cumchi ( double *x, double *df, double *cum, double *ccum );
static void cumgam ( double *x, double *a, double *cum, double *ccum );
static double dpmpar ( int *i );
static double erf1 ( double *x );
static double erfc1 ( int *ind, double *x );
static double exparg ( int *l );
static double fifdmax1 ( double a, double b );
static double fifdsign ( double mag, double sign );
static long fifidint ( double a );
static long fifmod ( long a, long b );
static double gam1 ( double *a );
static void gamma_inc ( double *a, double *x, double *ans, double *qans, int *ind );
static double gamma_x ( double *a );
static int ipmpar ( int *i );
static double rexp ( double *x );
static double rlog ( double *x );

};

#endif
