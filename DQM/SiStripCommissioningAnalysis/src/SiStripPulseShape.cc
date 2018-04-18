#include "DQM/SiStripCommissioningAnalysis/interface/SiStripPulseShape.h"
#include <TF1.h>
#include <TMath.h>

/* New Pulse Shape Fit by G. Auzinger June 2017
following methods are used to describe a pulse shape for a 3 stage CR-CR-RC pre-amplifier (CR) 
+ shaper (CR + RC) with time constants x, y, z respectively
some special cases (x=0. y=z, x=0 && y=z) are considered (divergence of equation) and
the shaper CR y is approximated via a pol(6) of x which gives a resonable respresentation and
reduces the number of free parameters -- this shape is derived from first principle and can
be used to fit peak mode signals and deconvolution mode signals
following parameters are added in addition to the time constants x, z:
a_0 ... baseline offset amplitude
s   ... scale parameter
t_0 ... turn_on time of the pulse
par[5] ... scale parameter for the time slices in deco mode
*/


double pulse_raw(double x, double y, double z, double t)
{
  double result1, result2, result3;

  result1 = z * y * exp(-t / y);
  result1/= pow(y,2) - (x + z) * y + z * x;

  result2 = z * x * exp(-t / x);
  result2/= pow(x,2) - (x - z) * y - z * x;

  result3 = pow(z,2) * exp(-t / z);
  result3/= pow(z,2) + (x - z) * y - z * x;

  return result1 + result2 + result3;
}

double pulse_x0(double y, double z, double t)
{
  return z / (y - z) * (exp(-t / y) - exp(-t / z));
}

double pulse_yz(double x, double z, double t)
{
  double result1, result2;

  result1 = exp(-t / x) - exp(-t / z);
  result1*= z * x / (z - x);

  result2 = t * exp(-t / z);

  return (result1 + result2) / (z - x);
}

double pulse_x0_yz(double z, double t)
{
  return t / z * exp(-t / z);
}

double pulse(double x, double y, double z, double t)
{
  if(x > y)
    {
      double pivot = x;
      x = y;
      y = pivot;
    }

  if((x == 0) && (y == z)) return pulse_x0_yz(z, t);

  else if(x == 0) return pulse_x0(y, z, t);

  else if(y == z) return pulse_yz(x, z, t);

  else return pulse_raw(x, y, z, t);
}

double get_compensation(double x)
{
       return 49.9581
              - 1.7941      * x
	 - 0.110089    * pow(x,2)
	 + 0.0113809   * pow(x,3)
	 - 0.000388111 * pow(x,4)
	 + 5.9761e-6   * pow(x,5)
	 - 3.51805e-8  * pow(x,6);
}

double fpeak(double *x, double *par)
{
  double xx  = par[0];
  double z   = par[1];
  double a_0 = par[2];
  double s   = par[3];
  double t_0 = par[4];
  double t   = x[0] - t_0;

  double y = get_compensation(xx);

  if(x[0] < t_0) return a_0;
  return a_0 + s * pulse(xx, y, z, t);
}

double fdeconv(double *x, double *par)
{
  double xm = par[5]*(x[0]-25);
  double xp = par[5]*(x[0]+25);
  double xz = par[5]*x[0];
  return 1.2131 * fpeak(&xp,par) - 1.4715 * fpeak(&xz,par) + 0.4463 * fpeak(&xm,par);
}

double fpeak_convoluted(double *x, double *par)
{
  TF1 f("peak_convoluted",fpeak,0,200,4);
  return f.IntegralError(x[0]-par[4]/2.,x[0]+par[4]/2.,par,nullptr,1.)/(par[4]);
} 

double fdeconv_convoluted(double *x, double *par)
{
  double xm = (x[0]-25);
  double xp = (x[0]+25);
  double xz = x[0];
  return 1.2131*fpeak_convoluted(&xp,par)-1.4715*fpeak_convoluted(&xz,par)+0.4463*fpeak_convoluted(&xm,par);
}
