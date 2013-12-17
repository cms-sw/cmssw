/** \file
 *
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 */

/* This Class Header */
#include "RecoLocalMuon/DTSegment/src/DTLinearFit.h"

/* Collaborating Class Header */

/* C++ Headers */
#include <cmath>
using namespace std;

/* ====================================================================== */

/// Constructor
DTLinearFit::DTLinearFit() {

}

/// Destructor
DTLinearFit::~DTLinearFit() {
}

/* Operations */ 
void DTLinearFit::fit(const vector<float> & x,
                      const vector<float> & y, 
                      int nptfit,
                      const vector<float> & sigy, 
                      float& slope,
                      float& intercept,
                      double& chi2, 
                      float& covss,
                      float& covii,
                      float& covsi) const
{
  chi2=0;
  float sy = 0, sx = 0;
  float s11 = 0, sxy = 0, sxx = 0;
  for (int i = 0; i != nptfit; i++) {
    float sy2 = sigy[i]*sigy[i];
    sy  += y[i] / sy2;
    sxy += x[i]*y[i] / sy2;
    s11 += 1. / sy2;
    sx  += x[i] / sy2;
    sxx += x[i]*x[i] / sy2;
  }

  float delta = s11*sxx - sx*sx;

  if (delta==0) return;

  intercept = (sy*sxx - sxy*sx) / delta;
  slope = (sxy*s11 - sy*sx) / delta;

  covii =  sxx / delta;
  covss =  s11 / delta;
  covsi = -sx / delta;

  for (int j=0; j<nptfit; j++){
    const double ypred = intercept + slope*x[j];
    const double dy = (y[j] - ypred)/sigy[j];
    chi2 += dy*dy;
  }

}


void DTLinearFit::fitNpar( const int npar,
                           const vector<float>& xfit,
                           const vector<float>& yfit,
                           const vector<int>& lfit,
                           const vector<double>& tfit,
                           const vector<float> & sigy, 
                           float& aminf,
                           float& bminf,
                           float& cminf,
                           float& vminf,
                           double& chi2fit,
                           const bool debug=0) const { 

  int nptfit=xfit.size();
  if (nptfit<npar) return;

  double sx = 0.;
  double sxx = 0.;
  double sy = 0.;
  double sxy = 0.;
  double sl = 0.;
  double sl2 = 0.;
  double sly = 0.;
  double slx = 0.;
  double st = 0.;
  double st2 = 0.;
  double slt = 0.;
  double sltx = 0.;
  double slty = 0.;

  for (int j=0; j<nptfit; j++){
    sx += xfit[j];       
    sy += yfit[j];
    sxx+= xfit[j]*xfit[j];
    sxy+= xfit[j]*yfit[j];
    sl += lfit[j];       
    sl2+= lfit[j]*lfit[j];
    sly+= lfit[j]*yfit[j];
    slx+= lfit[j]*xfit[j];
    st += tfit[j];
    st2+= tfit[j]*tfit[j];
    slt+= lfit[j]*tfit[j];
    sltx+= lfit[j]*tfit[j]*xfit[j];
    slty+= lfit[j]*tfit[j]*yfit[j];
  } //end loop

  const double d1 = sy;
  const double d2 = sxy;
  const double d3 = sly;
  const double c1 = sl;
  const double c2 = slx;
  const double c3 = sl2;
  const double b1 = sx;
  const double b2 = sxx;
  const double b3 = slx;
  const double a1 = nptfit;
  const double a2 = sx;
  const double a3 = sl;

  const double b4 = b2*a1-b1*a2;
  const double c4 = c2*a1-c1*a2;
  const double d4 = d2*a1-d1*a2;
  const double b5 = a1*b3-a3*b1;
  const double c5 = a1*c3-a3*c1;
  const double d5 = a1*d3-d1*a3;
  const double a6 = slt;
  const double b6 = sltx;
  const double c6 = st;
  const double v6 = st2;	
  const double d6 = slty;

  chi2fit = -1.;

  if (npar==3) {
  
    if (((c5*b4-c4*b5)*b4*a1)!=0) {
      cminf = (d5*b4-d4*b5)/(c5*b4-c4*b5);
      if (fabs(cminf)<0.000001) cminf=0;
      aminf = d4/b4 -cminf *c4/b4;
      bminf = (d1/a1 -cminf*c1/a1 -aminf*b1/a1);

      chi2fit = 0.;
      for (int j=0; j<nptfit; j++){
        const double ypred = aminf*xfit[j] + bminf;
        const double ycorr = yfit[j]-cminf*lfit[j];
        const double dy = (ycorr - ypred)/sigy[j];
        chi2fit += dy*dy;
        
        if (tfit[j]==0.) continue;
        double xwire = yfit[j]-tfit[j];
        
        // define a small safety margin
        double margin = 0.1; 
//          cout << " pred: " << ypred << "  hit: " << ycorr << "chi2: " << dy*dy << " t0: " << -cminf/0.00543 << endl;
        
        // Sanity checks - check that the hit after the fit is still withing the DT cell volume
        if ((fabs(yfit[j]-xwire)>margin) && (fabs(ycorr-xwire)>margin) && ((yfit[j]-xwire)*(ycorr-xwire)<0)) {
//          cout << " segmentpred: " << ypred << "   hit: " << ycorr << "   xwire: " << xwire << "   yhit: " << yfit[j] << "   t0: " << -cminf/0.00543 << endl;
//          cout << " XXX hit moved across wire!!!" << endl;
          chi2fit=-1.;
          return;
        }  

        if (fabs(ypred-xwire)>2.1 + margin) {
//          cout << " segmentpred: " << ypred << "   hit: " << ycorr << "   xwire: " << xwire << "   yhit: " << yfit[j] << "   t0: " << -cminf/0.00543 << endl;
//          cout << " XXX segment outside chamber!!!" << endl;
          chi2fit=-1.;
          return;
        }  

        if (fabs(ycorr-xwire)>2.1 + margin) {
//          cout << " segmentpred: " << ypred << "   hit: " << ycorr << "   xwire: " << xwire << "   yhit: " << yfit[j] << "   t0: " << -cminf/0.00543 << endl;
//          cout << " XXX hit outside chamber!!!" << endl;
          chi2fit=-1.;
          return;
        }  
        if (fabs(chi2fit<0.0001)) chi2fit=0;
      } //end loop chi2
    }
  } else if (npar==4) {
    const double det = (a1*a1*(b2*v6 - b6*b6) - a1*(a2*a2*v6 - 2*a2*a6*b6 + a6*a6*b2 + b2*c6*c6 + b3*(b3*v6 - 2*b6*c6))
			  + a2*a2*c6*c6 + 2*a2*(a3*(b3*v6 - b6*c6) - a6*b3*c6) + a3*a3*(b6*b6 - b2*v6)
			  + a6*(2*a3*(b2*c6 - b3*b6) + a6*b3*b3)); 

    if (det != 0) { 
        // computation of a, b, c and v
        bminf = (a1*(a2*(b6*d6 - v6*d2) + a6*(b6*d2 - b2*d6) + d1*(b2*v6 - b6*b6)) - a2*(b3*(c6*d6 - v6*d3)
                 + c6*(b6*d3 - c6*d2)) + a3*(b2*(c6*d6 - v6*d3) + b3*(v6*d2 - b6*d6) + b6*(b6*d3 - c6*d2))
                 + a6*(b2*c6*d3 + b3*(b3*d6 - b6*d3 - c6*d2)) - d1*(b2*c6*c6 + b3*(b3*v6 - 2*b6*c6)))/det;
        aminf = - (a1*a1*(b6*d6 - v6*d2) - a1*(a2*(a6*d6 - v6*d1) - a6*a6*d2 + a6*b6*d1 + b3*(c6*d6 - v6*d3)
                 + c6*(b6*d3 - c6*d2)) + a2*(a3*(c6*d6 - v6*d3) + c6*(a6*d3 - c6*d1)) + a3*a3*(v6*d2 - b6*d6)
                 + a3*(a6*(b3*d6 + b6*d3 - 2*c6*d2) - d1*(b3*v6 - b6*c6)) - a6*b3*(a6*d3 - c6*d1))/det;
        cminf = -(a1*(b2*(c6*d6 - v6*d3) + b3*(v6*d2 - b6*d6) + b6*(b6*d3 - c6*d2)) + a2*a2*(v6*d3 - c6*d6)
                 + a2*(a3*(b6*d6 - v6*d2) + a6*(b3*d6 - 2*b6*d3 + c6*d2) - d1*(b3*v6 - b6*c6))
                 + a3*(d1*(b2*v6 - b6*b6) - a6*(b2*d6 - b6*d2)) + a6*(a6*(b2*d3 - b3*d2) - d1*(b2*c6 - b3*b6)))/det;
        vminf = - (a1*a1*(b2*d6 - b6*d2) - a1*(a2*a2*d6 - a2*(a6*d2 + b6*d1) + a6*b2*d1 + b2*c6*d3
                 + b3*(b3*d6 - b6*d3 - c6*d2)) + a2*a2*c6*d3 + a2*(a3*(2*b3*d6 - b6*d3 - c6*d2) - b3*(a6*d3 + c6*d1))
                 + a3*a3*(b6*d2 - b2*d6) + a3*(a6*(b2*d3 - b3*d2) + d1*(b2*c6 - b3*b6)) + a6*b3*b3*d1)/det;

        chi2fit = 0.;
        for (int j=0; j<nptfit; j++) {
          const double ypred = aminf*xfit[j] + bminf;
          const double dy = (yfit[j]+vminf*lfit[j]*tfit[j]-cminf*lfit[j] -ypred)/sigy[j];
          chi2fit += dy*dy;
        } //end loop chi2
      }
  }       
}


void DTLinearFit::fit3par( const vector<float>& xfit,
                           const vector<float>& yfit,
                           const vector<int>& lfit,
                           const int nptfit,
                           const vector<float> & sigy, 
                           float& aminf,
                           float& bminf,
                           float& cminf,
                           double& chi2fit,
                           const bool debug=0) const { 

  float vminf;
  vector <double> tfit( nptfit, 0.);
  
  fitNpar(3,xfit,yfit,lfit,tfit,sigy,aminf,bminf,cminf,vminf,chi2fit,debug);
}


void DTLinearFit::fit4Var( const vector<float>& xfit,
                           const vector<float>& yfit,
                           const vector<int>& lfit,
                           const vector<double>& tfit,
                           const int nptfit,
                           float& aminf,
                           float& bminf,
                           float& cminf,
                           float& vminf,
                           double& chi2fit,
                           const bool vdrift_4parfit=0,
                           const bool debug=0) const { 

  if (debug) cout << "Entering Fit4Var" << endl;

  const double sigma = 0.0295;// FIXME errors can be inserted .just load them/that is the usual TB resolution value for DT chambers 
  vector<float> sigy;

  aminf = 0.;
  bminf = 0.;
  cminf = -999.;
  vminf = 0.;

  int nppar = 0;
  double chi2fitN2 = -1. ;
  double chi2fit3 = -1.;
  double chi2fitN3 = -1. ;
  double chi2fitN4 = -1.;
  float bminf3 = bminf;
  float aminf3 = aminf;
  float cminf3 = cminf;
  int nppar2 = 0;
  int nppar3 = 0;
  int nppar4 = 0;

  cminf = -999.;
  vminf = 0.;

  for (int j=0; j<nptfit; j++)
    sigy.push_back(sigma);

  float a = 0.;
  float b = 0.;
  float covss, covii, covsi;

  fit(xfit,yfit,nptfit,sigy,b,a,chi2fit,covss,covii,covsi);

  bminf = b;
  aminf = a;
  nppar = 2; 
  nppar2 = nppar; 
  chi2fitN2 = chi2fit/(nptfit-2);

  // cout << "dt0 = 0  chi2fit = " << chi2fit << "  slope = "<<b<<endl;

  if (nptfit >= 3) {

    fitNpar(3,xfit,yfit,lfit,tfit,sigy,aminf,bminf,cminf,vminf,chi2fit,debug);
    chi2fit3 = chi2fit;
  
    if (cminf!=-999.) {
      nppar = 3;
      if (nptfit>3)
        chi2fitN3 = chi2fit /(nptfit-3);
    } else {
      bminf = b;
      aminf = a;
      chi2fitN3 = chi2fit /(nptfit-2);
    }

    bminf3 = bminf;
    aminf3 = aminf;
    cminf3 = cminf;
    nppar3 = nppar;

    if (debug) {
      cout << "dt0= 0 : slope 2 = " << b << " pos in  = " << a << " chi2fitN2 = " << chi2fitN2
	   << " nppar = " << nppar2 << " nptfit = " << nptfit << endl;
      cout << "dt0 = 0 : slope 3 = " << bminf << " pos out = " << aminf << " chi2fitN3 = "
	   << chi2fitN3 << " nppar = " << nppar3 << " T0_ev ns = " << cminf/0.00543 << endl;
      cout << " vdrift_4parfit "<< vdrift_4parfit<<endl;
    } 


    if (nptfit>=5) {
      fitNpar(4,xfit,yfit,lfit,tfit,sigy,aminf,bminf,cminf,vminf,chi2fit,debug);
       
      if (vminf!=0) { 
        nppar = 4;
        if (nptfit<=nppar){ 
          chi2fitN4=-1;
        } else{
          chi2fitN4= chi2fit / (nptfit-nppar); 
        }
      } else {
        if (nptfit <= nppar) chi2fitN4=-1;
          else chi2fitN4 = chi2fit / (nptfit-nppar); 
      }

      if (fabs(vminf) >= 0.29) {
        // for safety and for code construction..dont accept correction on dv/vdrift greater then 0.09
        vminf = 0.;
        cminf = cminf3;
        aminf = aminf3;
        bminf = bminf3;
        nppar = 3;
        chi2fit = chi2fit3;
      }

    }

    if (!vdrift_4parfit){         //if not required explicitly leave the t0 and track step as at step 3
                                  // just update vdrift value vmin for storing in the segments for monitoring
       cminf = cminf3;
       aminf = aminf3;
       bminf = bminf3;
       nppar = 3;
       chi2fit = chi2fit3;
    }

    nppar4 = nppar;

  }  //end nptfit >=3

  if (debug) {
    cout << "   dt0= 0 : slope 4  = " << bminf << " pos out = " << aminf <<" chi2fitN4 = " << chi2fitN4
	 << "  nppar= " << nppar4 << " T0_ev ns= " << cminf/0.00543 <<" delta v = " << vminf <<endl;
    cout << nptfit << " nptfit " << " end  chi2fit = " << chi2fit/ (nptfit-nppar ) << " T0_ev ns= " << cminf/0.00543 << " delta v = " << vminf <<endl;
  }

  if ( fabs(vminf) >= 0.09 && debug ) {  //checks only vdrift less then 10 % accepted
//    cout << "vminf gt 0.09 det=  " << endl;
//    cout << "dt0= 0 : slope 4 = "<< bminf << " pos out = " << aminf << " chi2fitN4 = " << chi2fitN4
//	 << " T0_ev ns = " << cminf/0.00543 << " delta v = "<< vminf << endl;
//    cout << "dt0 = 0 : slope 2 = "<< b << " pos in = " << a <<" chi2fitN2 = " << chi2fitN2
//	 << " nppar = " << nppar-1 << " nptfit = " << nptfit <<endl;
//    cout << "dt0 = 0 : slope 3 = " << bminf << " pos out = " << aminf << " chi2fitN3 = "
//	 << chi2fitN3 << " T0_ev ns = " << cminf/0.00543 << endl;
    cout << nptfit   <<" nptfit "<< "   end  chi2fit = " << chi2fit << "T0_ev ns= " << cminf/0.00543 << "delta v = "<< vminf <<endl;        
  }

  if (nptfit != nppar) chi2fit = chi2fit / (nptfit-nppar);
}
