#include "TGraph.h"
#include "TCanvas.h"
#include "TMath.h"

//J.Bendavid
//Script to optimize the number of hadronization steps for a given matching*filter efficiency
//Optimization is exact under the assumption that every input event has the same underlying matching*filter efficiency
//In practice this is not true for jet matching when mixing multiplicities, so the optimization becomes approximate.
//This assumption is also not strictly true in case gen filters include pt/acceptance cuts which correlate the filter
//efficiency with the parton kinematics from the hard event.

void calcnattempts() {
  
  const double eff=0.273*0.00122; // (54924/201000)*(67/54924)
  const unsigned int ntmax=1000;
  
  const double tlhe = 2.8;  //cpu time to generate an lhe event
  const double tps = 19.5e-3;  //cpu time to shower/hadronize an event
  const double ts = 70.6;   //cpu time to simulate an event
  
  double tmin = (1./eff)*(tlhe+tps) + ts;
  int ntmin = 1;
  double ninmin = 0.;
  double noutmin = 0.;
  
  printf("initial tmin = %5f\n",tmin);
  
  TGraph *htime = new TGraph;
  for (int nt=1; nt<=ntmax; ++nt) {
    double pk0 = pow(1.-eff,nt);
    //double pk0 = TMath::Binomial(nt,0)*pow(eff,0)*pow(1.-eff,nt-0);
    double r1 = 0.;
    double r2 = 0.;
    bool trippednan = false;
    for (int k=0; k<=nt; ++k) {
      double pk = 0;
      if (!trippednan) {
	pk = TMath::Binomial(nt,k)*pow(eff,k)*pow(1.-eff,nt-k);
	if (std::isnan(pk)) trippednan = true;
	//assert(!trippednan);
      }
      if (trippednan) {
	//printf("binomialcoeff = %5f\n", TMath::Binomial(nt,k));
	double lambda = (double)nt*eff;
	pk = TMath::PoissonI(k,lambda);
      }
      r1 += pk*k*k;
      r2 += pk*k;
    }
    double r = r1/(r2*r2);
    double nin = r;
    double nout = r*(1.-pk0);
    double tcpu = r*(tlhe + nt*tps + (1.-pk0)*ts);
    //double tcpu = r*(tlhe + nt*tps + ts);
    if (tcpu<tmin) {
      tmin = tcpu;
      ntmin = nt;
      ninmin = nin;
      noutmin = nout;
    }
    printf("pk0 = %5f, r = %5f, r1 = %5f, r2 = %5f\n",pk0, r,r1,r2);
    printf("nt = %i, NE = 1, Nin = %5f, Nout = %5f, tcpu = %5f\n",nt,nin,nout,tcpu);
    htime->SetPoint(nt-1,double(nt),tcpu);
  }
  htime->Draw();
  
  printf("Optimal nAttempts = %i, cpu time per unweighted event equivalent = %5f\n",ntmin, tmin);
  printf("NE = 1, Nin = %5f, Nout = %5f\n",ninmin,noutmin);
  printf("Nout/Nin = %5f\n",noutmin/ninmin);
  printf("NE/Nin (equivalent unweighted events per input event) = %5f\n",1./ninmin);
  
}
