#include <OnlineDB/CSCCondDB/interface/CSCThrTurnOnFcn.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <cmath>
#include <iostream>
#include <vector>
#include "TMath.h"

double CSCThrTurnOnFcn::operator()
(const std::vector<double>& par) const {
  
  double x,y,er,fn;
  double N=norm;
  double chi2 = 0.;

  int size=xdata.size();
  for(int i = 0; i < size; ++i) {
    x=xdata[i]; 
    y=ydata[i];	
    er=ery[i];

    fn=(x-par[0])/(par[1]*1.4142);
    fn=N * (1.- TMath::Erf(fn))/2.;

    double diff = y-fn;
    chi2 += diff*diff / (er*er);

    //    std::cout<<"CSC AFEB threshold fit "<<i+1<<" "<<x<<" "<<y<<" "
    //                   <<er<<" "<<fn<<" "<<chi2<<" "
    //                   <<par[0]<<" "<<par[1]<<"\n";

    LogDebug("CSC")<<" AFEB threshold fit "<<i+1<<" "<<x<<" "<<y<<" "
                   <<er<<" "<<fn<<" "<<chi2<<" "
                   <<par[0]<<" "<<par[1]<<"\n";
  }
  //  std::cout<<"Chi2 "<<chi2<<std::endl;
  return chi2;
}
