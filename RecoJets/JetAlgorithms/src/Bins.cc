#include "RecoJets/JetAlgorithms/interface/Bins.h"
#include <iostream>

using namespace std;





int Bins::getBins(double  *bins,int nBins,double MinBin,double MaxBin,bool log)
{
double incr;
if(log)
	{
	incr=TMath::Power(MaxBin/MinBin,1.0/double(nBins));
	bins[0]=MinBin;
	bins[nBins]=MaxBin;
	for(int i=1;i<nBins;i++)
		bins[i]=bins[i-1]*incr;
	}
else
	{
	incr=(MaxBin-MinBin)/nBins;
	bins[0]=MinBin;
	bins[nBins+1]=MaxBin;
	for(int i=1; i<nBins+1;i++)
		bins[i]=bins[i-1]+incr;
	}
return 0;
}


int Bins::getBin(int nBins,double  bins[],double value,double *x0,double *x1)
{
int R=0;
//int nBins=sizeof(Bins)/sizeof(double);//?
if(value <bins[0])return -1;
if(value >bins[nBins]) {
  *x0 = bins[nBins-1];
  *x1 = bins[nBins];
  return nBins-1;
}

for(R=0;R<nBins;R++)
	{
	if(bins[R]>value)break;	
	}
R--;
if(x0) *x0=bins[R];
if(x1) *x1=bins[R+1];
return R;	
}

void Bins::getBins_int( int nBins_total, Double_t* Lower, Double_t xmin, Double_t xmax, bool plotLog) {

  Double_t Lower_exact;
  int nBins = nBins_total-1;
  const double dx = (plotLog) ? pow((xmax / xmin), (1. / (double)nBins)) : ((xmax - xmin) / (double)nBins);
  Lower[0] = xmin;
  Lower_exact = Lower[0];
  for (int i = 1; i != nBins; ++i) {

    if (plotLog) {
      Lower_exact *= dx;
      Lower[i] = TMath::Ceil(Lower_exact);
    } else {
      Lower[i] = TMath::Ceil(Lower[i-1] + dx);
    }

  }

  Lower[nBins] = xmax;

}
