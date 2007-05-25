#include "FastSimulation/Utilities/interface/HistogramGenerator.h"

//#include "TAxis.h"

double HistogramGenerator::ersatzt(double x)
{
  int ibin=theXaxis->FindBin(x);
  //  std::cout << x << " Bin " << ibin << std::endl;
  double x1=myHisto->GetBinLowEdge(ibin);
  double x2=x1+myHisto->GetBinWidth(ibin);
  double y1=myHisto->GetBinContent(ibin);  
  double y2;
  if(ibin<nbins)
    y2=myHisto->GetBinContent(ibin+1);
  else
    y2=y1;
  // std::cout << " X1 " << x1 <<" X2 " << x2 << " Y1 " <<y1 << " Y2 " << y2 << std::endl;
  return y2 + (y1-y2)*(x2-x)/(x2-x1);
}
