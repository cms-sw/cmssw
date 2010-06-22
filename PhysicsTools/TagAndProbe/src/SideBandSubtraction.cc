#include "PhysicsTools/TagAndProbe/interface/SideBandSubtraction.hh"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <TH1F.h>

void SideBandSubtraction::Configure(const edm::ParameterSet& SBSPSet) {
  Peak_ = SBSPSet.getUntrackedParameter< double >("SBSPeak");
  SD_   = SBSPSet.getUntrackedParameter< double >("SBSStanDev");
}


void SideBandSubtraction::Subtract( const TH1F& Total, TH1F& Result){
  // Total Means signal plus background

  const double BinWidth  = Total.GetXaxis()->GetBinWidth(1);
  const int nbins = Total.GetNbinsX();
  const double xmin = Total.GetXaxis()->GetXmin();

  const int PeakBin = (int)((Peak_ - xmin)/BinWidth + 1); // Peak
  const double SDBin = (SD_/BinWidth); // Standard deviation
  const int I = (int)((3.0*SDBin > 1.0)  ?  3.0*SDBin  : 1 ); // Interval
  const int D = (int)((10.0*SDBin > 1.0) ?  10.0*SDBin : 1 );  // Distance from peak

  const double IntegralRight = Total.Integral(PeakBin + D, PeakBin + D + I);
  const double IntegralLeft = Total.Integral(PeakBin - D - I, PeakBin - D);

  double SubValue = 0.0;
  double NewValue = 0.0;

  const double Slope     = (IntegralRight - IntegralLeft)/
    (double)((2*D + I )*(I+1));
  const double Intercept = IntegralLeft/(double)(I+1) - 
    ((double)PeakBin - (double)D - (double)I/2.0)*Slope;

  for(int bin = 1; bin < (nbins + 1); bin++){
    SubValue = Slope*bin + Intercept;
    if(SubValue < 0)
      SubValue = 0;

    NewValue = Total.GetBinContent(bin)-SubValue;
    if(NewValue > 0){
      Result.SetBinContent(bin, NewValue);
    }
  }
  Result.SetEntries(Result.Integral());
}
