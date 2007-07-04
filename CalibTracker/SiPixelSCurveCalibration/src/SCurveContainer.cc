//This class is used to contain information used
//in creating SCurve Calibration Plots.

#include "CalibTracker/SiPixelSCurveCalibration/interface/SCurveContainer.h"

#include <iostream>

SCurveContainer::SCurveContainer() :
  vcalmin_(0), vcalmax_(0), vcalstep_(0), ntriggers_(0),
  rowmax_(0), colmax_(0), detid_(0)
{}

SCurveContainer::SCurveContainer(int vcalmin, int vcalmax, int vcalstep,
                      int ntriggers, int rowmax, int colmax,int detid) :
  vcalmin_(vcalmin), vcalmax_(vcalmax), vcalstep_(vcalstep), 
  ntriggers_(ntriggers), rowmax_(rowmax), colmax_(colmax), detid_(detid)
{
  eff_.resize(rowmax_);

  for(int i = 0; i != rowmax_; ++i)
    eff_[i].resize(colmax_);

  for(int i = 0; i != rowmax_; ++i)
    for(int j = 0; j != colmax_; ++j)
      eff_[i][j].resize(vcalmax_+1, 0.0);
}

SCurveContainer::~SCurveContainer()
{

}

double SCurveContainer::getEff(const int& vcal, const int& row, const int& col) const
{
  return eff_[row][col][vcal];
}

void SCurveContainer::setEff(const int& adc, const int& vcal, const int& row, const int& col)
{
  if(adc > 0) eff_[row][col][vcal] += 1.0/ntriggers_;
}

