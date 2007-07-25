// File: SCurveContainer.cc
// Description:  see SCurveContainer.h
// Author: Jason Keller (University of Nebraska)
//--------------------------------------------

#include "CalibTracker/SiPixelSCurveCalibration/interface/SCurveContainer.h"

SCurveContainer::SCurveContainer() :
  ntriggers_(0), rowmax_(0), colmax_(0), detid_(0) {}

SCurveContainer::SCurveContainer(const int& vcalmax, int ntriggers, 
                                 int rowmax, int colmax, int detid) : 
  ntriggers_(ntriggers), rowmax_(rowmax), colmax_(colmax), detid_(detid)
{
  eff_.resize(rowmax);

  for(int i = 0; i != rowmax; ++i)
    eff_[i].resize(colmax);

  for(int i = 0; i != rowmax; ++i)
    for(int j = 0; j != colmax; ++j)
      eff_[i][j].resize(vcalmax + 1, 0.0);
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

