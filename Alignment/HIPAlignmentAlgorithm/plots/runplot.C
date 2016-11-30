#include <iostream>
#include <vector>
#include "TROOT.h"
#include "TString.h"
#include "plotter.C"

using namespace std;

void runplot(TString obj, int iov, int iter){
  //***********DEFINED BY USER******************

  TString path = "../";

  // choose plot type from : "cov","shift","chi2","param","hitmap"
  vector<TString> plottype;
  //plottype.push_back("cov");
  //plottype.push_back("shift");
  plottype.push_back("chi2");
  //plottype.push_back("param");
  //plottype.push_back("hitmap");

  // plot all detectors together or individually
  bool MergeAllDet = 1;

  //*******************************************

  int Nplots = plottype.size();

  //plotting all detectors together

  if (MergeAllDet == 1){
    for (int i = 0; i < Nplots; i++) plotter(path.Data(), obj.Data(), iov, plottype.at(i).Data(), iter);
  }

  // plotting each detector separately, don't use this for hit-map.
  else{
    // det: 0=all,1=PXB, 2=PXF, 3=TIB, 4=TID, 5=TOB, 6=TEC
    for (int i = 0; i < Nplots; i++) {
      for (int det = 0; det < 6; det++) plotter(path.Data(), obj.Data(), iov, plottype.at(i).Data(), iter, det);
    }
  }
}

