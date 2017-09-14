#ifndef CONDCORE_ALIGNMENTPLUGINS_ALIGNMENTPAYLOADINSPECTORHELPER_H
#define CONDCORE_ALIGNMENTPLUGINS_ALIGNMENTPAYLOADINSPECTORHELPER_H

#include <vector>
#include <numeric>
#include <string>
#include "TH1.h"
#include "TPaveText.h"

namespace AlignmentPI {

  // M.M. 2017/09/12
  // As the matrix is symmetric, we map only 6/9 terms
  // More terms for the extended APE can be added to the following methods

  enum index {
    XX=1,
    XY=2,
    XZ=3,
    YZ=4,
    YY=5,
    ZZ=6
  };

  /*--------------------------------------------------------------------*/
  std::string getStringFromIndex (AlignmentPI::index i)
  /*--------------------------------------------------------------------*/
  {
    switch(i){
    case XX : return "XX";
    case XY : return "XY";
    case XZ : return "XZ";
    case YZ : return "YX";
    case YY : return "YY";
    case ZZ : return "ZZ";
    default : return "should never be here!";
    }
  }

  /*--------------------------------------------------------------------*/
  std::pair<int,int> getIndices(AlignmentPI::index i)
  /*--------------------------------------------------------------------*/    
  {
    switch(i){
    case XX : return std::make_pair(0,0);
    case XY : return std::make_pair(0,1);
    case XZ : return std::make_pair(0,2);
    case YZ : return std::make_pair(1,0);
    case YY : return std::make_pair(1,1);
    case ZZ : return std::make_pair(2,2);
    default : return std::make_pair(-1,-1);
    }
  }
  
  /*--------------------------------------------------------------------*/
  void makeNicePlotStyle(TH1 *hist)
  /*--------------------------------------------------------------------*/
  { 
    hist->SetStats(kFALSE);  
    hist->SetLineWidth(2);
    hist->GetXaxis()->CenterTitle(true);
    hist->GetYaxis()->CenterTitle(true);
    hist->GetXaxis()->SetTitleFont(42); 
    hist->GetYaxis()->SetTitleFont(42);  
    hist->GetXaxis()->SetTitleSize(0.05);
    hist->GetYaxis()->SetTitleSize(0.05);
    hist->GetXaxis()->SetTitleOffset(1.2);
    hist->GetYaxis()->SetTitleOffset(1.3);
    hist->GetXaxis()->SetLabelFont(42);
    hist->GetYaxis()->SetLabelFont(42);
    hist->GetYaxis()->SetLabelSize(.05);
    hist->GetXaxis()->SetLabelSize(.05);
  }
}

#endif
