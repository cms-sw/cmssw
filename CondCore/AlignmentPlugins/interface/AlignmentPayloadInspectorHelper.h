#ifndef CONDCORE_ALIGNMENTPLUGINS_ALIGNMENTPAYLOADINSPECTORHELPER_H
#define CONDCORE_ALIGNMENTPLUGINS_ALIGNMENTPAYLOADINSPECTORHELPER_H

#include <vector>
#include <numeric>
#include <string>
#include "TH1.h"
#include "TPaveStats.h"
#include "TStyle.h"
#include "TList.h"

namespace AlignmentPI {

  // size of the phase-I APE payload (including both SS + DS modules)
  static const unsigned int phase0size=19876;
  static const float cmToUm = 10000;      

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

  enum partitions {
    BPix=1,
    FPix=2,
    TIB=3,
    TID=4,
    TOB=5,
    TEC=6
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
  std::string getStringFromPart (AlignmentPI::partitions i)
  /*--------------------------------------------------------------------*/
  {
    switch(i){
    case BPix : return "BPix";
    case FPix : return "FPix";
    case TIB  : return "TIB";
    case TID  : return "TID";
    case TOB  : return "TOB";
    case TEC  : return "TEC";
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
  void makeNicePlotStyle(TH1 *hist,int color)
  /*--------------------------------------------------------------------*/
  { 

    hist->SetStats(kFALSE);

    hist->GetXaxis()->SetTitleColor(color);
    hist->SetLineColor(color);
    hist->SetTitleSize(0.08);
    hist->SetLineWidth(2);
    hist->GetXaxis()->CenterTitle(true);
    hist->GetYaxis()->CenterTitle(true);
    hist->GetXaxis()->SetTitleFont(42); 
    hist->GetYaxis()->SetTitleFont(42);  
    hist->GetXaxis()->SetNdivisions(505);
    hist->GetXaxis()->SetTitleSize(0.06);
    hist->GetYaxis()->SetTitleSize(0.06);
    hist->GetXaxis()->SetTitleOffset(1.0);
    hist->GetYaxis()->SetTitleOffset(1.3);
    hist->GetXaxis()->SetLabelFont(42);
    hist->GetYaxis()->SetLabelFont(42);
    hist->GetYaxis()->SetLabelSize(.05);
    hist->GetXaxis()->SetLabelSize(.05);

  }
  
  /*--------------------------------------------------------------------*/
  void makeNiceStats(TH1F* hist,AlignmentPI::partitions part,int color)
  /*--------------------------------------------------------------------*/
  {
    char   buffer[255]; 
    TPaveText* stat = new TPaveText(0.60,0.75,0.95,0.97,"NDC");
    sprintf(buffer,"%s \n",AlignmentPI::getStringFromPart(part).c_str());
    stat->AddText(buffer);

    sprintf(buffer,"Entries : %i\n",(int)hist->GetEntries());
    stat->AddText(buffer);
    
    sprintf(buffer,"Mean    : %6.2f\n",hist->GetMean());
    stat->AddText(buffer);
    
    sprintf(buffer,"RMS     : %6.2f\n",hist->GetRMS());
    stat->AddText(buffer);
    
    stat->SetLineColor(color);
    stat->SetTextColor(color);
    stat->SetFillColor(10);
    stat->SetShadowColor(10);
    stat->Draw(); 
  }
}

#endif
