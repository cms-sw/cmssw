// Plot IOV-dependent parameters
//
//
// Execute as (in an interactive ROOT session):
//
//  > .L createPedeParamIOVPlots.C
//  > compile()
//  > .L createPedeParamIOVPlots.C


#include "TString.h"
#include "TROOT.h"

#include "GFUtils/GFHistManager.h"
#include "PlotMillePedeIOV.h"


void compile() {
  gROOT->ProcessLine(".L allMillePede.C");
  gROOT->ProcessLine("allMillePede()");
  gROOT->ProcessLine(".L setGStyle.C");
  gROOT->ProcessLine("setGStyle()");
}

void createPedeParamIOVPlots(const TString& treeFile1, const TString& title1, const TString& treeFile2="", const TString& title2="") {
  const int subDetIds[6]       = { 1,      2,      3,     4,     5,     6     };
  const TString subDetNames[6] = { "BPIX", "FPIX", "TIB", "TID", "TOB", "TEC" };
  const TString coords[6]      = { "x",    "xz",   "z",   "z",   "z",   "z"   }; // printed in legend

  const bool has2 = treeFile2.Length()>0;

  TString outNamePrefix1(title1);
  outNamePrefix1.ReplaceAll(" ","_");
  TString outNamePrefix2(title2);
  outNamePrefix2.ReplaceAll(" ","_");

  PlotMillePedeIOV iov(treeFile1,-1,1); // large-scale hierarchy level
  iov.SetTitle(title1);
  for(int i = 0; i < 6; ++i) {
    iov.SetSubDetId(subDetIds[i]);
    iov.GetHistManager()->SetCanvasName(outNamePrefix1+"_PedeParamIOV_"+subDetNames[i]+"_");
    iov.DrawPedeParam(coords[i]);
  }
  if( has2 ) {
    PlotMillePedeIOV iov2(treeFile2,-1,1); // large-scale hierarchy level
    iov2.SetTitle(title2);
    for(int i = 0; i < 6; ++i) {
      iov2.SetSubDetId(subDetIds[i]);
      iov2.GetHistManager()->SetCanvasName(outNamePrefix2+"_PedeParamIOV_"+subDetNames[i]+"_");
      iov2.DrawPedeParam(coords[i]);
    }
  }
}
