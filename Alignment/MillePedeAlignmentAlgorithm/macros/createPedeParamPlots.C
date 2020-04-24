// Plot distributions of module-level parameters
//
// Execute as (in an interactive ROOT session):
//
//  > .L createPedeParamPlots.C
//  > compile()
//  > createPedeParamPlots("<path-to-treeFile_merge.root","<label>")
//
//
//
// NB: Currently, the treeFile-writing assumes the 5th  column in millepede.res
//     to be the parameter uncertainties (as it is the case when running pede in
//     inversion mode) and puts those values in the tree. If pede is not run in
//     inversion mode, the 5th column might be sth else and thus the plotted errors
//     do not make sense. --> Ignore all plots other than <name>_0_<det>.pdf in that
//     case!

#include "TString.h"
#include "TROOT.h"

#include "GFUtils/GFHistManager.h"
#include "PlotMillePede.h"


void compile() {
  gROOT->ProcessLine(".L allMillePede.C");
  gROOT->ProcessLine("allMillePede()");
  gROOT->ProcessLine(".L setGStyle.C");
  gROOT->ProcessLine("setGStyle()");
}

void createPedeParamPlots(const TString& treeFile1, const TString& title1, const TString& treeFile2="", const TString& title2="") {

  const int subDetIds[6]       = { 1,      2,      3,     4,     5,     6     };
  const TString subDetNames[6] = { "BPIX", "FPIX", "TIB", "TID", "TOB", "TEC" };
  const TString coords[6]      = { "x",    "xz",   "z",   "z",   "z",   "z"   }; // printed in legend

  const bool has2 = treeFile2.Length()>0;

  TString outNamePrefix1(title1);
  outNamePrefix1.ReplaceAll(" ","_");
  TString outNamePrefix2(title2);
  outNamePrefix2.ReplaceAll(" ","_");
  
   PlotMillePede* p1 = new PlotMillePede(treeFile1);
   if( !has2 ) p1->SetTitle(title1);
   p1->SetBowsParameters(true);
   p1->SetHieraLevel(0);		// lowest level

   PlotMillePede* p2 = NULL;
   if( has2 ) {
     p2 = new PlotMillePede(treeFile2);
     p2->SetTitle(title2);
     p2->SetBowsParameters(true);
     p2->SetHieraLevel(0);		// lowest level
   }

   for(int i = 0; i < 6; ++i) {
     p1->SetSubDetId(subDetIds[i]);
     p1->SetOutName(outNamePrefix1+"_PedeParam_"+subDetNames[i]+"_");
     p1->DrawPedeParam();

     if( has2 ) {
       p2->SetSubDetId(subDetIds[i]);
       p2->SetOutName(outNamePrefix2+"_PedeParam_"+subDetNames[i]+"_");
       p2->DrawPedeParam();
     }

     if( has2 ) {
       p1->SetOutName(outNamePrefix1+"-"+outNamePrefix2+"_PedeParam_"+subDetNames[i]+"_");
       p1->GetHistManager()->Overlay(p1->GetHistManager(),0,0,title1);
       p1->GetHistManager()->Overlay(p2->GetHistManager(),0,0,p2->GetTitle());
       p1->GetHistManager()->Draw();
       p1->GetHistManager()->SameWithStats(true);
       p1->GetHistManager()->Draw();
     }
   }

   // do the first plots again to have the correct title
   if( has2 ) {
     p1->SetTitle(title1);
     for(int i = 0; i < 6; ++i) {
       p1->SetSubDetId(subDetIds[i]);
       p1->SetOutName(outNamePrefix1+"_PedeParam_"+subDetNames[i]+"_");
       p1->DrawPedeParam();
     }
   }
}
