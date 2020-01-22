#include "TMath.h"
#include "TRint.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TLorentzVector.h"
#include "TCanvas.h"
#include "TH2.h"
#include "TH2F.h"
#include "TGaxis.h"
#include <fstream>
#include <iostream>
#include "TMath.h"
#include <string>
#include <map>



void links(){
//
// To see the output of this macro, click here.

//

#include "TMath.h"

gROOT->Reset();

gROOT->SetStyle("Plain");

gStyle->SetCanvasColor(kWhite);

gStyle->SetCanvasBorderMode(0);     // turn off canvas borders
gStyle->SetPadBorderMode(0);
gStyle->SetOptStat(1111);
gStyle->SetOptTitle(1);

  // For publishing:
  gStyle->SetLineWidth(1);
  gStyle->SetTextSize(1.1);
  gStyle->SetLabelSize(0.06,"xy");
  gStyle->SetTitleSize(0.06,"xy");
  gStyle->SetTitleOffset(1.2,"x");
  gStyle->SetTitleOffset(1.0,"y");
  gStyle->SetPadTopMargin(0.1);
  gStyle->SetPadRightMargin(0.1);
  gStyle->SetPadBottomMargin(0.16);
  gStyle->SetPadLeftMargin(0.12);





 const int nsp=27;

 int splinks[nsp];
 for (int i=0;i<nsp;i++) {
   splinks[i]=0;
 }

 TH1* hist1= new TH1F("DTC output links","DTC output links",10,-0.5,9.5);
 TH1* hist2= new TH1F("SP links","SP links",60,-0.5,59.5);
 TH1* hist3= new TH1F("DTC input links","DTC input links",80,-0.5,79.5);

 
 

 ifstream in("dtcs_T5v3_27SP_TM6_phi0_piover27_phioverlap_piover81_maxstub53_nonant_tracklet_1000evts.dat");
 //ifstream in("dtcs_tilted_20SP_pizza10.dat");

 int sp;

 int ndtclinks=0;

 map<string,int> dtcoutputs;

 //in >> tmp;


   string idtc;

   in >> idtc;

   in >> sp;
   
   while (in.good()){ 
     if (dtcoutputs.find(idtc) == dtcoutputs.end())
     {
       dtcoutputs[idtc] = 0;
     }
     dtcoutputs[idtc]++;
     splinks[sp]++;
     in >> idtc >> sp;
   }
   for (map<string,int>::const_iterator it=dtcoutputs.begin(); it!=dtcoutputs.end(); ++it)
   {
      hist1->Fill(it->second);
      if (it->second == 6) { cout << it->first << endl;};
   }
   for (int i=0; i<nsp; ++i)
   {
     hist2->Fill(splinks[i]);
   }


 map<string,int> dtcinputlinks;

 ifstream in2("modules_T5v3_27SP_nonant_tracklet.dat");

 int layer, ladder, module; 
 string dtc;

 in2 >> layer>>ladder>>module>>dtc;

 while (in2.good()){

   if (dtcinputlinks.find(dtc) == dtcinputlinks.end())
   {
     dtcinputlinks[dtc] = 0;
   }
   dtcinputlinks[dtc]++;
   in2 >> layer>>ladder>>module>>dtc;

 }

 for (map<string,int>::const_iterator it=dtcinputlinks.begin(); it!=dtcinputlinks.end(); ++it)
 {
   hist3->Fill(it->second);
 }


 TCanvas* c1 = new TCanvas("DTC links","DTC links",200,10,700,800);
 c1->Divide(1,1);
 c1->SetFillColor(0);
 c1->SetGrid();

 hist1->Draw();

 c1->Print("links.pdf(",".pdf");

 TCanvas* c2 = new TCanvas("SP links","SP links",200,10,700,800);
 c2->Divide(1,1);
 c2->SetFillColor(0);
 c2->SetGrid();

 hist2->Draw();

 c2->Print("links.pdf",".pdf");

 
 TCanvas* c3 = new TCanvas("DTC input links","DTC input links",200,10,700,800);
 c3->Divide(1,1);
 c3->SetFillColor(0);
 c3->SetGrid();

 hist3->Draw();

 c3->Print("links.pdf)",".pdf");

}


