//how to read the fit result in root after running several toys...., to be run interactively: root toyTooFitAnalyze.C.....  
{
#include "RooFitResult.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TFile.h"
#include "RooAbsReal.h"
//#include <iostream>



using namespace RooFit;
 int Lumi = 45;
 double Y_true= 17000;
 // 17000 for L= 45,  
 double emt_true= 0.998257;
 double ems_true= 0.989785;
 double emmNotIso_true= 0.982977;
 double emmHlt_true= 0.916176;


TCanvas c1; TH1D h_Yield("h_Yield", "h_Yield", 30, -5, 5);
TH1D h_emt("h_emt", "h_emt", 30, -5, 5);
TH1D h_ems("h_ems", "h_ems", 30, -5, 5);
TH1D h_emmNotIso("h_emmNotIso", "h_emmNotIso", 30, -5, 5);
TH1D h_emmHlt("h_emmHlt", "h_emmHlt", 30, -5, 5);

TH1D h_chi2("h_chi2", "h_chi2", 30, 0, 5);

TF1 f1("f1","gaus",-3,3);
f1.SetLineColor(kRed);

for (int i =1; i <= 1000; i++){
TFile f(Form("outputToy_%d./out_%d.root",Lumi,i));
 if (f.IsZombie()) cout << "file not exist " << endl;
 if (f.IsZombie()) continue;
RooFitResult* r = gDirectory->Get("totChi2;1");
			   //r->floatParsFinal().Print("s");
			   // without s return a list,  can we get the number?
			   //RooFitResult* r = gDirectory->Get("toy_totChi2;1");
			   // chi2
std::cout << " chi2 " << r->minNll() << "distance from FCN " << r->edm()<< std::endl;
 h_chi2.Fill(r->minNll()/55.);
			   //distamce form chi2.....
			   //r->edm();
			   // yield
r->floatParsFinal()[0]->Print();
RooRealVar * y = r->floatParsFinal().find("Yield");
RooRealVar * e_mt = r->floatParsFinal().find("eff_tk");
RooRealVar * e_ms = r->floatParsFinal().find("eff_sa");
RooRealVar * e_mmNotIso = r->floatParsFinal().find("eff_iso");
RooRealVar * e_mmHlt = r->floatParsFinal().find("eff_hlt");

 // RooAbsReal * z = new RooAbsReal(y);
std::cout << " Yield value  for the chi2 number " << i << " = " << y->getVal() << std::endl;
h_Yield.Fill((y->getVal() - Y_true )/ (y->getError())  );
h_emt.Fill((e_mt->getVal() - emt_true )/ (e_mt->getError())  );
h_ems.Fill((e_ms->getVal() - ems_true )/ (e_ms->getError())  );
h_emmNotIso.Fill((e_mmNotIso->getVal() - emmNotIso_true )/ (e_mmNotIso->getError())  );
h_emmHlt.Fill((e_mmHlt->getVal() - emmHlt_true )/ (e_mmHlt->getError())  );
//delete f;		   
}
 


 gStyle->SetOptStat(1110);
  gStyle->SetOptFit(1111);
   gStyle->SetStatFontSize(0.04);
   //gStyle->SetStatFontSize(0.1);
//gStyle->SetFitFormat("5.3g");
 h_Yield.Fit("f1","","",  -3, 3);
h_Yield.Draw();
f1.Draw("same");

c1.SaveAs(Form("outputToy_%d./toy_Yield_%d.eps",Lumi,Lumi));
h_chi2.Draw();

//f1.Draw("same");

c1.SaveAs(Form("outputToy_%d./toy_chi2_%d.eps",Lumi,Lumi));
h_emt.Draw();
h_emt.Fit("f1","","",  -3, 3);
f1.Draw("same");
c1.SaveAs(Form("outputToy_%d./toy_eff_tk_%d.eps", Lumi, Lumi));
h_ems.Draw();
 h_ems.Fit("f1","","",  -3, 3);
f1.Draw("same");
c1.SaveAs(Form("outputToy_%d./toy_eff_sa_%d.eps",Lumi,Lumi));
h_emmNotIso.Draw();
h_emmNotIso.Fit("f1","","",  -3, 3);
f1.Draw("same");
c1.SaveAs(Form("outputToy_%d./toy_eff_iso_%d.eps",Lumi,Lumi));   
h_emmHlt.Draw();
 h_emmHlt.Fit("f1","","",  -3, 3);
f1.Draw("same");
c1.SaveAs(Form("outputToy_%d./toy_eff_hlt_%d.eps",Lumi,Lumi));

}

