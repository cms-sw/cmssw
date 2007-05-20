#include <iostream>
#include <stdio.h>
#include <string.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include "TFile.h"
#include "TProfile.h"
#include "TF1.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TKey.h"
#include "TObject.h"
#include "TDirectory.h"
#include "TMath.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TROOT.h"

stringstream ss;
TFile *f;
std::map<std::string, TObjArray*> vObjectsList;
TH1F * A;
 
//* Fit function for LocalAngleVsPhi
Double_t fitf(Double_t *x, Double_t *par)
{
  Double_t fitval = (par[0]*tan(x[0]/180*3.1416) - 1 )/(par[0]+tan(x[0]/180*3.1416));
  return atan(fitval)/3.1416*180;
}

void doFit(TProfile *hpx, double& zero, double& C , double& Cerr)
{
  if(hpx==NULL){
    zero=-1000;
    C=-9999;
    return;
  }
  

  // Creates a Root function based on function fitf above
  TF1 *func = new TF1("fitf",fitf,5,120,1);

  // Sets initial values and parameter names
  func->SetParameter(0,1);
  func->SetParNames("C");
  func->SetParLimits(0,-100,100);

  // Fit histogram in range defined by function
  hpx->Fit("fitf","rm0");
  
  C=func->GetParameter(0);
  Cerr=func->GetParError(0);
  double ChiSqr=func->GetChisquare();
  zero=atan(1/C)/3.1416*180;
  if (zero<0)
    zero=180+zero;
}
void AngleVsPhiStudy(TProfile *histo){

  double zero, C, Cerr;
  doFit(histo,zero,C, Cerr);
  std::cout<<  "&&&&&& " << histo->GetTitle() << " FitParam " << C << " Cerr " << Cerr << " Phi(@zero) " << zero << " PhiErr " << Cerr/(1+C*C)/3.1416*180 << std::endl;;
}

void Navigate(){
  TIter nextkey(gDirectory->GetListOfKeys());
  TKey *key;
  while (key = (TKey*)nextkey()) {
    TObject *obj = key->ReadObj();
    //std::cout << " object " << obj->GetName() << " " << obj->GetTitle()<< std::endl;

    if ( obj->IsA()->InheritsFrom( "TDirectory" ) ) {
      //cout << "Found subdirectory " << obj->GetName() << " " << obj->GetTitle()<< " " << ((TDirectory*)obj)->GetPath()<< endl;

      f->cd(obj->GetTitle());
     
      Navigate();

      f->cd("..");
    } else if ( obj->IsA()->InheritsFrom( "TH1" ) ) {

      //HERE ADD YOUR QT
      if (strstr(obj->GetTitle(),"AngleVsPhi_TIB")!=NULL || strstr(obj->GetTitle(),"AngleVsPhi_TOB")!=NULL){

	//std::cout << "Found object " << obj->GetName() << " " << obj->GetTitle()<< std::endl;

	std::string vObjectsName="AngleVsPhi";
	TObjArray* vObjects = (TObjArray*) vObjectsList[vObjectsName];
	if (vObjects==NULL){
	  vObjects = new TObjArray();
	  vObjectsList[vObjectsName] = vObjects;
	}

	TProfile* h = (TProfile*)key->ReadObj();
        AngleVsPhiStudy(h);
	if (vObjects!=NULL){
	  vObjects->Add(h);
	}
      }
    }
  }
}

void TracksQT(char *input, char* output){
  
  f=new TFile(input,"READ");
  vObjectsList = std::map<std::string,TObjArray*>();

  Navigate();  

  TCanvas *c = new TCanvas();
  for (std::map<std::string,TObjArray*>::const_iterator iter=vObjectsList.begin();iter!=vObjectsList.end();++iter){
    int Nentries=iter->second->GetEntries();
    c->Divide(4,(int)Nentries/4+1);
    for (int ih=0; ih<Nentries;ih++){
      c->cd(ih+1);
      if (iter->first=="AngleVsPhi"){
	TProfile* tp = ((TProfile*) (* iter->second )[ih]);
	gStyle->SetOptStat(10);
	gStyle->SetOptFit(1111);
	//gStyle->SetStatFontSize(.1);
	tp->UseCurrentStyle();
	std::cout << "Histos " << ih << " name " //<< std::endl;
		  << tp->GetTitle() 
		  << std::endl;
	const Int_t kNotDraw = 1<<9;
	
	double binw=tp->GetBinWidth(0);
	double xmin=0;
	double xmax=130;
	int Nbinx= (int) (xmax-xmin)/binw + 1;
	
	if (A==NULL){
	  A = new TH1F("A","",Nbinx,xmin,xmin+Nbinx*binw);
	  A->SetMaximum(90.);
	  A->SetMinimum(-90.);
	  A->SetStats(0);
	  A->SetXTitle(tp->GetXaxis()->GetTitle());
	  A->SetYTitle(tp->GetYaxis()->GetTitle());
	}
	A->Draw();
	
	//	tp->GetFunction("fitf")->ResetBit(kNotDraw);
	tp->GetFunction("fitf")->SetLineWidth(1);
	tp->GetFunction("fitf")->SetLineColor(2);
	tp->Draw("sames");
	tp->GetFunction("fitf")->Draw("lsames");
	//c->Update();
	
      }
    }
    if (iter->first=="AngleVsPhi"){
      c->Print(TString(input).ReplaceAll(".root","_TracksQT_AngleVsPhi.eps"));
      c->Print(TString(input).ReplaceAll(".root","_TracksQT_AngleVsPhi.gif"));
    }
  }

  for (std::map<std::string,TObjArray*>::const_iterator iter=vObjectsList.begin();iter!=vObjectsList.end();++iter){
    for (int ih=0; ih<iter->second->GetEntries();ih++){
      if ((* iter->second )[ih])
	delete (* iter->second )[ih];
    }
    if (iter->second)
      delete iter->second;
  }
  if (A)
    delete A;
}
