#include <iostream>
#include <cmath>
#include <TCanvas.h>
#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TLorentzVector.h>
#include <exception>
#include "PhysicsTools/Utilities/interface/SideBandSubtraction.h"
#include "RooGenericPdf.h"
#include "RooPolynomial.h"
#include "RooGaussian.h"
#include "RooProdPdf.h"
#include "RooPlot.h"
#include "RooAddPdf.h"
#include "RooDataSet.h"

using std::vector;
using std::cout;
using std::endl;

using namespace RooFit;
using std::vector;
using std::string;

void print_plot(RooDataSet *dataSet, RooRealVar printVar, string outname,string cut)
{
  RooPlot *genericFrame = printVar.frame();
  dataSet->plotOn(genericFrame,Cut(cut.c_str()) );
  TCanvas genericCanvas;
  genericFrame->Draw();
  outname = outname + ".eps";
  genericCanvas.SaveAs(outname.c_str());
  outname.replace(outname.size()-3,outname.size(),"gif");
  genericCanvas.SaveAs(outname.c_str());
}
int main(void)
{
  //variables to generate data for
  RooRealVar gfrac("gfrac","fraction of gaussian",0.5);
  RooRealVar mass("mass","Separation Variable (Mass)",2,4,"GeV/c^2");//separation variable
  RooRealVar pt("pt","pt",2,4,"GeV");

  //signal peak
  RooRealVar mean("mean","mean of signal",3.2,2,4,"GeV");
  RooRealVar sigma("sigma","width of signal",0.2,"GeV");
  RooGaussian signal("signal","signal gaussian",pt,mean,sigma);

  //constant background
  RooPolynomial background("background","Constant Background",mass,RooArgList());

  //interest variable gaussian
  RooRealVar pt_sig_mean("pt_sig_mean","mean of pt signal",3.1,2,4,"GeV");
  RooRealVar pt_sig_sigma("pt_sig_sigma","width of pt signal",0.2,"GeV");
  RooGaussian pt_sig_gauss("pt_sig_gauss","pt signal gaussian",mass,pt_sig_mean,pt_sig_sigma);

  //interest variable background
  RooRealVar pt_bkg_mean("pt_bkg_mean","mean of pt background",2.5,2,4,"GeV/c");
  RooRealVar pt_bkg_sigma("pt_bkg_sigma","width of pt background",0.2,"GeV");
  RooGaussian pt_bkg_gauss("pt_bkg_gauss","Gaussian of pt background",pt,pt_bkg_mean,pt_bkg_sigma);

  //background and model pdf for SBS
  RooProdPdf bg_pdf("signal pdf","background*mass",RooArgSet(background,signal)); //model_pdf
  RooProdPdf signal_pdf("signal_pdf","pt_sig_gauss*pt_bkg_gauss",RooArgSet(pt_sig_gauss,pt_bkg_gauss));

  //Add the two
  RooAddPdf sum_pdf("sum_pdf","bg_pdf+signal_pdf",RooArgList(bg_pdf,signal_pdf),gfrac);

  RooDataSet *datasum = sum_pdf.generate(RooArgSet(mass,pt),25000);//data
  print_plot(datasum,pt,"pt_no_cut","");

  //build the sbs object
  vector<TH1F*> basehistos(0);
  TH1F ptHist("pt","Pt Hist",100,2,4);
  basehistos.push_back(&ptHist);

  SideBandSubtract sbs(&sum_pdf,&bg_pdf,datasum,&mass,basehistos,true);
  sbs.addSideBandRegion(2,2.8);
  sbs.addSignalRegion(2.8,3.4);
  sbs.addSideBandRegion(3.4,4.0);
  sbs.doGlobalFit();
  sbs.printResults();
  return 0;
}
