#define LUMINOSITY 0.874 //(in pb^-1)
#define NBINSPASS 24
#define NBINSFAIL 6
#include <iostream>
#include <iomanip>
#include <fstream>

ofstream effTextFile("efficiency.txt");



void SimplifiedElectronTagAndProbe(){

/////// Are you running over data or MC ???
   //TCut preCut("mcTrue");
    TCut preCut("");
    TCut cleanSC("probe_hadronicOverEm<0.15");

   TCut cleanGsf("(probe_gsfEle_HoverE<0.15) && (probe_gsfEle_HoverE<0.15)");
   TCut ID95("probe_passingId");
   TCut NotID95(cleanGsf && "!probe_passingId");
   TCut NotPPass("!probe_passing");
   TCut PassAll("probe_passingALL");
   TCut NotPassAll("!probe_passingALL");
   TCut ID80("probe_passingId80");
   TCut NotID80(cleanGsf && "!probe_passingId80");
   TCut EMINUS("probe_gsfEle_charge<0");
   TCut EMINUSSC("tag_gsfEle_charge>0");
   TCut EPLUS("probe_gsfEle_charge>0");
   TCut EPLUSSC("tag_gsfEle_charge<0");
   TCut BARREL("abs(probe_sc_eta)<1.4442");
   TCut BARRELSC("abs(probe_eta)<1.4442");
   TCut ENDCAPS("abs(probe_sc_eta)>1.566");
   TCut ENDCAPSSC("abs(probe_eta)>1.566");


// //////////////////////////////////////////////////////////
   effTextFile << "probe type" << "         efficiency " << "       Npass" <<  
      "       Nfail" << endl;
// //////////////////////////////////////////////////////////



// //////////////////////////////////////////////////////////
//   //  Super cluster --> gsfElectron efficiency
// //////////////////////////////////////////////////////////

   TCut GsfPass = preCut && cleanGsf;
   TCut GsfFail = cleanSC && preCut && NotPPass;
   ComputeEfficiency("Gsf", GsfPass, GsfFail);
   ComputeEfficiency("Gsf", GsfPass, GsfFail, "Barrel", BARREL, BARRELSC);
   ComputeEfficiency("Gsf", GsfPass, GsfFail, "Endcap", ENDCAPS, ENDCAPSSC);
   ComputeEfficiency("Gsf", GsfPass, GsfFail, "", "", "", "_eminus", EMINUS, EMINUSSC);
   ComputeEfficiency("Gsf", GsfPass, GsfFail, "", "", "", "_eplus", EPLUS, EPLUSSC);
   ComputeEfficiency("Gsf", GsfPass, GsfFail, "Barrel", BARREL, BARRELSC, "_eminus", EMINUS, EMINUSSC);
   ComputeEfficiency("Gsf", GsfPass, GsfFail, "Barrel",  BARREL, BARRELSC, "_eplus", EPLUS, EPLUSSC);
   ComputeEfficiency("Gsf", GsfPass, GsfFail, "Endcap", ENDCAPS, ENDCAPSSC, "_eminus", EMINUS, EMINUSSC);
   ComputeEfficiency("Gsf", GsfPass, GsfFail, "Endcap", ENDCAPS, ENDCAPSSC, "_eplus", EPLUS, EPLUSSC);
   ComputeEfficiency("Gsf", GsfPass, GsfFail, "EndcapPlus", "probe_sc_eta>1.566", "probe_eta>1.566");
   ComputeEfficiency("Gsf", GsfPass, GsfFail, "EndcapMinus", "probe_sc_eta<-1.566", "probe_eta<-1.566");

   cout << "########################################" << endl;



// //////////////////////////////////////////////////////////
//   //  gsfElectron --> WP-95 selection efficiency
// //////////////////////////////////////////////////////////


   TCut Id95Pass = preCut && ID95;
   TCut Id95Fail = preCut && NotID95;
   ComputeEfficiency("Id95", Id95Pass, Id95Fail);
   ComputeEfficiency("Id95", Id95Pass, Id95Fail, "Barrel", BARREL, BARREL);
   ComputeEfficiency("Id95", Id95Pass, Id95Fail, "Endcap", ENDCAPS, ENDCAPS);
   ComputeEfficiency("Id95", Id95Pass, Id95Fail, "", "", "", "_eminus", EMINUS, EMINUS);
   ComputeEfficiency("Id95", Id95Pass, Id95Fail, "", "", "", "_eplus", EPLUS, EPLUS);
   ComputeEfficiency("Id95", Id95Pass, Id95Fail, "Barrel", BARREL, BARREL, "_eminus", EMINUS, EMINUS);
   ComputeEfficiency("Id95", Id95Pass, Id95Fail, "Barrel",  BARREL, BARREL, "_eplus", EPLUS, EPLUS);
   ComputeEfficiency("Id95", Id95Pass, Id95Fail, "Endcap", ENDCAPS, ENDCAPS, "_eminus", EMINUS, EMINUS);
   ComputeEfficiency("Id95", Id95Pass, Id95Fail, "Endcap", ENDCAPS, ENDCAPS, "_eplus", EPLUS, EPLUS);
   ComputeEfficiency("Id95", Id95Pass, Id95Fail, "EndcapPlus", "probe_sc_eta>1.566", "probe_eta>1.566");
   ComputeEfficiency("Id95", Id95Pass, Id95Fail, "EndcapMinus", "probe_sc_eta<-1.566", "probe_eta<-1.566");

   cout << "########################################" << endl;



// //////////////////////////////////////////////////////////
//   //  gsfElectron --> WP-80 selection efficiency
// //////////////////////////////////////////////////////////


   TCut Id80Pass = preCut && ID80;
   TCut Id80Fail = preCut && NotID80;
   ComputeEfficiency("Id80", Id80Pass, Id80Fail);
   ComputeEfficiency("Id80", Id80Pass, Id80Fail, "Barrel", BARREL, BARREL);
   ComputeEfficiency("Id80", Id80Pass, Id80Fail, "Endcap", ENDCAPS, ENDCAPS);
   ComputeEfficiency("Id80", Id80Pass, Id80Fail, "", "", "", "_eminus", EMINUS, EMINUS);
   ComputeEfficiency("Id80", Id80Pass, Id80Fail, "", "", "", "_eplus", EPLUS, EPLUS);
   ComputeEfficiency("Id80", Id80Pass, Id80Fail, "Barrel", BARREL, BARREL, "_eminus", EMINUS, EMINUS);
   ComputeEfficiency("Id80", Id80Pass, Id80Fail, "Barrel",  BARREL, BARREL, "_eplus", EPLUS, EPLUS);
   ComputeEfficiency("Id80", Id80Pass, Id80Fail, "Endcap", ENDCAPS, ENDCAPS, "_eminus", EMINUS, EMINUS);
   ComputeEfficiency("Id80", Id80Pass, Id80Fail, "Endcap", ENDCAPS, ENDCAPS, "_eplus", EPLUS, EPLUS);
   ComputeEfficiency("Id80", Id80Pass, Id80Fail, "EndcapPlus", "probe_sc_eta>1.566", "probe_eta>1.566");
   ComputeEfficiency("Id80", Id80Pass, Id80Fail, "EndcapMinus", "probe_sc_eta<-1.566", "probe_eta<-1.566");

   cout << "########################################" << endl;



// //////////////////////////////////////////////////////////
//   //   WP-95 --> HLT triggering efficiency
// //////////////////////////////////////////////////////////


   TCut HLT95Pass = preCut && ID95 && PassAll;
   TCut HLT95Fail = preCut && ID80 && NotPassAll;
   ComputeEfficiency("HLT95", HLT95Pass, HLT95Fail);
   ComputeEfficiency("HLT95", HLT95Pass, HLT95Fail, "Barrel", BARREL, BARREL);
   ComputeEfficiency("HLT95", HLT95Pass, HLT95Fail, "Endcap", ENDCAPS, ENDCAPS);
   ComputeEfficiency("HLT95", HLT95Pass, HLT95Fail, "", "", "", "_eminus", EMINUS, EMINUS);
   ComputeEfficiency("HLT95", HLT95Pass, HLT95Fail, "", "", "", "_eplus", EPLUS, EPLUS);
   ComputeEfficiency("HLT95", HLT95Pass, HLT95Fail, "Barrel", BARREL, BARREL, "_eminus", EMINUS, EMINUS);
   ComputeEfficiency("HLT95", HLT95Pass, HLT95Fail, "Barrel",  BARREL, BARREL, "_eplus", EPLUS, EPLUS);
   ComputeEfficiency("HLT95", HLT95Pass, HLT95Fail, "Endcap", ENDCAPS, ENDCAPS, "_eminus", EMINUS, EMINUS);
   ComputeEfficiency("HLT95", HLT95Pass, HLT95Fail, "Endcap", ENDCAPS, ENDCAPS, "_eplus", EPLUS, EPLUS);
   ComputeEfficiency("HLT95", HLT95Pass, HLT95Fail, "EndcapPlus", "probe_sc_eta>1.566", "probe_eta>1.566");
   ComputeEfficiency("HLT95", HLT95Pass, HLT95Fail, "EndcapMinus", "probe_sc_eta<-1.566", "probe_eta<-1.566");

   cout << "########################################" << endl;




// //////////////////////////////////////////////////////////
//   //   WP-80 --> HLT triggering efficiency
// //////////////////////////////////////////////////////////

 
   TCut HLT80Pass = preCut && ID80 && PassAll;
   TCut HLT80Fail = preCut && ID80 && NotPassAll;
   ComputeEfficiency("HLT80", HLT80Pass, HLT80Fail);
   ComputeEfficiency("HLT80", HLT80Pass, HLT80Fail, "Barrel", BARREL, BARREL);
   ComputeEfficiency("HLT80", HLT80Pass, HLT80Fail, "Endcap", ENDCAPS, ENDCAPS);
   ComputeEfficiency("HLT80", HLT80Pass, HLT80Fail, "", "", "", "_eminus", EMINUS, EMINUS);
   ComputeEfficiency("HLT80", HLT80Pass, HLT80Fail, "", "", "", "_eplus", EPLUS, EPLUS);
   ComputeEfficiency("HLT80", HLT80Pass, HLT80Fail, "Barrel", BARREL, BARREL, "_eminus", EMINUS, EMINUS);
   ComputeEfficiency("HLT80", HLT80Pass, HLT80Fail, "Barrel",  BARREL, BARREL, "_eplus", EPLUS, EPLUS);
   ComputeEfficiency("HLT80", HLT80Pass, HLT80Fail, "Endcap", ENDCAPS, ENDCAPS, "_eminus", EMINUS, EMINUS);
   ComputeEfficiency("HLT80", HLT80Pass, HLT80Fail, "Endcap", ENDCAPS, ENDCAPS, "_eplus", EPLUS, EPLUS);
   ComputeEfficiency("HLT80", HLT80Pass, HLT80Fail, "EndcapPlus", "probe_sc_eta>1.566", "probe_eta>1.566");
   ComputeEfficiency("HLT80", HLT80Pass, HLT80Fail, "EndcapMinus", "probe_sc_eta<-1.566", "probe_eta<-1.566");

   cout << "########################################" << endl;

   effTextFile.close();
}











void ComputeEfficiency(char* primaryLabel, TCut primaryCutPass,
                       TCut primaryCutFail, char* secondaryLabel="", 
                       TCut secondaryCutPass="", TCut secondaryCutFail="",
                       char* tertiaryLabel="", 
                       TCut tertiaryCutPass = "", TCut tertiaryCutFail = "") 
{
    TFile* f = new TFile("allTPtrees_836nb.root");

  TTree* scTree = (TTree*) f->Get("PhotonToGsf/fitter_tree");
  TTree* gsfTree = (TTree*) f->Get("GsfToIso/fitter_tree");
  char namePass[50];
  char nameFail[50];

  sprintf(namePass,"Zmass%sPass%s%s",primaryLabel,secondaryLabel,tertiaryLabel);
  sprintf(nameFail,"Zmass%sFail%s%s",primaryLabel,secondaryLabel,tertiaryLabel);
  TH1F* histPass = createHistogram(namePass, 30);
  TH1F* histFail = createHistogram(nameFail, 12);
  gsfTree->Draw("1.02*mass>>"+TString(namePass), primaryCutPass && secondaryCutPass && 
  tertiaryCutPass,"goff");
  TString checkForSCTree(primaryLabel);
  TString plotVar = "1.03*mass>>"+TString(nameFail);
  if(checkForSCTree.Contains("Gsf"))
     scTree->Draw(plotVar,primaryCutFail && secondaryCutFail && tertiaryCutFail,"goff");
  else 
     gsfTree->Draw(plotVar,primaryCutFail && secondaryCutFail && tertiaryCutFail,"goff");
  ComputeEfficiency(*histPass, *histFail);

  delete histPass;
  delete histFail;
}






TH1F* createHistogram(char* name, int nbins=12) {
  TH1F* hist = new TH1F(name,name, nbins, 60, 120);
  hist->SetTitle("");
  char temp[100];
  sprintf(temp, "Events / %.1f GeV/c^{2}", 60./nbins);
  hist->GetXaxis()->SetTitle("m_{ee} (GeV/c^{2})");
  hist->GetYaxis()->SetTitle(temp);
  return hist;
}






double ErrorInProduct(double x, double errx, double y, 
                      double erry, double corr) {
   double xFrErr = errx/x;
   double yFrErr = erry/y;
   return sqrt(xFrErr**2 +yFrErr**2 + 2.0*corr*xFrErr*yFrErr)*x*y;
}


void ComputeEfficiency( TH1& hist_pass, TH1& hist_fail)
{


  TString effType = hist_pass.GetName();
  effType.ReplaceAll("Zmass","");
  effType.ReplaceAll("Pass","");

  // The fit variable - lepton invariant mass
  RooRealVar* rooMass_ = new RooRealVar("Mass","m_{ee}",60.0, 120.0, "GeV/c^{2}");
  RooRealVar Mass = *rooMass_;

  // Make the category variable that defines the two fits,
  // namely whether the probe passes or fails the eff criteria.
  RooCategory sample("sample","") ;
  sample.defineType("Pass", 1) ;
  sample.defineType("Fail", 2) ; 


  gROOT->cd();


  ///////// convert Histograms into RooDataHists
  RooDataHist* data_pass = new RooDataHist("data_pass","data_pass",
					  RooArgList(Mass), &hist_pass);
  RooDataHist* data_fail = new RooDataHist("data_fail","data_fail",
					  RooArgList(Mass), &hist_fail);

  RooDataHist* data = new RooDataHist( "fitData","fitData",
  RooArgList(Mass),RooFit::Index(sample),
  RooFit::Import("Pass",*data_pass), RooFit::Import("Fail",*data_fail) ); 




 // Signal pdf

  Zeelineshape_file =  new TFile("Zlineshapes.root", "READ");
  TH1* histbbpass = (TH1D*) Zeelineshape_file->Get("pass_BB");
  TH1* histebpass = (TH1D*) Zeelineshape_file->Get("pass_BE");
  TH1* histeepass = (TH1D*) Zeelineshape_file->Get("pass_EE");
  TH1D* th1 = (TH1D*) histbbpass->Clone("th1");
  th1->Add(histebpass);
  th1->Add(histeepass);
  RooDataHist* rdh = new RooDataHist("rdh","", Mass, th1);
  RooHistPdf* signalShapePdf = new RooHistPdf("signalShapePdf", "",
  RooArgSet(Mass), *rdh);



//   RooRealVar* Mean_   = new RooRealVar("Mean","Mean", 90.0, 86.2, 95.2);
//   RooRealVar* Width_  = new RooRealVar("Width","Width", 2.5);
//   RooRealVar* Resolution_  = new RooRealVar("Resolution","Resolution", 5.6, 0., 10.);
//   RooRealVar* MeanBifG_   = new RooRealVar("MeanBifG","MeanBifG", 87.50, 86.2, 95.2);
//   RooRealVar* WidthL_  = new RooRealVar("WidthL","WidthL", 3., 2., 6.);
//   RooRealVar* WidthR_  = new RooRealVar("WidthR","WidthR", 6.);
//   RooRealVar* BifurGaussFrac_ = new RooRealVar("BifurGaussFrac","",0.2, 0.0, 1.0);

//   // Voigtian
//   RooAbsPdf* voigtianPdf= new RooVoigtian("voigtianPdf", "", 
//   Mass, *Mean_, *Width_, *Resolution_);

//   RooAbsPdf* signalShapePdf= voigtianPdf;


//   // Bifurcated Gaussian
//   RooAbsPdf* bifurGaussPdf_ = new RooBifurGauss("bifurGaussPdf", "", 
//   Mass, *Mean_, *WidthL_, *WidthL_);

//   // Signal PDF 
//   RooAddPdf* signalShapePdf= new RooAddPdf("signalShapePdf", "", 
//   *voigtianPdf, *bifurGaussPdf_,*BifurGaussFrac_);

//   // Signal PDF --> Failing sample
//   RooRealVar* ResolutionFail_  = new RooRealVar("ResolutionFail","ResolutionFail", 5.6, 0., 10.);
//   RooAbsPdf* signalShapePdfFail= new RooVoigtian("signalShapePdfFail", "", 
//   Mass, *Mean_, *Width_, *ResolutionFail_);
 

  // Background PDF 
  RooRealVar* bkgShape = new RooRealVar("bkgShape","bkgShape",-0.2,-10.,0.);
  RooExponential* bkgShapePdf = new RooExponential("bkgShapePdf","bkgShapePdf",Mass, *bkgShape);


  // Now define some efficiency/yield variables  
  RooRealVar* numSignal = new RooRealVar("numSignal","numSignal", 4000.0, 0.0, 100000.0);
  RooRealVar* eff = new RooRealVar("eff","eff", 0.9, 0.5, 1.0);
  RooFormulaVar* nSigPass = new RooFormulaVar("nSigPass", "eff*numSignal", RooArgList(*eff,*numSignal));
  RooFormulaVar* nSigFail = new RooFormulaVar("nSigFail", "(1.0-eff)*numSignal", RooArgList(*eff,*numSignal));
  RooRealVar* nBkgPass = new RooRealVar("nBkgPass","nBkgPass", 1000.0, 0.0, 10000000.0);
  RooRealVar* nBkgFail = new RooRealVar("nBkgFail","nBkgFail", 1000.0, 0.0, 10000000.0);


  RooArgList componentsPass(*signalShapePdf,*bkgShapePdf);
  RooArgList yieldsPass(*nSigPass, *nBkgPass);
  //RooArgList componentsFail(*signalShapePdfFail,*bkgShapePdf);
  RooArgList componentsFail(*signalShapePdf,*bkgShapePdf);
  RooArgList yieldsFail(*nSigFail, *nBkgFail);


   RooAddPdf pdfPass("pdfPass","extended sum pdf", componentsPass, yieldsPass);
   RooAddPdf pdfFail("pdfFail","extended sum pdf", componentsFail, yieldsFail);



   // The total simultaneous fit ...
   RooSimultaneous totalPdf("totalPdf","totalPdf", sample);
   totalPdf.addPdf(pdfPass,"Pass");
   totalPdf.Print();
   totalPdf.addPdf(pdfFail,"Fail");
   totalPdf.Print();


  // ********* Do the Actual Fit ********** //  
   RooFitResult *fitResult = totalPdf.fitTo(*data, RooFit::Save(true), 
   RooFit::Extended(true), RooFit::PrintLevel(-1));
  fitResult->Print("v");

  double numerator = nSigPass->getVal();
  double nfails    = nSigFail->getVal();
  double denominator = numerator + nfails;




  // ********** Make and save Canvas for the plots ********** //
  gROOT->ProcessLine(".L ~/tdrstyle.C");
  setTDRStyle();
  tdrStyle->SetErrorX(0.5);
  tdrStyle->SetPadLeftMargin(0.19);
  tdrStyle->SetPadRightMargin(0.10);
  tdrStyle->SetPadBottomMargin(0.15);
  tdrStyle->SetLegendBorderSize(0);
  tdrStyle->SetTitleYOffset(1.5);
  RooAbsData::ErrorType errorType = RooAbsData::Poisson;

  TString cname = TString("fit_") + hist_pass.GetName();
  TCanvas* c = new TCanvas(cname,cname,500,500);
  RooPlot* frame1 = Mass.frame();
  frame1->SetMinimum(0);
  data_pass->plotOn(frame1,RooFit::DataError(errorType));
  pdfPass.plotOn(frame1,RooFit::ProjWData(*data_pass), 
  RooFit::Components(*bkgShapePdf),RooFit::LineColor(kRed));
  pdfPass.plotOn(frame1,RooFit::ProjWData(*data_pass));
  frame1->Draw("e0");

  TPaveText *plotlabel = new TPaveText(0.23,0.87,0.43,0.92,"NDC");
   plotlabel->SetTextColor(kBlack);
   plotlabel->SetFillColor(kWhite);
   plotlabel->SetBorderSize(0);
   plotlabel->SetTextAlign(12);
   plotlabel->SetTextSize(0.03);
   plotlabel->AddText("CMS Preliminary 2010");
  TPaveText *plotlabel2 = new TPaveText(0.23,0.82,0.43,0.87,"NDC");
   plotlabel2->SetTextColor(kBlack);
   plotlabel2->SetFillColor(kWhite);
   plotlabel2->SetBorderSize(0);
   plotlabel2->SetTextAlign(12);
   plotlabel2->SetTextSize(0.03);
   plotlabel2->AddText("#sqrt{s} = 7 TeV");
  TPaveText *plotlabel3 = new TPaveText(0.23,0.75,0.43,0.80,"NDC");
   plotlabel3->SetTextColor(kBlack);
   plotlabel3->SetFillColor(kWhite);
   plotlabel3->SetBorderSize(0);
   plotlabel3->SetTextAlign(12);
   plotlabel3->SetTextSize(0.03);
  char temp[100];
  sprintf(temp, "%.4f", LUMINOSITY);
  plotlabel3->AddText((string("#int#font[12]{L}dt = ") + 
  temp + string(" pb^{ -1}")).c_str());
  TPaveText *plotlabel4 = new TPaveText(0.6,0.82,0.8,0.87,"NDC");
   plotlabel4->SetTextColor(kBlack);
   plotlabel4->SetFillColor(kWhite);
   plotlabel4->SetBorderSize(0);
   plotlabel4->SetTextAlign(12);
   plotlabel4->SetTextSize(0.03);
   double nsig = numSignal->getVal();
   double nErr = numSignal->getError();
   double e = eff->getVal();
   double eErr = eff->getError();
   double corr = fitResult->correlation(*eff, *numSignal);
   double err = ErrorInProduct(nsig, nErr, e, eErr, corr);
   sprintf(temp, "Signal = %.2f #pm %.2f", nSigPass->getVal(), err);
   plotlabel4->AddText(temp);
  TPaveText *plotlabel5 = new TPaveText(0.6,0.77,0.8,0.82,"NDC");
   plotlabel5->SetTextColor(kBlack);
   plotlabel5->SetFillColor(kWhite);
   plotlabel5->SetBorderSize(0);
   plotlabel5->SetTextAlign(12);
   plotlabel5->SetTextSize(0.03);
   sprintf(temp, "Bkg = %.2f #pm %.2f", nBkgPass->getVal(), nBkgPass->getError());
   plotlabel5->AddText(temp);
  TPaveText *plotlabel6 = new TPaveText(0.6,0.87,0.8,0.92,"NDC");
   plotlabel6->SetTextColor(kBlack);
   plotlabel6->SetFillColor(kWhite);
   plotlabel6->SetBorderSize(0);
   plotlabel6->SetTextAlign(12);
   plotlabel6->SetTextSize(0.03);
   plotlabel6->AddText("Passing probes");
  TPaveText *plotlabel7 = new TPaveText(0.6,0.72,0.8,0.77,"NDC");
   plotlabel7->SetTextColor(kBlack);
   plotlabel7->SetFillColor(kWhite);
   plotlabel7->SetBorderSize(0);
   plotlabel7->SetTextAlign(12);
   plotlabel7->SetTextSize(0.03); 
   sprintf(temp, "Eff = %.3f #pm %.3f", eff->getVal(), eff->getErrorHi());
   plotlabel7->AddText(temp);
//   TPaveText *plotlabel9 = new TPaveText(0.6,0.67,0.8,0.72,"NDC");
//    plotlabel9->SetTextColor(kBlack);
//    plotlabel9->SetFillColor(kWhite);
//    plotlabel9->SetBorderSize(0);
//    plotlabel9->SetTextAlign(12);
//    plotlabel9->SetTextSize(0.03);
//    sprintf(temp, "Resolution = %.2f #pm %.2f", Resolution_->getVal(), Resolution_->getError());
//    plotlabel9->AddText(temp);
//   plotlabel->Draw();
//   plotlabel2->Draw();
//   plotlabel3->Draw();
  plotlabel4->Draw();
  plotlabel5->Draw();
  plotlabel6->Draw();
  plotlabel7->Draw();
//   plotlabel9->Draw();

  c->SaveAs( cname + TString(".eps"));
  c->SaveAs( cname + TString(".gif"));
  c->SaveAs( cname + TString(".root"));
  delete c;


  cname = TString("fit_") + hist_fail.GetName();
  TCanvas* c2 = new TCanvas(cname,cname,500,500);
  RooPlot* frame2 = Mass.frame();
  frame2->SetMinimum(0);
  data_fail->plotOn(frame2,RooFit::DataError(errorType));
  pdfFail.plotOn(frame2,RooFit::ProjWData(*data_fail), 
  RooFit::Components(*bkgShapePdf),RooFit::LineColor(kRed));
  pdfFail.plotOn(frame2,RooFit::ProjWData(*data_fail));
  frame2->Draw("e0");

  TPaveText *plotlabel = new TPaveText(0.23,0.87,0.43,0.92,"NDC");
   plotlabel->SetTextColor(kBlack);
   plotlabel->SetFillColor(kWhite);
   plotlabel->SetBorderSize(0);
   plotlabel->SetTextAlign(12);
   plotlabel->SetTextSize(0.03);
   plotlabel->AddText("CMS Preliminary 2010");
  TPaveText *plotlabel2 = new TPaveText(0.23,0.82,0.43,0.87,"NDC");
   plotlabel2->SetTextColor(kBlack);
   plotlabel2->SetFillColor(kWhite);
   plotlabel2->SetBorderSize(0);
   plotlabel2->SetTextAlign(12);
   plotlabel2->SetTextSize(0.03);
   plotlabel2->AddText("#sqrt{s} = 7 TeV");
  TPaveText *plotlabel3 = new TPaveText(0.23,0.75,0.43,0.80,"NDC");
   plotlabel3->SetTextColor(kBlack);
   plotlabel3->SetFillColor(kWhite);
   plotlabel3->SetBorderSize(0);
   plotlabel3->SetTextAlign(12);
   plotlabel3->SetTextSize(0.03);
  char temp[100];
  sprintf(temp, "%.4f", LUMINOSITY);
  plotlabel3->AddText((string("#int#font[12]{L}dt = ") + 
  temp + string(" pb^{ -1}")).c_str());
  TPaveText *plotlabel4 = new TPaveText(0.6,0.82,0.8,0.87,"NDC");
   plotlabel4->SetTextColor(kBlack);
   plotlabel4->SetFillColor(kWhite);
   plotlabel4->SetBorderSize(0);
   plotlabel4->SetTextAlign(12);
   plotlabel4->SetTextSize(0.03);
   double err = ErrorInProduct(nsig, nErr, 1.0-e, eErr, corr);
   sprintf(temp, "Signal = %.2f #pm %.2f", nSigFail->getVal(), err);
   plotlabel4->AddText(temp);
  TPaveText *plotlabel5 = new TPaveText(0.6,0.77,0.8,0.82,"NDC");
   plotlabel5->SetTextColor(kBlack);
   plotlabel5->SetFillColor(kWhite);
   plotlabel5->SetBorderSize(0);
   plotlabel5->SetTextAlign(12);
   plotlabel5->SetTextSize(0.03);
   sprintf(temp, "Bkg = %.2f #pm %.2f", nBkgFail->getVal(), nBkgFail->getError());
   plotlabel5->AddText(temp);
  TPaveText *plotlabel6 = new TPaveText(0.6,0.87,0.8,0.92,"NDC");
   plotlabel6->SetTextColor(kBlack);
   plotlabel6->SetFillColor(kWhite);
   plotlabel6->SetBorderSize(0);
   plotlabel6->SetTextAlign(12);
   plotlabel6->SetTextSize(0.03);
   plotlabel6->AddText("Failing probes");
  TPaveText *plotlabel7 = new TPaveText(0.6,0.72,0.8,0.77,"NDC");
   plotlabel7->SetTextColor(kBlack);
   plotlabel7->SetFillColor(kWhite);
   plotlabel7->SetBorderSize(0);
   plotlabel7->SetTextAlign(12);
   plotlabel7->SetTextSize(0.03);
   sprintf(temp, "Eff = %.3f #pm %.3f", eff->getVal(), eff->getErrorHi(), eff->getErrorLo());
   plotlabel7->AddText(temp);

//   plotlabel->Draw();
//   plotlabel2->Draw();
//   plotlabel3->Draw();
  plotlabel4->Draw();
  plotlabel5->Draw();
  plotlabel6->Draw();
  plotlabel7->Draw();

  c2->SaveAs( cname + TString(".eps"));
  c2->SaveAs( cname + TString(".gif"));
  c2->SaveAs( cname + TString(".root"));
  delete c2;

  // cout << "########################################" << endl;
  effType.ReplaceAll("Gsf","");
  effType.ReplaceAll("Id95_","");
  effType.ReplaceAll("Id95","");
  effType.ReplaceAll("Id80_","");
  effType.ReplaceAll("Id80","");
  effType.ReplaceAll("HLT95_","");
  effType.ReplaceAll("HLT95","");
  effType.ReplaceAll("HLT80_","");
  effType.ReplaceAll("HLT80","");

 char* effTypeToPrint = (char*) effType;

  effTextFile << effTypeToPrint << "    " 
       << setiosflags(ios::fixed) << setprecision(4) << eff.getVal() 
       << " + " << eff->getErrorHi() << " - " <<  eff->getErrorLo()
       << setiosflags(ios::fixed) << setprecision(2)
       << "    " << numerator << "    "  << nfails << endl;
  //cout << "########################################" << endl;
}


double ErrorInProduct(double x, double errx, double y, 
                      double erry, double corr) {
   double xFrErr = errx/x;
   double yFrErr = erry/y;
   return sqrt(xFrErr**2 +yFrErr**2 + 2.0*corr*xFrErr*yFrErr)*x*y;
}






void ClopperPearsonLimits(double numerator, double denominator, 
double &lowerLimit, double &upperLimit, const double CL_low=1.0, 
const double CL_high=1.0) 
{  
//Confidence intervals are in the units of \sigma.

   double ratio = numerator/denominator;
   
// first get the lower limit
   if(numerator==0)   lowerLimit = 0.0; 
   else { 
      double v=ratio/2; 
      double vsL=0; 
      double vsH=ratio; 
      double p=CL_low/100;
      while((vsH-vsL)>1e-5) { 
         if(BinP(denominator,v,numerator,denominator)>p) 
         { vsH=v; v=(vsL+v)/2; } 
         else { vsL=v; v=(v+vsH)/2; } 
      }
      lowerLimit = v; 
   }
   
// now get the upper limit
   if(numerator==denominator) upperLimit = 1.0;
   else { 
      double v=(1+ratio)/2; 
      double vsL=ratio; 
      double vsH=1; 
      double p=CL_high/100;
      while((vsH-vsL)>1e-5) { 
         if(BinP(denominator,v,0,numerator)<p) { vsH=v; v=(vsL+v)/2; } 
         else { vsL=v; v=(v+vsH)/2; } 
      }
      upperLimit = v;
   }
}




double BinP(int N, double p, int x1, int x2) {
   double q=p/(1-p); 
   int k=0; 
   double v = 1; 
   double s=0; 
   double tot=0.0;
    while(k<=N) {
       tot=tot+v;
       if(k>=x1 & k<=x2) { s=s+v; }
       if(tot>1e30){s=s/1e30; tot=tot/1e30; v=v/1e30;}
       k=k+1; 
       v=v*q*(N+1-k)/k;
    }
    return s/tot;
}



