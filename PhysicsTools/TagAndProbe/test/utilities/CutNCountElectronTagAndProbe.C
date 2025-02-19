#include <iostream>
#include <iomanip>



void CutNCountElectronTagAndProbe(){

/////// Are you running over data or MC ???
    TCut preCut("mcTrue");
    //TCut preCut("");

   TCut cleanSC("(( (abs(probe_eta)<1.5)"
   "&& ((probe_trkSumPtHollowConeDR03 + probe_ecalRecHitSumEtConeDR03 + probe_hcalTowerSumEtConeDR03)/probe_et < 0.07)"
   " && (probe_sigmaIetaIeta<0.01)"
   " && (probe_hadronicOverEm<0.15)"
   ")"
   " || ((abs(probe_eta)>1.5)"
   "&& ( (probe_trkSumPtHollowConeDR03 + probe_ecalRecHitSumEtConeDR03 + probe_hcalTowerSumEtConeDR03)/probe_et < 0.06 )"
   " && (probe_sigmaIetaIeta<0.03)"
   " && (probe_hadronicOverEm<0.15) "
   "))");


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
   cout << "probe type" << "         efficiency " << "       Npass" <<  
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
   cout << "########################################" << endl;
}











void ComputeEfficiency(char* primaryLabel, TCut primaryCutPass,
                       TCut primaryCutFail, char* secondaryLabel="", 
                       TCut secondaryCutPass="", TCut secondaryCutFail="",
                       char* tertiaryLabel="", 
                       TCut tertiaryCutPass = "", TCut tertiaryCutFail = "") 
{
   // TFile* f = new TFile("newhistos.root");
   TFile* f = new TFile("Zee_new80.root");

  TTree* scTree = (TTree*) f->Get("PhotonToGsf/fitter_tree");
  TTree* gsfTree = (TTree*) f->Get("GsfToIso/fitter_tree");
  char namePass[50];
  char nameFail[50];

  sprintf(namePass,"Zmass%sPass%s%s",primaryLabel,secondaryLabel,tertiaryLabel);
  sprintf(nameFail,"Zmass%sFail%s%s",primaryLabel,secondaryLabel,tertiaryLabel);
  TH1F* histPass = createHistogram(namePass, 30);
  TH1F* histFail = createHistogram(nameFail, 12);
  gsfTree->Draw("mass>>"+TString(namePass), primaryCutPass && secondaryCutPass && 
  tertiaryCutPass,"goff");
  TString checkForSCTree(primaryLabel);
  TString plotVar = "mass>>"+TString(nameFail);
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






void ComputeEfficiency( TH1& hist_pass, TH1& hist_fail)
{


  TString effType = hist_pass.GetName();
  effType.ReplaceAll("Zmass","");
  effType.ReplaceAll("Pass","");
 
  double lowerLim = 0.0;
  double upperLim = 1.0;
  double numerator = hist_pass.Integral();
  double nfails    = hist_fail.Integral();
  double bkgFractionInNum = 0.0;
  double bkgFractionInFail = 0.0;

//   if(effType.Contains("Gsf"))  { 
//      bkgFractionInNum = 0.05;
//      bkgFractionInFail = 0.02;
//   }
//   if(effType.Contains("Id"))  bkgFractionInFail= 0.05; 

  nfails  *= (1.0 - bkgFractionInFail); 
  numerator *= (1.0 - bkgFractionInNum); 

  double denominator = numerator + nfails;

  double eff = numerator/ denominator;
  //double err = sqrt(2*(1.0-eff)/totsig);
  ClopperPearsonLimits(numerator, denominator, lowerLim, upperLim);
  double errUp = upperLim -eff;
  double errLo = eff -  lowerLim;


  // ********** Make and save Canvas for the plots ********** //
  gROOT->ProcessLine(".L ~/tdrstyle.C");
  setTDRStyle();
  tdrStyle->SetErrorX(0.5);
  tdrStyle->SetPadLeftMargin(0.19);
  tdrStyle->SetPadRightMargin(0.10);
  tdrStyle->SetPadBottomMargin(0.15);
  tdrStyle->SetLegendBorderSize(0);
  tdrStyle->SetTitleYOffset(1.5);
  char temp[100];

  TString cname = TString("plot_") + effType;
  cname.ReplaceAll("Gsf","ScToGsf");
  cname.ReplaceAll("Id","GsfToWP");
  cname.Append("_Pass");
  TCanvas* c = new TCanvas(cname,cname,500,500);
  hist_pass.Draw("E1");
  TPaveText *plotlabel4 = new TPaveText(0.6,0.82,0.8,0.87,"NDC");
   plotlabel4->SetTextColor(kBlack);
   plotlabel4->SetFillColor(kWhite);
   plotlabel4->SetBorderSize(0);
   plotlabel4->SetTextAlign(12);
   plotlabel4->SetTextSize(0.03);
   sprintf(temp, "Signal = %.2f #pm %.2f", numerator, sqrt(numerator) );
   plotlabel4->AddText(temp);
  TPaveText *plotlabel5 = new TPaveText(0.6,0.77,0.8,0.82,"NDC");
   plotlabel5->SetTextColor(kBlack);
   plotlabel5->SetFillColor(kWhite);
   plotlabel5->SetBorderSize(0);
   plotlabel5->SetTextAlign(12);
   plotlabel5->SetTextSize(0.03);
   sprintf(temp, "Bkg fraction = %.2f", bkgFractionInNum);
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
   sprintf(temp, "Eff = %.3f + %.3f - %.3f", eff, errUp, errLo); 
   plotlabel7->AddText(temp);
  plotlabel4->Draw();
  plotlabel5->Draw();
  plotlabel6->Draw();
  plotlabel7->Draw();
  c->SaveAs( cname + TString(".eps"));
  c->SaveAs( cname + TString(".gif"));
  //c->SaveAs( cname + TString(".root"));
  delete c;


  cname.ReplaceAll("_Pass", "_Fail"); 
  TCanvas* c2 = new TCanvas(cname,cname,500,500);
  hist_fail.Draw("E1");

  TPaveText *plotlabel4 = new TPaveText(0.6,0.82,0.8,0.87,"NDC");
   plotlabel4->SetTextColor(kBlack);
   plotlabel4->SetFillColor(kWhite);
   plotlabel4->SetBorderSize(0);
   plotlabel4->SetTextAlign(12);
   plotlabel4->SetTextSize(0.03);
   sprintf(temp, "Signal = %.2f #pm %.2f", nfails, sqrt(nfails) );
   plotlabel4->AddText(temp);
  TPaveText *plotlabel5 = new TPaveText(0.6,0.77,0.8,0.82,"NDC");
   plotlabel5->SetTextColor(kBlack);
   plotlabel5->SetFillColor(kWhite);
   plotlabel5->SetBorderSize(0);
   plotlabel5->SetTextAlign(12);
   plotlabel5->SetTextSize(0.03);
   sprintf(temp, "Bkg fraction = %.2f", bkgFractionInFail);
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
   sprintf(temp, "Eff = %.3f + %.3f - %.3f", eff, errUp, errLo);
   plotlabel7->AddText(temp);
   plotlabel4->Draw();
   plotlabel5->Draw();
   plotlabel6->Draw();
   plotlabel7->Draw();

     c2->SaveAs( cname + TString(".eps"));
    c2->SaveAs( cname + TString(".gif"));
  //c2->SaveAs( cname + TString(".root"));
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

  cout << effTypeToPrint << "    " 
       << setiosflags(ios::fixed) << setprecision(4) << eff 
       << " + " << errUp << " - " <<  errLo 
       << setiosflags(ios::fixed) << setprecision(0)
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




