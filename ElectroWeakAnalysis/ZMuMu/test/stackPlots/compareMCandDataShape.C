#include <sstream>
#include <string>
#include "stack_common.h"

void compareMCandDataShape(){



const Color_t zLineColor = kBlack;
const Color_t zFillColor = kOrange-2;

const double lumi =0.077 ;
//const double lumi =0100.0 ;
const double lumiZ = 100. ;

/// cuts common....
TCut kin_common("zGoldenDau1Pt> 20 && zGoldenDau2Pt>20 && zGoldenDau1Iso03SumPt< 3.0 && zGoldenDau2Iso03SumPt < 3.0 && abs(zGoldenDau1Eta)<2.4 &&  abs(zGoldenDau2Eta)<2.4  && (zGoldenDau1HLTBit==1 || zGoldenDau2HLTBit==1)  && abs(zGoldenDau1dxyFromBS)<0.2 && abs(zGoldenDau2dxyFromBS)<0.2 && ( abs(zGoldenDau1dxyFromBS)<0.1 || abs(zGoldenDau2dxyFromBS)<0.1 ) ");



TCut dau1Loose(" (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>10 ");
TCut dau2Loose(" (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>10 ");

TCut dau1TightWP1("zGoldenDau1Chi2<10  && (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>10 && zGoldenDau1NofPixelHits>0 && zGoldenDau1NofMuonHits>0 &&  zGoldenDau1NofMuMatches>1  && zGoldenDau1TrackerMuonBit==1");
TCut dau2TightWP1("zGoldenDau2Chi2<10  && (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>10 && zGoldenDau2NofPixelHits>0 && zGoldenDau2NofMuonHits>0 &&  zGoldenDau2NofMuMatches>1  && zGoldenDau2TrackerMuonBit==1");


TCut dau1TightWP2("zGoldenDau1Chi2<10  && (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>10  && zGoldenDau1NofMuonHits>0   && zGoldenDau1TrackerMuonBit==1");
TCut dau2TightWP2("zGoldenDau2Chi2<10  && (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>10  && zGoldenDau2NofMuonHits>0   && zGoldenDau2TrackerMuonBit==1");

TCut dau1TightWP1_hltAlso("zGoldenDau1Chi2<10  && (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>10  && zGoldenDau1NofMuonHits>0   && zGoldenDau1TrackerMuonBit==1 && zGoldenDau1HLTBit==1");
TCut dau2TightWP1_hltAlso("zGoldenDau2Chi2<10  && (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>10  && zGoldenDau2NofMuonHits>0   && zGoldenDau2TrackerMuonBit==1 && zGoldenDau2HLTBit==1");

 
TCut massCut("zGoldenMass>60 && zGoldenMass<120 ");


void makeStack(TH1 * h1,  TH1 * hdata,
	       double min, int rebin) {
  setHisto(h1, zFillColor, zLineColor, lumi/lumiZ, rebin);

  THStack * hs = new THStack("hs","");


  hs->Add(h1);

   hs->Draw("HIST");
  if(hdata != 0) {
    hdata->SetMarkerStyle(20);
    hdata->SetMarkerSize(1.0);
    hdata->SetMarkerColor(kBlack);
    hdata->SetLineWidth(2);
    hdata->SetLineColor(kBlack);
    hdata->Rebin(rebin); 
    hdata->Draw("epsame");
    hdata->GetXaxis()->SetLabelSize(0);
    hdata->GetYaxis()->SetLabelSize(0);
    // log plots, so the maximum should be one order of magnitude more...
    //     hs->SetMaximum( pow(10 , 0.0 + int(log( hdata->GetMaximum() )  )  ));
  // lin plot 
	       	 hs->SetMaximum(  4 +  hdata->GetMaximum()  )  ;
    //    gStyle->SetErrorX(.5);
}
  hs->SetMinimum(min);
  hs->GetXaxis()->SetTitle("m_{#mu^{+} #mu^{-}} (GeV/c^{2})");

 std::string yTag = "";
  switch(rebin) {
  case 1: yTag = "events/(GeV/c^{2})"; break;
  case 2: yTag = "events/(GeV/c^{2})"; break;
  case 3: yTag = "events/(3 GeV/c^{2})"; break;
  case 4: yTag = "events/(4 GeV/c^{2})"; break;
  case 5: yTag = "events/(5 GeV/c^{2})"; break;
  case 10: yTag = "events/(10 GeV/c^{2})"; break;
  default:
    std::cerr << ">>> ERROR: set y tag for rebin = " << rebin << std::endl;
  };

  hs->GetYaxis()->SetTitle(yTag.c_str());
  hs->GetXaxis()->SetTitleSize(0.05);
  hs->GetYaxis()->SetTitleSize(0.05);
  hs->GetXaxis()->SetTitleOffset(1.2);
  hs->GetYaxis()->SetTitleOffset(1.2);
  //hs->GetYaxis()->SetLabelOffset(1.0);
  hs->GetXaxis()->SetLabelSize(.05);
  hs->GetYaxis()->SetLabelSize(.05);

  leg = new TLegend(0.75,0.55,0.90,0.7);
  if(hdata != 0)
    leg->AddEntry(hdata,"data");
  leg->AddEntry(h1,"Z#rightarrow#mu #mu","f");
  //leg->AddEntry(h5,"Z#rightarrow#tau #tau","f"); 
  leg->SetFillColor(kWhite);
  leg->SetFillColor(kWhite);
  leg->SetShadowColor(kBlack);
  leg->Draw();
  // c1->SetLogy();
  //  TPaveText *pave = new TPaveText( 0.5 * (hdata->GetXaxis()->GetXmax() - (hdata->GetXaxis()->GetXmin()))  , (hdata->GetMaximum()) +1 , hdata->GetXaxis()->GetXmax() , 10 * hdata->GetMaximum()  );

  TPaveText *pave = new TPaveText( 0.6  , 0.75 , 0.9 , 0.8  , "NDC");
  pave->SetFillColor(kWhite);
  pave->SetBorderSize(0);
  //  TText * t1 = pave->AddText("CMS Preliminary 2010");
  //  TText * t2 = pave->AddText("L_{int} = 61 nb^{ -1} #sqrt{s} = 7 TeV"); // change by hand, can be improved...........  
 TText * t = pave->AddText("#int L dt = 77 nb^{ -1}");
 t->SetTextColor(kBlack);
  // t2->SetTextColor(kBlack);
  pave->Draw(); 

  TPaveText *ppave = new TPaveText( 0.15 , 0.95 , 0.65 , 1.0  , "NDC");
   ppave->SetFillColor(kWhite);
 ppave->SetBorderSize(0);
  //  TText * t1 = pave->AddText("CMS Preliminary 2010");
  //  TText * t2 = pave->AddText("L_{int} = 61 nb^{ -1} #sqrt{s} = 7 TeV"); // change by hand, can be improved...........  
 TText * tt = ppave->AddText("CMS preliminary 2010");
 //  hs->SetTitle("             #sqrt{s} = 7 TeV");
 tt->SetTextColor(kBlack);
  // t2->SetTextColor(kBlack);
  ppave->Draw(); 

  TPaveText *pppave = new TPaveText( 0.6  , 0.95 , 1.0 , 1.0  , "NDC");
  pppave->SetFillColor(kWhite);
  pppave->SetBorderSize(0);
   TText * ttt = pppave->AddText("#sqrt{s} = 7 TeV");
 ttt->SetTextColor(kBlack);
  // t2->SetTextColor(kBlack);
  pppave->Draw(); 

}

void setHisto(TH1 * h, Color_t fill, Color_t line, double scale, int rebin) {
  h->SetFillColor(fill);
  h->SetLineColor(line);
  h->Scale(scale);
  h->Rebin(rebin);  
}

// allowing two variables, for plotting the muon variables...
void comparePlots(const char * var1, const char * var2,   TCut cut, int rebin, const char * plot,
	       double min = 0.001, unsigned int nbins, double xMin, double xMax,  bool doData = true) {



TChain * zEvents = new TChain("Events"); 


 zEvents->Add("/scratch2/users/degruttola/Spring10Ntuples_withIso03/NtupleLoose_zmmSpring10cteq66_100pb.root");
TChain * wEvents = new TChain("Events"); 

TChain * dataEvents= new TChain("Events");


dataEvents->Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_132440_135802.root");
dataEvents->Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_135821-137731.root");
dataEvents->Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_138737-138751_promptreco_FF.root");
dataEvents->Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_138_919_939.root");
dataEvents->Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_139020.root");
dataEvents->Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_139_096_103.root");
dataEvents->Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_139_195_239.root");
dataEvents->Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_139347.root");
dataEvents->Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_139_356_360.root");
dataEvents->Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_139_362_365.root");
dataEvents->Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_139_368_370.root");
dataEvents->Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_139_372_375.root");
dataEvents->Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_139_399_411.root");

// .040 pb

  TH1F *h1 = new TH1F ("h1", "h1", nbins, xMin, xMax);
  TH1F *hh1 = new TH1F ("hh1", "hh1", nbins, xMin, xMax);

  zEvents->Project("h1", var1, cut);
  zEvents->Project("hh1", var2, cut);
  h1->Add(hh1);

  TH1F *hdata = new TH1F ("hdata", "hdata", nbins, xMin, xMax);
  TH1F *hhdata = new TH1F ("hhdata", "hhdata", nbins, xMin, xMax);
  dataEvents->Project("hdata", var1, cut) ;
  dataEvents->Project("hhdata", var2, cut) ;
  hdata->Add(hhdata);
  makeStack(h1,  hdata, min, rebin);
  c1->SaveAs((std::string(plot)+".eps").c_str());
  c1->SaveAs((std::string(plot)+".gif").c_str());
  c1->SaveAs((std::string(plot)+".pdf").c_str());


  hdata->KolmogorovTest(h1, "D");
  hdata->Chi2Test(h1, "UWP");
 
 
}



 comparePlots("zGoldenMass", "",  kin_common + ( ( dau1Loose  && dau2TightWP1_hltAlso ) || ( dau2Loose  && dau1TightWP1_hltAlso )) ,2 , "compareZGoldenMass_b2",  0.001, 70, 60 ,130, true);

 TCut kin_common_woPt("zGoldenDau1Iso03SumPt< 3.0 && zGoldenDau2Iso03SumPt < 3.0 && abs(zGoldenDau1Eta)<2.4 &&  abs(zGoldenDau2Eta)<2.4  && (zGoldenDau1HLTBit==1 || zGoldenDau2HLTBit==1)  && abs(zGoldenDau1dxyFromBS)<0.2 && abs(zGoldenDau2dxyFromBS)<0.2 ");



  comparePlots("zGoldenDau1Pt", "zGoldenDau2Pt", massCut + kin_common_woPt + ( ( dau1Loose  && dau2TightWP1_hltAlso ) || ( dau2Loose  && dau1TightWP1_hltAlso )), 2, "zGoldenDauPt",  0.01, 100, 0 ,100, true);
  hs->GetXaxis()->SetTitle("p_{T #mu} [GeV]");
  string yTag = "events/2 [GeV]"; // use the correct rebin
  hs->GetYaxis()->SetTitle(yTag.c_str());
  c1->SaveAs("compareZGoldenDauPt_b2.gif");


}



#endif
