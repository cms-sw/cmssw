{
  gStyle -> SetOptStat(1);
  gStyle->SetPalette(1);
  gStyle->SetPadGridX(1);
  gStyle->SetPadGridY(1);
  //gStyle -> SetTitleFont(82);
  gStyle -> SetTextFont(102);
  //  Int_t fillstyle = 3008;
Int_t sample=0;

// TString imgpath("~/afs/public_html/validation/cone26piminus/");  
// TFile* tf = new TFile("./ValidFile_minus.root","OPEN");
// TString imgpath("~/afs/public_html/validation/cone26piplus/");  
// TFile* tf = new TFile("./ValidFile_plus.root","OPEN");

 TString imgpath("~/afs/public_html/validation/data900v2/");  
 TFile* tf = new TFile("./ValidFile_data900v3sp.root","OPEN");
// TFile* tf = new TFile("./ValidFile_MB900.root","OPEN");
 //TFile* tf = new TFile("./ValidFile_XX_4.root","OPEN");
  

  //TCut tr_cut = "eTrack>47 && eTrack<53";  label = new TText(0.97,0.1, "50 GeV"); sample=50;

//  TCut tr_cut = "eTrack>41 && eTrack<59";  label = new TText(0.97,0.1, "40-60 GeV"); sample=50;
//  TCut tr_cut = "eTrack>5 && eTrack<100";  label = new TText(0.97,0.1, "MinBias"); sample=900;
  TCut tr_cut = "eTrack>5 && eTrack<60";  label = new TText(0.97,0.1, "Data @ 900 GeV"); sample=900;

  
label -> SetNDC();
label->SetTextAlign(11); 
label->SetTextSize(0.05);
label->SetTextAngle(90);

TCut tr_quality = "numValidTrkHits>=13 && (abs(etaTrack) <= 1.47 || numValidTrkStrips>=9)"; 
 TCut mip_cut = "eECAL<1";
 TCut hit_dist = "iDr<1.5";
 TCut ptNear = "PtNearBy<2";
TCut selection = tr_cut && mip_cut && tr_quality && hit_dist && ptNear;
  

//  TTree* ftree = (TTree*)tf->Get("fTree");  
//  TTree* ttree = (TTree*)tf->Get("tTree");
  TTree* ftree = (TTree*)tf->Get("ValidationIsoTrk/fTree");  
  TTree* ttree = (TTree*)tf->Get("ValidationIsoTrk/tTree");
   Int_t  nentries = (Int_t)ftree->GetEntries();
  Double_t nent = 1.;
  TCanvas* c1 = new TCanvas("c1","all",0,0,350,350);
  c1-> cd();
    
  ftree -> Draw("eTrack>>eTrack", selection);
  eTrack -> SetTitle("eTrack");
label -> Draw();
  c1 -> SetLogy();
   c1->SaveAs(imgpath+"p01.png");  
   
  ftree -> Draw("eECAL>>eECAL", tr_cut, "");
  //  ftree -> Draw("eECAL>>eECAL", tr_cut&& tr_quality && hit_dist, "");
  eECAL -> SetLineColor(kRed-2);
  eECAL -> SetTitle("EM energy");
label -> Draw();
  c1->SaveAs(imgpath+"p02.png");  

  c1 -> SetLogy(0);

  ftree -> Draw("numHits>>nh2", selection&&"abs(iEta)>17");
nh2 -> SetBins(20, 0,20);
  ftree -> Draw("numHits>>nh1", selection&&"abs(iEta)<14", "same");
  nh2 -> SetTitle("Number of Hits");
  nh1 -> SetLineColor(kOrange+2);
  nh2 -> SetLineColor(kBlue+2);
  leg = new TLegend(0.7,0.6,0.99,0.83);
  leg -> SetTextSize(0.04);
  leg->AddEntry(nh1,"#splitline{HBarrel}{|iEta|<14}","l");
  leg->AddEntry(nh2,"#splitline{HEndcap}{|iEta|>17}","l");
  leg -> Draw();
label -> Draw();
  c1->SaveAs(imgpath+"p03.png");  

  ftree -> Draw("eTrack>>hh1", selection);
  ftree -> Draw("eClustAfter>>hh2", selection);
  ftree -> Draw("eClustBefore>>hh3", selection);
hh2 -> SetLineColor(kGreen+2);
hh3 -> SetLineColor(kRed+1);
  hh1 -> SetTitle("eTrack and eHcalClast");
  hh1 -> Draw("");
hh1 -> SetAxisRange(0.,60.);
  //Double_t hmax = hh2->GetMaximum();
  //hh2->SetMaximum(hmax + hmax*0.1);
  hh3 -> Draw("same");
  hh2 -> Draw("same");
  leg = new TLegend(0.62,0.55,0.99,0.75);
  leg -> SetTextSize(0.04);
  leg->AddEntry(hh1,"orig. track","l");
  leg->AddEntry(hh3,"tr. bef calib","l");
  leg->AddEntry(hh2,"tr. aft calib","l");
  leg -> Draw();
label -> Draw();
  c1->SaveAs(imgpath+"p04.png");  
  hh1 -> Delete();
  hh2 -> Delete();
  hh3 -> Delete();


  ftree -> Draw("iEta>>hh1", selection);
  hh1 -> SetTitle("iEta");
label -> Draw();
  c1->SaveAs(imgpath+"p05.png");  
  hh1 -> Delete();

  ftree -> Draw("iPhi>>hh2", selection&&"abs(iEta)>20");
  ftree -> Draw("iPhi>>hh1", selection&&"abs(iEta)<20", "same");
  hh1 -> SetLineColor(kRed+1);
  hh2 -> SetLineColor(kBlue+1);
  hh1 -> SetTitle("iPhi");
  leg = new TLegend(0.68,0.55,0.99,0.7);
if (sample==300){
  ftree -> Draw("iPhi>>hh3", selection&&"abs(iEta)<14", "same");
  ftree -> Draw("iPhi>>hh4", selection&&"abs(iEta)>14 && abs(iEta)<20", "same");
  hh3 -> SetLineColor(kGreen+1);
  hh4 -> SetLineColor(kOrange+1);
  leg = new TLegend(0.68,0.5,0.99,0.73);
  leg->AddEntry(hh3,"|iEta|<14","l");
  leg->AddEntry(hh4,"14<|iEta|<20","l");
}
  leg -> SetTextSize(0.036);
  leg->AddEntry(hh1,"|iEta|<20","l");
  leg->AddEntry(hh2,"|iEta|>20","l");
  leg -> Draw();
label -> Draw();
  c1->SaveAs(imgpath+"p06.png");  
  hh1 -> Delete();
  hh2 -> Delete();
   
  ftree -> Draw("eClustBefore/eTrack>>hh1", selection, "goff");
  hh1 -> SetTitle("Response = eHcalClast/eTrack");
  hh1 -> SetLineColor(kRed+1);
hh1 -> SetAxisRange(0., 5);
  hh1 -> Draw();
  ftree -> Draw("eClustAfter/eTrack>>hh2", selection, "same");
  hh2 -> SetLineColor(kGreen+2);
  leg = new TLegend(0.68,0.77,0.99,0.91);
  leg -> SetTextSize(0.04);
  leg->AddEntry(hh1,"bef calib","l");
  leg->AddEntry(hh2,"aft calib","l");
  leg -> Draw();
label -> Draw();
 Float_t mean1 = hh1 ->GetMean();
 Float_t mean2 = hh2 ->GetMean();
 Float_t rmsh1 = hh1 ->GetRMS();
 Float_t rmsh2 = hh2 ->GetRMS();
 //cout<<"mean1: "<<mean1<<"   mean2: "<<mean2<<endl;
 m1 = new TText(0.7,0.63, Form("mean1: %1.2f",mean1));  
 r1 = new TText(0.7,0.58, Form(" rms1: %1.2f",rmsh1));  
 m2 = new TText(0.7,0.45, Form("mean2: %1.2f", mean2));
 r2 = new TText(0.7,0.4, Form(" rms2: %1.2f", rmsh2));
 m1 -> SetNDC();
 m2 -> SetNDC();
 r1 -> SetNDC();
 r2 -> SetNDC();
 m1 -> Draw();  
 m2 -> Draw();  
 r1 -> Draw();  
 r2 -> Draw();  
 c1->SaveAs(imgpath+"p07a.png");  
  hh1 -> Delete();
  hh2 -> Delete();

//  ftree -> Draw("eClustAfter/eTrack>>hh2", selection&&"abs(iEta)<17 && eClustAfter/eTrack>0.2", 
//"goff");
//  ftree -> Draw("eClustBefore/eTrack>>hh1", selection&&"abs(iEta)<17 && eClustBefore/eTrack>0.2", 
//"goff");
  ftree -> Draw("eClustAfter/eTrack>>hh2", selection&&"abs(iEta)<17", "goff");
  ftree -> Draw("eClustBefore/eTrack>>hh1", selection&&"abs(iEta)<17", "goff");
  hh2 -> SetLineColor(kGreen+2);
  hh1 -> SetLineColor(kRed+1);
  hh2 -> SetTitle("Response in HBarrel |iEta|<17");
hh2 -> SetAxisRange(0., 5);
  hh2 -> Draw("");
  hh1 -> Draw("same");
  leg = new TLegend(0.68,0.77,0.99,0.91);
  leg -> SetTextSize(0.04);
  leg->AddEntry(hh1,"bef calib","l");
  leg->AddEntry(hh2,"aft calib","l");
  leg -> Draw();
label -> Draw();
 Float_t mean1 = hh1 ->GetMean();
 Float_t mean2 = hh2 ->GetMean();
 Float_t rmsh1 = hh1 ->GetRMS();
 Float_t rmsh2 = hh2 ->GetRMS();
 //cout<<"mean1: "<<mean1<<"   mean2: "<<mean2<<endl;
 m1 = new TText(0.7,0.63, Form("mean1: %1.2f",mean1));  
 r1 = new TText(0.7,0.58, Form(" rms1: %1.2f",rmsh1));  
 m2 = new TText(0.7,0.45, Form("mean2: %1.2f", mean2));
 r2 = new TText(0.7,0.4, Form(" rms2: %1.2f", rmsh2));
 m1 -> SetNDC();
 m2 -> SetNDC();
 r1 -> SetNDC();
 r2 -> SetNDC();
 m1 -> Draw();  
 m2 -> Draw();  
 r1 -> Draw();  
 r2 -> Draw();  
  c1->SaveAs(imgpath+"p07b.png");  
  hh1 -> Delete();
  hh2 -> Delete();

//  ftree -> Draw("eClustAfter/eTrack>>hh2", selection&&"abs(iEta)>17  && eClustAfter/eTrack>0.2");
//  ftree -> Draw("eClustBefore/eTrack>>hh1", selection&&"abs(iEta)>17  && eClustBefore/eTrack>0.2");
  ftree -> Draw("eClustAfter/eTrack>>hh2", selection&&"abs(iEta)>17");
  ftree -> Draw("eClustBefore/eTrack>>hh1", selection&&"abs(iEta)>17");
  hh2 -> SetLineColor(kGreen+2);
  hh1 -> SetLineColor(kRed+1);
  hh2 -> SetTitle("Response in HEndcap |iEta|>17");
 hh2 -> SetAxisRange(0., 5);
 hh2 -> Draw("");
  hh1 -> Draw("same");
  leg = new TLegend(0.68,0.77,0.99,0.91);
  leg -> SetTextSize(0.04);
  leg->AddEntry(hh1,"bef calib","l");
  leg->AddEntry(hh2,"aft calib","l");
  leg -> Draw();
label -> Draw();
 Float_t mean1 = hh1 ->GetMean();
 Float_t mean2 = hh2 ->GetMean();
 Float_t rmsh1 = hh1 ->GetRMS();
 Float_t rmsh2 = hh2 ->GetRMS();
 //cout<<"mean1: "<<mean1<<"   mean2: "<<mean2<<endl;
 m1 = new TText(0.7,0.63, Form("mean1: %1.2f",mean1));  
 r1 = new TText(0.7,0.58, Form(" rms1: %1.2f",rmsh1));  
 m2 = new TText(0.7,0.45, Form("mean2: %1.2f", mean2));
 r2 = new TText(0.7,0.4, Form(" rms2: %1.2f", rmsh2));
 m1 -> SetNDC();
 m2 -> SetNDC();
 r1 -> SetNDC();
 r2 -> SetNDC();
 m1 -> Draw();  
 m2 -> Draw();  
 r1 -> Draw();  
 r2 -> Draw();  
  c1->SaveAs(imgpath+"p07c.png");  
  hh1 -> Delete();
  hh2 -> Delete();


  ftree -> Draw("eClustAfter/eTrack>>hh2", selection&&"abs(iEta)<17");
  ftree -> Draw("eClustAfter/eTrack>>hh3", selection&&"abs(iEta)>17");
  hh2 -> SetLineColor(kYellow+2);
  hh3 -> SetLineColor(kBlue+2);
  hh3 -> SetTitle("Response after calibration");
  hh3 -> SetAxisRange(0., 5);
hh3 -> Draw("");
//Double_t hmax = hh1->GetMaximum();
//hh2->SetMaximum(hmax+ 100);
  hh2 -> Draw("same");
  leg = new TLegend(0.68,0.61,0.99,0.75);
  leg -> SetTextSize(0.04);
  leg->AddEntry(hh2,"HBarrel","l");
  leg->AddEntry(hh3,"HEndcap","l");
  leg -> Draw();
label -> Draw();
  c1->SaveAs(imgpath+"p07d.png");  
  hh2 -> Delete();
  hh3 -> Delete();

  ftree -> Draw("(eClustAfter+eECAL)/eTrack>>hh1", tr_cut&& tr_quality && hit_dist && ptNear&&"abs(iEta)<17", "goff");
  ftree -> Draw("eClustAfter/eTrack>>hh2", tr_cut&& tr_quality && hit_dist && ptNear&&"abs(iEta)<17", "goff");
  ftree -> Draw("eClustAfter/eTrack>>hh3", selection&&"abs(iEta)<17", "goff");
hh1 -> SetLineColor(kGreen+2);
hh2 -> SetLineColor(kBlue+3);
hh3 -> SetLineColor(kBlue);
  hh1 -> SetTitle("Response in HBarrel |iEta| < 17 after calibration");
  //  TH1F * hh1c =  hh1 -> DrawCopy("");
  // hh1c -> SetAxisRange(0.2,5.1);
  hh1 -> SetAxisRange(0., 5);
hh1 -> Draw("");
  hh2 -> Draw("same");
  hh3 -> Draw("same");
  leg = new TLegend(0.65,0.65,0.99,0.92);
  leg -> SetTextSize(0.04);
  leg->AddEntry(hh1,"#splitline{not MIP}{E=E_{had}+ E_{em}}","l");
  leg->AddEntry(hh2,"#splitline{not MIP}{E = E_{had}}","l");
  leg->AddEntry(hh3,"MIP, E_{em}<1","l");
  leg -> Draw();
label -> Draw();
  c1->SaveAs(imgpath+"p07e.png");  
  hh1 -> Delete();
  hh2 -> Delete();
  hh3 -> Delete();


  ftree -> Draw("(eClustAfter+eECAL)/eTrack>>hh1", tr_cut&& tr_quality && hit_dist && ptNear&&"abs(iEta)>17");
  ftree -> Draw("eClustAfter/eTrack>>hh2", tr_cut&& tr_quality && hit_dist && ptNear&&"abs(iEta)>17");
  ftree -> Draw("eClustAfter/eTrack>>hh3", selection&&"abs(iEta)>17");
  hh1 -> SetLineColor(kGreen+2);
hh2 -> SetLineColor(kBlue+3);
hh3 -> SetLineColor(kBlue);
  hh1 -> SetTitle("Response in HEndcap |iEta| > 17 after calibration");
hh1 -> SetAxisRange(0., 5);
  hh1 -> Draw("");
  hh2 -> Draw("same");
  hh3 -> Draw("same");
  leg = new TLegend(0.65,0.65,0.99,0.92);
  leg -> SetTextSize(0.04);
  leg->AddEntry(hh1,"#splitline{not MIP}{E=E_{had}+ E_{em}}","l");
  leg->AddEntry(hh2,"#splitline{not MIP}{E = E_{had}}","l");
  leg->AddEntry(hh3,"MIP, E_{em}<1","l");
  leg -> Draw();
label -> Draw();
  c1->SaveAs(imgpath+"p07f.png");  
  hh1 -> Delete();
  hh2 -> Delete();
  hh3 -> Delete();



  ftree -> Draw("eClustAfter/eTrack:eTrack>>hh1", selection && "abs(iEta) < 17", "prof");
  ftree -> Draw("eClustAfter/eTrack:eTrack>>hh2", selection && "abs(iEta) > 17", "prof");
  hh2 -> SetTitle("eHcalClast/eTrack vs eTrack (After Calib)");
  hh2 -> SetMaximum(2.3);
  hh2 -> SetMinimum(0.3);
 if (sample==10)   {hh2 -> SetMinimum(0.4);}
  hh2 -> SetLineColor(kBlue+2);
  hh1 -> SetLineColor(kGreen+2+2);
  hh1 -> Draw("same");
  leg = new TLegend(0.6,0.55,0.97,0.75);
  leg -> SetTextSize(0.04);
  leg->AddEntry(hh1,"|iEta| < 17","l");
  leg->AddEntry(hh2,"|iEta| > 17","l");
  leg -> Draw();
label -> Draw();
  c1 -> SaveAs(imgpath+"p08.png");  
  hh1 -> Delete();
  hh2 -> Delete();

  ftree -> Draw("eClustBefore/eTrack:eTrack>>hh1", selection, "prof");
  ftree -> Draw("eClustAfter/eTrack:eTrack>>hh2", selection, "prof");
  hh2 -> SetTitle("eHcalClast/eTrack vs eTrack");
  hh2 -> SetLineColor(kGreen+2);
  hh1 -> SetLineColor(kRed+1);
  hh2 -> SetMaximum(2.3);
  hh2 -> SetMinimum(0.3);
 if (sample==10)   {hh2 -> SetMinimum(0.4);}
  hh1 -> Draw("same");
  leg = new TLegend(0.6,0.55,0.97,0.75);
  leg -> SetTextSize(0.04);
  leg->AddEntry(hh1,"Before Calib","l");
  leg->AddEntry(hh2,"After Calib","l");
  leg -> Draw();
label -> Draw();
  c1 -> SaveAs(imgpath+"p09.png");  
  hh1 -> Delete();
  hh2 -> Delete();

  ftree -> Draw("eClustAfter/eTrack:eTrack>>hh1", selection, "prof");
  ftree -> Draw("eClustAfter/eTrack:eTrack>>hh2", tr_cut && tr_quality && hit_dist && ptNear, "prof same");
  ftree -> Draw("(eClustAfter+eECAL)/eTrack:eTrack>>hh3", tr_cut && tr_quality && hit_dist && ptNear, "prof same");
  hh1 -> SetTitle("eHcalClast/eTr vs eTr with and w/o ECAL");
  hh2 -> SetLineColor(kGreen+2);
  hh1 -> SetLineColor(kBlue);
  hh1 -> SetMaximum(2.3);
  hh1 -> SetMinimum(0.3);
  //hh1 -> Draw("same");
  leg = new TLegend(0.6,0.55,0.97,0.75);
  leg -> SetTextSize(0.04);
  leg->AddEntry(hh1, "MIP","l");
  leg->AddEntry(hh3, "E=E_{had}+E_{ecal}","l");
  leg->AddEntry(hh2,"E=E_{had}","l");
  leg -> Draw();
label -> Draw();
  c1 -> SaveAs(imgpath+"p10.png");  
  hh1 -> Delete();
  hh2 -> Delete();
  hh3 -> Delete();

  
  ftree -> Draw("eBeforeDepth2>>hh3", selection && "eBeforeDepth2>0 && abs(iEta)>17");
  ftree -> Draw("eAfterDepth2>>hh4", selection && "eAfterDepth2>0 && abs(iEta)>17");
  ftree -> Draw("eBeforeDepth1>>hh1", selection && "eBeforeDepth1>0 && abs(iEta)>17");
  ftree -> Draw("eAfterDepth1>>hh2", selection && "eAfterDepth1>0 && abs(iEta)>17");
 
  hh1 -> SetLineColor(kRed+1);
  hh2 -> SetLineColor(kGreen+2);
  hh3 -> SetLineColor(kBlue+1);
  hh4 -> SetLineColor(kGreen+5);
hh2 -> SetTitle("Energy in Depths 1 and 2");
hh2 -> SetAxisRange(0., 50);
hh2 ->Draw("");
hh3 ->Draw("same");
hh4 ->Draw("same");
hh1 ->Draw("same");
  leg = new TLegend(0.53,0.45,0.97,0.75);
  leg -> SetTextSize(0.04);
  leg->AddEntry(hh1,"d=1 Before Calib","l");
  leg->AddEntry(hh2,"d=1 After Calib","l");
  leg->AddEntry(hh3,"d=2 Before Calib","l");
  leg->AddEntry(hh4,"d=2 After Calib","l");
  leg -> Draw();
label -> Draw();
  c1->SaveAs(imgpath+"p11.png");  
  hh1 -> Delete();
  hh2 -> Delete();
  hh3 -> Delete();
  hh4 -> Delete();

  
  ftree -> Draw("log10(eAfterDepth1/eAfterDepth2)>>hh1", selection && "eAfterDepth2>1 && eAfterDepth1>1 && abs(iEta)>17", "");
  hh1 -> SetTitle("log10(eDepth1/eDepth2) at |iEta|>17");
  hh1 -> SetLineColor(kBlue);
//  c1 -> SetLogy();
label -> Draw();
  c1->SaveAs(imgpath+"p12.png");  
  hh1 -> Delete();
  c1 -> SetLogy(0);
  
  ftree -> Draw("eAfterDepth1/eAfterDepth2:iEta>>hh1", selection && "eAfterDepth2>0.1 && eAfterDepth1>0.1 && abs(iEta)>13", "prof");
  hh1 -> SetTitle("eDepth1/eDepth2 vs iEta");
  hh1 -> SetLineColor(kBlue);
label -> Draw();
  c1->SaveAs(imgpath+"p13.png");  
  hh1 -> Delete();
  
  ftree -> Draw("eAfterDepth2/eTrack:eAfterDepth1/eTrack>>hh1", selection && "abs(iEta)>17", "colz");
  hh1 -> SetTitle("eD2/p vs eD1/p for |iEta|>17");
  hh1 -> SetXTitle("d1/trP");
  hh1 -> SetYTitle("d2/trP");
label -> Draw();
  c1->SaveAs(imgpath+"p14a.png");  
  hh1 -> Delete();

  ftree -> Draw("iPhi:abs(iEta)>>hh1", selection && "abs(iEta)>17 && eAfterDepth1/eTrack<0.1 && eAfterDepth2/eTrack>0.6", "colz");
  hh1 -> SetTitle("iPhi vs iEta for eD1/p<0.1, eD2/p>0.6");
  hh1 -> SetXTitle("iEta");
  hh1 -> SetYTitle("iPhi");
label -> Draw();
  c1->SaveAs(imgpath+"p14b.png");  
  hh1 -> Delete();
  
  ftree -> Draw("eClustBefore/eTrack>>hh1", selection &&"abs(iEta)>17 && eAfterDepth1/eTrack<0.1 && eAfterDepth2/eTrack>0.6", "hist");
  hh1 -> SetTitle("Response for eD1/p<0.1, eD2/p>0.6");
  hh1 -> SetLineColor(kRed+1);
  hh1 -> Draw();
  ftree -> Draw("eClustAfter/eTrack>>hh2", selection &&"abs(iEta)>17 && eAfterDepth1/eTrack<0.1 && eAfterDepth2/eTrack>0.6", "same");
  hh2 -> SetLineColor(kGreen+2);
  leg = new TLegend(0.68,0.65,0.99,0.8);
  leg -> SetTextSize(0.036);
  leg->AddEntry(hh1,"Before Calib","l");
  leg->AddEntry(hh2,"After Calib","l");
  leg -> Draw();
label -> Draw();
  c1->SaveAs(imgpath+"p14c.png");  
  hh1 -> Delete();
  hh2 -> Delete();

  ftree -> Draw("eventNumber>>hh1", selection);
  hh1 -> SetTitle("Event Number");
  hh1 -> SetLineColor(kBlue);
label -> Draw();
  c1->SaveAs(imgpath+"p15.png");  
  hh1 -> Delete();

  ftree -> Draw("runNumber>>hh1", selection);
  hh1 -> SetTitle("Run Number");
hh1 -> SetLineColor(kBlue);
hh1 -> SetNdivisions(404);

label -> Draw();
  c1->SaveAs(imgpath+"p16.png");  
  hh1 -> Delete();


  ftree -> Draw("eClustAfter/eTrack:iPhi>>hh1", selection&&"abs(iEta)<17 && eClustAfter/eTrack>0.2", "prof");
  ftree -> Draw("eClustBefore/eTrack:iPhi>>hh2", selection&&"abs(iEta)<17 && eClustBefore/eTrack>0.2", "prof");
  hh2 -> SetLineColor(kRed+1);
  hh1 -> SetLineColor(kGreen+2);
  hh2 -> SetMaximum(1.2);
  hh2 -> SetMinimum(0.4);
 if (sample==10)   {hh2 -> SetMinimum(0.4);}
  hh2 -> SetTitle("eHcalClast/eTrack vs iPhi at HB");
hh1 -> Draw("same");
  leg = new TLegend(0.4,0.8,0.75,0.92);
  leg -> SetTextSize(0.04);
  leg->AddEntry(hh2,"Before Calib","l");
  leg->AddEntry(hh1,"After Calib","l");
  leg -> Draw();
label -> Draw();
  c1 -> SaveAs(imgpath+"p17.png");  
  hh1 -> Delete();
  hh2 -> Delete();


  ftree -> Draw("eClustAfter/eTrack:iEta>>hh1", selection&&"eClustAfter/eTrack>0.2", "prof");
  ftree -> Draw("eClustBefore/eTrack:iEta>>hh2", selection, "prof");
  hh2 -> SetMaximum(1.2);
  hh2 -> SetMinimum(0.4);
 if (sample==10)   {hh2 -> SetMinimum(0.4);}
  hh2 -> SetTitle("eHcalClast/eTrack vs iEta");
  hh2 -> SetLineColor(kRed+1);
  hh1 -> SetLineColor(kGreen+2);
  hh1 -> Draw("same");
  leg = new TLegend(0.4,0.8,0.75,0.92);
  leg -> SetTextSize(0.04);
  leg->AddEntry(hh2,"Before Calib","l");
  leg->AddEntry(hh1,"After Calib","l");
  leg -> Draw();
label -> Draw();
  c1 -> SaveAs(imgpath+"p18.png");  
  hh1 -> Delete();
  hh2  -> Delete();

  ftree -> Draw("eClustAfter/eTrack:iPhi>>hh1", selection && "abs(iEta) < 20", "prof");
  ftree -> Draw("eClustAfter/eTrack:iPhi>>hh2", selection && "abs(iEta) > 20", "prof");
  hh2 -> SetTitle("eHcalClast/eTrack vs iPhi (After Calib)");
  hh2 -> SetMaximum(1.2);
  hh2 -> SetMinimum(0.4);
 if (sample==10)   {hh2 -> SetMinimum(0.4);}
  hh2 -> SetLineColor(kBlue);
  hh1 -> SetLineColor(kGreen+2);
  hh1 -> Draw("same");
  leg = new TLegend(0.4,0.8,0.75,0.92);
  leg -> SetTextSize(0.04);
  leg->AddEntry(hh1,"|iEta| < 20","l");
  leg->AddEntry(hh2,"|iEta| > 20","l");
  leg -> Draw();
label -> Draw();
  c1 -> SaveAs(imgpath+"p19.png");  
  hh1 -> Delete();
  hh2 -> Delete();


  ftree -> Draw("eClustAfter/eTrack:iPhi>>hh2", selection && "iPhi%4==3 && abs(iEta)<20", "prof");
  ftree -> Draw("eClustAfter/eTrack:iPhi>>hh1", selection && "iPhi%4==0&& abs(iEta)<20", "prof");
  hh1 -> SetLineColor(kBlue);
  hh2 -> SetLineColor(kGreen+3);
  hh1 -> SetMaximum(1.2);
  hh1 -> SetMinimum(0.4);
 if (sample==10)   {hh1 -> SetMinimum(0.4);}
  hh1 -> SetTitle("eHcalClast/eTrack vs iPhi, |iEta|<20");
  hh2 -> Draw("same");
  leg = new TLegend(0.4,0.8,0.75,0.92);
  leg -> SetTextSize(0.04);
  leg->AddEntry(hh1,"iPhi%4=0","l");
  leg->AddEntry(hh2,"iPhi%4=3","l");
  leg -> Draw();
label -> Draw();
  c1 -> SaveAs(imgpath+"p20.png");  
  hh1 -> Delete();
  hh2 -> Delete();

  ftree -> Draw("eClustAfter/eTrack:iPhi>>hh1", selection && "abs(iEta)>20", "prof");
  ftree -> Draw("eClustBefore/eTrack:iPhi>>hh2", selection && "abs(iEta)>20", "prof");
  hh1 -> SetLineColor(kBlue+1);
  hh2 -> SetLineColor(kRed+1);
  hh1 -> SetMaximum(1.2);
  hh1 -> SetMinimum(0.4);
 if (sample==10)   {hh1 -> SetMinimum(0.4);}
  hh1 -> SetTitle("eHcalClast/eTrack vs iPhi, |iEta|>20");
hh1 -> Draw();
hh2 -> Draw("same");
  leg = new TLegend(0.4,0.8,0.75,0.92);
  leg -> SetTextSize(0.04);
  leg->AddEntry(hh1,"after calib","l");
  leg->AddEntry(hh2,"before calib","l");
  leg -> Draw();
label -> Draw();
  c1 -> SaveAs(imgpath+"p21.png");  
  hh1 -> Delete();
  hh2 -> Delete();

  c1 -> SetLogy();
  ftree -> Draw("e3x3After>>hh1", selection&&"e3x3After<100", "");
  hh1 -> SetLineColor(kBlue+1);
  hh1 -> SetTitle("e3x3");
label -> Draw();
  c1 -> SaveAs(imgpath+"p22.png");  
  hh1 -> Delete();


  ftree -> Draw("e3x3Before/eClustBefore>>hh1", selection && "eClustBefore>15", "");
  ftree -> Draw("e3x3After/eClustAfter>>hh2", selection && "eClustAfter>15", "");
  hh1 -> SetLineColor(kRed+1);
  hh2 -> SetLineColor(kGreen+2);
  hh2 -> SetTitle("e3x3/eHcalClast");
  hh1 -> Draw("same");
label -> Draw();
  c1 -> SaveAs(imgpath+"p23.png");  
  hh1 -> Delete();
  hh2 -> Delete();
  c1 -> SetLogy(0);


  ftree -> Draw("(e5x5After-e3x3After)/eTrack:eTrack>>hh2", selection && "abs(iEta)<17", "prof");
  ftree -> Draw("(e5x5After-e3x3After)/eTrack:eTrack>>hh1", selection && "abs(iEta)>17", "prof");
  hh1 -> SetLineColor(kBlue);
  hh2 -> SetLineColor(kGreen+2);
  hh1 -> SetTitle("(e5x5-e3x3)/eTrack vs eTrack");
  hh1 -> SetMaximum(0.3);
  if (sample=10) {  hh1 -> SetMaximum(0.4);}
hh2 -> Draw("same");
  leg = new TLegend(0.65,0.55,0.97,0.75);
  leg -> SetTextSize(0.04);
  leg->AddEntry(hh1,"|iEta| > 17","l");
  leg->AddEntry(hh2,"|iEta| < 17","l");
  leg -> Draw();
label -> Draw();
  c1 -> SaveAs(imgpath+"p24.png");  
  hh1 -> Delete();
  hh2 -> Delete();

  c1 -> SetLogy();
//  ftree -> Draw("eCentHitAfter/eClustAfter>>hh1", selection && "eClustAfter>10", "");
  ftree -> Draw("eCentHitBefore/eClustBefore>>hh1", selection && "eClustBefore>3", "");
  hh1 -> SetLineColor(kBlue+1);
  hh1 -> SetTitle("eCentral/eClust");
label -> Draw();
  c1 -> SaveAs(imgpath+"p25.png");  
  hh1 -> Delete();
  c1 -> SetLogy(0);

  ftree -> Draw("(eClustAfter-eCentHitAfter)/eTrack:eTrack>>hh2", selection && "abs(iEta) < 17", "prof");
  ftree -> Draw("(eClustAfter-eCentHitAfter)/eTrack:eTrack>>hh1", selection && "abs(iEta) > 17", "prof");
  hh1 -> SetLineColor(kBlue);
  hh2 -> SetLineColor(kGreen+2);
  hh1 -> SetTitle("(eClust-eCentHit)/eTrack vs eTrack");
  hh1 -> SetMaximum(0.9);
  hh2 -> Draw("same");
  leg = new TLegend(0.65,0.55,0.97,0.75);
  leg -> SetTextSize(0.04);
  leg->AddEntry(hh1,"|iEta| < 17","l");
  leg->AddEntry(hh2,"|iEta| > 17","l");
  leg -> Draw();
label -> Draw();
  c1 -> SaveAs(imgpath+"p26.png");  
  hh1 -> Delete();
  hh2 -> Delete();


  ftree -> Draw("(e5x5After-e3x3After)/eTrack:iEta>>hh1", selection, "prof");
  hh1 -> SetLineColor(kGreen+2);
  hh1 -> SetTitle("(e5x5-e3x3)/eTrack vs iEta");
  hh1 -> SetMaximum(0.4);
label -> Draw();
  c1 -> SaveAs(imgpath+"p27.png");  
  hh1 -> Delete();

  ftree -> Draw("(eClustAfter-eCentHitAfter)/eTrack:iEta>>hh1", selection, "prof");
  hh1 -> SetLineColor(kGreen+2);
  hh1 -> SetTitle("(eClust-eCentHit)/eTrack vs iEta");
  hh1 -> SetMaximum(0.6);
label -> Draw();
  c1 -> SaveAs(imgpath+"p28.png");  
  hh1 -> Delete();


  ftree -> Draw("delR>>hh1", selection, "");
  hh1 -> SetLineColor(kGreen+2);
  hh1 -> SetTitle("dR(#eta,#phi) maxHit - assosiator point");
  //  hh1 -> SetMaximum(0.6);
label -> Draw();
  c1 -> SaveAs(imgpath+"p29.png");  
  hh1 -> Delete();

  ftree -> Draw("(e5x5After-eClustAfter)/eTrack>>hh1", selection, "");
  hh1 -> SetLineColor(kBlue);
  hh1 -> SetTitle("(e5x5-eClust)/eTrack");
label -> Draw();
  c1 -> SaveAs(imgpath+"p30.png");  
  hh1 -> Delete();
  

/*  
//Uncomment this if you want these plots. (takes long time to plot)
 TH1F * rms1 = new TH1F("rms1","rms", 51, -25, 25);
 TH1F * rms2 = new TH1F("rms2","rms", 51, -25, 25);

 Float_t imean;
 Float_t irms;

  for (Int_t i=-22; i<23; i++)
    {
      if (i==0) continue;
      TCut myiEta = Form("iEta==%i",i);
     
      ftree -> Draw("eClustAfter/eTrack>>hh1", selection && myiEta, "goff");
      ftree -> Draw("eClustBefore/eTrack>>hh2", selection && myiEta, "goff");
      irms = hh1->GetRMS();
      rms1->Fill(i,irms);           
      irms = hh2->GetRMS();
      rms2->Fill(i,irms);     

    }

 rms1 -> SetTitle("rms(Resp) vs iEta");
 rms1 -> SetMarkerColor(kBlue);
 rms2 -> SetMarkerColor(kRed+1);
 rms1 -> SetMarkerStyle(23);
 rms2 -> SetMarkerStyle(27);
rms1 -> SetMaximum(0.3);//0.5 for 10 GeV
 rms1 -> Draw("p");
 rms2 -> Draw("p same");
label -> Draw();
  c1 -> SaveAs(imgpath+"p31.png");  
rms1 -> Delete();
rms2 -> Delete();


 TH1F * rms1 = new TH1F("rms1","rms", 51, -25, 25);
 TH1F * rms2 = new TH1F("rms2","rms", 51, -25, 25);


  for (Int_t i=-22; i<23; i++)
    {
      if (i==0) continue;
      TCut myiEta = Form("iEta==%i",i);     
      ftree -> Draw("eClustAfter/eTrack>>hh1", selection && myiEta, "goff");
      ftree -> Draw("eClustBefore/eTrack>>hh2", selection && myiEta, "goff");
      imean = hh1->GetMean();
      irms = hh1->GetRMS();
      rms1 -> Fill(i,irms/imean);           
      imean = hh2->GetMean();
      irms = hh2->GetRMS();
      rms2-> Fill(i,irms/imean);     

    }

 rms1 -> SetTitle("rms(Resp)/mean vs iEta");
 rms1 -> SetMarkerColor(kBlue);
 rms2 -> SetMarkerColor(kRed+1);
 rms1 -> SetMarkerStyle(23);
 rms2 -> SetMarkerStyle(27);
rms1 -> SetMaximum(0.3);//0.5 for 10 GeV
 rms1 -> Draw("p");
 rms2 -> Draw("p same");
label -> Draw();
  c1 -> SaveAs(imgpath+"p32.png");  

    hh1 -> Delete();
   hh2 -> Delete();
*/

  ftree -> Draw("iTime>>hh2", selection&&"abs(iEta)>17", "");
 ftree -> Draw("iTime>>hh1", selection&&"abs(iEta)<15", "same");
//hh1 -> SetAxisRange(-20, 20);
//hh2 -> SetMaximum(hh1->GetMaximum()+10);

  //hh1 -> SetLineColor(kBlue+1);
  hh2 -> SetTitle("time in central rechit");
  hh1 -> SetLineColor(kOrange+3);
  hh2 -> SetLineColor(kBlue+2);
  leg = new TLegend(0.7,0.7,0.99,0.83);
  leg -> SetTextSize(0.04);
  leg->AddEntry(hh1,"HBarrel","l");
  leg->AddEntry(hh2,"HEndcap","l");
  leg -> Draw();
label -> Draw();
  c1 -> SaveAs(imgpath+"p33.png");  
  hh1 -> Delete();
  hh2 -> Delete();


  ftree -> Draw("HTime>>hh1", selection&&"Iteration$<numHits && abs(iEta)<15 && abs(HTime)<100");
  ftree -> Draw("iTime>>hh2", selection&&"abs(iEta)<15", "same");
//hh1 -> SetMaximum(hh2->GetMaximum()+10);
  hh1 -> SetTitle("time in HBarrel");
  hh2 -> SetLineColor(kBlue+1);
  leg = new TLegend(0.5,0.7,0.99,0.83);
  leg -> SetTextSize(0.04);
  leg->AddEntry(hh1,"rechits in cluster","l");
  leg->AddEntry(hh2,"central hit","l");
  leg -> Draw();
label -> Draw();
  c1 -> SaveAs(imgpath+"p34.png");  
  hh1 -> Delete();
  hh2 -> Delete();

  ftree -> Draw("HTime>>hh1", selection&&"Iteration$<numHits && abs(iEta)>17 && abs(HTime)<100", "");
  ftree -> Draw("iTime>>hh2", selection&&"abs(iEta)>17 ", "same");
//hh1 -> SetMaximum(hh2->GetMaximum()+10);
  hh2 -> SetLineColor(kBlue+1);
  hh1 -> SetTitle("time in HEndcap");
  leg = new TLegend(0.5,0.7,0.99,0.83);
  leg -> SetTextSize(0.04);
  leg->AddEntry(hh1,"rechits in cluster","l");
  leg->AddEntry(hh2,"central hit","l");
  leg -> Draw();
label -> Draw();
  c1 -> SaveAs(imgpath+"p35.png");  
  hh1 -> Delete();
  hh2 -> Delete();


  c1 -> SetLogy();
  ftree -> Draw("PtNearBy>>hh1", "", "");
  hh1 -> SetTitle("pt near");
label -> Draw();
  c1 -> SaveAs(imgpath+"p36.png");  
  hh1 -> Delete();
  c1 -> SetLogy(0);

  ftree -> Draw("numValidTrkHits>>hh1", tr_cut && mip_cut && hit_dist, "");
//hh1 -> SetMaximum(hh2->GetMaximum()+10);
  hh1 -> SetTitle("track pattern: numberOfValidHits()");
label -> Draw();
  c1 -> SaveAs(imgpath+"p37.png");  
  hh1 -> Delete();

  c1 -> SetLogy(1);

  ftree -> Draw("numValidTrkStrips>>hh1", tr_cut && mip_cut && hit_dist &&"iEta<17", "");
  hh1 -> SetTitle("track pattern: numberOfValidStripTECHits(). HB");
  hh1 -> SetLineColor(kRed+1);
label -> Draw();
  c1 -> SaveAs(imgpath+"p38.png");  
  hh1 -> Delete();


  ftree -> Draw("numValidTrkStrips>>hh1", tr_cut && mip_cut && hit_dist && "iEta>18", "");
  hh1 -> SetTitle("track pattern: numberOfValidStripTECHits(). HE");
  hh1 -> SetLineColor(kBlue+1);
label -> Draw();
  c1 -> SaveAs(imgpath+"p39.png");  
  hh1 -> Delete();

  ftree -> Draw("sqrt((abs(iEta-iEtaTr)+abs(iPhi-iPhiTr)))>>hh1", selection, "");
  //  ftree -> Draw("sqrt(abs(iEta-iEtaTr)+abs(iPhi-iPhiTr))>>hh2", selection, "");
  // ftree -> Draw("abs(iEta-iEtaTr)>>hh3", selection, "same");
  hh1 -> SetLineColor(kGreen+2);
  //  hh2 -> SetLineColor(kRed);
  //hh3 -> SetLineColor(kBlue+2);
  hh1 -> SetTitle("maxHit - trackHit");
  //  hh1 -> SetMaximum(0.6);
label -> Draw();
  c1 -> SaveAs(imgpath+"p40.png");  
  hh1 -> Delete();

  ftree -> Draw("iDr>>hh1", tr_cut && mip_cut && tr_quality && ptNear, "");
//  ftree -> Draw("sqrt(dietatr*dietatr+diphitr*diphitr)>>hh1", tr_cut && mip_cut && tr_quality && 
//ptNear, "");
  hh1 -> SetLineColor(kGreen+2);
  hh1 -> SetTitle("maxHit - trackHit distance");
  hh1 -> SetXTitle("#sqrt{#Deltai#eta^2 + #Deltai#phi^2}");
label -> Draw();
  c1 -> SaveAs(imgpath+"p41.png");  
  hh1 -> Delete();

c1 -> SetLogy(0);

  ftree -> Draw("etaTrack>>hh1", selection, "");
  hh1 -> SetTitle("eta Track");
label -> Draw();
   c1->SaveAs(imgpath+"p42.png");  
   hh1 -> Delete();

  ftree -> Draw("phiTrack>>hh1", selection, "");
  hh1 -> SetTitle("phi Track");
label -> Draw();
   c1->SaveAs(imgpath+"p43.png");  
   hh1 -> Delete();


  ftree -> Draw("eECAL:iEta>>hh1", selection, "lego");
//  ftree -> Draw("eECAL>>hh1", selection&&"iEta>17", "");
  hh1 -> SetTitle("ecal energy cs iEta");
  hh1 -> SetYTitle("energy");
  hh1 -> SetXTitle("iEta");
label -> Draw();
   c1->SaveAs(imgpath+"p44.png");  
   hh1 -> Delete();

  c1 -> SetLogy();
  ftree -> Draw("eECAL>>hh1", selection&&"abs(iEta)>17", "");
 ftree -> Draw("eECAL>>hh2", selection&&"abs(iEta)<17", "same");
  hh1 -> SetTitle("ecal energy vs iEta");
  hh1 -> SetXTitle("energy");
  hh1 -> SetLineColor(kGreen+2);
  hh2 -> SetLineColor(kRed+2);
  leg = new TLegend(0.6,0.7,0.99,0.83);
  leg -> SetTextSize(0.04);
  leg->AddEntry(hh2,"|iEta|<17","l");
  leg->AddEntry(hh1,"|iEta|>17","l");
  leg -> Draw();
label -> Draw();
   c1->SaveAs(imgpath+"p45.png");  
   hh1 -> Delete();
   hh2 -> Delete();
  c1 -> SetLogy(0);


  ftree -> Draw("iEta:iPhi>>hh1", selection&&"abs(iEta)<20", "colz");
  hh1 -> SetTitle("iEta-iPhi occupancy");
  hh1 -> SetYTitle("iEta");
  hh1 -> SetXTitle("iPhi");
label -> Draw();
   c1->SaveAs(imgpath+"p46.png");  
   hh1 -> Delete();

  ftree -> Draw("iEta:iPhi>>hh1", selection&&"abs(iEta)>20", "colz");
  hh1 -> SetTitle("iEta-iPhi occupancy");
  hh1 -> SetYTitle("iEta");
  hh1 -> SetXTitle("iPhi");
label -> Draw();
   c1->SaveAs(imgpath+"p47.png");  
   hh1 -> Delete();

  ftree -> Draw("eCentHitAfter:iPhi>>hh1", selection&&"abs(iEta)==22", "prof");
  hh1 -> SetTitle("eCentralHit vs iPhi for |iEta|=22");
  hh1 -> SetYTitle("energy");
  hh1 -> SetXTitle("iPhi");
  hh1 -> SetMinimum(0);
  hh1 -> SetMaximum(20);
label -> Draw();
   c1->SaveAs(imgpath+"p48.png");  
   hh1 -> Delete();


  ftree -> Draw("eClustAfter/eTrack:iPhi>>hh1", selection&&"abs(iEta)==21", "prof");
  ftree -> Draw("eClustAfter/eTrack:iPhi>>hh2", selection&&"abs(iEta)==22", "prof same");
  ftree -> Draw("eClustAfter/eTrack:iPhi>>hh3", selection&&"abs(iEta)==23", "prof same");
  ftree -> Draw("eClustAfter/eTrack:iPhi>>hh4", selection&&"abs(iEta)==24", "prof same");
  hh1 -> SetTitle("Response vs iPhi");
  hh1 -> SetYTitle("Response");
  hh1 -> SetXTitle("iPhi");
  hh1 -> SetMaximum(1.5);
  hh1 -> SetMinimum(0.2);
  hh1 -> SetLineColor(kRed);
  hh2 -> SetLineColor(kGreen+2);
  hh3 -> SetLineColor(kBlue+2);
  hh4 -> SetLineColor(kYellow+2);

leg = new TLegend(0.7,0.75,0.99,0.93);
  leg -> SetTextSize(0.04);
  leg->AddEntry(hh1,"|iEta|=21","l");
  leg->AddEntry(hh2,"|iEta|=22","l");
  leg->AddEntry(hh3,"|iEta|=23","l");
  leg->AddEntry(hh4,"|iEta|=24","l");
  leg -> Draw();

label -> Draw();
 c1->SaveAs(imgpath+"p49.png");  
 hh1 -> Delete();
 hh2 -> Delete();
 hh3 -> Delete();
 hh4 -> Delete();


  ftree -> Draw("eClustAfter/eTrack:iPhi>>hh1(72,0,72)", selection&&"iEta==21", "prof");
  ftree -> Draw("eClustAfter/eTrack:iPhi>>hh2(72,0,72)", selection&&"iEta==22", "prof same");
  ftree -> Draw("eClustAfter/eTrack:iPhi>>hh3(72,0,72)", selection&&"iEta==-21", "prof same");
  ftree -> Draw("eClustAfter/eTrack:iPhi>>hh4(72,0,72)", selection&&"iEta==-22", "prof same");
  hh1 -> SetTitle("Response vs iPhi");
  hh1 -> SetYTitle("Response");
  hh1 -> SetXTitle("iPhi");
  hh1 -> SetMaximum(1.5);
  hh1 -> SetMinimum(0.2);
  hh1 -> SetLineColor(kRed);
  hh2 -> SetLineColor(kGreen+2);
  hh3 -> SetLineColor(kBlue+2);
  hh4 -> SetLineColor(kYellow+2);

leg = new TLegend(0.7,0.75,0.99,0.93);
  leg -> SetTextSize(0.04);
  leg->AddEntry(hh1,"iEta=21","l");
  leg->AddEntry(hh2,"iEta=22","l");
  leg->AddEntry(hh3,"iEta=-21","l");
  leg->AddEntry(hh4,"iEta=-22","l");
  leg -> Draw();

label -> Draw();
 c1->SaveAs(imgpath+"p50.png");  
 hh1 -> Delete();
 hh2 -> Delete();
 hh3 -> Delete();
 hh4 -> Delete();


  tf->Close();
}

