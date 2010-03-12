{
  gStyle -> SetOptStat(1);
  gStyle->SetPalette(1);
  gStyle->SetPadGridX(1);
  gStyle->SetPadGridY(1);
  //gStyle -> SetTitleFont(82);
  //  gStyle -> SetTextFont(102);
Int_t sample=0;

//TString imgpath("~/afs/public_html/test/");  
 TString imgpath("~/afs/public_html/pfcorrs/v01/");  

 TFile* fMC = new TFile("~/nobackup/arch/hcalCorrPFv9_35cm.root","OPEN");


  TCut tr_cut = "eTrack>45 && eTrack<55";  label = new TText(0.03,0.2, "Pions 50 GeV "); sample=50;
  
  label -> SetNDC(); label->SetTextAlign(11); label->SetTextSize(0.05); label->SetTextAngle(90); label->SetTextColor(kRed);

  TCut trqual = "trkQual==1 && numValidTrkHits>13 && (abs(iEta)<17 || numValidTrkStrips>9)";
  TCut mip_cut = "eECAL<1.";
  //TCut neutral_iso_cut ="(eECAL40cm-eECAL09cm)<8";
  TCut neutral_iso_cut ="";
  TCut hit_dist = "iDr<1.5";
  // TCut hit_dist = "delR<18";
  TCut mc_dist = "delRmc<3 || abs(iEta)>23";
  TCut low_resp_cut = "eHcalCone/eTrack>0.2";
  TCut maxPNearBy = "";
  //TCut maxPNearBy = "maxPNearBy<2";
  
  TCut selection = trqual && neutral_iso_cut && tr_cut && mip_cut && hit_dist && mc_dist && maxPNearBy && low_resp_cut;
  
  TTree* ftreeMC = (TTree*)fMC->Get("hcalPFcorrs/pfTree");
  
  
  TCanvas* c1 = new TCanvas("c1","all",0,0,350,350);
  c1-> cd();
  
  c1 -> SetLogy();
  
  
  ftreeMC -> Draw("eTrack>>hMC", selection);
  // hMC -> SetAxisRange(-, 55);
  hMC -> SetTitle("eTrack");
  hMC -> SetFillColor(kRed-10);
  label -> Draw();
  c1->SaveAs(imgpath+"p01.png");
  hMC->Delete();


  ftreeMC -> Draw("iEta>>hMC", selection);
  hMC -> SetTitle("iEta");
  hMC -> SetFillColor(kRed-10);
  label -> Draw();
  c1->SaveAs(imgpath+"p02.png");
  hMC->Delete();

  ftreeMC -> Draw("eECAL>>hMC", tr_cut && trqual, "hist");
  hMC -> SetTitle("EM energy");
  hMC -> SetFillColor(kRed-10);
  label -> Draw();
  c1->SaveAs(imgpath+"p03.png");  
  hMC->Delete();

  c1 -> SetLogy(0);

 ftreeMC -> Draw("iPhi>>hMC", selection&&"abs(iEta)<17");
 hMC -> SetTitle("iPhi at |iEta|<17)");
 hMC -> SetFillColor(kRed-10);
 label -> Draw();
 c1->SaveAs(imgpath+"p04.png");  
 hMC->Delete();
 

 ftreeMC -> Draw("iPhi>>hMC", selection&&"abs(iEta)>=17 && abs(iEta)<21");
 hMC -> SetTitle("iPhi at 17<=|iEta|<21)");
 hMC -> SetFillColor(kRed-10);
 label -> Draw();
 c1->SaveAs(imgpath+"p05.png");  
 hMC->Delete();
 

 ftreeMC -> Draw("iPhi>>hMC", selection&&"abs(iEta)>=21&&abs(iEta)<29");
 hMC -> SetTitle("iPhi at 21<=|iEta|<29");
 hMC -> SetFillColor(kRed-10);
 label -> Draw();
 c1->SaveAs(imgpath+"p06.png");  
 hMC->Delete();

 ftreeMC -> Draw("iPhi>>hMC", selection&&"abs(iEta)>=29");
 hMC -> SetTitle("iPhi at |iEta|>=29");
 hMC -> SetFillColor(kRed-10);
 label -> Draw();
 c1->SaveAs(imgpath+"p07.png");  
 hMC->Delete();
 


 ftreeMC -> Draw("UsedCells>>hMC", selection&&"abs(iEta)<17");
 hMC -> SetBins(20, 0,20);
 hMC -> SetTitle("Number of Hits, HB (|iEta|<17)");
 hMC -> SetFillColor(kRed-10);
 label -> Draw();
 hMC -> SetNdivisions(110);
 c1->SaveAs(imgpath+"p08.png");  
 hMC->Delete();
 

 ftreeMC -> Draw("UsedCells>>hMC", selection&&"abs(iEta)>=17 && abs(iEta)<21");
 hMC -> SetBins(20, 0,20);
 hMC -> SetTitle("Number of Hits, HE (17<=|iEta|<21)");
 hMC -> SetFillColor(kRed-10);
 label -> Draw();
 hMC -> SetNdivisions(110);
 c1->SaveAs(imgpath+"p09.png");  
 hMC->Delete();
 

 ftreeMC -> Draw("UsedCells>>hMC", selection&&"abs(iEta)>=21&&abs(iEta)<29");
 hMC -> SetBins(20, 0,20);
 hMC -> SetTitle("Number of Hits, HB (21<=|iEta|<29)");
 hMC -> SetFillColor(kRed-10);
 label -> Draw();
 hMC -> SetNdivisions(110);
 c1->SaveAs(imgpath+"p10.png");  
 hMC->Delete();

 ftreeMC -> Draw("UsedCells>>hMC", selection&&"abs(iEta)>=29");
 hMC -> SetBins(20, 0,20);
 hMC -> SetTitle("Number of Hits, HF (|iEta|>=29)");
 hMC -> SetFillColor(kRed-10);
 label -> Draw();
 hMC -> SetNdivisions(110);
 c1->SaveAs(imgpath+"p11.png");  
 hMC->Delete();
 
  ftreeMC -> Draw("eHcalCone>>hMC", selection);
  hMC -> SetTitle("eHcal in cone");
  hMC -> SetFillColor(kRed-10);
  //  hMC -> SetAxisRange(0.,50.);
  label -> Draw();
  c1->SaveAs(imgpath+"p12.png");  
  hMC -> Delete();


  ftreeMC -> Draw("eHcalCone/eTrack>>hMC", selection&&"abs(iEta)<17");
  hMC -> SetTitle("eHcal/eTrack at |iEta|<17");
  hMC -> SetFillColor(kRed-10);
  label -> Draw();
  c1->SaveAs(imgpath+"p13.png");  
  hMC -> Delete();


  ftreeMC -> Draw("eHcalCone/eTrack>>hMC", selection&&"abs(iEta)>=17 && abs(iEta)<21");
  hMC -> SetTitle("eHcal/eTrack at 17<=|iEta|<21");
  hMC -> SetFillColor(kRed-10);
  label -> Draw();
  c1->SaveAs(imgpath+"p14.png");  
  hMC -> Delete();


  ftreeMC -> Draw("eHcalCone/eTrack>>hMC", selection&&"abs(iEta)>=21 && abs(iEta)<29");
  hMC -> SetTitle("eHcal/eTrack at 21<=|iEta|<29");
  hMC -> SetFillColor(kRed-10);
  label -> Draw();
  c1->SaveAs(imgpath+"p15.png");  
  hMC -> Delete();


  ftreeMC -> Draw("eHcalCone/eTrack>>hMC", selection&&"abs(iEta)>=29");
  hMC -> SetTitle("eHcal/eTrack at |iEta|>=29");
  hMC -> SetFillColor(kRed-10);
  label -> Draw();
  c1->SaveAs(imgpath+"p16.png");  
  hMC -> Delete();

  ftreeMC -> Draw("eHcalCone/eTrack:iEta>>hMC", selection, "prof");
  hMC -> SetTitle("eHcal/eTrack vs iEta");
  hMC -> SetFillColor(kRed-10);
hMC -> SetMaximum(1.2);
  hMC -> SetMinimum(0.1);
  label -> Draw();
  c1->SaveAs(imgpath+"p16.png");  
  hMC -> Delete();

  ftreeMC -> Draw("eHcalCone/eTrack:iPhi>>hMC", selection, "prof");
  hMC -> SetTitle("eHcal/eTrack vs iPhi");
hMC -> SetMaximum(1.2);
  hMC -> SetMinimum(0.1);

  hMC -> SetFillColor(kRed-10);
  label -> Draw();
  c1->SaveAs(imgpath+"p17.png");  
  hMC -> Delete();

  fMC->Close();


}
