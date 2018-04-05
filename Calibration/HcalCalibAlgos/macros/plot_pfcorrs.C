{
  gStyle -> SetOptStat(1);
  gStyle->SetPalette(1);
  gStyle->SetPadGridX(1);
  gStyle->SetPadGridY(1);
  //gStyle -> SetTitleFont(82);
  //  gStyle -> SetTextFont(102);
Int_t sample=0;

//TString imgpath("~/afs/public_html/test/");  
 TString imgpath("~/afs/public_html/pfcorrs/v02/");  

 //TFile* fMC = new TFile("./HcalCorrPF.root","OPEN");
  TFile* fMC = new TFile("~/nobackup/arch/hcalCorrPFv13_35cm.root","OPEN");


 TCut tr_cut = "eParticle>45 && eParticle<55";  label = new TText(0.03,0.2, "Pions 50 GeV "); sample=50;
  
  label -> SetNDC(); label->SetTextAlign(11); label->SetTextSize(0.05); label->SetTextAngle(90); label->SetTextColor(kRed);

  TCut trkQual = "";
  //  TCut trkQual = "(trkQual[0]==1 && abs(etaParticle)<2.4) || abs(etaParticle)>2.4";
  TCut mip_cut = "eECAL09cm<1.";
    TCut hit_dist = "delR<16";
  TCut mc_dist = "";
  // TCut mc_dist = "(abs(etaParticle)<2.4 && delRmc[0]<2) || abs(etaParticle)>2.4";
  //TCut low_resp_cut = "";
  TCut low_resp_cut = "eHcalCone/eParticle>0.2";
  TCut maxPNearBy = "";
  //  TCut neutral_iso_cut ="";
  TCut neutral_iso_cut = "(abs(iEta)<=13&&(eECAL40cm-eECAL09cm)<6.8)||(abs(iEta)==14&&(eECAL40cm-eECAL09cm)<6.6)||(abs(iEta)==15&&(eECAL40cm-eECAL09cm)<6.6)||(abs(iEta)==16&&(eECAL40cm-eECAL09cm)<9.8)||(abs(iEta)==17&&(eECAL40cm-eECAL09cm)<10.4)||(abs(iEta)==18&&(eECAL40cm-eECAL09cm)<9.8)||(abs(iEta)==19&&(eECAL40cm-eECAL09cm)<11.0)||(abs(iEta)==20&&(eECAL40cm-eECAL09cm)<12.3)||(abs(iEta)==21&&(eECAL40cm-eECAL09cm)<13.6)||(abs(iEta)==22&&(eECAL40cm-eECAL09cm)<15.2)||(abs(iEta)==23&&(eECAL40cm-eECAL09cm)<15.4)||(abs(iEta)==24&&(eECAL40cm-eECAL09cm)<16.3)||(abs(iEta)==25&&(eECAL40cm-eECAL09cm)<16.1)||(abs(iEta)==26&&(eECAL40cm-eECAL09cm)<15.4)||(abs(iEta)==27&&(eECAL40cm-eECAL09cm)<15.4)||abs(iEta)>27";
  
  TCut selection = trkQual && neutral_iso_cut && tr_cut && mip_cut && hit_dist && mc_dist && maxPNearBy && low_resp_cut;
  
  TTree* ftreeMC = (TTree*)fMC->Get("hcalPFcorrs/pfTree");
  
  
  TCanvas* c1 = new TCanvas("c1","all",0,0,350,350);
  c1-> cd();
  
  c1 -> SetLogy();
  
  
  ftreeMC -> Draw("eParticle>>hMC", selection&&"abs(iEta)>29");
  // hMC -> SetAxisRange(-, 55);
  hMC -> SetTitle("eParticle");
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

  ftreeMC -> Draw("eECAL>>hMC", tr_cut && trkQual, "hist");
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


  ftreeMC -> Draw("eHcalCone/eParticle>>hMC", selection&&"abs(iEta)<17");
  hMC -> SetTitle("eHcal/eParticle at |iEta|<17");
  hMC -> SetFillColor(kRed-10);
  label -> Draw();
  c1->SaveAs(imgpath+"p13.png");  
  hMC -> Delete();


  ftreeMC -> Draw("eHcalCone/eParticle>>hMC", selection&&"abs(iEta)>=17 && abs(iEta)<21");
  hMC -> SetTitle("eHcal/eParticle at 17<=|iEta|<21");
  hMC -> SetFillColor(kRed-10);
  label -> Draw();
  c1->SaveAs(imgpath+"p14.png");  
  hMC -> Delete();


  ftreeMC -> Draw("eHcalCone/eParticle>>hMC", selection&&"abs(iEta)>=21 && abs(iEta)<29");
  hMC -> SetTitle("eHcal/eParticle at 21<=|iEta|<29");
  hMC -> SetFillColor(kRed-10);
  label -> Draw();
  c1->SaveAs(imgpath+"p15.png");  
  hMC -> Delete();


  ftreeMC -> Draw("eHcalCone/eParticle>>hMC", selection&&"abs(iEta)>=29");
  hMC -> SetTitle("eHcal/eParticle at |iEta|>=29");
  hMC -> SetFillColor(kRed-10);
  label -> Draw();
  c1->SaveAs(imgpath+"p16.png");  
  hMC -> Delete();

  ftreeMC -> Draw("eHcalCone/eParticle:iEta>>hMC(83,-42.5,42.5)", selection, "prof");
  hMC -> SetTitle("eHcal/eParticle vs iEta");
  hMC -> SetFillColor(kRed-10);
hMC -> SetMaximum(1.2);
  hMC -> SetMinimum(0.1);
  label -> Draw();
  c1->SaveAs(imgpath+"p17.png");  
  hMC -> Delete();

  ftreeMC -> Draw("eHcalCone/eParticle:iPhi>>hMC", selection, "prof");
  hMC -> SetTitle("eHcal/eParticle  vs iPhi");
hMC -> SetMaximum(1.2);
  hMC -> SetMinimum(0.1);

  hMC -> SetFillColor(kRed-10);
  label -> Draw();
  c1->SaveAs(imgpath+"p18.png");  
  hMC -> Delete();

  c1 -> SetLogy(0);

  ftreeMC -> Draw("eParticle/e5x5:iEta>>hMC(83,-42.5,42.5)", selection,"prof");
  hMC -> SetTitle("Corrections");
  hMC -> SetFillColor(kRed-10);
  label -> Draw();
  c1->SaveAs(imgpath+"p19.png");  
  hMC -> Delete();

  /*
  ftreeMC -> Draw("eParticle/eHcalCone:iEta>>hMC(83,-42.5,42.5)", selection,"prof");
  hMC -> SetTitle("Corrections");
  hMC -> SetFillColor(kRed-10);
  label -> Draw();
  c1->SaveAs(imgpath+"p19.png");  
  hMC -> Delete();
*/

  ftreeMC -> Draw("etaParticle>>hMC", selection);
  hMC -> SetTitle("eta of MC particle");
  hMC -> SetFillColor(kRed-10);
  label -> Draw();
  c1->SaveAs(imgpath+"p20.png");  
  hMC -> Delete();

  ftreeMC -> Draw("phiParticle>>hMC", selection, "");
  hMC -> SetTitle("phi of MC particle");
  hMC -> SetFillColor(kRed-10);
  label -> Draw();
  c1->SaveAs(imgpath+"p21.png");  
  hMC -> Delete();


  ftreeMC -> Draw("etaParticle:etaGPoint>>hMC", selection, "prof");
  hMC -> SetTitle("etaMC vs etaGPoint");
  hMC -> SetXTitle("eta");
  label -> Draw();
  c1->SaveAs(imgpath+"p22.png");  
  hMC -> Delete();

  ftreeMC -> Draw("phiParticle:phiGPoint>>hMC", selection, "prof");
  hMC -> SetTitle("phiMC vs phiGPoint");
  hMC -> SetXTitle("phi");
  label -> Draw();
  c1->SaveAs(imgpath+"p23.png");  
  hMC -> Delete();


  ftreeMC -> Draw("(eECAL40cm-eECAL09cm)>>hMC", selection, "");
  hMC -> SetTitle("neutral energy");
  hMC -> SetFillColor(kRed-10);
  label -> Draw();
  c1->SaveAs(imgpath+"p24.png");  
  hMC -> Delete();

  c1 -> SetLogy();

  ftreeMC -> Draw("delRmc>>hMC", "delRmc<1.", "");
  hMC -> SetTitle("#DeltaR(genTrack, recoTrack)");
  hMC -> SetFillColor(kRed-10);
  label -> Draw();
  c1->SaveAs(imgpath+"p25.png");  
  hMC -> Delete();


  ftreeMC -> Draw("delR>>hMC", selection, "");
  hMC -> SetTitle("#DeltaR(hotHit, trackHit)");
  hMC -> SetFillColor(kRed-10);
  label -> Draw();
  c1->SaveAs(imgpath+"p26.png");  
  hMC -> Delete();


 ftreeMC -> Draw("iPhi>>hMC", selection&&"abs(iEta)>=29");
 hMC -> SetTitle("iPhi at |iEta|>=29");
 hMC -> SetFillColor(kRed-10);
 label -> Draw();
 c1->SaveAs(imgpath+"p07.png");  
 hMC->Delete();
 


 ftreeMC -> Draw("iPhi>>hMC", selection&&"abs(iEta)>=29");
 hMC -> SetTitle("iPhi at |iEta|>=29");
 hMC -> SetFillColor(kRed-10);
 label -> Draw();
 c1->SaveAs(imgpath+"p07.png");  
 hMC->Delete();
 


  c1 -> SetLogy();


  gStyle -> SetOptStat(0);
 ftreeMC -> Draw("nTracks>>hMC", "abs(etaParticle)<2.","");
 hMC -> SetTitle("nTracks");
 hMC -> SetLineColor(kRed+2);
 c1->SaveAs(imgpath+"p30.png");

 hMC->Delete();


 ftreeMC -> Draw("trkQual[0]>>hMC1", "");
 hMC1 -> SetMinimum(0.1);
 ftreeMC -> Draw("trkQual[1]>>hMC2", "", "same");
 ftreeMC -> Draw("trkQual[2]>>hMC3", "", "same");
 ftreeMC -> Draw("trkQual[3]>>hMC4", "", "same");
  hMC1 -> SetTitle("trkQual");
 hMC1 -> SetLineColor(kRed+2);
 hMC2 -> SetLineColor(kBlue+2);
 hMC3 -> SetLineColor(kGreen+2);
 hMC4 -> SetLineColor(kYellow+2);
 c1->SaveAs(imgpath+"p31.png");

 hMC1->Delete();
 hMC2->Delete();
 hMC3->Delete();
 hMC4->Delete();

  c1 -> SetLogy();

 ftreeMC -> Draw("numValidTrkHits[0]>>hMC1", "");
 //hMC1 -> SetMinimum(0.1);
 ftreeMC -> Draw("numValidTrkHits[1]>>hMC2", "", "same");
 ftreeMC -> Draw("numValidTrkHits[2]>>hMC3", "", "same");
 ftreeMC -> Draw("numValidTrkHits[3]>>hMC4", "", "same");
  hMC1 -> SetTitle("numValidTrkHits");
 hMC1 -> SetLineColor(kRed+2);
 hMC2 -> SetLineColor(kBlue+2);
 hMC3 -> SetLineColor(kGreen+2);
 hMC4 -> SetLineColor(kYellow+2);
 c1->SaveAs(imgpath+"p32.png");

 hMC1->Delete();
 hMC2->Delete();
 hMC3->Delete();
 hMC4->Delete();

ftreeMC -> Draw("numValidTrkStrips[0]>>hMC1", "abs(trackEta)>1.7");
 // hMC1 -> SetMinimum(0.1);
 ftreeMC -> Draw("numValidTrkStrips[1]>>hMC2", "abs(trackEta)>1.7", "same");
 ftreeMC -> Draw("numValidTrkStrips[2]>>hMC3", "abs(trackEta)>1.7", "same");
 ftreeMC -> Draw("numValidTrkStrips[3]>>hMC4", "abs(trackEta)>1.7", "same");
  hMC1 -> SetTitle("numValidTrkStrips");
 hMC1 -> SetLineColor(kRed+2);
 hMC2 -> SetLineColor(kBlue+2);
 hMC3 -> SetLineColor(kGreen+2);
 hMC4 -> SetLineColor(kYellow+2);
 c1->SaveAs(imgpath+"p33.png");

 hMC1->Delete();
 hMC2->Delete();
 hMC3->Delete();
 hMC4->Delete();

 ftreeMC -> Draw("numLayers[0]>>hMC1", "");
 ftreeMC -> Draw("numLayers[1]>>hMC2", "", "same");
 ftreeMC -> Draw("numLayers[2]>>hMC3", "", "same");
 ftreeMC -> Draw("numLayers[3]>>hMC4", "", "same");
  hMC1 -> SetTitle("numLayers");
 hMC1 -> SetLineColor(kRed+2);
 hMC2 -> SetLineColor(kBlue+2);
 hMC3 -> SetLineColor(kGreen+2);
 hMC4 -> SetLineColor(kYellow+2);
 c1->SaveAs(imgpath+"p34.png");

 hMC1->Delete();
 hMC2->Delete();
 hMC3->Delete();
 hMC4->Delete();

ftreeMC -> Draw("trackP[0]>>hMC1", "trackP[0]<150");
 ftreeMC -> Draw("trackP[1]>>hMC2", "trackP[1]<150", "same");
 ftreeMC -> Draw("trackP[2]>>hMC3", "trackP[2]<150", "same");
 ftreeMC -> Draw("trackP[3]>>hMC4", "trackP[3]<150", "same");
  hMC1 -> SetTitle("trackP");
 hMC1 -> SetLineColor(kRed+2);
 hMC2 -> SetLineColor(kBlue+2);
 hMC3 -> SetLineColor(kGreen+2);
 hMC4 -> SetLineColor(kYellow+2);
 c1->SaveAs(imgpath+"p35.png");

 hMC1->Delete();
 hMC2->Delete();
 hMC3->Delete();
 hMC4->Delete();



 ftreeMC -> Draw("trackEta[0]>>hMC1", "");
 ftreeMC -> Draw("trackEta[1]>>hMC2", "", "same");
 ftreeMC -> Draw("trackEta[2]>>hMC3", "", "same");
 ftreeMC -> Draw("trackEta[3]>>hMC4", "", "same");
  hMC1 -> SetTitle("trackEta");
 hMC1 -> SetLineColor(kRed+2);
 hMC2 -> SetLineColor(kBlue+2);
 hMC3 -> SetLineColor(kGreen+2);
 hMC4 -> SetLineColor(kYellow+2);
 c1->SaveAs(imgpath+"p36.png");
hMC1->Delete();
 hMC2->Delete();
 hMC3->Delete();
 hMC4->Delete();

 c1 -> SetLogy(0);
 ftreeMC -> Draw("trackPhi[0]>>hMC1", "");
 ftreeMC -> Draw("trackPhi[1]>>hMC2", "", "same");
 ftreeMC -> Draw("trackPhi[2]>>hMC3", "", "same");
 ftreeMC -> Draw("trackPhi[3]>>hMC4", "", "same");
  hMC1 -> SetTitle("trackPhi");
 hMC1 -> SetLineColor(kRed+2);
 hMC2 -> SetLineColor(kBlue+2);
 hMC3 -> SetLineColor(kGreen+2);
 hMC4 -> SetLineColor(kYellow+2);
 c1->SaveAs(imgpath+"p37.png");

 hMC1->Delete();
 hMC2->Delete();
 hMC3->Delete();
 hMC4->Delete();

 c1-> SetLogy();

 ftreeMC -> Draw("delRmc[0]>>hMC1", "delRmc[0]<1.");
 ftreeMC -> Draw("delRmc[1]>>hMC2", "delRmc[1]<1.", "same");
 ftreeMC -> Draw("delRmc[2]>>hMC3", "delRmc[2]<1.", "same");
 ftreeMC -> Draw("delRmc[3]>>hMC4", "delRmc[3]<1.", "same");
  hMC1 -> SetTitle("delRmc");
 hMC1 -> SetLineColor(kRed+2);
 hMC2 -> SetLineColor(kBlue+2);
 hMC3 -> SetLineColor(kGreen+2);
 hMC4 -> SetLineColor(kYellow+2);
 c1->SaveAs(imgpath+"p38.png");

 hMC1->Delete();
 hMC2->Delete();
 hMC3->Delete();
 hMC4->Delete();

 c1-> SetLogy(0);

 ftreeMC -> Draw("sqrt(xAtHcal*xAtHcal+yAtHcal*yAtHcal):zAtHcal>>hMC", "","box");
  hMC -> SetTitle("r vs z");
  hMC -> SetFillColor(kRed+2);
  label -> Draw();
  c1->SaveAs(imgpath+"p39.png");
  hMC->Delete();

 ftreeMC -> Draw("zAtHcal:iEta>>hMC", "","box");
 hMC -> SetTitle("z vs iEta");
 hMC -> SetLineColor(kRed+2);
 c1->SaveAs(imgpath+"p40.png");

 hMC->Delete();


  fMC->Close();


}
