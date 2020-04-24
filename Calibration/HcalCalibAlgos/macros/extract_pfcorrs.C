
{
  #include <map>

  gStyle -> SetOptStat(0);
//  gStyle->SetPalette(1);
  gStyle->SetPadGridX(1);
  gStyle->SetPadGridY(1);

TString imgpath("~/afs/public_html/pfcorrs/v02/");

 TFile* fcorrs = new TFile("~/nobackup/arch/hcalCorrsFile_resp01.root","OPEN");
 TProfile* respcorrs = (TProfile*)fcorrs->Get("corrs1");

 TFile* tf = new TFile("~/nobackup/arch/hcalCorrPFv13_35cm.root","OPEN");
 //TFile* tf = new TFile("~/afs/arch/hcalCorrPFv2_26.2cm.root","OPEN");
//TFile* tf = new TFile("./HcalCorrPF.root","OPEN");
 ofstream new_pfcorrs("newPFcorrs.txt");
 ofstream new_pfcorrs_subnoise("newPFcorrsNoiseSubtracted.txt");

// ofstream new_pfcorrs("newPFcorrs26.2cm.txt");
// ofstream new_pfcorrs_subnoise("newPFcorrsNoiseSubtracted26.2cm.txt");
  

/*
 TString imgpath("~/afs/public_html/validation/pfcorrs/30cm/");
 TFile* tf = new TFile("./HcalCorrPFcone30cm.root","OPEN");    
 ofstream new_pfcorrs("newPFcorrs30cm.txt");
 ofstream new_pfcorrs_subnoise("newPFcorrsNoiseSubtracted30cm.txt");
*/
/*  
 TString imgpath("~/afs/public_html/validation/pfcorrs/40cm/");
 TFile* tf = new TFile("./HcalCorrPFcone40cm.root","OPEN");
 ofstream new_pfcorrs("newPFcorrs40cm.txt");
 ofstream new_pfcorrs_subnoise("newPFcorrsNoiseSubtracted.txt");
  */

/*
 TString imgpath("~/afs/public_html/validation/pfcorrs/50cm/");
 TFile* tf = new TFile("./HcalCorrPFcone50cm.root","OPEN");
 ofstream new_pfcorrs("newPFcorrs50cm.txt");
 ofstream new_pfcorrs_subnoise("newPFcorrsNoiseSubtracted50cm.txt");
*/

/*
 TProfile* p1 = (TProfile*)tf->Get("hcalRecoAnalyzer/enHcal");  
 TProfile* p2 = (TProfile*)tf->Get("hcalRecoAnalyzer/enHcalNoise");  
 TProfile* p3 = (TProfile*)tf->Get("hcalRecoAnalyzer/nCells");
 TProfile* p4 = (TProfile*)tf->Get("hcalRecoAnalyzer/nCellsNoise");
 TProfile* p5 = (TProfile*)tf->Get("hcalRecoAnalyzer/enEcal");
*/

 TCut tr_cut = "eParticle>45 && eParticle<55";  label = new TText(0.03,0.2, "50 GeV");
  
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
  

  TTree* pftree = (TTree*)tf->Get("hcalPFcorrs/pfTree");
//  pftree -> Draw("eHcalCone/eParticle:iEta>>p1(82, -40.5, 41.5)", "", "prof goff");
  pftree -> Draw("eHcalCone/eParticle:iEta>>p1(82, -40.5, 41.5)", selection, "prof goff");
  pftree -> Draw("(eHcalCone-eHcalConeNoise)/eParticle:iEta>>p2(82, -40.5, 41.5)", selection, "prof goff");
  pftree -> Draw("UsedCells:iEta>>p3(82, -40.5, 41.5)", selection, "prof goff");
  pftree -> Draw("UsedCellsNoise:iEta>>p4(82, -40.5, 41.5)", selection, "prof goff");
  pftree -> Draw("eECAL:iEta>>p5(82, -40.5, 41.5)", selection, "prof goff");

TProfile* diff1 = new TProfile("diff1", "(oldvalue - new)/oldvalue", 84,-42,42);
TProfile* diff2 = new TProfile("diff2", "(pfcorr - pfSubNoise)/pfcorr", 84,-42,42);


TProfile* corrs2 = new TProfile("corrs2", "PF corrs", 84,-42,42);
TProfile* corrs3 = new TProfile("corrs3", "PF corrs", 84,-42,42);
TProfile* corrs4 = new TProfile("corrs4", "PF corrs", 84,-42,42);

//ifstream old_pfcorrs("../data/HcalPFCorrs_v1.03_mc.txt");
//ifstream old_pfcorrs("../data/HcalPFCorrs_v2.00_mc.txt");
ifstream old_pfcorrs("../data/response_corrections.txt");


  //FILE *new_pfcorrs = fopen("new_pfcorrs.txt", "w+"); 
  Int_t   iEta;
  UInt_t  iPhi;
  Int_t  depth;
  //TString sdName;
  string sdName;
  UInt_t  detId;
  Float_t value;
//  HcalSubdetector sd;
  map<Int_t, Float_t> CorrValues;
  map<Int_t, Float_t> CorrValuesSubNoise;
  
for (Int_t i=0; i<=82; i++)
{

  if (p1->GetBinContent(i)!=0) CorrValues[p1->GetBinCenter(i)]= 1./p1->GetBinContent(i);
  if (p2->GetBinContent(i)!=0) CorrValuesSubNoise[p2->GetBinCenter(i)]= 1./p2->GetBinContent(i);

  cout<<"bin: "<< p1->GetBinCenter(i)<<"   response: "<< p1->GetBinContent(i)<<"   corrs: "<< CorrValues[p1->GetBinCenter(i)]<<"   subnoise corrs: "<< CorrValuesSubNoise[p1->GetBinCenter(i)]<<endl;  


//if(depth>0) {

//diff1-> Fill(iEta, (value-CorrValues[iEta])/value);
//   diff2-> Fill(i, (CorrValues[i] - CorrValuesSubNoise[i])/CorrValues[i]);

   //corrs1-> Fill(iEta, value);
 }  
 // }

for (Int_t i=-41; i<=41; i++)
{
  corrs2-> Fill(i, CorrValues[i]);
  corrs3-> Fill(i, CorrValuesSubNoise[i]);

 }



TCanvas* c1 = new TCanvas("c1","all",0,0,350,350);
  c1-> cd();
  p1 -> Draw("");
  // p1 -> SetMaximum(60.);
  p1 -> SetXTitle("iEta");
  c1->SaveAs(imgpath+"a01.png");  


  p2 -> Draw("");
  p2 -> SetXTitle("iEta");
   c1->SaveAs(imgpath+"a02.png");  


  p3 -> Draw("");
  p3 -> SetXTitle("iEta");
   c1->SaveAs(imgpath+"a03.png");  

  p4 -> Draw("");
  p4 -> SetXTitle("iEta");
   c1->SaveAs(imgpath+"a04.png");  


  p5 -> Draw("");
  p5 -> SetXTitle("iEta");
  c1->SaveAs(imgpath+"a05.png");  
  
  
  respcorrs -> Draw("");  
  respcorrs -> SetMaximum(1.6);
  respcorrs -> SetMinimum(0.6);
  corrs2 -> Draw("same");
  corrs2 -> SetLineColor(kBlue);  
  c1->SaveAs(imgpath+"c01.png");  

  respcorrs -> Draw("");  
  corrs3 -> Draw("same");
  respcorrs -> SetMaximum(1.6);
  respcorrs -> SetMinimum(0.6);
  //corrs2 -> SetLineColor(kBlue);  
  corrs3 -> SetLineColor(kRed+1);  
  c1->SaveAs(imgpath+"c02.png");  

  //diff1 -> Draw("");  
  //c1->SaveAs(imgpath+"c03.png");
  
  //diff2 -> Draw("");  
  //c1->SaveAs(imgpath+"c04.png");  
  

  tf->Close();



  std::string line;
  
  while (getline(old_pfcorrs, line)) 
    {
      if(!line.size() || line[0]=='#') 
	{
	  //fprintf(new_pfcorrs, "%1s%16s%16s%16s%16s%9s%11s\n", "#", "eta", "phi", "depth", "det", "value", "DetId");   
	  new_pfcorrs<<"#             eta             phi           depth        det           value       DetId"<<endl;
	  continue;
	}
      
      std::istringstream linestream(line);
      linestream >> iEta >> iPhi >> depth >> sdName >> value >> hex >> detId;
      // corrs1-> Fill(abs(iEta), value);
      
      //cout<<" I'm HERE"<<iEta<<iPhi<<endl;      


      //format:
      //               -1               1               1              HB    0.97635     42004081
      
      //  fprintf(new_pfcorrs, "%17i%16i%16i%16s%9.5f%11X\n", iEta, iPhi, depth, sdName, value, detId);     
      

      new_pfcorrs.width(17);
      new_pfcorrs<<dec<<iEta;
      new_pfcorrs.width(16);
      new_pfcorrs<<iPhi;
      new_pfcorrs.width(16);
      new_pfcorrs<<depth;
      new_pfcorrs.width(11);
      new_pfcorrs<<sdName;
      new_pfcorrs.width(16);
      new_pfcorrs.setf(ios::fixed, ios::floatfield);
      new_pfcorrs.precision(5); 
      if (sdName=="HO" || depth==-99) {new_pfcorrs<<value;} else new_pfcorrs<<CorrValues[iEta];
      new_pfcorrs.width(13);
      new_pfcorrs.setf(ios::uppercase);
      new_pfcorrs<<hex<<detId<<endl;

      
      new_pfcorrs_subnoise.width(17);
      new_pfcorrs_subnoise<<dec<<iEta;
      new_pfcorrs_subnoise.width(16);
      new_pfcorrs_subnoise<<iPhi;
      new_pfcorrs_subnoise.width(16);
      new_pfcorrs_subnoise<<depth;
      new_pfcorrs_subnoise.width(11);
      new_pfcorrs_subnoise<<sdName;
      new_pfcorrs_subnoise.width(16);
      new_pfcorrs_subnoise.setf(ios::fixed, ios::floatfield);
      new_pfcorrs_subnoise.precision(5); 
      if (sdName=="HO" || depth==-99) {new_pfcorrs_subnoise<<value;} else new_pfcorrs_subnoise<<CorrValues[iEta];
      new_pfcorrs_subnoise.width(13);
      new_pfcorrs_subnoise.setf(ios::uppercase);
      new_pfcorrs_subnoise<<hex<<detId<<endl;

    }
}

