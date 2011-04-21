
{
  #include <map>
  //#include <TMap.h>

  gStyle -> SetOptStat(0);
//  gStyle->SetPalette(1);
  gStyle->SetPadGridX(1);
  gStyle->SetPadGridY(1);
/*
 TFile* tf = new TFile("./HcalCorrPFcone26.2cm.root","OPEN");
 ofstream new_pfcorrs("newPFcorrs26.2cm.txt");
 ofstream new_pfcorrs_subnoise("newPFcorrsNoiseSubtracted26.2cm.txt");  
*/
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

 TString imgpath("~/afs/public_html/validation/pfcorrs/50cm/");
 TFile* tf = new TFile("./HcalCorrPFcone50cm.root","OPEN");
 ofstream new_pfcorrs("newPFcorrs50cm.txt");
 ofstream new_pfcorrs_subnoise("newPFcorrsNoiseSubtracted50cm.txt");


 TProfile* p1 = (TProfile*)tf->Get("enHcal");  
 TProfile* p2 = (TProfile*)tf->Get("enHcalNoise");  
 TProfile* p3 = (TProfile*)tf->Get("nCells");
 TProfile* p4 = (TProfile*)tf->Get("nCellsNoise");
 TProfile* p5 = (TProfile*)tf->Get("enEcal");

TProfile* diff1 = new TProfile("diff1", "(oldvalue - new)/oldvalue", 85,-42,42);
TProfile* diff2 = new TProfile("diff2", "(pfcorr - pfSubNoise)/pfcorr", 85,-42,42);

TProfile* corrs1 = new TProfile("corrs1", "PF corrs", 85,-42,42);
TProfile* corrs2 = new TProfile("corrs2", "PF corrs", 85,-42,42);
TProfile* corrs3 = new TProfile("corrs3", "PF corrs", 85,-42,42);
TProfile* corrs4 = new TProfile("corrs4", "PF corrs", 85,-42,42);

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
  
for (Int_t i=0; i<=84; i++)
{
//  cout<<"bin: "<< p1->GetBinCenter(i)<<"   bin content: " <<p1->GetBinContent(i)<<"   response: "<< p1->GetBinContent(i)/50.<<endl;
  if (p1->GetBinContent(i)!=0) CorrValues[p1->GetBinCenter(i)]= 50./p1->GetBinContent(i);
  if ((p1->GetBinContent(i)-p2->GetBinContent(i))!=0) CorrValuesSubNoise[p1->GetBinCenter(i)]= 50./(p1->GetBinContent(i) - p2->GetBinContent(i));
  }    

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
new_pfcorrs<<CorrValues[iEta];
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
new_pfcorrs_subnoise<<CorrValuesSubNoise[iEta];
new_pfcorrs_subnoise.width(13);
new_pfcorrs_subnoise.setf(ios::uppercase);
new_pfcorrs_subnoise<<hex<<detId<<endl;


 if(depth>0) {

   diff1-> Fill(iEta, (value-CorrValues[iEta])/value);
   diff2-> Fill(iEta, (CorrValues[iEta] - CorrValuesSubNoise[iEta])/CorrValues[iEta]);

   corrs1-> Fill(iEta, value);
   corrs2-> Fill(iEta, CorrValues[iEta]);
   corrs3-> Fill(iEta, CorrValuesSubNoise[iEta]);
 }

/*
cout.width(16);
cout<<iEta;
cout.width(16);
cout<<iPhi;
cout.width(16);
cout<<depth;
cout.width(16);
cout<<sdName;
cout.width(16);
cout<<value;
cout.width(16);
cout<<detId<<endl;
*/

}



TCanvas* c1 = new TCanvas("c1","all",0,0,350,350);
  c1-> cd();
  p1 -> Draw("");
  p1 -> SetMaximum(60.);
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


  corrs1 -> Draw("");  
  corrs2 -> Draw("same");
  corrs1 -> SetMaximum(2.);
  corrs2 -> SetLineColor(kBlue);  
  c1->SaveAs(imgpath+"c01.png");  

  corrs2 -> Draw("");  
  corrs3 -> Draw("same");
  corrs2 -> SetMaximum(2.);
  corrs2 -> SetLineColor(kBlue);  
  corrs3 -> SetLineColor(kRed+1);  
  c1->SaveAs(imgpath+"c02.png");  

  diff1 -> Draw("");  
  c1->SaveAs(imgpath+"c03.png");
  
  diff2 -> Draw("");  
  c1->SaveAs(imgpath+"c04.png");  
  

  tf->Close();
}

