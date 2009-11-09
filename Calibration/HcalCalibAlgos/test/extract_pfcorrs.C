
{
  #include <map>
  //#include <TMap.h>

//  gStyle -> SetOptStat(1);
//  gStyle->SetPalette(1);
//  gStyle->SetPadGridX(1);
//  gStyle->SetPadGridY(1);

 TString imgpath("~/afs/public_html/validation/pfcorrs/");  
 TFile* tf = new TFile("./HcalCorrPF_30.root","OPEN");
    
  TProfile* p1 = (TProfile*)tf->Get("DQMData/HcalCorrPF/HcalCorrPF_En_rechits_cone30cm_profile_vs_ieta_all_depths_woHO");  
  TProfile* p2 = (TProfile*)tf->Get("DQMData/HcalCorrPF/HcalCorrPF_En_rechits_cone30cm_profile_vs_ieta_all_depths_woHO_Noise");  
  
  TProfile* p3 = (TProfile*)tf->Get("DQMData/HcalCorrPF/HcalCorrPF_Ncells_vs_ieta_cone30cm");
  TProfile* p4 = (TProfile*)tf->Get("DQMData/HcalCorrPF/HcalCorrPF_Ncells_vs_ieta_Noise_cone30cm");

TProfile* diff = new TProfile("diff", "PF corrs difference", 85,-42,42);

//ifstream old_pfcorrs("../data/HcalPFCorrs_v1.03_mc.txt");
  ifstream old_pfcorrs("../data/DumpCondPFCorrs_Run1.txt");
  ofstream new_pfcorrs("new_pfcorrs.txt");

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
  
for (Int_t i=0; i<=84; i++)
{
//  cout<<"bin: "<< p1->GetBinCenter(i)<<"   bin content: " <<p1->GetBinContent(i)<<"   response: "<< p1->GetBinContent(i)/50.<<endl;
  if (p1->GetBinContent(i)!=0) CorrValues[p1->GetBinCenter(i)]= 50./p1->GetBinContent(i);
  }    

 std::string line;
  while (getline(old_pfcorrs, line)) 
{
if(!line.size() || line[0]=='#') 
  {
    //fprintf(new_pfcorrs, "%1s%16s%16s%16s%16s%9s%11s\n", "#", "eta", "phi", "depth", "det", "value", "DetId");   
    new_pfcorrs<<"#       eta       phi       depth        det       value           DetId"<<endl;
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


 if(depth>0) diff-> Fill(iEta, (value - CorrValues[iEta])/value);

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
   c1->SaveAs(imgpath+"a01.png");  

  p2 -> Draw("");
   c1->SaveAs(imgpath+"a02.png");  

  p3 -> Draw("");
   c1->SaveAs(imgpath+"a03.png");  

  p4 -> Draw("");
   c1->SaveAs(imgpath+"a04.png");  

  diff -> Draw("");
   c1->SaveAs(imgpath+"a05.png");  


  tf->Close();
}

