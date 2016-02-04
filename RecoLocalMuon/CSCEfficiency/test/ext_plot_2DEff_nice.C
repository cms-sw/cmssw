
#ifndef VECTOR
#define VECTOR
#include <vector>
#endif

#ifndef IOSTREAM
#define  IOSTREAM
#include <iostream>
#endif

#ifndef IOMANIP
#define IOMANIP
#include <iomanip>
#endif

{

  gStyle->SetTitleFillColor(0);
  gStyle->SetFrameFillColor(0);
  
  string MyDirectory = "/uscms/home/stoyan/work/csc_ana/CSCEff/dev/CMSSW_2_2_1/src/test/merged/";
  string MySubDirectory;
  string MyFileName;// = "efficiencies.root";
 
  string MyFullPath;
  char *file_name;

  MySubDirectory = "./";
  MyFileName = "efficiencies.root";
  MyFullPath =  MyDirectory +  MySubDirectory + MyFileName;
  file_name = MyFullPath.c_str();
  TFile *f1=
    (TFile*)gROOT->GetListOfFiles()->FindObject(file_name);
  if (!f1){
    TFile *f1 = new TFile(file_name);
  }

  
  std::vector < TFile * > DataFiles;
  DataFiles.push_back(f1);

  Int_t nx = 36;
  const Int_t ny = 16;
  
  char *chambers[nx]  = {"01","02","03","04","05","06","07","08","09","10",
		    "11","12","13","14","15","16","17","18","19","20",
			 "21","22","23","24","25","26","27","28","29","30",
			 "31","32","33","34","35","36"};
  char *types[ny] = {"ME-41","ME-32","ME-31","ME-22","ME-21","ME-13","ME-12","ME-11",
		     "ME+11","ME+12","ME+13","ME+21","ME+22","ME+31","ME+32","ME+41"};
  
  
  TH2F * data_p2;
  string histo = Form("h_rhEfficiency");

  //string histo = Form("h_segEfficiency");

  //string histo = Form("h_stripEfficiency");

  //string histo = Form("h_wireEfficiency");

  //string histo = Form("h_clctEfficiency");

  //string histo = Form("h_alctEfficiency");

  char *histo_full = histo.c_str();
  data_p2=(TH2F*)(*(DataFiles[0]))->Get(histo_full);
  
  TH2F *hold = (TH2F*)data_p2->Clone("old");
  
  int nbinsX = hold->GetNbinsX();
  int nbinsY = hold->GetNbinsY();
  

  TH2F *h2 = new TH2F("h2","RecHit efficiency (in %), errors represented by text; chamber number",nbinsX,0.,float(nbinsX), nbinsY,0.,float(nbinsY));
  //TH2F *h2 = new TH2F("h2","Segment efficiency (in %), errors represented by text; chamber number",nbinsX,0.,float(nbinsX), nbinsY,0.,float(nbinsY));
  //TH2F *h2 = new TH2F("h2","Strip efficiency (in %), errors represented by text; chamber number",nbinsX,0.,float(nbinsX), nbinsY,0.,float(nbinsY));
  //TH2F *h2 = new TH2F("h2","WireGroup efficiency (in %), errors represented by text; chamber number",nbinsX,0.,float(nbinsX), nbinsY,0.,float(nbinsY));
  //TH2F *h2 = new TH2F("h2","CLCT efficiency (in %), errors represented by text; chamber number",nbinsX,0.,float(nbinsX), nbinsY,0.,float(nbinsY));
  //TH2F *h2 = new TH2F("h2","ALCT efficiency (in %), errors represented by text; chamber number",nbinsX,0.,float(nbinsX), nbinsY,0.,float(nbinsY));


  TH2F *h3 = new TH2F("h2","RecHit efficiency (in %)",nbinsX,0.,float(nbinsX), nbinsY,0.,float(nbinsY));

  
  TCanvas *eff1 = new TCanvas("eff1","Efficiency",10,10,1200,800);
  gPad->SetFillColor(0);
  std::cout<<" Processing..."<<std::endl;  
  ofstream myfile;
  myfile.open ("efficiencies.txt");
  myfile<<"         "<<setw(7)<<"ME-41 "<<setw(7)<<"ME-32 "<<setw(7)<<"ME-31 "<<setw(7)<<"ME-22 "<<
    setw(7)<<"ME-21 "<<setw(7)<<"ME-13 "<<setw(7)<<"ME-12 "<<setw(7)<<"ME-11 "<<setw(7)<<
    "ME+11 "<<setw(7)<<"ME+12 "<<setw(7)<<"ME+13 "<<setw(7)<<"ME+21 "<<setw(7)<<
    "ME+22 "<<setw(7)<<"ME+31 "<<setw(7)<<"ME+32 "<<setw(7)<<"ME+41 "<<std::endl;
  for (int ibinX = 1 ; ibinX <= nbinsX ;  ibinX++ )  {
    myfile<<std::endl;
    myfile<<"CH"<<chambers[ibinX-1]<<"    ";
    for (int ibinY = 1 ; ibinY <= nbinsY ;  ibinY++ )  {
      Int_t ibin = hold->GetBin(ibinX,ibinY);
      double binCont = hold->GetBinContent(ibin) ;
      double binErr = hold->GetBinError(ibin) ;
      myfile<<" "<<setw(6)<<std::setprecision(4) <<binCont;
      int tmpCont = int(binCont*1000. + 0.5);
      int tmpErr = int(binErr*1000. + 0.5);
      double newCont = double(tmpCont)/10.;
      double newErr = double(tmpErr)/10.;
      h2->Fill(chambers[ibinX-1],types[ibinY-1],newCont);
      h3->Fill(chambers[ibinX-1],types[ibinY-1],newErr);
    }    
  }
  myfile.close();
  
  h2->SetStats(0);
  eff1->SetGrid();
  
  gStyle->SetPalette(1);
  h2->Draw("colz");  
  h3->Draw("sametext45"); 
  //gPad->GetFrame()->SetBorderMode(0);
  //eff1->GetFrame()->SetBorderMode(0);
}
