#include <iostream>
#include <vector>
#include "TKey.h"
#include "TMath.h"

Double_t mycurvefunc(Double_t *x, Double_t *par){
  double xval = x[0]-par[0];
  xval /=sqrt(2.)*par[1];
  double val=0.5+0.5*TMath::Erf(xval);
  return val;  
}
void make_scurves(){
  
  TFile *file = new TFile("SCurve_DQM_Calibrationsmall.root");
  file->cd();
  // make a loop over all plots
  TList *list = file->GetListOfKeys();
  gStyle->SetOptStat(0);
  Int_t nkeys = file->GetNkeys();
  TDirectory *dir = file->GetDirectory("DQMData");
  if(dir==0)
    return;


  TLatex CMSprelim(0.3,0.3,"CMS Preliminary");
  CMSprelim.SetTextColor(2);
  CMSprelim.SetTextSize(CMSprelim.GetTextSize()*1.2);
  
  TString comparestring = "Module";
  TString curvestring = "row";
  std::vector<TString> keylist;
  std::vector<TString> hist1list;
  std::vector<TString> hist2list;
  std::vector<TString> dirlist;
  std::vector<TString> notdonelist;
  std::vector<int> nsubdirs;
  TDirectory *dirsav = dir;
  list = dir->GetListOfKeys();
  int ikey=0;
  int localkey=0;
  int ntimes=0;

  TCanvas *curvecanvas = new TCanvas();
  for(ikey=0;ikey<list->GetEntries();  ikey++){
    TKey *thekey = (TKey*)list->At(ikey);
    if(thekey==0)
      continue;
    TString keyname=thekey->GetName();
    //    keyname.ReplaceAll(" ","");
    TString keytype=thekey->GetClassName();
    keytype.ReplaceAll(" ","");
    if(keyname=="EventInfo")
      continue;
    //    std::cout <<  keytype << " " << keyname << std::endl;
    if(keytype=="TDirectoryFile"){
      TString dirname=dir->GetPath();
      dirname+="/";
      dirname+=keyname;
      //      std::cout << dirname << std::endl;
      dir=file->GetDirectory(dirname);
  
      list=dir->GetListOfKeys();
      if(dirname.Contains(comparestring))
	dirlist.push_back(dirname);
      else{
	notdonelist.push_back(dirname);
	nsubdirs.push_back(-1);
      }
    }
  }
  int nempty=0;
  while(nempty!=notdonelist.size()){
    for(int idir=0; idir<notdonelist.size(); ++idir){
      if(nsubdirs[idir]==0)
	continue;
      //      std::cout << "now examining " << notdonelist[idir]<< " " << nsubdirs[idir] <<  std::endl;
      dir = file->GetDirectory(notdonelist[idir]); 
      //      std::cout << dir->GetName() << std::endl;
      list= dir->GetListOfKeys(); 
      //      std::cout << list->GetEntries() << std::endl;
      int ndirectories=0;
      for(ikey=0;ikey<list->GetEntries();  ikey++){
	TKey *thekey = (TKey*)list->At(ikey);
	if(thekey==0)
	  continue;
	TString keyname=thekey->GetName();
	//	keyname.ReplaceAll(" ","");
	TString keytype=thekey->GetClassName();
	keytype.ReplaceAll(" ","");
	if(keytype=="TDirectoryFile"){
	  TString dirname=dir->GetPath();
	  dirname+="/";
	  dirname+=keyname;
	  //	  std::cout << dirname << std::endl;
	  ndirectories++;
	  if(dirname.Contains(comparestring))
	    dirlist.push_back(dirname);
	  else{
	    notdonelist.push_back(dirname);
	    nsubdirs.push_back(-1);
	  }
	}
      }
      nsubdirs[idir]=ndirectories;
      // std::cout << "now done examining " << notdonelist[idir]<< " " << nsubdirs[idir] <<  std::endl;
    }
    // count number of done dirs;
    nempty=0;
    for(int i=0; i<nsubdirs.size(); i++){
      if(nsubdirs[i]!=-1)
	nempty++;
    }
  }
  gStyle->SetOptStat(0);
  int ncurves=0;
  for(int i=0; i<dirlist.size() ; ++i){
    //    std::cout << "good dir "  << dirlist[i] << std::endl;
    // now count histograms:
    dir = file->GetDirectory(dirlist[i]); 
    list= dir->GetListOfKeys(); 
    //      std::cout << list->GetEntries() << std::endl;
    for(ikey=0;ikey<list->GetEntries();  ikey++){
      //      std::cout << ikey << std::endl;
      TKey *thekey = (TKey*)list->At(ikey);
      if(thekey==0)
	continue;
      TString keyname=thekey->GetName();
      keyname.ReplaceAll(" ","");
      TString keytype=thekey->GetClassName();
      keytype.ReplaceAll(" ","");
      //      std::cout << ikey << " " <<  keyname << std::endl;

      if(keytype=="TH1F"){
       	std::cout << keyname << " " << keytype << std::endl;
	if(keyname.Contains(curvestring)){
	  std::cout << keyname << " " << keytype << std::endl;
	  dir=file->GetDirectory(dirlist[i]);
	  TH1F* temp = (TH1F*) dir->Get(keyname);
	  ncurves++;
	  std::cout << keyname << std::endl;
	  curvecanvas->cd();
	  curvecanvas->Clear();
	  dir=file->GetDirectory(dirlist[i]);
	  dir->cd();
	  gStyle->SetOptStat(0);
	  gStyle->SetOptFit(0);
	  temp->SetMarkerSize(2.0); 
	  temp->GetXaxis()->SetTitle("VCAL");
	  temp->GetYaxis()->SetTitle("Efficiency");
	  temp->DrawClone(); 
	  float gainandped[2];
	  getgainresult(keyname,file, dirlist[i],gainandped);
	  std::cout << gainandped[0] << " " << gainandped[1] << std::endl;
	  if(gainandped[0]>0. && gainandped[1]>0.){
	    TString funcname = keyname;
	    funcname+="func";
	    TF1 *func = new TF1(funcname,mycurvefunc,0,256,2);
	    func->SetLineWidth(3);
	    func->SetParameter(0,gainandped[1]);
	    func->SetParameter(1,gainandped[0]);
	    func->DrawClone("lsame");
	    CMSprelim.DrawText(gainandped[1]*1.05,temp->GetMinimum()+0.01,"CMS Pixel");
	    TLegend *legend = new TLegend(0.45,0.2,0.89,0.28);
	    //	    legend->SetFillStyle(0);
	    legend->SetFillColor(0);
	    legend->SetBorderSize(0);
 	    TString fitstring = "y=0.5+0.5 Erf(#frac{x-";
	    fitstring+= gainandped[1];
	    fitstring+="}{" ;
	    fitstring+= gainandped[0];
	    fitstring+=" #sqrt{2}})";
 	    legend->AddEntry(func,fitstring,"l");
 	    legend->DrawClone("same");
	    
	  }
	  curvecanvas->Update();
	  curvecanvas->Print(keyname+".jpg");
	  curvecanvas->Print(keyname+".eps");
	}
      }
    }
  }
  //  for(int i=0; i<notdonelist.size();++i){
  //    std::cout << notdonelist[i] << std::endl;
  //  }

  
}


float getgainresult(TString keyname, TFile *file, TString dirname, float *gainandped){

  //  std::cout << "examining histo " << keyname << std::endl;
  std::string strversion = keyname;
  std::string cutstring[20];
  int foundunderscore=0;
  for(int i=0; i<20 && foundunderscore<strversion.size(); i++){
    int found=strversion.find("_",foundunderscore);
    std::string thesubstr = strversion.substr(foundunderscore,found-foundunderscore);
    //std::cout << thesubstr << " " << found << std::endl;
    cutstring[i]=thesubstr;
    foundunderscore=found+1;
  }

  int row = atoi(cutstring[2].c_str());
  int col = atoi(cutstring[4].c_str());
  int detid = atoi(cutstring[6].c_str());
  //  std::cout << row << " " << col << " " << detid << std::endl;
  TString gainhisto = dirname;
  gainhisto+="/ScurveSigmas_siPixelCalibDigis_";
  gainhisto+=detid;
  TString pedhisto =  dirname;
  pedhisto+="/ScurveThresholds_siPixelCalibDigis_";
  pedhisto+=detid;
  //  std::cout << gainhisto << " " << pedhisto << std::endl;
  file->cd();
  TH2F *gain2d = file->Get(gainhisto);
  float gainval=0;
  float pedval=0;
  if(gain2d!=0)
    gainval = gain2d->GetBinContent(col,row);
  else return 0;
  TH2F *ped2d = file->Get(pedhisto);
  if(ped2d)
    pedval=ped2d->GetBinContent(col,row);
  else return 0;
  //std::cout << ped2d->GetNbinsX() << " " << ped2d->GetNbinsY() << std::endl;

  //std::cout << pedhisto << " " << gainhisto << " " << col+1 << " " << row+1 << " " <<  ped2d->GetBinContent(col+1,row+1) << " " << gain2d->GetBinContent(col+1,row+1) <<  std::endl;
  gainval = gain2d->GetBinContent(col,row);
  pedval = ped2d->GetBinContent(col,row);
  //  std::cout << gainhisto << " " << row << " " << col << " " << gainval << " " << pedval << std::endl;
  gainandped[0]=gainval;
  gainandped[1]=pedval;
  return 1;
}

