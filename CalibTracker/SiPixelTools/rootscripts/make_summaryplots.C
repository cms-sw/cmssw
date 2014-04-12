#include <iostream>
#include <vector>
#include "TKey.h"

void make_summaryplots(){
  
  //  TFile *file = new TFile("/tmp/fblekman/DQM_R000003238__A__B__C.root");
  TFile *file = new TFile("DQM_R000002445__A__B__C.root");
  file->cd();
  //  TDirectory *dir = file->GetDirectory("DQMData/Run 3238/Pixel/Run summary");
  TDirectory *dir = file->GetDirectory("DQMData/Run 2445/Pixel/Run summary");
  // make a loop over all plots
  TList *list = dir->GetListOfKeys();
  gStyle->SetOptStat(0);
  Int_t nkeys = file->GetNkeys();
  if(dir==0)
    return;

  std::cout << "now opening " << dir->GetName() << std::endl;

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
      //std::cout << "now examining " << notdonelist[idir]<< " " << nsubdirs[idir] <<  std::endl;
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
	keyname.ReplaceAll(" ","");
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
      if(keyname.Contains(curvestring))
	continue;
      if(keytype=="TH1F" || keytype=="TH2F" ){
	std::cout << keyname << std::endl;
	dir=file->GetDirectory(dirlist[i]);
	TH1F* temp = (TH1F*) dir->Get(keyname);
	curvecanvas->cd();
	curvecanvas->Clear();
	dir=file->GetDirectory(dirlist[i]);
	dir->cd();
	gStyle->SetOptStat(0);
	gStyle->SetOptFit(0);
	if(keytype=="TH1F"){
	  temp->GetXaxis()->SetTitle(temp->GetXaxis()->GetTitle());
	  temp->GetYaxis()->SetTitle(temp->GetYaxis()->GetTitle());
	  temp->DrawClone(); 
	  CMSprelim.DrawTextNDC(0.57,0.85,"CMS Pixel");
	}
	else{
	  temp->GetXaxis()->SetTitle("Columns");
	  temp->GetYaxis()->SetTitle("Rows");
	  temp->SetMinimum(0.);
	  temp->DrawClone("colz"); 
	  CMSprelim.DrawTextNDC(0.02,0.01,"CMS Pixel");
	}
	curvecanvas->Update();
	curvecanvas->Print(keyname+".jpg");
	curvecanvas->Print(keyname+".eps");
	  
      }
    }
  }
}



