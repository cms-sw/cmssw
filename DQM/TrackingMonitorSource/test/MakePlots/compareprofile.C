//#ifndef __CINT__
#include "TStyle.h"
#include "TFile.h"
#include "TH1.h"
#include "TProfile.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TMath.h"
#include "TLine.h"
#include "TText.h"
#include "TPaveText.h"
//#endif
#include <iostream>
#include <vector>
#include <fstream>
#include <cassert>

using namespace std;
typedef unsigned int uint;

void tokenize(const string& str, vector<string>& tokens, const string& delimiters=" ");
void openFiles(vector<TFile*>& v, vector<string>& l, const char* filename="filelist.txt");
void closeFiles(vector<TFile*>& v);
void readHistograms(vector<string>& v, const char* filename="highpurityProfile.txt");
void compareHisto(TCanvas* canvas, const vector<TFile*>& v, const vector<string>& lglist, const vector<string>& tokens); 
void compareHisto(const vector<TFile*>& v, const char* folder, const char* hname); 
void compareprofile() {
  vector<TFile*> filelist;
  vector<string> lglist;
  openFiles(filelist, lglist);
  cout << ">>> # of Root files: " << filelist.size() << endl;

  vector<string> hlist;
  readHistograms(hlist);
  cout << ">>> # of histograms: " << hlist.size() << endl;
  TCanvas* canvas = new TCanvas("canvas", "canvas");
  canvas->SetCanvasSize(1080, 720);

  for (uint i = 0; i < hlist.size(); ++i) {
    string line(hlist[i]);
    vector<string> tokens;
    tokenize(line, tokens, ":");
    assert(tokens.size() > 1);
    compareHisto(canvas, filelist, lglist, tokens);
  }
  closeFiles(filelist);
}
void compareHisto(const vector<TFile*>& v, const char* folder, const char* hname) 
{
  for (uint i = 0; i < v.size(); ++i) {
    TFile* f = v[i];
    if (i==0) f->cd("DQMData/Run 260627/StandaloneTrackMonitor/Run summary/highPurityTracks/");
    if (i==1) f->cd("DQMData/Run 1/StandaloneTrackMonitor/Run summary/highPurityTracks/");
    TProfile *h = dynamic_cast<TProfile*>(gDirectory->Get(hname));
    if (!h) continue;
    h->SetMarkerColor(i);
    h->Draw(i ? "SAME" : "");
  }
}
void compareHisto(TCanvas* canvas, const vector<TFile*>& v, const vector<string>& lglist, const vector<string>& tokens) 
{
  std::cout<<"ComparemHisto Start"<<std::endl;

  string folder(tokens[0]);
  string hname(tokens[1]);

  TPad* pad1 = new TPad("pad1", "pad1",0,0,1,1);
  pad1->Draw();
  pad1->cd();

  TLegend *legend = new TLegend(0.40,0.8,0.60,0.9);

  double hmax = -1;
  double nentriesdata = 0;
  for (uint i = 0; i < v.size(); ++i) {
    TFile* f = v[i];
    if (i==0) f->cd("DQMData/Run 260627/StandaloneTrackMonitor/Run summary/highPurityTracks/");
    if (i==1) f->cd("DQMData/Run 1/StandaloneTrackMonitor/Run summary/highPurityTracks/");
    if (i==2) f->cd("DQMData/Run 1/StandaloneTrackMonitor/Run summary/highPurityTracks/");
    //f->cd(folder.c_str());
    TProfile *h = dynamic_cast<TProfile*>(gDirectory->Get(hname.c_str()));
    //TH1* h = gDirectory->Get(hname.c_str());
    if (!h) continue;


    //if (i == 0) nentriesdata = h->Integral("width");
    //else h->(nentriesdata/h->Integral("width"));
    //if (i == 0) h->Scale(0.896042);
    //if (i == 1) h->Scale(1130*71.163*1.05*(5153./10000)*(1./101967));
    //cout << "i = " << i << ", Integral = " << h->Integral() << endl;
    if (h->GetMaximum() > hmax) hmax = h->GetMaximum();
    
    string option;//("E");
    if (!i) {
      if (tokens.size() > 2) h->GetXaxis()->SetTitle(tokens[2].c_str());
      if (tokens.size() > 3) h->GetYaxis()->SetTitle(tokens[3].c_str());
      h->GetXaxis()->SetLabelSize(0.03);
      h->GetYaxis()->SetLabelSize(0.03);
      h->GetXaxis()->SetTitleSize(0.03);
      h->GetYaxis()->SetTitleSize(0.03);
      h->GetXaxis()->SetTitleOffset(1.25);
      h->GetYaxis()->SetTitleOffset(1.25);
      h->SetMinimum(0.0);

      h->SetMarkerSize(0.7);
      h->SetMarkerStyle(20);
      h->SetMarkerColor(i+1);

      TPaveStats *hstats = new TPaveStats(0.99,0.99,0.99,0.99,"brNDC");
      hstats->SetTextColor(i+1);
      hstats->SetOptStat(1111);
      hstats->Draw();
      h->GetListOfFunctions()->Add(hstats);
      hstats->SetParent(h);
    }
    else {
      option = "HISTSAMES";
      h->SetLineWidth(2.3);
      h->SetLineStyle(1);
      h->SetLineColor(i+1);

      TPaveStats *hstats = new TPaveStats(0.99,0.99,0.99,0.99,"brNDC");
      hstats->SetTextColor(i+1);
      hstats->SetOptStat(1111);
      hstats->Draw();
      h->GetListOfFunctions()->Add(hstats);
      hstats->SetParent(h);
    }

    h->Draw(option.c_str());
    legend->AddEntry(h, lglist[i].c_str());
    legend->SetTextSize(0.025);
  }
  // correct maximum height
  TFile* f = v.at(0);
  f->cd("DQMData/Run 260627/StandaloneTrackMonitor/Run summary/highPurityTracks/");
  //f->cd(folder.c_str());
  TProfile *h = dynamic_cast<TProfile*>(gDirectory->Get(hname.c_str()));
  assert(h);

  double fct = (tokens.size() > 6 && tokens[6] == "log") ? 6 : 1.25;
  h->SetMaximum(fct * hmax);
  
  if (tokens.size() > 5 && tokens[5] == "log") pad1->SetLogx();
  if (tokens.size() > 6 && tokens[6] == "log") pad1->SetLogy();
  legend->Draw();
  pad1->Update();
  pad1->Modified();
  
  std::cout << "pad1 drawn successfully" << std::endl; 
  canvas->cd();
  canvas->Update();
  canvas->Modified();
  char fname[256];
  sprintf(fname, "%s.png", hname.c_str());
  canvas->Print(fname);
  cout << "Profile printed successfully" << endl;
  canvas->Clear();
  return;
}
void openFiles(vector<TFile*>& v1, vector<string>& v2, const char* filename) {
  static const int BUF_SIZE = 256;

  // Open the file containing the datacards
  ifstream fin(filename, ios::in);
  if (!fin) {
    cerr << "Input File: " << filename << " could not be opened!" << endl;
    return;
  }
  char buf[BUF_SIZE];
  while (fin.getline(buf, BUF_SIZE, '\n')) {  // Pops off the newline character
    string line(buf);
    if (line.empty()) continue;
    if (line.substr(0,2) == "//") continue;
    if (line.substr(0,1) == "#") continue;
    cout << "file: " << line << endl;

    vector<string> tokens;
    tokenize(line, tokens, ":");
    assert(tokens.size() > 1);

    TFile* f = TFile::Open(tokens[0].c_str());
    v1.push_back(f);
    v2.push_back(tokens[1].c_str());
  }
  fin.close();
}
void closeFiles(vector<TFile*>& v) {
  for (uint i = 0; i < v.size(); ++i) {
    TFile* f = v[i];
    f->Close();
  }
}
void readHistograms(vector<string>& v, const char* filename) {
  static const int BUF_SIZE = 512;

  // Open the file containing the datacards
  ifstream fin(filename, ios::in);    
  if (!fin) {
    cerr << "Input File: " << filename << " could not be opened!" << endl;
    return;
  }
  char buf[BUF_SIZE];
  while (fin.getline(buf, BUF_SIZE, '\n')) {  // Pops off the newline character
    string line(buf);
    if (line.empty()) continue;
    if (line.substr(0,2) == "//") continue;
    if (line.substr(0,1) == "#") continue;

    cout << "histogram: " << line << endl;
    v.push_back(line);
  }
  fin.close();
}
void tokenize(const string& str, vector<string>& tokens, const string& delimiters) {
  // Skip delimiters at beginning.
  string::size_type lastPos = str.find_first_not_of(delimiters, 0);

  // Find first "non-delimiter".
  string::size_type pos = str.find_first_of(delimiters, lastPos);

  while (string::npos != pos || string::npos != lastPos)  {
    // Found a token, add it to the vector.
    tokens.push_back(str.substr(lastPos, pos - lastPos));

    // Skip delimiters.  Note the "not_of"
    lastPos = str.find_first_not_of(delimiters, pos);

    // Find next "non-delimiter"
    pos = str.find_first_of(delimiters, lastPos);
  }
}
