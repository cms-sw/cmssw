#include "analyzer.h"

/****************************************************************************/
void analyzeSimTracks2D(const char *prod, const char *part)
{
  // sim -> ids:etas:pts:rhos:nhits:nrec
  // rec ->  qr:etar:ptr:d0r

  char fileName[256];
  sprintf(fileName,"../data/result_%s.root",prod);
  TFile file(fileName,"read");
  TNtuple* trackSim = (TNtuple*) file.Get("trackSim");

  // Selection on simulated
  TCut base = "rhos<0.2";

  // Particle specific
  TCut pid; int bins;
  if(part == "pion") { pid = "abs(ids)== 211"; bins = 50; }
  if(part == "kaon") { pid = "abs(ids)== 321"; bins = 25; }
  if(part == "prot") { pid = "abs(ids)==2212"; bins = 25; }

  // Declare histograms
  TH2F* hsim = new TH2F("hsim","hsim",bins,-3,3,bins,0,2); // simulated
  TH2F* hacc = (TH2F*)hsim->Clone("hacc");  // accepted
  TH2F* href = (TH2F*)hsim->Clone("href");  // reconstructed/efficiency
  TH2F* hmul = (TH2F*)hsim->Clone("hmul");  // multiply reconstructed

  // Fill histograms
  trackSim->Project("hsim", "pts:etas", base + pid);
  trackSim->Project("hacc", "pts:etas", base + pid + "nhits>=3");
  trackSim->Project("href", "pts:etas", base + pid + "nhits>=3 && nrec>0");
  trackSim->Project("hmul", "pts:etas", base + pid + "nhits>=3 && nrec>1");

  // Ratios
  href->Sumw2(); href->Divide(href,hacc,1,1,"B"); // efficiency
  hmul->Sumw2(); hmul->Divide(hmul,hacc,1,1,"B"); // multiple counting
  hacc->Sumw2(); hacc->Divide(hacc,hsim,1,1,"B"); // acceptance
  
  // Write out
  sprintf(fileName,"../out/algoEffic_EtaPt_%s_%s.dat", prod,part);
  printToFile(href,fileName);

  sprintf(fileName,"../out/multCount_EtaPt_%s_%s.dat", prod,part);
  printToFile(hmul,fileName);

  sprintf(fileName,"../out/geomAccep_EtaPt_%s_%s.dat", prod,part);
  printToFile(hacc,fileName);

  delete hsim; file.Close();
}

/****************************************************************************/
void analyzeSimTracks1D(const char *prod, const char *part, const char *var)
{
  // sim -> ids:etas:pts:rhos:nhits:nrec
  // rec ->  qr:etar:ptr:d0r

  char fileName[256];
  sprintf(fileName,"../data/result_%s.root",prod);
  TFile file(fileName,"read");
  TNtuple* trackSim = (TNtuple*) file.Get("trackSim");

  // Selection on simulated
  TCut base; float hmin,hmax; char varName[128];
  if(var == "Pt" )
  {
    base = "rhos<0.2 && abs(etas)<2";
    hmin =  0; hmax = 2;
    sprintf(varName,"%s", "pts");
  }
  if(var == "Eta")
  {
    base = "rhos<0.2";
    hmin = -3; hmax = 3;
    sprintf(varName,"%s", "etas");
  }

  // Particle specific
  TCut pid; int bins;
  if(part == "pion") { pid = "abs(ids)== 211"; bins = 100; }
  if(part == "kaon") { pid = "abs(ids)== 321"; bins = 100; }
  if(part == "prot") { pid = "abs(ids)==2212"; bins = 100; }
  
  // Declare histograms
  TH1F* hsim = new TH1F("hsim","hsim",
                           bins,hmin,hmax); // simulated
  TH1F* hacc = (TH1F*)hsim->Clone("hacc");  // accepted
  TH1F* href = (TH1F*)hsim->Clone("href");  // reconstructed/efficiency
  TH1F* hmul = (TH1F*)hsim->Clone("hmul");  // multiply reconstructed

  TH2F* hbiar = new TH2F("hbiar","hbiar",
              bins,hmin,hmax,bins,hmin,hmax); // bias
  TH2F* hbiav = (TH2F*)hbiar->Clone("hbiav"); // bias with vertex
  TH2F* hresr = new TH2F("hresr","hresr",
              bins,hmin,hmax,bins,-1,-1);     // resolution
  TH2F* hresv = (TH2F*)hresr->Clone("hresv"); // resolution with vertex

  trackSim->Project("hbiar","ptr:pts",    base + "nhits>=3 && nrec==1" + pid);
  trackSim->Project("hbiav","ptv:pts",    base + "nhits>=3 && nrec==1" + pid);
  trackSim->Project("hresr","ptr-pts:pts",base + "nhits>=3 && nrec==1" + pid);
  trackSim->Project("hresv","ptv-pts:pts",base + "nhits>=3 && nrec==1" + pid);

  hbiar->FitSlicesY(0,0,0);
  hbiav->FitSlicesY(0,0,0);
  hresr->FitSlicesY(0,0,0);
  hresv->FitSlicesY(0,0,0);

  TH1F *hbiar_1 = (TH1F*)gDirectory->Get("hbiar_1");
  TH1F *hbiav_1 = (TH1F*)gDirectory->Get("hbiav_1");
  TH1F *hresr_2 = (TH1F*)gDirectory->Get("hresr_2");
  TH1F *hresv_2 = (TH1F*)gDirectory->Get("hresv_2");

  // Fill histograms
  trackSim->Project("hsim", varName, base + pid);
  trackSim->Project("hacc", varName, base + pid + "nhits>=3");
  trackSim->Project("href", varName, base + pid + "nhits>=3 && nrec>0");
  trackSim->Project("hmul", varName, base + pid + "nhits>=3 && nrec>1");

  // Ratios
  href->Sumw2(); href->Divide(href,hacc,1,1,"B"); // efficiency
  hmul->Sumw2(); hmul->Divide(hmul,hacc,1,1,"B"); // multiple counting
  hacc->Sumw2(); hacc->Divide(hacc,hsim,1,1,"B"); // acceptance

  // Write out
  sprintf(fileName,"../out/algoEffic_%s_%s_%s.dat", var,prod,part);
  printToFile(href,fileName);

  sprintf(fileName,"../out/multCount_%s_%s_%s.dat", var,prod,part);
  printToFile(hmul,fileName);

  sprintf(fileName,"../out/geomAccep_%s_%s_%s.dat", var,prod,part);
  printToFile(hacc,fileName);

  sprintf(fileName,"../out/biasr_%s_%s_%s.dat", var,prod,part);
  printToFile(hbiar_1,fileName);
  sprintf(fileName,"../out/biasv_%s_%s_%s.dat", var,prod,part);
  printToFile(hbiav_1,fileName);

  sprintf(fileName,"../out/resolutionr_%s_%s_%s.dat", var,prod,part);
  printToFile(hresr_2,fileName);
  sprintf(fileName,"../out/resolutionv_%s_%s_%s.dat", var,prod,part);
  printToFile(hresv_2,fileName);

  delete hsim; file.Close();
}

/****************************************************************************/
void analyzeSimTracks1D(const char *prod, const char *part, const char *var)
{

  TH2F* hbiar = new TH2F("hbiar","hbiar",
              bins,hmin,hmax,bins,hmin,hmax); // bias
  TH2F* hbiav = (TH2F*)hbiar->Clone("hbiav"); // bias with vertex
  TH2F* hresr = new TH2F("hresr","hresr",
              bins,hmin,hmax,bins,-1,-1);     // resolution
  TH2F* hresv = (TH2F*)hresr->Clone("hresv"); // resolution with vertex
    
  trackSim->Project("hbiar","ptr:pts",    base + "nhits>=3 && nrec==1" +
pid);
  trackSim->Project("hbiav","ptv:pts",    base + "nhits>=3 && nrec==1" +
pid);
  trackSim->Project("hresr","ptr-pts:pts",base + "nhits>=3 && nrec==1" +
pid);
  trackSim->Project("hresv","ptv-pts:pts",base + "nhits>=3 && nrec==1" +
pid);

  hbiar->FitSlicesY(0,0,0);
  hbiav->FitSlicesY(0,0,0);
  hresr->FitSlicesY(0,0,0);
  hresv->FitSlicesY(0,0,0);

  TH1F *hbiar_1 = (TH1F*)gDirectory->Get("hbiar_1");
  TH1F *hbiav_1 = (TH1F*)gDirectory->Get("hbiav_1");
  TH1F *hresr_2 = (TH1F*)gDirectory->Get("hresr_2");
  TH1F *hresv_2 = (TH1F*)gDirectory->Get("hresv_2");
}

/****************************************************************************/
void analyzer()
{
  // Productions
  const char *prods[2] = {"pp","PbPb_pix"};

  // Particles
  const char *parts[3] = {"pion","kaon","prot"};

  // Variables
  const char *vars[2] = {"Eta","Pt"};


  for(int i = 0; i < 1; i ++) // !!!
  for(int j = 0; j < 1; j ++) // !!!
  {
    cerr << " " << prods[i] << " " << parts[j] << endl;

    for(int k = 0; k < 2; k++)
      analyzeSimTracks1D(prods[i], parts[j], vars[k]); 

//    analyzeSimTracks2D(prods[i], parts[j]); 
  }
}
