#include "analyzer.h"

// EtaMax = 1.6
#define Factor 8

static char sourceName[256], fileName[256], path[256];

/****************************************************************************/
void analyzeSimTracks(TChain *trackSim, const char *part, const char *var)
{
  // Selection on simulated
  TCut base;
  float hmin,hmax, emin,emax, pmin,pmax;
  char varName[128];

  if(var == "Pt" )
  {
    base = "rhos<0.2 && abs(etas)<1.6";
    hmin =  0; hmax = 2; sprintf(varName,"%s", "pts");
  }
  if(var == "LogPt" )
  {
    base = "rhos<0.2 && abs(etas)<1.6";
    hmin = log10(0.1); hmax = log10(10.); sprintf(varName,"%s", "log10(pts)");
  }
  if(var == "Eta")
  {
    base = "rhos<0.2";
    hmin = -3; hmax = 3; sprintf(varName,"%s", "etas");
  }
  if(var == "EtaPt")
  {
    base = "rhos<0.2";
    emin = -3; emax = 3;
    pmin =  0; pmax = 2; sprintf(varName,"%s", "pts:etas");
  }

  // Particle specific, bins
  TCut pid; int bins;
  if(part == "pion") { pid = "abs(ids)== 211"; bins = 200; }
  if(part == "kaon") { pid = "abs(ids)== 321"; bins = 100; }
  if(part == "prot") { pid = "abs(ids)==2212"; bins = 100; }

  if(var == "EtaPt") bins /= 2; // Less bins for 2D
  bins /= Factor;

  // Declare histograms
  if(var != "EtaPt")
  {
    TH1F* hsim = new TH1F("hsim","hsim",
                             bins,hmin,hmax); // simulated
    TH1F* hacc = (TH1F*)hsim->Clone("hacc");  // accepted
    TH1F* href = (TH1F*)hsim->Clone("href");  // reconstructed/efficiency
    TH1F* hmul = (TH1F*)hsim->Clone("hmul");  // multiply reconstructed
  }
  else
  {
    TH2F* hsim = new TH2F("hsim","hsim",
              bins,emin,emax,bins,pmin,pmax); // simulated
    TH2F* hacc = (TH2F*)hsim->Clone("hacc");  // accepted
    TH2F* href = (TH2F*)hsim->Clone("href");  // reconstructed/efficiency
    TH2F* hmul = (TH2F*)hsim->Clone("hmul");  // multiply reconstructed
  }

  // Fill histograms
  trackSim->Project("hsim", varName, base + pid);
  trackSim->Project("hacc", varName, base + pid + "nhits>=3");
  trackSim->Project("href", varName, base + pid + "nhits>=3 && nrec>0");
  trackSim->Project("hmul", varName, base + pid + "nhits>=3 && nrec>1");

  // Ratios
  hmul->Sumw2(); hmul->Divide(hmul,href,1,1,"B"); // multiple counting
  href->Sumw2(); href->Divide(href,hacc,1,1,"B"); // efficiency
  hacc->Sumw2(); hacc->Divide(hacc,hsim,1,1,"B"); // acceptance

  // Write out
  sprintf(fileName,"%s/algoEffic_%s_%s.dat", path,var,part);
  printToFile(href,fileName);

  sprintf(fileName,"%s/multCount_%s_%s.dat", path,var,part);
  printToFile(hmul,fileName);

  sprintf(fileName,"%s/geomAccep_%s_%s.dat", path,var,part);
  printToFile(hacc,fileName);

  delete hsim;
  delete hacc;
  delete href;
  delete hmul;
}

/****************************************************************************/
void analyzeRecTracks(TChain *trackRec, const char *part)
{
  // ptr replaced by by ptv for resolution plots

  TCut base = "nsim==1 && rhos<0.2 && abs(etas)<1.6";

  float hmin=0,hmax=2;

  // Particle specific, bins
  TCut pid; int bins;
  if(part == "pion") { pid = "abs(ids)== 211"; bins = 100; }
  if(part == "kaon") { pid = "abs(ids)== 321"; bins =  50; }
  if(part == "prot") { pid = "abs(ids)==2212"; bins =  50; }

  bins /= Factor;

  // Declare histograms
  TH2F* hbia = new TH2F("hbia","hbia", bins,hmin,hmax,bins,hmin,hmax); // bias
  TH2F* hres = new TH2F("hres","hres", bins,hmin,hmax,bins,-1,1);      // reso
  TH2F* hbir = new TH2F("hbir","hbir", bins,hmin,hmax,bins,hmin,hmax); // bias
  TH2F* hrer = new TH2F("hrer","hrer", bins,hmin,hmax,bins,-1,1);      // reso

  hmin = log10(0.1); hmax = log10(10.);
  TH2F* hbla = new TH2F("hbla","hbla", bins,hmin,hmax,bins,hmin,hmax); // bias
  TH2F* hrls = new TH2F("hrls","hrls", bins,hmin,hmax,bins,-1,1);      // reso
    
  // Fill histograms
  trackRec->Project("hbia","ptv:pts",    base + pid);
  trackRec->Project("hres","ptv-pts:pts",base + pid);
  trackRec->Project("hbir","pts:ptv",    base + pid);
  trackRec->Project("hrer","pts-ptv:ptv",base + pid);

  trackRec->Project("hbla","log10(ptv):log10(pts)",    base + pid);
  trackRec->Project("hrls","ptv-pts:log10(pts)",base + pid);

  // Fit slices with gaussian
  hbia->FitSlicesY(0,0,0); TH1F *hbia_1 = (TH1F*)gDirectory->Get("hbia_1");
  hres->FitSlicesY(0,0,0); TH1F *hres_2 = (TH1F*)gDirectory->Get("hres_2");
  hbir->FitSlicesY(0,0,0); TH1F *hbir_1 = (TH1F*)gDirectory->Get("hbir_1");
  hrer->FitSlicesY(0,0,0); TH1F *hrer_2 = (TH1F*)gDirectory->Get("hrer_2");

  hbla->FitSlicesY(0,0,0); TH1F *hbla_1 = (TH1F*)gDirectory->Get("hbla_1");
  hrls->FitSlicesY(0,0,0); TH1F *hrls_2 = (TH1F*)gDirectory->Get("hrls_2");

  // Write out
  sprintf(fileName,"%s/ptFit_%s.dat",     path,part);
  printToFile(hbia,fileName);

  sprintf(fileName,"%s/ptBias_%s_%s.dat", path,part);
  printToFile(hbia_1,fileName);
  sprintf(fileName,"%s/ptReso_%s_%s.dat", path,part);
  printToFile(hres_2,fileName);
  sprintf(fileName,"%s/ptBiar_%s_%s.dat", path,part);
  printToFile(hbir_1,fileName);
  sprintf(fileName,"%s/ptResr_%s_%s.dat", path,part);
  printToFile(hrer_2,fileName);

  sprintf(fileName,"%s/ptLogBias_%s_%s.dat", path,part);
  printToFile(hbla_1,fileName);
  sprintf(fileName,"%s/ptLogReso_%s_%s.dat", path,part);
  printToFile(hrls_2,fileName);

  delete hbia; delete hres;
  delete hbir; delete hrer;

  delete hbla; delete hrls;
}


/****************************************************************************/
void analyzeFakeTracks(TChain *trackRec, const char* var) 
{
  TCut base;
  float hmin,hmax, emin,emax,pmin,pmax;
  char varName[128];

  // added chi2 cut!

  if(var == "Pt" )
  {
    base = "abs(d0r)<1 && abs(etar)<1.6";
    hmin =  0; hmax = 2; sprintf(varName,"%s", "ptr");
  }
  if(var == "LogPt" )
  { 
    base = "abs(d0r)<1 && abs(etar)<1.6";
    hmin = log10(0.1); hmax = log10(10.); sprintf(varName,"%s", "log10(ptr)");
  }
  if(var == "Eta")
  {
    base = "abs(d0r)<1";
    hmin = -3; hmax = 3; sprintf(varName,"%s", "etar");
  }
  if(var == "EtaPt")
  {
    base = "abs(d0r)<1";
    emin = -3; emax = 3;
    pmin =  0; pmax = 2; sprintf(varName,"%s", "ptr:etar");
  }

  int bins = 100;
  if(var == "EtaPt") bins /= 2; // Less bins for 2D
  bins /= Factor;

  // Declare histograms
  if(var != "EtaPt")
  {
    TH1F* hrec = new TH1F("hrec","hrec",bins,hmin,hmax); // reconstucted
    TH1F* hfak = (TH1F*)hrec->Clone("hfak");             // fake
  }
  else
  {
    TH2F* hrec = new TH2F("hrec","hrec",
              bins,emin,emax,bins,pmin,pmax); // reconstructed
    TH2F* hfak = (TH2F*)hrec->Clone("hfak");  // fake
  }

  // Fill histograms
  trackRec->Project("hrec", varName, base            );
  trackRec->Project("hfak", varName, base + "nsim==0");

  // Ratios
  hfak->Sumw2(); hfak->Divide(hfak,hrec,1,1,"B");

  // Write out
  sprintf(fileName,"%s/fakeRate_%s.dat",path,var);
  printToFile(hfak,fileName);
 
  delete hrec;
  delete hfak;
}

/****************************************************************************/
void analyzeFeedDown(TChain *trackRec, const char* var,
                     const char* moth, const char* daug)
{
  TCut base;
  float hmin,hmax, emin,emax,pmin,pmax;
  char varName[128];

  if(var == "Pt" )
  {
    base = "nsim==1 && rhos<0.2 && abs(d0r)<1 && abs(etar)<1.6";
    hmin =  0; hmax = 2; sprintf(varName,"%s", "ptr");
  }
  if(var == "LogPt" )
  {
    base = "nsim==1 && rhos<0.2 && abs(d0r)<1 && abs(etar)<1.6";
    hmin = log10(0.1); hmax = log10(10.); sprintf(varName,"%s", "log10(ptr)");
  }
  if(var == "Eta")
  {
    base = "nsim==1 && rhos<0.2 && abs(d0r)<1";
    hmin = -3; hmax = 3; sprintf(varName,"%s", "etar");
  }
  if(var == "EtaPt")
  {
    base = "nsim==1 && rhos<0.2 && abs(d0r)<1";
    emin = -3; emax = 3;
    pmin =  0; pmax = 2; sprintf(varName,"%s", "ptr:etar");
  }
 
  TCut mothpid; 
  if(moth == "k0s"   ) mothpid = "parids == 310";
  if(moth == "lambda") mothpid = "abs(parids) == 3122";

  TCut daugpid;
  if(daug == "pion") daugpid = "abs(ids) == 211";
  if(daug == "prot") daugpid = "abs(ids) == 2212";

  int bins = 20;
  if(var == "EtaPt") bins /= 2; // Less bins for 2D
  bins /= Factor;

  // Declare histograms
  if(var != "EtaPt")
  {
    TH1F* hrec = new TH1F("hrec","hrec",bins,hmin,hmax); // reconstucted
    TH1F* hdec = (TH1F*)hrec->Clone("hdec");             // decay product
  }
  else
  {
    TH2F* hrec = new TH2F("hrec","hrec",
              bins,emin,emax,bins,pmin,pmax); // reconstructed
    TH2F* hdec = (TH2F*)hrec->Clone("hdec");  // decay product
  }

  // Fill histograms
  trackRec->Project("hrec", varName, base + daugpid);
  trackRec->Project("hdec", varName, base + daugpid + mothpid);

  // Ratios
  hdec->Sumw2(); hdec->Divide(hdec,hrec,1,1,"B");

  sprintf(fileName,"%s/feedDown_%s_%s-%s.dat",path,var,moth,daug);
  printToFile(hdec,fileName);

  delete hrec;
  delete hdec;
}

/****************************************************************************/
void analyzer()
{
  // Track type
  const char *types[2] = {"Pixel","Global"};

  // Productions
  const char *prods[2] = {"pp","PbPb"};

  // Particles
  const char *parts[3] = {"pion","kaon","prot"};

  // Variables
  const char *vars[4]  = {"Eta","Pt","LogPt","EtaPt"};

  for(int h = 0; h < 2; h++) // type
  {
    cerr << "========================================================" << endl;
    cerr << types[h] << endl;

    for(int i = 1; i < 2; i++) // prod
    {
      sprintf(sourceName, "/tmp/sikler/result%sTracks_%s_*.root",
                          types[h], prods[i]);

      sprintf(path, "../out/%s%s", prods[i],types[h]);

      //////////////////////////////////////////////////////////////// 
      // trackSim
      cerr << " chaining trackSim..";
      TChain *trackSim = new TChain("trackSim");
      trackSim->Add(sourceName);
      cerr << " [done]" << endl;;

      for(int j = 0; j < 3; j++) // part
      for(int k = 0; k < 4; k++) // var
      {
        cerr << " " << prods[i] << " " << parts[j] << " " << vars[k] << endl;
        analyzeSimTracks(trackSim, parts[j], vars[k]); 
      }

      delete trackSim;
 
      //////////////////////////////////////////////////////////////// 
      // trackRec
      cerr << " chaining trackRec..";
      TChain *trackRec = new TChain("trackRec");
      trackRec->Add(sourceName);
      cerr << " [done]" << endl;

      for(int j = 0; j < 3; j++) // part
      { 
        cerr << " " << prods[i] << " " << parts[j] << " ptBias/ptReso" << endl;
        analyzeRecTracks(trackRec, parts[j]); 
      }

      for(int k = 0; k < 4; k++) // var
      {
        cerr << " " << prods[i] << " " << vars[k] << " feed-down" << endl;
        analyzeFeedDown(trackRec, vars[k], "k0s",    "pion"); 
        analyzeFeedDown(trackRec, vars[k], "lambda", "prot"); 
        analyzeFeedDown(trackRec, vars[k], "lambda", "pion"); 
      }
  
      for(int k = 0; k < 4; k++) // var
      {
        cerr << " " << prods[i] << " " << vars[k] << " fakeRate" << endl;
        analyzeFakeTracks(trackRec, vars[k]); 
      }

      delete trackRec;
    }
  } 
}

