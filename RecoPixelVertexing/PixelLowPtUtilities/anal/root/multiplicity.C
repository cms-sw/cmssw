#include "analyzer.h"

/****************************************************************************/
void multiplicity()
{
  // recHits
  TChain *recHits = new TChain("multi");
  recHits->Add("../../test/multi.root");

  // simTrack
  TChain *simTracks = new TChain("trackSim");
  simTracks->Add("../../test/multi.root");

  int bins = 60;
  float emin = -3., emax = 3.;
  float amin =  0., amax = 250e+3.;

  TH2F* hbac = new TH2F("hbac","hbac", bins,emin,emax, bins,amin,amax);  
  TH2F* hpri = (TH2F*)hbac->Clone("hpri");
  TH2F* hloo = (TH2F*)hbac->Clone("hloo");

  TH2F* ebac = (TH2F*)hbac->Clone("ebac");
  TH2F* epri = (TH2F*)hbac->Clone("epri");
  TH2F* eloo = (TH2F*)hbac->Clone("eloo");

  TH2F* hall = (TH2F*)hbac->Clone("hall");

  TCut base = "eloss<300e+3";

  recHits->Project("hbac","eloss:eta", base + "type==1");
  recHits->Project("hpri","eloss:eta", base + "type==2");
  recHits->Project("hloo","eloss:eta", base + "type==3");

  recHits->Project("ebac","eloss:eta", base + "type==-1");
  recHits->Project("epri","eloss:eta", base + "type==-2");
  recHits->Project("eloo","eloss:eta", base + "type==-3");

  recHits->Project("hall","eloss:eta", base);

  // Write out
  printToFile(hbac,"../out/ppMulti/bac.dat");
  printToFile(hpri,"../out/ppMulti/pri.dat");
  printToFile(hloo,"../out/ppMulti/loo.dat");

  printToFile(ebac,"../out/ppMulti/ebac.dat");
  printToFile(epri,"../out/ppMulti/epri.dat");
  printToFile(eloo,"../out/ppMulti/eloo.dat");

  printToFile(hall,"../out/ppMulti/all.dat");


  TH1F* hsim = new TH1F("hsim","hsim", bins,emin,emax);
  TH1F* hsia = (TH1F*)hsim->Clone("hsia");
  TH1F* hrec = (TH1F*)hsim->Clone("hrec");
 
  simTracks->Project("hsim", "eta", "nvtx==1 && q!=0 && rho<0.2");
  simTracks->Project("hsia", "eta",            "q!=0 && rho<0.2");
  recHits->Project  ("hrec", "eta", "eloss*1e-3 > 21*cosh(eta) - 11");

  printToFile(hsim,"../out/ppMulti/sim.dat");
  printToFile(hsia,"../out/ppMulti/sia.dat");
  printToFile(hrec,"../out/ppMulti/rec.dat");

  hrec->Sumw2(); hrec->Divide(hrec,hsim,1,1,"B");
  printToFile(hrec,"../out/ppMulti/rat.dat");
}

