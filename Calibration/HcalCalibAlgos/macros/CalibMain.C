////////////////////////////////////////////////////////////////////////////////
//
//  This provides a provision of executing CalibMonitor, CalibProperties,
//  CalibTree, or CalibSplit in batch operation.
//
//  Usage:
//  ./calibMain.exe <mode> <other parameters depending on mode>
//  mode = 0:CalibMonitor, 1:CalibProperties, 2:CalibTree; 3: CalibSplit
//
//  Other parameters for CalibMonitor:
//  <InputFile> <HistogramFile> <Flag> <DirectoryName> <Prefix> <PUcorr>
//  <Truncate> <Nmax> <datamc> <numb> <usegen> <scale> <usescale> <etalo>
//  <etahi> <runlo> <runhi> <phimin> <phimax> <zside> <nvxlo> <nvxhi> <rbx>
//  <exclude> <etamax> <append> <all> <corrfile> <rcorfile> <dupfile>
//  <comfile> <outfile>
//
//  Other parameters for CalibProperties:
//  <InputFile> <HistogramFile> <Flag> <DirectoryName> <Prefix> <PUcorr>
//  <Truncate> <Nmax> <datamc> <usegen> <scale> <usescale> <etalo> <etahi>
//  <runlo> <runhi> <phimin> <phimax> <zside> <nvxlo> <nvxhi> <rbx>
//  <exclude> <etamax> <append> <all> <corrfile> <rcorfile> <dupfile>
//
//  Other parameters for CalibTree:
//  <InputFile> <OutputFile> <Flag> <DirectoryName> <Prefix> <PUcorr>
//  <Truncate> <Nmax> <maxIter> <corrfile> <applyl1> <l1cut> <useiter>
//  <useweight> <usemean> <nmin> <inverse> <ratmin> <ratmax> <ietamax>
//  <ietatrack> <sysmode> <rcorform> <usegen> <runlo> <runhi> <phimin>
//  <phimax> <zside> <nvxlo> <nvxhi> <rbx> <exclude> <higheta> <fraction>
//  <writehisto> <debug> <rcorfile> <dupfile> <treename>
//
//  Other parameters for CalibSplit:
//  <InputFile> <HistogramFile> <Flag> <DirectoryName> <Prefix> <PUcorr>
//  <Truncate> <Nmax> <pmin> <pmax> <debug>
//
//
////////////////////////////////////////////////////////////////////////////////

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TH1D.h>
#include <TH2F.h>
#include <TProfile.h>
#include <TFitResult.h>
#include <TFitResultPtr.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TPaveStats.h>
#include <TPaveText.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <string>

void unpackDetId(unsigned int, int&, int&, int&, int&, int&);

#include "CalibMonitor.C"
#include "CalibPlotProperties.C"
#include "CalibTree.C"

int main(Int_t argc, Char_t* argv[]) {
  if (argc < 10) {
    std::cerr << "Please give N arguments \n"
              << "Mode (0 CalibMonitor; 1 CalibProperties; 2 CalibTree; 3 CalibSplit)\n"
              << "Input File Name\n"
              << "Output File Name(ROOT)\n"
              << "Flag\n"
              << "Directory Name\n"
              << "Prefix\n"
              << "PUcorr\n"
              << "Truncate\n"
              << "Nmax\n"
              << " .... Other parameters depending on mode\n"
              << std::endl;
    return -1;
  }

  int mode = std::atoi(argv[1]);
  const char* infile = argv[2];
  std::string histfile(argv[3]);
  int flag = std::atoi(argv[4]);
  const char* dirname = argv[5];
  std::string prefix(argv[6]);
  int pucorr = std::atoi(argv[7]);
  int truncate = std::atoi(argv[8]);
  Long64_t nmax = static_cast<Long64_t>(std::atoi(argv[9]));

  if (mode == 0) {
    // CalibMonitor
    bool datamc = (argc > 10) ? (std::atoi(argv[10]) > 0) : true;
    int numb = (argc > 11) ? std::atoi(argv[11]) : 50;
    bool usegen = (argc > 12) ? (std::atoi(argv[12]) > 0) : false;
    double scale = (argc > 13) ? std::atof(argv[13]) : 1.0;
    int usescale = (argc > 14) ? std::atoi(argv[14]) : 0;
    int etalo = (argc > 15) ? std::atoi(argv[15]) : 0;
    int etahi = (argc > 16) ? std::atoi(argv[16]) : 30;
    int runlo = (argc > 17) ? std::atoi(argv[17]) : 0;
    int runhi = (argc > 18) ? std::atoi(argv[18]) : 99999999;
    int phimin = (argc > 19) ? std::atoi(argv[19]) : 1;
    int phimax = (argc > 20) ? std::atoi(argv[20]) : 72;
    int zside = (argc > 21) ? std::atoi(argv[21]) : 1;
    int nvxlo = (argc > 22) ? std::atoi(argv[22]) : 0;
    int nvxhi = (argc > 23) ? std::atoi(argv[23]) : 1000;
    int rbx = (argc > 24) ? std::atoi(argv[24]) : 0;
    bool exclude = (argc > 25) ? (std::atoi(argv[25]) > 0) : false;
    bool etamax = (argc > 26) ? (std::atoi(argv[26]) > 0) : false;
    bool append = (argc > 27) ? (std::atoi(argv[27]) > 0) : true;
    bool all = (argc > 28) ? (std::atoi(argv[28]) > 0) : true;
    const char* corrfile = (argc > 29) ? argv[29] : "";
    const char* rcorfile = (argc > 30) ? argv[30] : "";
    const char* dupfile = (argc > 31) ? argv[31] : "";
    const char* comfile = (argc > 32) ? argv[32] : "";
    const char* outfile = (argc > 33) ? argv[33] : "";
    CalibMonitor c1(infile,
                    dirname,
                    dupfile,
                    comfile,
                    outfile,
                    prefix,
                    corrfile,
                    rcorfile,
                    pucorr,
                    flag,
                    numb,
                    datamc,
                    truncate,
                    usegen,
                    scale,
                    usescale,
                    etalo,
                    etahi,
                    runlo,
                    runhi,
                    phimin,
                    phimax,
                    zside,
                    nvxlo,
                    nvxhi,
                    rbx,
                    exclude,
                    etamax);
    c1.Loop(nmax);
    c1.savePlot(histfile, append, all);
  } else if (mode == 1) {
    // CalibPlotProperties
    bool datamc = (argc > 10) ? (std::atoi(argv[10]) > 0) : true;
    bool usegen = (argc > 11) ? (std::atoi(argv[11]) > 0) : false;
    double scale = (argc > 12) ? std::atof(argv[12]) : 1.0;
    int usescale = (argc > 13) ? std::atoi(argv[13]) : 0;
    int etalo = (argc > 14) ? std::atoi(argv[14]) : 0;
    int etahi = (argc > 15) ? std::atoi(argv[15]) : 30;
    int runlo = (argc > 16) ? std::atoi(argv[16]) : 0;
    int runhi = (argc > 17) ? std::atoi(argv[17]) : 99999999;
    int phimin = (argc > 18) ? std::atoi(argv[18]) : 1;
    int phimax = (argc > 19) ? std::atoi(argv[19]) : 72;
    int zside = (argc > 20) ? std::atoi(argv[20]) : 1;
    int nvxlo = (argc > 21) ? std::atoi(argv[21]) : 0;
    int nvxhi = (argc > 22) ? std::atoi(argv[22]) : 1000;
    int rbx = (argc > 23) ? std::atoi(argv[23]) : 0;
    bool exclude = (argc > 24) ? (std::atoi(argv[24]) > 0) : false;
    bool etamax = (argc > 25) ? (std::atoi(argv[25]) > 0) : false;
    bool append = (argc > 26) ? (std::atoi(argv[26]) > 0) : true;
    bool all = (argc > 27) ? (std::atoi(argv[27]) > 0) : true;
    const char* corrfile = (argc > 28) ? argv[28] : "";
    const char* rcorfile = (argc > 29) ? argv[29] : "";
    const char* dupfile = (argc > 30) ? argv[30] : "";
    bool debug(false);
    CalibPlotProperties c1(infile,
                           dirname,
                           dupfile,
                           prefix,
                           corrfile,
                           rcorfile,
                           pucorr,
                           flag,
                           datamc,
                           truncate,
                           usegen,
                           scale,
                           usescale,
                           etalo,
                           etahi,
                           runlo,
                           runhi,
                           phimin,
                           phimax,
                           zside,
                           nvxlo,
                           nvxhi,
                           rbx,
                           exclude,
                           etamax);
    c1.Loop(nmax);
    c1.savePlot(histfile, append, all, debug);
  } else if (mode == 2) {
    // CalibTree
    int maxIter = (argc > 10) ? std::atoi(argv[10]) : 30;
    const char* corrfile = (argc > 11) ? argv[11] : "";
    int applyl1 = (argc > 12) ? std::atoi(argv[12]) : 1;
    double l1cut = (argc > 13) ? std::atof(argv[13]) : 0.5;
    bool useiter = (argc > 14) ? (std::atoi(argv[14]) > 0) : true;
    bool useweight = (argc > 15) ? (std::atoi(argv[15]) > 0) : true;
    bool usemean = (argc > 16) ? (std::atoi(argv[16]) > 0) : false;
    int nmin = (argc > 17) ? std::atoi(argv[17]) : 0;
    bool inverse = (argc > 18) ? (std::atoi(argv[18]) > 0) : true;
    double ratmin = (argc > 19) ? std::atof(argv[19]) : 0.25;
    double ratmax = (argc > 20) ? std::atof(argv[20]) : 3.0;
    int ietamax = (argc > 21) ? std::atoi(argv[21]) : 25;
    int ietatrack = (argc > 22) ? std::atoi(argv[22]) : -1;
    int sysmode = (argc > 23) ? std::atoi(argv[23]) : -1;
    int rcorform = (argc > 24) ? std::atoi(argv[24]) : 0;
    bool usegen = (argc > 25) ? (std::atoi(argv[25]) > 0) : false;
    int runlo = (argc > 26) ? std::atoi(argv[26]) : 0;
    int runhi = (argc > 27) ? std::atoi(argv[27]) : 99999999;
    int phimin = (argc > 28) ? std::atoi(argv[28]) : 1;
    int phimax = (argc > 29) ? std::atoi(argv[29]) : 72;
    int zside = (argc > 30) ? std::atoi(argv[30]) : 0;
    int nvxlo = (argc > 31) ? std::atoi(argv[31]) : 0;
    int nvxhi = (argc > 32) ? std::atoi(argv[32]) : 1000;
    int rbx = (argc > 33) ? std::atoi(argv[33]) : 0;
    bool exclude = (argc > 34) ? (std::atoi(argv[34]) > 0) : false;
    int higheta = (argc > 35) ? std::atoi(argv[35]) : 1;
    double fraction = (argc > 36) ? std::atof(argv[36]) : 1.0;
    bool writehisto = (argc > 37) ? (std::atoi(argv[37]) > 0) : false;
    bool debug = (argc > 38) ? (std::atoi(argv[38]) > 0) : false;
    const char* rcorfile = (argc > 39) ? argv[39] : "";
    const char* dupfile = (argc > 40) ? argv[40] : "";
    const char* treename = (argc > 41) ? argv[41] : "CalibTree";

    char name[500];
    sprintf(name, "%s/%s", dirname, treename);
    TChain* chain = new TChain(name);
    std::cout << "Create a chain for " << name << " from " << infile << std::endl;
    if (!fillChain(chain, infile)) {
      std::cout << "*****No valid tree chain can be obtained*****" << std::endl;
    } else {
      std::cout << "Proceed with a tree chain with " << chain->GetEntries() << " entries" << std::endl;
      Long64_t nentryTot = chain->GetEntries();
      Long64_t nentries = (fraction > 0.01 && fraction < 0.99) ? (Long64_t)(fraction * nentryTot) : nentryTot;
      static const int maxIterMax = 100;
      if (maxIter > maxIterMax)
        maxIter = maxIterMax;
      std::cout << "Tree " << name << " " << chain << " in directory " << dirname << " from file " << infile
                << " with nentries (tracks): " << nentries << std::endl;
      unsigned int k(0), kmax(maxIter);
      CalibTree t(dupfile,
                  rcorfile,
                  truncate,
                  useiter,
                  usemean,
                  runlo,
                  runhi,
                  phimin,
                  phimax,
                  zside,
                  nvxlo,
                  nvxhi,
                  sysmode,
                  rbx,
                  pucorr,
                  rcorform,
                  usegen,
                  exclude,
                  higheta,
                  chain);
      t.h_pbyE = new TH1D("pbyE", "pbyE", 100, -1.0, 9.0);
      t.h_Ebyp_bfr = new TProfile("Ebyp_bfr", "Ebyp_bfr", 60, -30, 30, 0, 10);
      t.h_Ebyp_aftr = new TProfile("Ebyp_aftr", "Ebyp_aftr", 60, -30, 30, 0, 10);
      t.h_cvg = new TH1D("Cvg0", "Convergence", kmax, 0, kmax);
      t.h_cvg->SetMarkerStyle(7);
      t.h_cvg->SetMarkerSize(5.0);

      TFile* fout = new TFile(histfile.c_str(), "RECREATE");
      std::cout << "Output file: " << histfile << " opened in recreate mode" << std::endl;
      fout->cd();

      double cvgs[maxIterMax], itrs[maxIterMax];
      t.getDetId(fraction, ietatrack, debug, nmax);

      for (; k <= kmax; ++k) {
        std::cout << "Calling Loop() " << k << "th time" << std::endl;
        double cvg = t.Loop(k,
                            fout,
                            useweight,
                            nmin,
                            inverse,
                            ratmin,
                            ratmax,
                            ietamax,
                            ietatrack,
                            applyl1,
                            l1cut,
                            k == kmax,
                            fraction,
                            writehisto,
                            debug,
                            nmax);
        itrs[k] = k;
        cvgs[k] = cvg;
        if (cvg < 0.00001)
          break;
      }

      t.writeCorrFactor(corrfile, ietamax);

      fout->cd();
      TGraph* g_cvg;
      g_cvg = new TGraph(k, itrs, cvgs);
      g_cvg->SetMarkerStyle(7);
      g_cvg->SetMarkerSize(5.0);
      g_cvg->Draw("AP");
      g_cvg->Write("Cvg");
      std::cout << "Finish looping after " << k << " iterations" << std::endl;
      t.makeplots(ratmin, ratmax, ietamax, useweight, fraction, debug, nmax);
      fout->Close();
    }
  } else {
    // CalibSplit
    double pmin = (argc > 10) ? std::atof(argv[10]) : 40.0;
    double pmax = (argc > 11) ? std::atof(argv[11]) : 60.0;
    bool debug = (argc > 12) ? (std::atoi(argv[12]) > 0) : false;
    CalibSplit c1(infile, dirname, histfile, pmin, pmax, debug);
    c1.Loop(nmax);
  }
  return 0;
}
