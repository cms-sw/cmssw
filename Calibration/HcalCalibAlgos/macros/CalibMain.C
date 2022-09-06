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
  if (argc < 8) {
    std::cerr << "Please give N arguments \n"
              << "Mode (0 CalibMonitor; 1 CalibProperties; 2 CalibTree)\n"
              << "Input File Name\n"
              << "Output File Name(ROOT)\n"
              << "Correction File Name\n"
              << "Directory Name\n"
              << "Duplicate File Name\n"
              << "Prefix\n"
              << "PUcorr\n"
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

  if (mode == 0) {
    // CalibMonitor
    bool datamc = (argc > 8) ? (std::atoi(argv[8]) < 1) : true;
    int numb = (argc > 9) ? std::atoi(argv[9]) : 50;
    int truncate = (argc > 10) ? std::atoi(argv[10]) : 0;
    const char* corrfile = (argc > 11) ? argv[11] : "";
    const char* dupfile = (argc > 12) ? argv[12] : "";
    const char* comfile = (argc > 13) ? argv[13] : "";
    const char* outfile = (argc > 14) ? argv[14] : "";
    const char* rcorfile = (argc > 15) ? argv[15] : "";
    bool usegen = (argc > 16) ? (std::atoi(argv[16]) < 1) : false;
    double scale = (argc > 17) ? std::atof(argv[17]) : 1.0;
    int usescale = (argc > 18) ? std::atoi(argv[18]) : 0;
    int etalo = (argc > 19) ? std::atoi(argv[19]) : 0;
    int etahi = (argc > 20) ? std::atoi(argv[20]) : 30;
    int runlo = (argc > 21) ? std::atoi(argv[21]) : 0;
    int runhi = (argc > 22) ? std::atoi(argv[22]) : 99999999;
    int phimin = (argc > 23) ? std::atoi(argv[23]) : 1;
    int phimax = (argc > 24) ? std::atoi(argv[24]) : 30;
    int zside = (argc > 25) ? std::atoi(argv[25]) : 1;
    int nvxlo = (argc > 26) ? std::atoi(argv[26]) : 0;
    int nvxhi = (argc > 27) ? std::atoi(argv[27]) : 1000;
    int rbx = (argc > 28) ? std::atoi(argv[28]) : 0;
    bool exclude = (argc > 29) ? (std::atoi(argv[29]) > 0) : false;
    bool etamax = (argc > 30) ? (std::atoi(argv[30]) > 0) : false;
    bool append = (argc > 31) ? (std::atoi(argv[31]) > 0) : true;
    bool all = (argc > 32) ? (std::atoi(argv[32]) > 0) : true;
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
    c1.Loop();
    c1.savePlot(histfile, append, all);
  } else if (mode == 1) {
    // CalibPlotProperties
    bool datamc = (argc > 8) ? (std::atoi(argv[8]) < 1) : true;
    int truncate = (argc > 9) ? std::atoi(argv[9]) : 0;
    const char* corrfile = (argc > 10) ? argv[10] : "";
    const char* dupfile = (argc > 11) ? argv[11] : "";
    const char* rcorfile = (argc > 12) ? argv[12] : "";
    bool usegen = (argc > 13) ? (std::atoi(argv[13]) < 1) : false;
    double scale = (argc > 14) ? std::atof(argv[14]) : 1.0;
    int usescale = (argc > 15) ? std::atoi(argv[15]) : 0;
    int etalo = (argc > 16) ? std::atoi(argv[16]) : 0;
    int etahi = (argc > 17) ? std::atoi(argv[17]) : 30;
    int runlo = (argc > 18) ? std::atoi(argv[18]) : 0;
    int runhi = (argc > 19) ? std::atoi(argv[19]) : 99999999;
    int phimin = (argc > 20) ? std::atoi(argv[20]) : 1;
    int phimax = (argc > 21) ? std::atoi(argv[21]) : 30;
    int zside = (argc > 22) ? std::atoi(argv[22]) : 1;
    int nvxlo = (argc > 23) ? std::atoi(argv[23]) : 0;
    int nvxhi = (argc > 24) ? std::atoi(argv[24]) : 1000;
    int rbx = (argc > 25) ? std::atoi(argv[25]) : 0;
    bool exclude = (argc > 26) ? (std::atoi(argv[26]) > 0) : false;
    bool etamax = (argc > 27) ? (std::atoi(argv[27]) > 0) : false;
    bool append = (argc > 28) ? (std::atoi(argv[28]) > 0) : true;
    bool all = (argc > 29) ? (std::atoi(argv[29]) > 0) : true;
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
    c1.Loop();
    c1.savePlot(histfile, append, all, debug);
  } else {
    // CalibTree
    int truncate = (argc > 8) ? std::atoi(argv[8]) : 0;
    int maxIter = (argc > 9) ? std::atoi(argv[9]) : 30;
    const char* corrfile = (argc > 10) ? argv[10] : "";
    int applyl1 = (argc > 11) ? std::atoi(argv[11]) : 1;
    double l1cut = (argc > 12) ? std::atof(argv[12]) : 0.5;
    const char* treename = (argc > 13) ? argv[13] : "CalibTree";
    const char* dupfile = (argc > 14) ? argv[14] : "";
    const char* rcorfile = (argc > 15) ? argv[15] : "";
    bool useiter = (argc > 16) ? (std::atoi(argv[16]) < 1) : true;
    bool useweight = (argc > 17) ? (std::atoi(argv[17]) < 1) : true;
    bool usemean = (argc > 18) ? (std::atoi(argv[18]) < 1) : false;
    int nmin = (argc > 19) ? std::atoi(argv[19]) : 0;
    bool inverse = (argc > 20) ? (std::atoi(argv[20]) < 1) : true;
    double ratmin = (argc > 21) ? std::atof(argv[21]) : 0.25;
    double ratmax = (argc > 22) ? std::atof(argv[22]) : 3.0;
    int ietamax = (argc > 23) ? std::atoi(argv[23]) : 25;
    int ietatrack = (argc > 24) ? std::atoi(argv[24]) : -1;
    int sysmode = (argc > 25) ? std::atoi(argv[25]) : -1;
    int rcorform = (argc > 26) ? std::atoi(argv[26]) : 0;
    bool usegen = (argc > 27) ? (std::atoi(argv[27]) < 1) : false;
    int runlo = (argc > 28) ? std::atoi(argv[28]) : 0;
    int runhi = (argc > 29) ? std::atoi(argv[29]) : 99999999;
    int phimin = (argc > 30) ? std::atoi(argv[30]) : 1;
    int phimax = (argc > 31) ? std::atoi(argv[31]) : 30;
    int zside = (argc > 32) ? std::atoi(argv[32]) : 0;
    int nvxlo = (argc > 33) ? std::atoi(argv[33]) : 0;
    int nvxhi = (argc > 34) ? std::atoi(argv[34]) : 1000;
    int rbx = (argc > 35) ? std::atoi(argv[35]) : 0;
    bool exclude = (argc > 36) ? (std::atoi(argv[36]) > 0) : false;
    int higheta = (argc > 37) ? std::atoi(argv[37]) : 1;
    double fraction = (argc > 38) ? std::atof(argv[38]) : 1.0;
    bool writehisto = (argc > 39) ? (std::atoi(argv[39]) > 0) : false;
    bool debug = (argc > 40) ? (std::atoi(argv[40]) > 0) : false;

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
      t.getDetId(fraction, ietatrack, debug);

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
                            debug);
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
      t.makeplots(ratmin, ratmax, ietamax, useweight, fraction, debug);
      fout->Close();
    }
  }

  return 0;
}
