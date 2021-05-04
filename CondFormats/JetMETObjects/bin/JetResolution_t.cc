////////////////////////////////////////////////////////////////////////////////
//
// JetResolution_t
// ---------------
//
//            11/05/2010 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////

#include "CondFormats/JetMETObjects/interface/JetResolution.h"

#include <TROOT.h>
#include <TApplication.h>
#include <TRandom3.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TH1F.h>
#include <TMath.h>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>

using namespace std;

//______________________________________________________________________________
int main(int argc, char** argv) {
  if (argc > 1 && string(argv[1]) == "--help") {
    cout << "USAGE: JetResolution_t --era <era> --alg <alg> --nevts <n> --gaussian" << endl;
    return 0;
  }

  string era("Spring10");
  string alg("AK5Calo");
  unsigned nevts(10000);
  bool doGaussian(false);

  for (int i = 1; i < argc; i++) {
    if (string(argv[i]) == "--era") {
      era = argv[i + 1];
      i++;
    } else if (string(argv[i]) == "--alg") {
      alg = argv[i + 1];
      i++;
    } else if (string(argv[i]) == "--nevts") {
      stringstream ss;
      ss << argv[i + 1];
      ss >> nevts;
      i++;
    } else if (string(argv[i]) == "--gaussian") {
      doGaussian = true;
    } else {
      cerr << "ERROR: unknown option '" << argv[i] << "'" << endl;
      return 0;
    }
  }

  cout << "era:      " << era << endl;
  cout << "alg:      " << alg << endl;
  cout << "nevts:    " << nevts << endl;
  cout << "gaussian: " << doGaussian << endl << endl;

  string cmssw_base(std::getenv("CMSSW_BASE"));
  string cmssw_release_base(std::getenv("CMSSW_RELEASE_BASE"));
  string path = cmssw_base + "/src/CondFormats/JetMETObjects/data";
  struct stat st;
  if (stat(path.c_str(), &st) != 0)
    path = cmssw_release_base + "/src/CondFormats/JetMETObjects/data";
  if (stat(path.c_str(), &st) != 0) {
    cerr << "ERROR: tried to set path but failed, abort." << endl;
    return 0;
  }

  string ptFileName = path + "/" + era + "_PtResolution_" + alg + ".txt";
  string etaFileName = path + "/" + era + "_EtaResolution_" + alg + ".txt";
  string phiFileName = path + "/" + era + "_PhiResolution_" + alg + ".txt";

  cout << ptFileName << endl;
  cout << etaFileName << endl;
  cout << phiFileName << endl;
  cout << endl;

  JetResolution ptResol(ptFileName, doGaussian);
  JetResolution etaResol(etaFileName, doGaussian);
  JetResolution phiResol(phiFileName, doGaussian);

  // SIMPLE TEST
  float pt = 200. * gRandom->Exp(0.1);
  float eta = gRandom->Uniform(-5.0, 5.0);
  float phi = gRandom->Uniform(-TMath::Pi(), TMath::Pi());

  cout << "pT=" << pt << " eta=" << eta << " phi=" << phi << endl;

  TF1* fPtResol = ptResol.resolutionEtaPt(eta, pt);
  TF1* fEtaResol = etaResol.resolutionEtaPt(eta, pt);
  TF1* fPhiResol = phiResol.resolutionEtaPt(eta, pt);

  fPtResol->Print();
  cout << endl;
  fEtaResol->Print();
  cout << endl;
  fPhiResol->Print();
  cout << endl;

  // GENERATE SPECTRA AND SMEAR
  TH1F* hRndPt = new TH1F("RndPt", ";random number", 200, 0.0, 5.0);
  TH1F* hGenPt = new TH1F("GenPt", ";p_{T} [GeV]", 100, 0., 250.);
  TH1F* hJetPt = new TH1F("JetPt", ";p_{T} [GeV]", 100, 0., 250.);

  TH1F* hRndEta = new TH1F("RndEta", ";random number", 200, -5.0, 5.0);
  TH1F* hGenEta = new TH1F("GenEta", ";#eta", 51, -5., 5.);
  TH1F* hJetEta = new TH1F("JetEta", ";#eta", 51, -5., 5.);

  TH1F* hRndPhi = new TH1F("RndPhi", ";random number", 200, -3.15, 3.15);
  TH1F* hGenPhi = new TH1F("GenPhi", ";#varphi", 41, -3.15, 3.15);
  TH1F* hJetPhi = new TH1F("JetPhi", ";#varphi", 41, -3.15, 3.15);

  for (unsigned ievt = 0; ievt < nevts; ievt++) {
    if ((ievt + 1) % 1000 == 0)
      cout << ievt + 1 << " events processed." << endl;

    float genpt = 200. * gRandom->Exp(0.1);
    if (genpt < 1.0)
      continue;
    float geneta = gRandom->Uniform(-5.0, 5.0);
    float genphi = gRandom->Uniform(-TMath::Pi(), TMath::Pi());

    fPtResol = ptResol.resolutionEtaPt(geneta, genpt);
    fEtaResol = etaResol.resolutionEtaPt(geneta, genpt);
    fPhiResol = phiResol.resolutionEtaPt(geneta, genpt);

    float rndpt = fPtResol->GetRandom();
    float rndeta = fEtaResol->GetRandom();
    float rndphi = fPhiResol->GetRandom();

    float jetpt = rndpt * genpt;
    float jeteta = rndeta + geneta;
    float jetphi = rndphi + genphi;

    hRndPt->Fill(rndpt);
    hGenPt->Fill(genpt);
    hJetPt->Fill(jetpt);

    hRndEta->Fill(rndeta);
    hGenEta->Fill(geneta);
    hJetEta->Fill(jeteta);

    hRndPhi->Fill(rndphi);
    hGenPhi->Fill(genphi);
    hJetPhi->Fill(jetphi);
  }

  // RUN ROOT APPLICATION AND DRAW BOTH DISTRIBUTIONS
  argc = 1;
  TApplication* app = new TApplication("JetResolution_t", &argc, argv);
  gROOT->SetStyle("Plain");

  // PLOT RESOLUTION FOR DIFFERENT ETA BINS
  TCanvas* cResolution = new TCanvas("Resolution", "Resolution", 0, 0, 1000, 400);
  cResolution->Divide(3, 1);

  cResolution->cd(1);
  gPad->SetLogx();

  TF1* fPtEta1 = ptResol.parameterEta("sigma", 0.25);
  TF1* fPtEta2 = ptResol.parameterEta("sigma", 1.75);
  TF1* fPtEta3 = ptResol.parameterEta("sigma", 2.75);

  fPtEta1->SetLineWidth(1);
  fPtEta2->SetLineWidth(1);
  fPtEta3->SetLineWidth(1);
  fPtEta1->SetNpx(500);
  fPtEta2->SetNpx(500);
  fPtEta3->SetNpx(500);
  fPtEta1->SetLineColor(kRed);
  fPtEta2->SetLineColor(kBlue);
  fPtEta3->SetLineColor(kGreen);
  fPtEta1->SetRange(5.0, 500.);
  fPtEta2->SetRange(5.0, 500.);
  fPtEta3->SetRange(5.0, 500.);
  fPtEta1->Draw();
  fPtEta2->Draw("SAME");
  fPtEta3->Draw("SAME");

  fPtEta1->SetTitle("p_{T} - Resolution");
  fPtEta1->GetHistogram()->SetXTitle("p_{T} [GeV]");
  fPtEta1->GetHistogram()->SetYTitle("#sigma(p_{T})/p_{T}");
  fPtEta1->GetHistogram()->GetYaxis()->CenterTitle();
  fPtEta1->GetHistogram()->GetYaxis()->SetTitleOffset(1.33);

  TLegend* legPtRes = new TLegend(0.6, 0.85, 0.85, 0.6);
  legPtRes->SetFillColor(10);
  legPtRes->AddEntry(fPtEta1, "#eta=0.25", "l");
  legPtRes->AddEntry(fPtEta2, "#eta=1.75", "l");
  legPtRes->AddEntry(fPtEta3, "#eta=2.75", "l");
  legPtRes->Draw();

  cResolution->cd(2);
  gPad->SetLogx();

  TF1* fEtaEta1 = etaResol.parameterEta("sigma", 0.25);
  TF1* fEtaEta2 = etaResol.parameterEta("sigma", 1.75);
  TF1* fEtaEta3 = etaResol.parameterEta("sigma", 2.75);

  fEtaEta1->SetLineWidth(1);
  fEtaEta2->SetLineWidth(1);
  fEtaEta3->SetLineWidth(1);
  fEtaEta1->SetNpx(500);
  fEtaEta2->SetNpx(500);
  fEtaEta3->SetNpx(500);
  fEtaEta1->SetLineColor(kRed);
  fEtaEta2->SetLineColor(kBlue);
  fEtaEta3->SetLineColor(kGreen);
  fEtaEta1->SetRange(5.0, 500.);
  fEtaEta2->SetRange(5.0, 500.);
  fEtaEta3->SetRange(5.0, 500.);
  fEtaEta1->Draw();
  fEtaEta2->Draw("SAME");
  fEtaEta3->Draw("SAME");

  fEtaEta1->SetTitle("#eta - Resolution");
  fEtaEta1->GetHistogram()->SetXTitle("p_{T} [GeV]");
  fEtaEta1->GetHistogram()->SetYTitle("#sigma(#eta)");
  fEtaEta1->GetHistogram()->GetYaxis()->CenterTitle();
  fEtaEta1->GetHistogram()->GetYaxis()->SetTitleOffset(1.33);

  TLegend* legEtaRes = new TLegend(0.6, 0.85, 0.85, 0.6);
  legEtaRes->SetFillColor(10);
  legEtaRes->AddEntry(fEtaEta1, "#eta=0.25", "l");
  legEtaRes->AddEntry(fEtaEta2, "#eta=1.75", "l");
  legEtaRes->AddEntry(fEtaEta3, "#eta=2.75", "l");
  legEtaRes->Draw();

  cResolution->cd(3);
  gPad->SetLogx();

  TF1* fPhiEta1 = phiResol.parameterEta("sigma", 0.25);
  TF1* fPhiEta2 = phiResol.parameterEta("sigma", 1.75);
  TF1* fPhiEta3 = phiResol.parameterEta("sigma", 2.75);

  fPhiEta1->SetLineWidth(1);
  fPhiEta2->SetLineWidth(1);
  fPhiEta3->SetLineWidth(1);
  fPhiEta1->SetNpx(500);
  fPhiEta2->SetNpx(500);
  fPhiEta3->SetNpx(500);
  fPhiEta1->SetLineColor(kRed);
  fPhiEta2->SetLineColor(kBlue);
  fPhiEta3->SetLineColor(kGreen);
  fPhiEta1->SetRange(5.0, 500.);
  fPhiEta2->SetRange(5.0, 500.);
  fPhiEta3->SetRange(5.0, 500.);
  fPhiEta1->Draw();
  fPhiEta2->Draw("SAME");
  fPhiEta3->Draw("SAME");

  fPhiEta1->SetTitle("#varphi - Resolution");
  fPhiEta1->GetHistogram()->SetXTitle("p_{T} [GeV]");
  fPhiEta1->GetHistogram()->SetYTitle("#sigma(#varphi)");
  fPhiEta1->GetHistogram()->GetYaxis()->CenterTitle();
  fPhiEta1->GetHistogram()->GetYaxis()->SetTitleOffset(1.33);

  TLegend* legPhiRes = new TLegend(0.6, 0.85, 0.85, 0.6);
  legPhiRes->SetFillColor(10);
  legPhiRes->AddEntry(fPhiEta1, "#eta=0.25", "l");
  legPhiRes->AddEntry(fPhiEta2, "#eta=1.75", "l");
  legPhiRes->AddEntry(fPhiEta3, "#eta=2.75", "l");
  legPhiRes->Draw();

  // PLOT GEN VS SMEARED DISTRIBUTIONS
  TCanvas* cSmearing = new TCanvas("Smearing", "Smearing", 100, 100, 1000, 600);
  cSmearing->Divide(3, 2);

  cSmearing->cd(1);
  gPad->SetLogy();
  hRndPt->Draw();

  cSmearing->cd(2);
  gPad->SetLogy();
  hRndEta->Draw();

  cSmearing->cd(3);
  gPad->SetLogy();
  hRndPhi->Draw();

  cSmearing->cd(4);
  gPad->SetLogy();
  hGenPt->Draw();
  hJetPt->Draw("SAME");
  hJetPt->SetLineColor(kRed);
  if (hGenPt->GetMaximum() < hJetPt->GetMaximum())
    hGenPt->SetMaximum(1.1 * hJetPt->GetMaximum());
  TLegend* legPt = new TLegend(0.6, 0.8, 0.85, 0.65);
  legPt->SetFillColor(10);
  legPt->AddEntry(hGenPt, "generated", "l");
  legPt->AddEntry(hJetPt, "smeared", "l");
  legPt->Draw();

  cSmearing->cd(5);
  hGenEta->Draw();
  hJetEta->Draw("SAME");
  hJetEta->SetLineColor(kRed);
  if (hGenEta->GetMaximum() < hJetEta->GetMaximum())
    hGenEta->SetMaximum(1.1 * hJetEta->GetMaximum());
  hGenEta->SetMinimum(0.0);
  hGenEta->SetMaximum(1.5 * hGenEta->GetMaximum());
  TLegend* legEta = new TLegend(0.6, 0.8, 0.85, 0.65);
  legEta->SetFillColor(10);
  legEta->AddEntry(hGenEta, "generated", "l");
  legEta->AddEntry(hJetEta, "smeared", "l");
  legEta->Draw();

  cSmearing->cd(6);
  hGenPhi->Draw();
  hJetPhi->Draw("SAME");
  hJetPhi->SetLineColor(kRed);
  if (hGenPhi->GetMaximum() < hJetPhi->GetMaximum())
    hGenPhi->SetMaximum(1.1 * hJetPhi->GetMaximum());
  hGenPhi->SetMinimum(0.0);
  hGenPhi->SetMaximum(1.5 * hGenPhi->GetMaximum());
  TLegend* legPhi = new TLegend(0.6, 0.8, 0.85, 0.65);
  legPhi->SetFillColor(10);
  legPhi->AddEntry(hGenPhi, "generated", "l");
  legPhi->AddEntry(hJetPhi, "smeared", "l");
  legPhi->Draw();

  app->Run();

  return 0;
}
