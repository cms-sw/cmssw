#include <ROOT/RDataFrame.hxx>
#include <ROOT/RVec.hxx>
#include <TCanvas.h>
#include <TF1.h>
#include <TFile.h>
#include <TH1D.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TLorentzVector.h>
#include <TMath.h>
#include <TProfile.h>
#include <TRegexp.h>
#include <TStopwatch.h>
#include <TString.h>
#include <TTree.h>
#include <algorithm>
#include <cmath>
#include <iomanip>  // Include the header for setw
#include <iostream>
#include <map>
#include <string>

// analysis types
enum anaKind { sagitta_t = 0, d0_t = 1, dz_t = 2 };

// number of bins for the correction map
unsigned int k_NBINS = 48;

// maximum number of iterations
int k_MAXITERATIONS = 30;

static constexpr const char* const analysis_types[] = {
    "sagitta",  //  0 - sagitta bias
    "d0",       //  1 - transverse IP bias
    "dz",       //  2 - longitudinal IPbias
};

// physics constants
static constexpr double k_muMass = 0.105658;  // GeV
static constexpr double MZ_PDG = 91.1876;     // in GeV

// ansatz from eq. 5 of https://arxiv.org/pdf/2212.07338.pdf
//________________________________________________________________
double pTcorrector(const double& unCorrPt, const double& delta, const float& charge) {
  return unCorrPt / (1 - charge * delta * unCorrPt);
}

//________________________________________________________________
float calcDeltaSagitta(float charge, float frac, float avgSagitta, float thisTrackPt) {
  return (-charge * 0.5 * frac * ((1 + charge * thisTrackPt * avgSagitta) / thisTrackPt));
}

//________________________________________________________________
std::pair<int, int> findEtaPhiBin(const TH2* h2, const float& eta, const float& phi) {
  const auto& etaBin = h2->GetXaxis()->FindBin(eta) - 1;
  const auto& phiBin = h2->GetYaxis()->FindBin(phi) - 1;

  if (etaBin < 0 || etaBin > int(k_NBINS - 1)) {
    std::cout << "etaBin: " << etaBin << " eta: " << eta << std::endl;
    return {-1., -1.};
  }

  if (phiBin < 0 || phiBin > int(k_NBINS - 1)) {
    std::cout << "phiBin: " << phiBin << " phi: " << phi << std::endl;
    return {-1., -1.};
  }
  return std::make_pair(etaBin, phiBin);
}

//________________________________________________________________
float updateSagittaMap(TTree* tree,
                       std::map<std::pair<int, int>, float>& theMap,
                       TH2F*& hSagitta,
                       const int iteration,
                       float oldMaxCorrection);

//________________________________________________________________
float updateSagittaMapFast(TTree* tree,
                           std::map<std::pair<int, int>, float>& theMap,
                           TH2F*& hSagitta,
                           const int iteration,
                           float oldMaxCorrection);

//________________________________________________________________
float updateIPMap(
    TTree* tree, std::map<std::pair<int, int>, float>& theMap, TH2F*& hIP, const int iteration, anaKind type);

//_______________________________________________________________
float updateIPMapFast(
    TTree* tree, std::map<std::pair<int, int>, float>& theMap, TH2F*& hIP, const int iteration, anaKind type);

// Enum to specify the formatting type
enum TitleFormatType { MEAN, RMS };

//________________________________________________________________
TString updateTitle(const TString& title, TitleFormatType formatType) {
  // Regular expression to match units in square brackets
  TRegexp re("\\[.*\\]");
  TString units;
  TString cleanTitle = title;

  // Check if units are present and extract them
  if (cleanTitle.Index(re) != kNPOS) {
    units = cleanTitle(re);                         // Extract the unit part
    cleanTitle(re) = "";                            // Remove the unit from the cleanTitle
    cleanTitle = cleanTitle.Strip(TString::kBoth);  // Remove extra spaces
  }

  // Construct the new title based on the formatType
  TString newTitle;
  switch (formatType) {
    case MEAN:
      if (!units.IsNull()) {
        newTitle = TString::Format("#LT%s#GT %s", cleanTitle.Data(), units.Data());
      } else {
        newTitle = TString::Format("#LT%s#GT", cleanTitle.Data());
      }
      break;
    case RMS:
      if (!units.IsNull()) {
        newTitle = TString::Format("RMS (%s) %s", cleanTitle.Data(), units.Data());
      } else {
        newTitle = TString::Format("RMS (%s)", cleanTitle.Data());
      }
      break;
  }

  // Set the new title to the x-axis of h_prof
  return newTitle;
}

//________________________________________________________________
std::pair<TH1F*, TH1F*> makeProfileVsEta(TH2F* h2) {
  const auto& nBinsX = h2->GetNbinsX();
  const auto& nBinsY = h2->GetNbinsY();

  float xmin = h2->GetXaxis()->GetXmin();
  float xmax = h2->GetXaxis()->GetXmax();

  std::vector<float> average;
  std::vector<float> RMS;
  std::vector<float> errorOnRMS;

  for (int i = 1; i <= nBinsX; i++) {
    float avgInY{0.};
    for (int j = 1; j <= nBinsY; j++) {
      avgInY += h2->GetBinContent(i, j);
    }
    avgInY /= nBinsY;
    average.push_back(avgInY);
  }

  for (int i = 1; i <= nBinsX; i++) {
    float RMSInY{0.};
    for (int j = 1; j <= nBinsY; j++) {
      RMSInY += std::pow((h2->GetBinContent(i, j) - average[i - 1]), 2);
    }
    RMSInY /= nBinsY;
    RMS.push_back(std::sqrt(RMSInY));
  }

  TH1F* h_prof = new TH1F("avgInEta", "profile vs #eta", nBinsX, xmin, xmax);
  h_prof->GetXaxis()->SetTitle(h2->GetXaxis()->GetTitle());
  h_prof->GetYaxis()->SetTitle(updateTitle(h2->GetZaxis()->GetTitle(), TitleFormatType::MEAN));
  for (int i = 1; i <= nBinsX; i++) {
    std::cout << i << " = " << average[i - 1] << std::endl;
    h_prof->SetBinContent(i, average[i - 1]);
    h_prof->SetBinError(i, RMS[i - 1] / nBinsY);
  }

  TH1F* h_RMS = new TH1F("RMSInEta", "RMS vs #eta", nBinsX, xmin, xmax);
  h_RMS->GetXaxis()->SetTitle(h2->GetXaxis()->GetTitle());
  h_RMS->GetYaxis()->SetTitle(updateTitle(h2->GetZaxis()->GetTitle(), TitleFormatType::RMS));
  for (int i = 1; i <= nBinsX; i++) {
    std::cout << i << " = " << average[i - 1] << std::endl;
    h_RMS->SetBinContent(i, RMS[i - 1]);
    h_RMS->SetBinError(i, RMS[i - 1] / std::sqrt(2 * (nBinsY - 1)));
  }

  return std::make_pair(h_prof, h_RMS);
}

//________________________________________________________________
std::pair<TH1F*, TH1F*> makeProfileVsPhi(TH2F* h2) {
  const auto& nBinsX = h2->GetNbinsX();
  const auto& nBinsY = h2->GetNbinsY();

  float ymin = h2->GetYaxis()->GetXmin();
  float ymax = h2->GetYaxis()->GetXmax();

  std::vector<float> average;
  std::vector<float> RMS;

  for (int j = 1; j <= nBinsY; j++) {
    float avgInX{0.};
    for (int i = 1; i <= nBinsX; i++) {
      avgInX += h2->GetBinContent(i, j);
    }
    avgInX /= nBinsX;
    average.push_back(avgInX);
  }

  for (int j = 1; j <= nBinsY; j++) {
    float RMSInX{0.};
    for (int i = 1; i <= nBinsX; i++) {
      RMSInX += std::pow((h2->GetBinContent(i, j) - average[j - 1]), 2);
    }
    RMSInX /= nBinsX;
    RMS.push_back(std::sqrt(RMSInX));
  }

  TH1F* h_prof = new TH1F("avgInPhi", "profile vs #phi", nBinsY, ymin, ymax);
  h_prof->GetXaxis()->SetTitle(h2->GetYaxis()->GetTitle());
  h_prof->GetYaxis()->SetTitle(updateTitle(h2->GetZaxis()->GetTitle(), TitleFormatType::MEAN));
  for (int i = 1; i <= nBinsY; i++) {
    std::cout << i << " = " << average[i - 1] << std::endl;
    h_prof->SetBinContent(i, average[i - 1]);
    h_prof->SetBinError(i, RMS[i - 1] / nBinsX);
  }

  TH1F* h_RMS = new TH1F("RMSInPhi", "RMS vs #phi", nBinsY, ymin, ymax);
  h_RMS->GetXaxis()->SetTitle(h2->GetYaxis()->GetTitle());
  h_RMS->GetYaxis()->SetTitle(updateTitle(h2->GetZaxis()->GetTitle(), TitleFormatType::RMS));
  for (int i = 1; i <= nBinsY; i++) {
    std::cout << i << " = " << average[i - 1] << std::endl;
    h_RMS->SetBinContent(i, RMS[i - 1]);
    h_RMS->SetBinError(i, RMS[i - 1] / std::sqrt(2 * (nBinsX - 1)));
  }

  return std::make_pair(h_prof, h_RMS);
}

//________________________________________________________________
void analyzeDiMuonBiases_fast(const char* inputFile, anaKind type) {
  TStopwatch timer;
  timer.Start();

  std::cout << "I AM GOING TO PERFORM THE " << analysis_types[type] << " analysis!" << std::endl;

  ROOT::EnableImplicitMT();  // Enable multi-threading

  TFile* file = TFile::Open(inputFile);
  //TTree* tree = dynamic_cast<TTree*>(file->Get("myanalysis/tree"));
  TTree* tree = dynamic_cast<TTree*>(file->Get("ZtoMMNtuple/tree"));

  if (tree) {
    tree->ls();
  } else {
    std::cout << "could not open the input tree from file.\n exiting";
    exit(EXIT_FAILURE);
  }
  // prepare the output file name
  std::string outputString(inputFile);
  std::string oldSubstring = "ZmmNtuple_";
  std::string newSubstring = "histos_";
  outputString.replace(outputString.find(oldSubstring), oldSubstring.length(), newSubstring);
  std::size_t pos = outputString.find(".root");

  // If ".root" is found, inject the given const char*
  TString toInject = TString::Format("__%s", analysis_types[type]);
  if (pos != std::string::npos) {
    outputString.insert(pos, toInject.Data());
  }

  // one for iteration

  TFile* outputFile = TFile::Open(outputString.c_str(), "RECREATE");
  TH1F* hMass = new TH1F("hMass", "mass;m_{#mu#mu} [GeV];events", 80, 70., 110.);
  TH1F* hPt = new TH1F("hPt", ";muon p_{T} [GeV];events", 100, 0., 200.);
  TH1F* hZPt = new TH1F("hZPt", ";di-muon system p_{T} [GeV];events", 100, 0., 200.);

  TH1F* hPtPlus = new TH1F("hPtPlus", ";positive muon p_{T} [GeV];events", 100, 0., 200.);
  TH1F* hPtMinus = new TH1F("hPtMinus", ";negative muon p_{T} [GeV];events", 100, 0., 200.);

  TH1F* hPtCorr = new TH1F("hPtCorr", ";corrected muon p_{T} [GeV];events", 100, 0., 200.);
  TH1F* hPtPlusCorr = new TH1F("hPtPlusCorr", ";corrected positive muon p_{T} [GeV];events", 100, 0., 200.);
  TH1F* hPtMinusCorr = new TH1F("hPtMinusCorr", ";corrected negative muon p_{T} [GeV];events", 100, 0., 200.);

  TH1F* hMassCorr = new TH1F("hMassCorr", "corrected mass;corrected m_{#mu#mu} [GeV];events", 80, 70., 110.);
  TH1F* hDeltaMassCorr = new TH1F(
      "hDeltaMassCorr",
      "#Delta Mass (corrected - corrected) ;#Delta m_{#mu#mu} (m^{corr}_{#mu#mu} - m^{uncorr}_{#mu#mu}) [GeV];events",
      100,
      -10.,
      10.);
  TH1F* hDeltaPtCorr =
      new TH1F("hDeltaPtCorr", "#Delta p_{T};#Delta p_{T} (p^{corr}_{T}-p^{uncorr}_{T}) [GeV];events", 100, -10., 10.);

  TH1F* hEta = new TH1F("hEta", ";muon #eta;events", 100, -3.0, 3.0);

  TH1F* hDeltaD0 = new TH1F("hDeltaD0", ";#Deltad_{0} [cm];events", 100, -0.01, 0.01);
  TH1F* hDeltaDz = new TH1F("hDeltaDz", ";#Deltad_{z} [cm];events", 100, -0.1, 0.1);

  TH1F* hPhi = new TH1F("hPhi", ";muon #phi;events", 100, -TMath::Pi(), TMath::Pi());
  TH1F* hCorrection = new TH1F(
      "hCorrection",
      TString::Format("largest correction vs iteration;iteration number; #delta_{%s} correction ", analysis_types[type])
          .Data(),
      k_MAXITERATIONS,
      0.,
      k_MAXITERATIONS);

  TH2F* hSagitta = new TH2F("hSagitta",
                            "#delta_{sagitta};muon #eta;muon #phi;#delta_{sagitta} [TeV^{-1}]",
                            k_NBINS,
                            -2.5,
                            2.5,
                            k_NBINS,
                            -TMath::Pi(),
                            TMath::Pi());

  const char* IPtype = (type == anaKind::d0_t) ? "d_{0}" : "d_{z}";
  TH2F* hIP = new TH2F("hIP",
                       TString::Format("#delta_{%s};muon #eta;muon #phi;#delta_{%s} [#mum]", IPtype, IPtype).Data(),
                       k_NBINS,
                       -2.47,
                       2.47,
                       k_NBINS,
                       -TMath::Pi(),
                       TMath::Pi());

  std::map<std::pair<int, int>, float> sagittaCorrections;
  std::map<std::pair<int, int>, float> IPCorrections;
  //sagittaCorrections.reserve(k_NBINS * k_NBINS);

  // initialize the map
  for (unsigned int i = 0; i < k_NBINS; i++) {
    for (unsigned int j = 0; j < k_NBINS; j++) {
      const auto& index = std::make_pair(i, j);
      //sagittaCorrections[index] = 3.435/10e3 ; // from GEN-SIM
      sagittaCorrections[index] = 0.f;
      IPCorrections[index] = 0.f;
    }
  }

  float maxCorrection{1.f};
  int iteration = 0;
  float threshold = (type == sagitta_t) ? 1e-10 : 1e-10;

  while ((std::abs(maxCorrection) > threshold) && iteration < k_MAXITERATIONS) {
    float oldMaxCorrection = maxCorrection;
    if (type == sagitta_t) {
      maxCorrection = updateSagittaMapFast(tree, sagittaCorrections, hSagitta, iteration, oldMaxCorrection);
    } else {
      maxCorrection = updateIPMapFast(tree, IPCorrections, hIP, iteration, type);
    }

    std::cout << "iteration: " << iteration << " maxCorrection: " << maxCorrection << std::endl;
    hCorrection->SetBinContent(iteration, maxCorrection);
    iteration++;
  }

  Float_t posTrackEta, negTrackEta, posTrackPhi, negTrackPhi, posTrackPt, negTrackPt;
  Float_t posTrackDz, negTrackDz, posTrackD0, negTrackD0;

  tree->SetBranchAddress("posTrackEta", &posTrackEta);
  tree->SetBranchAddress("negTrackEta", &negTrackEta);
  tree->SetBranchAddress("posTrackPhi", &posTrackPhi);
  tree->SetBranchAddress("negTrackPhi", &negTrackPhi);
  tree->SetBranchAddress("posTrackPt", &posTrackPt);
  tree->SetBranchAddress("negTrackPt", &negTrackPt);
  tree->SetBranchAddress("posTrackDz", &posTrackDz);
  tree->SetBranchAddress("negTrackDz", &negTrackDz);
  tree->SetBranchAddress("posTrackD0", &posTrackD0);
  tree->SetBranchAddress("negTrackD0", &negTrackD0);

  Float_t invMass;

  // Fill control histograms
  for (Long64_t i = 0; i < tree->GetEntries(); i++) {
    tree->GetEntry(i);

    TLorentzVector posTrack, negTrack, mother;
    posTrack.SetPtEtaPhiM(posTrackPt, posTrackEta, posTrackPhi, k_muMass);  // assume muon mass for tracks
    negTrack.SetPtEtaPhiM(negTrackPt, negTrackEta, negTrackPhi, k_muMass);
    mother = posTrack + negTrack;

    hDeltaD0->Fill(posTrackD0 - negTrackD0);
    hDeltaDz->Fill(posTrackDz - negTrackDz);

    hPt->Fill(posTrackPt);
    hPt->Fill(negTrackPt);

    hPtPlus->Fill(posTrackPt);
    hPtMinus->Fill(negTrackPt);

    hEta->Fill(posTrackEta);
    hEta->Fill(negTrackEta);

    hPhi->Fill(posTrackPhi);
    hPhi->Fill(negTrackPhi);

    invMass = mother.M();
    hMass->Fill(invMass);
    hZPt->Fill(mother.Pt());

    const auto& indexPlus = findEtaPhiBin(hSagitta, posTrackEta, posTrackPhi);
    float deltaPlus = sagittaCorrections[indexPlus];

    const auto& indexMinus = findEtaPhiBin(hSagitta, negTrackEta, negTrackPhi);
    float deltaMinus = sagittaCorrections[indexMinus];

    TLorentzVector posTrackCorr, negTrackCorr, motherCorr;
    //posTrackCorr.SetPtEtaPhiM(posTrackPt/(1+posTrackPt*deltaPlus), posTrackEta, posTrackPhi, k_muMass);  // assume muon mass for tracks
    //negTrackCorr.SetPtEtaPhiM(negTrackPt/(1-negTrackPt*deltaMinus), negTrackEta, negTrackPhi, k_muMass);

    posTrackCorr.SetPtEtaPhiM(
        pTcorrector(posTrackPt, deltaPlus, 1.), posTrackEta, posTrackPhi, k_muMass);  // assume muon mass for tracks
    negTrackCorr.SetPtEtaPhiM(pTcorrector(negTrackPt, deltaMinus, -1.), negTrackEta, negTrackPhi, k_muMass);

    hPtPlusCorr->Fill(posTrackCorr.Pt());
    hPtMinusCorr->Fill(negTrackCorr.Pt());

    hDeltaPtCorr->Fill(posTrackCorr.Pt() - posTrackPt);
    hDeltaPtCorr->Fill(negTrackCorr.Pt() - negTrackPt);

    //std::cout << "original pT: " << posTrackPt << " corrected pT: " << posTrackPt / (1+posTrackPt*deltaPlus)  << std::endl;
    //std::cout << "original pT: " << negTrackPt << " corrected pT: " << negTrackPt / (1-negTrackPt*deltaMinus) << std::endl;

    motherCorr = posTrackCorr + negTrackCorr;

    hMassCorr->Fill(motherCorr.M());
    hDeltaMassCorr->Fill(mother.M() - motherCorr.M());
    hPtCorr->Fill(posTrackCorr.Pt());
    hPtCorr->Fill(negTrackCorr.Pt());
  }

  //for (unsigned int iteration = 0; iteration < 1; iteration++) {
  //float max = updateSagittaMap(tree, sagittaCorrections, hSagitta, 0);  // zero-th iteration
  // std::cout << "maximal correction update is: " << max << std::endl;
  // }
  //updateSagittaMap(tree,sagittaCorrections,hSagitta,1); // first iteration
  //updateSagittaMap(tree,sagittaCorrections,hSagitta,2); // second iteration
  //updateSagittaMap(tree,sagittaCorrections,hSagitta,3); // third iteration

  for (unsigned int i = 0; i < k_NBINS; i++) {
    for (unsigned int j = 0; j < k_NBINS; j++) {
      const auto& index = std::make_pair(i, j);
      if (type == sagitta_t) {
        hSagitta->SetBinContent(i + 1, j + 1, sagittaCorrections[index] * 10e3);  // 1/GeV = 1000/TeV
      } else {
        hIP->SetBinContent(i + 1, j + 1, IPCorrections[index] * 10e4);  // 1cm = 10e4um
      }
    }
  }

  // The first argument is the name of the new TProfile,
  // the second argument is the first bin to include (1 in this case, to skip underflow),
  // the third argument is the last bin to include (-1 in this case, to skip overflow),
  // and the fourth argument is an option "s" to compute the profile using weights from the bin contents.

  //TH1D* projVsEta = hSagitta->ProjectionX("projVsEta", 1, k_NBINS);
  //projVsEta->GetYaxis()->SetTitle("#delta_{sagitta} [TeV^{-1}]");
  //TH1D* projVsPhi = hSagitta->ProjectionY("projVsPhi", 1, k_NBINS);
  //projVsPhi->GetYaxis()->SetTitle("#delta_{sagitta} [TeV^{-1}]");
  //TProfile* profVsEta = hSagitta->ProfileX("_pfx",1,47,"s");
  //TProfile* profVsPhi = hSagitta->ProfileY("_pfy",1,47,"s");

  //const auto& momVsEta = makeProfileVsEta(hSagitta);
  //const auto& momVsPhi = makeProfileVsPhi(hSagitta);

  std::pair<TH1F*, TH1F*> momVsEta = std::make_pair(nullptr, nullptr);
  std::pair<TH1F*, TH1F*> momVsPhi = std::make_pair(nullptr, nullptr);
  if (type == sagitta_t) {
    momVsEta = makeProfileVsEta(hSagitta);
    momVsPhi = makeProfileVsPhi(hSagitta);
  } else {
    momVsEta = makeProfileVsEta(hIP);
    momVsPhi = makeProfileVsPhi(hIP);
  }

  hDeltaD0->Write();
  hDeltaDz->Write();

  hMass->Write();
  hPt->Write();
  hPtPlus->Write();
  hPtMinus->Write();

  hZPt->Write();

  hEta->Write();
  hPhi->Write();

  if (type == sagitta_t) {
    hMassCorr->Write();
    hPtCorr->Write();
    hPtPlusCorr->Write();
    hPtMinusCorr->Write();
    hDeltaMassCorr->Write();
    hDeltaPtCorr->Write();
  }

  hCorrection->Write();
  if (type == sagitta_t) {
    hSagitta->SetOption("colz");
    hSagitta->Write();
  } else {
    hIP->SetOption("colz");
    hIP->Write();
  }

  momVsEta.first->Write();
  momVsPhi.first->Write();
  momVsEta.second->Write();
  momVsPhi.second->Write();

  //projVsEta->Write();
  //projVsPhi->Write();
  //profVsEta->Write();
  //profVsPhi->Write();

  outputFile->Close();
  file->Close();

  timer.Stop();
  timer.Print();
}

//________________________________________________________________
float fitResiduals(TH1* histogram) {
  TF1* gaussianFunc = new TF1("gaussianFunc", "gaus", 0, 100);
  // Perform an initial fit to get the mean (μ) and width (σ) of the Gaussian
  histogram->Fit(gaussianFunc, "Q");

  double prevMean = gaussianFunc->GetParameter(1);   // Mean of the previous iteration
  double prevSigma = gaussianFunc->GetParameter(2);  // Sigma of the previous iteration

  int maxIterations = 10;    // Set a maximum number of iterations
  double tolerance = 0.001;  // Define a tolerance to check for convergence

  for (int iteration = 0; iteration < maxIterations; ++iteration) {
    // Set the fit range to 1.5 σ of the previous iteration
    double lowerLimit = prevMean - 1.5 * prevSigma;
    double upperLimit = prevMean + 1.5 * prevSigma;
    gaussianFunc->SetRange(lowerLimit, upperLimit);

    // Perform the fit within the restricted range
    histogram->Fit(gaussianFunc, "Q");

    // Get the fit results for this iteration
    double currentMean = gaussianFunc->GetParameter(1);
    double currentSigma = gaussianFunc->GetParameter(2);

    // Check for convergence
    if (abs(currentMean - prevMean) < tolerance && abs(currentSigma - prevSigma) < tolerance) {
      //cout << "Converged after " << iteration + 1 << " iterations." << endl;
      break;
    }

    prevMean = currentMean;
    prevSigma = currentSigma;
  }

  return prevMean;
}

//________________________________________________________________
float updateSagittaMap(TTree* tree,
                       std::map<std::pair<int, int>, float>& theMap,
                       TH2F*& hSagitta,
                       const int iteration,
                       float oldMaxCorrection) {
  std::cout << "calling the updateSagittaMap" << std::endl;

  std::map<std::pair<int, int>, float> deltaCorrection;
  //deltaCorrection.reserve(k_NBINS * k_NBINS);
  std::map<std::pair<int, int>, float> countsPerBin;
  //countsPerBin.reserve(k_NBINS * k_NBINS);
  std::map<std::pair<int, int>, TH1F*> deltaInBin;

  //float maxRange = (iteration == 0) ? 1. : 1./std::pow(2,iteration);
  float maxRange = 0.01;

  for (unsigned int i = 0; i < k_NBINS; i++) {
    for (unsigned int j = 0; j < k_NBINS; j++) {
      const auto& index = std::make_pair(i, j);
      deltaCorrection[index] = 0.f;
      countsPerBin[index] = 0.f;
      deltaInBin[index] = new TH1F(
          Form("delta_iter_%i_%i_%i", iteration, i, j),
          Form("iteration %i : #delta_%i_%i;estimated #delta_{s} -true #delta_{s} [GeV^{-1}];n.muons", iteration, i, j),
          100,
          -maxRange,
          maxRange);
      //deltaInBin[index]->GetXaxis()->SetCanExtend(kTRUE);
    }
  }

  Float_t posTrackEta, negTrackEta, posTrackPhi, negTrackPhi, posTrackPt, negTrackPt, invMass;
  Float_t genPosMuonEta, genNegMuonEta, genPosMuonPhi, genNegMuonPhi, genPosMuonPt, genNegMuonPt;  // gen info

  tree->SetBranchAddress("posTrackEta", &posTrackEta);
  tree->SetBranchAddress("negTrackEta", &negTrackEta);
  tree->SetBranchAddress("posTrackPhi", &posTrackPhi);
  tree->SetBranchAddress("negTrackPhi", &negTrackPhi);
  tree->SetBranchAddress("posTrackPt", &posTrackPt);
  tree->SetBranchAddress("negTrackPt", &negTrackPt);

  // gen info
  tree->SetBranchAddress("genPosMuonEta", &genPosMuonEta);
  tree->SetBranchAddress("genNegMuonEta", &genNegMuonEta);
  tree->SetBranchAddress("genPosMuonPhi", &genPosMuonPhi);
  tree->SetBranchAddress("genNegMuonPhi", &genNegMuonPhi);
  tree->SetBranchAddress("genPosMuonPt", &genPosMuonPt);
  tree->SetBranchAddress("genNegMuonPt", &genNegMuonPt);

  for (Long64_t i = 0; i < tree->GetEntries(); i++) {
    tree->GetEntry(i);
    TLorentzVector posTrack, negTrack, mother;
    posTrack.SetPtEtaPhiM(posTrackPt, posTrackEta, posTrackPhi, k_muMass);  // assume muon mass for tracks
    negTrack.SetPtEtaPhiM(negTrackPt, negTrackEta, negTrackPhi, k_muMass);
    mother = posTrack + negTrack;
    invMass = mother.M();

    //std::cout << "invmass:" << invMass << std::endl;

    //now deal with positive muon
    const auto& indexPlus = findEtaPhiBin(hSagitta, posTrackEta, posTrackPhi);
    float deltaSagittaPlus = theMap[indexPlus];

    //now deal with negative muon
    const auto& indexMinus = findEtaPhiBin(hSagitta, negTrackEta, negTrackPhi);
    float deltaSagittaMinus = theMap[indexMinus];

    if (indexPlus == std::make_pair(-1, -1) || indexMinus == std::make_pair(-1, -1)) {
      continue;
    }

    /*
    float deltaMass{0.f};
    if (iteration == 0) {
      deltaMass = invMass * invMass - (MZ_PDG * MZ_PDG);
    } else {
      deltaMass = MZ_PDG * (posTrackPt * deltaSagittaPlus - negTrackPt * deltaSagittaMinus);
    }
    
    float deltaDeltaSagittaPlus =
        -(deltaMass / (2 * MZ_PDG)) * ((1 + posTrackPt * deltaSagittaPlus) / posTrackPt);
    float deltaDeltaSagittaMinus =
        (deltaMass / (2 * MZ_PDG)) * ((1 - negTrackPt * deltaSagittaMinus) / negTrackPt);

    */

    float frac;
    if (iteration != 0) {
      frac = (posTrackPt * deltaSagittaPlus - negTrackPt * deltaSagittaMinus);
    } else {
      frac = (invMass - MZ_PDG) / MZ_PDG;
    }

    /*
    TLorentzVector posTrackCorr, negTrackCorr, motherCorr;
    posTrackCorr.SetPtEtaPhiM(pTcorrector(posTrackPt,deltaSagittaPlus,1.), posTrackEta, posTrackPhi, k_muMass);  // assume muon mass for tracks
    negTrackCorr.SetPtEtaPhiM(pTcorrector(negTrackPt,deltaSagittaMinus,-1.), negTrackEta, negTrackPhi, k_muMass);
    motherCorr = posTrackCorr + negTrackCorr;
    
    float frac = (motherCorr.M() - MZ_PDG) / MZ_PDG;
    */

    float deltaDeltaSagittaPlus = calcDeltaSagitta(1., frac, deltaSagittaPlus, posTrackPt);
    float deltaDeltaSagittaMinus = calcDeltaSagitta(-1., frac, deltaSagittaMinus, negTrackPt);

    // inverting equation 5 of https://arxiv.org/pdf/2212.07338.pdf
    double trueDeltaSagittaPlus = (genPosMuonPt - posTrackPt) / (genPosMuonPt * posTrackPt);
    double trueDeltaSagittaMinus = (negTrackPt - genNegMuonPt) / (genNegMuonPt * negTrackPt);

    //std::cout << "deltaMass: " << std::setw(10) << frac
    //<< " DeltaDeltaSag(+)-true: " << (deltaSagittaPlus + deltaDeltaSagittaPlus)   - trueDeltaSagittaPlus
    //	      << " DeltaDeltaSag(-)-true: " << (deltaSagittaMinus + deltaDeltaSagittaMinus) - trueDeltaSagittaMinus << std::endl;

    deltaCorrection[indexPlus] += deltaDeltaSagittaPlus;
    deltaCorrection[indexMinus] += deltaDeltaSagittaMinus;

    deltaInBin[indexPlus]->Fill((deltaSagittaPlus + deltaDeltaSagittaPlus) - trueDeltaSagittaPlus);
    deltaInBin[indexMinus]->Fill((deltaSagittaMinus + deltaDeltaSagittaMinus) - trueDeltaSagittaMinus);

    countsPerBin[indexPlus] += 1.;
    countsPerBin[indexMinus] += 1.;
  }

  //TFile* iterFile = TFile::Open(Form("iteration_%i.root",iteration), "RECREATE");

  for (unsigned int i = 0; i < k_NBINS; i++) {
    for (unsigned int j = 0; j < k_NBINS; j++) {
      const auto& index = std::make_pair(i, j);
      //std::cout << index.first << ", " << index.second << " value: " << deltaCorrection[index] << " / " << countsPerBin[index];
      deltaCorrection[index] /= countsPerBin[index];
      //deltaCorrection[index] = fitResiduals(deltaInBin[index]);
      deltaInBin[index]->Write();
      //std::cout << " =  " << deltaCorrection[index] << std::endl;
    }
  }

  //iterFile->Close();

  std::cout << " ================================ iteration: " << iteration << std::endl;
  for (unsigned int i = 0; i < k_NBINS; i++) {
    for (unsigned int j = 0; j < k_NBINS; j++) {
      const auto& index = std::make_pair(i, j);
      //std::cout << i << ", " << j << " initial: " << theMap[index] << " correction: " << deltaCorrection[index]
      //          << std::endl;
      theMap[index] += deltaCorrection[index];
    }
  }

  // find the largest element of the correction of this iteration
  auto maxIter = std::max_element(deltaCorrection.begin(), deltaCorrection.end(), [](const auto& a, const auto& b) {
    return std::abs(a.second) < std::abs(b.second);
  });

  return maxIter->second;
}

float updateSagittaMapFast(TTree* tree,
                           std::map<std::pair<int, int>, float>& theMap,
                           TH2F*& hSagitta,
                           const int iteration,
                           float oldMaxCorrection) {
  std::cout << "Calling updateSagittaMap with RDataFrame" << std::endl;

  ROOT::RDataFrame df(*tree);

  // Initialize maps for delta correction and counts per bin
  std::map<std::pair<int, int>, float> deltaCorrection;
  std::map<std::pair<int, int>, float> countsPerBin;

  /*
  std::map<std::pair<int, int>, TH1F*> deltaInBin;

  float maxRange = 0.01;

  for (unsigned int i = 0; i < k_NBINS; i++) {
    for (unsigned int j = 0; j < k_NBINS; j++) {
      const auto& index = std::make_pair(i, j);
      deltaCorrection[index] = 0.f;
      countsPerBin[index] = 0.f;
      deltaInBin[index] = new TH1F(
          Form("delta_iter_%i_%i_%i", iteration, i, j),
          Form("iteration %i : #delta_%i_%i;estimated #delta_{s} -true #delta_{s} [GeV^{-1}];n.muons", iteration, i, j),
          100,
          -maxRange,
          maxRange);
    }
  }
  */

  const double mZfromPDG = 91.1876;  // in GeV

  // Define necessary columns
  auto df2 =
      df.Define("posTrack",
                [](float pt, float eta, float phi) -> TLorentzVector {
                  TLorentzVector v;
                  v.SetPtEtaPhiM(pt, eta, phi, k_muMass);
                  return v;
                },
                {"posTrackPt", "posTrackEta", "posTrackPhi"})
          .Define("negTrack",
                  [](float pt, float eta, float phi) -> TLorentzVector {
                    TLorentzVector v;
                    v.SetPtEtaPhiM(pt, eta, phi, k_muMass);
                    return v;
                  },
                  {"negTrackPt", "negTrackEta", "negTrackPhi"})
          .Define("mother",
                  [](const TLorentzVector& posTrack, const TLorentzVector& negTrack) -> TLorentzVector {
                    return posTrack + negTrack;
                  },
                  {"posTrack", "negTrack"})
          .Define("invMass", [](const TLorentzVector& mother) -> float { return mother.M(); }, {"mother"})
          .Define("indexPlus",
                  [hSagitta](float eta, float phi) -> std::pair<int, int> { return findEtaPhiBin(hSagitta, eta, phi); },
                  {"posTrackEta", "posTrackPhi"})
          .Define("indexMinus",
                  [hSagitta](float eta, float phi) -> std::pair<int, int> { return findEtaPhiBin(hSagitta, eta, phi); },
                  {"negTrackEta", "negTrackPhi"})
          .Define("deltaSagittaPlus",
                  [&theMap](const std::pair<int, int>& index) -> float { return theMap.at(index); },
                  {"indexPlus"})
          .Define("deltaSagittaMinus",
                  [&theMap](const std::pair<int, int>& index) -> float { return theMap.at(index); },
                  {"indexMinus"});

  auto df3 = df2.Filter(
                    [](const std::pair<int, int>& indexPlus, const std::pair<int, int>& indexMinus) {
                      return indexPlus != std::make_pair(-1, -1) && indexMinus != std::make_pair(-1, -1);
                    },
                    {"indexPlus", "indexMinus"})
                 .Define("frac",
                         [iteration, mZfromPDG](float invMass,
                                                float posTrackPt,
                                                float deltaSagittaPlus,
                                                float negTrackPt,
                                                float deltaSagittaMinus) -> float {
                           return (iteration != 0) ? (posTrackPt * deltaSagittaPlus - negTrackPt * deltaSagittaMinus)
                                                   : (invMass - mZfromPDG) / mZfromPDG;
                         },
                         {"invMass", "posTrackPt", "deltaSagittaPlus", "negTrackPt", "deltaSagittaMinus"})
                 .Define("deltaDeltaSagittaPlus",
                         [](float frac, float deltaSagittaPlus, float posTrackPt) -> float {
                           return calcDeltaSagitta(1.f, frac, deltaSagittaPlus, posTrackPt);
                         },
                         {"frac", "deltaSagittaPlus", "posTrackPt"})
                 .Define("deltaDeltaSagittaMinus",
                         [](float frac, float deltaSagittaMinus, float negTrackPt) -> float {
                           return calcDeltaSagitta(-1.f, frac, deltaSagittaMinus, negTrackPt);
                         },
                         {"frac", "deltaSagittaMinus", "negTrackPt"});
  // .Define("trueDeltaSagittaPlus", [](float genPt, float trackPt) -> double {
  //       return (genPt - trackPt) / (genPt * trackPt);
  //     }, {"genPosMuonPt", "posTrackPt"})
  // .Define("trueDeltaSagittaMinus", [](float trackPt, float genPt) -> double {
  //       return (trackPt - genPt) / (genPt * trackPt);
  //     }, {"negTrackPt", "genNegMuonPt"});

  // Accumulate corrections
  df3.Foreach(
      [&](const std::pair<int, int>& indexPlus,
          float deltaDeltaSagittaPlus,
          float deltaSagittaPlus,
          const std::pair<int, int>& indexMinus,
          float deltaDeltaSagittaMinus,
          float deltaSagittaMinus) {
#pragma omp critical
        {
          deltaCorrection[indexPlus] += deltaDeltaSagittaPlus;
          deltaCorrection[indexMinus] += deltaDeltaSagittaMinus;
          //deltaInBin[indexPlus]->Fill((deltaSagittaPlus + deltaDeltaSagittaPlus) - trueDeltaSagittaPlus);
          //deltaInBin[indexMinus]->Fill((deltaSagittaMinus + deltaDeltaSagittaMinus) - trueDeltaSagittaMinus);
          countsPerBin[indexPlus] += 1.f;
          countsPerBin[indexMinus] += 1.f;
        }
      },
      {"indexPlus",
       "deltaDeltaSagittaPlus",
       "deltaSagittaPlus",
       "indexMinus",
       "deltaDeltaSagittaMinus",
       "deltaSagittaMinus"});

  // Normalize corrections
  for (unsigned int i = 0; i < k_NBINS; i++) {
    for (unsigned int j = 0; j < k_NBINS; j++) {
      const auto& index = std::make_pair(i, j);
      deltaCorrection[index] /= countsPerBin[index];
      //deltaInBin[index]->Write();
    }
  }

  std::cout << " ================================ iteration: " << iteration << std::endl;
  for (unsigned int i = 0; i < k_NBINS; i++) {
    for (unsigned int j = 0; j < k_NBINS; j++) {
      const auto& index = std::make_pair(i, j);
      theMap[index] += deltaCorrection[index];
    }
  }

  // Find the largest element of the correction for this iteration
  auto maxIter = std::max_element(deltaCorrection.begin(), deltaCorrection.end(), [](const auto& a, const auto& b) {
    return std::abs(a.second) < std::abs(b.second);
  });

  return maxIter->second;
}

//________________________________________________________________
float updateIPMap(
    TTree* tree, std::map<std::pair<int, int>, float>& theMap, TH2F*& hIP, const int iteration, anaKind type) {
  std::cout << "calling the updateIPMap" << std::endl;

  std::map<std::pair<int, int>, float> deltaCorrection;
  //deltaCorrection.reserve(k_NBINS * k_NBINS);
  std::map<std::pair<int, int>, float> countsPerBin;
  //countsPerBin.reserve(k_NBINS * k_NBINS);

  for (unsigned int i = 0; i < k_NBINS; i++) {
    for (unsigned int j = 0; j < k_NBINS; j++) {
      const auto& index = std::make_pair(i, j);
      deltaCorrection[index] = 0.f;
      countsPerBin[index] = 0.f;
    }
  }

  float posTrackEta, negTrackEta, posTrackPhi, negTrackPhi, posTrackIP, negTrackIP, invMass;
  tree->SetBranchAddress("posTrackEta", &posTrackEta);
  tree->SetBranchAddress("negTrackEta", &negTrackEta);
  tree->SetBranchAddress("posTrackPhi", &posTrackPhi);
  tree->SetBranchAddress("negTrackPhi", &negTrackPhi);
  if (type == d0_t) {
    tree->SetBranchAddress("posTrackD0", &posTrackIP);
    tree->SetBranchAddress("negTrackD0", &negTrackIP);
  } else if (type == dz_t) {
    tree->SetBranchAddress("posTrackDz", &posTrackIP);
    tree->SetBranchAddress("negTrackDz", &negTrackIP);
  } else {
    std::cout << "error, you have been calling this script with the wrong settings!";
    exit(EXIT_FAILURE);
  }

  for (Long64_t i = 0; i < tree->GetEntries(); i++) {
    tree->GetEntry(i);

    // deal with the positive muon
    const auto& indexPlus = findEtaPhiBin(hIP, posTrackEta, posTrackPhi);
    float deltaIPPlus = theMap[indexPlus];

    //now deal with negative muon
    const auto& indexMinus = findEtaPhiBin(hIP, negTrackEta, negTrackPhi);
    float deltaIPMinus = theMap[indexMinus];

    float IPdistance = std::abs(((posTrackIP + deltaIPPlus) - (negTrackIP + deltaIPMinus)));

    //this converges
    float deltaDeltaIPPlus =
        (posTrackIP + deltaIPPlus) < (negTrackIP + deltaIPMinus) ? IPdistance / 2. : -IPdistance / 2.;
    float deltaDeltaIPMinus =
        (posTrackIP + deltaIPPlus) < (negTrackIP + deltaIPMinus) ? -IPdistance / 2. : IPdistance / 2.;

    //std::cout << "deltaMass: " << deltaMass << " DeltaDeltaSagPlus: " <<   deltaDeltaIPPlus << " DeltaDeltaSagMinus: " << deltaDeltaIPMinus << std::endl;

    deltaCorrection[indexPlus] += deltaDeltaIPPlus;
    deltaCorrection[indexMinus] += deltaDeltaIPMinus;

    countsPerBin[indexPlus] += 1.;
    countsPerBin[indexMinus] += 1.;
  }

  for (unsigned int i = 0; i < k_NBINS; i++) {
    for (unsigned int j = 0; j < k_NBINS; j++) {
      const auto& index = std::make_pair(i, j);
      //std::cout << index.first << ", " << index.second << " value: " << deltaCorrection[index] << " / " << countsPerBin[index];
      deltaCorrection[index] /= countsPerBin[index];
      //std::cout << " =  " << deltaCorrection[index] << std::endl;
    }
  }

  std::cout << " ================================ iteration: " << iteration << std::endl;
  for (unsigned int i = 0; i < k_NBINS; i++) {
    for (unsigned int j = 0; j < k_NBINS; j++) {
      const auto& index = std::make_pair(i, j);
      //std::cout << i << ", " << j << " initial: " << theMap[index] << " correction: " << deltaCorrection[index]
      //          << std::endl;
      theMap[index] += deltaCorrection[index];
    }
  }

  // find the largest element of the correction of this iteration
  auto maxIter = std::max_element(deltaCorrection.begin(), deltaCorrection.end(), [](const auto& a, const auto& b) {
    return std::abs(a.second) < std::abs(b.second);
  });

  return maxIter->second;
}

float updateIPMapFast(
    TTree* tree, std::map<std::pair<int, int>, float>& theMap, TH2F*& hIP, const int iteration, anaKind type) {
  std::cout << "Calling updateIPMap with RDataFrame" << std::endl;

  ROOT::RDataFrame df(*tree);

  std::map<std::pair<int, int>, float> deltaCorrection;
  std::map<std::pair<int, int>, float> countsPerBin;
  for (unsigned int i = 0; i < k_NBINS; i++) {
    for (unsigned int j = 0; j < k_NBINS; j++) {
      const auto& index = std::make_pair(i, j);
      deltaCorrection[index] = 0.f;
      countsPerBin[index] = 0.f;
    }
  }

  // Define the necessary columns based on anaKind
  auto columns = type == d0_t ? df.Define("posTrackIP", [](float d0) -> float { return d0; }, {"posTrackD0"})
                                    .Define("negTrackIP", [](float d0) -> float { return d0; }, {"negTrackD0"})
                              : df.Define("posTrackIP", [](float dz) -> float { return dz; }, {"posTrackDz"})
                                    .Define("negTrackIP", [](float dz) -> float { return dz; }, {"negTrackDz"});

  auto result =
      columns
          .Define("indexPlus",
                  [hIP](float posTrackEta, float posTrackPhi) -> std::pair<int, int> {
                    return findEtaPhiBin(hIP, posTrackEta, posTrackPhi);
                  },
                  {"posTrackEta", "posTrackPhi"})
          .Define("indexMinus",
                  [hIP](float negTrackEta, float negTrackPhi) -> std::pair<int, int> {
                    return findEtaPhiBin(hIP, negTrackEta, negTrackPhi);
                  },
                  {"negTrackEta", "negTrackPhi"})
          .Define("deltaIPPlus",
                  [theMap](const std::pair<int, int>& indexPlus) -> float { return theMap.at(indexPlus); },
                  {"indexPlus"})
          .Define("deltaIPMinus",
                  [theMap](const std::pair<int, int>& indexMinus) -> float { return theMap.at(indexMinus); },
                  {"indexMinus"})
          .Define("IPdistance",
                  [](float posTrackIP, float deltaIPPlus, float negTrackIP, float deltaIPMinus) -> float {
                    return std::abs((posTrackIP + deltaIPPlus) - (negTrackIP + deltaIPMinus));
                  },
                  {"posTrackIP", "deltaIPPlus", "negTrackIP", "deltaIPMinus"})
          .Define(
              "deltaDeltaIPPlus",
              [](float posTrackIP, float deltaIPPlus, float negTrackIP, float deltaIPMinus, float IPdistance) -> float {
                return (posTrackIP + deltaIPPlus) < (negTrackIP + deltaIPMinus) ? IPdistance / 2. : -IPdistance / 2.;
              },
              {"posTrackIP", "deltaIPPlus", "negTrackIP", "deltaIPMinus", "IPdistance"})
          .Define(
              "deltaDeltaIPMinus",
              [](float posTrackIP, float deltaIPPlus, float negTrackIP, float deltaIPMinus, float IPdistance) -> float {
                return (posTrackIP + deltaIPPlus) < (negTrackIP + deltaIPMinus) ? -IPdistance / 2. : IPdistance / 2.;
              },
              {"posTrackIP", "deltaIPPlus", "negTrackIP", "deltaIPMinus", "IPdistance"});

  // Update the corrections in parallel
  result.Foreach(
      [&](const std::pair<int, int>& indexPlus,
          float deltaDeltaIPPlus,
          const std::pair<int, int>& indexMinus,
          float deltaDeltaIPMinus) {
#pragma omp critical
        {
          deltaCorrection[indexPlus] += deltaDeltaIPPlus;
          deltaCorrection[indexMinus] += deltaDeltaIPMinus;
          countsPerBin[indexPlus] += 1.f;
          countsPerBin[indexMinus] += 1.f;
        }
      },
      {"indexPlus", "deltaDeltaIPPlus", "indexMinus", "deltaDeltaIPMinus"});

  // Normalize the corrections
  for (unsigned int i = 0; i < k_NBINS; i++) {
    for (unsigned int j = 0; j < k_NBINS; j++) {
      const auto& index = std::make_pair(i, j);
      deltaCorrection[index] /= countsPerBin[index];
    }
  }

  // Update the map with the new corrections
  std::cout << " ================================ iteration: " << iteration << std::endl;
  for (unsigned int i = 0; i < k_NBINS; i++) {
    for (unsigned int j = 0; j < k_NBINS; j++) {
      const auto& index = std::make_pair(i, j);
      theMap[index] += deltaCorrection[index];
    }
  }

  // Find the largest correction in this iteration
  auto maxIter = std::max_element(deltaCorrection.begin(), deltaCorrection.end(), [](const auto& a, const auto& b) {
    return std::abs(a.second) < std::abs(b.second);
  });

  return maxIter->second;
}
