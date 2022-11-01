// user includes
#include "DQMOffline/Alignment/interface/DiMuonMassBiasClient.h"
#include "DataFormats/Histograms/interface/DQMToken.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/TagAndProbe/interface/RooCMSShape.h"

// RooFit includes
#include "TCanvas.h"
#include "RooAddPdf.h"
#include "RooDataHist.h"
#include "RooExponential.h"
#include "RooGaussian.h"
#include "RooPlot.h"
#include "RooRealVar.h"
#include "RooVoigtian.h"

//-----------------------------------------------------------------------------------
DiMuonMassBiasClient::DiMuonMassBiasClient(edm::ParameterSet const& iConfig)
    : TopFolder_(iConfig.getParameter<std::string>("FolderName")),
      fitBackground_(iConfig.getParameter<bool>("fitBackground")),
      useRooCMSShape_(iConfig.getParameter<bool>("useRooCMSShape")),
      debugMode_(iConfig.getParameter<bool>("debugMode")),
      MEtoHarvest_(iConfig.getParameter<std::vector<std::string>>("MEtoHarvest"))
//-----------------------------------------------------------------------------------
{
  edm::LogInfo("DiMuonMassBiasClient") << "DiMuonMassBiasClient::Constructing DiMuonMassBiasClient ";

  // fill the parameters for the fit
  edm::ParameterSet fit_par = iConfig.getParameter<edm::ParameterSet>("fit_par");
  diMuonMassBias::fillArrayF(meanConfig_, fit_par, "mean_par");
  diMuonMassBias::fillArrayF(widthConfig_, fit_par, "width_par");
  diMuonMassBias::fillArrayF(sigmaConfig_, fit_par, "sigma_par");

  if (debugMode_) {
    edm::LogPrint("DiMuonMassBiasClient")
        << "mean: " << meanConfig_[0] << " (" << meanConfig_[1] << "," << meanConfig_[2] << ") " << std::endl;
    edm::LogPrint("DiMuonMassBiasClient")
        << "width: " << widthConfig_[0] << " (" << widthConfig_[1] << "," << widthConfig_[2] << ")" << std::endl;
    edm::LogPrint("DiMuonMassBiasClient")
        << "sigma: " << sigmaConfig_[0] << " (" << sigmaConfig_[1] << "," << sigmaConfig_[2] << ")" << std::endl;
  }
}

//-----------------------------------------------------------------------------------
DiMuonMassBiasClient::~DiMuonMassBiasClient()
//-----------------------------------------------------------------------------------
{
  edm::LogInfo("DiMuonMassBiasClient") << "DiMuonMassBiasClient::Deleting DiMuonMassBiasClient ";
}

//-----------------------------------------------------------------------------------
void DiMuonMassBiasClient::beginJob(void)
//-----------------------------------------------------------------------------------
{
  edm::LogInfo("DiMuonMassBiasClient") << "DiMuonMassBiasClient::beginJob done";
}

//-----------------------------------------------------------------------------------
void DiMuonMassBiasClient::beginRun(edm::Run const& run, edm::EventSetup const& eSetup)
//-----------------------------------------------------------------------------------
{
  edm::LogInfo("DiMuonMassBiasClient") << "DiMuonMassBiasClient:: Begining of Run";
}

//-----------------------------------------------------------------------------------
void DiMuonMassBiasClient::bookMEs(DQMStore::IBooker& iBooker)
//-----------------------------------------------------------------------------------
{
  iBooker.setCurrentFolder(TopFolder_ + "/DiMuonMassBiasMonitor/MassBias/Profiles");
  for (const auto& [key, ME] : harvestTargets_) {
    if (ME == nullptr) {
      edm::LogError("DiMuonMassBiasClient") << "could not find MonitorElement for key: " << key << std::endl;
      continue;
    }

    const auto& title = ME->getTitle();
    const auto& xtitle = ME->getAxisTitle(1);
    const auto& ytitle = ME->getAxisTitle(2);

    const auto& nxbins = ME->getNbinsX();
    const auto& xmin = ME->getAxisMin(1);
    const auto& xmax = ME->getAxisMax(1);

    MonitorElement* meanToBook =
        iBooker.book1D(("Mean" + key), (title + ";" + xtitle + ";" + ytitle), nxbins, xmin, xmax);
    meanProfiles_.insert({key, meanToBook});

    MonitorElement* sigmaToBook =
        iBooker.book1D(("Sigma" + key), (title + ";" + xtitle + ";" + "#sigma of " + ytitle), nxbins, xmin, xmax);
    widthProfiles_.insert({key, sigmaToBook});
  }
}

//-----------------------------------------------------------------------------------
void DiMuonMassBiasClient::getMEsToHarvest(DQMStore::IGetter& iGetter)
//-----------------------------------------------------------------------------------
{
  std::string inFolder = TopFolder_ + "/DiMuonMassBiasMonitor/MassBias/";

  //loop on the list of histograms to harvest
  for (const auto& hname : MEtoHarvest_) {
    MonitorElement* toHarvest = iGetter.get(inFolder + hname);

    if (toHarvest == nullptr) {
      edm::LogError("DiMuonMassBiasClient") << "could not find input MonitorElement: " << inFolder + hname << std::endl;
      continue;
    }

    harvestTargets_.insert({hname, toHarvest});
  }
}

//-----------------------------------------------------------------------------------
void DiMuonMassBiasClient::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter)
//-----------------------------------------------------------------------------------
{
  edm::LogInfo("DiMuonMassBiasClient") << "DiMuonMassBiasClient::endLuminosityBlock";

  getMEsToHarvest(igetter);
  bookMEs(ibooker);

  for (const auto& [key, ME] : harvestTargets_) {
    if (debugMode_)
      edm::LogPrint("DiMuonMassBiasClient") << "dealing with key: " << key << std::endl;
    TH2F* bareHisto = ME->getTH2F();
    for (int bin = 1; bin <= ME->getNbinsX(); bin++) {
      if (debugMode_)
        edm::LogPrint("DiMuonMassBiasClient") << "dealing with bin: " << bin << std::endl;
      TH1D* Proj = bareHisto->ProjectionY(Form("%s_proj_%i", key.c_str(), bin), bin, bin);
      diMuonMassBias::fitOutputs results = fitVoigt(Proj);

      if (results.isInvalid()) {
        edm::LogWarning("DiMuonMassBiasClient") << "the current bin has invalid data" << std::endl;
        continue;
      }

      // fill the mean profiles
      const Measurement1D& bias = results.getBias();
      meanProfiles_[key]->setBinContent(bin, bias.value());
      meanProfiles_[key]->setBinError(bin, bias.error());

      // fill the width profiles
      const Measurement1D& width = results.getWidth();
      widthProfiles_[key]->setBinContent(bin, width.value());
      widthProfiles_[key]->setBinError(bin, width.error());
    }
  }
}

//-----------------------------------------------------------------------------------
diMuonMassBias::fitOutputs DiMuonMassBiasClient::fitVoigt(TH1* hist, const bool& fitBackground) const
//-----------------------------------------------------------------------------------
{
  if (hist->GetEntries() < diMuonMassBias::minimumHits) {
    edm::LogWarning("DiMuonMassBiasClient") << " Input histogram:" << hist->GetName() << " has not enough entries ("
                                            << hist->GetEntries() << ") for a meaningful Voigtian fit!\n"
                                            << "Skipping!";

    return diMuonMassBias::fitOutputs(Measurement1D(0., 0.), Measurement1D(0., 0.));
  }

  TCanvas* c1 = new TCanvas();
  if (debugMode_) {
    c1->Clear();
    c1->SetLeftMargin(0.15);
    c1->SetRightMargin(0.10);
  }

  // silence messages
  RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);

  Double_t xmin = hist->GetXaxis()->GetXmin();
  Double_t xmax = hist->GetXaxis()->GetXmax();

  if (debugMode_) {
    edm::LogPrint("DiMuonMassBiasClient") << "fitting range: (" << xmin << "-" << xmax << ")" << std::endl;
  }

  RooRealVar InvMass("InvMass", "di-muon mass M(#mu^{+}#mu^{-}) [GeV]", xmin, xmax);
  std::unique_ptr<RooPlot> frame{InvMass.frame()};
  RooDataHist datahist("datahist", "datahist", InvMass, RooFit::Import(*hist));
  datahist.plotOn(frame.get());

  // parmaeters of the Voigtian
  RooRealVar mean("#mu", "mean", meanConfig_[0], meanConfig_[1], meanConfig_[2]);          //90.0, 60.0, 120.0 (for Z)
  RooRealVar width("width", "width", widthConfig_[0], widthConfig_[1], widthConfig_[2]);   // 5.0,  0.0, 120.0 (for Z)
  RooRealVar sigma("#sigma", "sigma", sigmaConfig_[0], sigmaConfig_[1], sigmaConfig_[2]);  // 5.0,  0.0, 120.0 (for Z)
  RooVoigtian voigt("voigt", "voigt", InvMass, mean, width, sigma);

  // for the simple background fit
  RooRealVar lambda("#lambda", "slope", 0., -50., 50.);
  RooExponential expo("expo", "expo", InvMass, lambda);

  // for the more refined background fit
  RooRealVar exp_alpha("#alpha", "alpha", 40.0, 20.0, 160.0);
  RooRealVar exp_beta("#beta", "beta", 0.05, 0.0, 2.0);
  RooRealVar exp_gamma("#gamma", "gamma", 0.02, 0.0, 0.1);
  RooRealVar exp_peak("peak", "peak", meanConfig_[0]);
  RooCMSShape exp_pdf("exp_pdf", "bkg shape", InvMass, exp_alpha, exp_beta, exp_gamma, exp_peak);

  // define the signal and background fractions
  RooRealVar b("N_{b}", "Number of background events", 0, hist->GetEntries() / 10.);
  RooRealVar s("N_{s}", "Number of signal events", 0, hist->GetEntries());

  if (fitBackground_) {
    const auto& listPdf = useRooCMSShape_ ? RooArgList(voigt, exp_pdf) : RooArgList(voigt, expo);
    RooAddPdf fullModel("fullModel", "Signal + Background Model", listPdf, RooArgList(s, b));
    fullModel.fitTo(datahist, RooFit::PrintLevel(-1));
    fullModel.plotOn(frame.get(), RooFit::LineColor(kRed));
    if (useRooCMSShape_) {
      fullModel.plotOn(frame.get(), RooFit::Components(exp_pdf), RooFit::LineStyle(kDashed));  //Other option
    } else {
      fullModel.plotOn(frame.get(), RooFit::Components(expo), RooFit::LineStyle(kDashed));  //Other option
    }
    fullModel.paramOn(frame.get(), RooFit::Layout(0.65, 0.90, 0.90));
  } else {
    voigt.fitTo(datahist, RooFit::PrintLevel(-1));
    voigt.plotOn(frame.get(), RooFit::LineColor(kRed));            //this will show fit overlay on canvas
    voigt.paramOn(frame.get(), RooFit::Layout(0.65, 0.90, 0.90));  //this will display the fit parameters on canvas
  }

  // Redraw data on top and print / store everything
  datahist.plotOn(frame.get());
  frame->GetYaxis()->SetTitle("n. of events");
  TString histName = hist->GetName();
  frame->SetName("frame" + histName);
  frame->SetTitle(hist->GetTitle());
  frame->Draw();

  if (debugMode_) {
    c1->Print("fit_debug" + histName + ".pdf");
  }
  delete c1;

  float mass_mean = mean.getVal();
  float mass_sigma = sigma.getVal();

  float mass_mean_err = mean.getError();
  float mass_sigma_err = sigma.getError();

  Measurement1D resultM(mass_mean, mass_mean_err);
  Measurement1D resultW(mass_sigma, mass_sigma_err);

  return diMuonMassBias::fitOutputs(resultM, resultW);
}

//-----------------------------------------------------------------------------------
void DiMuonMassBiasClient::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
//-----------------------------------------------------------------------------------
{
  edm::ParameterSetDescription desc;
  desc.add<std::string>("FolderName", "DiMuonMassBiasMonitor");
  desc.add<bool>("fitBackground", false);
  desc.add<bool>("useRooCMSShape", false);
  desc.add<bool>("debugMode", false);

  edm::ParameterSetDescription fit_par;
  fit_par.add<std::vector<double>>("mean_par",
                                   {std::numeric_limits<float>::max(),
                                    std::numeric_limits<float>::max(),
                                    std::numeric_limits<float>::max()});  // par = mean

  fit_par.add<std::vector<double>>("width_par",
                                   {std::numeric_limits<float>::max(),
                                    std::numeric_limits<float>::max(),
                                    std::numeric_limits<float>::max()});  // par = width

  fit_par.add<std::vector<double>>("sigma_par",
                                   {std::numeric_limits<float>::max(),
                                    std::numeric_limits<float>::max(),
                                    std::numeric_limits<float>::max()});  // par = sigma

  desc.add<edm::ParameterSetDescription>("fit_par", fit_par);

  desc.add<std::vector<std::string>>("MEtoHarvest",
                                     {"DiMuMassVsMuMuPhi",
                                      "DiMuMassVsMuMuEta",
                                      "DiMuMassVsMuPlusPhi",
                                      "DiMuMassVsMuPlusEta",
                                      "DiMuMassVsMuMinusPhi",
                                      "DiMuMassVsMuMinusEta",
                                      "DiMuMassVsMuMuDeltaEta",
                                      "DiMuMassVsCosThetaCS"});
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(DiMuonMassBiasClient);
