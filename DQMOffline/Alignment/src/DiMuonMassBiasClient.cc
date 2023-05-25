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
#include "RooCBShape.h"

//-----------------------------------------------------------------------------------
DiMuonMassBiasClient::DiMuonMassBiasClient(edm::ParameterSet const& iConfig)
    : TopFolder_(iConfig.getParameter<std::string>("FolderName")),
      useTH1s_(iConfig.getParameter<bool>("useTH1s")),
      fitBackground_(iConfig.getParameter<bool>("fitBackground")),
      useRooCBShape_(iConfig.getParameter<bool>("useRooCBShape")),
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
    meanHistos_.insert({key, meanToBook});

    MonitorElement* sigmaToBook =
        iBooker.book1D(("Sigma" + key), (title + ";" + xtitle + ";" + "#sigma of " + ytitle), nxbins, xmin, xmax);
    widthHistos_.insert({key, sigmaToBook});
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
void DiMuonMassBiasClient::fitAndFillProfile(std::pair<std::string, MonitorElement*> toHarvest,
                                             DQMStore::IBooker& iBooker)
//-----------------------------------------------------------------------------------
{
  const auto& key = toHarvest.first;
  const auto& ME = toHarvest.second;

  if (debugMode_)
    edm::LogPrint("DiMuonMassBiasClient") << "dealing with key: " << key << std::endl;

  if (ME == nullptr) {
    edm::LogError("DiMuonMassBiasClient") << "could not find MonitorElement for key: " << key << std::endl;
    return;
  }

  const auto& title = ME->getTitle();
  const auto& xtitle = ME->getAxisTitle(1);
  const auto& ytitle = ME->getAxisTitle(2);

  const auto& nxbins = ME->getNbinsX();
  const auto& xmin = ME->getAxisMin(1);
  const auto& xmax = ME->getAxisMax(1);

  TProfile* p_mean = new TProfile(
      ("Mean" + key).c_str(), (title + ";" + xtitle + ";#LT" + ytitle + "#GT").c_str(), nxbins, xmin, xmax, "g");

  TProfile* p_width = new TProfile(
      ("Sigma" + key).c_str(), (title + ";" + xtitle + ";#sigma of " + ytitle).c_str(), nxbins, xmin, xmax, "g");

  p_mean->Sumw2();
  p_width->Sumw2();

  TH2F* bareHisto = ME->getTH2F();
  for (int bin = 1; bin <= nxbins; bin++) {
    const auto& xaxis = bareHisto->GetXaxis();
    const auto& low_edge = xaxis->GetBinLowEdge(bin);
    const auto& high_edge = xaxis->GetBinUpEdge(bin);

    if (debugMode_)
      edm::LogPrint("DiMuonMassBiasClient") << "dealing with bin: " << bin << " range: (" << std::setprecision(2)
                                            << low_edge << "," << std::setprecision(2) << high_edge << ")";

    TH1D* Proj = bareHisto->ProjectionY(Form("%s_proj_%i", key.c_str(), bin), bin, bin);
    Proj->SetTitle(Form("%s #in (%.2f,%.2f), bin: %i", Proj->GetTitle(), low_edge, high_edge, bin));

    diMuonMassBias::fitOutputs results = fitLineShape(Proj);

    if (results.isInvalid()) {
      edm::LogWarning("DiMuonMassBiasClient") << "the current bin has invalid data" << std::endl;
      continue;
    }

    // fill the mean profiles
    const Measurement1D& bias = results.getBias();

    // ============================================= DISCLAIMER ================================================
    // N.B. this is sort of a hack in order to fill arbitrarily both central values and error bars of a TProfile.
    // Choosing the option "g" in the constructor the bin error will be 1/sqrt(W(j)), where W(j) is the sum of weights.
    // Filling the sum of weights with the 1 / err^2, the bin error automatically becomes "err".
    // In order to avoid the central value to be shifted, that's divided by 1 / err^2 as well.
    // For more information, please consult the https://root.cern.ch/doc/master/classTProfile.html

    p_mean->SetBinContent(bin, bias.value() / (bias.error() * bias.error()));
    p_mean->SetBinEntries(bin, 1. / (bias.error() * bias.error()));

    if (debugMode_)
      LogDebug("DiMuonBassBiasClient") << " Bin: " << bin << " value:  " << bias.value() << " from profile ( "
                                       << p_mean->GetBinContent(bin) << ") - error:  " << bias.error()
                                       << "  from profile ( " << p_mean->GetBinError(bin) << " )";

    // fill the width profiles
    const Measurement1D& width = results.getWidth();

    // see discussion above
    p_width->SetBinContent(bin, width.value() / (width.error() * width.error()));
    p_width->SetBinEntries(bin, 1. / (width.error() * width.error()));
  }

  // now book the profiles
  iBooker.setCurrentFolder(TopFolder_ + "/DiMuonMassBiasMonitor/MassBias/Profiles");
  MonitorElement* meanToBook = iBooker.bookProfile(p_mean->GetName(), p_mean);
  meanProfiles_.insert({key, meanToBook});

  MonitorElement* sigmaToBook = iBooker.bookProfile(p_width->GetName(), p_width);
  widthProfiles_.insert({key, sigmaToBook});

  delete p_mean;
  delete p_width;
}

//-----------------------------------------------------------------------------------
void DiMuonMassBiasClient::fitAndFillHisto(std::pair<std::string, MonitorElement*> toHarvest,
                                           DQMStore::IBooker& iBooker)
//-----------------------------------------------------------------------------------
{
  const auto& key = toHarvest.first;
  const auto& ME = toHarvest.second;

  if (debugMode_)
    edm::LogPrint("DiMuonMassBiasClient") << "dealing with key: " << key << std::endl;

  if (ME == nullptr) {
    edm::LogError("DiMuonMassBiasClient") << "could not find MonitorElement for key: " << key << std::endl;
    return;
  }

  TH2F* bareHisto = ME->getTH2F();
  for (int bin = 1; bin <= ME->getNbinsX(); bin++) {
    const auto& xaxis = bareHisto->GetXaxis();
    const auto& low_edge = xaxis->GetBinLowEdge(bin);
    const auto& high_edge = xaxis->GetBinUpEdge(bin);

    if (debugMode_)
      edm::LogPrint("DiMuonMassBiasClient") << "dealing with bin: " << bin << " range: (" << std::setprecision(2)
                                            << low_edge << "," << std::setprecision(2) << high_edge << ")";
    TH1D* Proj = bareHisto->ProjectionY(Form("%s_proj_%i", key.c_str(), bin), bin, bin);
    Proj->SetTitle(Form("%s #in (%.2f,%.2f), bin: %i", Proj->GetTitle(), low_edge, high_edge, bin));

    diMuonMassBias::fitOutputs results = fitLineShape(Proj);

    if (results.isInvalid()) {
      edm::LogWarning("DiMuonMassBiasClient") << "the current bin has invalid data" << std::endl;
      continue;
    }

    // fill the mean profiles
    const Measurement1D& bias = results.getBias();
    meanHistos_[key]->setBinContent(bin, bias.value());
    meanHistos_[key]->setBinError(bin, bias.error());

    // fill the width profiles
    const Measurement1D& width = results.getWidth();
    widthHistos_[key]->setBinContent(bin, width.value());
    widthHistos_[key]->setBinError(bin, width.error());
  }
}

//-----------------------------------------------------------------------------------
void DiMuonMassBiasClient::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter)
//-----------------------------------------------------------------------------------
{
  edm::LogInfo("DiMuonMassBiasClient") << "DiMuonMassBiasClient::endLuminosityBlock";

  getMEsToHarvest(igetter);

  // book the histograms upfront
  if (useTH1s_) {
    bookMEs(ibooker);
  }

  for (const auto& element : harvestTargets_) {
    if (!useTH1s_) {
      // if using profiles
      this->fitAndFillProfile(element, ibooker);
    } else {
      // if using histograms
      this->fitAndFillHisto(element, ibooker);
    }
  }
}

//-----------------------------------------------------------------------------------
diMuonMassBias::fitOutputs DiMuonMassBiasClient::fitLineShape(TH1* hist, const bool& fitBackground) const
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

  // parameters of the Voigtian
  RooRealVar mean("#mu", "mean", meanConfig_[0], meanConfig_[1], meanConfig_[2]);          //90.0, 60.0, 120.0 (for Z)
  RooRealVar width("width", "width", widthConfig_[0], widthConfig_[1], widthConfig_[2]);   // 5.0,  0.0, 120.0 (for Z)
  RooRealVar sigma("#sigma", "sigma", sigmaConfig_[0], sigmaConfig_[1], sigmaConfig_[2]);  // 5.0,  0.0, 120.0 (for Z)
  RooVoigtian voigt("voigt", "voigt", InvMass, mean, width, sigma);

  // parameters of the Crystal-ball
  RooRealVar peakCB("peakCB", "peakCB", meanConfig_[0], meanConfig_[1], meanConfig_[2]);
  RooRealVar sigmaCB("#sigma", "sigma", sigmaConfig_[0], sigmaConfig_[1], sigmaConfig_[2]);
  RooRealVar alphaCB("#alpha", "alpha", 1., 0., 10.);
  RooRealVar nCB("n", "n", 1., 0., 100.);
  RooCBShape crystalball("crystalball", "crystalball", InvMass, peakCB, sigmaCB, alphaCB, nCB);

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
    RooArgList listPdf;
    if (useRooCBShape_) {
      if (useRooCMSShape_) {
        // crystal-ball + CMS-shape fit
        listPdf.add(crystalball);
        listPdf.add(exp_pdf);
      } else {
        // crystal-ball + exponential fit
        listPdf.add(crystalball);
        listPdf.add(expo);
      }
    } else {
      if (useRooCMSShape_) {
        // voigtian + CMS-shape fit
        listPdf.add(voigt);
        listPdf.add(exp_pdf);
      } else {
        // voigtian + exponential fit
        listPdf.add(voigt);
        listPdf.add(expo);
      }
    }

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
    if (useRooCBShape_) {
      // use crystal-ball for a fit-only signal
      crystalball.fitTo(datahist, RooFit::PrintLevel(-1));
      crystalball.plotOn(frame.get(), RooFit::LineColor(kRed));  //this will show fit overlay on canvas
      crystalball.paramOn(frame.get(),
                          RooFit::Layout(0.65, 0.90, 0.90));  //this will display the fit parameters on canvas
    } else {
      // use voigtian for a fit-only signal
      voigt.fitTo(datahist, RooFit::PrintLevel(-1));
      voigt.plotOn(frame.get(), RooFit::LineColor(kRed));            //this will show fit overlay on canvas
      voigt.paramOn(frame.get(), RooFit::Layout(0.65, 0.90, 0.90));  //this will display the fit parameters on canvas
    }
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

  float mass_mean = useRooCBShape_ ? peakCB.getVal() : mean.getVal();
  float mass_sigma = useRooCBShape_ ? sigmaCB.getVal() : sigma.getVal();

  float mass_mean_err = useRooCBShape_ ? peakCB.getError() : mean.getError();
  float mass_sigma_err = useRooCBShape_ ? sigmaCB.getError() : sigma.getError();

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
  desc.add<bool>("useTH1s", false);
  desc.add<bool>("fitBackground", false);
  desc.add<bool>("useRooCMSShape", false);
  desc.add<bool>("useRooCBShape", false);
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
