#include "Alignment/OfflineValidation/interface/FitWithRooFit.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Import TH1 histogram into a RooDataHist
std::unique_ptr<RooDataHist> FitWithRooFit::importTH1(TH1* histo, double xMin, double xMax) {
  if ((xMin == xMax) && xMin == 0) {
    xMin = histo->GetXaxis()->GetXmin();
    xMax = histo->GetXaxis()->GetXmax();
  }
  // Declare observable x
  RooRealVar x("InvMass", "di-muon mass M(#mu^{+}#mu^{-}) [GeV]", xMin, xMax);
  // Create a binned dataset that imports contents of TH1 and associates its contents to observable 'x'
  return std::make_unique<RooDataHist>("dh", "dh", x, RooFit::Import(*histo));
}

// Plot and fit a RooDataHist fitting signal and background
void FitWithRooFit::fit(
    TH1* histo, const TString signalType, const TString backgroundType, double xMin, double xMax, bool sumW2Error) {
  reinitializeParameters();

  std::unique_ptr<RooDataHist> dh = importTH1(histo, xMin, xMax);
  RooRealVar x(*static_cast<RooRealVar*>(dh->get()->find("x")));

  // Make plot of binned dataset showing Poisson error bars (RooFit default)
  RooPlot* frame = x.frame(RooFit::Title("di-muon mass M(#mu^{+}#mu^{-}) [GeV]"));
  frame->SetName(TString(histo->GetName()) + "_frame");
  dh->plotOn(frame);

  // Build the composite model
  std::unique_ptr<RooAbsPdf> model = buildModel(&x, signalType, backgroundType);

  std::unique_ptr<RooAbsReal> chi2{model->createChi2(*dh, RooFit::DataError(RooAbsData::SumW2))};

  // Fit the composite model
  // -----------------------
  // Fit with likelihood
  if (!useChi2_) {
    model->fitTo(*dh, RooFit::SumW2Error(sumW2Error));
  }
  // Fit with chi^2
  else {
    edm::LogWarning("FitWithRooFit") << "FITTING WITH CHI^2";
    RooMinimizer m(*chi2);
    m.migrad();
    m.hesse();
    // RooFitResult* r_chi2_wgt = m.save();
  }
  model->plotOn(frame, RooFit::LineColor(kRed));
  model->plotOn(frame, RooFit::Components(backgroundType), RooFit::LineStyle(kDashed));
  model->paramOn(frame,
                 RooFit::Label("fit result"),
                 RooFit::Layout(0.65, 0.90, 0.90),
                 RooFit::Format("NEU", RooFit::AutoPrecision(2)));

  // TODO: fix next lines to get the prob(chi2) (ndof should be dynamically set according to the choosen pdf)
  // double chi2 = xframe.chiSquare("model","data",ndof);
  // double ndoff = xframeGetNbinsX();
  // double chi2prob = TMath::Prob(chi2,ndoff);

  // P l o t   a n d   f i t   a   R o o D a t a H i s t   w i t h   i n t e r n a l   e r r o r s
  // ---------------------------------------------------------------------------------------------

  // If histogram has custom error (i.e. its contents is does not originate from a Poisson process
  // but e.g. is a sum of weighted events) you can data with symmetric 'sum-of-weights' error instead
  // (same error bars as shown by ROOT)
  RooPlot* frame2 = x.frame(RooFit::Title("di-muon mass M(#mu^{+}#mu^{-}) [GeV]"));
  dh->plotOn(frame2, RooFit::DataError(RooAbsData::SumW2));
  model->plotOn(frame2, RooFit::LineColor(kRed));
  model->plotOn(frame2, RooFit::Components(backgroundType), RooFit::LineStyle(kDashed));
  model->paramOn(frame2,
                 RooFit::Label("fit result"),
                 RooFit::Layout(0.65, 0.90, 0.90),
                 RooFit::Format("NEU", RooFit::AutoPrecision(2)));

  // Please note that error bars shown (Poisson or SumW2) are for visualization only, the are NOT used
  // in a maximum likelihood fit
  //
  // A (binned) ML fit will ALWAYS assume the Poisson error interpretation of data (the mathematical definition
  // of likelihood does not take any external definition of errors). Data with non-unit weights can only be correctly
  // fitted with a chi^2 fit (see rf602_chi2fit.C)

  // Draw all frames on a canvas
  // if( canvas == 0 ) {
  //   canvas = new TCanvas("rf102_dataimport","rf102_dataimport",800,800);
  //   canvas->cd(1);
  // }
  // canvas->Divide(2,1);
  // canvas->cd(1) ; frame->Draw();
  // canvas->cd(2) ; frame2->Draw();

  frame2->Draw();
}

// Initialization methods for all the parameters
void FitWithRooFit::initMean(double value, double min, double max, const TString& name, const TString& title) {
  mean_ = std::make_unique<RooRealVar>(name, title, value, min, max);
  initVal_mean = value;
}

void FitWithRooFit::initMean2(double value, double min, double max, const TString& name, const TString& title) {
  mean2_ = std::make_unique<RooRealVar>(name, title, value, min, max);
  initVal_mean2 = value;
}

void FitWithRooFit::initMean3(double value, double min, double max, const TString& name, const TString& title) {
  mean3_ = std::make_unique<RooRealVar>(name, title, value, min, max);
  initVal_mean3 = value;
}

void FitWithRooFit::initSigma(double value, double min, double max, const TString& name, const TString& title) {
  sigma_ = std::make_unique<RooRealVar>(name, title, value, min, max);
  initVal_sigma = value;
}

void FitWithRooFit::initSigma2(double value, double min, double max, const TString& name, const TString& title) {
  sigma2_ = std::make_unique<RooRealVar>(name, title, value, min, max);
  initVal_sigma2 = value;
}

void FitWithRooFit::initSigma3(double value, double min, double max, const TString& name, const TString& title) {
  sigma3_ = std::make_unique<RooRealVar>(name, title, value, min, max);
  initVal_sigma3 = value;
}

void FitWithRooFit::initGamma(double value, double min, double max, const TString& name, const TString& title) {
  gamma_ = std::make_unique<RooRealVar>(name, title, value, min, max);
  initVal_gamma = value;
}

void FitWithRooFit::initGaussFrac(double value, double min, double max, const TString& name, const TString& title) {
  gaussFrac_ = std::make_unique<RooRealVar>(name, title, value, min, max);
  initVal_gaussFrac = value;
}

void FitWithRooFit::initGaussFrac2(double value, double min, double max, const TString& name, const TString& title) {
  gaussFrac2_ = std::make_unique<RooRealVar>(name, title, value, min, max);
  initVal_gaussFrac2 = value;
}

void FitWithRooFit::initExpCoeffA0(double value, double min, double max, const TString& name, const TString& title) {
  expCoeffa0_ = std::make_unique<RooRealVar>(name, title, value, min, max);
  initVal_expCoeffa0 = value;
}

void FitWithRooFit::initExpCoeffA1(double value, double min, double max, const TString& name, const TString& title) {
  expCoeffa1_ = std::make_unique<RooRealVar>(name, title, value, min, max);
  initVal_expCoeffa1 = value;
}

void FitWithRooFit::initExpCoeffA2(double value, double min, double max, const TString& name, const TString& title) {
  expCoeffa2_ = std::make_unique<RooRealVar>(name, title, value, min, max);
  initVal_expCoeffa2 = value;
}

void FitWithRooFit::initFsig(double value, double min, double max, const TString& name, const TString& title) {
  fsig_ = std::make_unique<RooRealVar>(name, title, value, min, max);
  initVal_fsig = value;
}

void FitWithRooFit::initA0(double value, double min, double max, const TString& name, const TString& title) {
  a0_ = std::make_unique<RooRealVar>(name, title, value, min, max);
  initVal_a0 = value;
}

void FitWithRooFit::initA1(double value, double min, double max, const TString& name, const TString& title) {
  a1_ = std::make_unique<RooRealVar>(name, title, value, min, max);
  initVal_a1 = value;
}

void FitWithRooFit::initA2(double value, double min, double max, const TString& name, const TString& title) {
  a2_ = std::make_unique<RooRealVar>(name, title, value, min, max);
  initVal_a2 = value;
}

void FitWithRooFit::initA3(double value, double min, double max, const TString& name, const TString& title) {
  a3_ = std::make_unique<RooRealVar>(name, title, value, min, max);
  initVal_a3 = value;
}

void FitWithRooFit::initA4(double value, double min, double max, const TString& name, const TString& title) {
  a4_ = std::make_unique<RooRealVar>(name, title, value, min, max);
  initVal_a4 = value;
}

void FitWithRooFit::initA5(double value, double min, double max, const TString& name, const TString& title) {
  a5_ = std::make_unique<RooRealVar>(name, title, value, min, max);
  initVal_a5 = value;
}

void FitWithRooFit::initA6(double value, double min, double max, const TString& name, const TString& title) {
  a6_ = std::make_unique<RooRealVar>(name, title, value, min, max);
  initVal_a6 = value;
}

void FitWithRooFit::initAlpha(double value, double min, double max, const TString& name, const TString& title) {
  alpha_ = std::make_unique<RooRealVar>(name, title, value, min, max);
  initVal_alpha = value;
}

void FitWithRooFit::initN(double value, double min, double max, const TString& name, const TString& title) {
  n_ = std::make_unique<RooRealVar>(name, title, value, min, max);
  initVal_n = value;
}

void FitWithRooFit::initFGCB(double value, double min, double max, const TString& name, const TString& title) {
  fGCB_ = std::make_unique<RooRealVar>(name, title, value, min, max);
  initVal_fGCB = value;
}

void FitWithRooFit::reinitializeParameters() {
  auto initParam = [](std::unique_ptr<RooRealVar>& var, double val) {
    if (var)
      var->setVal(val);
  };

  initParam(mean_, initVal_mean);
  initParam(mean2_, initVal_mean2);
  initParam(mean3_, initVal_mean3);
  initParam(sigma_, initVal_sigma);
  initParam(sigma2_, initVal_sigma2);
  initParam(sigma3_, initVal_sigma3);
  initParam(gamma_, initVal_gamma);
  initParam(gaussFrac_, initVal_gaussFrac);
  initParam(gaussFrac2_, initVal_gaussFrac2);
  initParam(expCoeffa0_, initVal_expCoeffa0);
  initParam(expCoeffa1_, initVal_expCoeffa1);
  initParam(expCoeffa2_, initVal_expCoeffa2);
  initParam(fsig_, initVal_fsig);
  initParam(a0_, initVal_a0);
  initParam(a1_, initVal_a1);
  initParam(a2_, initVal_a2);
  initParam(a3_, initVal_a3);
  initParam(a4_, initVal_a4);
  initParam(a5_, initVal_a5);
  initParam(a6_, initVal_a6);
  initParam(alpha_, initVal_alpha);
  initParam(n_, initVal_n);
  initParam(fGCB_, initVal_fGCB);
}

/// Build the model for the specified signal type
std::unique_ptr<RooAbsPdf> FitWithRooFit::buildSignalModel(RooRealVar* x, const TString& signalType) {
  if (signalType == "gaussian") {
    // Fit a Gaussian p.d.f to the data
    if ((mean_ == nullptr) || (sigma_ == nullptr)) {
      edm::LogError("FitWithRooFit")
          << "Error: one or more parameters are not initialized. Please be sure to initialize mean and sigma";
      exit(1);
    }
    return std::make_unique<RooGaussian>("gauss", "gauss", *x, *mean_, *sigma_);
  } else if (signalType == "doubleGaussian") {
    // Fit with double gaussian
    if ((mean_ == nullptr) || (sigma_ == nullptr) || (sigma2_ == nullptr)) {
      edm::LogError("FitWithRooFit")
          << "Error: one or more parameters are not initialized. Please be sure to initialize mean, sigma and sigma2";
      exit(1);
    }
    RooGaussModel* gaussModel = new RooGaussModel("gaussModel", "gaussModel", *x, *mean_, *sigma_);
    RooGaussModel* gaussModel2 = new RooGaussModel("gaussModel2", "gaussModel2", *x, *mean_, *sigma2_);
    RooArgList components{*gaussModel, *gaussModel2};
    auto out = std::make_unique<RooAddModel>("doubleGaussian", "double gaussian", components, *gaussFrac_);
    out->addOwnedComponents(components);
    return out;
  } else if (signalType == "tripleGaussian") {
    // Fit with triple gaussian
    if ((mean_ == nullptr) || (mean2_ == nullptr) || (mean3_ == nullptr) || (sigma_ == nullptr) ||
        (sigma2_ == nullptr) || (sigma3_ == nullptr)) {
      edm::LogError("FitWithRooFit")
          << "Error: one or more parameters are not initialized. Please be sure to initialize mean, mean2, "
             "mean3, sigma, sigma2, sigma3";
      exit(1);
    }
    RooGaussModel* gaussModel = new RooGaussModel("gaussModel", "gaussModel", *x, *mean_, *sigma_);
    RooGaussModel* gaussModel2 = new RooGaussModel("gaussModel2", "gaussModel2", *x, *mean2_, *sigma2_);
    RooGaussModel* gaussModel3 = new RooGaussModel("gaussModel3", "gaussModel3", *x, *mean3_, *sigma3_);
    RooArgList components{*gaussModel, *gaussModel2, *gaussModel3};
    auto out = std::make_unique<RooAddModel>(
        "tripleGaussian", "triple gaussian", components, RooArgList{*gaussFrac_, *gaussFrac2_});
    out->addOwnedComponents(components);
    return out;
  } else if (signalType == "breitWigner") {
    // Fit a Breit-Wigner
    if ((mean_ == nullptr) || (gamma_ == nullptr)) {
      edm::LogError("FitWithRooFit")
          << "Error: one or more parameters are not initialized. Please be sure to initialize mean and gamma";
      exit(1);
    }
    return std::make_unique<RooBreitWigner>("breiWign", "breitWign", *x, *mean_, *gamma_);
  } else if (signalType == "relBreitWigner") {
    // Fit a relativistic Breit-Wigner
    if ((mean_ == nullptr) || (gamma_ == nullptr)) {
      edm::LogError("FitWithRooFit")
          << "Error: one or more parameters are not initialized. Please be sure to initialize mean and gamma";
      exit(1);
    }
    return std::make_unique<RooGenericPdf>("Relativistic Breit-Wigner",
                                           "RBW",
                                           "@0/(pow(@0*@0 - @1*@1,2) + @2*@2*@0*@0*@0*@0/(@1*@1))",
                                           RooArgList{*x, *mean_, *gamma_});
  } else if (signalType == "voigtian") {
    // Fit a Voigtian
    if ((mean_ == nullptr) || (sigma_ == nullptr) || (gamma_ == nullptr)) {
      edm::LogError("FitWithRooFit")
          << "Error: one or more parameters are not initialized. Please be sure to initialize mean, sigma and gamma";
      exit(1);
    }
    return std::make_unique<RooVoigtian>("voigt", "voigt", *x, *mean_, *gamma_, *sigma_);
  } else if (signalType == "crystalBall") {
    // Fit a CrystalBall
    if ((mean_ == nullptr) || (sigma_ == nullptr) || (alpha_ == nullptr) || (n_ == nullptr)) {
      edm::LogError("FitWithRooFit")
          << "Error: one or more parameters are not initialized. Please be sure to initialize mean, sigma, "
             "alpha and n";
      exit(1);
    }
    return std::make_unique<RooCBShape>("crystalBall", "crystalBall", *x, *mean_, *sigma_, *alpha_, *n_);
  } else if (signalType == "breitWignerTimesCB") {
    // Fit a Breit Wigner convoluted with a CrystalBall
    if ((mean_ == nullptr) || (mean2_ == nullptr) || (sigma_ == nullptr) || (gamma_ == nullptr) ||
        (alpha_ == nullptr) || (n_ == nullptr)) {
      edm::LogError("FitWithRooFit")
          << "Error: one or more parameters are not initialized. Please be sure to initialize mean, mean2, "
             "sigma, gamma, alpha and n";
      exit(1);
    }
    RooAbsPdf* bw = new RooBreitWigner("breiWigner", "breitWigner", *x, *mean_, *gamma_);
    RooAbsPdf* cb = new RooCBShape("crystalBall", "crystalBall", *x, *mean2_, *sigma_, *alpha_, *n_);
    auto out = std::make_unique<RooFFTConvPdf>("breitWignerTimesCB", "breitWignerTimesCB", *x, *bw, *cb);
    out->addOwnedComponents({*bw, *cb});
    return out;
  } else if (signalType == "relBreitWignerTimesCB") {
    // Fit a relativistic Breit Wigner convoluted with a CrystalBall
    if ((mean_ == nullptr) || (mean2_ == nullptr) || (sigma_ == nullptr) || (gamma_ == nullptr) ||
        (alpha_ == nullptr) || (n_ == nullptr)) {
      edm::LogError("FitWithRooFit")
          << "Error: one or more parameters are not initialized. Please be sure to initialize mean, mean2, "
             "sigma, gamma, alpha and n";
      exit(1);
    }
    RooGenericPdf* bw = new RooGenericPdf("Relativistic Breit-Wigner",
                                          "RBW",
                                          "@0/(pow(@0*@0 - @1*@1,2) + @2*@2*@0*@0*@0*@0/(@1*@1))",
                                          {*x, *mean_, *gamma_});
    RooAbsPdf* cb = new RooCBShape("crystalBall", "crystalBall", *x, *mean2_, *sigma_, *alpha_, *n_);
    auto out = std::make_unique<RooFFTConvPdf>("relBreitWignerTimesCB", "relBreitWignerTimesCB", *x, *bw, *cb);
    out->addOwnedComponents({*bw, *cb});
    return out;
  } else if (signalType == "gaussianPlusCrystalBall") {
    // Fit a Gaussian + CrystalBall with the same mean
    if ((mean_ == nullptr) || (sigma_ == nullptr) || (alpha_ == nullptr) || (n_ == nullptr) || (sigma2_ == nullptr) ||
        (fGCB_ == nullptr)) {
      edm::LogError("FitWithRooFit")
          << "Error: one or more parameters are not initialized. Please be sure to initialize mean, sigma, "
             "sigma2, alpha, n and fGCB";
      exit(1);
    }
    RooAbsPdf* tempCB = new RooCBShape("crystalBall", "crystalBall", *x, *mean_, *sigma_, *alpha_, *n_);
    RooAbsPdf* tempGaussian = new RooGaussian("gauss", "gauss", *x, *mean_, *sigma2_);
    RooArgList components{*tempCB, *tempGaussian};

    auto out = std::make_unique<RooAddPdf>("gaussianPlusCrystalBall", "gaussianPlusCrystalBall", components, *fGCB_);
    out->addOwnedComponents(components);
    return out;
  } else if (signalType == "voigtianPlusCrystalBall") {
    // Fit a Voigtian + CrystalBall with the same mean
    if ((mean_ == nullptr) || (sigma_ == nullptr) || (gamma_ == nullptr) || (alpha_ == nullptr) || (n_ == nullptr) ||
        (sigma2_ == nullptr) || (fGCB_ == nullptr)) {
      edm::LogError("FitWithRooFit")
          << "Error: one or more parameters are not initialized. Please be sure to initialize mean, gamma, "
             "sigma, sigma2, alpha, n and fGCB";
      exit(1);
    }
    RooAbsPdf* tempVoigt = new RooVoigtian("voigt", "voigt", *x, *mean_, *gamma_, *sigma_);
    RooAbsPdf* tempCB = new RooCBShape("crystalBall", "crystalBall", *x, *mean_, *sigma2_, *alpha_, *n_);
    RooArgList components{*tempVoigt, *tempCB};

    auto out = std::make_unique<RooAddPdf>("voigtianPlusCrystalBall", "voigtianPlusCrystalBall", components, *fGCB_);
    out->addOwnedComponents(components);
    return out;
  } else if (signalType == "breitWignerPlusCrystalBall") {
    // Fit a Breit-Wigner + CrystalBall with the same mean
    if ((mean_ == nullptr) || (gamma_ == nullptr) || (alpha_ == nullptr) || (n_ == nullptr) || (sigma2_ == nullptr) ||
        (fGCB_ == nullptr)) {
      edm::LogError("FitWithRooFit")
          << "Error: one or more parameters are not initialized. Please be sure to initialize mean, gamma, "
             "sigma, alpha, n and fGCB";
      exit(1);
    }
    RooAbsPdf* tempBW = new RooBreitWigner("breitWign", "breitWign", *x, *mean_, *gamma_);
    RooAbsPdf* tempCB = new RooCBShape("crystalBall", "crystalBall", *x, *mean_, *sigma2_, *alpha_, *n_);
    RooArgList components{*tempCB, *tempBW};

    auto out =
        std::make_unique<RooAddPdf>("breitWignerPlusCrystalBall", "breitWignerPlusCrystalBall", components, *fGCB_);
    out->addOwnedComponents(components);
    return out;
  }

  else if (signalType != "") {
    edm::LogError("FitWithRooFit") << "Unknown signal function: " << signalType << ". Signal will not be in the model";
  }
  return nullptr;
}

/// Build the model for the specified background type
std::unique_ptr<RooAbsPdf> FitWithRooFit::buildBackgroundModel(RooRealVar* x, const TString& backgroundType) {
  if (backgroundType == "exponential") {
    // Add an exponential for the background
    if ((expCoeffa1_ == nullptr) || (fsig_ == nullptr)) {
      edm::LogError("FitWithRooFit")
          << "Error: one or more parameters are not initialized. Please be sure to initialize expCoeffa1 and fsig";
      exit(1);
    }
    return std::make_unique<RooExponential>("exponential", "exponential", *x, *expCoeffa1_);
  }

  if (backgroundType == "exponentialpol") {
    // Add an exponential for the background
    if ((expCoeffa0_ == nullptr) || (expCoeffa1_ == nullptr) || (expCoeffa2_ == nullptr) || (fsig_ == nullptr)) {
      edm::LogError("FitWithRooFit")
          << "Error: one or more parameters are not initialized. Please be sure to initialize expCoeff and fsig";
      exit(1);
    }
    return std::make_unique<RooGenericPdf>("exponential",
                                           "exponential",
                                           "TMath::Exp(@1+@2*@0+@3*@0*@0)",
                                           RooArgList{*x, *expCoeffa0_, *expCoeffa1_, *expCoeffa2_});
  }

  else if (backgroundType == "chebychev0") {
    // Add a linear background
    if (a0_ == nullptr) {
      edm::LogError("FitWithRooFit")
          << "Error: one or more parameters are not initialized. Please be sure to initialize a0";
      exit(1);
    }
    return std::make_unique<RooChebychev>("chebychev0", "chebychev0", *x, *a0_);
  } else if (backgroundType == "chebychev1") {
    // Add a 2nd order chebychev polynomial background
    if ((a0_ == nullptr) || (a1_ == nullptr)) {
      edm::LogError("FitWithRooFit")
          << "Error: one or more parameters are not initialized. Please be sure to initialize a0 and a1";
      exit(1);
    }
    return std::make_unique<RooChebychev>("chebychev1", "chebychev1", *x, RooArgList{*a0_, *a1_});
  } else if (backgroundType == "chebychev3") {
    // Add a 3rd order chebychev polynomial background
    if ((a0_ == nullptr) || (a1_ == nullptr) || (a2_ == nullptr) || (a3_ == nullptr)) {
      edm::LogError("FitWithRooFit")
          << "Error: one or more parameters are not initialized. Please be sure to initialize a0, a1, a2 and a3";
      exit(1);
    }
    return std::make_unique<RooChebychev>("3rdOrderPol", "3rdOrderPol", *x, RooArgList{*a0_, *a1_, *a2_, *a3_});
  }

  else if (backgroundType == "chebychev6") {
    // Add a 6th order chebychev polynomial background
    if ((a0_ == nullptr) || (a1_ == nullptr) || (a2_ == nullptr) || (a3_ == nullptr) || (a4_ == nullptr) ||
        (a5_ == nullptr) || (a6_ == nullptr)) {
      edm::LogError("FitWithRooFit")
          << "Error: one or more parameters are not initialized. Please be sure to initialize a0, a1, a2, a3, "
             "a4, a5 and a6";
      exit(1);
    }
    return std::make_unique<RooChebychev>(
        "6thOrderPol", "6thOrderPol", *x, RooArgList{*a0_, *a1_, *a2_, *a3_, *a4_, *a5_, *a6_});
  }
  return nullptr;
}

/// Build the model to fit
std::unique_ptr<RooAbsPdf> FitWithRooFit::buildModel(RooRealVar* x,
                                                     const TString& signalType,
                                                     const TString& backgroundType) {
  std::unique_ptr<RooAbsPdf> model;

  std::unique_ptr<RooAbsPdf> signal = buildSignalModel(x, signalType);
  std::unique_ptr<RooAbsPdf> background = buildBackgroundModel(x, backgroundType);

  if ((signal != nullptr) && (background != nullptr)) {
    // Combine signal and background pdfs
    edm::LogPrint("FitWithRooFit") << "Building model with signal and backgound";
    RooArgList components{*signal.release(), *background.release()};
    model = std::make_unique<RooAddPdf>("model", "model", components, *fsig_);
    model->addOwnedComponents(components);
  } else if (signal != nullptr) {
    edm::LogPrint("FitWithRooFit") << "Building model with signal";
    model = std::move(signal);
  } else if (background != nullptr) {
    edm::LogPrint("FitWithRooFit") << "Building model with backgound";
    model = std::move(background);
  } else {
    edm::LogWarning("FitWithRooFit") << "Nothing to fit";
    exit(0);
  }
  return model;
}
