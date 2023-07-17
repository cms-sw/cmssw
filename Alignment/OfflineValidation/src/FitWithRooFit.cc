#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Alignment/OfflineValidation/interface/FitWithRooFit.h"

// Import TH1 histogram into a RooDataHist
rooPair FitWithRooFit::importTH1(TH1* histo, const double& inputXmin, const double& inputXmax) {
  double xMin = inputXmin;
  double xMax = inputXmax;
  if ((xMin == xMax) && xMin == 0) {
    xMin = histo->GetXaxis()->GetXmin();
    xMax = histo->GetXaxis()->GetXmax();
  }
  // Declare observable x
  RooRealVar x("InvMass", "di-muon mass M(#mu^{+}#mu^{-}) [GeV]", xMin, xMax);
  // Create a binned dataset that imports contents of TH1 and associates its contents to observable 'x'
  return (std::make_pair(x, new RooDataHist("dh", "dh", x, RooFit::Import(*histo))));
}

// Plot and fit a RooDataHist fitting signal and background
void FitWithRooFit::fit(TH1* histo,
                        const TString signalType,
                        const TString backgroundType,
                        const double& xMin,
                        const double& xMax,
                        bool sumW2Error) {
  reinitializeParameters();

  rooPair imported = importTH1(histo, xMin, xMax);
  RooRealVar x(imported.first);
  RooDataHist* dh = imported.second;

  // Make plot of binned dataset showing Poisson error bars (RooFit default)
  RooPlot* frame = x.frame(RooFit::Title("di-muon mass M(#mu^{+}#mu^{-}) [GeV]"));
  frame->SetName(TString(histo->GetName()) + "_frame");
  dh->plotOn(frame);

  // Build the composite model
  RooAbsPdf* model = buildModel(&x, signalType, backgroundType);

  std::unique_ptr<RooAbsReal> chi2{model->createChi2(*dh, RooFit::DataError(RooAbsData::SumW2))};

  // Fit the composite model
  // -----------------------
  // Fit with likelihood
  if (!useChi2_) {
    if (sumW2Error)
      model->fitTo(*dh, RooFit::Save(), RooFit::SumW2Error(kTRUE));
    else
      model->fitTo(*dh);
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
void FitWithRooFit::initMean(
    const double& value, const double& min, const double& max, const TString& name, const TString& title) {
  if (mean_ != nullptr)
    delete mean_;
  mean_ = new RooRealVar(name, title, value, min, max);
  initVal_mean = value;
}

void FitWithRooFit::initMean2(
    const double& value, const double& min, const double& max, const TString& name, const TString& title) {
  if (mean2_ != nullptr)
    delete mean2_;
  mean2_ = new RooRealVar(name, title, value, min, max);
  initVal_mean2 = value;
}

void FitWithRooFit::initMean3(
    const double& value, const double& min, const double& max, const TString& name, const TString& title) {
  if (mean3_ != nullptr)
    delete mean3_;
  mean3_ = new RooRealVar(name, title, value, min, max);
  initVal_mean3 = value;
}

void FitWithRooFit::initSigma(
    const double& value, const double& min, const double& max, const TString& name, const TString& title) {
  if (sigma_ != nullptr)
    delete sigma_;
  sigma_ = new RooRealVar(name, title, value, min, max);
  initVal_sigma = value;
}

void FitWithRooFit::initSigma2(
    const double& value, const double& min, const double& max, const TString& name, const TString& title) {
  if (sigma2_ != nullptr)
    delete sigma2_;
  sigma2_ = new RooRealVar(name, title, value, min, max);
  initVal_sigma2 = value;
}

void FitWithRooFit::initSigma3(
    const double& value, const double& min, const double& max, const TString& name, const TString& title) {
  if (sigma3_ != nullptr)
    delete sigma3_;
  sigma3_ = new RooRealVar(name, title, value, min, max);
  initVal_sigma3 = value;
}

void FitWithRooFit::initGamma(
    const double& value, const double& min, const double& max, const TString& name, const TString& title) {
  if (gamma_ != nullptr)
    delete gamma_;
  gamma_ = new RooRealVar(name, title, value, min, max);
  initVal_gamma = value;
}

void FitWithRooFit::initGaussFrac(
    const double& value, const double& min, const double& max, const TString& name, const TString& title) {
  if (gaussFrac_ != nullptr)
    delete gaussFrac_;
  gaussFrac_ = new RooRealVar(name, title, value, min, max);
  initVal_gaussFrac = value;
}

void FitWithRooFit::initGaussFrac2(
    const double& value, const double& min, const double& max, const TString& name, const TString& title) {
  if (gaussFrac2_ != nullptr)
    delete gaussFrac2_;
  gaussFrac2_ = new RooRealVar(name, title, value, min, max);
  initVal_gaussFrac2 = value;
}

void FitWithRooFit::initExpCoeffA0(
    const double& value, const double& min, const double& max, const TString& name, const TString& title) {
  if (expCoeffa0_ != nullptr)
    delete expCoeffa0_;
  expCoeffa0_ = new RooRealVar(name, title, value, min, max);
  initVal_expCoeffa0 = value;
}

void FitWithRooFit::initExpCoeffA1(
    const double& value, const double& min, const double& max, const TString& name, const TString& title) {
  if (expCoeffa1_ != nullptr)
    delete expCoeffa1_;
  expCoeffa1_ = new RooRealVar(name, title, value, min, max);
  initVal_expCoeffa1 = value;
}

void FitWithRooFit::initExpCoeffA2(
    const double& value, const double& min, const double& max, const TString& name, const TString& title) {
  if (expCoeffa2_ != nullptr)
    delete expCoeffa2_;
  expCoeffa2_ = new RooRealVar(name, title, value, min, max);
  initVal_expCoeffa2 = value;
}

void FitWithRooFit::initFsig(
    const double& value, const double& min, const double& max, const TString& name, const TString& title) {
  if (fsig_ != nullptr)
    delete fsig_;
  fsig_ = new RooRealVar(name, title, value, min, max);
  initVal_fsig = value;
}

void FitWithRooFit::initA0(
    const double& value, const double& min, const double& max, const TString& name, const TString& title) {
  if (a0_ != nullptr)
    delete a0_;
  a0_ = new RooRealVar(name, title, value, min, max);
  initVal_a0 = value;
}

void FitWithRooFit::initA1(
    const double& value, const double& min, const double& max, const TString& name, const TString& title) {
  if (a1_ != nullptr)
    delete a1_;
  a1_ = new RooRealVar(name, title, value, min, max);
  initVal_a1 = value;
}

void FitWithRooFit::initA2(
    const double& value, const double& min, const double& max, const TString& name, const TString& title) {
  if (a2_ != nullptr)
    delete a2_;
  a2_ = new RooRealVar(name, title, value, min, max);
  initVal_a2 = value;
}

void FitWithRooFit::initA3(
    const double& value, const double& min, const double& max, const TString& name, const TString& title) {
  if (a3_ != nullptr)
    delete a3_;
  a3_ = new RooRealVar(name, title, value, min, max);
  initVal_a3 = value;
}

void FitWithRooFit::initA4(
    const double& value, const double& min, const double& max, const TString& name, const TString& title) {
  if (a4_ != nullptr)
    delete a4_;
  a4_ = new RooRealVar(name, title, value, min, max);
  initVal_a4 = value;
}

void FitWithRooFit::initA5(
    const double& value, const double& min, const double& max, const TString& name, const TString& title) {
  if (a5_ != nullptr)
    delete a5_;
  a5_ = new RooRealVar(name, title, value, min, max);
  initVal_a5 = value;
}

void FitWithRooFit::initA6(
    const double& value, const double& min, const double& max, const TString& name, const TString& title) {
  if (a6_ != nullptr)
    delete a6_;
  a6_ = new RooRealVar(name, title, value, min, max);
  initVal_a6 = value;
}

void FitWithRooFit::initAlpha(
    const double& value, const double& min, const double& max, const TString& name, const TString& title) {
  if (alpha_ != nullptr)
    delete alpha_;
  alpha_ = new RooRealVar(name, title, value, min, max);
  initVal_alpha = value;
}

void FitWithRooFit::initN(
    const double& value, const double& min, const double& max, const TString& name, const TString& title) {
  if (n_ != nullptr)
    delete n_;
  n_ = new RooRealVar(name, title, value, min, max);
  initVal_n = value;
}

void FitWithRooFit::initFGCB(
    const double& value, const double& min, const double& max, const TString& name, const TString& title) {
  if (fGCB_ != nullptr)
    delete fGCB_;
  fGCB_ = new RooRealVar(name, title, value, min, max);
  initVal_fGCB = value;
}

void FitWithRooFit::reinitializeParameters() {
  if (mean_ != nullptr)
    mean_->setVal(initVal_mean);
  if (mean2_ != nullptr)
    mean2_->setVal(initVal_mean2);
  if (mean3_ != nullptr)
    mean3_->setVal(initVal_mean3);
  if (sigma_ != nullptr)
    sigma_->setVal(initVal_sigma);
  if (sigma2_ != nullptr)
    sigma2_->setVal(initVal_sigma2);
  if (sigma3_ != nullptr)
    sigma3_->setVal(initVal_sigma3);
  if (gamma_ != nullptr)
    gamma_->setVal(initVal_gamma);
  if (gaussFrac_ != nullptr)
    gaussFrac_->setVal(initVal_gaussFrac);
  if (gaussFrac2_ != nullptr)
    gaussFrac2_->setVal(initVal_gaussFrac2);
  if (expCoeffa0_ != nullptr)
    expCoeffa0_->setVal(initVal_expCoeffa0);
  if (expCoeffa1_ != nullptr)
    expCoeffa1_->setVal(initVal_expCoeffa1);
  if (expCoeffa2_ != nullptr)
    expCoeffa2_->setVal(initVal_expCoeffa2);
  if (fsig_ != nullptr)
    fsig_->setVal(initVal_fsig);
  if (a0_ != nullptr)
    a0_->setVal(initVal_a0);
  if (a1_ != nullptr)
    a1_->setVal(initVal_a1);
  if (a2_ != nullptr)
    a2_->setVal(initVal_a2);
  if (a3_ != nullptr)
    a3_->setVal(initVal_a3);
  if (a4_ != nullptr)
    a4_->setVal(initVal_a4);
  if (a5_ != nullptr)
    a5_->setVal(initVal_a5);
  if (a6_ != nullptr)
    a6_->setVal(initVal_a6);
  if (alpha_ != nullptr)
    alpha_->setVal(initVal_alpha);
  if (n_ != nullptr)
    n_->setVal(initVal_n);
  if (fGCB_ != nullptr)
    fGCB_->setVal(initVal_fGCB);
}

/// Build the model for the specified signal type
RooAbsPdf* FitWithRooFit::buildSignalModel(RooRealVar* x, const TString& signalType) {
  RooAbsPdf* signal = nullptr;
  if (signalType == "gaussian") {
    // Fit a Gaussian p.d.f to the data
    if ((mean_ == nullptr) || (sigma_ == nullptr)) {
      edm::LogError("FitWithRooFit")
          << "Error: one or more parameters are not initialized. Please be sure to initialize mean and sigma";
      exit(1);
    }
    signal = new RooGaussian("gauss", "gauss", *x, *mean_, *sigma_);
  } else if (signalType == "doubleGaussian") {
    // Fit with double gaussian
    if ((mean_ == nullptr) || (sigma_ == nullptr) || (sigma2_ == nullptr)) {
      edm::LogError("FitWithRooFit")
          << "Error: one or more parameters are not initialized. Please be sure to initialize mean, sigma and sigma2";
      exit(1);
    }
    RooGaussModel* gaussModel = new RooGaussModel("gaussModel", "gaussModel", *x, *mean_, *sigma_);
    RooGaussModel* gaussModel2 = new RooGaussModel("gaussModel2", "gaussModel2", *x, *mean_, *sigma2_);
    signal = new RooAddModel("doubleGaussian", "double gaussian", RooArgList(*gaussModel, *gaussModel2), *gaussFrac_);
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
    signal = new RooAddModel("tripleGaussian",
                             "triple gaussian",
                             RooArgList(*gaussModel, *gaussModel2, *gaussModel3),
                             RooArgList(*gaussFrac_, *gaussFrac2_));
  } else if (signalType == "breitWigner") {
    // Fit a Breit-Wigner
    if ((mean_ == nullptr) || (gamma_ == nullptr)) {
      edm::LogError("FitWithRooFit")
          << "Error: one or more parameters are not initialized. Please be sure to initialize mean and gamma";
      exit(1);
    }
    signal = new RooBreitWigner("breiWign", "breitWign", *x, *mean_, *gamma_);
  } else if (signalType == "relBreitWigner") {
    // Fit a relativistic Breit-Wigner
    if ((mean_ == nullptr) || (gamma_ == nullptr)) {
      edm::LogError("FitWithRooFit")
          << "Error: one or more parameters are not initialized. Please be sure to initialize mean and gamma";
      exit(1);
    }
    signal = new RooGenericPdf("Relativistic Breit-Wigner",
                               "RBW",
                               "@0/(pow(@0*@0 - @1*@1,2) + @2*@2*@0*@0*@0*@0/(@1*@1))",
                               RooArgList(*x, *mean_, *gamma_));
  } else if (signalType == "voigtian") {
    // Fit a Voigtian
    if ((mean_ == nullptr) || (sigma_ == nullptr) || (gamma_ == nullptr)) {
      edm::LogError("FitWithRooFit")
          << "Error: one or more parameters are not initialized. Please be sure to initialize mean, sigma and gamma";
      exit(1);
    }
    signal = new RooVoigtian("voigt", "voigt", *x, *mean_, *gamma_, *sigma_);
  } else if (signalType == "crystalBall") {
    // Fit a CrystalBall
    if ((mean_ == nullptr) || (sigma_ == nullptr) || (alpha_ == nullptr) || (n_ == nullptr)) {
      edm::LogError("FitWithRooFit")
          << "Error: one or more parameters are not initialized. Please be sure to initialize mean, sigma, "
             "alpha and n";
      exit(1);
    }
    signal = new RooCBShape("crystalBall", "crystalBall", *x, *mean_, *sigma_, *alpha_, *n_);
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
    signal = new RooFFTConvPdf("breitWignerTimesCB", "breitWignerTimesCB", *x, *bw, *cb);
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
                                          RooArgList(*x, *mean_, *gamma_));
    RooAbsPdf* cb = new RooCBShape("crystalBall", "crystalBall", *x, *mean2_, *sigma_, *alpha_, *n_);
    signal = new RooFFTConvPdf("relBreitWignerTimesCB", "relBreitWignerTimesCB", *x, *bw, *cb);
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

    signal =
        new RooAddPdf("gaussianPlusCrystalBall", "gaussianPlusCrystalBall", RooArgList(*tempCB, *tempGaussian), *fGCB_);
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

    signal =
        new RooAddPdf("voigtianPlusCrystalBall", "voigtianPlusCrystalBall", RooArgList(*tempCB, *tempVoigt), *fGCB_);
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

    signal =
        new RooAddPdf("breitWignerPlusCrystalBall", "breitWignerPlusCrystalBall", RooArgList(*tempCB, *tempBW), *fGCB_);
  }

  else if (signalType != "") {
    edm::LogError("FitWithRooFit") << "Unknown signal function: " << signalType << ". Signal will not be in the model";
  }
  return signal;
}

/// Build the model for the specified background type
RooAbsPdf* FitWithRooFit::buildBackgroundModel(RooRealVar* x, const TString& backgroundType) {
  RooAbsPdf* background = nullptr;
  if (backgroundType == "exponential") {
    // Add an exponential for the background
    if ((expCoeffa1_ == nullptr) || (fsig_ == nullptr)) {
      edm::LogError("FitWithRooFit")
          << "Error: one or more parameters are not initialized. Please be sure to initialize expCoeffa1 and fsig";
      exit(1);
    }
    background = new RooExponential("exponential", "exponential", *x, *expCoeffa1_);
  }

  if (backgroundType == "exponentialpol") {
    // Add an exponential for the background
    if ((expCoeffa0_ == nullptr) || (expCoeffa1_ == nullptr) || (expCoeffa2_ == nullptr) || (fsig_ == nullptr)) {
      edm::LogError("FitWithRooFit")
          << "Error: one or more parameters are not initialized. Please be sure to initialize expCoeff and fsig";
      exit(1);
    }
    background = new RooGenericPdf("exponential",
                                   "exponential",
                                   "TMath::Exp(@1+@2*@0+@3*@0*@0)",
                                   RooArgList(*x, *expCoeffa0_, *expCoeffa1_, *expCoeffa2_));
  }

  else if (backgroundType == "chebychev0") {
    // Add a linear background
    if (a0_ == nullptr) {
      edm::LogError("FitWithRooFit")
          << "Error: one or more parameters are not initialized. Please be sure to initialize a0";
      exit(1);
    }
    background = new RooChebychev("chebychev0", "chebychev0", *x, *a0_);
  } else if (backgroundType == "chebychev1") {
    // Add a 2nd order chebychev polynomial background
    if ((a0_ == nullptr) || (a1_ == nullptr)) {
      edm::LogError("FitWithRooFit")
          << "Error: one or more parameters are not initialized. Please be sure to initialize a0 and a1";
      exit(1);
    }
    background = new RooChebychev("chebychev1", "chebychev1", *x, RooArgList(*a0_, *a1_));
  } else if (backgroundType == "chebychev3") {
    // Add a 3rd order chebychev polynomial background
    if ((a0_ == nullptr) || (a1_ == nullptr) || (a2_ == nullptr) || (a3_ == nullptr)) {
      edm::LogError("FitWithRooFit")
          << "Error: one or more parameters are not initialized. Please be sure to initialize a0, a1, a2 and a3";
      exit(1);
    }
    background = new RooChebychev("3rdOrderPol", "3rdOrderPol", *x, RooArgList(*a0_, *a1_, *a2_, *a3_));
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
    background =
        new RooChebychev("6thOrderPol", "6thOrderPol", *x, RooArgList(*a0_, *a1_, *a2_, *a3_, *a4_, *a5_, *a6_));
  }

  return background;
}

/// Build the model to fit
RooAbsPdf* FitWithRooFit::buildModel(RooRealVar* x, const TString& signalType, const TString& backgroundType) {
  RooAbsPdf* model = nullptr;

  RooAbsPdf* signal = buildSignalModel(x, signalType);
  RooAbsPdf* background = buildBackgroundModel(x, backgroundType);

  if ((signal != nullptr) && (background != nullptr)) {
    // Combine signal and background pdfs
    edm::LogPrint("FitWithRooFit") << "Building model with signal and backgound";
    model = new RooAddPdf("model", "model", RooArgList(*signal, *background), *fsig_);
  } else if (signal != nullptr) {
    edm::LogPrint("FitWithRooFit") << "Building model with signal";
    model = signal;
  } else if (background != nullptr) {
    edm::LogPrint("FitWithRooFit") << "Building model with backgound";
    model = background;
  } else {
    edm::LogWarning("FitWithRooFit") << "Nothing to fit";
    exit(0);
  }
  return model;
}
