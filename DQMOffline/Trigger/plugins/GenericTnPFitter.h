#include "RooGaussian.h"
#include "RooChebychev.h"
#include "RooVoigtian.h"
#include "RooExponential.h"
#include "RooRealVar.h"
#include "RooFormulaVar.h"
#include "RooDataHist.h"
#include "RooAddPdf.h"
#include "RooGlobalFunc.h"
#include "RooCategory.h"
#include "RooSimultaneous.h"
#include "RooFitResult.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TH3F.h"
#include "TProfile2D.h"
#include "TCanvas.h"
#include "RooPlot.h"

namespace dqmTnP {

  class AbstractFitter {
  protected:
    RooRealVar mass;
    RooRealVar mean;
    double expectedMean;
    RooRealVar sigma;
    double expectedSigma;
    RooRealVar efficiency;
    RooRealVar nSignalAll;
    RooFormulaVar nSignalPass;
    RooFormulaVar nSignalFail;
    RooRealVar nBackgroundFail;
    RooRealVar nBackgroundPass;
    RooCategory category;
    RooSimultaneous simPdf;
    RooDataHist* data;
    double chi2;
    bool verbose;

  public:
    AbstractFitter(bool verbose_ = false)
        : mass("mass", "mass", 0., 100., "GeV"),
          mean("mean", "mean", 0., 100., "GeV"),
          sigma("sigma", "sigma", 0., 100., "GeV"),
          efficiency("efficiency", "efficiency", 0.5, 0.0, 1.0),
          nSignalAll("nSignalAll", "nSignalAll", 0., 1e10),
          nSignalPass("nSignalPass", "nSignalAll*efficiency", RooArgList(nSignalAll, efficiency)),
          nSignalFail("nSignalFail", "nSignalAll*(1-efficiency)", RooArgList(nSignalAll, efficiency)),
          nBackgroundFail("nBackgroundFail", "nBackgroundFail", 0., 1e10),
          nBackgroundPass("nBackgroundPass", "nBackgroundPass", 0., 1e10),
          category("category", "category"),
          simPdf("simPdf", "simPdf", category),
          data(nullptr),
          verbose(verbose_) {
      //turn on/off default messaging of roofit
      RooMsgService::instance().setSilentMode(!verbose ? kTRUE : kFALSE);
      for (int i = 0; i < RooMsgService::instance().numStreams(); i++) {
        RooMsgService::instance().setStreamStatus(i, verbose ? kTRUE : kFALSE);
      }
      category.defineType("pass");
      category.defineType("fail");
    };
    virtual ~AbstractFitter() = default;
    ;
    void setup(double expectedMean_, double massLow, double massHigh, double expectedSigma_) {
      expectedMean = expectedMean_;
      expectedSigma = expectedSigma_;
      mass.setRange(massLow, massHigh);
      mean.setRange(massLow, massHigh);
    }
    virtual void fit(TH1* num, TH1* den) = 0;
    double getEfficiency() { return efficiency.getVal(); }
    double getEfficiencyError() { return efficiency.getError(); }
    double getChi2() { return chi2; }
    void savePlot(const TString& name) {
      using namespace RooFit;
      RooPlot* frame = mass.frame(Name(name), Title("Failing and Passing Probe Distributions"));
      data->plotOn(frame, Cut("category==category::pass"), LineColor(kGreen), MarkerColor(kGreen));
      data->plotOn(frame, Cut("category==category::fail"), LineColor(kRed), MarkerColor(kRed));
      simPdf.plotOn(frame, Slice(category, "pass"), ProjWData(category, *data), LineColor(kGreen));
      simPdf.plotOn(frame, Slice(category, "fail"), ProjWData(category, *data), LineColor(kRed));
      simPdf.paramOn(frame, Layout(0.58, 0.99, 0.99));
      data->statOn(frame, Layout(0.70, 0.99, 0.5));
      frame->Write();
      delete frame;
    }

    TString calculateEfficiency(
        TH3* pass, TH3* all, int massDimension, TProfile2D*& eff, TProfile2D*& effChi2, const TString& plotName = "") {
      //sort out the TAxis
      TAxis *par1Axis, *par2Axis, *massAxis;
      int par1C, par2C, massC;
      if (massDimension == 1) {
        massAxis = all->GetXaxis();
        massC = 1;
        par1Axis = all->GetYaxis();
        par1C = all->GetXaxis()->GetNbins() + 2;
        par2Axis = all->GetZaxis();
        par2C = (all->GetXaxis()->GetNbins() + 2) * (all->GetYaxis()->GetNbins() + 2);
      } else if (massDimension == 2) {
        par1Axis = all->GetXaxis();
        par1C = 1;
        massAxis = all->GetYaxis();
        massC = all->GetXaxis()->GetNbins() + 2;
        par2Axis = all->GetZaxis();
        par2C = (all->GetXaxis()->GetNbins() + 2) * (all->GetYaxis()->GetNbins() + 2);
      } else if (massDimension == 3) {
        par1Axis = all->GetXaxis();
        par1C = 1;
        par2Axis = all->GetYaxis();
        par2C = all->GetXaxis()->GetNbins() + 2;
        massAxis = all->GetZaxis();
        massC = (all->GetXaxis()->GetNbins() + 2) * (all->GetYaxis()->GetNbins() + 2);
      } else {
        return "massDimension > 3 !, skipping...";
      }

      //create eff and effChi2 TProfiles
      if (!par1Axis || !par2Axis)
        return "No par1Axis or par2Axis!";
      if (par1Axis->GetXbins()->GetSize() == 0 && par2Axis->GetXbins()->GetSize() == 0) {
        eff = new TProfile2D("efficiency",
                             "efficiency",
                             par1Axis->GetNbins(),
                             par1Axis->GetXmin(),
                             par1Axis->GetXmax(),
                             par2Axis->GetNbins(),
                             par2Axis->GetXmin(),
                             par2Axis->GetXmax());
      } else if (par1Axis->GetXbins()->GetSize() == 0) {
        eff = new TProfile2D("efficiency",
                             "efficiency",
                             par1Axis->GetNbins(),
                             par1Axis->GetXmin(),
                             par1Axis->GetXmax(),
                             par2Axis->GetNbins(),
                             par2Axis->GetXbins()->GetArray());
      } else if (par2Axis->GetXbins()->GetSize() == 0) {
        eff = new TProfile2D("efficiency",
                             "efficiency",
                             par1Axis->GetNbins(),
                             par1Axis->GetXbins()->GetArray(),
                             par2Axis->GetNbins(),
                             par2Axis->GetXmin(),
                             par2Axis->GetXmax());
      } else {
        eff = new TProfile2D("efficiency",
                             "efficiency",
                             par1Axis->GetNbins(),
                             par1Axis->GetXbins()->GetArray(),
                             par2Axis->GetNbins(),
                             par2Axis->GetXbins()->GetArray());
      }
      eff->SetTitle("");
      eff->SetXTitle(par1Axis->GetTitle());
      eff->SetYTitle(par2Axis->GetTitle());
      eff->SetStats(kFALSE);
      effChi2 = (TProfile2D*)eff->Clone("efficiencyChi2");
      eff->SetZTitle("Efficiency");
      eff->SetOption("colztexte");
      eff->GetZaxis()->SetRangeUser(-0.001, 1.001);
      effChi2->SetZTitle("Chi^2/NDF");
      effChi2->SetOption("colztext");

      //create the 1D mass distribution container histograms
      TH1D* all1D = (massAxis->GetXbins()->GetSize() == 0)
                        ? new TH1D("all1D", "all1D", massAxis->GetNbins(), massAxis->GetXmin(), massAxis->GetXmax())
                        : new TH1D("all1D", "all1D", massAxis->GetNbins(), massAxis->GetXbins()->GetArray());
      auto* pass1D = (TH1D*)all1D->Clone("pass1D");

      //for each parameter bin fit the mass distributions
      for (int par1 = 1; par1 <= par1Axis->GetNbins(); par1++) {
        for (int par2 = 1; par2 <= par2Axis->GetNbins(); par2++) {
          for (int mass = 1; mass <= massAxis->GetNbins(); mass++) {
            int index = par1 * par1C + par2 * par2C + mass * massC;
            all1D->SetBinContent(mass, all->GetBinContent(index));
            pass1D->SetBinContent(mass, pass->GetBinContent(index));
          }
          fit(pass1D, all1D);
          int index = par1 + par2 * (par1Axis->GetNbins() + 2);
          eff->SetBinContent(index, getEfficiency());
          eff->SetBinEntries(index, 1);
          eff->SetBinError(index,
                           sqrt(getEfficiency() * getEfficiency() + getEfficiencyError() * getEfficiencyError()));
          effChi2->SetBinContent(index, getChi2());
          effChi2->SetBinEntries(index, 1);
          if (plotName != "") {
            savePlot(TString::Format("%s_%d_%d", plotName.Data(), par1, par2));
          }
        }
      }
      delete all1D;
      delete pass1D;
      return "";  //OK
    }

    TString calculateEfficiency(
        TH2* pass, TH2* all, int massDimension, TProfile*& eff, TProfile*& effChi2, const TString& plotName = "") {
      //sort out the TAxis
      TAxis *par1Axis, *massAxis;
      int par1C, massC;
      if (massDimension == 1) {
        massAxis = all->GetXaxis();
        massC = 1;
        par1Axis = all->GetYaxis();
        par1C = all->GetXaxis()->GetNbins() + 2;
      } else if (massDimension == 2) {
        par1Axis = all->GetXaxis();
        par1C = 1;
        massAxis = all->GetYaxis();
        massC = all->GetXaxis()->GetNbins() + 2;
      } else {
        return "massDimension > 2 !, skipping...";
      }

      //create eff and effChi2 TProfiles
      if (!par1Axis)
        return "No par1Axis!";
      eff =
          (par1Axis->GetXbins()->GetSize() == 0)
              ? new TProfile("efficiency", "efficiency", par1Axis->GetNbins(), par1Axis->GetXmin(), par1Axis->GetXmax())
              : new TProfile("efficiency", "efficiency", par1Axis->GetNbins(), par1Axis->GetXbins()->GetArray());
      eff->SetTitle("");
      eff->SetXTitle(par1Axis->GetTitle());
      eff->SetLineColor(2);
      eff->SetLineWidth(2);
      eff->SetMarkerStyle(20);
      eff->SetMarkerSize(0.8);
      eff->SetStats(kFALSE);
      effChi2 = (TProfile*)eff->Clone("efficiencyChi2");
      eff->SetYTitle("Efficiency");
      eff->SetOption("PE");
      eff->GetYaxis()->SetRangeUser(-0.001, 1.001);
      effChi2->SetYTitle("Chi^2/NDF");
      effChi2->SetOption("HIST");

      //create the 1D mass distribution container histograms
      TH1D* all1D = (massAxis->GetXbins()->GetSize() == 0)
                        ? new TH1D("all1D", "all1D", massAxis->GetNbins(), massAxis->GetXmin(), massAxis->GetXmax())
                        : new TH1D("all1D", "all1D", massAxis->GetNbins(), massAxis->GetXbins()->GetArray());
      auto* pass1D = (TH1D*)all1D->Clone("pass1D");

      //for each parameter bin fit the mass distributions
      for (int par1 = 1; par1 <= par1Axis->GetNbins(); par1++) {
        for (int mass = 1; mass <= massAxis->GetNbins(); mass++) {
          int index = par1 * par1C + mass * massC;
          all1D->SetBinContent(mass, all->GetBinContent(index));
          pass1D->SetBinContent(mass, pass->GetBinContent(index));
        }
        fit(pass1D, all1D);
        int index = par1;
        eff->SetBinContent(index, getEfficiency());
        eff->SetBinEntries(index, 1);
        eff->SetBinError(index, sqrt(getEfficiency() * getEfficiency() + getEfficiencyError() * getEfficiencyError()));
        effChi2->SetBinContent(index, getChi2());
        effChi2->SetBinEntries(index, 1);
        if (plotName != "") {
          savePlot(TString::Format("%s_%d", plotName.Data(), par1));
        }
      }
      delete all1D;
      delete pass1D;
      return "";  //OK
    }
  };

  //concrete fitter: Gaussian signal plus linear background
  class GaussianPlusLinearFitter : public AbstractFitter {
  protected:
    RooGaussian gaussian;
    RooRealVar slopeFail;
    RooChebychev linearFail;
    RooRealVar slopePass;
    RooChebychev linearPass;
    RooAddPdf pdfFail;
    RooAddPdf pdfPass;

  public:
    GaussianPlusLinearFitter(bool verbose = false)
        : AbstractFitter(verbose),
          gaussian("gaussian", "gaussian", mass, mean, sigma),
          slopeFail("slopeFail", "slopeFail", 0., -1., 1.),
          linearFail("linearFail", "linearFail", mass, slopeFail),
          slopePass("slopePass", "slopePass", 0., -1., 1.),
          linearPass("linearPass", "linearPass", mass, slopePass),
          pdfFail("pdfFail", "pdfFail", RooArgList(gaussian, linearFail), RooArgList(nSignalFail, nBackgroundFail)),
          pdfPass("pdfPass", "pdfPass", RooArgList(gaussian, linearPass), RooArgList(nSignalPass, nBackgroundPass)) {
      simPdf.addPdf(pdfFail, "fail");
      simPdf.addPdf(pdfPass, "pass");
    };
    ~GaussianPlusLinearFitter() override = default;
    ;
    void fit(TH1* pass, TH1* all) override {
      using namespace RooFit;
      all->Add(pass, -1);
      TH1*& fail = all;
      if (!data)
        delete data;
      data = new RooDataHist("data", "data", mass, Index(category), Import("fail", *fail), Import("pass", *pass));
      if (pass->Integral() + fail->Integral() < 5) {
        efficiency.setVal(0.5);
        efficiency.setError(0.5);
        chi2 = 0;
        return;
      }
      mean.setVal(expectedMean);
      sigma.setVal(expectedSigma);
      efficiency.setVal(pass->Integral() / (pass->Integral() + fail->Integral()));
      nSignalAll.setVal(0.5 * (fail->Integral() + pass->Integral()));
      nBackgroundFail.setVal(0.5 * fail->Integral());
      nBackgroundPass.setVal(0.5 * pass->Integral());
      slopeFail.setVal(0.);
      slopePass.setVal(0.);
      if (verbose) {
        simPdf.fitTo(*data);
      } else {
        simPdf.fitTo(*data, Verbose(kFALSE), PrintLevel(-1), Warnings(kFALSE), PrintEvalErrors(-1));
      }
      RooDataHist dataFail("fail", "fail", mass, fail);
      RooDataHist dataPass("pass", "pass", mass, pass);
      using AbsRealPtr = std::unique_ptr<RooAbsReal>;
      const double chi2Fail = AbsRealPtr(pdfFail.createChi2(dataFail, DataError(RooAbsData::Poisson)))->getVal();
      const double chi2Pass = AbsRealPtr(pdfPass.createChi2(dataPass, DataError(RooAbsData::Poisson)))->getVal();
      chi2 = (chi2Fail + chi2Pass) / (2 * pass->GetNbinsX() - 8);
      if (chi2 > 3) {
        efficiency.setVal(0.5);
        efficiency.setError(0.5);
      }
    }
  };

  //concrete fitter: voigtian signal plus exponential background
  class VoigtianPlusExponentialFitter : public AbstractFitter {
  protected:
    RooRealVar width;
    RooVoigtian voigtian;
    RooRealVar slopeFail;
    RooExponential exponentialFail;
    RooRealVar slopePass;
    RooExponential exponentialPass;
    RooAddPdf pdfFail;
    RooAddPdf pdfPass;

  public:
    VoigtianPlusExponentialFitter(bool verbose = false)
        : AbstractFitter(verbose),
          width("width", "width", 2.5, "GeV"),
          voigtian("voigtian", "voigtian", mass, mean, width, sigma),
          slopeFail("slopeFail", "slopeFail", 0., -1., 0.),
          exponentialFail("linearFail", "linearFail", mass, slopeFail),
          slopePass("slopePass", "slopePass", 0., -1., 0.),
          exponentialPass("linearPass", "linearPass", mass, slopePass),
          pdfFail(
              "pdfFail", "pdfFail", RooArgList(voigtian, exponentialFail), RooArgList(nSignalFail, nBackgroundFail)),
          pdfPass(
              "pdfPass", "pdfPass", RooArgList(voigtian, exponentialPass), RooArgList(nSignalPass, nBackgroundPass)) {
      width.setConstant(kTRUE);
      simPdf.addPdf(pdfFail, "fail");
      simPdf.addPdf(pdfPass, "pass");
    };
    ~VoigtianPlusExponentialFitter() override = default;
    ;
    void setup(double expectedMean_, double massLow, double massHigh, double expectedSigma_, double width_) {
      expectedMean = expectedMean_;
      expectedSigma = expectedSigma_;
      mass.setRange(massLow, massHigh);
      mean.setRange(massLow, massHigh);
      width.setVal(width_);
    }
    void fit(TH1* pass, TH1* all) override {
      using namespace RooFit;
      all->Add(pass, -1);
      TH1*& fail = all;
      if (!data)
        delete data;
      data = new RooDataHist("data", "data", mass, Index(category), Import("fail", *fail), Import("pass", *pass));
      if (pass->Integral() + fail->Integral() < 5) {
        efficiency.setVal(0.5);
        efficiency.setError(0.5);
        chi2 = 0;
        return;
      }
      mean.setVal(expectedMean);
      sigma.setVal(expectedSigma);
      efficiency.setVal(pass->Integral() / (pass->Integral() + fail->Integral()));
      nSignalAll.setVal(0.5 * (fail->Integral() + pass->Integral()));
      nBackgroundFail.setVal(0.5 * fail->Integral());
      nBackgroundPass.setVal(0.5 * pass->Integral());
      slopeFail.setVal(0.);
      slopePass.setVal(0.);
      if (verbose) {
        simPdf.fitTo(*data);
      } else {
        simPdf.fitTo(*data, Verbose(kFALSE), PrintLevel(-1), Warnings(kFALSE), PrintEvalErrors(-1));
      }
      RooDataHist dataFail("fail", "fail", mass, fail);
      RooDataHist dataPass("pass", "pass", mass, pass);
      using AbsRealPtr = std::unique_ptr<RooAbsReal>;
      const double chi2Fail = AbsRealPtr(pdfFail.createChi2(dataFail, DataError(RooAbsData::Poisson)))->getVal();
      const double chi2Pass = AbsRealPtr(pdfPass.createChi2(dataPass, DataError(RooAbsData::Poisson)))->getVal();
      chi2 = (chi2Fail + chi2Pass) / (2 * all->GetNbinsX() - 8);
      if (chi2 > 3) {
        efficiency.setVal(0.5);
        efficiency.setError(0.5);
      }
    }
  };

}  //namespace dqmTnP
