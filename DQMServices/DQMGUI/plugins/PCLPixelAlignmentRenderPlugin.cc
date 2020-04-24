#include "DQMServices/DQMGUI/interface/DQMRenderPlugin.h"
#include "utils.h"
#include "TCanvas.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TObjString.h"
#include "TProfile2D.h"
#include "TStyle.h"
#include "TGaxis.h"
#include "TPaveStats.h"
#include "TList.h"
#include "TLine.h"
#include <cassert>

class PCLPixelAlignmentRenderPlugin : public DQMRenderPlugin {
  std::array<double, 6> sigCut_;
  std::array<double, 6> cut_;
  std::array<double, 6> maxMoveCut_;
  std::array<double, 6> maxErrorCut_;

public:
  void initialise(int, char **) override {}

  bool applies(const VisDQMObject &o, const VisDQMImgInfo &) override {
    if (o.name.find("SiPixelAli/") != std::string::npos)
      return true;
    else
      return false;
  }

  void preDraw(TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo &, VisDQMRenderInfo &) override {
    c->cd();

    if (dynamic_cast<TH1F *>(o.object)) {
      this->preDrawTH1F(c, o);
    } else if (dynamic_cast<TH2F *>(o.object)) {
      this->preDrawTH2F(c, o);
    } else if (dynamic_cast<TProfile *>(o.object)) {
      this->preDrawTProfile(c, o);
    } else if (dynamic_cast<TProfile2D *>(o.object)) {
      this->preDrawTProfile2D(c, o);
    }
  }

  void postDraw(TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo &) override {
    c->cd();

    if (o.name.find("SiPixelAli") != std::string::npos && o.name.find("PedeExitCode") != std::string::npos) {
      // Clear the default string from the canvas
      c->Clear();

      TObjString *tos = dynamic_cast<TObjString *>(o.object);
      auto exitCode = TString((tos->GetString())(0, 6));
      TText *t = new TText(.5, .5, tos->GetString());
      t->SetTextFont(63);
      t->SetTextAlign(22);
      t->SetTextSize(18);
      // from Pede manual: http://www.desy.de/~kleinwrt/MP2/doc/html/exit_code_page.html
      // all exit codes <  10 are normal endings
      // all exit codes >= 10 indicated an aborted measurement
      if (exitCode.Atoi() >= 10) {
        t->SetTextColor(kRed);
      } else {
        t->SetTextColor(kGreen + 2);
      }
      t->Draw();
    }

    if (dynamic_cast<TH1F *>(o.object)) {
      this->postDrawTH1F(c, o);
    } else if (dynamic_cast<TH2F *>(o.object)) {
      this->postDrawTH2F(c, o);
    } else if (dynamic_cast<TProfile *>(o.object)) {
      this->postDrawTProfile(c, o);
    } else if (dynamic_cast<TProfile2D *>(o.object)) {
      this->postDrawTProfile2D(c, o);
    }
  }

private:
  void preDrawTH1F(TCanvas *c, const VisDQMObject &o) {
    gStyle->SetCanvasBorderMode(0);
    gStyle->SetPadBorderMode(0);
    gStyle->SetPadBorderSize(0);

    gStyle->SetOptStat(111110);
    gStyle->SetTitleSize(0.06, "");
    gStyle->SetTitleX(0.18);

    TH1F *obj = dynamic_cast<TH1F *>(o.object);
    assert(obj);

    c->SetLogy(0);
    c->SetTopMargin(0.08);
    c->SetBottomMargin(0.14);
    c->SetLeftMargin(0.11);
    c->SetRightMargin(0.09);

    // dirty trick to ensure compatibility
    // with old histogram format

    const int &nBins = obj->GetNbinsX();
    if (nBins > 11) {
      for (size_t i = 0; i < 6; i++) {
        cut_[i] = obj->GetBinContent(8 + 5 * i);
        sigCut_[i] = obj->GetBinContent(9 + 5 * i);
        maxMoveCut_[i] = obj->GetBinContent(10 + 5 * i);
        maxErrorCut_[i] = obj->GetBinContent(11 + 5 * i);
      }
    } else {
      for (size_t i = 0; i < 6; i++) {
        cut_[i] = obj->GetBinContent(8);
        sigCut_[i] = obj->GetBinContent(9);
        maxMoveCut_[i] = obj->GetBinContent(10);
        maxErrorCut_[i] = obj->GetBinContent(11);
      }
    }

    obj->SetLineColor(kBlack);
    obj->SetLineWidth(2);
    obj->SetFillColor(kGreen + 3);

    double max = -1000;
    double min = 1000;

    for (int i = 1; i < 7; i++) {
      if (obj->GetBinContent(i) < min)
        min = obj->GetBinContent(i);
      if (obj->GetBinContent(i) > max)
        max = obj->GetBinContent(i);

      if (fabs(obj->GetBinContent(i)) > maxMoveCut_[i - 1])
        obj->SetFillColor(kRed);
      else if (obj->GetBinContent(i) > cut_[i - 1]) {
        if (obj->GetBinError(i) > maxErrorCut_[i - 1]) {
          obj->SetFillColor(kRed);
        } else if (fabs(obj->GetBinContent(i)) / obj->GetBinError(i) > sigCut_[i - 1]) {
          obj->SetFillColor(kGreen + 3);
        }
      }
    }

    obj->GetXaxis()->SetBinLabel(1, "FPIX(x+,z-)");
    obj->GetXaxis()->SetBinLabel(2, "FPIX(x-,z-)");
    obj->GetXaxis()->SetBinLabel(3, "BPIX(x+)");
    obj->GetXaxis()->SetBinLabel(4, "BPIX(x-)");
    obj->GetXaxis()->SetBinLabel(5, "FPIX(x+,z+)");
    obj->GetXaxis()->SetBinLabel(6, "FPIX(x-,z+)");

    obj->GetXaxis()->SetTitleSize(0.06);
    obj->GetXaxis()->SetLabelSize(0.06);
    obj->GetXaxis()->SetTitleOffset(1.05);
    obj->GetYaxis()->SetTitleOffset(0.95);
    obj->GetXaxis()->SetRangeUser(0, 6);

    obj->GetYaxis()->SetTitleSize(0.06);
    obj->GetYaxis()->SetLabelSize(0.06);

    double maxCut_ = *std::max_element(cut_.begin(), cut_.end());

    obj->GetYaxis()->SetRangeUser(std::min(min - 20, -maxCut_ - 20), std::max(max + 20, maxCut_ + 20));

    obj->SetStats(kFALSE);
    obj->SetFillStyle(3017);
    obj->SetOption("histe");
  }

  void preDrawTH2F(TCanvas *, const VisDQMObject &o) {
    TH2F *obj = dynamic_cast<TH2F *>(o.object);
    assert(obj);

    if (o.name.find("SiPixelAli") != std::string::npos && o.name.find("statusResults") != std::string::npos) {
      gPad->SetGrid();
      dqm::utils::reportSummaryMapPalette(obj);
      obj->SetMarkerSize(1.5);
      obj->SetOption("colztext");
      return;
    }
  }

  void preDrawTProfile(TCanvas *, const VisDQMObject &o) {
    TProfile *obj = dynamic_cast<TProfile *>(o.object);
    assert(obj);
  }

  void preDrawTProfile2D(TCanvas *, const VisDQMObject &o) {
    TProfile2D *obj = dynamic_cast<TProfile2D *>(o.object);
    assert(obj);
  }

  void postDrawTH1F(TCanvas *, const VisDQMObject &o) {
    TH1F *obj = dynamic_cast<TH1F *>(o.object);
    assert(obj);

    gStyle->SetOptStat(111110);

    obj->SetStats(kFALSE);

    // dirty trick to ensure compatibility
    // with old histogram format

    const int &nBins = obj->GetNbinsX();
    if (nBins > 11) {
      for (size_t i = 0; i < 6; i++) {
        cut_[i] = obj->GetBinContent(8 + 5 * i);
        sigCut_[i] = obj->GetBinContent(9 + 5 * i);
        maxMoveCut_[i] = obj->GetBinContent(10 + 5 * i);
        maxErrorCut_[i] = obj->GetBinContent(11 + 5 * i);
      }
    } else {
      for (size_t i = 0; i < 6; i++) {
        cut_[i] = obj->GetBinContent(8);
        sigCut_[i] = obj->GetBinContent(9);
        maxMoveCut_[i] = obj->GetBinContent(10);
        maxErrorCut_[i] = obj->GetBinContent(11);
      }
    }

    TLine *line = new TLine();
    line->SetBit(kCanDelete);
    line->SetLineWidth(2);
    line->SetLineColor(kRed + 2);

    for (size_t i = 0; i < 6; i++) {
      line->DrawLine(i, cut_[i], i + 1, cut_[i]);
      line->DrawLine(i, -cut_[i], i + 1, -cut_[i]);
    }

    line->SetLineColor(kBlue + 2);
    line->DrawLine(0, 0, 6, 0);

    TText *t_text = new TText();
    t_text->SetBit(kCanDelete);
    t_text->SetNDC(true);

    bool hitMax = false;
    bool moved = false;
    bool hitMaxError = false;
    bool sigMove = false;

    for (int i = 1; i < 7; i++) {
      if (fabs(obj->GetBinContent(i)) > maxMoveCut_[i - 1]) {
        hitMax = true;
      } else if (fabs(obj->GetBinContent(i)) > cut_[i - 1]) {
        moved = true;
        if (obj->GetBinError(i) > maxErrorCut_[i - 1]) {
          hitMaxError = true;
        } else if (fabs(obj->GetBinContent(i)) / obj->GetBinError(i) > sigCut_[i - 1]) {
          sigMove = true;
        }
      }
    }
    if (hitMax) {
      obj->SetFillColor(kRed);
      t_text->DrawText(0.25, 0.8, "Exceeds Maximum Movement");
    } else if (moved) {
      if (hitMaxError) {
        obj->SetFillColor(kRed);
        t_text->DrawText(0.25, 0.8, "Movement uncertainty exceeds maximum");
      } else if (sigMove) {
        obj->SetFillColor(kOrange);
        t_text->DrawText(0.25, 0.8, "Significant movement observed");
      }
    } else
      t_text->DrawText(0.25, 0.8, "Movement within limits");

    return;
  }

  void postDrawTH2F(TCanvas *, const VisDQMObject &o) {
    TH2F *obj = dynamic_cast<TH2F *>(o.object);
    assert(obj);

    obj->SetStats(kFALSE);
  }

  void postDrawTProfile(TCanvas *, const VisDQMObject &o) {
    TProfile *obj = dynamic_cast<TProfile *>(o.object);
    assert(obj);
  }

  void postDrawTProfile2D(TCanvas *, const VisDQMObject &o) {
    TProfile2D *obj = dynamic_cast<TProfile2D *>(o.object);
    assert(obj);
  }
};

static PCLPixelAlignmentRenderPlugin instance;
