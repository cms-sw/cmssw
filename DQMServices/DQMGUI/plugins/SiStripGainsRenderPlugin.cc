/*!
  \file SiStripGainsRenderPlugin
  \brief Display Plugin for SiStrip Particle Gain DQM Histograms
  \author A. Di Mattia
  \version $Revision: 1.0 $
  \date $Date: 2017/06/01 10:51:56 $
*/

#include "DQMServices/DQMGUI/interface/DQMRenderPlugin.h"
#include "utils.h"

#include "TProfile2D.h"
#include "TProfile.h"
#include "TPaveStats.h"
#include "TH1F.h"
#include "TH2D.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TText.h"
#include "TLine.h"
#include "TMath.h"
#include "TLegend.h"
#include "TString.h"
#include "TLatex.h"
#include <cassert>

class SiStripGainsRenderPlugin : public DQMRenderPlugin {
public:
  virtual bool applies(const VisDQMObject &o, const VisDQMImgInfo &) {
    if (o.name.find("SiStripGains") != std::string::npos)
      return true;
    return false;
  }

  virtual void preDraw(TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo &, VisDQMRenderInfo &) {
    c->cd();

    // This applies to all
    gStyle->SetCanvasBorderMode(0);
    gStyle->SetPadBorderMode(0);
    gStyle->SetPadBorderSize(0);
    gStyle->SetLegendBorderSize(0);
    gStyle->SetOptTitle(0);

    if (dynamic_cast<TH2D *>(o.object) && (o.name.find("Error") != std::string::npos)) {
      preDrawTH2DError(c, o);
    } else if (dynamic_cast<TH2D *>(o.object)) {
      preDrawTH2D(c, o);
    } else if (dynamic_cast<TH2S *>(o.object)) {
      preDrawTH2S(c, o);
    } else if (dynamic_cast<TH1D *>(o.object)) {
      preDrawTH1D(c, o);
    }
  }

  virtual void postDraw(TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo &) {
    c->cd();

    if (dynamic_cast<TH2D *>(o.object) && (o.name.find("Error") != std::string::npos)) {
      postDrawTH2DError(c, o);
    } else if (dynamic_cast<TH2D *>(o.object)) {
      postDrawTH2D(c, o);
    } else if (dynamic_cast<TH2S *>(o.object)) {
      postDrawTH2S(c, o);
    } else if (dynamic_cast<TH1D *>(o.object)) {
      postDrawTH1D(c, o);
    }
  }

private:
  void preDrawTH2DError(TCanvas *, const VisDQMObject &o) {
    TH2D *obj = dynamic_cast<TH2D *>(o.object);
    assert(obj);

    const UInt_t nRGBs = 5;
    UInt_t NCont = 255;

    Double_t stops[nRGBs] = {0.00, 0.34, 0.61, 0.84, 1.00};
    Double_t red[nRGBs] = {0.00, 0.00, 0.87, 1.00, 0.51};
    Double_t green[nRGBs] = {0.00, 0.81, 1.00, 0.20, 0.00};
    Double_t blue[nRGBs] = {0.51, 1.00, 0.12, 0.00, 0.00};

    TColor::CreateGradientColorTable(nRGBs, stops, red, green, blue, NCont);
    gStyle->SetNumberContours(NCont);

    obj->SetStats(kFALSE);
    obj->SetOption("colz");
  }

  void preDrawTH2S(TCanvas *, const VisDQMObject &o) {
    TH2S *obj = dynamic_cast<TH2S *>(o.object);
    assert(obj);

    const UInt_t nRGBs = 5;
    UInt_t NCont = 255;

    Double_t stops[nRGBs] = {0.00, 0.34, 0.61, 0.84, 1.00};
    Double_t red[nRGBs] = {0.00, 0.00, 0.87, 1.00, 0.51};
    Double_t green[nRGBs] = {0.00, 0.81, 1.00, 0.20, 0.00};
    Double_t blue[nRGBs] = {0.51, 1.00, 0.12, 0.00, 0.00};

    TColor::CreateGradientColorTable(nRGBs, stops, red, green, blue, NCont);
    gStyle->SetNumberContours(NCont);

    obj->SetOption("colz");
  }

  void preDrawTH2D(TCanvas *, const VisDQMObject &o) {
    TH2D *obj = dynamic_cast<TH2D *>(o.object);
    assert(obj);

    obj->SetMarkerColor(4);
    if (o.name.find("NoMPV") == std::string::npos)
      obj->SetStats(kFALSE);

    if (o.name.find("TIB") != std::string::npos) {
      obj->SetLineColor(kRed - 7);
      obj->SetMarkerColor(kRed - 7);
      obj->SetMarkerStyle(22);
      obj->SetMarkerSize(.4);
    } else if (o.name.find("TOB") != std::string::npos) {
      obj->SetLineColor(kMagenta - 8);
      obj->SetMarkerColor(kMagenta - 8);
      obj->SetMarkerStyle(23);
      obj->SetMarkerSize(.4);
    } else if (o.name.find("TID") != std::string::npos) {
      obj->SetLineColor(kOrange - 8);
      obj->SetMarkerColor(kOrange - 8);
      obj->SetMarkerStyle(21);
      obj->SetMarkerSize(.4);
    } else if (o.name.find("TEC") != std::string::npos) {
      obj->SetLineColor(kGreen - 8);
      obj->SetMarkerColor(kGreen - 8);
      obj->SetMarkerStyle(20);
      obj->SetMarkerSize(.4);
    }
  }

  void preDrawTH1D(TCanvas *, const VisDQMObject &o) {
    TH1D *obj = dynamic_cast<TH1D *>(o.object);
    assert(obj);

    obj->SetLineColor(4);

    if (o.name.find("TIB") != std::string::npos) {
      obj->SetLineColor(kRed - 7);
      obj->SetMarkerColor(kRed - 7);
      obj->SetMarkerStyle(22);
    } else if (o.name.find("TOB") != std::string::npos) {
      obj->SetLineColor(kMagenta - 8);
      obj->SetMarkerColor(kMagenta - 8);
      obj->SetMarkerStyle(23);
    } else if (o.name.find("TID") != std::string::npos) {
      obj->SetLineColor(kOrange - 8);
      obj->SetMarkerColor(kOrange - 8);
      obj->SetMarkerStyle(21);
    } else if (o.name.find("TEC") != std::string::npos) {
      obj->SetLineColor(kGreen - 8);
      obj->SetMarkerColor(kGreen - 8);
      obj->SetMarkerStyle(20);
    }
  }

  void preDrawTH1F(TCanvas *, const VisDQMObject &o) {
    TH1F *obj = dynamic_cast<TH1F *>(o.object);
    assert(obj);
  }

  void preDrawTProfile2D(TCanvas *, const VisDQMObject &o) {
    TProfile2D *obj = dynamic_cast<TProfile2D *>(o.object);
    assert(obj);
  }
  void preDrawTProfile(TCanvas *, const VisDQMObject &o) {
    TProfile *obj = dynamic_cast<TProfile *>(o.object);
    assert(obj);
  }

  void postDrawTH1F(TCanvas *, const VisDQMObject &o) {
    TText tt;
    tt.SetTextSize(0.12);
    if (o.flags == 0)
      return;
    else {
      if (o.flags & DQM_PROP_REPORT_ERROR) {
        tt.SetTextColor(2);
        tt.DrawTextNDC(0.5, 0.5, "Error");
      } else if (o.flags & DQM_PROP_REPORT_WARN) {
        tt.SetTextColor(5);
        tt.DrawTextNDC(0.5, 0.5, "Warning");
      } else if (o.flags & DQM_PROP_REPORT_OTHER) {
        tt.SetTextColor(1);
        tt.DrawTextNDC(0.5, 0.5, "Other ");
      }
    }
  }

  void postDrawTH1D(TCanvas *, const VisDQMObject &o) {
    TH1D *obj = dynamic_cast<TH1D *>(o.object);
    assert(obj);

    TAxis *xa = obj->GetXaxis();
    TAxis *ya = obj->GetYaxis();

    xa->SetTitleOffset(0.9);
    xa->SetTitleSize(0.05);
    xa->SetLabelSize(0.04);

    ya->SetTitleOffset(0.9);
    ya->SetTitleSize(0.05);
    ya->SetLabelSize(0.04);

    TLine tl;
    tl.SetLineColor(7);
    tl.SetLineWidth(3);
    tl.SetLineStyle(7);

    TLegend *legend = 0;
    if (o.name.find("MPVError") != std::string::npos) {
      obj->SetLineColor(4);
      xa->SetTitle("Error on MPV [ADC/mm]");
      ya->SetTitle("Number of APVs");
      gPad->SetLogy(1);

      int bmin = xa->FindBin(3.3);
      int integral = obj->Integral(bmin, xa->GetNbins() + 1);
      double percentage = 100. * (integral / obj->GetEntries());

      TText tt;
      tt.SetTextSize(0.045);
      tt.SetTextColor(4);

      TString ts = TString::Format("APVs with error > 1.3%%: %d, (%5.2f%%)", integral, percentage);

      tt.DrawTextNDC(0.28, 0.5, ts.Data());
    } else if (o.name.find("MPV") != std::string::npos) {
      xa->SetTitle("MPV [ADC/mm]");
      ya->SetTitle("Number of Clusters");
      //legend = new TLegend(.34,.77,.48,.86);
      legend = new TLegend(.57, .77, .74, .86);
      legend->SetBit(kCanDelete);
    } else if (o.name.find("WRTP") != std::string::npos) {
      xa->SetTitle("New Gain / Previous Gain");
      ya->SetTitle("Number of APVs");
      legend = new TLegend(.52, .77, .65, .86);
      legend->SetBit(kCanDelete);
    } else if (o.name.find("/Gains") != std::string::npos) {
      xa->SetTitle("SiStrip Gain values from fit");
      ya->SetTitle("Number of APVs");
      legend = new TLegend(.52, .77, .65, .86);
      legend->SetBit(kCanDelete);
    } else {
      tl.DrawLine(300., 0., 300., obj->GetMaximum());
      xa->SetTitle("Cluster Charge [ADC/mm]");
      ya->SetTitle("Number of Clusters");
      legend = new TLegend(.52, .77, .65, .86);
      legend->SetBit(kCanDelete);
    }

    gPad->Update();

    TPaveStats *st = (TPaveStats *)obj->GetListOfFunctions()->FindObject("stats");
    if (st != 0) {
      st->SetBorderSize(0);
      st->SetOptStat(1110);
      st->SetTextColor(obj->GetLineColor());
      if (o.name.find("MPV") != std::string::npos) {
        st->SetX1NDC(.12);
        st->SetX2NDC(.35);
        st->SetY1NDC(.73);
        st->SetY2NDC(.89);
      } else {
        st->SetX1NDC(.66);
        st->SetX2NDC(.89);
        st->SetY1NDC(.73);
        st->SetY2NDC(.89);
      }
    }

    if (legend != 0) {
      if (o.name.find("TIB") != std::string::npos)
        legend->AddEntry(obj, "TIB", "l");
      else if (o.name.find("TOB") != std::string::npos)
        legend->AddEntry(obj, "TOB", "l");
      else if (o.name.find("TID") != std::string::npos) {
        if (o.name.find("TIDM") != std::string::npos || o.name.find("TIDmi") != std::string::npos)
          legend->AddEntry(obj, "TID-", "l");
        else if (o.name.find("TIDP") != std::string::npos || o.name.find("TIDpl") != std::string::npos)
          legend->AddEntry(obj, "TID+", "l");
        else
          legend->AddEntry(obj, "TID", "l");
      } else if (o.name.find("TEC") != std::string::npos) {
        if (o.name.find("TECP1") != std::string::npos)
          legend->AddEntry(obj, "TEC+, thin", "l");
        else if (o.name.find("TECP2") != std::string::npos)
          legend->AddEntry(obj, "TEC+, thick", "l");
        else if (o.name.find("TECM1") != std::string::npos)
          legend->AddEntry(obj, "TEC-, thin", "l");
        else if (o.name.find("TECM2") != std::string::npos)
          legend->AddEntry(obj, "TEC-, thick", "l");
        else if (o.name.find("TECP") != std::string::npos)
          legend->AddEntry(obj, "TEC+, thick", "l");
        else if (o.name.find("TECM") != std::string::npos)
          legend->AddEntry(obj, "TEC-, thin", "l");
        else if (o.name.find("TECmi") != std::string::npos)
          legend->AddEntry(obj, "TEC-", "l");
        else if (o.name.find("TECpl") != std::string::npos)
          legend->AddEntry(obj, "TEC+", "l");
        else
          legend->AddEntry(obj, "TEC", "l");
      }
      legend->Draw();
    }

    TText tt;
    tt.SetTextSize(0.12);
    if (o.flags == 0)
      return;
    else {
      if (o.flags & DQM_PROP_REPORT_ERROR) {
        tt.SetTextColor(2);
        tt.DrawTextNDC(0.5, 0.5, "Error");
      } else if (o.flags & DQM_PROP_REPORT_WARN) {
        tt.SetTextColor(5);
        tt.DrawTextNDC(0.5, 0.5, "Warning");
      } else if (o.flags & DQM_PROP_REPORT_OTHER) {
        tt.SetTextColor(1);
        tt.DrawTextNDC(0.5, 0.5, "Other ");
      }
    }
  }

  void postDrawTH2DError(TCanvas *, const VisDQMObject &o) {
    TH2D *obj = dynamic_cast<TH2D *>(o.object);
    assert(obj);

    TAxis *xa = obj->GetXaxis();
    TAxis *ya = obj->GetYaxis();

    xa->SetTitleOffset(0.9);
    xa->SetTitleSize(0.05);
    xa->SetLabelSize(0.04);

    ya->SetTitleOffset(0.7);
    ya->SetTitleSize(0.05);
    ya->SetLabelSize(0.04);
    ya->SetTitle("Error on MPV [ADC/mm]");

    if (o.name.find("VsEta") != std::string::npos) {
      xa->SetTitle("module #eta");
    } else if (o.name.find("VsPhi") != std::string::npos) {
      xa->SetTitle("module #phi");
    } else if (o.name.find("VsN") != std::string::npos) {
      xa->SetTitle("Number of Entries");
    } else if (o.name.find("VsMPV") != std::string::npos) {
      xa->SetTitle("MPV [ADC/mm]");
    }

    std::string name = o.name.substr(o.name.rfind("/") + 1);
    /*
      TLine tl1;
      tl1.SetLineColor(2);
      tl1.SetLineWidth(3);
      float mask1_xmin = 26.5;
      float mask1_xmax = 29.5;
      float mask1_ymin = 166.5;
      float mask1_ymax = 236.5;

      float mask2_xmin = 37.5;
      float mask2_xmax = 39.5;
      float mask2_ymin = 387.5;
      float mask2_ymax = 458.5;

      TLine tl2;
      tl2.SetLineColor(921); // 15?
      tl2.SetLineWidth(2);
      tl2.SetLineStyle(7);
*/
    TText tt;
    tt.SetTextSize(0.12);

    if (o.flags != 0) {
      if (o.flags & DQM_PROP_REPORT_ERROR) {
        tt.SetTextColor(2);
        tt.DrawTextNDC(0.5, 0.5, "Error");
      } else if (o.flags & DQM_PROP_REPORT_WARN) {
        tt.SetTextColor(5);
        tt.DrawTextNDC(0.5, 0.5, "Warning");
      } else if (o.flags & DQM_PROP_REPORT_OTHER) {
        tt.SetTextColor(1);
        tt.DrawTextNDC(0.5, 0.5, "Other ");
      }
    }
  }

  void postDrawTH2S(TCanvas *, const VisDQMObject &o) {
    TH2S *obj = dynamic_cast<TH2S *>(o.object);
    assert(obj);

    TAxis *xa = obj->GetXaxis();
    TAxis *ya = obj->GetYaxis();

    xa->SetTitleOffset(0.9);
    xa->SetTitleSize(0.05);
    xa->SetLabelSize(0.04);

    ya->SetTitleOffset(0.9);
    ya->SetTitleSize(0.05);
    ya->SetLabelSize(0.04);

    if (o.name.find("Pathlength") != std::string::npos) {
      ya->SetTitle("Cluster Charge [ADC]");
      xa->SetTitle("track path [mm]");
    } else {
      ya->SetTitle("Cluster Charge [ADC/mm]");
      xa->SetTitle("module index");
    }

    TText tt;
    tt.SetTextSize(0.12);

    if (o.flags != 0) {
      if (o.flags & DQM_PROP_REPORT_ERROR) {
        tt.SetTextColor(2);
        tt.DrawTextNDC(0.5, 0.5, "Error");
      } else if (o.flags & DQM_PROP_REPORT_WARN) {
        tt.SetTextColor(5);
        tt.DrawTextNDC(0.5, 0.5, "Warning");
      } else if (o.flags & DQM_PROP_REPORT_OTHER) {
        tt.SetTextColor(1);
        tt.DrawTextNDC(0.5, 0.5, "Other ");
      }
    }
  }

  void postDrawTH2D(TCanvas *, const VisDQMObject &o) {
    TH2D *obj = dynamic_cast<TH2D *>(o.object);
    assert(obj);

    TAxis *xa = obj->GetXaxis();
    TAxis *ya = obj->GetYaxis();

    xa->SetTitleOffset(0.9);
    xa->SetTitleSize(0.05);
    xa->SetLabelSize(0.04);

    ya->SetTitleOffset(0.9);
    ya->SetTitleSize(0.05);
    ya->SetLabelSize(0.04);

    TLine tl;
    tl.SetLineColor(7);
    tl.SetLineWidth(3);
    tl.SetLineStyle(7);

    if (o.name.find("GainVs") != std::string::npos) {
      tl.DrawLineNDC(0.1, 0.1, 0.9, 0.9);
      xa->SetTitle("Previous Gain");
      ya->SetTitle("New Gain");
    } else if (o.name.find("MPVvs") != std::string::npos) {
      tl.DrawLineNDC(0.1, 0.5, 0.9, 0.5);
      ya->SetTitle("MPV [ADC/mm]");
      if (o.name.find("Eta") != std::string::npos)
        xa->SetTitle("module #eta");
      else if (o.name.find("Phi") != std::string::npos)
        xa->SetTitle("module #phi");
    } else if (o.name.find("NoMPV") != std::string::npos) {
      xa->SetTitle("Z [cm]");
      ya->SetTitle("R [cm]");

      gPad->Update();

      TPaveStats *st = (TPaveStats *)obj->GetListOfFunctions()->FindObject("stats");
      if (st != 0) {
        st->SetBorderSize(0);
        st->SetOptStat(10);
        st->SetTextColor(4);
        st->SetX1NDC(.69);
        st->SetX2NDC(.89);
        st->SetY1NDC(.81);
        st->SetY2NDC(.89);
      }

      TLatex myt;
      myt.SetTextFont(42);
      myt.SetTextSize(0.06);
      myt.SetTextColor(kBlue);
      myt.DrawLatexNDC(0.15, 0.20, o.name.find("fit") != std::string::npos ? "NO FIT" : "MASKED");
      myt.Draw();
    }

    TText tt;
    tt.SetTextSize(0.12);

    if (o.flags != 0) {
      if (o.flags & DQM_PROP_REPORT_ERROR) {
        tt.SetTextColor(2);
        tt.DrawTextNDC(0.5, 0.5, "Error");
      } else if (o.flags & DQM_PROP_REPORT_WARN) {
        tt.SetTextColor(5);
        tt.DrawTextNDC(0.5, 0.5, "Warning");
      } else if (o.flags & DQM_PROP_REPORT_OTHER) {
        tt.SetTextColor(1);
        tt.DrawTextNDC(0.5, 0.5, "Other ");
      }
    }
  }

  void postDrawTProfile(TCanvas *, const VisDQMObject &o) {
    TProfile *obj = dynamic_cast<TProfile *>(o.object);
    assert(obj);
  }
};

static SiStripGainsRenderPlugin instance;
