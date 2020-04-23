#include "DQMServices/DQMGUI/interface/DQMRenderPlugin.h"
#include "utils.h"
#include "TProfile2D.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TColor.h"
#include <iostream>
#include <cassert>
#include "TLine.h"
#include "TText.h"
#include "TPaletteAxis.h"
#include "TH2F.h"
#include "TList.h"

class RPCRenderPlugin : public DQMRenderPlugin {
public:
  bool applies(const VisDQMObject &o, const VisDQMImgInfo &) override {
    if (o.name.find("RPC/RecHits") != std::string::npos)
      return true;
    if (o.name.find("RPC/Noise") != std::string::npos)
      return true;
    if (o.name.find("RPC/Muon") != std::string::npos)
      return true;
    if (o.name.find("RPC/DCSInfo") != std::string::npos)
      return true;
    if (o.name.find("RPC/RPCEfficiency") != std::string::npos)
      return true;
    if (o.name.find("RPC/FEDIntegrity") != std::string::npos)
      return true;
    if (o.name.find("RPC/EventInfo") != std::string::npos)
      return true;
    if (o.name.find("RPC/AllHits") != std::string::npos)
      return true;
    return false;
  }

  void preDraw(TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo &, VisDQMRenderInfo &) override {
    c->cd();

    if (dynamic_cast<TH2 *>(o.object)) {
      preDrawTH2(c, o);
    } else if (dynamic_cast<TH1 *>(o.object)) {
      preDrawTH1(c, o);
    }
  }

  void postDraw(TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo &) override {
    if (dynamic_cast<TH2 *>(o.object))
      postDrawTH2(c, o);
  }

private:
  void preDrawTH1(TCanvas *c __attribute__((unused)), const VisDQMObject &o) {
    TH1 *obj = dynamic_cast<TH1 *>(o.object);
    assert(obj);

    if (o.name.find("BX") != std::string::npos) {
      obj->StatOverflows(false);
    }
  }

  void preDrawTH2(TCanvas *c, const VisDQMObject &o) {
    TH2F *obj = dynamic_cast<TH2F *>(o.object);
    assert(obj);

    // This applies to all
    gStyle->SetCanvasBorderMode(0);
    gStyle->SetCanvasColor(kWhite);
    gStyle->SetPadBorderMode(0);
    gStyle->SetPadBorderSize(0);
    gStyle->SetOptStat(10);
    gStyle->SetPalette(1);
    if (o.name.find("Occupancy_for_") != std::string::npos && o.name.find("SummaryHistograms") != std::string::npos) {
      obj->SetOption("colztext");
    } else {
      obj->SetOption("colz");
    }
    obj->SetStats(kFALSE);

    obj->GetXaxis()->SetNdivisions(-510);
    obj->GetYaxis()->SetNdivisions(-510);
    obj->GetXaxis()->CenterLabels();
    obj->GetYaxis()->CenterLabels();
    c->SetGridx();
    c->SetGridy();

    if (o.name.find("SummaryMap") != std::string::npos) {
      dqm::utils::reportSummaryMapPalette(obj);
      return;
    }

    if (o.name.find("Occupancy") != std::string::npos) {
      if ((o.name.find("_for_Barrel") != std::string::npos || o.name.find("_for_Endcap") != std::string::npos) &&
          o.name.find("SummaryHistograms") != std::string::npos) {
        obj->SetStats(kFALSE);
      } else {
        obj->SetStats(kTRUE);
      }
      return;
    }

    if (o.name.find("AfterPulse") != std::string::npos) {
      obj->SetStats(kTRUE);
      obj->SetOption("SCAT");
      return;
    }
  }

  void postDrawTH2(TCanvas *c, const VisDQMObject &o) {
    TH2 *obj = dynamic_cast<TH2 *>(o.object);
    assert(obj);

    if ((o.name.find("Roll_vs_Sector_Wheel") != std::string::npos && o.name.find("Occupancy") == std::string::npos) ||
        o.name.find("VStatus_Wheel") != std::string::npos) {
      TLine line;
      line.SetLineWidth(2);
      line.DrawLine(0.5, 17.5, 3.5, 17.5);
      line.DrawLine(3.5, 17.5, 3.5, 21.5);
      line.DrawLine(4.5, 21.5, 4.5, 17.5);
      line.DrawLine(4.5, 17.5, 8.5, 17.5);
      line.DrawLine(8.5, 17.5, 8.5, 15.5);
      line.DrawLine(8.5, 15.5, 9.5, 15.5);
      line.DrawLine(9.5, 15.5, 9.5, 17.5);
      line.DrawLine(9.5, 17.5, 10.5, 17.5);
      line.DrawLine(10.5, 17.5, 10.5, 15.5);
      line.DrawLine(10.5, 15.5, 11.5, 15.5);
      line.DrawLine(11.5, 15.5, 11.5, 17.5);
      line.DrawLine(11.5, 17.5, 12.5, 17.5);

      for (int x = 1; x < 13; x++) {
        for (int y = 18; y < 22; y++) {
          if (x != 4)
            obj->SetBinContent(x, y, -1);
        }
      }

      obj->SetBinContent(9, 16, -1);
      obj->SetBinContent(9, 17, -1);
      obj->SetBinContent(11, 16, -1);
      obj->SetBinContent(11, 17, -1);
    }

    if (o.name.find("SummaryMap") != std::string::npos) {  //report summary map

      TLine line;  // draw lines to delimitate Barrel and Endcaps
      line.SetLineWidth(1);
      line.DrawLine(-3.5, 0.5, -3.5, 6.5);
      line.DrawLine(-7.5, 6.5, -3.5, 6.5);
      line.DrawLine(-2.5, 0.5, -2.5, 12.5);
      line.DrawLine(2.5, 0.5, 2.5, 12.5);
      line.DrawLine(-2.5, 12.5, 2.5, 12.5);

      line.DrawLine(3.5, 0.5, 3.5, 6.5);
      line.DrawLine(3.5, 6.5, 7.5, 6.5);
      line.DrawLine(7.5, 0.5, 7.5, 6.5);
      return;
    }

    if (o.name.find("Occupancy") != std::string::npos && o.name.find("SummaryBySectors") != std::string::npos &&
        o.name.find("Wheel") != std::string::npos) {
      //sector occupancy plots
      TLine line;
      line.SetLineWidth(2);  // Draw lines to delimitate the end of the roll
      //rb1in
      line.DrawLine(91, 0.5, 91, 2.5);
      line.DrawLine(91, 2.5, 85, 2.5);
      //rb1out
      line.DrawLine(85, 2.5, 85, 4.5);
      line.DrawLine(85, 4.5, 91, 4.5);
      //rb2in and rb2out
      if (o.name.find("Wheel_-2") != std::string::npos || o.name.find("Wheel_2") != std::string::npos) {
        line.DrawLine(91, 4.5, 91, 6.5);
        line.DrawLine(91, 6.5, 85, 6.5);
        line.DrawLine(85, 6.5, 85, 9.5);
      } else {
        line.DrawLine(91, 4.5, 91, 7.5);
        line.DrawLine(91, 7.5, 85, 7.5);
        line.DrawLine(85, 7.5, 85, 9.5);
      }
      line.DrawLine(85, 9.5, 43, 9.5);
      //rb3
      line.DrawLine(43, 9.5, 43, 13.5);

      //  gPad->Update();
      //         TPaletteAxis *palette;
      //         palette = (TPaletteAxis*)obj->GetListOfFunctions()->FindObject("palette");
      //         palette->GetAxis()->SetLabelSize(0.024);
      //         gPad->Update();
    }

    if (o.name.find("OccupancyNormByGeoAndRPCEvents") != std::string::npos) {
      obj->SetMaximum(0.2);
      return;
    }

    if (o.name.find("AfterPulse") != std::string::npos) {  //afterpulse 2D plots
      obj->GetXaxis()->LabelsOption("v");
      obj->GetXaxis()->SetLabelSize(0.03);
      obj->GetXaxis()->SetLabelOffset(0.005);
      //obj->GetXaxis()->SetNdivisions(-510);
      obj->GetYaxis()->SetLabelSize(0.03);
      obj->GetYaxis()->SetLabelOffset(0.005);
      // obj->GetYaxis()->SetNdivisions(-510);
      return;
    }

    if (o.name.find("ClusterSizeIn1Bin") != std::string::npos) {
      obj->SetMinimum(-1.e-15);
      obj->SetMaximum(1.0);

      int colorPalette[20];
      for (int i = 0; i < 15; i++) {
        colorPalette[i] = 416;
      }
      colorPalette[15] = 400;  // Yallow
      colorPalette[16] = 807;  // Orange
      colorPalette[17] = 632;  // Red
      colorPalette[18] = 632;
      colorPalette[19] = 632;

      gStyle->SetPalette(20, colorPalette);
      return;
    }

    if (o.name.find("ClusterSizeMean") != std::string::npos) {
      obj->SetMinimum(0.0);
      obj->SetMaximum(5.0);

      int colorPalette_m[5];

      colorPalette_m[0] = 400;
      colorPalette_m[1] = 416;  // Yallow
      colorPalette_m[2] = 416;  // Orange
      colorPalette_m[3] = 807;  // Red
      colorPalette_m[4] = 632;

      gStyle->SetPalette(5, colorPalette_m);
      return;
    }

    if (o.name.find("AsymmetryLeftRight") != std::string::npos) {
      obj->SetMinimum(-1.e-15);
      obj->SetMaximum(1.0);

      int colorPalette1[10];

      colorPalette1[0] = 416;
      colorPalette1[1] = 416;
      colorPalette1[2] = 416;

      colorPalette1[3] = 400;  // Yallow
      colorPalette1[4] = 400;  // Yallow

      colorPalette1[5] = 807;  // Orange
      colorPalette1[6] = 807;  // Orange

      colorPalette1[7] = 632;  // Red
      colorPalette1[8] = 632;
      colorPalette1[9] = 632;

      gStyle->SetPalette(10, colorPalette1);
      return;
    }

    if (o.name.find("DeadChannelFraction") != std::string::npos) {
      obj->SetMinimum(-1.e-15);
      obj->SetMaximum(1.0);

      int colorPalette2[10];

      colorPalette2[0] = 416;
      colorPalette2[1] = 416;
      colorPalette2[2] = 416;

      colorPalette2[3] = 400;  // Yallow
      colorPalette2[4] = 400;  // Yallow
      colorPalette2[5] = 400;  // Yallow

      colorPalette2[6] = 807;  // Orange
      colorPalette2[7] = 807;  // Orange

      colorPalette2[8] = 632;  // Red
      colorPalette2[9] = 632;

      gStyle->SetPalette(10, colorPalette2);
      return;
    }

    if (o.name.find("RPC_System_Quality_Overview") != std::string::npos) {
      gStyle->SetPaintTextFormat(".2f");

      obj->GetXaxis()->SetTitle("Fraction of RPC States");
      obj->SetOption("text");
      obj->SetStats(kTRUE);
      return;
    }

    if (o.name.find("RPCChamberQuality") != std::string::npos) {
      obj->SetMinimum(0.5);
      obj->SetMaximum(7.5);

      int colorPalette3[7];

      colorPalette3[0] = 416;  // Gren OK
      colorPalette3[1] = 860;  // Blue OFF
      colorPalette3[2] = 400;  // Yallow Noisily Strip
      colorPalette3[3] = 807;  // Orange Noisily Chamber
      colorPalette3[4] = 616;  // Pink Partly Dead
      colorPalette3[5] = 632;  // red Dead
      colorPalette3[6] = 432;  // Bad Shape

      gStyle->SetPalette(7, colorPalette3);

      //  c->cd();
      //         gPad->Update();

      //         TPaletteAxis *palette;
      //         palette = (TPaletteAxis*)obj->GetListOfFunctions()->FindObject("palette");
      //         palette->GetAxis()->SetLabelSize(0);

      // 	palette->GetAxis()->SetTitle("OK        OFF        Nois.St       Nois.Ch   Part.Dead       Dead    Bad.Shape");

      //         palette->SetTitleOffset(0.25);
      //         palette->SetTitleSize(0.025);
      return;
    }

    if (o.name.find("VStatus_Wheel") != std::string::npos) {
      obj->SetMinimum(-0.5);
      obj->SetMaximum(2.5);

      int colorPalette4[3];

      colorPalette4[1] = 416;  // Blue OFF
      colorPalette4[0] = 860;  // Green ON
      colorPalette4[2] = 400;  // Yallow Error
      gStyle->SetPalette(3, colorPalette4);

      c->cd();
      gPad->Update();

      TPaletteAxis *palette;
      palette = (TPaletteAxis *)obj->GetListOfFunctions()->FindObject("palette");
      palette->GetAxis()->SetLabelSize(0);
      palette->GetAxis()->SetTitle(
          "OFF                                     ON                                 Error                    ");
      palette->SetTitleOffset(0.3);
      palette->SetTitleSize(0.025);

      return;
    }

    if (o.name.find("RPCNoisyStrips") != std::string::npos) {
      obj->SetMinimum(-0.5);
      obj->SetMaximum(5.5);

      int colorPalette5[6];
      colorPalette5[0] = 416;  // G
      colorPalette5[1] = 400;  // Y
      colorPalette5[2] = 400;  // Y
      colorPalette5[3] = 807;  // O
      colorPalette5[4] = 632;  // R
      colorPalette5[5] = 632;  // R

      gStyle->SetPalette(6, colorPalette5);

      return;
    }

    if (o.name.find("NumberOfDigi_Mean") != std::string::npos) {
      obj->SetMinimum(0.5);
      obj->SetMaximum(10.5);

      int colorPalette5[5];

      colorPalette5[0] = 416;  // G
      colorPalette5[1] = 416;  // G
      colorPalette5[2] = 400;  // Y
      colorPalette5[3] = 807;  // O
      colorPalette5[4] = 632;  // R

      gStyle->SetPalette(5, colorPalette5);
      return;
    }

    //for Offline DQM
    if (o.name.find("Efficiency_Roll") != std::string::npos) {
      //obj->Reset();
      obj->SetMinimum(0.0);
      obj->SetMaximum(100.0);

      int colorPaletteEff[10];
      colorPaletteEff[0] = 632;  // R
      colorPaletteEff[1] = 632;  // R
      colorPaletteEff[2] = 632;  // R
      colorPaletteEff[3] = 632;  // R
      colorPaletteEff[4] = 632;  // R
      colorPaletteEff[5] = 632;  // R
      colorPaletteEff[6] = 632;  // R
      colorPaletteEff[7] = 807;  // O
      colorPaletteEff[8] = 400;  // Y
      colorPaletteEff[9] = 416;  // G

      gStyle->SetPalette(10, colorPaletteEff);
      return;
    }

    if (o.name.find("rpcHVStatus") != std::string::npos) {
      //obj->Reset();
      obj->SetMinimum(-0.5);
      obj->SetMaximum(1.5);

      int colorPaletteDCS[2];
      colorPaletteDCS[0] = 632;  // Red
      colorPaletteDCS[1] = 416;  // Green

      gStyle->SetPalette(2, colorPaletteDCS);
      return;
    }

    if (o.name.find("RollPercentage") != std::string::npos) {
      obj->SetOption("text");
      return;
    }
  }
};

static RPCRenderPlugin instance;
