#include "DQMServices/DQMGUI/interface/DQMRenderPlugin.h"
#include "utils.h"
#include "TLegend.h"
#include "TH2F.h"
#include "TH3F.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TColor.h"
#include <cassert>
#include "TROOT.h"
#include "TLatex.h"
#include "TLine.h"
#include "TBox.h"
#include "TPaletteAxis.h"
#include "TGaxis.h"
#include "TText.h"

class CASTORRenderPlugin : public DQMRenderPlugin {
public:
  ////---- define the histograms
  virtual bool applies(const VisDQMObject &o, const VisDQMImgInfo &) {
    ////---- determine whether the object is a CASTOR object
    if ((o.name.find("Castor/EventInfo/reportSummaryMap") != std::string::npos) ||
        (o.name.find("Castor/CastorPSMonitor/CASTOR Digi ChannelSummaryMap") != std::string::npos) ||
        (o.name.find("Castor/CastorPSMonitor/CASTOR Digi SaturationSummaryMap") != std::string::npos) ||
        (o.name.find("Castor/CastorChannelQuality/RecHitEnergyBasedSummaryMap") != std::string::npos) ||
        (o.name.find("Castor/CastorPSMonitor/CASTOR Digi Occupancy Map") != std::string::npos) ||
        (o.name.find("Castor/CastorPSMonitor/Castor Pulse Shape for sector=1 (in all 14 modules)") !=
         std::string::npos) ||
        (o.name.find("Castor/CastorPSMonitor/Castor Pulse Shape for sector=2 (in all 14 modules)") !=
         std::string::npos) ||
        (o.name.find("Castor/CastorPSMonitor/Castor Pulse Shape for sector=3 (in all 14 modules)") !=
         std::string::npos) ||
        (o.name.find("Castor/CastorPSMonitor/Castor Pulse Shape for sector=4 (in all 14 modules)") !=
         std::string::npos) ||
        (o.name.find("Castor/CastorPSMonitor/Castor Pulse Shape for sector=5 (in all 14 modules)") !=
         std::string::npos) ||
        (o.name.find("Castor/CastorPSMonitor/Castor Pulse Shape for sector=6 (in all 14 modules)") !=
         std::string::npos) ||
        (o.name.find("Castor/CastorPSMonitor/Castor Pulse Shape for sector=7 (in all 14 modules)") !=
         std::string::npos) ||
        (o.name.find("Castor/CastorPSMonitor/Castor Pulse Shape for sector=8 (in all 14 modules)") !=
         std::string::npos) ||
        (o.name.find("Castor/CastorPSMonitor/Castor Pulse Shape for sector=9 (in all 14 modules)") !=
         std::string::npos) ||
        (o.name.find("Castor/CastorPSMonitor/Castor Pulse Shape for sector=10 (in all 14 modules)") !=
         std::string::npos) ||
        (o.name.find("Castor/CastorPSMonitor/Castor Pulse Shape for sector=11 (in all 14 modules)") !=
         std::string::npos) ||
        (o.name.find("Castor/CastorPSMonitor/Castor Pulse Shape for sector=12 (in all 14 modules)") !=
         std::string::npos) ||
        (o.name.find("Castor/CastorPSMonitor/Castor Pulse Shape for sector=13 (in all 14 modules)") !=
         std::string::npos) ||
        (o.name.find("Castor/CastorPSMonitor/Castor Pulse Shape for sector=14 (in all 14 modules)") !=
         std::string::npos) ||
        (o.name.find("Castor/CastorPSMonitor/Castor Pulse Shape for sector=15 (in all 14 modules)") !=
         std::string::npos) ||
        (o.name.find("Castor/CastorPSMonitor/Castor Pulse Shape for sector=16 (in all 14 modules)") !=
         std::string::npos) ||
        (o.name.find("Castor/CastorRecHitMonitor/CastorRecHit Energy in modules- above threshold") !=
         std::string::npos) ||
        (o.name.find("Castor/CastorRecHitMonitor/CastorRecHit Energy in sectors- above threshold") !=
         std::string::npos) ||
        (o.name.find("Castor/CastorEventDisplay/CASTOR 3D hits- cumulative") != std::string::npos) ||
        (o.name.find("Castor/CastorEventDisplay/CASTOR 3D hits- event with the largest deposited E") !=
         std::string::npos) ||
        (o.name.find("Castor/CastorRecHitMonitor/EnergyFraction/Fraction of the total energy in CASTOR modules") !=
         std::string::npos) ||
        (o.name.find("Castor/CastorRecHitMonitor/EnergyFraction/Fraction of the total energy in CASTOR sectors") !=
         std::string::npos) ||
        (o.name.find("Castor/CastorRecHitMonitor/CastorRecHits Occupancy Map") != std::string::npos) ||
        (o.name.find("Castor/CastorDataIntegrityMonitor/CASTOR spigot status") != std::string::npos) ||
        (o.name.find("Castor/CastorDataIntegrityMonitor/CASTOR FEDFatal errors") != std::string::npos) ||
        (o.name.find("Castor/CastorEventProducts/CastorEventProduct") != std::string::npos))
      return true;

    return false;
  }

  //==========================================================//
  //==================== preDraw ============================//
  //==========================================================//

  virtual void preDraw(TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo &, VisDQMRenderInfo &) {
    c->cd();

    ////---- TH3
    if (dynamic_cast<TH3 *>(o.object)) {
      preDrawTH3(c, o);
    }

    ////---- TH2
    if (dynamic_cast<TH2 *>(o.object)) {
      preDrawTH2(c, o);
    }

    ////---- TH1
    else if (dynamic_cast<TH1 *>(o.object)) {
      preDrawTH1(c, o);
    }
  }

  //==========================================================//
  //==================== postDraw ============================//
  //==========================================================//

  virtual void postDraw(TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo &) {
    c->cd();

    ////--- TH3
    if (dynamic_cast<TH3 *>(o.object)) {
      postDrawTH3(c, o);
    }

    ////--- TH2
    if (dynamic_cast<TH2 *>(o.object)) {
      postDrawTH2(c, o);
    }

    ////--- TH1
    if (dynamic_cast<TH1 *>(o.object)) {
      postDrawTH1(c, o);
    }
  }

private:
  //==========================================================//
  //==================== preDrawTH3 ==========================//
  //==========================================================//

  void preDrawTH3(TCanvas *c, const VisDQMObject &o) {
    TH3F *obj = dynamic_cast<TH3F *>(o.object);
    assert(obj);

    gStyle->SetCanvasBorderMode(0);
    gStyle->SetPadBorderMode(0);
    gStyle->SetPadBorderSize(0);

    obj->GetXaxis()->SetLabelSize(0.04);
    obj->GetYaxis()->SetLabelSize(0.04);
    obj->GetZaxis()->SetLabelSize(0.04);
    obj->GetXaxis()->SetTitle("|Z| [cm]");  //swap z and y axis and x and y
    obj->GetYaxis()->SetTitle("X [cm]");
    obj->GetZaxis()->SetTitle("Y [cm]");
    obj->GetXaxis()->SetTitleOffset(0.12);
    obj->GetYaxis()->SetTitleOffset(0.12);
    obj->GetZaxis()->SetTitleOffset(0.12);

    if (o.name.find("CASTOR 3D hits- cumulative") != std::string::npos) {
      gStyle->SetOptStat(0);
      obj->SetStats(kFALSE);
      c->SetGrid(1);
      return;
    }

    if (o.name.find("CASTOR 3D hits- event with the largest deposited E") != std::string::npos) {
      //gStyle->SetPalette(1);
      obj->SetStats(kFALSE);
      obj->SetOption("LEGO");
      c->SetGrid(1);
      return;
    }
  }

  //==========================================================//
  //==================== postDrawTH3 ==========================//
  //==========================================================//

  void postDrawTH3(TCanvas *, const VisDQMObject &o) {
    if (o.name.find("testOccupancy3D") != std::string::npos) {
      //-- leave it empty for the moment
      return;
    }
  }

  //==========================================================//
  //==================== preDrawTH2 ==========================//
  //==========================================================//

  void preDrawTH2(TCanvas *c, const VisDQMObject &o) {
    TH2F *obj = dynamic_cast<TH2F *>(o.object);
    assert(obj);

    gStyle->SetCanvasBorderMode(0);
    gStyle->SetPadBorderMode(0);
    gStyle->SetPadBorderSize(0);
    gStyle->SetOptStat(0);
    obj->SetStats(kFALSE);
    obj->SetOption("colz");
    obj->GetXaxis()->SetLabelSize(0.05);
    obj->GetYaxis()->SetLabelSize(0.05);

    if (o.name.find("Castor/EventInfo/reportSummaryMap") != std::string::npos) {
      obj->GetXaxis()->SetNdivisions(15, true);
      obj->GetYaxis()->SetNdivisions(17, true);
      obj->GetXaxis()->SetLabelSize(0.07);
      obj->GetYaxis()->SetLabelSize(0.07);

      obj->GetXaxis()->SetBinLabel(1, "1");
      obj->GetXaxis()->SetBinLabel(2, "2");
      obj->GetXaxis()->SetBinLabel(3, "3");
      obj->GetXaxis()->SetBinLabel(4, "4");
      obj->GetXaxis()->SetBinLabel(5, "5");
      obj->GetXaxis()->SetBinLabel(6, "6");
      obj->GetXaxis()->SetBinLabel(7, "7");
      obj->GetXaxis()->SetBinLabel(8, "8");
      obj->GetXaxis()->SetBinLabel(9, "9");
      obj->GetXaxis()->SetBinLabel(10, "10");
      obj->GetXaxis()->SetBinLabel(11, "11");
      obj->GetXaxis()->SetBinLabel(12, "12");
      obj->GetXaxis()->SetBinLabel(13, "13");
      obj->GetXaxis()->SetBinLabel(14, "14");

      obj->GetYaxis()->SetBinLabel(1, "1");
      obj->GetYaxis()->SetBinLabel(2, "2");
      obj->GetYaxis()->SetBinLabel(3, "3");
      obj->GetYaxis()->SetBinLabel(4, "4");
      obj->GetYaxis()->SetBinLabel(5, "5");
      obj->GetYaxis()->SetBinLabel(6, "6");
      obj->GetYaxis()->SetBinLabel(7, "7");
      obj->GetYaxis()->SetBinLabel(8, "8");
      obj->GetYaxis()->SetBinLabel(9, "9");
      obj->GetYaxis()->SetBinLabel(10, "10");
      obj->GetYaxis()->SetBinLabel(11, "11");
      obj->GetYaxis()->SetBinLabel(12, "12");
      obj->GetYaxis()->SetBinLabel(13, "13");
      obj->GetYaxis()->SetBinLabel(14, "14");
      obj->GetYaxis()->SetBinLabel(15, "15");
      obj->GetYaxis()->SetBinLabel(16, "16");

      obj->GetXaxis()->SetTitle("z-module");
      obj->GetYaxis()->SetTitle("#phi-sector");

      obj->GetXaxis()->CenterLabels();
      obj->GetYaxis()->CenterLabels();
      c->SetGrid(1, 1);
      dqm::utils::reportSummaryMapPalette(obj);
      obj->SetMinimum(0.0);
      obj->SetMaximum(+1.0);

      return;
    }

    if (o.name.find("ChannelSummaryMap") != std::string::npos) {
      obj->GetXaxis()->SetNdivisions(15, true);
      obj->GetYaxis()->SetNdivisions(17, true);
      obj->GetXaxis()->SetLabelSize(0.07);
      obj->GetYaxis()->SetLabelSize(0.07);

      obj->GetXaxis()->SetBinLabel(1, "1");
      obj->GetXaxis()->SetBinLabel(2, "2");
      obj->GetXaxis()->SetBinLabel(3, "3");
      obj->GetXaxis()->SetBinLabel(4, "4");
      obj->GetXaxis()->SetBinLabel(5, "5");
      obj->GetXaxis()->SetBinLabel(6, "6");
      obj->GetXaxis()->SetBinLabel(7, "7");
      obj->GetXaxis()->SetBinLabel(8, "8");
      obj->GetXaxis()->SetBinLabel(9, "9");
      obj->GetXaxis()->SetBinLabel(10, "10");
      obj->GetXaxis()->SetBinLabel(11, "11");
      obj->GetXaxis()->SetBinLabel(12, "12");
      obj->GetXaxis()->SetBinLabel(13, "13");
      obj->GetXaxis()->SetBinLabel(14, "14");

      obj->GetYaxis()->SetBinLabel(1, "1");
      obj->GetYaxis()->SetBinLabel(2, "2");
      obj->GetYaxis()->SetBinLabel(3, "3");
      obj->GetYaxis()->SetBinLabel(4, "4");
      obj->GetYaxis()->SetBinLabel(5, "5");
      obj->GetYaxis()->SetBinLabel(6, "6");
      obj->GetYaxis()->SetBinLabel(7, "7");
      obj->GetYaxis()->SetBinLabel(8, "8");
      obj->GetYaxis()->SetBinLabel(9, "9");
      obj->GetYaxis()->SetBinLabel(10, "10");
      obj->GetYaxis()->SetBinLabel(11, "11");
      obj->GetYaxis()->SetBinLabel(12, "12");
      obj->GetYaxis()->SetBinLabel(13, "13");
      obj->GetYaxis()->SetBinLabel(14, "14");
      obj->GetYaxis()->SetBinLabel(15, "15");
      obj->GetYaxis()->SetBinLabel(16, "16");

      obj->GetXaxis()->SetTitle("z-module");
      obj->GetYaxis()->SetTitle("#phi-sector");

      obj->GetXaxis()->CenterLabels();
      obj->GetYaxis()->CenterLabels();
      c->SetGrid(1, 1);
      //dqm::utils::reportSummaryMapPalette(obj);
      obj->SetMinimum(-1.0);
      obj->SetMaximum(+1.0);
      int colorError1[4];
      colorError1[0] = 632;  // kRed
      colorError1[1] = 800;  // kOrange
      colorError1[2] = 432;  // kCyan
      colorError1[3] = 416;  // kGreen
      gStyle->SetPalette(4, colorError1);
      gStyle->SetPaintTextFormat("+g");
      return;
    }

    if (o.name.find("SaturationSummaryMap") != std::string::npos) {
      obj->GetXaxis()->SetNdivisions(15, true);
      obj->GetYaxis()->SetNdivisions(17, true);
      obj->GetXaxis()->SetLabelSize(0.07);
      obj->GetYaxis()->SetLabelSize(0.07);

      obj->GetXaxis()->SetBinLabel(1, "1");
      obj->GetXaxis()->SetBinLabel(2, "2");
      obj->GetXaxis()->SetBinLabel(3, "3");
      obj->GetXaxis()->SetBinLabel(4, "4");
      obj->GetXaxis()->SetBinLabel(5, "5");
      obj->GetXaxis()->SetBinLabel(6, "6");
      obj->GetXaxis()->SetBinLabel(7, "7");
      obj->GetXaxis()->SetBinLabel(8, "8");
      obj->GetXaxis()->SetBinLabel(9, "9");
      obj->GetXaxis()->SetBinLabel(10, "10");
      obj->GetXaxis()->SetBinLabel(11, "11");
      obj->GetXaxis()->SetBinLabel(12, "12");
      obj->GetXaxis()->SetBinLabel(13, "13");
      obj->GetXaxis()->SetBinLabel(14, "14");

      obj->GetYaxis()->SetBinLabel(1, "1");
      obj->GetYaxis()->SetBinLabel(2, "2");
      obj->GetYaxis()->SetBinLabel(3, "3");
      obj->GetYaxis()->SetBinLabel(4, "4");
      obj->GetYaxis()->SetBinLabel(5, "5");
      obj->GetYaxis()->SetBinLabel(6, "6");
      obj->GetYaxis()->SetBinLabel(7, "7");
      obj->GetYaxis()->SetBinLabel(8, "8");
      obj->GetYaxis()->SetBinLabel(9, "9");
      obj->GetYaxis()->SetBinLabel(10, "10");
      obj->GetYaxis()->SetBinLabel(11, "11");
      obj->GetYaxis()->SetBinLabel(12, "12");
      obj->GetYaxis()->SetBinLabel(13, "13");
      obj->GetYaxis()->SetBinLabel(14, "14");
      obj->GetYaxis()->SetBinLabel(15, "15");
      obj->GetYaxis()->SetBinLabel(16, "16");

      obj->GetXaxis()->SetTitle("z-module");
      obj->GetYaxis()->SetTitle("#phi-sector");

      obj->GetXaxis()->CenterLabels();
      obj->GetYaxis()->CenterLabels();
      c->SetGrid(1, 1);
      //dqm::utils::reportSummaryMapPalette(obj);
      obj->SetMinimum(-1.0);
      obj->SetMaximum(+1.0);
      int colorError1[3];
      colorError1[0] = 800;  // kOrange
      colorError1[1] = 400;  // kYellow
      colorError1[2] = 416;  // kGreen
      gStyle->SetPalette(3, colorError1);
      gStyle->SetPaintTextFormat("+g");

      return;
    }

    if (o.name.find("RecHitEnergyBasedSummaryMap") != std::string::npos) {
      obj->GetXaxis()->SetNdivisions(15, true);
      obj->GetYaxis()->SetNdivisions(17, true);
      obj->GetXaxis()->SetLabelSize(0.07);
      obj->GetYaxis()->SetLabelSize(0.07);

      obj->GetXaxis()->SetBinLabel(1, "1");
      obj->GetXaxis()->SetBinLabel(2, "2");
      obj->GetXaxis()->SetBinLabel(3, "3");
      obj->GetXaxis()->SetBinLabel(4, "4");
      obj->GetXaxis()->SetBinLabel(5, "5");
      obj->GetXaxis()->SetBinLabel(6, "6");
      obj->GetXaxis()->SetBinLabel(7, "7");
      obj->GetXaxis()->SetBinLabel(8, "8");
      obj->GetXaxis()->SetBinLabel(9, "9");
      obj->GetXaxis()->SetBinLabel(10, "10");
      obj->GetXaxis()->SetBinLabel(11, "11");
      obj->GetXaxis()->SetBinLabel(12, "12");
      obj->GetXaxis()->SetBinLabel(13, "13");
      obj->GetXaxis()->SetBinLabel(14, "14");

      obj->GetYaxis()->SetBinLabel(1, "1");
      obj->GetYaxis()->SetBinLabel(2, "2");
      obj->GetYaxis()->SetBinLabel(3, "3");
      obj->GetYaxis()->SetBinLabel(4, "4");
      obj->GetYaxis()->SetBinLabel(5, "5");
      obj->GetYaxis()->SetBinLabel(6, "6");
      obj->GetYaxis()->SetBinLabel(7, "7");
      obj->GetYaxis()->SetBinLabel(8, "8");
      obj->GetYaxis()->SetBinLabel(9, "9");
      obj->GetYaxis()->SetBinLabel(10, "10");
      obj->GetYaxis()->SetBinLabel(11, "11");
      obj->GetYaxis()->SetBinLabel(12, "12");
      obj->GetYaxis()->SetBinLabel(13, "13");
      obj->GetYaxis()->SetBinLabel(14, "14");
      obj->GetYaxis()->SetBinLabel(15, "15");
      obj->GetYaxis()->SetBinLabel(16, "16");

      obj->GetXaxis()->SetTitle("z-module");
      obj->GetYaxis()->SetTitle("#phi-sector");

      obj->GetXaxis()->CenterLabels();
      obj->GetYaxis()->CenterLabels();
      c->SetGrid(1, 1);
      //dqm::utils::reportSummaryMapPalette(obj);
      obj->SetMinimum(-1.0);
      obj->SetMaximum(+1.0);
      int colorError1[3];
      colorError1[0] = 632;  // kRed
      colorError1[1] = 400;  // kYellow
      colorError1[2] = 416;  // kGreen
      gStyle->SetPalette(3, colorError1);
      return;
    }

    if (o.name.find("CastorRecHit 2D Energy Map- above threshold") != std::string::npos) {
      obj->GetXaxis()->SetNdivisions(15, true);
      obj->GetYaxis()->SetNdivisions(17, true);
      obj->GetXaxis()->SetLabelSize(0.07);
      obj->GetYaxis()->SetLabelSize(0.07);

      obj->GetXaxis()->SetBinLabel(1, "1");
      obj->GetXaxis()->SetBinLabel(2, "2");
      obj->GetXaxis()->SetBinLabel(3, "3");
      obj->GetXaxis()->SetBinLabel(4, "4");
      obj->GetXaxis()->SetBinLabel(5, "5");
      obj->GetXaxis()->SetBinLabel(6, "6");
      obj->GetXaxis()->SetBinLabel(7, "7");
      obj->GetXaxis()->SetBinLabel(8, "8");
      obj->GetXaxis()->SetBinLabel(9, "9");
      obj->GetXaxis()->SetBinLabel(10, "10");
      obj->GetXaxis()->SetBinLabel(11, "11");
      obj->GetXaxis()->SetBinLabel(12, "12");
      obj->GetXaxis()->SetBinLabel(13, "13");
      obj->GetXaxis()->SetBinLabel(14, "14");

      obj->GetYaxis()->SetBinLabel(1, "1");
      obj->GetYaxis()->SetBinLabel(2, "2");
      obj->GetYaxis()->SetBinLabel(3, "3");
      obj->GetYaxis()->SetBinLabel(4, "4");
      obj->GetYaxis()->SetBinLabel(5, "5");
      obj->GetYaxis()->SetBinLabel(6, "6");
      obj->GetYaxis()->SetBinLabel(7, "7");
      obj->GetYaxis()->SetBinLabel(8, "8");
      obj->GetYaxis()->SetBinLabel(9, "9");
      obj->GetYaxis()->SetBinLabel(10, "10");
      obj->GetYaxis()->SetBinLabel(11, "11");
      obj->GetYaxis()->SetBinLabel(12, "12");
      obj->GetYaxis()->SetBinLabel(13, "13");
      obj->GetYaxis()->SetBinLabel(14, "14");
      obj->GetYaxis()->SetBinLabel(15, "15");
      obj->GetYaxis()->SetBinLabel(16, "16");

      obj->GetXaxis()->SetTitle("z-module");
      obj->GetYaxis()->SetTitle("#phi-sector");

      obj->GetXaxis()->CenterLabels();
      obj->GetYaxis()->CenterLabels();
      gStyle->SetPalette(1);
      c->SetGrid(1, 1);
      return;
    }

    if (o.name.find("CastorRecHits Occupancy Map") != std::string::npos) {
      obj->GetXaxis()->SetNdivisions(15, true);
      obj->GetYaxis()->SetNdivisions(17, true);
      obj->GetXaxis()->SetLabelSize(0.07);
      obj->GetYaxis()->SetLabelSize(0.07);

      obj->GetXaxis()->SetBinLabel(1, "1");
      obj->GetXaxis()->SetBinLabel(2, "2");
      obj->GetXaxis()->SetBinLabel(3, "3");
      obj->GetXaxis()->SetBinLabel(4, "4");
      obj->GetXaxis()->SetBinLabel(5, "5");
      obj->GetXaxis()->SetBinLabel(6, "6");
      obj->GetXaxis()->SetBinLabel(7, "7");
      obj->GetXaxis()->SetBinLabel(8, "8");
      obj->GetXaxis()->SetBinLabel(9, "9");
      obj->GetXaxis()->SetBinLabel(10, "10");
      obj->GetXaxis()->SetBinLabel(11, "11");
      obj->GetXaxis()->SetBinLabel(12, "12");
      obj->GetXaxis()->SetBinLabel(13, "13");
      obj->GetXaxis()->SetBinLabel(14, "14");

      obj->GetYaxis()->SetBinLabel(1, "1");
      obj->GetYaxis()->SetBinLabel(2, "2");
      obj->GetYaxis()->SetBinLabel(3, "3");
      obj->GetYaxis()->SetBinLabel(4, "4");
      obj->GetYaxis()->SetBinLabel(5, "5");
      obj->GetYaxis()->SetBinLabel(6, "6");
      obj->GetYaxis()->SetBinLabel(7, "7");
      obj->GetYaxis()->SetBinLabel(8, "8");
      obj->GetYaxis()->SetBinLabel(9, "9");
      obj->GetYaxis()->SetBinLabel(10, "10");
      obj->GetYaxis()->SetBinLabel(11, "11");
      obj->GetYaxis()->SetBinLabel(12, "12");
      obj->GetYaxis()->SetBinLabel(13, "13");
      obj->GetYaxis()->SetBinLabel(14, "14");
      obj->GetYaxis()->SetBinLabel(15, "15");
      obj->GetYaxis()->SetBinLabel(16, "16");

      obj->GetXaxis()->SetTitle("z-module");
      obj->GetYaxis()->SetTitle("#phi-sector");

      obj->GetXaxis()->CenterLabels();
      obj->GetYaxis()->CenterLabels();
      gStyle->SetPalette(1);
      c->SetGrid(1, 1);
      return;
    }

    if (o.name.find("CASTOR Digi Occupancy Map") != std::string::npos) {
      obj->GetXaxis()->SetNdivisions(15, true);
      obj->GetYaxis()->SetNdivisions(17, true);
      obj->GetXaxis()->SetLabelSize(0.07);
      obj->GetYaxis()->SetLabelSize(0.07);

      obj->GetXaxis()->SetBinLabel(1, "1");
      obj->GetXaxis()->SetBinLabel(2, "2");
      obj->GetXaxis()->SetBinLabel(3, "3");
      obj->GetXaxis()->SetBinLabel(4, "4");
      obj->GetXaxis()->SetBinLabel(5, "5");
      obj->GetXaxis()->SetBinLabel(6, "6");
      obj->GetXaxis()->SetBinLabel(7, "7");
      obj->GetXaxis()->SetBinLabel(8, "8");
      obj->GetXaxis()->SetBinLabel(9, "9");
      obj->GetXaxis()->SetBinLabel(10, "10");
      obj->GetXaxis()->SetBinLabel(11, "11");
      obj->GetXaxis()->SetBinLabel(12, "12");
      obj->GetXaxis()->SetBinLabel(13, "13");
      obj->GetXaxis()->SetBinLabel(14, "14");

      obj->GetYaxis()->SetBinLabel(1, "1");
      obj->GetYaxis()->SetBinLabel(2, "2");
      obj->GetYaxis()->SetBinLabel(3, "3");
      obj->GetYaxis()->SetBinLabel(4, "4");
      obj->GetYaxis()->SetBinLabel(5, "5");
      obj->GetYaxis()->SetBinLabel(6, "6");
      obj->GetYaxis()->SetBinLabel(7, "7");
      obj->GetYaxis()->SetBinLabel(8, "8");
      obj->GetYaxis()->SetBinLabel(9, "9");
      obj->GetYaxis()->SetBinLabel(10, "10");
      obj->GetYaxis()->SetBinLabel(11, "11");
      obj->GetYaxis()->SetBinLabel(12, "12");
      obj->GetYaxis()->SetBinLabel(13, "13");
      obj->GetYaxis()->SetBinLabel(14, "14");
      obj->GetYaxis()->SetBinLabel(15, "15");
      obj->GetYaxis()->SetBinLabel(16, "16");

      obj->GetXaxis()->SetTitle("z-module");
      obj->GetYaxis()->SetTitle("#phi-sector");

      obj->GetXaxis()->CenterLabels();
      obj->GetYaxis()->CenterLabels();
      gStyle->SetPalette(1);
      c->SetGrid(1, 1);
      return;
    }

    if (o.name.find("CASTOR spigot status") != std::string::npos) {
      obj->GetXaxis()->SetNdivisions(15, true);
      obj->GetYaxis()->SetNdivisions(3, true);
      obj->GetXaxis()->SetLabelSize(0.07);
      obj->GetYaxis()->SetLabelSize(0.07);

      obj->GetXaxis()->SetBinLabel(1, "1");
      obj->GetXaxis()->SetBinLabel(2, "2");
      obj->GetXaxis()->SetBinLabel(3, "3");
      obj->GetXaxis()->SetBinLabel(4, "4");
      obj->GetXaxis()->SetBinLabel(5, "5");
      obj->GetXaxis()->SetBinLabel(6, "6");
      obj->GetXaxis()->SetBinLabel(7, "7");
      obj->GetXaxis()->SetBinLabel(8, "8");
      obj->GetXaxis()->SetBinLabel(9, "9");
      obj->GetXaxis()->SetBinLabel(10, "10");
      obj->GetXaxis()->SetBinLabel(11, "11");
      obj->GetXaxis()->SetBinLabel(12, "12");
      obj->GetXaxis()->SetBinLabel(13, "13");
      obj->GetXaxis()->SetBinLabel(14, "14");
      obj->GetXaxis()->SetBinLabel(15, "15");

      obj->GetYaxis()->SetBinLabel(1, "690");
      obj->GetYaxis()->SetBinLabel(2, "691");
      obj->GetYaxis()->SetBinLabel(3, "692");

      obj->GetXaxis()->SetTitle("input socket on CASTOR DCC");
      obj->GetYaxis()->SetTitle("FED ID");

      obj->GetXaxis()->CenterLabels();
      obj->GetYaxis()->CenterLabels();

      obj->SetMinimum(-1.0);
      obj->SetMaximum(+1.0);
      int colorError1[3];
      colorError1[0] = 632;  // kRed
      colorError1[1] = 400;  // kYellow
      colorError1[2] = 416;  // kGreen
      gStyle->SetPalette(3, colorError1);

      c->SetGrid(1, 1);
      return;
    }

    if (o.name.find("CastorEventProduct") != std::string::npos) {
      obj->GetXaxis()->SetNdivisions(5, true);
      obj->GetYaxis()->SetNdivisions(1, true);
      obj->GetXaxis()->SetLabelSize(0.07);
      obj->GetYaxis()->SetLabelSize(0.05);

      obj->GetXaxis()->SetBinLabel(1, "Raw");
      obj->GetXaxis()->SetBinLabel(2, "Digi");
      obj->GetXaxis()->SetBinLabel(3, "RecHits");
      obj->GetXaxis()->SetBinLabel(4, "Towers");
      obj->GetXaxis()->SetBinLabel(5, "BasicJets");
      obj->GetXaxis()->SetBinLabel(6, "JetID");
      obj->GetYaxis()->SetBinLabel(1, "Status");
      obj->GetYaxis()->SetTitle("Present/Not Present");

      obj->SetMinimum(-1.0);
      obj->SetMaximum(+1.0);

      int colorError1[2];
      colorError1[0] = 632;  // kRed
      colorError1[1] = 416;  // kGreen
      gStyle->SetPalette(2, colorError1);

      return;
    }
  }

  //==========================================================//
  //==================== postDrawTH2 ==========================//
  //==========================================================//

  void postDrawTH2(TCanvas *, const VisDQMObject &o) {
    if (o.name.find("ChannelSummaryMap") != std::string::npos) {
      TBox *b_box_g = new TBox();
      TBox *b_box_c = new TBox();
      TBox *b_box_o = new TBox();
      TBox *b_box_r = new TBox();

      b_box_g->SetFillColor(416);
      b_box_c->SetFillColor(432);
      b_box_o->SetFillColor(800);
      b_box_r->SetFillColor(632);

      TLegend *leg = new TLegend(0.1, 0.75, 0.9, 0.9);  //(0.16, 0.11, 0.44, 0.38);
      leg->AddEntry(b_box_g, "Good (signal)", "f");
      leg->AddEntry(b_box_c, "Good (pedestal)", "f");
      leg->AddEntry(b_box_o, "Saturated Channel (in >5% of events)", "f");
      leg->AddEntry(b_box_r, "Dead Channel", "f");
      leg->Draw();

      // TLine* my_line = new TLine();
      // my_line->DrawLine(0,18.07,14,18.07);
    }

    if (o.name.find("SaturationSummaryMap") != std::string::npos) {
      TBox *b_box_g = new TBox();
      TBox *b_box_y = new TBox();
      TBox *b_box_o = new TBox();

      b_box_g->SetFillColor(416);
      b_box_y->SetFillColor(400);
      b_box_o->SetFillColor(800);

      TLegend *leg = new TLegend(0.1, 0.75, 0.9, 0.9);  //(0.16, 0.11, 0.44, 0.38);
      leg->AddEntry(b_box_g, "No saturation", "f");
      leg->AddEntry(b_box_y, "Saturation in <5% of events", "f");
      leg->AddEntry(b_box_o, "Saturation in >5% of events", "f");
      leg->Draw();
    }

    if (o.name.find("CASTOR spigot status") != std::string::npos) {
      TBox *b_box_g = new TBox();
      TBox *b_box_y = new TBox();
      TBox *b_box_r = new TBox();

      b_box_g->SetFillColor(416);
      b_box_y->SetFillColor(400);
      b_box_r->SetFillColor(632);

      TLegend *leg = new TLegend(0.1, 0.70, 0.9, 0.9);  //(0.16, 0.11, 0.44, 0.38);
      leg->AddEntry(b_box_g, "OK", "f");
      leg->AddEntry(b_box_y, "Problematic in <5% of events", "f");
      leg->AddEntry(b_box_r, "Problematic in >5% of events", "f");
      leg->Draw();
    }

    if (o.name.find("testOccupancy") != std::string::npos) {
      TH2F *histo = dynamic_cast<TH2F *>(o.object);
      int nBinsX = histo->GetNbinsX();
      for (int i = 0; i != 12; ++i) {
        if (histo->GetBinContent(nBinsX + 1, i + 1) == -1) {
          TLine *lineLow = new TLine(1, i, nBinsX, i);
          lineLow->SetLineColor(kRed);
          TLine *lineHigh = new TLine(1, i + 1, nBinsX, i + 1);
          lineHigh->SetLineColor(kRed);
          lineLow->Draw("same");
          lineHigh->Draw("same");
        }
      }
      return;
    }
  }

  //==========================================================//
  //==================== preDrawTH1 ==========================//
  //==========================================================//

  void preDrawTH1(TCanvas *c, const VisDQMObject &o) {
    TH1 *obj = dynamic_cast<TH1 *>(o.object);
    assert(obj);

    // This applies to all
    //gStyle->SetCanvasBorderMode(0);
    //gStyle->SetPadBorderMode(0);
    //gStyle->SetPadBorderSize(0);
    //obj->SetStats(kFALSE);
    //gStyle->SetLabelSize(0.7);
    //obj->GetXaxis()->SetLabelSize(0.05);
    //obj->GetYaxis()->SetLabelSize(0.05);

    if (o.name.find("Castor Pulse Shape for sector=1 (in all 14 modules)") != std::string::npos ||
        o.name.find("Castor Pulse Shape for sector=2 (in all 14 modules)") != std::string::npos ||
        o.name.find("Castor Pulse Shape for sector=3 (in all 14 modules)") != std::string::npos ||
        o.name.find("Castor Pulse Shape for sector=4 (in all 14 modules)") != std::string::npos ||
        o.name.find("Castor Pulse Shape for sector=5 (in all 14 modules)") != std::string::npos ||
        o.name.find("Castor Pulse Shape for sector=6 (in all 14 modules)") != std::string::npos ||
        o.name.find("Castor Pulse Shape for sector=7 (in all 14 modules)") != std::string::npos ||
        o.name.find("Castor Pulse Shape for sector=8 (in all 14 modules)") != std::string::npos ||
        o.name.find("Castor Pulse Shape for sector=9 (in all 14 modules)") != std::string::npos ||
        o.name.find("Castor Pulse Shape for sector=10 (in all 14 modules)") != std::string::npos ||
        o.name.find("Castor Pulse Shape for sector=11 (in all 14 modules)") != std::string::npos ||
        o.name.find("Castor Pulse Shape for sector=12 (in all 14 modules)") != std::string::npos ||
        o.name.find("Castor Pulse Shape for sector=13 (in all 14 modules)") != std::string::npos ||
        o.name.find("Castor Pulse Shape for sector=14 (in all 14 modules)") != std::string::npos ||
        o.name.find("Castor Pulse Shape for sector=15 (in all 14 modules)") != std::string::npos ||
        o.name.find("Castor Pulse Shape for sector=16 (in all 14 modules)") != std::string::npos) {
      obj->SetStats(kFALSE);
      obj->GetXaxis()->SetLabelSize(0.04);
      obj->GetXaxis()->SetNdivisions(1020);
      c->SetGridx(1);
      return;
    }

    if (o.name.find("CastorRecHit Energy in modules- above threshold") != std::string::npos) {
      obj->GetXaxis()->SetBinLabel(1, "1");
      obj->GetXaxis()->SetBinLabel(2, "2");
      obj->GetXaxis()->SetBinLabel(3, "3");
      obj->GetXaxis()->SetBinLabel(4, "4");
      obj->GetXaxis()->SetBinLabel(5, "5");
      obj->GetXaxis()->SetBinLabel(6, "6");
      obj->GetXaxis()->SetBinLabel(7, "7");
      obj->GetXaxis()->SetBinLabel(8, "8");
      obj->GetXaxis()->SetBinLabel(9, "9");
      obj->GetXaxis()->SetBinLabel(10, "10");
      obj->GetXaxis()->SetBinLabel(11, "11");
      obj->GetXaxis()->SetBinLabel(12, "12");
      obj->GetXaxis()->SetBinLabel(13, "13");
      obj->GetXaxis()->SetBinLabel(14, "14");
      obj->GetXaxis()->SetTitle("z-module");

      obj->GetXaxis()->SetLabelSize(0.07);
      obj->GetXaxis()->SetNdivisions(1020);
      obj->GetXaxis()->CenterLabels();
      obj->GetYaxis()->CenterLabels();

      return;
    }

    if (o.name.find("Fraction of the total energy in CASTOR modules") != std::string::npos) {
      obj->GetXaxis()->SetBinLabel(1, "1");
      obj->GetXaxis()->SetBinLabel(2, "2");
      obj->GetXaxis()->SetBinLabel(3, "3");
      obj->GetXaxis()->SetBinLabel(4, "4");
      obj->GetXaxis()->SetBinLabel(5, "5");
      obj->GetXaxis()->SetBinLabel(6, "6");
      obj->GetXaxis()->SetBinLabel(7, "7");
      obj->GetXaxis()->SetBinLabel(8, "8");
      obj->GetXaxis()->SetBinLabel(9, "9");
      obj->GetXaxis()->SetBinLabel(10, "10");
      obj->GetXaxis()->SetBinLabel(11, "11");
      obj->GetXaxis()->SetBinLabel(12, "12");
      obj->GetXaxis()->SetBinLabel(13, "13");
      obj->GetXaxis()->SetBinLabel(14, "14");
      obj->GetXaxis()->SetTitle("z-module");

      obj->GetXaxis()->SetLabelSize(0.07);
      obj->GetXaxis()->SetNdivisions(1020);
      obj->GetXaxis()->CenterLabels();
      obj->GetYaxis()->CenterLabels();
      obj->GetYaxis()->SetTitle("fraction of the total energy");
      obj->GetYaxis()->SetRange(0, 1);
      obj->SetStats(kFALSE);

      return;
    }

    if (o.name.find("CastorRecHit Energy in sectors- above threshold") != std::string::npos) {
      obj->GetXaxis()->SetBinLabel(1, "1");
      obj->GetXaxis()->SetBinLabel(2, "2");
      obj->GetXaxis()->SetBinLabel(3, "3");
      obj->GetXaxis()->SetBinLabel(4, "4");
      obj->GetXaxis()->SetBinLabel(5, "5");
      obj->GetXaxis()->SetBinLabel(6, "6");
      obj->GetXaxis()->SetBinLabel(7, "7");
      obj->GetXaxis()->SetBinLabel(8, "8");
      obj->GetXaxis()->SetBinLabel(9, "9");
      obj->GetXaxis()->SetBinLabel(10, "10");
      obj->GetXaxis()->SetBinLabel(11, "11");
      obj->GetXaxis()->SetBinLabel(12, "12");
      obj->GetXaxis()->SetBinLabel(13, "13");
      obj->GetXaxis()->SetBinLabel(14, "14");
      obj->GetXaxis()->SetBinLabel(15, "15");
      obj->GetXaxis()->SetBinLabel(16, "16");
      obj->GetXaxis()->SetTitle("#phi-sector");

      obj->GetXaxis()->SetLabelSize(0.07);
      obj->GetXaxis()->SetNdivisions(1020);
      obj->GetXaxis()->CenterLabels();
      obj->GetYaxis()->CenterLabels();
      return;
    }

    if (o.name.find("Fraction of the total energy in CASTOR sectors") != std::string::npos) {
      obj->GetXaxis()->SetBinLabel(1, "1");
      obj->GetXaxis()->SetBinLabel(2, "2");
      obj->GetXaxis()->SetBinLabel(3, "3");
      obj->GetXaxis()->SetBinLabel(4, "4");
      obj->GetXaxis()->SetBinLabel(5, "5");
      obj->GetXaxis()->SetBinLabel(6, "6");
      obj->GetXaxis()->SetBinLabel(7, "7");
      obj->GetXaxis()->SetBinLabel(8, "8");
      obj->GetXaxis()->SetBinLabel(9, "9");
      obj->GetXaxis()->SetBinLabel(10, "10");
      obj->GetXaxis()->SetBinLabel(11, "11");
      obj->GetXaxis()->SetBinLabel(12, "12");
      obj->GetXaxis()->SetBinLabel(13, "13");
      obj->GetXaxis()->SetBinLabel(14, "14");
      obj->GetXaxis()->SetBinLabel(15, "15");
      obj->GetXaxis()->SetBinLabel(16, "16");
      obj->GetXaxis()->SetTitle("#phi-sector");

      obj->GetXaxis()->SetLabelSize(0.07);
      obj->GetXaxis()->SetNdivisions(1020);
      obj->GetXaxis()->CenterLabels();
      obj->GetYaxis()->CenterLabels();
      obj->GetYaxis()->SetTitle("fraction of the total energy");
      obj->GetYaxis()->SetRange(0, 1);
      obj->SetStats(kFALSE);
      return;
    }

    if (o.name.find("CASTOR FEDFatal errors") != std::string::npos) {
      obj->GetXaxis()->SetBinLabel(1, "690");
      obj->GetXaxis()->SetBinLabel(2, "691");
      obj->GetXaxis()->SetBinLabel(3, "692");
      obj->GetXaxis()->SetTitle("FED ID");

      obj->GetXaxis()->SetLabelSize(0.07);
      obj->GetXaxis()->CenterLabels();

      return;
    }
  }
  //==========================================================//
  //==================== postDrawTH1 ==========================//
  //==========================================================//

  void postDrawTH1(TCanvas *, const VisDQMObject &o) {
    TH1 *obj = dynamic_cast<TH1 *>(o.object);
    assert(obj);

    if (o.name.find("CASTOR FEDFatal errors") != std::string::npos) {
      TText t;

      if (obj->GetEntries() == 0) {
        t.SetTextColor(3);
        t.SetTextSize(0.07);
        t.DrawText(690.3, 0.5, "OK: No Fatal Errors Found");
      }

      else {
        t.SetTextColor(2);
        t.SetTextSize(0.07);
        t.DrawText(690.3, 1.0, "ATTENTION: Fatal Errors Found");
      }

      gPad->SetGridx();
      return;
    }

    if (o.name.find("CastorEventProduct") != std::string::npos) {
      obj->SetMarkerSize(3);         // set font size to 3x normal
      obj->SetOption("text90colz");  // draw marker at 90 degrees

      gPad->SetGridx();
      return;
    }
  }
};

static CASTORRenderPlugin instance;
