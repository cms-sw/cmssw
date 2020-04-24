/**
 * \class L1TEMURenderPlugin
 *
 *
 * Description: render plugin for L1 emulator DQM histograms.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Lorenzo Agostino
 *      Initial version - based on code from HcalRenderPlugin
 *
 * \author: Vasile Mihai Ghete   - HEPHY Vienna
 *      New render plugin for report summary map
 *
 *
 * $Date: 2012/06/13 11:06:06 $
 * $Revision: 1.13 $
 *
 */

// system include files
#include <cassert>

// user include files

//    base class
#include "DQMServices/DQMGUI/interface/DQMRenderPlugin.h"

#include "TProfile2D.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TText.h"
#include "TBox.h"
#include "TLine.h"
#include "TLegend.h"
#include "TPRegexp.h"

#include "QualityTestStatusRenderPlugin.h"

class L1TEMURenderPlugin : public DQMRenderPlugin {
public:
  // determine whether core object is an L1TEMU object
  bool applies(const VisDQMObject& dqmObj, const VisDQMImgInfo&) override {
    if (dqmObj.name.find("L1TEMU/") != std::string::npos) {
      // return true for all L1TEMU, except L1TdeRCT, L1TStage2 and reportSummaryMap
      if (dqmObj.name.find("L1TEMU/L1TdeRCT/") != std::string::npos ||
          dqmObj.name.find("L1TStage2") != std::string::npos ||
          dqmObj.name.find("reportSummaryMap") != std::string::npos) {
        return false;
      } else {
        return true;
      }
    }

    return false;
  }

  // pre-draw, separated per histogram type
  void preDraw(TCanvas* canvas, const VisDQMObject& dqmObj, const VisDQMImgInfo&, VisDQMRenderInfo&) override {
    canvas->cd();

    if (dynamic_cast<TH2F*>(dqmObj.object)) {
      // object is TH2 histogram
      preDrawTH2F(canvas, dqmObj);

    } else if (dynamic_cast<TH1F*>(dqmObj.object)) {
      // object is TH1 histogram
      preDrawTH1F(canvas, dqmObj);
    }
  }

  // post-draw, separated per histogram type
  void postDraw(TCanvas* canvas, const VisDQMObject& dqmObj, const VisDQMImgInfo&) override {
    if (dynamic_cast<TH2F*>(dqmObj.object)) {
      // object is TH2 histogram
      postDrawTH2F(canvas, dqmObj);

    } else if (dynamic_cast<TH1F*>(dqmObj.object)) {
      // object is TH1 histogram
      postDrawTH1F(canvas, dqmObj);
    }
  }

private:
  void preDrawTH1F(TCanvas*, const VisDQMObject& dqmObj) {
    TH1F* objTH = dynamic_cast<TH1F*>(dqmObj.object);

    // checks that object indeed exists
    assert(objTH);

    // no other rendering changes
  }

  void preDrawTH2F(TCanvas*, const VisDQMObject& dqmObj) {
    TH2F* obj = dynamic_cast<TH2F*>(dqmObj.object);

    // checks that object indeed exists
    assert(obj);

    // specific rendering of L1TEMU reportSummaryMap, using
    // dqm::QualityTestStatusRenderPlugin::reportSummaryMapPalette(obj)

    if (dqmObj.name.find("reportSummaryMap") != std::string::npos) {
      obj->SetStats(kFALSE);
      dqm::QualityTestStatusRenderPlugin::reportSummaryMapPalette(obj);

      obj->GetXaxis()->SetLabelSize(0.1);

      obj->GetXaxis()->CenterLabels();
      obj->GetYaxis()->CenterLabels();

      return;
    }

    // pre-draw rendering of other L1TEMU TH2F histograms

    gStyle->SetCanvasBorderMode(0);
    gStyle->SetPadBorderMode(0);
    gStyle->SetPadBorderSize(0);

    //gStyle->SetOptStat( 0 );
    //obj->SetStats( kFALSE );

    // label format
    TAxis* xa = obj->GetXaxis();
    TAxis* ya = obj->GetYaxis();

    xa->SetTitleOffset(0.7);
    xa->SetTitleSize(0.05);
    xa->SetLabelSize(0.04);

    ya->SetTitleOffset(0.7);
    ya->SetTitleSize(0.05);
    ya->SetLabelSize(0.04);

    // set 2D histogram drawing option to "colz"
    //  "COL"
    //    A box is drawn for each cell with a color scale varying with contents.
    //    All the none empty bins are painted.
    //    Empty bins are not painted unless some bins have a negative content because
    //    in that case the null bins might be not empty.
    //  "COLZ" = "COL" +
    //    The color palette is also drawn.

    gStyle->SetPalette(1);
    obj->SetOption("colz");
  }

  void postDrawTH1F(TCanvas*, const VisDQMObject&) {
    // use DQM default rendering
  }

  void postDrawTH2F(TCanvas*, const VisDQMObject& dqmObj) {
    TH2F* obj = dynamic_cast<TH2F*>(dqmObj.object);

    // checks that object indeed exists
    assert(obj);

    if (dqmObj.name.find("reportSummaryMap") != std::string::npos) {
      TLine* l_line = new TLine();
      TText* t_text = new TText();

      t_text->DrawText(2.25, 14.3, "Mu");
      t_text->DrawText(2.25, 13.3, "NoIsoEG");
      t_text->DrawText(2.25, 12.3, "IsoEG");
      t_text->DrawText(2.25, 11.3, "CenJet");
      t_text->DrawText(2.25, 10.3, "ForJet");
      t_text->DrawText(2.25, 9.3, "Tau");
      t_text->DrawText(2.25, 8.3, "ETT");
      t_text->DrawText(2.25, 7.3, "ETM");
      t_text->DrawText(2.25, 6.3, "HTT");
      t_text->DrawText(2.25, 5.3, "HTM");
      t_text->DrawText(2.25, 4.3, "HfBitCounts");
      t_text->DrawText(2.25, 3.3, "HfRingEtSums");
      t_text->DrawText(2.25, 2.3, "GtExternal");
      t_text->DrawText(2.25, 1.3, "TechTrig");

      t_text->DrawText(1.25, 11.3, "GT");
      t_text->DrawText(1.25, 10.3, "GMT");
      t_text->DrawText(1.25, 9.3, "RPC");
      t_text->DrawText(1.25, 8.3, "CSC TF");
      t_text->DrawText(1.25, 7.3, "CSC TPG");
      t_text->DrawText(1.25, 6.3, "DT TF");
      t_text->DrawText(1.25, 5.3, "DT TPG");
      t_text->DrawText(1.25, 4.3, "Stage1Layer2");
      t_text->DrawText(1.25, 3.3, "RCT");
      t_text->DrawText(1.25, 2.3, "HCAL TPG");
      t_text->DrawText(1.25, 1.3, "ECAL TPG");

      l_line->SetLineWidth(2);

      // vertical line

      l_line->DrawLine(2, 1, 2, 15);

      // horizontal lines

      l_line->DrawLine(1, 1, 3, 1);
      l_line->DrawLine(1, 2, 3, 2);
      l_line->DrawLine(1, 3, 3, 3);
      l_line->DrawLine(1, 4, 3, 4);
      l_line->DrawLine(1, 5, 3, 5);
      l_line->DrawLine(1, 6, 3, 6);
      l_line->DrawLine(1, 7, 3, 7);
      l_line->DrawLine(1, 8, 3, 8);
      l_line->DrawLine(1, 9, 3, 9);
      l_line->DrawLine(1, 10, 3, 10);
      l_line->DrawLine(1, 11, 3, 11);
      l_line->DrawLine(1, 12, 3, 12);
      l_line->DrawLine(2, 13, 3, 13);
      l_line->DrawLine(2, 14, 3, 14);

      return;
    }

    // post-draw rendering of other L1TEMU TH2F histograms
  }
};

static L1TEMURenderPlugin instance;
