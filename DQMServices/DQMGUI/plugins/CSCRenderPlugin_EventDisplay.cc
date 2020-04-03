/*
 * =====================================================================================
 *
 *       Filename:  EventDisplay.cc
 *
 *    Description:  draw event display based on encoded historgam chamber
 *
 *        Version:  1.0
 *        Created:  12/12/2009 08:57:49 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), Valdas.Rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#include "CSCRenderPlugin_EventDisplay.h"

/**
 * @brief  Constructor
 */
EventDisplay::EventDisplay() {

  greyscaleExec = new TExec("cscdqm_greyscaleExec", " \
    Double_t Red[2]    = { 1.00, 0.00 };\
    Double_t Green[2]  = { 1.00, 0.00 };\
    Double_t Blue[2]   = { 1.00, 0.00 };\
    Double_t Length[2] = { 0.00, 1.00 };\
    TColor::CreateGradientColorTable(2, Length, Red, Green, Blue, 50);\
  ");
  normalExec = new TExec("cscdqm_normalExec", " \
    gStyle->SetPalette(1,0);\
  ");

  histos[0] = new TH2F("EventDisplay_h1", "Anode Hit Timing and Quality", HISTO_WIDTH, 0, HISTO_WIDTH, 6, 0, 6);
  histos[0]->GetXaxis()->SetLabelSize(0.0);
  histos[0]->GetXaxis()->SetTickLength(0.0);
  histos[0]->GetYaxis()->SetLabelSize(0.0);
  histos[0]->GetYaxis()->SetTickLength(0.0);
  histos[0]->SetBinContent(1, 1, -5.0);
  histos[0]->SetBinContent(1, 2,  5.0);
  histos[0]->SetStats(kFALSE);

  histos[1] = new TH2F("EventDisplay_h2", "Comparator Hit Timing and Quality", HISTO_WIDTH, 0, HISTO_WIDTH, 6, 0, 6);
  histos[1]->GetXaxis()->SetLabelSize(0.0);
  histos[1]->GetXaxis()->SetTickLength(0.0);
  histos[1]->GetYaxis()->SetLabelSize(0.0);
  histos[1]->GetYaxis()->SetTickLength(0.0);
  histos[1]->SetBinContent(1, 1, -5.0);
  histos[1]->SetBinContent(1, 2,  5.0);
  histos[1]->SetStats(kFALSE);

  histos[2] = new TH2F("EventDisplay_h3", "SCA Charges", HISTO_WIDTH, 0, HISTO_WIDTH, 6, 0, 6);
  histos[2]->GetXaxis()->SetLabelSize(0.0);
  histos[2]->GetXaxis()->SetTickLength(0.0);
  histos[2]->GetYaxis()->SetLabelSize(0.0);
  histos[2]->GetYaxis()->SetTickLength(0.0);
  histos[2]->SetBinContent(1, 1, 0.0);
  histos[2]->SetBinContent(1, 2, 1000.0);
  histos[2]->SetStats(kFALSE);

  bBlank = new TBox(-1.0, -1.0, HISTO_WIDTH + 4, 7);
  bBlank->SetFillColor(0);
  bBlank->SetLineColor(0);
  bBlank->SetLineStyle(0);

  tTitle = 0;

  for (int h = 0; h < 3; h++) {
    for (int l = 0; l < 6; l++) {
      for (int x = 0; x < 224; x++) {
        bBox[h][l][x] = 0;
        bKey[h][x] = 0;
        tKey[h][x] = 0;
        tXLabel[h][x] = 0;
      }
      tYLabel[l] = 0;
    }
  }

  tLayer = 0;

  tXTitle[0] = 0;
  tXTitle[1] = 0;
  tXTitle[2] = 0;

}

/**
 * @brief  Destructor
 */
EventDisplay::~EventDisplay() {

  delete histos[0];
  delete histos[1];
  delete histos[2];
  delete bBlank;

  if (tXTitle[0]) delete tXTitle[0];
  if (tXTitle[1]) delete tXTitle[1];
  if (tXTitle[2]) delete tXTitle[2];

  delete greyscaleExec;
  delete normalExec;

}

/**
 * @brief  Draw a single chamber event based on encoded histogram
 * @param  me oncoded histogram
 */
void EventDisplay::drawSingleChamber(TH2*& data) {

  gPad->SetLeftMargin(0.0);
  gPad->SetRightMargin(0.0);
  gPad->SetTopMargin(0.0);
  gPad->SetBottomMargin(0.0);

  pad0 = new TPad("pad0", "Title Pad", 0.00, 0.95, 1.0, 1.00, 0);
  pad1 = new TPad("pad1", "Top pad",   0.00, 0.70, 1.0, 0.95, 0);
  pad2 = new TPad("pad2", "Mid pad",   0.00, 0.25, 1.0, 0.50, 0);
  pad3 = new TPad("pad3", "Bot pad",   0.00, 0.00, 1.0, 0.25, 0);
  pad4 = new TPad("pad4", "Cover pad", 0.00, 0.00, 1.0, 1.00, 0);

  pad4->Draw();
  pad0->Draw();
  pad1->Draw();
  pad2->Draw();
  pad3->Draw();

  int endcap  = (int) data->GetBinContent(1, 1);
  int station = (int) data->GetBinContent(1, 2);
  int ring    = (int) data->GetBinContent(1, 3);
  int chamber = (int) data->GetBinContent(1, 4);
  //int crate   = (int) data->GetBinContent(1, 5);
  //int dmb     = (int) data->GetBinContent(1, 6);
  int event   = (int) data->GetBinContent(1, 7);

  if (event > 0) {

    // *************************************
    // Anode hits and ALCTs
    // *************************************
    {
      pad1->cd();

      gPad->SetLeftMargin(0.05);
      gPad->SetRightMargin(0.08);
      gPad->SetTopMargin(0.08);
      gPad->SetBottomMargin(0.12);

      drawEventDisplayGrid(0, data, 4, 2, 3,
                          countWiregroups(station, ring), 0.0f, -5.0f,
                          5.0f, 0, -3, -8,
                          "wiregroup #", false);
    }

    // *************************************
    // Cathode hits and CLCTs
    // *************************************
    {
      pad2->cd();

      gPad->SetLeftMargin(0.05);
      gPad->SetRightMargin(0.08);
      gPad->SetTopMargin(0.08);
      gPad->SetBottomMargin(0.12);

      drawEventDisplayGrid(1, data, 12, 10, 11,
                          (countStrips(station, ring) + countStripsNose(station, ring)) * 2, (countStripsNose(station, ring) > 0 ? 0.0f : 1.0f), -5.0f, 5.0f,
                          (countStripsNose(station, ring) > 0 ? countStrips(station, ring) * 2 : 0), 155, -7,
                          "half-strip #", false);
    }

    // *************************************
    // SCA Charges
    // *************************************
    {
      pad3->cd();

      gPad->SetLeftMargin(0.05);
      gPad->SetRightMargin(0.08);
      gPad->SetTopMargin(0.08);
      gPad->SetBottomMargin(0.12);

      drawEventDisplayGrid(2, data, 18, -1, -1,
                          (countStrips(station, ring) + countStripsNose(station, ring)), (countStripsNose(station, ring) > 0 ? 0.0f : 0.5f), 0.0f, 1000.0f,
                          (countStripsNose(station, ring) > 0 ? countStrips(station, ring) : 0), 0, 0,
                          "strip #", true);
    }

  }

  pad0->cd();

  TString t = Form("No event");
  if (event != 0) {
    t = Form("Chamber ME%s%d/%d/%d Event #%d", (endcap == 1 ? "+" : "-"), station, ring, chamber, event);
  }

  if (tTitle == 0) {
    tTitle = new TText(0.02, 0.30, t);
    tTitle->SetTextAlign(11);
    tTitle->SetTextFont(62);
    tTitle->SetTextSize(0.6);
  } else {
    tTitle->SetText(0.02, 0.30, t);
  }
  tTitle->Draw();

}

void EventDisplay::drawEventDisplayGrid(int hnum, TH2* data, int data_first_col, int data_time_col, int data_quality_col,
                                        int count_x, float shift_x, float min_z, float max_z, int split_after_x, int time_corr, int d_corr,
                                        const char* title_x, bool greyscale) {

  TObject *post_draw[160 * 2];
  int p_post_draw = 0;

  histos[hnum]->Draw("colz");
  if (greyscale) {
    greyscaleExec->Draw();
    histos[hnum]->Draw("colz same");
  } else {
    normalExec->Draw();
    histos[hnum]->Draw("colz same");
  }

  bBlank->Draw("l");

  float w = (float) HISTO_WIDTH / (count_x + (split_after_x == 0 ? 0 : 2 ));
  if (split_after_x == 0) split_after_x = count_x;

  for (int l = 0; l < 6; l++) {

    int y = 6 - l;

    for (int xg = 0; xg < count_x; xg++) {

      int section_shift = (xg + 1 > split_after_x ? 2 : 0);

      float x = (shift_x * w * ((l + 1) % 2)) + (float) xg * w + section_shift * w;
      int d = (int) data->GetBinContent(data_first_col + l, xg + 1);

      int time = 0;
      if (data_time_col >= 0) {
        time = (int) data->GetBinContent(data_time_col, xg + 1);
      }

      int quality = 0;
      if (data_quality_col >= 0) {
        quality = (int) data->GetBinContent(data_quality_col, xg + 1);
      }

      int color = 0;
      if (d > 0) {
        if (data_time_col >= 0) {
          d = d - 1;
        }
        d += d_corr;
        if (d > max_z) d = (int) max_z;
        if (d < min_z) d = (int) min_z;

        float df = (float) d / (float) (max_z - min_z);

        if (greyscale) {
          color = TColor::GetColor(df, df, df);
        } else {
          color = 51 + (int) (df * 49.0);
        }

      }

      if (bBox[hnum][l][xg] != 0) {
        delete bBox[hnum][l][xg];
      }
      bBox[hnum][l][xg] = new TBox(x, y, x + w, y - 1);
      bBox[hnum][l][xg]->SetLineColor(15);
      bBox[hnum][l][xg]->SetLineStyle(1);
      bBox[hnum][l][xg]->SetFillColor(color);
      bBox[hnum][l][xg]->Draw("l");

      if (l == 2 && quality > 0) {

        time += time_corr;
        if (time > max_z) time = (int) max_z;
        if (time < min_z) time = (int) min_z;

        color = 51 + (int) (((float) time / (float) (max_z - min_z)) * 49.0);

        if (bKey[hnum][xg] != 0) {
          delete bKey[hnum][xg];
        }
        bKey[hnum][xg] = new TBox(x - w * 0.25, y - 0.25, x + w * 1.25, y - 1 + 0.25);
        bKey[hnum][xg]->SetLineColor(1);
        bKey[hnum][xg]->SetLineStyle(1);
        bKey[hnum][xg]->SetFillColor(color);
        post_draw[p_post_draw++] = bKey[hnum][xg];

        TString h = Form("%d", quality);
        if (tKey[hnum][xg] == 0) {
          tKey[hnum][xg] = new TText(x + w / 2, y - 0.53, h);
          tKey[hnum][xg]->SetTextAlign(22);
          tKey[hnum][xg]->SetTextFont(42);
          tKey[hnum][xg]->SetTextSize(0.05);
        } else {
          tKey[hnum][xg]->SetText(x + w / 2, y - 0.53, h);
        }
        post_draw[p_post_draw++] = tKey[hnum][xg];

      }

      if (l == 5 && (count_x < 100 || (count_x > 100 && (xg + 1) % 2))) {
        TString ts = Form("%d", (section_shift > 0 ? xg - split_after_x + 1 : xg + 1));
        if (tXLabel[hnum][xg] == 0) {
          tXLabel[hnum][xg] = new TText(x + w / 2, y - 1.2, ts);
          tXLabel[hnum][xg]->SetTextFont(42);
          tXLabel[hnum][xg]->SetTextSize(0.03);
        } else {
          tXLabel[hnum][xg]->SetText(x + w / 2, y - 1.2, ts);
        }
        tXLabel[hnum][xg]->SetTextAlign(count_x >= 100 ? 32 : 22);
        tXLabel[hnum][xg]->SetTextAngle(count_x >= 100 ? 90 : 0);
        tXLabel[hnum][xg]->Draw();
      }

    }

    if (tYLabel[l] == 0) {
      TString ts = Form("%d", l + 1);
      tYLabel[l] = new TText(-2, y - 0.5, ts);
      tYLabel[l]->SetTextAlign(22);
      tYLabel[l]->SetTextFont(42);
      tYLabel[l]->SetTextSize(0.04);
    }
    tYLabel[l]->Draw();

  }

  if (tLayer == 0) {
    tLayer = new TText(-7, 3, "layer");
    tLayer->SetTextAlign(22);
    tLayer->SetTextFont(42);
    tLayer->SetTextSize(0.05);
    tLayer->SetTextAngle(90.0);
  }
  tLayer->Draw();

  if (tXTitle[hnum] == 0) {
    tXTitle[hnum] = new TText(HISTO_WIDTH, -0.7, title_x);
    tXTitle[hnum]->SetTextAlign(32);
    tXTitle[hnum]->SetTextFont(42);
    tXTitle[hnum]->SetTextSize(0.05);
  }
  tXTitle[hnum]->Draw();

  for (int i = 0; i < p_post_draw; i++) {
    post_draw[i]->Draw("l");
  }

}

/**
 * @brief  Wiregroup number on chamber by station and ring
 * @param  station station
 * @param  ring ring
 * @return wiregroups per chamber
 */
int EventDisplay::countWiregroups(int station, int ring) const {
  if (station == 1) {
    if (ring == 1) return 48;
    if (ring == 2) return 32;
    if (ring == 3) return 112;
  }
  if (station == 2) {
    if (ring == 1) return 112;
  }
  if (ring == 1) return 96;
  return 64;
}

/**
 * @brief  Number of strips per chamber by station and ring
 * @param  station station
 * @param  ring ring
 * @return number of strips in chamber
 */
int EventDisplay::countStrips(int station, int ring) const {
  if (station == 1 && (ring == 1 || ring == 3)) return 64;
  return 80;
}

/**
 * @brief  Number of strips in inner corner (exception for ME1/1)
 * @param  station station
 * @param  ring ring
 * @return number of strips in inner corner
 */
int EventDisplay::countStripsNose(int station, int ring) const {
  if (station == 1 && ring == 1) return 48;
  return 0;
}
