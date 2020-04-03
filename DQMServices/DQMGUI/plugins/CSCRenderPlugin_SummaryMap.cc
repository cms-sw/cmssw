/*
 * =====================================================================================
 *
 *       Filename:  CSCRenderPlugin_SummaryMap.cc
 *
 *    Description:  Makes a real CSC map out of the dummy histogram. Actually it streches ME(+|-)2/1,
 *    ME(+|-)3/1, ME(+|-)4/1 chambers to the full extent of the diagram. Initial algorithm implementation
 *    was dome by YP and the port to DQM was done by VR.
 *
 *        Version:  1.0
 *        Created:  04/09/2008 04:57:49 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Yuriy Pakhotin (YP), pakhotin@ufl.edu; Valdas Rapsevicius (VR), Valdas.Rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#include "CSCRenderPlugin_SummaryMap.h"
#include <math.h>
#include <string>
#include <iostream>
#include <bitset>
#include <TH1.h>
#include <TH2.h>
#include <TBox.h>
#include <TText.h>
#include <TPRegexp.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TLine.h>

SummaryMap::SummaryMap() : detector(N_TICS, N_TICS) {

  h1 = new TH2F("h1", "", 10, -2.5, 2.5, 10, 0.0, 2.0*3.14159);
  h1->GetXaxis()->SetTitle("#eta");
  h1->GetXaxis()->SetTitleOffset(1.2);
  h1->GetXaxis()->CenterTitle(true);
  h1->GetXaxis()->SetLabelSize(0.03);
  h1->GetXaxis()->SetTicks("+-");
  h1->GetXaxis()->SetNdivisions(20510, kTRUE);
  h1->GetYaxis()->SetTitle("#phi");
  h1->GetYaxis()->SetTitleOffset(-1.2);
  h1->GetYaxis()->SetTicks("+-");
  h1->GetYaxis()->SetNdivisions(21010, kTRUE);
  h1->GetYaxis()->CenterTitle(true);
  h1->GetYaxis()->SetLabelSize(0.03);
  h1->SetStats(kFALSE);

  bBlank = new TBox(-2.5, 0.0, 2.5, 2.0*3.14159);
  bBlank->SetFillColor(18);
  bBlank->SetLineColor(1);
  bBlank->SetLineStyle(1);

  bEmptyMinus = new TBox(-2.45, 0.0, -1.0, 2.0*3.14159);
  bEmptyMinus->SetFillColor(10);
  bEmptyMinus->SetLineColor(1);
  bEmptyMinus->SetLineStyle(1);

  bEmptyPlus = new TBox(2.45, 0.0, 1.0, 2.0*3.14159);
  bEmptyPlus->SetFillColor(10);
  bEmptyPlus->SetLineColor(1);
  bEmptyPlus->SetLineStyle(1);

  for(unsigned int x = 0; x < N_TICS; x++) {
    for(unsigned int y = 0; y < N_TICS; y++) {
      bDetector[x][y] = 0;
    }
    if (x > 0) {
      lDetector[x - 1][0] = 0;
      lDetector[x - 1][1] = 0;
    }
  }

  tDetector = 0;

  for (int station = 0; station < 4; station++) {
    for (int i = 0; i < N_ELEMENTS; i++) {
      bStation[station][i] = 0;
    }
    for (int i = 0; i < 3456; i++) {
      lStation[station][i] = 0;
    }
    for (int i = 0; i < 864; i++) {
      tStationCSC_label[station][i] = 0;
    }
    for (int i = 0; i < 6; i++) {
      tStationRing_label[station][i] = 0;
    }
  }

  tStation_minus_label = 0;
  tStation_plus_label = 0;
  tStation_title = 0;
}

SummaryMap::~SummaryMap()
{
  delete h1;
  delete bBlank;
  delete bEmptyMinus;
  delete bEmptyPlus;
}

void SummaryMap::drawDetector(TH2* me) {

  gStyle->SetPalette(1,0);

  h1->Draw();
  bBlank->Draw("l");
  bEmptyMinus->Draw("l");
  bEmptyPlus->Draw("l");

  float xd = 5.0 / N_TICS, yd = 1.0 * (2.0 * 3.14159) / N_TICS;

  float xmin, xmax, ymin, ymax;

  for(unsigned int x = 0; x < N_TICS; x++)
  {
    xmin = -2.5 + xd * x;
    xmax = xmin + xd;

    if (xmin == -2.5 || xmax == 2.5) continue;
    if (xmin >= -1 && xmax <= 1)     continue;

    for(unsigned int y = 0; y < N_TICS; y++)
    {
      ymin = yd * y;
      ymax = ymin + yd;

      int value = int(me->GetBinContent(x + 1, y + 1));

      if (value != 0)
      {

        if (bDetector[x][y] == 0) {
          bDetector[x][y] = new TBox(xmin, ymin, xmax, ymax);
          bDetector[x][y]->SetFillStyle(1001);
        }

        switch (value)
        {
        case -1:
          // Error (RED)
          bDetector[x][y]->SetFillColor(2);
          break;
        case 1:
          // OK (GREEN)
          bDetector[x][y]->SetFillColor(8);
          break;
        case 2:
          // Swithed off (DARK GREY)
          bDetector[x][y]->SetFillColor(17);
        }
        bDetector[x][y]->Draw("");

      }

    }
  }

  for(unsigned int x = 1; x < N_TICS; x++) {
    if (lDetector[x - 1][0] == 0) {
      lDetector[x - 1][0] = new TLine(-2.5 + xd * x, 0.0, -2.5 + xd * x, 2.0 * 3.14159);
      lDetector[x - 1][0]->SetLineColor(12);
      lDetector[x - 1][0]->SetLineStyle(1);
      lDetector[x - 1][0]->SetLineWidth(1);
    }
    lDetector[x - 1][0]->Draw();
    if (lDetector[x - 1][1] == 0) {
      lDetector[x - 1][1] = new TLine(-2.5, yd * x, 2.5, yd * x);
      lDetector[x - 1][1]->SetLineColor(12);
      lDetector[x - 1][1]->SetLineStyle(1);
      lDetector[x - 1][1]->SetLineWidth(1);
    }
    lDetector[x - 1][1]->Draw();
  }

  if (tDetector == 0) {
    tDetector = new TText(0.0, 2.0 * 3.14159 + 0.5, me->GetTitle());
    tDetector->SetTextAlign(22);
    tDetector->SetTextFont(62);
    tDetector->SetTextSize(0.04);
  } else {
    tDetector->SetText(0.0, 2.0 * 3.14159 + 0.5, me->GetTitle());
  }
  tDetector->Draw();

}

void SummaryMap::drawStation(TH2* me, const int station) {

  cscdqm::Address adr;

  gStyle->SetPalette(1,0);

  h1->Draw();
  bBlank->Draw("l");

  float x_min_chamber = FLT_MAX, x_max_chamber = FLT_MIN;
  float y_min_chamber = FLT_MAX, y_max_chamber = FLT_MIN;

  const cscdqm::AddressBox *box;
  adr.mask.side = adr.mask.ring = adr.mask.chamber  = adr.mask.layer = adr.mask.cfeb = adr.mask.hv = false;
  adr.mask.station = true;
  adr.station = station;

  unsigned int i = 0, p_hw = 0, p_l = 0, p_csc = 0, p_ring = 0;
  while(detector.NextAddressBox(i, box, adr)) {

    if (bStation[station - 1][p_hw] == 0) {
      bStation[station - 1][p_hw] = new TBox(box->xmin, box->ymin, box->xmax, box->ymax);
      bStation[station - 1][p_hw]->SetLineColor(12);
      bStation[station - 1][p_hw]->SetLineStyle(1);
    }

    unsigned int x = 1 + (box->adr.side - 1) * 9 + (box->adr.ring - 1) * 3 + (box->adr.hv - 1);
    unsigned int y = 1 + (box->adr.chamber - 1) * 5 + (box->adr.cfeb - 1);

    switch ((int) me->GetBinContent(x, y))
    {
    case -1:
      // Error (RED)
      bStation[station - 1][p_hw]->SetFillColor(2);
      break;
    case 1:
      // OK (GREEN)
      bStation[station - 1][p_hw]->SetFillColor(8);
      break;
    case 2:
      // Swithed off (DARK GREY)
      bStation[station - 1][p_hw]->SetFillColor(16);
      break;
    case 0:
      // No data (WHITE)
      bStation[station - 1][p_hw]->SetFillColor(10);
      break;
    default:
      // Application error!? Can not be this... (MAGENTA)
      bStation[station - 1][p_hw]->SetFillColor(6);
    }

    bStation[station - 1][p_hw]->Draw("l");
    p_hw++;

    // If this is the last hw element in the chamber - proceed drawing chamber
    if(box->adr.cfeb == detector.NumberOfChamberCFEBs(box->adr.station, box->adr.ring) &&
       box->adr.hv == detector.NumberOfChamberHVs(box->adr.station, box->adr.ring))
    {
      x_max_chamber = box->xmax;
      y_max_chamber = box->ymax;

      if (lStation[station - 1][p_l] == 0) {
        lStation[station - 1][p_l]     = new TLine(x_min_chamber, y_min_chamber, x_min_chamber, y_max_chamber);
        lStation[station - 1][p_l + 1] = new TLine(x_max_chamber, y_min_chamber, x_max_chamber, y_max_chamber);
        lStation[station - 1][p_l + 2] = new TLine(x_min_chamber, y_min_chamber, x_max_chamber, y_min_chamber);
        lStation[station - 1][p_l + 3] = new TLine(x_min_chamber, y_max_chamber, x_max_chamber, y_max_chamber);
        for(int n_l = 0; n_l < 4; n_l++) {
          lStation[station - 1][p_l + n_l]->SetLineColor(1);
          lStation[station - 1][p_l + n_l]->SetLineStyle(1);
          lStation[station - 1][p_l + n_l]->SetLineWidth(1);
        }
      }

      for(int n_l = 0; n_l < 4; n_l++) {
        lStation[station - 1][p_l + n_l]->Draw();
      }

      p_l += 4;

      if (tStationCSC_label[station - 1][p_csc] == 0) {
        TString ChamberID = Form("%d", box->adr.chamber);
        tStationCSC_label[station - 1][p_csc] = new TText((x_min_chamber + x_max_chamber) / 2.0, (y_min_chamber + y_max_chamber) / 2.0, ChamberID);
        tStationCSC_label[station - 1][p_csc]->SetTextAlign(22);
        tStationCSC_label[station - 1][p_csc]->SetTextFont(42);
        tStationCSC_label[station - 1][p_csc]->SetTextSize(0.02);
      }
      tStationCSC_label[station - 1][p_csc]->Draw();
      p_csc++;

      // Last HW element in Ring? display ring label
      if(box->adr.chamber == detector.NumberOfChambers(box->adr.station, box->adr.ring))
      {
        if (tStationRing_label[station - 1][p_ring] == 0) {
          TString ringID = Form("%d", box->adr.ring);
          tStationRing_label[station - 1][p_ring] = new TText((x_min_chamber + x_max_chamber) / 2.0, 2.0 * 3.14159 + 0.1, ringID);
          tStationRing_label[station - 1][p_ring]->SetTextAlign(22);
          tStationRing_label[station - 1][p_ring]->SetTextFont(62);
          tStationRing_label[station - 1][p_ring]->SetTextSize(0.02);
        }
        tStationRing_label[station - 1][p_ring]->Draw();
        p_ring++;
      }
    }
    else if (box->adr.cfeb == 1 && box->adr.hv == 1)
    {
      x_min_chamber = box->xmin;
      y_min_chamber = box->ymin;
    }
  }

  TString stationID_minus = Form("ME-%d", station);
  if (tStation_minus_label == 0) {
    tStation_minus_label = new TText(-1.7, 2.0 * 3.14159 + 0.25, stationID_minus);
    tStation_minus_label->SetTextAlign(22);
    tStation_minus_label->SetTextFont(62);
    tStation_minus_label->SetTextSize(0.02);
  } else {
    tStation_minus_label->SetText(-1.7, 2.0 * 3.14159 + 0.25, stationID_minus);
  }
  tStation_minus_label->Draw();

  TString stationID_plus = Form("ME+%d", station);
  if (tStation_plus_label == 0) {
    tStation_plus_label = new TText(1.7, 2.0 * 3.14159 + 0.25, stationID_plus);
    tStation_plus_label->SetTextAlign(22);
    tStation_plus_label->SetTextFont(62);
    tStation_plus_label->SetTextSize(0.02);
  } else {
    tStation_plus_label->SetText(1.7, 2.0 * 3.14159 + 0.25, stationID_plus);
  }
  tStation_plus_label->Draw();

  if (tStation_title == 0) {
    tStation_title = new TText(0.0, 2.0 * 3.14159 + 0.5, me->GetTitle());
    tStation_title->SetTextAlign(22);
    tStation_title->SetTextFont(62);
    tStation_title->SetTextSize(0.04);
  } else {
    tStation_title->SetText(0.0, 2.0 * 3.14159 + 0.5, me->GetTitle());
  }
  tStation_title->Draw();

}
