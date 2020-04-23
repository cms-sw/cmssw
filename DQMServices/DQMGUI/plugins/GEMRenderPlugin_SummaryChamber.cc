/*
 * =====================================================================================
 *
 *       Filename:  SummaryChamber.cc
 *       (the original: CSCRenderPlugin_ChamberMap.cc)
 *
 *    Description:  Makes a real GEM map out of the dummy histogram.
 *                  For more description, see CSCRenderPlugin_ChamberMap.cc
 *
 *        Version:  0.1
 *        Created:  22/06/2019
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Yuriy Pakhotin (YP), pakhotin@ufl.edu; Valdas Rapsevicius (VR), Valdas.Rapsevicius@cern.ch
 *         (the original)
 *        Company:  CERN, CH
 *         Copier:  Byeonghak Ko, bko@cern.ch, University of Seoul
 *
 * =====================================================================================
 */

#include "GEMRenderPlugin_SummaryChamber.h"
#include <unordered_map>

uint32_t ChIdToInt(ChamberID &id) {
  return ((id.nRegion + 1) << 11) | (id.nStation << 9) | (id.nLayer << 6) | (id.nChamber << 0);
}

SummaryChamber::SummaryChamber() {
  m_nNumLayer = 0;
  m_nNumChamber = 36;

  m_fScaleX = 1.005;
  m_fScaleY = 1.0;

  std::vector<ChamberID> listChPre;

  for (Int_t nR = -1; nR <= 1; nR += 2)
    for (Int_t nS = 1; nS <= 2; nS++)
      //for ( Int_t nS = 1 ; nS <= 1 ; nS++ ) // When watching only GE11
      for (Int_t nL = 1; nL <= 2; nL++) {
        ChamberID idNew;

        idNew.nRegion = nR;
        idNew.nStation = nS;
        idNew.nLayer = nL;
        idNew.nChamber = 0;
        idNew.bIsDouble = (nS == 2);

        listChPre.push_back(idNew);
      }

  auto lambdaChamber = [](ChamberID a, ChamberID b) -> bool {
    Int_t nA = a.nRegion * (20 * a.nStation + a.nLayer);  // Maybe 10, not 20, is enough
    Int_t nB = b.nRegion * (20 * b.nStation + b.nLayer);
    return nA > nB;
  };

  //std::sort(listChPre.begin(), listChPre.end(), lambdaChamber);
  for (unsigned int i = 0; i < (unsigned int)listChPre.size(); i++) {
    for (unsigned int j = i + 1; j < (unsigned int)listChPre.size(); j++) {
      if (!lambdaChamber(listChPre[i], listChPre[j])) {
        auto objS = listChPre[i];
        listChPre[i] = listChPre[j];
        listChPre[j] = objS;
      }
    }
  }

  for (auto id : listChPre) {
    m_nNumLayer++;
    id.nIdx = m_nNumLayer;
    Int_t nNumChSub = (!id.bIsDouble ? m_nNumChamber : m_nNumChamber / 2);

    for (Int_t nC = 1; nC <= nNumChSub; nC++) {
      id.nChamber = nC;
      uint32_t nIdxFull = ChIdToInt(id);
      bGEM_ChInfo[nIdxFull] = id;

      bGEM_box[nIdxFull] = nullptr;
      bGEM_label[nIdxFull] = nullptr;
    }
  }

  bBlank = new TBox(0.0, 0.0, 1.0 + m_fScaleX * m_nNumChamber, m_fScaleY * m_nNumLayer);
  bBlank->SetFillColor(0);
  bBlank->SetLineColor(1);
  bBlank->SetLineStyle(1);

  for (int i = 0; i < 5; i++) {
    bLegend[i] = nullptr;
    tLegend[i] = nullptr;
  }

  tStatusTitle = new TText(3.5 + m_fScaleX * m_nNumChamber, 17.5, "Status");
  tStatusTitle->SetTextAlign(22);
  tStatusTitle->SetTextFont(42);
  tStatusTitle->SetTextSize(0.02);

  tLegendTitle = new TText(3.5 + m_fScaleX * m_nNumChamber, 13.5, "Legend");
  tLegendTitle->SetTextAlign(22);
  tLegendTitle->SetTextFont(42);
  tLegendTitle->SetTextSize(0.02);
}

SummaryChamber::~SummaryChamber() {
  delete bBlank;
  delete tStatusTitle;
  delete tLegendTitle;
}

// Transform chamber ID to local canvas coordinates

float SummaryChamber::GetXmin(ChamberID &id) const {
  return m_fScaleX * (id.nChamber - 1) * (!id.bIsDouble ? 1.0 : 2.0);
}
float SummaryChamber::GetXmax(ChamberID &id) const { return m_fScaleX * (id.nChamber) * (!id.bIsDouble ? 1.0 : 2.0); }
float SummaryChamber::GetYmin(ChamberID &id) const { return m_fScaleY * (id.nIdx - 1); }
float SummaryChamber::GetYmax(ChamberID &id) const { return m_fScaleY * (id.nIdx); }

void SummaryChamber::drawStats(TH2 *&me) {
  gStyle->SetPalette(1, nullptr);

  // Useless labels for the current layout
  for (int i = 0; i < me->GetNbinsX(); i++)
    me->GetXaxis()->SetBinLabel(i + 1, "");

  /** Cosmetics... :P */
  me->GetXaxis()->SetTitle("Chamber");
  me->GetXaxis()->CenterTitle(true);
  me->GetXaxis()->SetLabelSize(0.0);
  me->GetXaxis()->SetTicks("0");
  me->GetXaxis()->SetNdivisions(0);
  me->GetXaxis()->SetTickLength(0.0);

  me->SetStats(false);
  me->Draw("");

  bBlank->SetFillStyle(1001);
  bBlank->Draw("");

  std::bitset<10> legend;
  legend.reset();

  /** VR: Making it floats and moving up */
  float fXMin, fXMax, fYMin, fYMax;
  float BinContent = 0;
  int fillColor = 0;

  unsigned int status_all = 0, status_bad = 0;

  int nT = 0;
  for (auto itemBox : bGEM_box) {
    nT++;
    auto gid = bGEM_ChInfo[itemBox.first];

    fXMin = GetXmin(gid);
    fXMax = GetXmax(gid);
    fYMin = GetYmin(gid);
    fYMax = GetYmax(gid);

    BinContent = 0;
    fillColor = 0;

    /** VR: if the station/ring is an exceptional one (less chambers) we should
     * correct x coordinates of source. Casts are just to avoid warnings :) */
    BinContent = (float)me->GetBinContent(gid.nChamber, gid.nIdx);

    fillColor = int(BinContent);
    if (fillColor < 0 || fillColor > 5)
      fillColor = 0;
    legend.set(fillColor);

    switch (fillColor) {
      // No data, no error
      case 0:
        fillColor = COLOR_WHITE;
        status_all += 1;
        break;
      // Data, no error
      case 1:
        fillColor = COLOR_GREEN;
        status_all += 1;
        break;
      // Error, hot
      case 2:
        fillColor = COLOR_RED;
        status_all += 1;
        status_bad += 1;
        break;
      // Cold
      case 3:
        fillColor = COLOR_BLUE;
        status_all += 1;
        status_bad += 1;
        break;
      // Masked
      case 4:
        fillColor = COLOR_GREY;
        break;
      // Standby
      case 5:
        fillColor = COLOR_YELLOW;
        status_all += 1;
        status_bad += 1;
        break;
    }

    if (bGEM_box[itemBox.first] == nullptr) {
      bGEM_box[itemBox.first] = new TBox(fXMin, fYMin, fXMax, fYMax);

      bGEM_box[itemBox.first]->SetLineColor(1);
      bGEM_box[itemBox.first]->SetLineStyle(2);
    }

    bGEM_box[itemBox.first]->SetFillColor(fillColor);
    bGEM_box[itemBox.first]->Draw("l");

    if (bGEM_label[itemBox.first] == nullptr) {
      TString strChamberID = Form("%d", gid.nChamber);
      bGEM_label[itemBox.first] = new TText((fXMin + fXMax) / 2.0, (fYMin + fYMax) / 2.0, strChamberID);

      bGEM_label[itemBox.first]->SetTextAlign(22);
      bGEM_label[itemBox.first]->SetTextFont(42);
      bGEM_label[itemBox.first]->SetTextSize(0.015);
    }

    bGEM_label[itemBox.first]->Draw();
  }

  unsigned int legendBoxIndex = 2;
  std::string meTitle(me->GetTitle());

  tStatusTitle->Draw();
  tLegendTitle->Draw();

  // Only standby plus possibly masked?
  if (legend == 0x20 || legend == 0x30) {
    meTitle.append(" (STANDBY)");
    me->SetTitle(meTitle.c_str());

    printLegendBox(0, "BAD", COLOR_RED);
    printLegendBox(legendBoxIndex++, "Standby", COLOR_YELLOW);
  } else {
    double status = 1.0;

    if (status_all > 0) {
      status = status - (1.0 * status_bad) / (1.0 * status_all);
      meTitle.append(" (%4.1f%%)");
      TString statusStr = Form(meTitle.c_str(), status * 100.0);
      me->SetTitle(statusStr);
    }

    if (status >= 0.75) {
      printLegendBox(0, "GOOD", COLOR_GREEN);
    } else {
      printLegendBox(0, "BAD", COLOR_RED);
    }

    if (legend.test(0))
      printLegendBox(legendBoxIndex++, "OK/No Data", COLOR_WHITE);
    if (legend.test(1))
      printLegendBox(legendBoxIndex++, "OK/Data", COLOR_GREEN);
    if (legend.test(2))
      printLegendBox(legendBoxIndex++, "Error/Hot", COLOR_RED);
    if (legend.test(3))
      printLegendBox(legendBoxIndex++, "Cold", COLOR_BLUE);
    if (legend.test(4))
      printLegendBox(legendBoxIndex++, "Masked", COLOR_GREY);
    if (legend.test(5))
      printLegendBox(legendBoxIndex++, "Standby", COLOR_YELLOW);
  }
}

void SummaryChamber::printLegendBox(const unsigned int &number, const std::string title, int color) {
  /*if (bLegend[number] == 0) {
      bLegend[number] = new TBox(38, 17 - number * 2, 41, 17 - number * 2 - 1);
      bLegend[number]->SetLineColor(1);
      bLegend[number]->SetLineStyle(2);
  }
  bLegend[number]->SetFillColor(color);
  bLegend[number]->Draw("l");
  
  if (tLegend[number] == 0) {
      tLegend[number] = new TText((38 + 41) / 2.0, (2 * (17 - number * 2) - 1) / 2.0, title.c_str());
      tLegend[number]->SetTextAlign(22);
      tLegend[number]->SetTextFont(42);
      tLegend[number]->SetTextSize(0.015);
  } else {
      tLegend[number]->SetText((38 + 41) / 2.0, (2 * (17 - number * 2) - 1) / 2.0, title.c_str());
  }
  tLegend[number]->Draw();*/

  Float_t fBasisX = m_nNumChamber + 0.7;
  Float_t fBasisY = 6.0;
  Float_t fRatioY = m_nNumLayer / 23.0;

  Int_t nNumItem = 6;

  if (bLegend[number] == nullptr) {
    bLegend[number] = new TBox(fBasisX,
                               fRatioY * (fBasisY + 2 * (nNumItem - number)),
                               fBasisX + 3,
                               fRatioY * (fBasisY + 2 * (nNumItem - number) - 1));
    bLegend[number]->SetLineColor(1);
    bLegend[number]->SetLineStyle(2);
  }
  bLegend[number]->SetFillColor(color);
  bLegend[number]->Draw("l");

  if (tLegend[number] == nullptr) {
    tLegend[number] = new TText(fBasisX + 1.5, fRatioY * (fBasisY + 2 * (nNumItem - number) - 0.5), title.c_str());
    tLegend[number]->SetTextAlign(22);
    tLegend[number]->SetTextFont(42);
    tLegend[number]->SetTextSize(0.015);
  } else {
    tLegend[number]->SetText(0, 0, title.c_str());
  }
  tLegend[number]->Draw();
}
