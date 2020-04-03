/*
 * =====================================================================================
 *
 *       Filename:  ChamberMap.cc
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

#include "CSCRenderPlugin_ChamberMap.h"

ChamberMap::ChamberMap() {

    bBlank = new TBox(1.0, 0.0, 37, 18);
    bBlank->SetFillColor(0);
    bBlank->SetLineColor(1);
    bBlank->SetLineStyle(1);

    for (int n_side = 1; n_side <= 2; n_side++) {
        for (int station = 1; station <= 4; station++) {
            for (int n_ring = 1; n_ring <= N_ring(station); n_ring++) {
                for (int n_chamber = 1; n_chamber <= N_chamber(station, n_ring); n_chamber++) {
                    bCSC_box[n_side - 1][station - 1][n_ring - 1][n_chamber - 1] = 0;
                    tCSC_label[n_side - 1][station - 1][n_ring - 1][n_chamber - 1] = 0;
                }
            }
        }
    }

    for (int i = 0; i < 5; i++) {
        bLegend[i] = 0;
        tLegend[i] = 0;
    }

    tStatusTitle = new TText(39.5, 17.5, "Status");
    tStatusTitle->SetTextAlign(22);
    tStatusTitle->SetTextFont(42);
    tStatusTitle->SetTextSize(0.02);

    tLegendTitle = new TText(39.5, 13.5, "Legend");
    tLegendTitle->SetTextAlign(22);
    tLegendTitle->SetTextFont(42);
    tLegendTitle->SetTextSize(0.02);

}

ChamberMap::~ChamberMap() {
    delete bBlank;
    delete tStatusTitle;
    delete tLegendTitle;
}

// Transform chamber ID to local canvas coordinates

float ChamberMap::Xmin_local_derived_from_ChamberID(int /* side */, int station, int ring, int chamber) const {
    float x;

    if ((station == 2 || station == 3 || station == 4) && ring == 1) {
        x = (float) ((chamber - 1)*2);
    } else {
        x = (float) (chamber - 1);
    }

    return x;
}

// Transform chamber ID to local canvas coordinates

float ChamberMap::Xmax_local_derived_from_ChamberID(int /* side */, int station, int ring, int chamber) const {
    float x;

    if ((station == 2 || station == 3 || station == 4) && ring == 1) {
        x = (float) ((chamber)*2);
    } else {
        x = (float) (chamber);
    }

    return x;
}

// Transform chamber ID to local canvas coordinates

float ChamberMap::Ymin_local_derived_from_ChamberID(int side, int station, int ring, int /* chamber */) const {
    float y = 0;
    float offset = 0.0;

    if (side == 2) { // "-" side
        offset = 0.0;
        if (station == 4 && ring == 2) y = offset + 0.0;
        if (station == 4 && ring == 1) y = offset + 1.0;
        if (station == 3 && ring == 2) y = offset + 2.0;
        if (station == 3 && ring == 1) y = offset + 3.0;
        if (station == 2 && ring == 2) y = offset + 4.0;
        if (station == 2 && ring == 1) y = offset + 5.0;
        if (station == 1 && ring == 3) y = offset + 6.0;
        if (station == 1 && ring == 2) y = offset + 7.0;
        if (station == 1 && ring == 1) y = offset + 8.0;
    }

    if (side == 1) {// "+" side
        offset = 9.0;
        if (station == 1 && ring == 1) y = offset + 0.0;
        if (station == 1 && ring == 2) y = offset + 1.0;
        if (station == 1 && ring == 3) y = offset + 2.0;
        if (station == 2 && ring == 1) y = offset + 3.0;
        if (station == 2 && ring == 2) y = offset + 4.0;
        if (station == 3 && ring == 1) y = offset + 5.0;
        if (station == 3 && ring == 2) y = offset + 6.0;
        if (station == 4 && ring == 1) y = offset + 7.0;
        if (station == 4 && ring == 2) y = offset + 8.0;
    }

    return y;
}

// Transform chamber ID to local canvas coordinates

float ChamberMap::Ymax_local_derived_from_ChamberID(int side, int station, int ring, int /* chamber */) const {
    float y = 0;
    float offset = 0.0;

    if (side == 2) { // "-" side
        offset = 1.0;
        if (station == 4 && ring == 2) y = offset + 0.0;
        if (station == 4 && ring == 1) y = offset + 1.0;
        if (station == 3 && ring == 2) y = offset + 2.0;
        if (station == 3 && ring == 1) y = offset + 3.0;
        if (station == 2 && ring == 2) y = offset + 4.0;
        if (station == 2 && ring == 1) y = offset + 5.0;
        if (station == 1 && ring == 3) y = offset + 6.0;
        if (station == 1 && ring == 2) y = offset + 7.0;
        if (station == 1 && ring == 1) y = offset + 8.0;
    }
    if (side == 1) {// "+" side
        offset = 10.0;
        if (station == 1 && ring == 1) y = offset + 0.0;
        if (station == 1 && ring == 2) y = offset + 1.0;
        if (station == 1 && ring == 3) y = offset + 2.0;
        if (station == 2 && ring == 1) y = offset + 3.0;
        if (station == 2 && ring == 2) y = offset + 4.0;
        if (station == 3 && ring == 1) y = offset + 5.0;
        if (station == 3 && ring == 2) y = offset + 6.0;
        if (station == 4 && ring == 1) y = offset + 7.0;
        if (station == 4 && ring == 2) y = offset + 8.0;
    }
    return y;
}

// Ring number

int ChamberMap::N_ring(int station) const {
    int n_ring = 0;
    if (station == 1) n_ring = 3;
    if (station == 2) n_ring = 2;
    if (station == 3) n_ring = 2;
    if (station == 4) n_ring = 2;
    return n_ring;
}

// Chamber number

int ChamberMap::N_chamber(int station, int ring) const {
    int n_chambers;
    if (station == 1) n_chambers = 36;
    else {
        if (ring == 1) n_chambers = 18;
        else n_chambers = 36;
    }
    return n_chambers;
}

void ChamberMap::draw(TH2*& me) {

    gStyle->SetPalette(1, 0);

    /** VR: Moved this up and made float */
    float HistoMaxValue = me->GetMaximum();
    float HistoMinValue = me->GetMinimum();

    /** Cosmetics... */
    me->GetXaxis()->SetTitle("Chamber");
    me->GetXaxis()->CenterTitle(true);
    me->GetXaxis()->SetLabelSize(0.0);
    me->GetXaxis()->SetTicks("0");
    me->GetXaxis()->SetNdivisions(0);
    me->GetXaxis()->SetTickLength(0.0);

    me->Draw("colz");

    bBlank->Draw("l");

    /** VR: Making it floats and moving up */
    float x_min_chamber, x_max_chamber;
    float y_min_chamber, y_max_chamber;
    float BinContent = 0;
    int fillColor = 0;

    for (int n_side = 1; n_side <= 2; n_side++) {
        for (int station = 1; station <= 4; station++) {
            for (int n_ring = 1; n_ring <= N_ring(station); n_ring++) {
                for (int n_chamber = 1; n_chamber <= N_chamber(station, n_ring); n_chamber++) {
                    x_min_chamber = Xmin_local_derived_from_ChamberID(n_side, station, n_ring, n_chamber);
                    x_max_chamber = Xmax_local_derived_from_ChamberID(n_side, station, n_ring, n_chamber);
                    y_min_chamber = Ymin_local_derived_from_ChamberID(n_side, station, n_ring, n_chamber);
                    y_max_chamber = Ymax_local_derived_from_ChamberID(n_side, station, n_ring, n_chamber);

                    BinContent = 0;
                    fillColor = 0;

                    /** VR: if the station/ring is an exceptional one (less chambers) we should
                     * correct x coordinates of source. Casts are just to avoid warnings :) */
                    if (station > 1 && n_ring == 1) {
                        BinContent = (float) me->GetBinContent((int) x_max_chamber / 2, (int) y_max_chamber);
                    } else {
                        BinContent = (float) me->GetBinContent((int) x_max_chamber, (int) y_max_chamber);
                    }
                    if (BinContent != 0) {
                        /** VR: color calculation differs for linear and log10 scales though... */
                        if (gPad->GetLogz() == 1) {
                            fillColor = 51 + (int) (((log10(BinContent) - log10(HistoMaxValue) + 3) / 3) * 49.0);
                        } else {
                            fillColor = 51 + (int) (((BinContent - HistoMinValue) / (HistoMaxValue - HistoMinValue)) * 49.0);
                        }
                        /** VR: just to be sure :) */
                        if (fillColor > 99) fillColor = 99;
                        if (fillColor < 51) fillColor = 51;

                    }

                    if (bCSC_box[n_side - 1][station - 1][n_ring - 1][n_chamber - 1] == 0) {
                        bCSC_box[n_side - 1][station - 1][n_ring - 1][n_chamber - 1] = new TBox(x_min_chamber + 1, y_min_chamber, x_max_chamber + 1, y_max_chamber);
                        bCSC_box[n_side - 1][station - 1][n_ring - 1][n_chamber - 1]->SetLineColor(1);
                        bCSC_box[n_side - 1][station - 1][n_ring - 1][n_chamber - 1]->SetLineStyle(2);
                    }
                    bCSC_box[n_side - 1][station - 1][n_ring - 1][n_chamber - 1]->SetFillColor(fillColor);
                    bCSC_box[n_side - 1][station - 1][n_ring - 1][n_chamber - 1]->Draw("l");

                    if (tCSC_label[n_side - 1][station - 1][n_ring - 1][n_chamber - 1] == 0) {
                        TString ChamberID = Form("%d", n_chamber);
                        tCSC_label[n_side - 1][station - 1][n_ring - 1][n_chamber - 1] = new TText((x_min_chamber + x_max_chamber) / 2.0 + 1, (y_min_chamber + y_max_chamber) / 2.0, ChamberID);
                        tCSC_label[n_side - 1][station - 1][n_ring - 1][n_chamber - 1]->SetTextAlign(22);
                        tCSC_label[n_side - 1][station - 1][n_ring - 1][n_chamber - 1]->SetTextFont(42);
                        tCSC_label[n_side - 1][station - 1][n_ring - 1][n_chamber - 1]->SetTextSize(0.015);
                    }
                    tCSC_label[n_side - 1][station - 1][n_ring - 1][n_chamber - 1]->Draw();

                }
            }
        }
    }

}

void ChamberMap::drawStats(TH2*& me) {

    gStyle->SetPalette(1, 0);

    /** Cosmetics... */
    me->GetXaxis()->SetTitle("Chamber");
    me->GetXaxis()->CenterTitle(true);
    me->GetXaxis()->SetLabelSize(0.0);
    me->GetXaxis()->SetTicks("0");
    me->GetXaxis()->SetNdivisions(0);
    me->GetXaxis()->SetTickLength(0.0);

    me->SetStats(false);
    me->Draw("col");

    TBox* bBlank = new TBox(1.0, 0.0, 37, 18);
    bBlank->SetFillColor(0);
    bBlank->SetLineColor(1);
    bBlank->SetLineStyle(1);
    bBlank->Draw("l");

    std::bitset < 10 > legend;
    legend.reset();

    unsigned int status_all = 0, status_bad = 0;

    /** VR: Making it floats and moving up */
    float x_min_chamber, x_max_chamber;
    float y_min_chamber, y_max_chamber;
    float BinContent = 0;
    int fillColor = 0;

    for (int n_side = 1; n_side <= 2; n_side++) {
        for (int station = 1; station <= 4; station++) {
            for (int n_ring = 1; n_ring <= N_ring(station); n_ring++) {
                for (int n_chamber = 1; n_chamber <= N_chamber(station, n_ring); n_chamber++) {
                    x_min_chamber = Xmin_local_derived_from_ChamberID(n_side, station, n_ring, n_chamber);
                    x_max_chamber = Xmax_local_derived_from_ChamberID(n_side, station, n_ring, n_chamber);
                    y_min_chamber = Ymin_local_derived_from_ChamberID(n_side, station, n_ring, n_chamber);
                    y_max_chamber = Ymax_local_derived_from_ChamberID(n_side, station, n_ring, n_chamber);

                    BinContent = 0;
                    fillColor = 0;

                    /** VR: if the station/ring is an exceptional one (less chambers) we should
                     * correct x coordinates of source. Casts are just to avoid warnings :) */
                    if (station > 1 && n_ring == 1) {
                        BinContent = (float) me->GetBinContent((int) x_max_chamber / 2, (int) y_max_chamber);
                    } else {
                        BinContent = (float) me->GetBinContent((int) x_max_chamber, (int) y_max_chamber);
                    }

                    fillColor = int(BinContent);

                    if (fillColor < 0 || fillColor > 5) fillColor = 0;
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

                    if (bCSC_box[n_side - 1][station - 1][n_ring - 1][n_chamber - 1] == 0) {
                        bCSC_box[n_side - 1][station - 1][n_ring - 1][n_chamber - 1] = new TBox(x_min_chamber + 1, y_min_chamber, x_max_chamber + 1, y_max_chamber);
                        bCSC_box[n_side - 1][station - 1][n_ring - 1][n_chamber - 1]->SetLineColor(1);
                        bCSC_box[n_side - 1][station - 1][n_ring - 1][n_chamber - 1]->SetLineStyle(2);
                    }
                    bCSC_box[n_side - 1][station - 1][n_ring - 1][n_chamber - 1]->SetFillColor(fillColor);
                    bCSC_box[n_side - 1][station - 1][n_ring - 1][n_chamber - 1]->Draw("l");

                    if (tCSC_label[n_side - 1][station - 1][n_ring - 1][n_chamber - 1] == 0) {
                        TString ChamberID = Form("%d", n_chamber);
                        tCSC_label[n_side - 1][station - 1][n_ring - 1][n_chamber - 1] = new TText((x_min_chamber + x_max_chamber) / 2.0 + 1, (y_min_chamber + y_max_chamber) / 2.0, ChamberID);
                        tCSC_label[n_side - 1][station - 1][n_ring - 1][n_chamber - 1]->SetTextAlign(22);
                        tCSC_label[n_side - 1][station - 1][n_ring - 1][n_chamber - 1]->SetTextFont(42);
                        tCSC_label[n_side - 1][station - 1][n_ring - 1][n_chamber - 1]->SetTextSize(0.015);
                    }
                    tCSC_label[n_side - 1][station - 1][n_ring - 1][n_chamber - 1]->Draw();

                }
            }
        }
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

        if (legend.test(0)) printLegendBox(legendBoxIndex++, "OK/No Data", COLOR_WHITE);
        if (legend.test(1)) printLegendBox(legendBoxIndex++, "OK/Data", COLOR_GREEN);
        if (legend.test(2)) printLegendBox(legendBoxIndex++, "Error/Hot", COLOR_RED);
        if (legend.test(3)) printLegendBox(legendBoxIndex++, "Cold", COLOR_BLUE);
        if (legend.test(4)) printLegendBox(legendBoxIndex++, "Masked", COLOR_GREY);
        if (legend.test(5)) printLegendBox(legendBoxIndex++, "Standby", COLOR_YELLOW);
    }

}

void ChamberMap::printLegendBox(const unsigned int& number, const std::string title, int color) {

    if (bLegend[number] == 0) {
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
    tLegend[number]->Draw();

}
