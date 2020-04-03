#ifndef CSCRenderPlugin_ChamberMap_H
#define CSCRenderPlugin_ChamberMap_H

/*
 * =====================================================================================
 *
 *       Filename:  CSCRenderPlugin_ChamberMap.h
 *
 *    Description:  CSC Histogram Rendering Plugin
 *
 *        Version:  1.0
 *        Created:  05/06/2008 03:50:48 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), Valdas.Rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

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

/**
 * @class ChamberMap
 * @brief Class that draws CSC Map diagram
 */
class ChamberMap {

  private:

    static const unsigned short COLOR_WHITE  = 0;
    static const unsigned short COLOR_GREEN  = 3;
    static const unsigned short COLOR_RED    = 2;
    static const unsigned short COLOR_BLUE   = 4;
    static const unsigned short COLOR_GREY   = 17;
    static const unsigned short COLOR_YELLOW = 5;

    TBox*  bBlank;
    TBox*  bCSC_box[2][4][3][36];
    TText* tCSC_label[2][4][3][36];
    TBox*  bLegend[10];
    TText* tLegend[10];
    TText* tStatusTitle;
    TText* tLegendTitle;

  public:

    ChamberMap();
    ~ChamberMap();
    void draw(TH2*& me);
    void drawStats(TH2*& me);

  private:

    void printLegendBox(const unsigned int& number, const std::string title, int color);
    float Xmin_local_derived_from_ChamberID(int side, int station, int ring, int chamber) const;
    float Xmax_local_derived_from_ChamberID(int side, int station, int ring, int chamber) const;
    float Ymin_local_derived_from_ChamberID(int side, int station, int ring, int chamber) const;
    float Ymax_local_derived_from_ChamberID(int side, int station, int ring, int chamber) const;
    int N_ring(int station) const;
    int N_chamber(int station, int ring) const;

};

#endif
