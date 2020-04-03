#ifndef CSCRenderPlugin_EventDisplay_H
#define CSCRenderPlugin_EventDisplay_H

/*
 * =====================================================================================
 *
 *       Filename:  CSCRenderPlugin_EventDisplay.h
 *
 *    Description:  CSC Histogram Rendering Plugin
 *
 *        Version:  1.0
 *        Created:  12/12/2008 07:50:48 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), Valdas.Rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#include <string>
#include <iostream>
#include <TH1.h>
#include <TH2.h>
#include <TBox.h>
#include <TText.h>
#include <TPRegexp.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TExec.h>
#include <TColor.h>

/**
 * @class EventDisplay
 * @brief Class that draws CSC Map diagram
 */
class EventDisplay {

  private:

    static const unsigned int   HISTO_WIDTH = 225;

    TPad* pad0;
    TPad* pad1;
    TPad* pad2;
    TPad* pad3;
    TPad* pad4;

    TH2F* histos[3];

    TText* tTitle;
    TText* tLayer;
    TText* tYLabel[6];
    TText* tXTitle[3];
    TText* tXLabel[3][224];
    TBox*  bBlank;
    TBox*  bBox[3][6][224];
    TBox*  bKey[3][224];
    TText* tKey[3][224];

    TExec* greyscaleExec;
    TExec* normalExec;

  public:

    EventDisplay();
    ~EventDisplay();
    void drawSingleChamber(TH2*& data);

  private:

    int countWiregroups(int station, int ring) const;
    int countStrips(int station, int ring) const;
    int countStripsNose(int station, int ring) const;
    void drawEventDisplayGrid(int hnum, TH2* data, int data_first_col, int data_time_col, int data_quality_col,
                              int count_x, float shift_x, float min_z, float max_z, int split_after_x, int time_corr, int d_corr,
                              const char* title_x, bool greyscale);

};

#endif
