#ifndef CSCRenderPlugin_SummaryMap_H
#define CSCRenderPlugin_SummaryMap_H

/*
 * =====================================================================================
 *
 *       Filename:  CSCRenderPlugin_SummaryMap.h
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

#define CSC_RENDER_PLUGIN

#ifdef DQMLOCAL
#include "DQM/CSCMonitorModule/interface/CSCDQM_Detector.h"
#else
#include "CSCDQM_Detector.h"
#endif

#define N_TICS    100

class TH2;
class TH2F;
class TBox;
class TLine;
class TText;

/**
 * @class SummaryMap
 * @brief Class that draws CSC Summary Map diagram
 */
class SummaryMap {

  public:

    SummaryMap();
    ~SummaryMap();
    void drawDetector(TH2* me);
    void drawStation(TH2* me, const int station);

  private:

    cscdqm::Detector detector;

    TBox  *bDetector[N_TICS][N_TICS];
    TLine *lDetector[N_TICS - 1][2];
    TText *tDetector;

    TBox *bStation[4][N_ELEMENTS];
    TLine *lStation[4][3456];
    TText *tStationCSC_label[4][864];
    TText *tStationRing_label[4][6];
    TText *tStation_minus_label;
    TText *tStation_plus_label;
    TText *tStation_title;

    TBox* bEmptyPlus;
    TH2*  h1;
    TBox* bBlank;
    TBox* bEmptyMinus;
};

#endif
