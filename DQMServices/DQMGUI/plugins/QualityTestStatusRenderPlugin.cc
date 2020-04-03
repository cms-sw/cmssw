/**
 * \class QualityTestStatusRenderPlugin
 *
 *
 * Description: see header file.
 *
 *
 * \author: Vasile Mihai Ghete   - HEPHY Vienna
 *
 *
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "QualityTestStatusRenderPlugin.h"

// system include files

// user include files
#include "TH2.h"
#include "TStyle.h"
#include "TColor.h"
#include "TROOT.h"

# include "DQM/DQMDefinitions.h"

void dqm::QualityTestStatusRenderPlugin::reportSummaryMapPalette(TH2* obj) {

    /**
     DQM/DQMDefinitions.h

     Numeric constants for quality test results.  The smaller the
     number, the less severe the message.

     namespace qstatus
     {
         static const int OTHER          =  30;  //< Anything but 'ok','warning' or 'error'.
         static const int DISABLED       =  50;  //< Test has been disabled.
         static const int INVALID        =  60;  //< Problem preventing test from running.
         static const int INSUF_STAT     =  70;  //< Insufficient statistics.
         static const int DID_NOT_RUN    =  90;  //< Algorithm did not run.
         static const int STATUS_OK      =  100; //< Test was successful.
         static const int WARNING        =  200; //< Test had some problems.
         static const int ERROR          =  300; //< Test has failed.
     }

     */

    // number of colors to be defined
    // corresponds to maximum value in histogram
    // for cell content N it will use palette[N] color
    const int colorNum = dqm::qstatus::ERROR;

    // custom color palette
    static int pallete[colorNum];

    if (!QualityTestStatusRenderPlugin::init) {

        QualityTestStatusRenderPlugin::init = true;

        // Red (index 0), Green (index 1), Blue (index 2) components for each colorNum
        // ROOT uses in SetRGB arithmetic representation (float from 0 to 1)
        float rgb[colorNum][3];

        // use fixed colors, no gradients
        for (int i = 0; i < colorNum; i++) {

            if (i <= dqm::qstatus::OTHER + (dqm::qstatus::DISABLED - dqm::qstatus::OTHER)/2) {
                // white
                rgb[i][0] = 1.0;
                rgb[i][1] = 1.0;
                rgb[i][2] = 1.0;
            } else if (i <= dqm::qstatus::DISABLED + (dqm::qstatus::INVALID - dqm::qstatus::DISABLED)/2) {
                // light grey (no utils color)
                rgb[i][0] = 0.83;
                rgb[i][1] = 0.83;
                rgb[i][2] = 0.83;
            } else if (i <= dqm::qstatus::INVALID + (dqm::qstatus::INSUF_STAT - dqm::qstatus::INVALID)/2) {
                // dark orange (no utils color)
                rgb[i][0] = 1.00;
                rgb[i][1] = 0.60;
                rgb[i][2] = 0.20;
            } else if (i <= dqm::qstatus::INSUF_STAT + (dqm::qstatus::DID_NOT_RUN - dqm::qstatus::INSUF_STAT)/2) {
                // light green (no utils color)
                rgb[i][0] = 0.80;
                rgb[i][1] = 1.0;
                rgb[i][2] = 0.80;
            } else if (i <= dqm::qstatus::DID_NOT_RUN + (dqm::qstatus::STATUS_OK - dqm::qstatus::DID_NOT_RUN)/2) {
                // light yellow (no utils color)
                rgb[i][0] = 1.00;
                rgb[i][1] = 1.00;
                rgb[i][2] = 0.80;
            } else if (i <= dqm::qstatus::STATUS_OK + (dqm::qstatus::WARNING - dqm::qstatus::STATUS_OK)/2) {
                // utils green
                rgb[i][0] = 0.00;
                rgb[i][1] = 0.80;
                rgb[i][2] = 0.00;
            } else if (i <= dqm::qstatus::WARNING + (dqm::qstatus::ERROR - dqm::qstatus::WARNING)/2) {
                // utils yellow
                rgb[i][0] = 0.98;
                rgb[i][1] = 0.79;
                rgb[i][2] = 0.00;
            } else if (i <= dqm::qstatus::ERROR + (dqm::qstatus::ERROR - dqm::qstatus::WARNING)/2) {
                // utils red
                rgb[i][0] = 0.80;
                rgb[i][1] = 0.00;
                rgb[i][2] = 0.00;
            } else {

                // do nothing, it should not arrive here
            }

            pallete[i] = TColor::GetColor(rgb[i][0], rgb[i][1], rgb[i][2]);
        }
    }

    gStyle->SetPalette(colorNum, pallete);

    if (obj) {
        obj->SetMinimum(-1.e-15);
        obj->SetMaximum(+300.0);

        // drawing option
        //  "COL"
        //    A box is drawn for each cell with a color scale varying with contents.
        //    All the none empty bins are painted.
        //    Empty bins are not painted unless some bins have a negative content because
        //    in that case the null bins might be not empty.
        //  "COLZ" = "COL" +
        //    The color palette is also drawn.
        obj->SetOption("colz");

    }
}

// initialization of static constants
bool dqm::QualityTestStatusRenderPlugin::init = false;
