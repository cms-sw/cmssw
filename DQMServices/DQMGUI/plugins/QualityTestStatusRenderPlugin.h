#ifndef DQM_RenderPlugins_QualityTestStatusRenderPlugin_H
#define DQM_RenderPlugins_QualityTestStatusRenderPlugin_H

/**
 * \class QualityTestStatusRenderPlugin
 *
 *
 * Description: render plugin for histograms filled with status of quality tests.
 *
 * Implementation:
 *
 *      Render histograms filled with status of quality tests. The status is defined in
 *      DQM/DQMDefinitions.h
 *
 *      Try to keep it compatible with utils.h from the same directory
 *
 *
 * \author: Vasile Mihai Ghete   - HEPHY Vienna
 *
 *
 * $Date$
 * $Revision$
 *
 */

// system include files

// user include files
class TH2;

namespace dqm {
    class QualityTestStatusRenderPlugin {

    public:
        static void reportSummaryMapPalette(TH2* obj);

    private:
        static bool init;

    };

}

#endif
