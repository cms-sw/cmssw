/*!
  \file SiPixel2DTemplateDBObject_PayloadInspector
  \Payload Inspector Plugin for SiPixel2DTemplateDBObject
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2020/04/16 18:00:00 $
*/

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/SiPixelPlugins/interface/SiPixelPayloadInspectorHelper.h"
#include "CondCore/SiPixelPlugins/interface/SiPixelTemplateHelper.h"

#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"

// the data format of the condition to be inspected
#include "CondFormats/SiPixelObjects/interface/SiPixel2DTemplateDBObject.h"
#include "CondFormats/SiPixelTransient/interface/SiPixelTemplate2D.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <memory>
#include <map>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <boost/range/adaptor/indexed.hpp>

// include ROOT
#include "TH2.h"
#include "TProfile2D.h"
#include "TH2Poly.h"
#include "TGraph.h"
#include "TH2F.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TGraph.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPave.h"
#include "TPaveStats.h"
#include "TGaxis.h"

namespace {

  //***********************************************
  // Display of Template Titles
  // **********************************************/
  using namespace templateHelper;
  using SiPixel2DTemplateTitles_Display =
      SiPixelTitles_Display<SiPixel2DTemplateDBObject, SiPixelTemplateStore2D, SiPixelTemplate2D>;

  //***********************************************
  // Display of 2DTemplate Header
  // **********************************************/
  using SiPixel2DTemplateHeaderTable =
      SiPixelHeaderTable<SiPixel2DTemplateDBObject, SiPixelTemplateStore2D, SiPixelTemplate2D>;

  //***********************************************
  // TH2Poly Map of IDs
  //***********************************************/
  using SiPixel2DTemplateIDsBPixMap = SiPixelIDs<SiPixel2DTemplateDBObject, SiPixelPI::t_barrel>;
  using SiPixel2DTemplateIDsFPixMap = SiPixelIDs<SiPixel2DTemplateDBObject, SiPixelPI::t_forward>;

  //************************************************
  // Full Pixel Tracker Map of Template IDs
  // ***********************************************/
  using SiPixel2DTemplateIDsFullPixelMap =
      SiPixelFullPixelIDMap<SiPixel2DTemplateDBObject, SiPixelTemplateStore2D, SiPixelTemplate2D>;

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(SiPixel2DTemplateDBObject) {
  PAYLOAD_INSPECTOR_CLASS(SiPixel2DTemplateTitles_Display);
  PAYLOAD_INSPECTOR_CLASS(SiPixel2DTemplateHeaderTable);
  PAYLOAD_INSPECTOR_CLASS(SiPixel2DTemplateIDsBPixMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixel2DTemplateIDsFPixMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixel2DTemplateIDsFullPixelMap);
}
