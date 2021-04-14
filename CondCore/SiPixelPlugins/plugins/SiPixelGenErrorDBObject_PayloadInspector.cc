/*!
  \file SiPixelGenErrorDBObject_PayloadInspector
  \Payload Inspector Plugin for SiPixelGenError
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
#include "CondFormats/SiPixelObjects/interface/SiPixelGenErrorDBObject.h"
#include "CondFormats/SiPixelTransient/interface/SiPixelGenError.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <memory>
#include <map>
#include <sstream>
#include <iostream>
#include <algorithm>

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
  using SiPixelGenErrorTitles_Display =
      SiPixelTitles_Display<SiPixelGenErrorDBObject, SiPixelGenErrorStore, SiPixelGenError>;

  //***********************************************
  // Display of GenError Header
  // **********************************************/
  using SiPixelGenErrorHeaderTable = SiPixelHeaderTable<SiPixelGenErrorDBObject, SiPixelGenErrorStore, SiPixelGenError>;

  //***********************************************
  // TH2Poly Map of IDs
  //***********************************************/
  using SiPixelGenErrorIDsBPixMap = SiPixelIDs<SiPixelGenErrorDBObject, SiPixelPI::t_barrel>;
  using SiPixelGenErrorIDsFPixMap = SiPixelIDs<SiPixelGenErrorDBObject, SiPixelPI::t_forward>;
  using SiPixelGenErrorIDsMap = SiPixelIDs<SiPixelGenErrorDBObject, SiPixelPI::t_all>;

  //************************************************
  // Full Pixel Tracker Map of Template IDs
  // ***********************************************/
  using SiPixelGenErrorIDsFullPixelMap =
      SiPixelFullPixelIDMap<SiPixelGenErrorDBObject, SiPixelGenErrorStore, SiPixelGenError>;

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(SiPixelGenErrorDBObject) {
  PAYLOAD_INSPECTOR_CLASS(SiPixelGenErrorTitles_Display);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGenErrorHeaderTable);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGenErrorIDsBPixMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGenErrorIDsFPixMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGenErrorIDsMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGenErrorIDsFullPixelMap);
}
