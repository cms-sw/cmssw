/*!
  \file SiPixelTemplateDBObject_PayloadInspector
  \Payload Inspector Plugin for SiPixelTemplateDBObject
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2020/04/16 18:00:00 $
*/

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/SiPixelPlugins/interface/SiPixelPayloadInspectorHelper.h"
#include "CondCore/SiPixelPlugins/interface/Phase1PixelMaps.h"
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
  using SiPixelGenErrorTitles_Display =
      SiPixelTitles_Display<SiPixelGenErrorDBObject, SiPixelGenErrorStore, SiPixelGenError>;

  //***********************************************
  // Display of GenError Header
  // **********************************************/
  using SiPixelGenErrorHeaderTable = SiPixelHeaderTable<SiPixelGenErrorDBObject, SiPixelGenErrorStore, SiPixelGenError>;

  //***********************************************
  // testing TH2Poly classes for plotting
  //***********************************************/
  template <SiPixelPI::DetType myType>
  class SiPixelGenErrorIDs
      : public cond::payloadInspector::PlotImage<SiPixelGenErrorDBObject, cond::payloadInspector::SINGLE_IOV> {
  public:
    SiPixelGenErrorIDs()
        : cond::payloadInspector::PlotImage<SiPixelGenErrorDBObject, cond::payloadInspector::SINGLE_IOV>(
              "SiPixelGenError ID Values") {}

    bool fill() override {
      gStyle->SetPalette(kRainBow);

      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<SiPixelGenErrorDBObject> payload = fetchPayload(std::get<1>(iov));

      if (payload.get()) {
        // Book the TH2Poly
        Phase1PixelMaps theMaps("text");
        if (myType == SiPixelPI::t_barrel) {
          theMaps.bookBarrelHistograms("generrorIDsBarrel", "genErrorIDs", "genError IDs");
          // book the barrel bins of the TH2Poly
          theMaps.bookBarrelBins("generrorIDsBarrel");
        } else if (myType == SiPixelPI::t_forward) {
          theMaps.bookForwardHistograms("generrorIDsForward", "genErrorIDs", "genError IDs");
          // book the forward bins of the TH2Poly
          theMaps.bookForwardBins("generrorIDsForward");
        }

        std::map<unsigned int, short> templMap = payload->getGenErrorIDs();

        if (templMap.size() == SiPixelPI::phase0size || templMap.size() > SiPixelPI::phase1size) {
          edm::LogError("SiPixelGenErrorDBObject_PayloadInspector")
              << "There are " << templMap.size()
              << " DetIds in this payload. SiPixelTempateIDs maps are not supported for non-Phase1 Pixel geometries !";
          TCanvas canvas("Canv", "Canv", 1200, 1000);
          SiPixelPI::displayNotSupported(canvas, templMap.size());
          std::string fileName(m_imageFileName);
          canvas.SaveAs(fileName.c_str());
          return false;
        } else {
          if (templMap.size() < SiPixelPI::phase1size) {
            edm::LogWarning("SiPixelGenErrorDBObject_PayloadInspector")
                << "\n ********* WARNING! ********* \n There are " << templMap.size() << " DetIds in this payload !"
                << "\n **************************** \n";
          }
        }

        for (auto const& entry : templMap) {
          COUT << "DetID: " << entry.first << " generror ID: " << entry.second << std::endl;
          auto detid = DetId(entry.first);
          if ((detid.subdetId() == PixelSubdetector::PixelBarrel) && (myType == SiPixelPI::t_barrel)) {
            theMaps.fillBarrelBin("generrorIDsBarrel", entry.first, entry.second);
          } else if ((detid.subdetId() == PixelSubdetector::PixelEndcap) && (myType == SiPixelPI::t_forward)) {
            theMaps.fillForwardBin("generrorIDsForward", entry.first, entry.second);
          }
        }

        theMaps.beautifyAllHistograms();

        TCanvas canvas("Canv", "Canv", (myType == SiPixelPI::t_barrel) ? 1200 : 1500, 1000);
        if (myType == SiPixelPI::t_barrel) {
          theMaps.DrawBarrelMaps("generrorIDsBarrel", canvas);
        } else if (myType == SiPixelPI::t_forward) {
          theMaps.DrawForwardMaps("generrorIDsForward", canvas);
        }

        canvas.cd();

        std::string fileName(m_imageFileName);
        canvas.SaveAs(fileName.c_str());
      }
      return true;
    }
  };

  using SiPixelGenErrorIDsBPixMap = SiPixelGenErrorIDs<SiPixelPI::t_barrel>;
  using SiPixelGenErrorIDsFPixMap = SiPixelGenErrorIDs<SiPixelPI::t_forward>;

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(SiPixelGenErrorDBObject) {
  PAYLOAD_INSPECTOR_CLASS(SiPixelGenErrorTitles_Display);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGenErrorHeaderTable);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGenErrorIDsBPixMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGenErrorIDsFPixMap);
}
