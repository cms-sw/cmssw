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

#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"

// the data format of the condition to be inspected
#include "CondFormats/SiPixelObjects/interface/SiPixelTemplateDBObject.h"
#include "CondFormats/SiPixelTransient/interface/SiPixelTemplate.h"
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

namespace {

  enum MapType { t_barrel = 0, t_forward = 1 };

  /************************************************
    test class
  *************************************************/

  class SiPixelTemplateDBObjectTest : public cond::payloadInspector::Histogram1D<SiPixelTemplateDBObject> {
  public:
    SiPixelTemplateDBObjectTest()
        : cond::payloadInspector::Histogram1D<SiPixelTemplateDBObject>(
              "SiPixelTemplateDBObject test", "SiPixelTemplateDBObject test", 10, 0.0, 100.) {
      Base::setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash>>& iovs) override {
      for (auto const& iov : iovs) {
        std::vector<SiPixelTemplateStore> thePixelTemp_;

        std::shared_ptr<SiPixelTemplateDBObject> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          if (!SiPixelTemplate::pushfile(*payload, thePixelTemp_)) {
            throw cms::Exception("") << "\nERROR: Templates not filled correctly. Check the sqlite file. Using "
                                        "SiPixelTemplateDBObject version "
                                     << payload->version() << "\n\n";
          }

          SiPixelTemplate templ(thePixelTemp_);

          for (const auto& theTemp : thePixelTemp_) {
            std::cout << "\n\n"
                      << "Template ID = " << theTemp.head.ID << ", Template Version " << theTemp.head.templ_version
                      << ", Bfield = " << theTemp.head.Bfield << ", NTy = " << theTemp.head.NTy
                      << ", NTyx = " << theTemp.head.NTyx << ", NTxx = " << theTemp.head.NTxx
                      << ", Dtype = " << theTemp.head.Dtype << ", Bias voltage " << theTemp.head.Vbias
                      << ", temperature " << theTemp.head.temperature << ", fluence " << theTemp.head.fluence
                      << ", Q-scaling factor " << theTemp.head.qscale << ", 1/2 multi dcol threshold "
                      << theTemp.head.s50 << ", 1/2 single dcol threshold " << theTemp.head.ss50 << ", y Lorentz Width "
                      << theTemp.head.lorywidth << ", y Lorentz Bias " << theTemp.head.lorybias << ", x Lorentz width "
                      << theTemp.head.lorxwidth << ", x Lorentz Bias " << theTemp.head.lorxbias
                      << ", Q/Q_avg fractions for Qbin defs " << theTemp.head.fbin[0] << ", " << theTemp.head.fbin[1]
                      << ", " << theTemp.head.fbin[2] << ", pixel x-size " << theTemp.head.xsize << ", y-size "
                      << theTemp.head.ysize << ", zsize " << theTemp.head.zsize << "\n"
                      << std::endl;
          }

          std::map<unsigned int, short> templMap = payload->getTemplateIDs();
          for (auto const& entry : templMap) {
            std::cout << "DetID: " << entry.first << " template ID: " << entry.second << std::endl;
            templ.interpolate(entry.second, 0.f, 0.f, 1.f, 1.f);

            std::cout << "\t lorywidth  " << templ.lorywidth() << " lorxwidth: " << templ.lorxwidth() << " lorybias "
                      << templ.lorybias() << " lorxbias: " << templ.lorxbias() << "\n"
                      << std::endl;
          }

          fillWithValue(1.);

        }  // payload
      }    // iovs
      return true;
    }  // fill
  };

  /************************************************
  // testing TH2Poly classes for plotting
  *************************************************/
  template <MapType myType>
  class SiPixelTemplateLA : public cond::payloadInspector::PlotImage<SiPixelTemplateDBObject> {
    struct header_info {
      int ID;             //!< template ID number
      float lorywidth;    //!< estimate of y-lorentz width for optimal resolution
      float lorxwidth;    //!< estimate of x-lorentz width for optimal resolution
      float lorybias;     //!< estimate of y-lorentz bias
      float lorxbias;     //!< estimate of x-lorentz bias
      float Vbias;        //!< detector bias potential in Volts
      float temperature;  //!< detector temperature in deg K
      int templ_version;  //!< Version number of the template to ensure code compatibility
      float Bfield;       //!< Bfield in Tesla
      float xsize;        //!< pixel size (for future use in upgraded geometry)
      float ysize;        //!< pixel size (for future use in upgraded geometry)
      float zsize;        //!< pixel size (for future use in upgraded geometry)
    };

  public:
    SiPixelTemplateLA()
        : cond::payloadInspector::PlotImage<SiPixelTemplateDBObject>("SiPixelTemplate assumed value of uH") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash>>& iovs) override {
      //gStyle->SetPalette(kRainBow);
      auto iov = iovs.front();

      std::vector<SiPixelTemplateStore> thePixelTemp_;
      std::shared_ptr<SiPixelTemplateDBObject> payload = fetchPayload(std::get<1>(iov));

      if (payload.get()) {
        if (!SiPixelTemplate::pushfile(*payload, thePixelTemp_)) {
          throw cms::Exception("") << "\nERROR: Templates not filled correctly. Check the sqlite file. Using "
                                      "SiPixelTemplateDBObject version "
                                   << payload->version() << "\n\n";
        }

        // store the map of ID / interesting quantities
        SiPixelTemplate templ(thePixelTemp_);
        std::map<int, header_info> theInfos;
        for (const auto& theTemp : thePixelTemp_) {
          header_info info;
          info.ID = theTemp.head.ID;
          info.lorywidth = theTemp.head.lorywidth;
          info.lorxwidth = theTemp.head.lorxwidth;
          info.lorybias = theTemp.head.lorybias;
          info.lorxbias = theTemp.head.lorxbias;
          info.Vbias = theTemp.head.Vbias;
          info.temperature = theTemp.head.temperature;
          info.templ_version = theTemp.head.templ_version;
          info.Bfield = theTemp.head.Bfield;
          info.xsize = theTemp.head.xsize;
          info.ysize = theTemp.head.ysize;
          info.zsize = theTemp.head.zsize;

          theInfos[theTemp.head.ID] = info;
        }

        // Book the TH2Poly
        Phase1PixelMaps theMaps("COLZ L");

        if (myType == t_barrel) {
          theMaps.bookBarrelHistograms("templateLABarrel");
          theMaps.bookBarrelBins("templateLABarrel");
        } else if (myType == t_forward) {
          theMaps.bookForwardHistograms("templateLAForward");
          theMaps.bookForwardBins("templateLAForward");
        }

        std::map<unsigned int, short> templMap = payload->getTemplateIDs();
        if (templMap.size() != SiPixelPI::phase1size) {
          edm::LogError("SiPixelTemplateDBObject_PayloadInspector")
              << "SiPixelTempateLA maps are not supported for non-Phase1 Pixel geometries !";
          std::string phase = (templMap.size() < SiPixelPI::phase1size) ? "Phase-0" : "Phase-2";
          TCanvas canvas("Canv", "Canv", 1200, 1000);
          canvas.cd();
          TLatex t2;
          t2.SetTextAlign(21);
          t2.SetTextSize(0.1);
          t2.SetTextAngle(45);
          t2.SetTextColor(kRed);
          t2.DrawLatexNDC(0.6, 0.50, Form("%s NOT SUPPORTED!", phase.c_str()));
          std::string fileName(m_imageFileName);
          canvas.SaveAs(fileName.c_str());
          return false;
        }

        for (auto const& entry : templMap) {
          templ.interpolate(entry.second, 0.f, 0.f, 1.f, 1.f);

          //mu_H = lorentz width / sensor thickness / B field
          float uH = templ.lorxwidth() / theInfos[entry.second].zsize / theInfos[entry.second].Bfield;
          COUT << "uH: " << uH << " lor x width:" << templ.lorxwidth() << " z size: " << theInfos[entry.second].zsize
               << " B-field: " << theInfos[entry.second].Bfield << std::endl;

          auto detid = DetId(entry.first);
          if ((detid.subdetId() == PixelSubdetector::PixelBarrel) && (myType == t_barrel)) {
            theMaps.fillBarrelBin("templateLABarrel", entry.first, uH);
          } else if ((detid.subdetId() == PixelSubdetector::PixelEndcap) && (myType == t_forward)) {
            theMaps.fillForwardBin("templateLAForward", entry.first, uH);
          }
        }

        theMaps.beautifyAllHistograms();

        TCanvas canvas("Canv", "Canv", (myType == t_barrel) ? 1200 : 1500, 1000);
        if (myType == t_barrel) {
          theMaps.DrawBarrelMaps("templateLABarrel", canvas);
        } else if (myType == t_forward) {
          theMaps.DrawForwardMaps("templateLAForward", canvas);
        }

        canvas.cd();
        std::string fileName(m_imageFileName);
        canvas.SaveAs(fileName.c_str());
      }
      return true;
    }  // fill
  };

  using SiPixelTemplateLABPixMap = SiPixelTemplateLA<t_barrel>;
  using SiPixelTemplateLAFPixMap = SiPixelTemplateLA<t_forward>;

  /************************************************
  // testing TH2Poly classes for plotting
  *************************************************/
  template <MapType myType>
  class SiPixelTemplateIDs : public cond::payloadInspector::PlotImage<SiPixelTemplateDBObject> {
  public:
    SiPixelTemplateIDs() : cond::payloadInspector::PlotImage<SiPixelTemplateDBObject>("SiPixelTemplate ID Values") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash>>& iovs) override {
      gStyle->SetPalette(kRainBow);
      auto iov = iovs.front();
      std::shared_ptr<SiPixelTemplateDBObject> payload = fetchPayload(std::get<1>(iov));

      if (payload.get()) {
        // Book the TH2Poly
        Phase1PixelMaps theMaps("text");
        if (myType == t_barrel) {
          theMaps.bookBarrelHistograms("templateIDsBarrel");
          // book the barrel bins of the TH2Poly
          theMaps.bookBarrelBins("templateIDsBarrel");
        } else if (myType == t_forward) {
          theMaps.bookForwardHistograms("templateIDsForward");
          // book the forward bins of the TH2Poly
          theMaps.bookForwardBins("templateIDsForward");
        }

        std::map<unsigned int, short> templMap = payload->getTemplateIDs();
        if (templMap.size() != SiPixelPI::phase1size) {
          edm::LogError("SiPixelTemplateDBObject_PayloadInspector")
              << "SiPixelTempateIDs maps are not supported for non-Phase1 Pixel geometries !";
          std::string phase = (templMap.size() < SiPixelPI::phase1size) ? "Phase-0" : "Phase-2";
          TCanvas canvas("Canv", "Canv", 1200, 1000);
          canvas.cd();
          TLatex t2;
          t2.SetTextAlign(21);
          t2.SetTextSize(0.1);
          t2.SetTextAngle(45);
          t2.SetTextColor(kRed);
          t2.DrawLatexNDC(0.6, 0.50, Form("%s  NOT SUPPORTED!", phase.c_str()));
          std::string fileName(m_imageFileName);
          canvas.SaveAs(fileName.c_str());
          return false;
        }

        /*
        std::vector<unsigned int> detids;
        std::transform(templMap.begin(),
                       templMap.end(),
                       std::back_inserter(detids),
                       [](const std::map<unsigned int, short>::value_type& pair) { return pair.first; });
	*/

        for (auto const& entry : templMap) {
          COUT << "DetID: " << entry.first << " template ID: " << entry.second << std::endl;
          auto detid = DetId(entry.first);
          if ((detid.subdetId() == PixelSubdetector::PixelBarrel) && (myType == t_barrel)) {
            theMaps.fillBarrelBin("templateIDsBarrel", entry.first, entry.second);
          } else if ((detid.subdetId() == PixelSubdetector::PixelEndcap) && (myType == t_forward)) {
            theMaps.fillForwardBin("templateIDsForward", entry.first, entry.second);
          }
        }

        theMaps.beautifyAllHistograms();

        TCanvas canvas("Canv", "Canv", (myType == t_barrel) ? 1200 : 1500, 1000);
        if (myType == t_barrel) {
          theMaps.DrawBarrelMaps("templateIDsBarrel", canvas);
        } else if (myType == t_forward) {
          theMaps.DrawForwardMaps("templateIDsForward", canvas);
        }

        canvas.cd();

        std::string fileName(m_imageFileName);
        canvas.SaveAs(fileName.c_str());
      }
      return true;
    }
  };

  using SiPixelTemplateIDsBPixMap = SiPixelTemplateIDs<t_barrel>;
  using SiPixelTemplateIDsFPixMap = SiPixelTemplateIDs<t_forward>;

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(SiPixelTemplateDBObject) {
  PAYLOAD_INSPECTOR_CLASS(SiPixelTemplateDBObjectTest);
  PAYLOAD_INSPECTOR_CLASS(SiPixelTemplateIDsBPixMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelTemplateIDsFPixMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelTemplateLABPixMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelTemplateLAFPixMap);
}
