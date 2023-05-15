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
#include "CondCore/SiPixelPlugins/interface/SiPixelTemplateHelper.h"
#include "DQM/TrackerRemapper/interface/Phase1PixelMaps.h"
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
#include "TGaxis.h"

namespace {

  using namespace cond::payloadInspector;

  /************************************************
    test class
  *************************************************/
  class SiPixelTemplateDBObjectTest : public Histogram1D<SiPixelTemplateDBObject, SINGLE_IOV> {
  public:
    SiPixelTemplateDBObjectTest()
        : Histogram1D<SiPixelTemplateDBObject, SINGLE_IOV>(
              "SiPixelTemplateDBObject test", "SiPixelTemplateDBObject test", 10, 0.0, 100.) {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::vector<SiPixelTemplateStore> thePixelTemp_;
        std::shared_ptr<SiPixelTemplateDBObject> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          if (!SiPixelTemplate::pushfile(*payload, thePixelTemp_)) {
            throw cms::Exception("SiPixelTemplateDBObject_PayloadInspector")
                << "\nERROR: Templates not filled correctly. Check the conditions. Using "
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
  template <SiPixelPI::DetType myType>
  class SiPixelTemplateLA : public PlotImage<SiPixelTemplateDBObject, SINGLE_IOV> {
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
    SiPixelTemplateLA() : PlotImage<SiPixelTemplateDBObject, SINGLE_IOV>("SiPixelTemplate assumed value of uH") {}

    bool fill() override {
      gStyle->SetPalette(kRainBow);
      TGaxis::SetMaxDigits(2);

      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::vector<SiPixelTemplateStore> thePixelTemp_;
      std::shared_ptr<SiPixelTemplateDBObject> payload = fetchPayload(std::get<1>(iov));

      if (payload.get()) {
        if (!SiPixelTemplate::pushfile(*payload, thePixelTemp_)) {
          throw cms::Exception("SiPixelTemplateDBObject_PayloadInspector")
              << "\nERROR: Templates not filled correctly. Check the conditions. Using "
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
        Phase1PixelMaps theMaps("");
        if (myType == SiPixelPI::t_all) {
          theMaps.resetOption("COLZA L");
        } else {
          theMaps.resetOption("COLZL");
        }

        if (myType == SiPixelPI::t_barrel) {
          theMaps.bookBarrelHistograms("templateLABarrel", "#muH", "#mu_{H} [1/T]");
        } else if (myType == SiPixelPI::t_forward) {
          theMaps.bookForwardHistograms("templateLAForward", "#muH", "#mu_{H} [1/T]");
        } else if (myType == SiPixelPI::t_all) {
          theMaps.bookBarrelHistograms("templateLA", "#muH", "#mu_{H} [1/T]");
          theMaps.bookForwardHistograms("templateLA", "#muH", "#mu_{H} [1/T]");
        } else {
          edm::LogError("SiPixelTemplateDBObject_PayloadInspector")
              << " un-recognized detector type " << myType << std::endl;
          return false;
        }

        std::map<unsigned int, short> templMap = payload->getTemplateIDs();
        if (templMap.size() == SiPixelPI::phase0size || templMap.size() > SiPixelPI::phase1size) {
          edm::LogError("SiPixelTemplateDBObject_PayloadInspector")
              << "There are " << templMap.size()
              << " DetIds in this payload. SiPixelTempate Lorentz Angle maps are not supported for non-Phase1 Pixel "
                 "geometries !";
          TCanvas canvas("Canv", "Canv", 1200, 1000);
          SiPixelPI::displayNotSupported(canvas, templMap.size());
          std::string fileName(m_imageFileName);
          canvas.SaveAs(fileName.c_str());
          return false;
        } else {
          if (templMap.size() < SiPixelPI::phase1size) {
            edm::LogWarning("SiPixelTemplateDBObject_PayloadInspector")
                << "\n ********* WARNING! ********* \n There are " << templMap.size() << " DetIds in this payload !"
                << "\n **************************** \n";
          }
        }

        for (auto const& entry : templMap) {
          templ.interpolate(entry.second, 0.f, 0.f, 1.f, 1.f);

          //mu_H = lorentz width / sensor thickness / B field
          float uH = templ.lorxwidth() / theInfos[entry.second].zsize / theInfos[entry.second].Bfield;
          COUT << "uH: " << uH << " lor x width:" << templ.lorxwidth() << " z size: " << theInfos[entry.second].zsize
               << " B-field: " << theInfos[entry.second].Bfield << std::endl;

          auto detid = DetId(entry.first);
          if (myType == SiPixelPI::t_all) {
            if ((detid.subdetId() == PixelSubdetector::PixelBarrel)) {
              theMaps.fillBarrelBin("templateLA", entry.first, uH);
            }
            if ((detid.subdetId() == PixelSubdetector::PixelEndcap)) {
              theMaps.fillForwardBin("templateLA", entry.first, uH);
            }
          } else if ((detid.subdetId() == PixelSubdetector::PixelBarrel) && (myType == SiPixelPI::t_barrel)) {
            theMaps.fillBarrelBin("templateLABarrel", entry.first, uH);
          } else if ((detid.subdetId() == PixelSubdetector::PixelEndcap) && (myType == SiPixelPI::t_forward)) {
            theMaps.fillForwardBin("templateLAForward", entry.first, uH);
          }
        }

        theMaps.beautifyAllHistograms();

        TCanvas canvas("Canv", "Canv", (myType == SiPixelPI::t_barrel) ? 1200 : 1600, 1000);
        if (myType == SiPixelPI::t_barrel) {
          theMaps.drawBarrelMaps("templateLABarrel", canvas);
        } else if (myType == SiPixelPI::t_forward) {
          theMaps.drawForwardMaps("templateLAForward", canvas);
        } else if (myType == SiPixelPI::t_all) {
          theMaps.drawSummaryMaps("templateLA", canvas);
        }

        canvas.cd();
        std::string fileName(m_imageFileName);
        canvas.SaveAs(fileName.c_str());
      }
      return true;
    }  // fill
  };

  using SiPixelTemplateLABPixMap = SiPixelTemplateLA<SiPixelPI::t_barrel>;
  using SiPixelTemplateLAFPixMap = SiPixelTemplateLA<SiPixelPI::t_forward>;
  using SiPixelTemplateLAMap = SiPixelTemplateLA<SiPixelPI::t_all>;

  using namespace templateHelper;

  //************************************************
  // Full Pixel Tracker Map of Template IDs
  // ***********************************************/
  using SiPixelTemplateIDsFullPixelMap =
      SiPixelFullPixelIDMap<SiPixelTemplateDBObject, SiPixelTemplateStore, SiPixelTemplate>;

  //************************************************
  // Display of Template Titles
  // **********************************************/
  using SiPixelTemplateTitles_Display =
      SiPixelTitles_Display<SiPixelTemplateDBObject, SiPixelTemplateStore, SiPixelTemplate>;

  //***********************************************
  // Display of Template Header
  // **********************************************/
  using SiPixelTemplateHeaderTable = SiPixelHeaderTable<SiPixelTemplateDBObject, SiPixelTemplateStore, SiPixelTemplate>;

  //***********************************************
  // TH2Poly Map of IDs
  //***********************************************/
  using SiPixelTemplateIDsBPixMap = SiPixelIDs<SiPixelTemplateDBObject, SiPixelPI::t_barrel>;
  using SiPixelTemplateIDsFPixMap = SiPixelIDs<SiPixelTemplateDBObject, SiPixelPI::t_forward>;
  using SiPixelTemplateIDsMap = SiPixelIDs<SiPixelTemplateDBObject, SiPixelPI::t_all>;

  //************************************************
  // TH2Poly Map of qScale
  //***********************************************/
  using SiPixelTemplateQScaleBPixMap = SiPixelTemplateHeaderInfo<SiPixelTemplateDBObject,
                                                                 SiPixelTemplateStore,
                                                                 SiPixelTemplate,
                                                                 SiPixelPI::t_barrel,
                                                                 headerParam::k_qscale>;
  using SiPixelTemplateQScaleFPixMap = SiPixelTemplateHeaderInfo<SiPixelTemplateDBObject,
                                                                 SiPixelTemplateStore,
                                                                 SiPixelTemplate,
                                                                 SiPixelPI::t_forward,
                                                                 headerParam::k_qscale>;
  using SiPixelTemplateQScaleMap = SiPixelTemplateHeaderInfo<SiPixelTemplateDBObject,
                                                             SiPixelTemplateStore,
                                                             SiPixelTemplate,
                                                             SiPixelPI::t_all,
                                                             headerParam::k_qscale>;

  //************************************************
  // TH2Poly Map of Vbias
  //***********************************************/
  using SiPixelTemplateVbiasBPixMap = SiPixelTemplateHeaderInfo<SiPixelTemplateDBObject,
                                                                SiPixelTemplateStore,
                                                                SiPixelTemplate,
                                                                SiPixelPI::t_barrel,
                                                                headerParam::k_Vbias>;
  using SiPixelTemplateVbiasFPixMap = SiPixelTemplateHeaderInfo<SiPixelTemplateDBObject,
                                                                SiPixelTemplateStore,
                                                                SiPixelTemplate,
                                                                SiPixelPI::t_forward,
                                                                headerParam::k_Vbias>;
  using SiPixelTemplateVbiasMap = SiPixelTemplateHeaderInfo<SiPixelTemplateDBObject,
                                                            SiPixelTemplateStore,
                                                            SiPixelTemplate,
                                                            SiPixelPI::t_all,
                                                            headerParam::k_Vbias>;
}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(SiPixelTemplateDBObject) {
  PAYLOAD_INSPECTOR_CLASS(SiPixelTemplateDBObjectTest);
  PAYLOAD_INSPECTOR_CLASS(SiPixelTemplateIDsFullPixelMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelTemplateTitles_Display);
  PAYLOAD_INSPECTOR_CLASS(SiPixelTemplateHeaderTable);
  PAYLOAD_INSPECTOR_CLASS(SiPixelTemplateIDsMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelTemplateIDsBPixMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelTemplateIDsFPixMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelTemplateLAMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelTemplateLABPixMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelTemplateLAFPixMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelTemplateQScaleBPixMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelTemplateQScaleFPixMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelTemplateQScaleMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelTemplateVbiasBPixMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelTemplateVbiasFPixMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelTemplateVbiasMap);
}
