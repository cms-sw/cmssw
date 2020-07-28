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

  /************************************************
    test class
  *************************************************/
  class SiPixelTemplateDBObjectTest
      : public cond::payloadInspector::Histogram1D<SiPixelTemplateDBObject, cond::payloadInspector::SINGLE_IOV> {
  public:
    SiPixelTemplateDBObjectTest()
        : cond::payloadInspector::Histogram1D<SiPixelTemplateDBObject, cond::payloadInspector::SINGLE_IOV>(
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
  // header plotting
  *************************************************/
  class SiPixelTemplateHeaderTable
      : public cond::payloadInspector::PlotImage<SiPixelTemplateDBObject, cond::payloadInspector::SINGLE_IOV> {
  public:
    SiPixelTemplateHeaderTable()
        : cond::payloadInspector::PlotImage<SiPixelTemplateDBObject, cond::payloadInspector::SINGLE_IOV>(
              "SiPixelTemplateDBObject Header summary") {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      auto tagname = tag.name;
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
        TCanvas canvas("Template Header Summary", "Template Header summary", 1400, 1000);
        canvas.cd();

        unsigned int tempSize = thePixelTemp_.size();

        canvas.SetTopMargin(0.07);
        canvas.SetBottomMargin(0.06);
        canvas.SetLeftMargin(0.17);
        canvas.SetRightMargin(0.03);
        canvas.Modified();
        canvas.SetGrid();

        auto h2_TemplateHeaders = std::make_unique<TH2F>("Header", ";;", tempSize, 0, tempSize, 6, 0., 6.);
        h2_TemplateHeaders->SetStats(false);

        for (const auto& theTemp : thePixelTemp_ | boost::adaptors::indexed(1)) {
          auto tempValue = theTemp.value();
          auto tempIndex = theTemp.index();
          float uH = roundoff(tempValue.head.lorxwidth / tempValue.head.zsize / tempValue.head.Bfield, 4);
          h2_TemplateHeaders->SetBinContent(tempIndex, 6, tempValue.head.ID);
          h2_TemplateHeaders->SetBinContent(tempIndex, 5, tempValue.head.Bfield);
          h2_TemplateHeaders->SetBinContent(tempIndex, 4, uH);
          h2_TemplateHeaders->SetBinContent(tempIndex, 3, tempValue.head.xsize);
          h2_TemplateHeaders->SetBinContent(tempIndex, 2, tempValue.head.ysize);
          h2_TemplateHeaders->SetBinContent(tempIndex, 1, tempValue.head.zsize);
          h2_TemplateHeaders->GetYaxis()->SetBinLabel(6, "TemplateID");
          h2_TemplateHeaders->GetYaxis()->SetBinLabel(5, "B-field [T]");
          h2_TemplateHeaders->GetYaxis()->SetBinLabel(4, "#mu_{H} [1/T]");
          h2_TemplateHeaders->GetYaxis()->SetBinLabel(3, "x-size [#mum]");
          h2_TemplateHeaders->GetYaxis()->SetBinLabel(2, "y-size [#mum]");
          h2_TemplateHeaders->GetYaxis()->SetBinLabel(1, "z-size [#mum]");
          h2_TemplateHeaders->GetXaxis()->SetBinLabel(tempIndex, "");
        }

        h2_TemplateHeaders->GetXaxis()->LabelsOption("h");
        h2_TemplateHeaders->GetXaxis()->SetNdivisions(500 + tempSize, false);
        h2_TemplateHeaders->GetYaxis()->SetLabelSize(0.05);
        h2_TemplateHeaders->SetMarkerSize(1.5);

        canvas.cd();
        h2_TemplateHeaders->Draw("TEXT");

        auto ltx = TLatex();
        ltx.SetTextFont(62);
        ltx.SetTextColor(kBlue);
        ltx.SetTextSize(0.045);
        ltx.SetTextAlign(11);
        ltx.DrawLatexNDC(gPad->GetLeftMargin(),
                         1 - gPad->GetTopMargin() + 0.01,
                         (tagname + ", IOV:" + std::to_string(std::get<0>(iov))).c_str());

        std::string fileName(m_imageFileName);
        canvas.SaveAs(fileName.c_str());
      }
      return true;
    }

    float roundoff(float value, unsigned char prec) {
      float pow_10 = pow(10.0f, (float)prec);
      return round(value * pow_10) / pow_10;
    }
  };

  /************************************************
  // testing TH2Poly classes for plotting
  *************************************************/
  template <SiPixelPI::DetType myType>
  class SiPixelTemplateLA
      : public cond::payloadInspector::PlotImage<SiPixelTemplateDBObject, cond::payloadInspector::SINGLE_IOV> {
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
        : cond::payloadInspector::PlotImage<SiPixelTemplateDBObject, cond::payloadInspector::SINGLE_IOV>(
              "SiPixelTemplate assumed value of uH") {}

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
        Phase1PixelMaps theMaps("COLZ L");

        if (myType == SiPixelPI::t_barrel) {
          theMaps.bookBarrelHistograms("templateLABarrel", "#muH", "#mu_{H} [1/T]");
          theMaps.bookBarrelBins("templateLABarrel");
        } else if (myType == SiPixelPI::t_forward) {
          theMaps.bookForwardHistograms("templateLAForward", "#muH", "#mu_{H} [1/T]");
          theMaps.bookForwardBins("templateLAForward");
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
          if ((detid.subdetId() == PixelSubdetector::PixelBarrel) && (myType == SiPixelPI::t_barrel)) {
            theMaps.fillBarrelBin("templateLABarrel", entry.first, uH);
          } else if ((detid.subdetId() == PixelSubdetector::PixelEndcap) && (myType == SiPixelPI::t_forward)) {
            theMaps.fillForwardBin("templateLAForward", entry.first, uH);
          }
        }

        theMaps.beautifyAllHistograms();

        TCanvas canvas("Canv", "Canv", (myType == SiPixelPI::t_barrel) ? 1200 : 1600, 1000);
        if (myType == SiPixelPI::t_barrel) {
          theMaps.DrawBarrelMaps("templateLABarrel", canvas);
        } else if (myType == SiPixelPI::t_forward) {
          theMaps.DrawForwardMaps("templateLAForward", canvas);
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

  /************************************************
  // testing TH2Poly classes for plotting
  *************************************************/
  template <SiPixelPI::DetType myType>
  class SiPixelTemplateIDs
      : public cond::payloadInspector::PlotImage<SiPixelTemplateDBObject, cond::payloadInspector::SINGLE_IOV> {
  public:
    SiPixelTemplateIDs()
        : cond::payloadInspector::PlotImage<SiPixelTemplateDBObject, cond::payloadInspector::SINGLE_IOV>(
              "SiPixelTemplate ID Values") {}

    bool fill() override {
      gStyle->SetPalette(kRainBow);

      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<SiPixelTemplateDBObject> payload = fetchPayload(std::get<1>(iov));

      if (payload.get()) {
        // Book the TH2Poly
        Phase1PixelMaps theMaps("text");
        if (myType == SiPixelPI::t_barrel) {
          theMaps.bookBarrelHistograms("templateIDsBarrel", "templateIDs", "template IDs");
          // book the barrel bins of the TH2Poly
          theMaps.bookBarrelBins("templateIDsBarrel");
        } else if (myType == SiPixelPI::t_forward) {
          theMaps.bookForwardHistograms("templateIDsForward", "templateIDs", "template IDs");
          // book the forward bins of the TH2Poly
          theMaps.bookForwardBins("templateIDsForward");
        }

        std::map<unsigned int, short> templMap = payload->getTemplateIDs();

        if (templMap.size() == SiPixelPI::phase0size || templMap.size() > SiPixelPI::phase1size) {
          edm::LogError("SiPixelTemplateDBObject_PayloadInspector")
              << "There are " << templMap.size()
              << " DetIds in this payload. SiPixelTempateIDs maps are not supported for non-Phase1 Pixel geometries !";
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
          if ((detid.subdetId() == PixelSubdetector::PixelBarrel) && (myType == SiPixelPI::t_barrel)) {
            theMaps.fillBarrelBin("templateIDsBarrel", entry.first, entry.second);
          } else if ((detid.subdetId() == PixelSubdetector::PixelEndcap) && (myType == SiPixelPI::t_forward)) {
            theMaps.fillForwardBin("templateIDsForward", entry.first, entry.second);
          }
        }

        theMaps.beautifyAllHistograms();

        TCanvas canvas("Canv", "Canv", (myType == SiPixelPI::t_barrel) ? 1200 : 1500, 1000);
        if (myType == SiPixelPI::t_barrel) {
          theMaps.DrawBarrelMaps("templateIDsBarrel", canvas);
        } else if (myType == SiPixelPI::t_forward) {
          theMaps.DrawForwardMaps("templateIDsForward", canvas);
        }

        canvas.cd();

        std::string fileName(m_imageFileName);
        canvas.SaveAs(fileName.c_str());
      }
      return true;
    }
  };

  using SiPixelTemplateIDsBPixMap = SiPixelTemplateIDs<SiPixelPI::t_barrel>;
  using SiPixelTemplateIDsFPixMap = SiPixelTemplateIDs<SiPixelPI::t_forward>;

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(SiPixelTemplateDBObject) {
  PAYLOAD_INSPECTOR_CLASS(SiPixelTemplateDBObjectTest);
  PAYLOAD_INSPECTOR_CLASS(SiPixelTemplateHeaderTable);
  PAYLOAD_INSPECTOR_CLASS(SiPixelTemplateIDsBPixMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelTemplateIDsFPixMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelTemplateLABPixMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelTemplateLAFPixMap);
}
