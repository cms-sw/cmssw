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

  /************************************************
  // header plotting
  *************************************************/
  class SiPixelGenErrorHeaderTable
      : public cond::payloadInspector::PlotImage<SiPixelGenErrorDBObject, cond::payloadInspector::SINGLE_IOV> {
  public:
    SiPixelGenErrorHeaderTable()
        : cond::payloadInspector::PlotImage<SiPixelGenErrorDBObject, cond::payloadInspector::SINGLE_IOV>(
              "SiPixelGenErrorDBObject Header summary") {}

    bool fill() override {
      gStyle->SetHistMinimumZero();  // will display zero as zero in the text map

      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      auto tagname = tag.name;
      std::vector<SiPixelGenErrorStore> thePixelTemp_;
      std::shared_ptr<SiPixelGenErrorDBObject> payload = fetchPayload(std::get<1>(iov));

      if (payload.get()) {
        if (!SiPixelGenError::pushfile(*payload, thePixelTemp_)) {
          throw cms::Exception("SiPixelGenErrorDBObject_PayloadInspector")
              << "\nERROR: GenErrors not filled correctly. Check the conditions. Using "
                 "SiPixelGenErrorDBObject version "
              << payload->version() << "\n\n";
        }

        // store the map of ID / interesting quantities
        SiPixelGenError templ(thePixelTemp_);
        TCanvas canvas("GenError Header Summary", "GenError Header summary", 1400, 1000);
        canvas.cd();

        unsigned int tempSize = thePixelTemp_.size();

        canvas.SetTopMargin(0.07);
        canvas.SetBottomMargin(0.06);
        canvas.SetLeftMargin(0.17);
        canvas.SetRightMargin(0.03);
        canvas.Modified();
        canvas.SetGrid();

        auto h2_GenErrorHeaders = std::make_unique<TH2F>("Header", ";;", tempSize, 0, tempSize, 6, 0., 6.);
        h2_GenErrorHeaders->SetStats(false);

        for (const auto& theTemp : thePixelTemp_ | boost::adaptors::indexed(1)) {
          auto tempValue = theTemp.value();
          auto tempIndex = theTemp.index();
          float uH = -99.;
          if (tempValue.head.Bfield != 0.) {
            uH = roundoff(tempValue.head.lorxwidth / tempValue.head.zsize / tempValue.head.Bfield, 4);
          }
          h2_GenErrorHeaders->SetBinContent(tempIndex, 6, tempValue.head.ID);
          h2_GenErrorHeaders->SetBinContent(tempIndex, 5, tempValue.head.Bfield);
          h2_GenErrorHeaders->SetBinContent(tempIndex, 4, uH);
          h2_GenErrorHeaders->SetBinContent(tempIndex, 3, tempValue.head.xsize);
          h2_GenErrorHeaders->SetBinContent(tempIndex, 2, tempValue.head.ysize);
          h2_GenErrorHeaders->SetBinContent(tempIndex, 1, tempValue.head.zsize);
          h2_GenErrorHeaders->GetYaxis()->SetBinLabel(6, "GenErrorID");
          h2_GenErrorHeaders->GetYaxis()->SetBinLabel(5, "B-field [T]");
          h2_GenErrorHeaders->GetYaxis()->SetBinLabel(4, "#mu_{H} [1/T]");
          h2_GenErrorHeaders->GetYaxis()->SetBinLabel(3, "x-size [#mum]");
          h2_GenErrorHeaders->GetYaxis()->SetBinLabel(2, "y-size [#mum]");
          h2_GenErrorHeaders->GetYaxis()->SetBinLabel(1, "z-size [#mum]");
          h2_GenErrorHeaders->GetXaxis()->SetBinLabel(tempIndex, "");
        }

        h2_GenErrorHeaders->GetXaxis()->LabelsOption("h");
        h2_GenErrorHeaders->GetXaxis()->SetNdivisions(500 + tempSize, false);
        h2_GenErrorHeaders->GetYaxis()->SetLabelSize(0.05);
        h2_GenErrorHeaders->SetMarkerSize(1.5);

        canvas.cd();
        h2_GenErrorHeaders->Draw("TEXT");

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
  PAYLOAD_INSPECTOR_CLASS(SiPixelGenErrorHeaderTable);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGenErrorIDsBPixMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGenErrorIDsFPixMap);
}
