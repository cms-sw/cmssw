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
      gStyle->SetPalette(kMint);     // for the ghost plot (colored BPix and FPix bins)

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

        auto h2_Header = std::make_unique<TH2F>("Header", ";;", tempSize, 0, tempSize, 12, 0., 12.);
        auto h2_ghost = std::make_unique<TH2F>("ghost", ";;", tempSize, 0, tempSize, 12, 0., 12.);
        h2_Header->SetStats(false);
        h2_ghost->SetStats(false);

        int tempVersion = -999;

        for (const auto& theTemp : thePixelTemp_ | boost::adaptors::indexed(1)) {
          auto tempValue = theTemp.value();
          auto idx = theTemp.index();
          float uH = -99.;
          if (tempValue.head.Bfield != 0.) {
            uH = roundoff(tempValue.head.lorxwidth / tempValue.head.zsize / tempValue.head.Bfield, 4);
          }

          // clang-format off
          h2_Header->SetBinContent(idx, 12, tempValue.head.ID);     //!< template ID number
          h2_Header->SetBinContent(idx, 11, tempValue.head.Bfield); //!< Bfield in Tesla
          h2_Header->SetBinContent(idx, 10, uH);                    //!< hall mobility
          h2_Header->SetBinContent(idx, 9, tempValue.head.xsize);   //!< pixel size (for future use in upgraded geometry)
          h2_Header->SetBinContent(idx, 8, tempValue.head.ysize);   //!< pixel size (for future use in upgraded geometry)
          h2_Header->SetBinContent(idx, 7, tempValue.head.zsize);   //!< pixel size (for future use in upgraded geometry)
          h2_Header->SetBinContent(idx, 6, tempValue.head.NTy);     //!< number of Template y entries
          h2_Header->SetBinContent(idx, 5, tempValue.head.NTyx);    //!< number of Template y-slices of x entries
          h2_Header->SetBinContent(idx, 4, tempValue.head.NTxx);    //!< number of Template x-entries in each slice
          h2_Header->SetBinContent(idx, 3, tempValue.head.Dtype);   //!< detector type (0=BPix, 1=FPix)
          h2_Header->SetBinContent(idx, 2, tempValue.head.qscale);  //!< Charge scaling to match cmssw and pixelav
          h2_Header->SetBinContent(idx, 1, tempValue.head.Vbias);   //!< detector bias potential in Volts
          // clang-format on

          h2_Header->GetYaxis()->SetBinLabel(12, "GenErrorID");
          h2_Header->GetYaxis()->SetBinLabel(11, "B-field [T]");
          h2_Header->GetYaxis()->SetBinLabel(10, "#mu_{H} [1/T]");
          h2_Header->GetYaxis()->SetBinLabel(9, "x-size [#mum]");
          h2_Header->GetYaxis()->SetBinLabel(8, "y-size [#mum]");
          h2_Header->GetYaxis()->SetBinLabel(7, "z-size [#mum]");
          h2_Header->GetYaxis()->SetBinLabel(6, "NTy");
          h2_Header->GetYaxis()->SetBinLabel(5, "NTyx");
          h2_Header->GetYaxis()->SetBinLabel(4, "NTxx");
          h2_Header->GetYaxis()->SetBinLabel(3, "DetectorType");
          h2_Header->GetYaxis()->SetBinLabel(2, "qScale");
          h2_Header->GetYaxis()->SetBinLabel(1, "VBias [V]");
          h2_Header->GetXaxis()->SetBinLabel(idx, "");

          for (unsigned int iy = 1; iy <= 12; iy++) {
            if (tempValue.head.Dtype != 0 || uH < 0) {
              h2_ghost->SetBinContent(idx, iy, 1);
            } else {
              h2_ghost->SetBinContent(idx, iy, -1);
            }
            h2_ghost->GetYaxis()->SetBinLabel(iy, h2_Header->GetYaxis()->GetBinLabel(iy));
            h2_ghost->GetXaxis()->SetBinLabel(idx, "");
          }

          if (tempValue.head.templ_version != tempVersion) {
            tempVersion = tempValue.head.templ_version;
          }
        }

        h2_Header->GetXaxis()->LabelsOption("h");
        h2_Header->GetXaxis()->SetNdivisions(500 + tempSize, false);
        h2_Header->GetYaxis()->SetLabelSize(0.05);
        h2_Header->SetMarkerSize(1.5);

        h2_ghost->GetXaxis()->LabelsOption("h");
        h2_ghost->GetXaxis()->SetNdivisions(500 + tempSize, false);
        h2_ghost->GetYaxis()->SetLabelSize(0.05);

        canvas.cd();
        h2_ghost->Draw("col");
        h2_Header->Draw("TEXTsame");

        TPaveText ksPt(0, 0, 0.88, 0.04, "NDC");
        ksPt.SetBorderSize(0);
        ksPt.SetFillColor(0);
        const char* textToAdd = Form(
            "Template Version: #color[2]{%i}. Payload hash: #color[2]{%s}", tempVersion, (std::get<1>(iov)).c_str());
        ksPt.AddText(textToAdd);
        ksPt.Draw();

        auto ltx = TLatex();
        ltx.SetTextFont(62);
        ltx.SetTextSize(0.040);
        ltx.SetTextAlign(11);
        ltx.DrawLatexNDC(
            gPad->GetLeftMargin(),
            1 - gPad->GetTopMargin() + 0.01,
            ("#color[4]{" + tagname + "}, IOV: #color[4]{" + std::to_string(std::get<0>(iov)) + "}").c_str());

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
