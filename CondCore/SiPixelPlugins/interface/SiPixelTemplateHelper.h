#ifndef CONDCORE_SIPIXELPLUGINS_SIPIXELTEMPLATEHELPER_H
#define CONDCORE_SIPIXELPLUGINS_SIPIXELTEMPLATEHELPER_H

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/SiPixelPlugins/interface/SiPixelPayloadInspectorHelper.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelTemplateDBObject.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGenErrorDBObject.h"
#include "CondFormats/SiPixelTransient/interface/SiPixelTemplate.h"
#include "DQM/TrackerRemapper/interface/Phase1PixelMaps.h"
#include "DQM/TrackerRemapper/interface/Phase1PixelSummaryMap.h"

#include <type_traits>
#include <memory>
#include <sstream>
#include <fmt/printf.h>
#include <boost/range/adaptor/indexed.hpp>

// include ROOT
#include "TH2F.h"
#include "TH1F.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPave.h"
#include "TPaveStats.h"
#include "TGaxis.h"

namespace templateHelper {

  //************************************************
  // Display of Template/GenError Titles
  // *************************************************/
  template <class PayloadType, class StoreType, class TransientType>
  class SiPixelTitles_Display
      : public cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV> {
  public:
    SiPixelTitles_Display()
        : cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV>(
              "Table of SiPixelTemplate/GenError titles") {
      if constexpr (std::is_same_v<PayloadType, SiPixelGenErrorDBObject>) {
        isTemplate_ = false;
        label_ = "SiPixelGenErrorDBObject_PayloadInspector";
      } else {
        isTemplate_ = true;
        label_ = "SiPixelTemplateDBObject_PayloadInspector";
      }
    }

    bool fill() override {
      auto tag = cond::payloadInspector::PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      auto tagname = tag.name;
      std::vector<StoreType> thePixelTemp_;
      std::shared_ptr<PayloadType> payload = this->fetchPayload(std::get<1>(iov));

      std::string IOVsince = std::to_string(std::get<0>(iov));

      if (payload.get()) {
        if (!TransientType::pushfile(*payload, thePixelTemp_)) {
          throw cms::Exception(label_) << "\nERROR:" << (isTemplate_ ? "Templates" : "GenErrors")
                                       << " not filled correctly."
                                       << " Check the conditions. Using "
                                       << (isTemplate_ ? "SiPixelTemplateDBObject" : "SiPixelGenErrorDBObject")
                                       << " version " << payload->version() << "\n\n";
        }

        unsigned int mapsize = thePixelTemp_.size();
        float pitch = 1. / (mapsize * 1.1);

        float y, x1, x2;
        std::vector<float> y_x1, y_x2, y_line;
        std::vector<std::string> s_x1, s_x2, s_x3;

        // starting table at y=1.0 (top of the canvas)
        // first column is at 0.02, second column at 0.32 NDC
        y = 1.0;
        x1 = 0.02;
        x2 = x1 + 0.30;

        y -= pitch;
        y_x1.push_back(y);
        s_x1.push_back(Form("#scale[1.2]{%s}", (isTemplate_ ? "Template ID" : "GenError ID")));
        y_x2.push_back(y);
        s_x2.push_back(Form("#scale[1.2]{#color[4]{%s} in IOV: #color[4]{%s}}", tagname.c_str(), IOVsince.c_str()));

        y -= pitch / 2.;
        y_line.push_back(y);

        for (const auto& element : thePixelTemp_) {
          y -= pitch;
          y_x1.push_back(y);
          s_x1.push_back(std::to_string(element.head.ID));

          y_x2.push_back(y);
          s_x2.push_back(Form("#color[2]{%s}", element.head.title));

          y_line.push_back(y - (pitch / 2.));
        }

        const auto& c_title = fmt::sprintf("%s titles", (isTemplate_ ? "Template" : "GenError"));
        TCanvas canvas(c_title.c_str(), c_title.c_str(), 2000, std::max(y_x1.size(), y_x2.size()) * 40);
        TLatex l;
        // Draw the columns titles
        l.SetTextAlign(12);

        float newpitch = 1 / (std::max(y_x1.size(), y_x2.size()) * 1.1);
        float factor = newpitch / pitch;
        l.SetTextSize(newpitch - 0.002);
        canvas.cd();
        for (unsigned int i = 0; i < y_x1.size(); i++) {
          l.DrawLatexNDC(x1, 1 - (1 - y_x1[i]) * factor, s_x1[i].c_str());
        }

        for (unsigned int i = 0; i < y_x2.size(); i++) {
          l.DrawLatexNDC(x2, 1 - (1 - y_x2[i]) * factor, s_x2[i].c_str());
        }

        canvas.cd();
        canvas.Update();

        TLine lines[y_line.size()];
        unsigned int iL = 0;
        for (const auto& line : y_line) {
          lines[iL] = TLine(gPad->GetUxmin(), 1 - (1 - line) * factor, gPad->GetUxmax(), 1 - (1 - line) * factor);
          lines[iL].SetLineWidth(1);
          lines[iL].SetLineStyle(9);
          lines[iL].SetLineColor(2);
          lines[iL].Draw("same");
          iL++;
        }

        std::string fileName(this->m_imageFileName);
        canvas.SaveAs(fileName.c_str());

      }  // if payload.get()
      return true;
    }

  protected:
    bool isTemplate_;
    std::string label_;
  };

  /************************************************
  // header plotting
  *************************************************/
  template <class PayloadType, class StoreType, class TransientType>
  class SiPixelHeaderTable : public cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV> {
  public:
    SiPixelHeaderTable()
        : cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV>(
              "SiPixel CPE Conditions Header summary") {
      if constexpr (std::is_same_v<PayloadType, SiPixelGenErrorDBObject>) {
        isTemplate_ = false;
        label_ = "SiPixelGenErrorDBObject_PayloadInspector";
      } else {
        isTemplate_ = true;
        label_ = "SiPixelTemplateDBObject_PayloadInspector";
      }
    }

    bool fill() override {
      gStyle->SetHistMinimumZero();  // will display zero as zero in the text map
      gStyle->SetPalette(kMint);     // for the ghost plot (colored BPix and FPix bins)

      auto tag = cond::payloadInspector::PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      auto tagname = tag.name;
      std::vector<StoreType> thePixelTemp_;
      std::shared_ptr<PayloadType> payload = this->fetchPayload(std::get<1>(iov));

      if (payload.get()) {
        if (!TransientType::pushfile(*payload, thePixelTemp_)) {
          throw cms::Exception(label_) << "\nERROR:" << (isTemplate_ ? "Templates" : "GenErrors")
                                       << " not filled correctly."
                                       << " Check the conditions. Using "
                                       << (isTemplate_ ? "SiPixelTemplateDBObject" : "SiPixelGenErrorDBObject")
                                       << payload->version() << "\n\n";
        }

        // store the map of ID / interesting quantities
        TransientType templ(thePixelTemp_);
        TCanvas canvas("Header Summary", "Header summary", 1400, 1000);
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

          h2_Header->GetYaxis()->SetBinLabel(12, (isTemplate_ ? "TemplateID" : "GenErrorID"));
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
            // Some of the Phase-2 templates have DType = 0 for all partitions (TBPX, TEPX, TFPX)
            // so they are distinguished by the uH strength value (<0).
            // To avoid changing the behaviour of 0T payload (uH=-99) that case is treated separately
            if (tempValue.head.Dtype != 0 || (uH < 0 && uH > -99)) {
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
        const char* textToAdd = Form("%s Version: #color[2]{%i}. Payload hash: #color[2]{%s}",
                                     (isTemplate_ ? "Template" : "GenError"),
                                     tempVersion,
                                     (std::get<1>(iov)).c_str());
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

        std::string fileName(this->m_imageFileName);
        canvas.SaveAs(fileName.c_str());
      }
      return true;
    }

    float roundoff(float value, unsigned char prec) {
      float pow_10 = pow(10.0f, (float)prec);
      return round(value * pow_10) / pow_10;
    }

  protected:
    bool isTemplate_;
    std::string label_;
  };

  //***********************************************
  // TH2Poly Map of IDs
  //***********************************************/
  template <class PayloadType, SiPixelPI::DetType myType>
  class SiPixelIDs : public cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV> {
  public:
    SiPixelIDs()
        : cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV>(
              "SiPixelMap of Template / GenError ID Values") {
      if constexpr (std::is_same_v<PayloadType, SiPixelGenErrorDBObject>) {
        isTemplate_ = false;
        label_ = "SiPixelGenErrorDBObject_PayloadInspector";
      } else {
        isTemplate_ = true;
        label_ = "SiPixelTemplateDBObject_PayloadInspector";
      }
    }

    bool fill() override {
      gStyle->SetPalette(kRainBow);

      auto tag = cond::payloadInspector::PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<PayloadType> payload = this->fetchPayload(std::get<1>(iov));

      std::string barrelName_ = fmt::sprintf("%sIDsBarrel", (isTemplate_ ? "template" : "genError"));
      std::string endcapName_ = fmt::sprintf("%sIDsForward", (isTemplate_ ? "template" : "genError"));
      std::string title_ = fmt::sprintf("%s IDs", (isTemplate_ ? "template" : "genError"));

      if (payload.get()) {
        // Book the TH2Poly
        Phase1PixelMaps theMaps("text");
        if (myType == SiPixelPI::t_barrel) {
          // book the barrel bins of the TH2Poly
          theMaps.bookBarrelHistograms(barrelName_, title_.c_str(), title_.c_str());
        } else if (myType == SiPixelPI::t_forward) {
          // book the forward bins of the TH2Poly
          theMaps.bookForwardHistograms(endcapName_, title_.c_str(), title_.c_str());
        }

        std::map<unsigned int, short> templMap;
        if constexpr (std::is_same_v<PayloadType, SiPixelGenErrorDBObject>) {
          templMap = payload->getGenErrorIDs();
        } else {
          templMap = payload->getTemplateIDs();
        }

        if (templMap.size() == SiPixelPI::phase0size || templMap.size() > SiPixelPI::phase1size) {
          edm::LogError(label_)
              << "There are " << templMap.size()
              << " DetIds in this payload. SiPixelIDs maps are not supported for non-Phase1 Pixel geometries !";
          TCanvas canvas("Canv", "Canv", 1200, 1000);
          SiPixelPI::displayNotSupported(canvas, templMap.size());
          std::string fileName(this->m_imageFileName);
          canvas.SaveAs(fileName.c_str());
          return false;
        } else {
          if (templMap.size() < SiPixelPI::phase1size) {
            edm::LogWarning(label_) << "\n ********* WARNING! ********* \n There are " << templMap.size()
                                    << " DetIds in this payload !"
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
          COUT << "DetID: " << entry.first << fmt::sprintf("%s ID ", (isTemplate_ ? "Template" : "GenError"))
               << entry.second << std::endl;
          auto detid = DetId(entry.first);
          if ((detid.subdetId() == PixelSubdetector::PixelBarrel) && (myType == SiPixelPI::t_barrel)) {
            theMaps.fillBarrelBin(barrelName_, entry.first, entry.second);
          } else if ((detid.subdetId() == PixelSubdetector::PixelEndcap) && (myType == SiPixelPI::t_forward)) {
            theMaps.fillForwardBin(endcapName_, entry.first, entry.second);
          }
        }

        theMaps.beautifyAllHistograms();

        TCanvas canvas("Canv", "Canv", (myType == SiPixelPI::t_barrel) ? 1200 : 1500, 1000);
        if (myType == SiPixelPI::t_barrel) {
          theMaps.drawBarrelMaps(barrelName_, canvas);
        } else if (myType == SiPixelPI::t_forward) {
          theMaps.drawForwardMaps(endcapName_, canvas);
        }

        canvas.cd();

        std::string fileName(this->m_imageFileName);
        canvas.SaveAs(fileName.c_str());
      }
      return true;
    }

  protected:
    bool isTemplate_;
    std::string label_;
  };

  /************************************************
   Full Pixel Tracker Map class
  *************************************************/
  template <class PayloadType, class StoreType, class TransientType>
  class SiPixelFullPixelIDMap
      : public cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV> {
  public:
    SiPixelFullPixelIDMap()
        : cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV>(
              "SiPixel CPE conditions Map of IDs") {
      if constexpr (std::is_same_v<PayloadType, SiPixelGenErrorDBObject>) {
        isTemplate_ = false;
        label_ = "SiPixelGenErrorDBObject_PayloadInspector";
      } else {
        isTemplate_ = true;
        label_ = "SiPixelTemplateDBObject_PayloadInspector";
      }
    }

    bool fill() override {
      gStyle->SetPalette(1);
      auto tag = cond::payloadInspector::PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::vector<StoreType> thePixelTemp_;
      std::shared_ptr<PayloadType> payload = this->fetchPayload(std::get<1>(iov));

      std::string payloadString = (isTemplate_ ? "Templates" : "GenErrors");

      if (payload.get()) {
        if (!TransientType::pushfile(*payload, thePixelTemp_)) {
          throw cms::Exception(label_) << "\nERROR: " << payloadString
                                       << " not filled correctly. Check the conditions. Using "
                                       << (isTemplate_ ? "SiPixelTemplateDBObject" : "SiPixelGenErrorDBObject")
                                       << payload->version() << "\n\n";
        }

        Phase1PixelSummaryMap fullMap("", fmt::sprintf("%s IDs", payloadString), fmt::sprintf("%s ID", payloadString));
        fullMap.createTrackerBaseMap();

        std::map<unsigned int, short> templMap;
        if constexpr (std::is_same_v<PayloadType, SiPixelGenErrorDBObject>) {
          templMap = payload->getGenErrorIDs();
        } else {
          templMap = payload->getTemplateIDs();
        }

        for (const auto& entry : templMap) {
          fullMap.fillTrackerMap(entry.first, entry.second);
        }

        if (templMap.size() == SiPixelPI::phase0size || templMap.size() > SiPixelPI::phase1size) {
          edm::LogError(label_)
              << "There are " << templMap.size()
              << " DetIds in this payload. SiPixelIDs maps are not supported for non-Phase1 Pixel geometries !";
          TCanvas canvas("Canv", "Canv", 1200, 1000);
          SiPixelPI::displayNotSupported(canvas, templMap.size());
          std::string fileName(this->m_imageFileName);
          canvas.SaveAs(fileName.c_str());
          return false;
        } else {
          if (templMap.size() < SiPixelPI::phase1size) {
            edm::LogWarning(label_) << "\n ********* WARNING! ********* \n There are " << templMap.size()
                                    << " DetIds in this payload !"
                                    << "\n **************************** \n";
          }
        }

        TCanvas canvas("Canv", "Canv", 3000, 2000);
        fullMap.printTrackerMap(canvas);

        //fmt::sprintf("#color[2]{%s, IOV %i}",tag.name,std::get<0>(iov));

        auto ltx = TLatex();
        ltx.SetTextFont(62);
        ltx.SetTextSize(0.025);
        ltx.SetTextAlign(11);
        ltx.DrawLatexNDC(
            gPad->GetLeftMargin() + 0.01,
            gPad->GetBottomMargin() + 0.01,
            ("#color[4]{" + tag.name + "}, IOV: #color[4]{" + std::to_string(std::get<0>(iov)) + "}").c_str());

        std::string fileName(this->m_imageFileName);
        canvas.SaveAs(fileName.c_str());
      }
      return true;
    }

  protected:
    bool isTemplate_;
    std::string label_;
  };

  enum headerParam {
    k_ID = 0,
    k_NTy = 1,
    k_NTyx = 2,
    k_NTxx = 3,
    k_Dtype = 4,
    k_qscale = 5,
    k_lorywidth = 6,
    k_lorxwidth = 7,
    k_lorybias = 8,
    k_lorxbias = 9,
    k_Vbias = 10,
    k_temperature = 11,
    k_fluence = 12,
    k_s50 = 13,
    k_ss50 = 14,
    k_title = 15,
    k_templ_version = 16,
    k_Bfield = 17,
    k_fbin = 18,
    k_END_OF_TYPES = 19,
  };

  static constexpr const char* header_types[] = {"ID;templated ID",
                                                 "NTy;number of template y entries",
                                                 "NTyx;number of template y-slices of x entries",
                                                 "NTxx;number of template x-entries in each slice",
                                                 "Dtype;detector type (0=BPix, 1=FPix)",
                                                 "qScale;charge scaling correction",
                                                 "lorxwidth;estimate of the y-Lorentz width",
                                                 "lorywidth;estimate of the x-Lorentz width",
                                                 "lorybias;estimate of the y-Lorentz bias",
                                                 "lorxbias;estimate of the x-Lorentz bias",
                                                 "Vbias;detector bias [V]",
                                                 "temperature;detector temperature [K]",
                                                 "fluence;radiation fluence [n_{eq}/cm^{2}] ",
                                                 "s50;1/2 of the multihit dcol threshold [e]",
                                                 "ss50;1/2 of the single hit dcol threshold [e]",
                                                 "title;title",
                                                 "template version;template version number",
                                                 "B-field;B-field [T]",
                                                 "fbin;Qbin in Q_{clus}/Q_{avg}",
                                                 "NOT HERE;NOT HERE"};

  // class to display values of the template header information in a Phase1 Pixel Map
  template <class PayloadType, class StoreType, class TransientType, SiPixelPI::DetType myType, headerParam myParam>
  class SiPixelTemplateHeaderInfo
      : public cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV> {
    struct header_info {
      int ID;             //!< template ID number
      int NTy;            //!< number of Template y entries
      int NTyx;           //!< number of Template y-slices of x entries
      int NTxx;           //!< number of Template x-entries in each slice
      int Dtype;          //!< detector type (0=BPix, 1=FPix)
      float qscale;       //!< Charge scaling to match cmssw and pixelav
      float lorywidth;    //!< estimate of y-lorentz width for optimal resolution
      float lorxwidth;    //!< estimate of x-lorentz width for optimal resolution
      float lorybias;     //!< estimate of y-lorentz bias
      float lorxbias;     //!< estimate of x-lorentz bias
      float Vbias;        //!< detector bias potential in Volts
      float temperature;  //!< detector temperature in deg K
      float fluence;      //!< radiation fluence in n_eq/cm^2
      float s50;          //!< 1/2 of the multihit dcol threshold in electrons
      float ss50;         //!< 1/2 of the single hit dcol threshold in electrons
      char title[80];     //!< template title
      int templ_version;  //!< Version number of the template to ensure code compatibility
      float Bfield;       //!< Bfield in Tesla
      float fbin[3];      //!< The QBin definitions in Q_clus/Q_avg
    };

  public:
    SiPixelTemplateHeaderInfo()
        : cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV>(
              "SiPixel CPE conditions Map of header quantities") {
      if constexpr (std::is_same_v<PayloadType, SiPixelGenErrorDBObject>) {
        isTemplate_ = false;
        label_ = "SiPixelGenErrorDBObject_PayloadInspector";
      } else {
        isTemplate_ = true;
        label_ = "SiPixelTemplateDBObject_PayloadInspector";
      }
    }

    bool fill() override {
      gStyle->SetPalette(kRainBow);
      if (!(myParam == headerParam::k_Vbias || myParam == headerParam::k_Dtype)) {
        TGaxis::SetMaxDigits(2);
      }

      auto tag = cond::payloadInspector::PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::vector<StoreType> thePixelTemp_;
      std::shared_ptr<PayloadType> payload = this->fetchPayload(std::get<1>(iov));

      if (payload.get()) {
        if (!TransientType::pushfile(*payload, thePixelTemp_)) {
          throw cms::Exception(label_) << "\nERROR: Templates not filled correctly. Check the conditions. Using "
                                          "payload version "
                                       << payload->version() << "\n\n";
        }

        // store the map of ID / interesting quantities
        SiPixelTemplate templ(thePixelTemp_);
        for (const auto& theTemp : thePixelTemp_) {
          header_info info;
          info.ID = theTemp.head.ID;
          info.NTy = theTemp.head.NTy;
          info.NTyx = theTemp.head.NTyx;
          info.NTxx = theTemp.head.NTxx;
          info.Dtype = theTemp.head.Dtype;
          info.qscale = theTemp.head.qscale;
          info.lorywidth = theTemp.head.lorywidth;
          info.lorxwidth = theTemp.head.lorxwidth;
          info.lorybias = theTemp.head.lorybias;
          info.lorxbias = theTemp.head.lorxbias;
          info.Vbias = theTemp.head.Vbias;
          info.temperature = theTemp.head.temperature;
          info.fluence = theTemp.head.fluence;
          info.s50 = theTemp.head.s50;
          info.ss50 = theTemp.head.ss50;
          info.templ_version = theTemp.head.templ_version;
          info.Bfield = theTemp.head.Bfield;
          theInfos_[theTemp.head.ID] = info;
        }

        // Book the TH2Poly
        Phase1PixelMaps theMaps("");
        if (myType == SiPixelPI::t_all) {
          theMaps.resetOption("COLZA L");
        } else {
          theMaps.resetOption("COLZL");
        }

        std::string input{header_types[myParam]};
        std::string delimiter = ";";
        std::string first = input.substr(0, input.find(delimiter));
        std::string second = input.substr(input.find(delimiter) + 1);

        if (myType == SiPixelPI::t_barrel) {
          theMaps.bookBarrelHistograms("templateLABarrel", first.c_str(), second.c_str());
        } else if (myType == SiPixelPI::t_forward) {
          theMaps.bookForwardHistograms("templateLAForward", first.c_str(), second.c_str());
        } else if (myType == SiPixelPI::t_all) {
          theMaps.bookBarrelHistograms("templateLA", first.c_str(), second.c_str());
          theMaps.bookForwardHistograms("templateLA", first.c_str(), second.c_str());
        } else {
          edm::LogError(label_) << " un-recognized detector type " << myType << std::endl;
          return false;
        }

        std::map<unsigned int, short> templMap = payload->getTemplateIDs();
        if (templMap.size() == SiPixelPI::phase0size || templMap.size() > SiPixelPI::phase1size) {
          edm::LogError(label_)
              << "There are " << templMap.size()
              << " DetIds in this payload. SiPixelTempate Lorentz Angle maps are not supported for non-Phase1 Pixel "
                 "geometries !";
          TCanvas canvas("Canv", "Canv", 1200, 1000);
          SiPixelPI::displayNotSupported(canvas, templMap.size());
          std::string fileName(this->m_imageFileName);
          canvas.SaveAs(fileName.c_str());
          return false;
        } else {
          if (templMap.size() < SiPixelPI::phase1size) {
            edm::LogWarning(label_) << "\n ********* WARNING! ********* \n There are " << templMap.size()
                                    << " DetIds in this payload !"
                                    << "\n **************************** \n";
          }
        }

        for (auto const& entry : templMap) {
          //templ.interpolate(entry.second, 0.f, 0.f, 1.f, 1.f);

          const auto& theInfo = theInfos_[entry.second];

          std::function<float(headerParam, header_info)> cutFunctor = [](headerParam my_param, header_info myInfo) {
            float ret(-999.);
            switch (my_param) {
              case k_ID:
                return (float)myInfo.ID;
              case k_NTy:
                return (float)myInfo.NTy;
              case k_NTyx:
                return (float)myInfo.NTyx;
              case k_NTxx:
                return (float)myInfo.NTxx;
              case k_Dtype:
                return (float)myInfo.Dtype;
              case k_qscale:
                return (float)myInfo.qscale;
              case k_lorywidth:
                return (float)myInfo.lorywidth;
              case k_lorxwidth:
                return (float)myInfo.lorxwidth;
              case k_lorybias:
                return (float)myInfo.lorybias;
              case k_lorxbias:
                return (float)myInfo.lorxbias;
              case k_Vbias:
                return (float)myInfo.Vbias;
              case k_temperature:
                return (float)myInfo.temperature;
              case k_fluence:
                return (float)myInfo.fluence;
              case k_s50:
                return (float)myInfo.s50;
              case k_ss50:
                return (float)myInfo.ss50;
              case k_title:
                return (float)myInfo.templ_version;
              case k_Bfield:
                return (float)myInfo.Bfield;
              case k_END_OF_TYPES:
                return ret;
              default:
                return ret;
            }
          };

          auto detid = DetId(entry.first);
          if (myType == SiPixelPI::t_all) {
            if ((detid.subdetId() == PixelSubdetector::PixelBarrel)) {
              theMaps.fillBarrelBin("templateLA", entry.first, cutFunctor(myParam, theInfo));
            }
            if ((detid.subdetId() == PixelSubdetector::PixelEndcap)) {
              theMaps.fillForwardBin("templateLA", entry.first, cutFunctor(myParam, theInfo));
            }
          } else if ((detid.subdetId() == PixelSubdetector::PixelBarrel) && (myType == SiPixelPI::t_barrel)) {
            theMaps.fillBarrelBin("templateLABarrel", entry.first, cutFunctor(myParam, theInfo));
          } else if ((detid.subdetId() == PixelSubdetector::PixelEndcap) && (myType == SiPixelPI::t_forward)) {
            theMaps.fillForwardBin("templateLAForward", entry.first, cutFunctor(myParam, theInfo));
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
        std::string fileName(this->m_imageFileName);
        canvas.SaveAs(fileName.c_str());
      }
      return true;
    }  // fill

  protected:
    bool isTemplate_;
    std::string label_;

  private:
    std::map<int, header_info> theInfos_;
  };

}  // namespace templateHelper

#endif
