/*!
  \file AlignPCLThresholds_PayloadInspector
  \Payload Inspector Plugin for Alignment PCL thresholds
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2017/10/19 12:51:00 $
*/

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/PCLConfig/interface/AlignPCLThresholds.h"

#include <memory>
#include <sstream>
#include <iostream>
#include <functional>

// include ROOT
#include "TH2F.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"

namespace {

  enum types { DELTA, SIG, MAXMOVE, MAXERR, END_OF_TYPES };

  /************************************************
    Display of AlignPCLThresholds
  *************************************************/
  class AlignPCLThresholds_Display : public cond::payloadInspector::PlotImage<AlignPCLThresholds> {
  public:
    AlignPCLThresholds_Display()
        : cond::payloadInspector::PlotImage<AlignPCLThresholds>("Display of threshold parameters for SiPixelAli PCL") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      auto iov = iovs.front();
      std::shared_ptr<AlignPCLThresholds> payload = fetchPayload(std::get<1>(iov));
      auto alignables = payload->getAlignableList();

      TCanvas canvas("Alignment PCL thresholds summary", "Alignment PCL thresholds summary", 1500, 800);
      canvas.cd();

      canvas.SetTopMargin(0.07);
      canvas.SetBottomMargin(0.06);
      canvas.SetLeftMargin(0.11);
      canvas.SetRightMargin(0.05);
      canvas.Modified();
      canvas.SetGrid();

      auto Thresholds = std::make_unique<TH2F>(
          "Thresholds", "Alignment parameter thresholds", alignables.size(), 0, alignables.size(), 24, 0, 24);
      Thresholds->SetStats(false);

      std::function<float(types, std::string, AlignPCLThresholds::coordType)> cutFunctor =
          [&payload](types my_type, std::string alignable, AlignPCLThresholds::coordType coord) {
            float ret(-999.);
            switch (my_type) {
              case DELTA:
                return payload->getCut(alignable, coord);
              case SIG:
                return payload->getSigCut(alignable, coord);
              case MAXMOVE:
                return payload->getMaxMoveCut(alignable, coord);
              case MAXERR:
                return payload->getMaxErrorCut(alignable, coord);
              case END_OF_TYPES:
                return ret;
              default:
                return ret;
            }
          };

      unsigned int xBin = 0;
      for (const auto& alignable : alignables) {
        xBin++;

        auto xLabel = replaceAll(replaceAll(alignable, "minus", "(-)"), "plus", "(+)");
        Thresholds->GetXaxis()->SetBinLabel(xBin, (xLabel).c_str());
        unsigned int yBin = 24;
        for (int foo = AlignPCLThresholds::X; foo != AlignPCLThresholds::extra_DOF; foo++) {
          AlignPCLThresholds::coordType coord = static_cast<AlignPCLThresholds::coordType>(foo);
          for (int bar = types::DELTA; bar != types::END_OF_TYPES; bar++) {
            types type = static_cast<types>(bar);
            std::string theLabel = getStringFromTypeEnum(type) + getStringFromCoordEnum(coord);
            if (xBin == 1) {
              Thresholds->GetYaxis()->SetBinLabel(yBin, theLabel.c_str());
            }

            Thresholds->SetBinContent(xBin, yBin, cutFunctor(type, alignable, coord));

            yBin--;
          }  // loop on types
        }    // loop on coordinates
      }      // loop on alignables

      Thresholds->GetXaxis()->LabelsOption("h");
      Thresholds->Draw("TEXT");

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }

    /************************************************/
    std::string getStringFromCoordEnum(const AlignPCLThresholds::coordType& coord) {
      switch (coord) {
        case AlignPCLThresholds::X:
          return "X";
        case AlignPCLThresholds::Y:
          return "Y";
        case AlignPCLThresholds::Z:
          return "Z";
        case AlignPCLThresholds::theta_X:
          return "#theta_{X}";
        case AlignPCLThresholds::theta_Y:
          return "#theta_{Y}";
        case AlignPCLThresholds::theta_Z:
          return "#theta_{Z}";
        default:
          return "should never be here";
      }
    }

    /************************************************/
    std::string getStringFromTypeEnum(const types& type) {
      switch (type) {
        case types::DELTA:
          return "#Delta";
        case types::SIG:
          return "#Delta/#sigma ";
        case types::MAXMOVE:
          return "max. move ";
        case types::MAXERR:
          return "max. err ";
        default:
          return "should never be here";
      }
    }

    /************************************************/
    std::string replaceAll(const std::string& str, const std::string& from, const std::string& to) {
      std::string out(str);

      if (from.empty())
        return out;
      size_t start_pos = 0;
      while ((start_pos = out.find(from, start_pos)) != std::string::npos) {
        out.replace(start_pos, from.length(), to);
        start_pos += to.length();  // In case 'to' contains 'from', like replacing 'x' with 'yx'
      }
      return out;
    }
  };
}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(AlignPCLThresholds) { PAYLOAD_INSPECTOR_CLASS(AlignPCLThresholds_Display); }
