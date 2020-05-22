/*!
  \file SiPixelVCal_PayloadInspector
  \Payload Inspector Plugin for SiPixel Lorentz angles
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2019/06/20 10:59:56 $
*/

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelVCal.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"
#include "CondCore/SiPixelPlugins/interface/SiPixelPayloadInspectorHelper.h"
#include "CondCore/SiPixelPlugins/interface/PixelRegionContainers.h"

#include <memory>
#include <sstream>

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

namespace {

  namespace SiPixelVCalPI {
    enum type { t_slope = 0, t_offset = 1 };
  }

  /************************************************
    1d histogram of SiPixelVCal of 1 IOV 
  *************************************************/
  // inherit from one of the predefined plot class: Histogram1D
  template <SiPixelVCalPI::type myType>
  class SiPixelVCalValue : public cond::payloadInspector::Histogram1D<SiPixelVCal, cond::payloadInspector::SINGLE_IOV> {
  public:
    SiPixelVCalValue()
        : cond::payloadInspector::Histogram1D<SiPixelVCal, cond::payloadInspector::SINGLE_IOV>(
              "SiPixel VCal values",
              "SiPixel VCal values",
              100,
              myType == SiPixelVCalPI::t_slope ? 40. : -700,
              myType == SiPixelVCalPI::t_slope ? 60. : 0) {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const &iov : tag.iovs) {
        std::shared_ptr<SiPixelVCal> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          auto VCalMap_ = payload->getSlopeAndOffset();
          for (const auto &element : VCalMap_) {
            if (myType == SiPixelVCalPI::t_slope) {
              fillWithValue(element.second.slope);
            } else if (myType == SiPixelVCalPI::t_offset) {
              fillWithValue(element.second.offset);
            }
          }
        }  // payload
      }    // iovs
      return true;
    }  // fill
  };

  using SiPixelVCalSlopeValue = SiPixelVCalValue<SiPixelVCalPI::t_slope>;
  using SiPixelVCalOffsetValue = SiPixelVCalValue<SiPixelVCalPI::t_offset>;

}  // namespace

PAYLOAD_INSPECTOR_MODULE(SiPixelVCal) {
  PAYLOAD_INSPECTOR_CLASS(SiPixelVCalSlopeValue);
  PAYLOAD_INSPECTOR_CLASS(SiPixelVCalOffsetValue);
}
