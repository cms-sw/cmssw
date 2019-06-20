/*!
  \file SiPixelLorentzAngle_PayloadInspector
  \Payload Inspector Plugin for SiPixel Lorentz angles
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2019/06/20 10:59:56 $
*/

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelLorentzAngle.h"
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"

#include <memory>
#include <sstream>

// include ROOT
#include "TH2F.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPave.h"
#include "TPaveStats.h"
#include "TGaxis.h"

namespace {

  /************************************************
    1d histogram of SiPixelLorentzAngle of 1 IOV 
  *************************************************/

  // inherit from one of the predefined plot class: Histogram1D
  class SiPixelLorentzAngleValue : public cond::payloadInspector::Histogram1D<SiPixelLorentzAngle> {
  public:
    SiPixelLorentzAngleValue()
      : cond::payloadInspector::Histogram1D<SiPixelLorentzAngle>(
								 "SiPixel LorentzAngle values", "SiPixel LorentzAngle values", 100, 0.0, 0.05) {
      Base::setSingleIov(true);
    }
    
    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> > &iovs) override {
      for (auto const &iov : iovs) {
        std::shared_ptr<SiPixelLorentzAngle> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          std::map<uint32_t, float> LAMap_ = payload->getLorentzAngles();
	  
          for (const auto &element : LAMap_) {
            fillWithValue(element.second);
          }
        }  // payload
      }    // iovs
      return true;
    }  // fill
  }; 
}  // namespace

PAYLOAD_INSPECTOR_MODULE(SiPixelLorentzAngle) {
  PAYLOAD_INSPECTOR_CLASS(SiPixelLorentzAngleValue);
}
