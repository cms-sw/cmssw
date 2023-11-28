// authors: Jan Kaspar (jan.kaspar@gmail.com)

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondTools/RunInfo/interface/LHCInfoCombined.h"

#include "CondFormats/DataRecord/interface/CTPPSOpticsRcd.h"
#include "CondFormats/DataRecord/interface/CTPPSInterpolatedOpticsRcd.h"

#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "CondFormats/PPSObjects/interface/LHCOpticalFunctionsSetCollection.h"
#include "CondFormats/PPSObjects/interface/LHCInterpolatedOpticalFunctionsSetCollection.h"

class CTPPSInterpolatedOpticalFunctionsESSource : public edm::ESProducer {
public:
  CTPPSInterpolatedOpticalFunctionsESSource(const edm::ParameterSet &);
  ~CTPPSInterpolatedOpticalFunctionsESSource() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  std::shared_ptr<LHCInterpolatedOpticalFunctionsSetCollection> produce(const CTPPSInterpolatedOpticsRcd &);

private:
  edm::ESGetToken<LHCOpticalFunctionsSetCollection, CTPPSOpticsRcd> opticsToken_;
  edm::ESGetToken<LHCInfo, LHCInfoRcd> lhcInfoToken_;
  edm::ESGetToken<LHCInfoPerLS, LHCInfoPerLSRcd> lhcInfoPerLSToken_;
  edm::ESGetToken<LHCInfoPerFill, LHCInfoPerFillRcd> lhcInfoPerFillToken_;
  std::shared_ptr<LHCInterpolatedOpticalFunctionsSetCollection> currentData_;
  float currentCrossingAngle_;
  bool currentDataValid_;
  const bool useNewLHCInfo_;
};

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

CTPPSInterpolatedOpticalFunctionsESSource::CTPPSInterpolatedOpticalFunctionsESSource(const edm::ParameterSet &iConfig)
    : currentCrossingAngle_(LHCInfoCombined::crossingAngleInvalid),
      currentDataValid_(false),
      useNewLHCInfo_(iConfig.getParameter<bool>("useNewLHCInfo")) {
  auto cc = setWhatProduced(this, iConfig.getParameter<std::string>("opticsLabel"));
  opticsToken_ = cc.consumes(edm::ESInputTag("", iConfig.getParameter<std::string>("opticsLabel")));
  lhcInfoToken_ = cc.consumes(edm::ESInputTag("", iConfig.getParameter<std::string>("lhcInfoLabel")));
  lhcInfoPerLSToken_ = cc.consumes(edm::ESInputTag("", iConfig.getParameter<std::string>("lhcInfoPerLSLabel")));
  lhcInfoPerFillToken_ = cc.consumes(edm::ESInputTag("", iConfig.getParameter<std::string>("lhcInfoPerFillLabel")));
}

//----------------------------------------------------------------------------------------------------

void CTPPSInterpolatedOpticalFunctionsESSource::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("lhcInfoLabel", "")->setComment("label of the LHCInfo record");
  desc.add<std::string>("lhcInfoPerFillLabel", "")->setComment("label of the LHCInfoPerFill record");
  desc.add<std::string>("lhcInfoPerLSLabel", "")->setComment("label of the LHCInfoPerLS record");
  desc.add<std::string>("opticsLabel", "")->setComment("label of the optics records");
  desc.add<bool>("useNewLHCInfo", false)->setComment("flag whether to use new LHCInfoPer* records or old LHCInfo");

  descriptions.add("ctppsInterpolatedOpticalFunctionsESSource", desc);
}

//----------------------------------------------------------------------------------------------------

std::shared_ptr<LHCInterpolatedOpticalFunctionsSetCollection> CTPPSInterpolatedOpticalFunctionsESSource::produce(
    const CTPPSInterpolatedOpticsRcd &iRecord) {
  // get the input data
  LHCOpticalFunctionsSetCollection const &ofColl = iRecord.get(opticsToken_);

  auto lhcInfoCombined = LHCInfoCombined::createLHCInfoCombined<
      CTPPSInterpolatedOpticsRcd,
      edm::mpl::Vector<CTPPSOpticsRcd, LHCInfoRcd, LHCInfoPerFillRcd, LHCInfoPerLSRcd>>(
      iRecord, lhcInfoPerLSToken_, lhcInfoPerFillToken_, lhcInfoToken_, useNewLHCInfo_);

  // is there anything to do?
  if (currentDataValid_ && lhcInfoCombined.crossingAngle() == currentCrossingAngle_)
    return currentData_;

  // is crossing angle reasonable (LHCInfo is correctly filled in DB)?
  if (lhcInfoCombined.isCrossingAngleInvalid()) {
    edm::LogInfo("CTPPSInterpolatedOpticalFunctionsESSource")
        << "Invalid crossing angle, no optical functions produced.";

    currentDataValid_ = false;
    currentCrossingAngle_ = LHCInfoCombined::crossingAngleInvalid;
    currentData_ = std::make_shared<LHCInterpolatedOpticalFunctionsSetCollection>();

    return currentData_;
  }

  // set new crossing angle
  currentCrossingAngle_ = lhcInfoCombined.crossingAngle();
  edm::LogInfo("CTPPSInterpolatedOpticalFunctionsESSource")
      << "Crossing angle has changed to " << currentCrossingAngle_ << ".";

  // is input optics available ?
  if (ofColl.empty()) {
    edm::LogInfo("CTPPSInterpolatedOpticalFunctionsESSource")
        << "No input optics available, no optical functions produced.";

    currentDataValid_ = false;
    currentData_ = std::make_shared<LHCInterpolatedOpticalFunctionsSetCollection>();

    return currentData_;
  }

  // regular case with single-xangle input
  if (ofColl.size() == 1) {
    const auto &it = ofColl.begin();

    // does the input xangle correspond to the actual one?
    if (fabs(currentCrossingAngle_ - it->first) > 1e-6)
      throw cms::Exception("CTPPSInterpolatedOpticalFunctionsESSource")
          << "Cannot interpolate: input given only for xangle " << it->first << " while interpolation requested for "
          << currentCrossingAngle_ << ".";

    currentData_ = std::make_shared<LHCInterpolatedOpticalFunctionsSetCollection>();
    for (const auto &rp_p : it->second) {
      const auto rpId = rp_p.first;
      LHCInterpolatedOpticalFunctionsSet iof(rp_p.second);
      iof.initializeSplines();
      currentData_->emplace(rpId, std::move(iof));
    }

    currentDataValid_ = true;
  }

  // regular case with multi-xangle input
  if (ofColl.size() > 1) {
    // find the closest xangle points for interpolation
    auto it1 = ofColl.begin();
    auto it2 = std::next(it1);

    if (currentCrossingAngle_ > it1->first) {
      for (; it1 != ofColl.end(); ++it1) {
        it2 = std::next(it1);

        if (it2 == ofColl.end()) {
          it2 = it1;
          it1 = std::prev(it1);
          break;
        }

        if (it1->first <= currentCrossingAngle_ && currentCrossingAngle_ < it2->first)
          break;
      }
    }

    const auto &xangle1 = it1->first;
    const auto &xangle2 = it2->first;

    const auto &ofs1 = it1->second;
    const auto &ofs2 = it2->second;

    // do the interpoaltion RP by RP
    currentData_ = std::make_shared<LHCInterpolatedOpticalFunctionsSetCollection>();
    for (const auto &rp_p : ofs1) {
      const auto rpId = rp_p.first;
      const auto &rp_it2 = ofs2.find(rpId);
      if (rp_it2 == ofs2.end())
        throw cms::Exception("CTPPSInterpolatedOpticalFunctionsESSource") << "RP mismatch between ofs1 and ofs2.";

      const auto &of1 = rp_p.second;
      const auto &of2 = rp_it2->second;

      const size_t num_xi_vals1 = of1.getXiValues().size();
      const size_t num_xi_vals2 = of2.getXiValues().size();

      if (num_xi_vals1 != num_xi_vals2)
        throw cms::Exception("CTPPSInterpolatedOpticalFunctionsESSource") << "Size mismatch between ofs1 and ofs2.";

      const size_t num_xi_vals = num_xi_vals1;

      LHCInterpolatedOpticalFunctionsSet iof;
      iof.m_z = of1.getScoringPlaneZ();
      iof.m_fcn_values.resize(LHCInterpolatedOpticalFunctionsSet::nFunctions);
      iof.m_xi_values.resize(num_xi_vals);

      for (size_t fi = 0; fi < of1.getFcnValues().size(); ++fi) {
        iof.m_fcn_values[fi].resize(num_xi_vals);

        for (size_t pi = 0; pi < num_xi_vals; ++pi) {
          double xi = of1.getXiValues()[pi];
          double xi_control = of2.getXiValues()[pi];

          if (fabs(xi - xi_control) > 1e-6)
            throw cms::Exception("CTPPSInterpolatedOpticalFunctionsESSource") << "Xi mismatch between ofs1 and ofs2.";

          iof.m_xi_values[pi] = xi;

          double v1 = of1.getFcnValues()[fi][pi];
          double v2 = of2.getFcnValues()[fi][pi];
          iof.m_fcn_values[fi][pi] = v1 + (v2 - v1) / (xangle2 - xangle1) * (currentCrossingAngle_ - xangle1);
        }
      }

      iof.initializeSplines();

      currentDataValid_ = true;
      currentData_->emplace(rpId, std::move(iof));
    }
  }

  return currentData_;
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_EVENTSETUP_MODULE(CTPPSInterpolatedOpticalFunctionsESSource);
