/****************************************************************************
 * Authors:
 *   Jan Kašpar
 ****************************************************************************/

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondTools/RunInfo/interface/LHCInfoCombined.h"

//----------------------------------------------------------------------------------------------------

class XangleBetaStarFilter : public edm::stream::EDFilter<> {
public:
  explicit XangleBetaStarFilter(const edm::ParameterSet &);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  edm::ESGetToken<LHCInfo, LHCInfoRcd> lhcInfoToken_;
  edm::ESGetToken<LHCInfoPerLS, LHCInfoPerLSRcd> lhcInfoPerLSToken_;
  edm::ESGetToken<LHCInfoPerFill, LHCInfoPerFillRcd> lhcInfoPerFillToken_;

  bool useNewLHCInfo_;

  double xangle_min_;
  double xangle_max_;

  double beta_star_min_;
  double beta_star_max_;

  bool filter(edm::Event &, const edm::EventSetup &) override;
};

//----------------------------------------------------------------------------------------------------

XangleBetaStarFilter::XangleBetaStarFilter(const edm::ParameterSet &iConfig)
    : lhcInfoToken_(
          esConsumes<LHCInfo, LHCInfoRcd>(edm::ESInputTag{"", iConfig.getParameter<std::string>("lhcInfoLabel")})),
      lhcInfoPerLSToken_(esConsumes<LHCInfoPerLS, LHCInfoPerLSRcd>(
          edm::ESInputTag{"", iConfig.getParameter<std::string>("lhcInfoPerLSLabel")})),
      lhcInfoPerFillToken_(esConsumes<LHCInfoPerFill, LHCInfoPerFillRcd>(
          edm::ESInputTag{"", iConfig.getParameter<std::string>("lhcInfoPerFillLabel")})),

      useNewLHCInfo_(iConfig.getParameter<bool>("useNewLHCInfo")),

      xangle_min_(iConfig.getParameter<double>("xangle_min")),
      xangle_max_(iConfig.getParameter<double>("xangle_max")),
      beta_star_min_(iConfig.getParameter<double>("beta_star_min")),
      beta_star_max_(iConfig.getParameter<double>("beta_star_max")) {}

//----------------------------------------------------------------------------------------------------

void XangleBetaStarFilter::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("lhcInfoLabel", "")->setComment("label of the LHCInfo record");
  desc.add<std::string>("lhcInfoPerLSLabel", "")->setComment("label of the LHCInfoPerLS record");
  desc.add<std::string>("lhcInfoPerFillLabel", "")->setComment("label of the LHCInfoPerFill record");

  desc.add<bool>("useNewLHCInfo", false)->setComment("flag whether to use new LHCInfoPer* records or old LHCInfo");

  desc.add<double>("xangle_min", 0.);
  desc.add<double>("xangle_max", 1000.);

  desc.add<double>("beta_star_min", 0.);
  desc.add<double>("beta_star_max", 1000.);

  descriptions.add("xangleBetaStarFilter", desc);
}

//----------------------------------------------------------------------------------------------------

bool XangleBetaStarFilter::filter(edm::Event & /*iEvent*/, const edm::EventSetup &iSetup) {
  LHCInfoCombined lhcInfoCombined(iSetup, lhcInfoPerLSToken_, lhcInfoPerFillToken_, lhcInfoToken_, useNewLHCInfo_);

  return (xangle_min_ <= lhcInfoCombined.crossingAngle() && lhcInfoCombined.crossingAngle() < xangle_max_) &&
         (beta_star_min_ <= lhcInfoCombined.betaStarX && lhcInfoCombined.betaStarX < beta_star_max_);
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(XangleBetaStarFilter);
