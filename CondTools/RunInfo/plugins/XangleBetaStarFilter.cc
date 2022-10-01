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
#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"

//----------------------------------------------------------------------------------------------------

class XangleBetaStarFilter : public edm::stream::EDFilter<> {
public:
  explicit XangleBetaStarFilter(const edm::ParameterSet &);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  edm::ESGetToken<LHCInfo, LHCInfoRcd> lhcInfoToken_;

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

      xangle_min_(iConfig.getParameter<double>("xangle_min")),
      xangle_max_(iConfig.getParameter<double>("xangle_max")),
      beta_star_min_(iConfig.getParameter<double>("beta_star_min")),
      beta_star_max_(iConfig.getParameter<double>("beta_star_max")) {}

//----------------------------------------------------------------------------------------------------

void XangleBetaStarFilter::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("lhcInfoLabel", "")->setComment("label of the LHCInfo record");

  desc.add<double>("xangle_min", 0.);
  desc.add<double>("xangle_max", 1000.);

  desc.add<double>("beta_star_min", 0.);
  desc.add<double>("beta_star_max", 1000.);

  descriptions.add("xangleBetaStarFilter", desc);
}

//----------------------------------------------------------------------------------------------------

bool XangleBetaStarFilter::filter(edm::Event & /*iEvent*/, const edm::EventSetup &iSetup) {
  const auto &lhcInfo = iSetup.getData(lhcInfoToken_);

  return (xangle_min_ <= lhcInfo.crossingAngle() && lhcInfo.crossingAngle() < xangle_max_) &&
         (beta_star_min_ <= lhcInfo.betaStar() && lhcInfo.betaStar() < beta_star_max_);
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(XangleBetaStarFilter);
