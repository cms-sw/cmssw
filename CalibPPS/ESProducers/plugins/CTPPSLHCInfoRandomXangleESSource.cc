// Original Author:  Jan Kašpar

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/JamesRandom.h"

#include "TFile.h"
#include "TH2D.h"

//----------------------------------------------------------------------------------------------------

/**
 * \brief Provides LHCInfo data necessary for CTPPS reconstruction (and direct simulation).
 **/
class CTPPSLHCInfoRandomXangleESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  CTPPSLHCInfoRandomXangleESSource(const edm::ParameterSet &);
  edm::ESProducts<std::unique_ptr<LHCInfo>> produce(const LHCInfoRcd &);
  static void fillDescriptions(edm::ConfigurationDescriptions &);

private:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
                      const edm::IOVSyncValue &,
                      edm::ValidityInterval &) override;

  std::string m_label;

  unsigned int m_generateEveryNEvents;

  double m_beamEnergy;

  std::unique_ptr<CLHEP::HepRandomEngine> m_engine;

  struct BinData {
    double min, max;
    double xangle, betaStar;
  };

  std::vector<BinData> xangleBetaStarBins;
};

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

CTPPSLHCInfoRandomXangleESSource::CTPPSLHCInfoRandomXangleESSource(const edm::ParameterSet &conf)
    : m_label(conf.getParameter<std::string>("label")),

      m_generateEveryNEvents(conf.getParameter<unsigned int>("generateEveryNEvents")),

      m_beamEnergy(conf.getParameter<double>("beamEnergy")),

      m_engine(new CLHEP::HepJamesRandom(conf.getParameter<unsigned int>("seed"))) {
  // get input beta* vs. xangle histogram
  const auto &xangleBetaStarHistogramFile = conf.getParameter<std::string>("xangleBetaStarHistogramFile");
  const auto &xangleBetaStarHistogramObject = conf.getParameter<std::string>("xangleBetaStarHistogramObject");

  edm::FileInPath fip(xangleBetaStarHistogramFile);
  std::unique_ptr<TFile> f_in(TFile::Open(fip.fullPath().c_str()));
  if (!f_in)
    throw cms::Exception("PPS") << "Cannot open input file '" << xangleBetaStarHistogramFile << "'.";

  TH2D *h_xangle_beta_star = (TH2D *)f_in->Get(xangleBetaStarHistogramObject.c_str());
  if (!h_xangle_beta_star)
    throw cms::Exception("PPS") << "Cannot load input object '" << xangleBetaStarHistogramObject << "'.";

  // parse histogram
  double sum = 0.;
  for (int x = 1; x <= h_xangle_beta_star->GetNbinsX(); ++x) {
    for (int y = 1; y <= h_xangle_beta_star->GetNbinsY(); ++y)
      sum += h_xangle_beta_star->GetBinContent(x, y);
  }

  double cw = 0;
  for (int x = 1; x <= h_xangle_beta_star->GetNbinsX(); ++x) {
    for (int y = 1; y <= h_xangle_beta_star->GetNbinsY(); ++y) {
      const double c = h_xangle_beta_star->GetBinContent(x, y);
      const double xangle = h_xangle_beta_star->GetXaxis()->GetBinCenter(x);
      const double betaStar = h_xangle_beta_star->GetYaxis()->GetBinCenter(y);

      if (c > 0.) {
        const double rc = c / sum;
        xangleBetaStarBins.push_back({cw, cw + rc, xangle, betaStar});
        cw += rc;
      }
    }
  }

  setWhatProduced(this, m_label);
  findingRecord<LHCInfoRcd>();
}

//----------------------------------------------------------------------------------------------------

void CTPPSLHCInfoRandomXangleESSource::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("label", "")->setComment("label of the LHCInfo record");

  desc.add<unsigned int>("seed", 1)->setComment("random seed");

  desc.add<unsigned int>("generateEveryNEvents", 1)->setComment("how often to generate new xangle");

  desc.add<std::string>("xangleBetaStarHistogramFile", "")->setComment("ROOT file with xangle distribution");
  desc.add<std::string>("xangleBetaStarHistogramObject", "")->setComment("xangle distribution object in the ROOT file");

  desc.add<double>("beamEnergy", 0.)->setComment("beam energy");

  descriptions.add("ctppsLHCInfoRandomXangleESSource", desc);
}

//----------------------------------------------------------------------------------------------------

void CTPPSLHCInfoRandomXangleESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &key,
                                                      const edm::IOVSyncValue &iosv,
                                                      edm::ValidityInterval &oValidity) {
  edm::EventID beginEvent = iosv.eventID();
  edm::EventID endEvent(beginEvent.run(), beginEvent.luminosityBlock(), beginEvent.event() + m_generateEveryNEvents);
  oValidity = edm::ValidityInterval(edm::IOVSyncValue(beginEvent), edm::IOVSyncValue(endEvent));
}

//----------------------------------------------------------------------------------------------------

edm::ESProducts<std::unique_ptr<LHCInfo>> CTPPSLHCInfoRandomXangleESSource::produce(const LHCInfoRcd &) {
  auto output = std::make_unique<LHCInfo>();

  double xangle = 0., betaStar = 0.;
  const double u = CLHEP::RandFlat::shoot(m_engine.get(), 0., 1.);
  for (const auto &d : xangleBetaStarBins) {
    if (d.min <= u && u <= d.max) {
      xangle = d.xangle;
      betaStar = d.betaStar;
      break;
    }
  }

  output->setEnergy(m_beamEnergy);
  output->setCrossingAngle(xangle);
  output->setBetaStar(betaStar);

  return edm::es::products(std::move(output));
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_EVENTSETUP_SOURCE(CTPPSLHCInfoRandomXangleESSource);