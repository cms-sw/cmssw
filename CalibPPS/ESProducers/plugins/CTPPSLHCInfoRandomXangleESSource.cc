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
#include "TH1D.h"

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
  double m_betaStar;

  std::unique_ptr<CLHEP::HepRandomEngine> m_engine;

  struct BinData {
    double min, max, xangle;
  };

  std::vector<BinData> binData;
};

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

CTPPSLHCInfoRandomXangleESSource::CTPPSLHCInfoRandomXangleESSource(const edm::ParameterSet &conf)
    : m_label(conf.getParameter<std::string>("label")),

      m_generateEveryNEvents(conf.getParameter<unsigned int>("generateEveryNEvents")),

      m_beamEnergy(conf.getParameter<double>("beamEnergy")),
      m_betaStar(conf.getParameter<double>("betaStar")),

      m_engine(new CLHEP::HepJamesRandom(conf.getParameter<unsigned int>("seed"))) {
  const auto &xangleHistogramFile = conf.getParameter<std::string>("xangleHistogramFile");
  const auto &xangleHistogramObject = conf.getParameter<std::string>("xangleHistogramObject");

  TFile *f_in = TFile::Open(xangleHistogramFile.c_str());
  TH1D *h_xangle = (TH1D *)f_in->Get(xangleHistogramObject.c_str());

  double s = 0.;
  for (int bi = 1; bi <= h_xangle->GetNbinsX(); ++bi)
    s += h_xangle->GetBinContent(bi);

  double cw = 0.;
  for (int bi = 1; bi <= h_xangle->GetNbinsX(); ++bi) {
    double xangle = h_xangle->GetBinCenter(bi);
    double w = h_xangle->GetBinContent(bi) / s;

    binData.push_back({cw, cw + w, xangle});

    cw += w;
  }

  delete f_in;

  setWhatProduced(this, m_label);
  findingRecord<LHCInfoRcd>();
}

//----------------------------------------------------------------------------------------------------

void CTPPSLHCInfoRandomXangleESSource::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("label", "")->setComment("label of the LHCInfo record");

  desc.add<unsigned int>("seed", 1)->setComment("random seed");

  desc.add<unsigned int>("generateEveryNEvents", 1)->setComment("how often to generate new xangle");

  desc.add<std::string>("xangleHistogramFile", "")->setComment("ROOT file with xangle distribution");
  desc.add<std::string>("xangleHistogramObject", "")->setComment("xangle distribution object in the ROOT file");

  desc.add<double>("beamEnergy", 0.)->setComment("beam energy");
  desc.add<double>("betaStar", 0.)->setComment("beta*");

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

  const double u = CLHEP::RandFlat::shoot(m_engine.get(), 0., 1.);

  double xangle = 0.;
  for (const auto &d : binData) {
    if (d.min <= u && u <= d.max) {
      xangle = d.xangle;
      break;
    }
  }

  output->setEnergy(m_beamEnergy);
  output->setCrossingAngle(xangle);
  output->setBetaStar(m_betaStar);

  return edm::es::products(std::move(output));
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_EVENTSETUP_SOURCE(CTPPSLHCInfoRandomXangleESSource);
