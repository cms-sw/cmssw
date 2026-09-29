// Original Author:  Jan Kašpar

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducts.h"

#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"

//----------------------------------------------------------------------------------------------------

/**
 * \brief Provides LHCInfo data necessary for CTPPS reconstruction (and direct simulation).
 **/
class CTPPSLHCInfoESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  CTPPSLHCInfoESSource(const edm::ParameterSet &);
  edm::ESProducts<std::unique_ptr<LHCInfo>> produce(const LHCInfoRcd &);
  static void fillDescriptions(edm::ConfigurationDescriptions &);

private:
  bool isConcurrentFinder() const override { return true; }
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
                      const edm::IOVSyncValue &,
                      edm::ValidityInterval &) override;

  const std::string m_label;

  const edm::EventRange m_validityRange;
  const double m_beamEnergy;
  const double m_betaStar;
  const double m_xangle;
};

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

CTPPSLHCInfoESSource::CTPPSLHCInfoESSource(const edm::ParameterSet &conf)
    : m_label(conf.getParameter<std::string>("label")),
      m_validityRange(conf.getParameter<edm::EventRange>("validityRange")),
      m_beamEnergy(conf.getParameter<double>("beamEnergy")),
      m_betaStar(conf.getParameter<double>("betaStar")),
      m_xangle(conf.getParameter<double>("xangle")) {
  setWhatProduced(this, m_label);
  findingRecord<LHCInfoRcd>();
}

//----------------------------------------------------------------------------------------------------

void CTPPSLHCInfoESSource::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("label", "")->setComment("label of the LHCInfo record");

  desc.add<edm::EventRange>("validityRange", edm::EventRange())->setComment("interval of validity");

  desc.add<double>("beamEnergy", 0.)->setComment("beam energy");
  desc.add<double>("betaStar", 0.)->setComment("beta*");
  desc.add<double>("xangle", 0.)->setComment("crossing angle");

  descriptions.add("ctppsLHCInfoESSource", desc);
}

//----------------------------------------------------------------------------------------------------

void CTPPSLHCInfoESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &key,
                                          const edm::IOVSyncValue &iosv,
                                          edm::ValidityInterval &oValidity) {
  if (edm::contains(m_validityRange, iosv.eventID())) {
    oValidity = edm::ValidityInterval(edm::IOVSyncValue(m_validityRange.startEventID()),
                                      edm::IOVSyncValue(m_validityRange.endEventID()));
  } else {
    if (iosv.eventID() < m_validityRange.startEventID()) {
      edm::RunNumber_t run = m_validityRange.startEventID().run();
      edm::LuminosityBlockNumber_t lb = m_validityRange.startEventID().luminosityBlock();
      edm::EventID endEvent =
          (lb > 1) ? edm::EventID(run, lb - 1, 0) : edm::EventID(run - 1, edm::EventID::maxLuminosityBlockNumber(), 0);

      oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue(endEvent));
    } else {
      edm::RunNumber_t run = m_validityRange.startEventID().run();
      edm::LuminosityBlockNumber_t lb = m_validityRange.startEventID().luminosityBlock();
      edm::EventID beginEvent = (lb < edm::EventID::maxLuminosityBlockNumber() - 1) ? edm::EventID(run, lb + 1, 0)
                                                                                    : edm::EventID(run + 1, 0, 0);

      oValidity = edm::ValidityInterval(edm::IOVSyncValue(beginEvent), edm::IOVSyncValue::endOfTime());
    }
  }
}

//----------------------------------------------------------------------------------------------------

edm::ESProducts<std::unique_ptr<LHCInfo>> CTPPSLHCInfoESSource::produce(const LHCInfoRcd &iRcd) {
  auto output = std::make_unique<LHCInfo>();

  if (edm::contains(m_validityRange, iRcd.validityInterval().first().eventID())) {
    output->setEnergy(m_beamEnergy);
    output->setBetaStar(m_betaStar);
    output->setCrossingAngle(m_xangle);
  } else {
    output->setEnergy(0.);
    output->setBetaStar(0.);
    output->setCrossingAngle(0.);
  }

  return edm::es::products(std::move(output));
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_EVENTSETUP_SOURCE(CTPPSLHCInfoESSource);
