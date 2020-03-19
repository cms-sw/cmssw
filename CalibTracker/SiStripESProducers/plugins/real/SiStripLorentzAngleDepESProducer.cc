// -*- C++ -*-
//
// Package:    SiStripLorentzAngleDepESProducer
// Class:      SiStripLorentzAngleDepESProducer
//
/**\class SiStripLorentzAngleDepESProducer SiStripLorentzAngleDepESProducer.h CalibTracker/SiStripESProducers/plugins/real/SiStripLorentzAngleDepESProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Michael Segala and Rebeca Gonzalez Suarez
//         Created:  15/02/2011
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Utilities/interface/ESProductTag.h"

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"

class SiStripLorentzAngleDepESProducer : public edm::ESProducer {
public:
  SiStripLorentzAngleDepESProducer(const edm::ParameterSet&);

  std::shared_ptr<SiStripLorentzAngle const> produce(const SiStripLorentzAngleDepRcd&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::ESGetToken<SiStripLorentzAngle, SiStripLorentzAngleRcd> lorentzAngleToken_;
};

SiStripLorentzAngleDepESProducer::SiStripLorentzAngleDepESProducer(const edm::ParameterSet& iConfig) {
  auto const getLatency = iConfig.getParameter<edm::ParameterSet>("LatencyRecord");
  // How useful the "record" parameter really is?
  if (getLatency.getParameter<std::string>("record") != "SiStripLatencyRcd") {
    throw edm::Exception(edm::errors::Configuration,
                         "[SiStripLorentzAngleDepESProducer::ctor] No Latency Record found ");
  }

  auto const getPeak = iConfig.getParameter<edm::ParameterSet>("LorentzAnglePeakMode");
  if (getPeak.getParameter<std::string>("record") != "SiStripLorentzAngleRcd") {
    throw edm::Exception(edm::errors::Configuration,
                         "[SiStripLorentzAngleDepESProducer::ctor] No Lorentz Angle Record found ");
  }

  auto const getDeconv = iConfig.getParameter<edm::ParameterSet>("LorentzAngleDeconvMode");
  // How useful the "record" parameter really is?
  if (getDeconv.getParameter<std::string>("record") != "SiStripLorentzAngleRcd") {
    throw edm::Exception(edm::errors::Configuration,
                         "[SiStripLorentzAngleDepESProducer::ctor] No Lorentz Angle Record found ");
  }

  auto const peakLabel{getPeak.getUntrackedParameter<std::string>("label")};
  auto const deconvLabel{getDeconv.getUntrackedParameter<std::string>("label")};
  setWhatProduced(this).setMayConsume(
      lorentzAngleToken_,
      [peakLabel, deconvLabel](auto const& get, edm::ESTransientHandle<SiStripLatency> iLatency) {
        if (iLatency->singleReadOutMode() == 1) {
          return get("", peakLabel);
        }
        return get("", deconvLabel);
      },
      edm::ESProductTag<SiStripLatency, SiStripLatencyRcd>("", getLatency.getUntrackedParameter<std::string>("label")));

  edm::LogInfo("SiStripLorentzAngleDepESProducer") << "ctor" << std::endl;
}

std::shared_ptr<SiStripLorentzAngle const> SiStripLorentzAngleDepESProducer::produce(
    const SiStripLorentzAngleDepRcd& iRecord) {
  edm::LogInfo("SiStripLorentzAngleDepESProducer") << "Producer called" << std::endl;

  //tell shared_ptr not to delete the product since it is already owned by the record
  return std::shared_ptr<SiStripLorentzAngle const>(&iRecord.get(lorentzAngleToken_), [](auto) {});
}

void SiStripLorentzAngleDepESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  {
    edm::ParameterSetDescription latency;
    latency.add<std::string>("record", "SiStripLatencyRcd");
    latency.addUntracked<std::string>("label", "");

    desc.add<edm::ParameterSetDescription>("LatencyRecord", latency);
  }

  {
    edm::ParameterSetDescription peak;
    peak.add<std::string>("record", "SiStripLorentzAngleRcd");
    peak.addUntracked<std::string>("label", "peak");

    desc.add<edm::ParameterSetDescription>("LorentzAnglePeakMode", peak);
  }

  {
    edm::ParameterSetDescription deconv;
    deconv.add<std::string>("record", "SiStripLorentzAngleRcd");
    deconv.addUntracked<std::string>("label", "deconvolution");

    desc.add<edm::ParameterSetDescription>("LorentzAngleDeconvMode", deconv);
  }

  descriptions.addDefault(desc);
}

DEFINE_FWK_EVENTSETUP_MODULE(SiStripLorentzAngleDepESProducer);
