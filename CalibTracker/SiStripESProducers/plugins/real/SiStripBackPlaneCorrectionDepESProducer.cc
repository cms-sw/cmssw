// -*- C++ -*-
//
// Package:    SiStripBackPlaneCorrectionDepESProducer
// Class:      SiStripBackPlaneCorrectionDepESProducer
//
/**\class SiStripBackPlaneCorrectionDepESProducer SiStripBackPlaneCorrectionDepESProducer.h CalibTracker/SiStripESProducers/plugins/real/SiStripBackPlaneCorrectionDepESProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Loic Quertenmont inspired from the SiStripLorentzAngleDepESProducer
//         Created:  07/03/2013
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

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "CondFormats/SiStripObjects/interface/SiStripBackPlaneCorrection.h"
#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"

#include "FWCore/Utilities/interface/ESProductTag.h"

class SiStripBackPlaneCorrectionDepESProducer : public edm::ESProducer {
public:
  SiStripBackPlaneCorrectionDepESProducer(const edm::ParameterSet&);

  std::shared_ptr<SiStripBackPlaneCorrection const> produce(const SiStripBackPlaneCorrectionDepRcd&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::ESGetToken<SiStripBackPlaneCorrection, SiStripBackPlaneCorrectionRcd> backPlaneCorrectionToken_;
};

SiStripBackPlaneCorrectionDepESProducer::SiStripBackPlaneCorrectionDepESProducer(const edm::ParameterSet& iConfig) {
  edm::LogInfo("SiStripBackPlaneCorrectionDepESProducer") << "ctor" << std::endl;

  auto getLatency = iConfig.getParameter<edm::ParameterSet>("LatencyRecord");
  // How useful the "record" parameter really is?
  if (getLatency.getParameter<std::string>("record") != "SiStripLatencyRcd") {
    throw edm::Exception(edm::errors::Configuration,
                         "[SiStripBackPlaneCorrectionDepESProducer::ctor] No Latency Record found ");
  }

  auto getPeak = iConfig.getParameter<edm::ParameterSet>("BackPlaneCorrectionPeakMode");
  if (getPeak.getParameter<std::string>("record") != "SiStripBackPlaneCorrectionRcd") {
    throw edm::Exception(edm::errors::Configuration,
                         "[SiStripBackPlaneCorrectionDepESProducer::ctor] No Lorentz Angle Record found ");
  }

  auto getDeconv = iConfig.getParameter<edm::ParameterSet>("BackPlaneCorrectionDeconvMode");
  // How useful the "record" parameter really is?
  if (getDeconv.getParameter<std::string>("record") != "SiStripBackPlaneCorrectionRcd") {
    throw edm::Exception(edm::errors::Configuration,
                         "[SiStripBackPlaneCorrectionDepESProducer::ctor] No Lorentz Angle Record found ");
  }

  auto peakLabel{getPeak.getUntrackedParameter<std::string>("label")};
  auto deconvLabel{getDeconv.getUntrackedParameter<std::string>("label")};

  setWhatProduced(this).setMayConsume(
      backPlaneCorrectionToken_,
      [peakLabel, deconvLabel](auto const& get, edm::ESTransientHandle<SiStripLatency> iLatency) {
        if (iLatency->singleReadOutMode() == 1) {
          return get("", peakLabel);
        }
        return get("", deconvLabel);
      },
      edm::ESProductTag<SiStripLatency, SiStripLatencyRcd>("", getLatency.getUntrackedParameter<std::string>("label")));
}

std::shared_ptr<SiStripBackPlaneCorrection const> SiStripBackPlaneCorrectionDepESProducer::produce(
    const SiStripBackPlaneCorrectionDepRcd& iRecord) {
  edm::LogInfo("SiStripBackPlaneCorrectionDepESProducer") << "Producer called" << std::endl;

  //tell shared_ptr not to delete the product since it is already owned by the record
  return std::shared_ptr<SiStripBackPlaneCorrection const>(&iRecord.get(backPlaneCorrectionToken_), [](auto) {});
}

void SiStripBackPlaneCorrectionDepESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  {
    edm::ParameterSetDescription latency;
    latency.add<std::string>("record", "SiStripLatencyRcd");
    latency.addUntracked<std::string>("label", "");

    desc.add<edm::ParameterSetDescription>("LatencyRecord", latency);
  }

  {
    edm::ParameterSetDescription peak;
    peak.add<std::string>("record", "SiStripBackPlaneCorrectionRcd");
    peak.addUntracked<std::string>("label", "peak");

    desc.add<edm::ParameterSetDescription>("BackPlaneCorrectionPeakMode", peak);
  }

  {
    edm::ParameterSetDescription deconv;
    deconv.add<std::string>("record", "SiStripBackPlaneCorrectionRcd");
    deconv.addUntracked<std::string>("label", "deconvolution");

    desc.add<edm::ParameterSetDescription>("BackPlaneCorrectionDeconvMode", deconv);
  }

  descriptions.addDefault(desc);
}

DEFINE_FWK_EVENTSETUP_MODULE(SiStripBackPlaneCorrectionDepESProducer);
