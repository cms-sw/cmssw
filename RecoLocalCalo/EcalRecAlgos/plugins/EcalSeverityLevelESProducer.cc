#include <memory>
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"

/*
  Provide  a hook to retrieve the EcalSeverityLevelAlgo
  through the EventSetup 
 
  Appartently there is no smarter way to do it in CMSSW

  Author: Stefano Argiro
 */

class EcalSeverityLevelESProducer : public edm::ESProducer {
public:
  EcalSeverityLevelESProducer(const edm::ParameterSet& iConfig);

  typedef std::shared_ptr<EcalSeverityLevelAlgo> ReturnType;

  ReturnType produce(const EcalSeverityLevelAlgoRcd& iRecord);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void setupChannelStatus(const EcalChannelStatusRcd&, EcalSeverityLevelAlgo*);

  using HostType = edm::ESProductHost<EcalSeverityLevelAlgo, EcalChannelStatusRcd>;

  edm::ReusableObjectHolder<HostType> holder_;
  edm::ParameterSet const pset_;
  edm::ESGetToken<EcalChannelStatus, EcalChannelStatusRcd> const channelToken_;
};

EcalSeverityLevelESProducer::EcalSeverityLevelESProducer(const edm::ParameterSet& iConfig)
    : pset_(iConfig), channelToken_(setWhatProduced(this).consumesFrom<EcalChannelStatus, EcalChannelStatusRcd>()) {}

EcalSeverityLevelESProducer::ReturnType EcalSeverityLevelESProducer::produce(const EcalSeverityLevelAlgoRcd& iRecord) {
  auto host = holder_.makeOrGet([this]() { return new HostType(pset_); });

  host->ifRecordChanges<EcalChannelStatusRcd>(iRecord,
                                              [this, h = host.get()](auto const& rec) { setupChannelStatus(rec, h); });

  return host;
}

void EcalSeverityLevelESProducer::setupChannelStatus(const EcalChannelStatusRcd& chs, EcalSeverityLevelAlgo* algo) {
  algo->setChannelStatus(chs.get(channelToken_));
}

void EcalSeverityLevelESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::vector<std::string>>("kGood",
                                       {
                                           "kGood",
                                       });
    psd0.add<std::vector<std::string>>("kProblematic",
                                       {
                                           "kPoorReco",
                                           "kPoorCalib",
                                           "kNoisy",
                                           "kSaturated",
                                       });
    psd0.add<std::vector<std::string>>("kRecovered",
                                       {
                                           "kLeadingEdgeRecovered",
                                           "kTowerRecovered",
                                       });
    psd0.add<std::vector<std::string>>("kTime",
                                       {
                                           "kOutOfTime",
                                       });
    psd0.add<std::vector<std::string>>("kWeird",
                                       {
                                           "kWeird",
                                           "kDiWeird",
                                       });
    psd0.add<std::vector<std::string>>("kBad",
                                       {
                                           "kFaultyHardware",
                                           "kDead",
                                           "kKilled",
                                       });
    desc.add<edm::ParameterSetDescription>("flagMask", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::vector<std::string>>("kGood",
                                       {
                                           "kOk",
                                       });
    psd0.add<std::vector<std::string>>("kProblematic",
                                       {
                                           "kDAC",
                                           "kNoLaser",
                                           "kNoisy",
                                           "kNNoisy",
                                           "kNNNoisy",
                                           "kNNNNoisy",
                                           "kNNNNNoisy",
                                           "kFixedG6",
                                           "kFixedG1",
                                           "kFixedG0",
                                       });
    psd0.add<std::vector<std::string>>("kRecovered", {});
    psd0.add<std::vector<std::string>>("kTime", {});
    psd0.add<std::vector<std::string>>("kWeird", {});
    psd0.add<std::vector<std::string>>("kBad",
                                       {
                                           "kNonRespondingIsolated",
                                           "kDeadVFE",
                                           "kDeadFE",
                                           "kNoDataNoTP",
                                       });
    desc.add<edm::ParameterSetDescription>("dbstatusMask", psd0);
  }
  desc.add<double>("timeThresh", 2.0);
  descriptions.add("ecalSeverityLevel", desc);
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(EcalSeverityLevelESProducer);
