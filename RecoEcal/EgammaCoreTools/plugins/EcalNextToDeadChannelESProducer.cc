#include <memory>
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalNextToDeadChannel.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalNextToDeadChannelRcd.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalTools.h"

/**
   ESProducer to fill the EcalNextToDeadChannel record
   starting from EcalChannelStatus information
   

   \author Stefano Argiro
   \date 18 May 2011
*/

class EcalNextToDeadChannelESProducer : public edm::ESProducer {
public:
  EcalNextToDeadChannelESProducer(const edm::ParameterSet& iConfig);

  typedef std::shared_ptr<EcalNextToDeadChannel> ReturnType;

  ReturnType produce(const EcalNextToDeadChannelRcd& iRecord);

private:
  void setupNextToDeadChannels(const EcalChannelStatusRcd&, EcalNextToDeadChannel*);

  using HostType = edm::ESProductHost<EcalNextToDeadChannel, EcalChannelStatusRcd>;

  edm::ReusableObjectHolder<HostType> holder_;

  edm::ESGetToken<EcalChannelStatus, EcalChannelStatusRcd> const channelToken_;
  // threshold above which a channel will be considered "dead"
  int statusThreshold_;
};

EcalNextToDeadChannelESProducer::EcalNextToDeadChannelESProducer(const edm::ParameterSet& iConfig)
    : channelToken_(setWhatProduced(this).consumesFrom<EcalChannelStatus, EcalChannelStatusRcd>()) {
  statusThreshold_ = iConfig.getParameter<int>("channelStatusThresholdForDead");
}

EcalNextToDeadChannelESProducer::ReturnType EcalNextToDeadChannelESProducer::produce(
    const EcalNextToDeadChannelRcd& iRecord) {
  auto host = holder_.makeOrGet([]() { return new HostType; });

  host->ifRecordChanges<EcalChannelStatusRcd>(
      iRecord, [this, h = host.get()](auto const& rec) { setupNextToDeadChannels(rec, h); });

  return host;
}

void EcalNextToDeadChannelESProducer::setupNextToDeadChannels(const EcalChannelStatusRcd& chs,
                                                              EcalNextToDeadChannel* rcd) {
  rcd->clear();

  // Find channels next to dead ones and fill corresponding record

  EcalChannelStatus const& h = chs.get(channelToken_);

  for (int ieta = -EBDetId::MAX_IETA; ieta <= EBDetId::MAX_IETA; ++ieta) {
    if (ieta == 0)
      continue;
    for (int iphi = EBDetId::MIN_IPHI; iphi <= EBDetId::MAX_IPHI; ++iphi) {
      if (EBDetId::validDetId(ieta, iphi)) {
        EBDetId detid(ieta, iphi);

        if (EcalTools::isNextToDeadFromNeighbours(detid, h, statusThreshold_)) {
          rcd->setValue(detid, 1);
        };
      }
    }  // for phi
  }    // for eta

  // endcap

  for (int iX = EEDetId::IX_MIN; iX <= EEDetId::IX_MAX; ++iX) {
    for (int iY = EEDetId::IY_MIN; iY <= EEDetId::IY_MAX; ++iY) {
      if (EEDetId::validDetId(iX, iY, 1)) {
        EEDetId detid(iX, iY, 1);

        if (EcalTools::isNextToDeadFromNeighbours(detid, h, statusThreshold_)) {
          rcd->setValue(detid, 1);
        };
      }

      if (EEDetId::validDetId(iX, iY, -1)) {
        EEDetId detid(iX, iY, -1);

        if (EcalTools::isNextToDeadFromNeighbours(detid, h, statusThreshold_)) {
          rcd->setValue(detid, 1);
        };
      }
    }  // for iy
  }    // for ix
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(EcalNextToDeadChannelESProducer);
