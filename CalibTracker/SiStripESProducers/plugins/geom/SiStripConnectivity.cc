#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CalibTracker/Records/interface/SiStripFecCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include <memory>

class SiStripConnectivity : public edm::ESProducer {
public:
  SiStripConnectivity(const edm::ParameterSet&);
  ~SiStripConnectivity() override;

  std::unique_ptr<SiStripFecCabling> produceFecCabling(const SiStripFecCablingRcd&);
  std::unique_ptr<SiStripDetCabling> produceDetCabling(const SiStripDetCablingRcd&);

private:
  struct FecTokens {
    FecTokens(edm::ESConsumesCollector&& cc) : fedCabling(cc.consumes()) {}
    const edm::ESGetToken<SiStripFedCabling, SiStripFedCablingRcd> fedCabling;
  };
  struct DetTokens {
    DetTokens(edm::ESConsumesCollector&& cc) : fedCabling(cc.consumes()), tTopo(cc.consumes()) {}
    const edm::ESGetToken<SiStripFedCabling, SiStripFedCablingRcd> fedCabling;
    const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopo;
  };
  const FecTokens fecTokens_;
  const DetTokens detTokens_;
};

SiStripConnectivity::SiStripConnectivity(const edm::ParameterSet& p)
    : fecTokens_(setWhatProduced(this, &SiStripConnectivity::produceFecCabling)),
      detTokens_(setWhatProduced(this, &SiStripConnectivity::produceDetCabling)) {}

SiStripConnectivity::~SiStripConnectivity() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

// ------------ methods called to produce the data  ------------

std::unique_ptr<SiStripFecCabling> SiStripConnectivity::produceFecCabling(const SiStripFecCablingRcd& iRecord) {
  //here build an object of type SiStripFecCabling using  **ONLY** the information from class SiStripFedCabling,
  return std::make_unique<SiStripFecCabling>(iRecord.get(fecTokens_.fedCabling));
}

std::unique_ptr<SiStripDetCabling> SiStripConnectivity::produceDetCabling(const SiStripDetCablingRcd& iRecord) {
  //here build an object of type SiStripDetCabling using  **ONLY** the information from class SiStripFedCabling,
  return std::make_unique<SiStripDetCabling>(iRecord.get(detTokens_.fedCabling), &iRecord.get(detTokens_.tTopo));
}

// define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(SiStripConnectivity);
