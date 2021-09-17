#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTInputProducer.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCT.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h"

#include <vector>
using std::vector;

#include <iostream>
using std::cout;
using std::endl;

L1RCTInputProducer::L1RCTInputProducer(const edm::ParameterSet &conf)
    : rctLookupTables(new L1RCTLookupTables),
      rct(new L1RCT(rctLookupTables)),
      useEcal(conf.getParameter<bool>("useEcal")),
      useHcal(conf.getParameter<bool>("useHcal")),
      ecalDigisLabel(conf.getParameter<edm::InputTag>("ecalDigisLabel")),
      hcalDigisLabel(conf.getParameter<edm::InputTag>("hcalDigisLabel")),
      rctParametersToken(esConsumes<L1RCTParameters, L1RCTParametersRcd>()),
      channelMaskToken(esConsumes<L1RCTChannelMask, L1RCTChannelMaskRcd>()),
      ecalScaleToken(esConsumes<L1CaloEcalScale, L1CaloEcalScaleRcd>()),
      hcalScaleToken(esConsumes<L1CaloHcalScale, L1CaloHcalScaleRcd>()),
      emScaleToken(esConsumes<L1CaloEtScale, L1EmEtScaleRcd>()) {
  produces<std::vector<unsigned short>>("rctCrate");
  produces<std::vector<unsigned short>>("rctCard");
  produces<std::vector<unsigned short>>("rctTower");
  produces<std::vector<unsigned int>>("rctEGammaET");
  produces<std::vector<bool>>("rctHoEFGVetoBit");
  produces<std::vector<unsigned int>>("rctJetMETET");
  produces<std::vector<bool>>("rctTowerActivityBit");
  produces<std::vector<bool>>("rctTowerMIPBit");
  produces<std::vector<unsigned short>>("rctHFCrate");
  produces<std::vector<unsigned short>>("rctHFRegion");
  produces<std::vector<unsigned int>>("rctHFET");
  produces<std::vector<bool>>("rctHFFG");
}

L1RCTInputProducer::~L1RCTInputProducer() {
  if (rct != nullptr)
    delete rct;
  if (rctLookupTables != nullptr)
    delete rctLookupTables;
}

void L1RCTInputProducer::produce(edm::Event &event, const edm::EventSetup &eventSetup) {
  // Refresh configuration information every event
  // Hopefully, this does not take too much time
  // There should be a call back function in future to
  // handle changes in configuration

  edm::ESHandle<L1RCTParameters> rctParameters = eventSetup.getHandle(rctParametersToken);
  const L1RCTParameters *r = rctParameters.product();
  edm::ESHandle<L1RCTChannelMask> channelMask = eventSetup.getHandle(channelMaskToken);
  const L1RCTChannelMask *c = channelMask.product();
  edm::ESHandle<L1CaloEcalScale> ecalScale = eventSetup.getHandle(ecalScaleToken);
  const L1CaloEcalScale *e = ecalScale.product();
  edm::ESHandle<L1CaloHcalScale> hcalScale = eventSetup.getHandle(hcalScaleToken);
  const L1CaloHcalScale *h = hcalScale.product();
  edm::ESHandle<L1CaloEtScale> emScale = eventSetup.getHandle(emScaleToken);
  const L1CaloEtScale *s = emScale.product();

  rctLookupTables->setRCTParameters(r);
  rctLookupTables->setChannelMask(c);
  rctLookupTables->setHcalScale(h);
  rctLookupTables->setEcalScale(e);
  rctLookupTables->setL1CaloEtScale(s);

  edm::Handle<EcalTrigPrimDigiCollection> ecal;
  edm::Handle<HcalTrigPrimDigiCollection> hcal;

  if (useEcal) {
    event.getByLabel(ecalDigisLabel, ecal);
  }
  if (useHcal) {
    event.getByLabel(hcalDigisLabel, hcal);
  }

  EcalTrigPrimDigiCollection ecalColl;
  HcalTrigPrimDigiCollection hcalColl;
  if (ecal.isValid()) {
    ecalColl = *ecal;
  }
  if (hcal.isValid()) {
    hcalColl = *hcal;
  }

  rct->digiInput(ecalColl, hcalColl);

  // Stuff to create

  std::unique_ptr<std::vector<unsigned short>> rctCrate(new std::vector<unsigned short>);
  std::unique_ptr<std::vector<unsigned short>> rctCard(new std::vector<unsigned short>);
  std::unique_ptr<std::vector<unsigned short>> rctTower(new std::vector<unsigned short>);
  std::unique_ptr<std::vector<unsigned int>> rctEGammaET(new std::vector<unsigned int>);
  std::unique_ptr<std::vector<bool>> rctHoEFGVetoBit(new std::vector<bool>);
  std::unique_ptr<std::vector<unsigned int>> rctJetMETET(new std::vector<unsigned int>);
  std::unique_ptr<std::vector<bool>> rctTowerActivityBit(new std::vector<bool>);
  std::unique_ptr<std::vector<bool>> rctTowerMIPBit(new std::vector<bool>);

  for (int crate = 0; crate < 18; crate++) {
    for (int card = 0; card < 7; card++) {
      for (int tower = 0; tower < 32; tower++) {
        unsigned short ecalCompressedET = rct->ecalCompressedET(crate, card, tower);
        unsigned short ecalFineGrainBit = rct->ecalFineGrainBit(crate, card, tower);
        unsigned short hcalCompressedET = rct->hcalCompressedET(crate, card, tower);
        unsigned int lutBits =
            rctLookupTables->lookup(ecalCompressedET, hcalCompressedET, ecalFineGrainBit, crate, card, tower);
        unsigned int eGammaETCode = lutBits & 0x0000007F;
        bool hOeFGVetoBit = (lutBits >> 7) & 0x00000001;
        unsigned int jetMETETCode = (lutBits >> 8) & 0x000001FF;
        bool activityBit = (lutBits >> 17) & 0x00000001;
        if (eGammaETCode > 0 || jetMETETCode > 0 || hOeFGVetoBit || activityBit) {
          rctCrate->push_back(crate);
          rctCard->push_back(card);
          rctTower->push_back(tower);
          rctEGammaET->push_back(eGammaETCode);
          rctHoEFGVetoBit->push_back(hOeFGVetoBit);
          rctJetMETET->push_back(jetMETETCode);
          rctTowerActivityBit->push_back(activityBit);
          rctTowerMIPBit->push_back(false);  // FIXME: MIP bit is not yet defined
        }
      }
    }
  }

  std::unique_ptr<std::vector<unsigned short>> rctHFCrate(new std::vector<unsigned short>);
  std::unique_ptr<std::vector<unsigned short>> rctHFRegion(new std::vector<unsigned short>);
  std::unique_ptr<std::vector<unsigned int>> rctHFET(new std::vector<unsigned int>);
  std::unique_ptr<std::vector<bool>> rctHFFG(new std::vector<bool>);
  for (int crate = 0; crate < 18; crate++) {
    for (int hfRegion = 0; hfRegion < 8; hfRegion++) {
      unsigned short hfCompressedET = rct->hfCompressedET(crate, hfRegion);
      unsigned int hfETCode = rctLookupTables->lookup(hfCompressedET, crate, 999, hfRegion);
      if (hfETCode > 0) {
        rctHFCrate->push_back(crate);
        rctHFRegion->push_back(hfRegion);
        rctHFET->push_back(hfETCode);
        rctHFFG->push_back(false);  // FIXME: HF FG is not yet defined
      }
    }
  }

  // putting stuff back into event
  event.put(std::move(rctCrate), "rctCrate");
  event.put(std::move(rctCard), "rctCard");
  event.put(std::move(rctTower), "rctTower");
  event.put(std::move(rctEGammaET), "rctEGammaET");
  event.put(std::move(rctHoEFGVetoBit), "rctHoEFGVetoBit");
  event.put(std::move(rctJetMETET), "rctJetMETET");
  event.put(std::move(rctTowerActivityBit), "rctTowerActivityBit");
  event.put(std::move(rctTowerMIPBit), "rctTowerMIPBit");
  event.put(std::move(rctHFCrate), "rctHFCrate");
  event.put(std::move(rctHFRegion), "rctHFRegion");
  event.put(std::move(rctHFET), "rctHFET");
  event.put(std::move(rctHFFG), "rctHFFG");
}
