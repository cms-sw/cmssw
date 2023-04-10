// -*- C++ -*-
//
// Package:    L1Trigger/L1TCaloLayer1
// Class:      L1TCaloLayer1
//
/**\class L1TCaloLayer1 L1TCaloLayer1.cc L1Trigger/L1TCaloLayer1/plugins/L1TCaloLayer1.cc

   Description: [one line class summary]

   Implementation:
   [Notes on implementation]
*/
//
// Original Author:  Sridhara Rao Dasu
//         Created:  Thu, 08 Oct 2015 09:20:16 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "L1Trigger/L1TCaloLayer1/src/UCTLayer1.hh"
#include "L1Trigger/L1TCaloLayer1/src/UCTCrate.hh"
#include "L1Trigger/L1TCaloLayer1/src/UCTCard.hh"
#include "L1Trigger/L1TCaloLayer1/src/UCTRegion.hh"
#include "L1Trigger/L1TCaloLayer1/src/UCTTower.hh"

#include "L1Trigger/L1TCaloLayer1/src/UCTGeometry.hh"
#include "L1Trigger/L1TCaloLayer1/src/UCTLogging.hh"

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

#include "L1Trigger/L1TCaloLayer1/src/L1TCaloLayer1FetchLUTs.hh"

using namespace l1t;
using namespace l1tcalo;

//
// class declaration
//

class L1TCaloLayer1 : public edm::stream::EDProducer<> {
public:
  explicit L1TCaloLayer1(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  void beginRun(edm::Run const&, edm::EventSetup const&) override;

  //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
  //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------

  edm::EDGetTokenT<EcalTrigPrimDigiCollection> ecalTPSource;
  edm::EDGetTokenT<HcalTrigPrimDigiCollection> hcalTPSource;
  edm::EDPutTokenT<CaloTowerBxCollection> towerPutToken;
  edm::EDPutTokenT<L1CaloRegionCollection> regionPutToken;
  const L1TCaloLayer1FetchLUTsTokens lutsTokens;

  std::vector<std::array<std::array<std::array<uint32_t, nEtBins>, nCalSideBins>, nCalEtaBins> > ecalLUT;
  std::vector<std::array<std::array<std::array<uint32_t, nEtBins>, nCalSideBins>, nCalEtaBins> > hcalLUT;
  std::vector<std::array<std::array<uint32_t, nEtBins>, nHfEtaBins> > hfLUT;
  std::vector<unsigned long long int> hcalFBLUT;

  std::vector<unsigned int> ePhiMap;
  std::vector<unsigned int> hPhiMap;
  std::vector<unsigned int> hfPhiMap;

  std::vector<UCTTower*> twrList;

  bool useLSB;
  bool useCalib;
  bool useECALLUT;
  bool useHCALLUT;
  bool useHFLUT;
  bool useHCALFBLUT;
  bool verbose;
  bool unpackHcalMask;
  bool unpackEcalMask;
  int fwVersion;

  std::unique_ptr<UCTLayer1> layer1;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1TCaloLayer1::L1TCaloLayer1(const edm::ParameterSet& iConfig)
    : ecalTPSource(consumes<EcalTrigPrimDigiCollection>(iConfig.getParameter<edm::InputTag>("ecalToken"))),
      hcalTPSource(consumes<HcalTrigPrimDigiCollection>(iConfig.getParameter<edm::InputTag>("hcalToken"))),
      towerPutToken{produces<CaloTowerBxCollection>()},
      regionPutToken{produces<L1CaloRegionCollection>()},
      lutsTokens{esConsumes<edm::Transition::BeginRun>(),
                 esConsumes<edm::Transition::BeginRun>(),
                 esConsumes<edm::Transition::BeginRun>()},
      ePhiMap(72 * 2, 0),
      hPhiMap(72 * 2, 0),
      hfPhiMap(72 * 2, 0),
      useLSB(iConfig.getParameter<bool>("useLSB")),
      useCalib(iConfig.getParameter<bool>("useCalib")),
      useECALLUT(iConfig.getParameter<bool>("useECALLUT")),
      useHCALLUT(iConfig.getParameter<bool>("useHCALLUT")),
      useHFLUT(iConfig.getParameter<bool>("useHFLUT")),
      useHCALFBLUT(iConfig.getParameter<bool>("useHCALFBLUT")),
      verbose(iConfig.getUntrackedParameter<bool>("verbose")),
      unpackHcalMask(iConfig.getParameter<bool>("unpackHcalMask")),
      unpackEcalMask(iConfig.getParameter<bool>("unpackEcalMask")),
      fwVersion(iConfig.getParameter<int>("firmwareVersion")) {
  // See UCTLayer1.hh for firmware version definitions
  layer1 = std::make_unique<UCTLayer1>(fwVersion);

  vector<UCTCrate*> crates = layer1->getCrates();
  for (uint32_t crt = 0; crt < crates.size(); crt++) {
    vector<UCTCard*> cards = crates[crt]->getCards();
    for (uint32_t crd = 0; crd < cards.size(); crd++) {
      vector<UCTRegion*> regions = cards[crd]->getRegions();
      for (uint32_t rgn = 0; rgn < regions.size(); rgn++) {
        vector<UCTTower*> towers = regions[rgn]->getTowers();
        for (uint32_t twr = 0; twr < towers.size(); twr++) {
          twrList.push_back(towers[twr]);
        }
      }
    }
  }

  // This sort corresponds to the sort condition on
  // the output CaloTowerBxCollection
  std::sort(twrList.begin(), twrList.end(), [](UCTTower* a, UCTTower* b) {
    return CaloTools::caloTowerHash(a->caloEta(), a->caloPhi()) < CaloTools::caloTowerHash(b->caloEta(), b->caloPhi());
  });
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void L1TCaloLayer1::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  edm::Handle<EcalTrigPrimDigiCollection> ecalTPs;
  iEvent.getByToken(ecalTPSource, ecalTPs);
  edm::Handle<HcalTrigPrimDigiCollection> hcalTPs;
  iEvent.getByToken(hcalTPSource, hcalTPs);

  CaloTowerBxCollection towersColl;
  L1CaloRegionCollection rgnCollection;

  if (!layer1->clearEvent()) {
    LOG_ERROR << "UCT: Failed to clear event" << std::endl;
    return;
  }

  for (const auto& ecalTp : *ecalTPs) {
    if (unpackEcalMask && ((ecalTp.sample(0).raw() >> 13) & 0x1))
      continue;
    int caloEta = ecalTp.id().ieta();
    int caloPhi = ecalTp.id().iphi();
    int et = ecalTp.compressedEt();
    bool fgVeto = ecalTp.fineGrain();
    UCTTowerIndex t = UCTTowerIndex(caloEta, caloPhi);
    if (!layer1->setECALData(t, fgVeto, et)) {
      LOG_ERROR << "UCT: Failed loading an ECAL tower" << std::endl;
      return;
    }
  }

  if (hcalTPs.isValid()) {
    for (const auto& hcalTp : *hcalTPs) {
      if (unpackHcalMask && ((hcalTp.sample(0).raw() >> 13) & 0x1))
        continue;
      int caloEta = hcalTp.id().ieta();
      uint32_t absCaloEta = std::abs(caloEta);
      // Tower 29 is not used by Layer-1
      if (absCaloEta == 29) {
        continue;
      }
      // Prevent usage of HF TPs with Layer-1 emulator if HCAL TPs are old style
      else if (hcalTp.id().version() == 0 && absCaloEta > 29) {
        continue;
      } else if (absCaloEta <= 41) {
        int caloPhi = hcalTp.id().iphi();
        int et = hcalTp.SOI_compressedEt();
        bool fg = hcalTp.t0().fineGrain(0);   // depth
        bool fg2 = hcalTp.t0().fineGrain(1);  // prompt
        bool fg3 = hcalTp.t0().fineGrain(2);  // delay 1
        bool fg4 = hcalTp.t0().fineGrain(3);  // delay 2
        // note that hcalTp.t0().fineGrain(4) and hcalTp.t0().fineGrain(5) are the reserved MIP bits (not used for LLP logic)
        if (caloPhi <= 72) {
          UCTTowerIndex t = UCTTowerIndex(caloEta, caloPhi);
          uint32_t featureBits = 0;
          if (absCaloEta > 29) {
            if (fg)
              featureBits |= 0b01;
            // fg2 should only be set for HF
            if (fg2)
              featureBits |= 0b10;
          } else if (absCaloEta < 16)
            featureBits |= (fg | ((!fg2) & (fg3 | fg4)));  // depth | (!prompt & (delay1 | delay2))
          if (!layer1->setHCALData(t, featureBits, et)) {
            LOG_ERROR << "caloEta = " << caloEta << "; caloPhi =" << caloPhi << std::endl;
            LOG_ERROR << "UCT: Failed loading an HCAL tower" << std::endl;
            return;
          }
        } else {
          LOG_ERROR << "Illegal Tower: caloEta = " << caloEta << "; caloPhi =" << caloPhi << "; et = " << et
                    << std::endl;
        }
      } else {
        LOG_ERROR << "Illegal Tower: caloEta = " << caloEta << std::endl;
      }
    }
  }

  //Process
  if (!layer1->process()) {
    LOG_ERROR << "UCT: Failed to process layer 1" << std::endl;
  }

  int theBX = 0;  // Currently we only read and process the "hit" BX only

  for (uint32_t twr = 0; twr < twrList.size(); twr++) {
    CaloTower caloTower;
    caloTower.setHwPt(twrList[twr]->et());          // Bits 0-8 of the 16-bit word per the interface protocol document
    caloTower.setHwEtRatio(twrList[twr]->er());     // Bits 9-11 of the 16-bit word per the interface protocol document
    caloTower.setHwQual(twrList[twr]->miscBits());  // Bits 12-15 of the 16-bit word per the interface protocol document
    caloTower.setHwEta(twrList[twr]->caloEta());    // caloEta = 1-28 and 30-41
    caloTower.setHwPhi(twrList[twr]->caloPhi());    // caloPhi = 1-72
    caloTower.setHwEtEm(twrList[twr]->getEcalET());   // This is provided as a courtesy - not available to hardware
    caloTower.setHwEtHad(twrList[twr]->getHcalET());  // This is provided as a courtesy - not available to hardware
    towersColl.push_back(theBX, caloTower);
  }

  iEvent.emplace(towerPutToken, std::move(towersColl));

  UCTGeometry g;
  vector<UCTCrate*> crates = layer1->getCrates();
  for (uint32_t crt = 0; crt < crates.size(); crt++) {
    vector<UCTCard*> cards = crates[crt]->getCards();
    for (uint32_t crd = 0; crd < cards.size(); crd++) {
      vector<UCTRegion*> regions = cards[crd]->getRegions();
      for (uint32_t rgn = 0; rgn < regions.size(); rgn++) {
        uint32_t rawData = regions[rgn]->rawData();
        uint32_t regionData = rawData & 0x0000FFFF;
        uint32_t crate = regions[rgn]->getCrate();
        uint32_t card = regions[rgn]->getCard();
        uint32_t region = regions[rgn]->getRegion();
        bool negativeEta = regions[rgn]->isNegativeEta();
        uint32_t rPhi = g.getUCTRegionPhiIndex(crate, card);
        if (region < NRegionsInCard) {  // We only store the Barrel and Endcap - HF has changed in the upgrade
          uint32_t rEta =
              10 -
              region;  // UCT region is 0-6 for B/E but GCT eta goes 0-21, 0-3 -HF, 4-10 -B/E, 11-17 +B/E, 18-21 +HF
          if (!negativeEta)
            rEta = 11 + region;  // Positive eta portion is offset by 11
          rgnCollection.push_back(L1CaloRegion((uint16_t)regionData, (unsigned)rEta, (unsigned)rPhi, (int16_t)0));
        }
      }
    }
  }
  iEvent.emplace(regionPutToken, std::move(rgnCollection));
}

// ------------ method called when starting to processes a run  ------------
void L1TCaloLayer1::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  if (!L1TCaloLayer1FetchLUTs(lutsTokens,
                              iSetup,
                              ecalLUT,
                              hcalLUT,
                              hfLUT,
                              hcalFBLUT,
                              ePhiMap,
                              hPhiMap,
                              hfPhiMap,
                              useLSB,
                              useCalib,
                              useECALLUT,
                              useHCALLUT,
                              useHFLUT,
                              useHCALFBLUT,
                              fwVersion)) {
    LOG_ERROR << "L1TCaloLayer1::beginRun: failed to fetch LUTS - using unity" << std::endl;
    std::array<std::array<std::array<uint32_t, nEtBins>, nCalSideBins>, nCalEtaBins> eCalLayer1EtaSideEtArray;
    std::array<std::array<std::array<uint32_t, nEtBins>, nCalSideBins>, nCalEtaBins> hCalLayer1EtaSideEtArray;
    std::array<std::array<uint32_t, nEtBins>, nHfEtaBins> hfLayer1EtaEtArray;
    ecalLUT.push_back(eCalLayer1EtaSideEtArray);
    hcalLUT.push_back(hCalLayer1EtaSideEtArray);
    hfLUT.push_back(hfLayer1EtaEtArray);
  }
  for (uint32_t twr = 0; twr < twrList.size(); twr++) {
    // Map goes minus 1 .. 72 plus 1 .. 72 -> 0 .. 143
    int iphi = twrList[twr]->caloPhi();
    int ieta = twrList[twr]->caloEta();
    if (ieta < 0) {
      iphi -= 1;
    } else {
      iphi += 71;
    }
    twrList[twr]->setECALLUT(&ecalLUT[ePhiMap[iphi]]);
    twrList[twr]->setHCALLUT(&hcalLUT[hPhiMap[iphi]]);
    twrList[twr]->setHFLUT(&hfLUT[hfPhiMap[iphi]]);
  }
}

// ------------ method called when ending the processing of a run  ------------
/*
  void
  L1TCaloLayer1::endRun(edm::Run const&, edm::EventSetup const&)
  {
  }
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
  void
  L1TCaloLayer1::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
  {
  }
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
  void
  L1TCaloLayer1::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
  {
  }
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void L1TCaloLayer1::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //Description set to reflect default present in simCaloStage2Layer1Digis_cfi.py
  //Currently redundant, but could be adjusted to provide defaults in case additional LUT
  //checks are added and before other configurations adjust to match.
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("ecalToken", edm::InputTag("simEcalTriggerPrimitiveDigis"));
  desc.add<edm::InputTag>("hcalToken", edm::InputTag("simHcalTriggerPrimitiveDigis"));
  desc.add<bool>("useLSB", true);
  desc.add<bool>("useCalib", true);
  desc.add<bool>("useECALLUT", true);
  desc.add<bool>("useHCALLUT", true);
  desc.add<bool>("useHFLUT", true);
  desc.add<bool>("useHCALFBLUT", false);
  desc.addUntracked<bool>("verbose", false);
  desc.add<bool>("unpackEcalMask", false);
  desc.add<bool>("unpackHcalMask", false);
  desc.add<int>("firmwareVersion", 1);
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TCaloLayer1);
/* vim: set ts=8 sw=2 tw=0 et :*/
