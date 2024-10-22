#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTTPGProvider.h"

L1RCTTPGProvider::L1RCTTPGProvider(const edm::ParameterSet &iConfig)
    : ecalTPG_(consumes(iConfig.getParameter<edm::InputTag>("ecalTPGs"))),
      hcalTPG_(consumes(iConfig.getParameter<edm::InputTag>("hcalTPGs"))),
      useHcalCosmicTiming(iConfig.getParameter<bool>("useHCALCosmicTiming")),
      useEcalCosmicTiming(iConfig.getParameter<bool>("useECALCosmicTiming")),
      preSamples(iConfig.getParameter<int>("preSamples")),
      postSamples(iConfig.getParameter<int>("postSamples")),
      hfShift(iConfig.getParameter<int>("HFShift")),
      hbShift(iConfig.getParameter<int>("HBShift")) {
  // Output :The new manipulated TPGs
  // make it smart - to name the collections
  // correctly
  char ecal_label[200];
  char hcal_label[200];

  for (int i = preSamples; i > 0; --i) {
    sprintf(ecal_label, "ECALBxminus%d", i);
    sprintf(hcal_label, "HCALBxminus%d", i);
    produces<EcalTrigPrimDigiCollection>(ecal_label);
    produces<HcalTrigPrimDigiCollection>(hcal_label);
  }

  produces<EcalTrigPrimDigiCollection>("ECALBx0");
  produces<HcalTrigPrimDigiCollection>("HCALBx0");

  for (int i = 0; i < postSamples; ++i) {
    sprintf(ecal_label, "ECALBxplus%d", i + 1);
    sprintf(hcal_label, "HCALBxplus%d", i + 1);
    produces<EcalTrigPrimDigiCollection>(ecal_label);
    produces<HcalTrigPrimDigiCollection>(hcal_label);
  }
}

L1RCTTPGProvider::~L1RCTTPGProvider() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void L1RCTTPGProvider::produce(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const {
  using namespace edm;
  // Declare handles
  Handle<EcalTrigPrimDigiCollection> ecal;
  Handle<HcalTrigPrimDigiCollection> hcal;

  // Declare vector of collection to send for output !

  std::vector<EcalTrigPrimDigiCollection> ecalColl(preSamples + 1 + postSamples);
  std::vector<HcalTrigPrimDigiCollection> hcalColl(preSamples + 1 + postSamples);

  unsigned nSamples = preSamples + postSamples + 1;

  if ((ecal = iEvent.getHandle(ecalTPG_)))
    if (ecal.isValid()) {
      // loop through all ecal digis
      for (EcalTrigPrimDigiCollection::const_iterator ecal_it = ecal->begin(); ecal_it != ecal->end(); ecal_it++) {
        short zside = ecal_it->id().zside();
        unsigned short ietaAbs = ecal_it->id().ietaAbs();
        short iphi = ecal_it->id().iphi();
        unsigned short digiSize = ecal_it->size();
        unsigned short nSOI = (unsigned short)(ecal_it->sampleOfInterest());
        if (digiSize < nSamples || nSOI < preSamples || ((int)(digiSize - nSOI) < (int)(nSamples - preSamples))) {
          unsigned short preLoopsZero = (unsigned short)(preSamples)-nSOI;

          // fill extra bx's at beginning with zeros
          for (int sample = 0; sample < preLoopsZero; sample++) {
            // fill first few with zeros
            EcalTriggerPrimitiveDigi ecalDigi(
                EcalTrigTowerDetId((int)zside, EcalTriggerTower, (int)ietaAbs, (int)iphi));
            ecalDigi.setSize(1);
            ecalDigi.setSample(0, EcalTriggerPrimitiveSample(0, false, 0));
            ecalColl[sample].push_back(ecalDigi);
          }

          // loop through existing data
          for (int sample = preLoopsZero; sample < (preLoopsZero + digiSize); sample++) {
            // go through data
            EcalTriggerPrimitiveDigi ecalDigi(
                EcalTrigTowerDetId((int)zside, EcalTriggerTower, (int)ietaAbs, (int)iphi));
            ecalDigi.setSize(1);
            if (useEcalCosmicTiming && iphi >= 1 && iphi <= 36) {
              if (nSOI < (preSamples + 1)) {
                edm::LogWarning("TooLittleData") << "ECAL data needs at least one presample "
                                                 << "more than the number requested "
                                                 << "to use ecal cosmic timing mod!  "
                                                 << "reverting to useEcalCosmicTiming = false "
                                                 << "for rest of job.";
                useEcalCosmicTiming = false;
              } else {
                ecalDigi.setSample(0,
                                   EcalTriggerPrimitiveSample(ecal_it->sample(nSOI + sample - preSamples - 1).raw()));
              }
            }
            // else
            if ((!useEcalCosmicTiming) || (iphi >= 37 && iphi <= 72)) {
              ecalDigi.setSample(0, EcalTriggerPrimitiveSample(ecal_it->sample(nSOI + sample - preSamples).raw()));
            }
            ecalColl[sample].push_back(ecalDigi);
          }

          // fill extra bx's at end with zeros
          for (unsigned int sample = (preLoopsZero + digiSize); sample < nSamples; sample++) {
            // fill zeros!
            EcalTriggerPrimitiveDigi ecalDigi(
                EcalTrigTowerDetId((int)zside, EcalTriggerTower, (int)ietaAbs, (int)iphi));
            ecalDigi.setSize(1);
            ecalDigi.setSample(0, EcalTriggerPrimitiveSample(0, false, 0));
            ecalColl[sample].push_back(ecalDigi);
          }
        } else {
          for (unsigned short sample = 0; sample < nSamples; sample++) {
            // put each time sample into its own digi
            short zside = ecal_it->id().zside();
            unsigned short ietaAbs = ecal_it->id().ietaAbs();
            short iphi = ecal_it->id().iphi();
            EcalTriggerPrimitiveDigi ecalDigi(
                EcalTrigTowerDetId((int)zside, EcalTriggerTower, (int)ietaAbs, (int)iphi));
            ecalDigi.setSize(1);

            if (useEcalCosmicTiming && iphi >= 1 && iphi <= 36) {
              if (nSOI < (preSamples + 1)) {
                edm::LogWarning("TooLittleData") << "ECAL data needs at least one presample "
                                                 << "more than the number requested "
                                                 << "to use ecal cosmic timing mod!  "
                                                 << "reverting to useEcalCosmicTiming = false "
                                                 << "for rest of job.";
                useEcalCosmicTiming = false;
              } else {
                ecalDigi.setSample(0,
                                   EcalTriggerPrimitiveSample(
                                       ecal_it->sample(ecal_it->sampleOfInterest() + sample - preSamples - 1).raw()));
              }
            }
            // else
            if ((!useEcalCosmicTiming) || (iphi >= 37 && iphi <= 72)) {
              ecalDigi.setSample(
                  0,
                  EcalTriggerPrimitiveSample(ecal_it->sample(ecal_it->sampleOfInterest() + sample - preSamples).raw()));
            }
            // push back each digi into correct "time sample" of coll
            ecalColl[sample].push_back(ecalDigi);
          }
        }
      }
    }

  if ((hcal = iEvent.getHandle(hcalTPG_)))
    if (hcal.isValid()) {
      // loop through all hcal digis
      for (HcalTrigPrimDigiCollection::const_iterator hcal_it = hcal->begin(); hcal_it != hcal->end(); hcal_it++) {
        short ieta = hcal_it->id().ieta();
        short iphi = hcal_it->id().iphi();
        // loop through time samples for each digi
        unsigned short digiSize = hcal_it->size();
        // (size of each digi must be no less than nSamples)
        unsigned short nSOI = (unsigned short)(hcal_it->presamples());
        if (digiSize < nSamples || nSOI < preSamples || ((int)(digiSize - nSOI) < (int)(nSamples - preSamples))) {
          unsigned short preLoopsZero = (unsigned short)(preSamples)-nSOI;
          // fill extra bx's at beginning with zeros
          for (int sample = 0; sample < preLoopsZero; sample++) {
            // fill first few with zeros
            HcalTriggerPrimitiveDigi hcalDigi(HcalTrigTowerDetId((int)ieta, (int)iphi));
            hcalDigi.setSize(1);
            hcalDigi.setPresamples(0);
            hcalDigi.setSample(0, HcalTriggerPrimitiveSample(0, false, 0, 0));
            hcalColl[sample].push_back(hcalDigi);
          }

          // loop through existing data
          for (int sample = preLoopsZero; sample < (preLoopsZero + digiSize); sample++) {
            // go through data
            HcalTriggerPrimitiveDigi hcalDigi(HcalTrigTowerDetId((int)ieta, (int)iphi));
            hcalDigi.setSize(1);
            hcalDigi.setPresamples(0);

            if (useHcalCosmicTiming && iphi >= 1 && iphi <= 36) {
              if (nSOI < (preSamples + 1)) {
                edm::LogWarning("TooLittleData") << "HCAL data needs at least one presample "
                                                 << "more than the number requested "
                                                 << "to use hcal cosmic timing mod!  "
                                                 << "reverting to useHcalCosmicTiming = false "
                                                 << "for rest of job.";
                useHcalCosmicTiming = false;
              } else {
                hcalDigi.setSample(
                    0,
                    HcalTriggerPrimitiveSample(hcal_it->sample(hcal_it->presamples() + sample - preSamples - 1).raw()));
              }
            }
            // else
            if ((!useHcalCosmicTiming) || (iphi >= 37 && iphi <= 72)) {
              hcalDigi.setSample(
                  0, HcalTriggerPrimitiveSample(hcal_it->sample(hcal_it->presamples() + sample - preSamples).raw()));
            }
            hcalColl[sample].push_back(hcalDigi);
          }

          // fill extra bx's at end with zeros
          for (unsigned int sample = (preLoopsZero + digiSize); sample < nSamples; sample++) {
            // fill zeros!
            HcalTriggerPrimitiveDigi hcalDigi(HcalTrigTowerDetId((int)ieta, (int)iphi));
            hcalDigi.setSize(1);
            hcalDigi.setPresamples(0);
            hcalDigi.setSample(0, HcalTriggerPrimitiveSample(0, false, 0, 0));
            hcalColl[sample].push_back(hcalDigi);
          }
        } else {
          for (unsigned short sample = 0; sample < nSamples; sample++) {
            // put each (relevant) time sample into its own digi
            HcalTriggerPrimitiveDigi hcalDigi(HcalTrigTowerDetId((int)ieta, (int)iphi));
            hcalDigi.setSize(1);
            hcalDigi.setPresamples(0);

            if (useHcalCosmicTiming && iphi >= 1 && iphi <= 36) {
              if (nSOI < (preSamples + 1)) {
                edm::LogWarning("TooLittleData") << "HCAL data needs at least one presample "
                                                 << "more than the number requested "
                                                 << "to use hcal cosmic timing mod!  "
                                                 << "reverting to useHcalCosmicTiming = false "
                                                 << "for rest of job.";
                useHcalCosmicTiming = false;
              } else {
                hcalDigi.setSample(
                    0,
                    HcalTriggerPrimitiveSample(hcal_it->sample(hcal_it->presamples() + sample - preSamples - 1).raw()));
              }
            }
            // else
            if ((!useHcalCosmicTiming) || (iphi >= 37 && iphi <= 72)) {
              if (ieta > -29 && ieta < 29)
                hcalDigi.setSample(0,
                                   HcalTriggerPrimitiveSample(
                                       hcal_it->sample(hcal_it->presamples() + sample - preSamples + hbShift).raw()));
              if (ieta <= -29 || ieta >= 29)
                hcalDigi.setSample(0,
                                   HcalTriggerPrimitiveSample(
                                       hcal_it->sample(hcal_it->presamples() + sample - preSamples + hfShift).raw()));
            }
            hcalColl[sample].push_back(hcalDigi);
          }
        }
      }
    }

  // Now put the events back to file

  for (int i = 0; i < preSamples; ++i) {
    char ecal_label[200];
    char hcal_label[200];

    sprintf(ecal_label, "ECALBxminus%d", preSamples - i);
    sprintf(hcal_label, "HCALBxminus%d", preSamples - i);

    std::unique_ptr<EcalTrigPrimDigiCollection> ecalIn(new EcalTrigPrimDigiCollection);
    std::unique_ptr<HcalTrigPrimDigiCollection> hcalIn(new HcalTrigPrimDigiCollection);
    for (unsigned int j = 0; j < ecalColl[i].size(); ++j) {
      ecalIn->push_back((ecalColl[i])[j]);
    }
    for (unsigned int j = 0; j < hcalColl[i].size(); ++j)
      hcalIn->push_back((hcalColl[i])[j]);

    iEvent.put(std::move(ecalIn), ecal_label);
    iEvent.put(std::move(hcalIn), hcal_label);
  }

  std::unique_ptr<EcalTrigPrimDigiCollection> ecal0(new EcalTrigPrimDigiCollection);
  std::unique_ptr<HcalTrigPrimDigiCollection> hcal0(new HcalTrigPrimDigiCollection);
  for (unsigned int j = 0; j < ecalColl[preSamples].size(); ++j)
    ecal0->push_back((ecalColl[preSamples])[j]);
  for (unsigned int j = 0; j < hcalColl[preSamples].size(); ++j)
    hcal0->push_back((hcalColl[preSamples])[j]);

  iEvent.put(std::move(ecal0), "ECALBx0");
  iEvent.put(std::move(hcal0), "HCALBx0");

  for (int i = preSamples + 1; i < preSamples + postSamples + 1; ++i) {
    char ecal_label[200];
    char hcal_label[200];

    sprintf(ecal_label, "ECALBxplus%d", i - preSamples);
    sprintf(hcal_label, "HCALBxplus%d", i - preSamples);

    std::unique_ptr<EcalTrigPrimDigiCollection> ecalIn2(new EcalTrigPrimDigiCollection);
    std::unique_ptr<HcalTrigPrimDigiCollection> hcalIn2(new HcalTrigPrimDigiCollection);

    for (unsigned int j = 0; j < ecalColl[i].size(); ++j)
      ecalIn2->push_back((ecalColl[i])[j]);

    for (unsigned int j = 0; j < hcalColl[i].size(); ++j)
      hcalIn2->push_back((hcalColl[i])[j]);

    iEvent.put(std::move(ecalIn2), ecal_label);
    iEvent.put(std::move(hcalIn2), hcal_label);
  }
}
