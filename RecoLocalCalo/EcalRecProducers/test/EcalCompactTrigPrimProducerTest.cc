#include "RecoLocalCalo/EcalRecProducers/test/EcalCompactTrigPrimProducerTest.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <iostream>
#include <iomanip>
#include <sstream>

void EcalCompactTrigPrimProducerTest::analyze(edm::Event const& event, edm::EventSetup const& c) {
  edm::Handle<EcalTrigPrimDigiCollection> hTpDigis;
  event.getByToken(tpDigiToken_, hTpDigis);

  const EcalTrigPrimDigiCollection* trigPrims = hTpDigis.product();

  edm::Handle<EcalTrigPrimCompactColl> hTpRecs;
  event.getByToken(tpRecToken_, hTpRecs);

  const EcalTrigPrimCompactColl* trigPrimRecs = hTpRecs.product();

  int nTps = 0;
  err_ = false;
  for (EcalTrigPrimDigiCollection::const_iterator trigPrim = trigPrims->begin(); trigPrim != trigPrims->end();
       ++trigPrim) {
    const EcalTrigTowerDetId& ttId = trigPrim->id();

    if ((trigPrim->sample(trigPrim->sampleOfInterest()).raw() & 0x1FFF) != (trigPrimRecs->raw(ttId) & 0x1FFF)) {
      err("Different TP (0x") << std::hex << trigPrim->sample(trigPrim->sampleOfInterest()).raw() << " -- "
                              << trigPrimRecs->raw(ttId) << std::dec << ") for " << ttId << "\n";
    }
    if (trigPrim->compressedEt() != trigPrimRecs->compressedEt(ttId))
      err("\tDifferent compressed Et\n");
    if (trigPrim->fineGrain() != trigPrimRecs->fineGrain(ttId))
      err("\tDifferent FGVB\n") << ttId << "\n";
    if (trigPrim->ttFlag() != trigPrimRecs->ttFlag(ttId))
      err("\tDifferent compressed TTF\n") << ttId << "\n";
    if (trigPrim->l1aSpike() != trigPrimRecs->l1aSpike(ttId))
      err("\tDifferent compressed L1Spike flag\n");
    if (trigPrim->compressedEt() != 0)
      ++nCompressEt_;
    if (trigPrim->fineGrain() != 0)
      ++nFineGrain_;
    if (trigPrim->ttFlag() != 0)
      ++nTTF_;
    if (trigPrim->l1aSpike() != 0)
      ++nL1aSpike_;
    ++nTps;
  }
  if (nTps != 4032)
    err("Unexpected number of TPs: ") << nTps << "\n";

  if (!err_)
    std::cout << "Validation of compact trigger primitive collection " << (err_ ? "failed" : "succeeded") << "\n";

  if (err_) {
    std::cout << "Cannot check compact to legacy collection convertion because of previous failure\n";
  } else {
    //test compact to legacy collection convertion:
    EcalTrigPrimDigiCollection col2;
    trigPrimRecs->toEcalTrigPrimDigiCollection(col2);
    if (col2.size() != trigPrims->size()) {
      err("Collection size error!\n");
      err_ = true;
    } else {
      EcalTrigPrimDigiCollection::const_iterator trigPrim2 = col2.begin();
      for (EcalTrigPrimDigiCollection::const_iterator trigPrim = trigPrims->begin();
           trigPrim != trigPrims->end() && !err_;
           ++trigPrim, ++trigPrim2) {
        if ((trigPrim->sample(trigPrim->sampleOfInterest()).raw() &
             0x1FFF)  //masked unused bits (see flaw in unpacker that modifies them)
            != (trigPrim2->sample(0).raw() & 0x1FFF))
          err("Trig prim differs: ") << *trigPrim << " (" << std::hex
                                     << (trigPrim->sample(trigPrim->sampleOfInterest()).raw()) << std::dec << ") "
                                     << "  --  " << *trigPrim2 << " (" << std::hex
                                     << (trigPrim2->sample(trigPrim2->sampleOfInterest()).raw()) << std::dec << ") "
                                     << "\n";  //err_ = true;
      }
    }
    std::cout << "Validation of compact-to-legacy trigger primitive collection conversion "
              << (err_ ? "failed" : "succeeded") << "\n";
  }

  //test of skimmed TP collection
  edm::Handle<EcalTrigPrimDigiCollection> hSkimTpDigis;
  event.getByToken(tpSkimToken_, hSkimTpDigis);
  const EcalTrigPrimDigiCollection* skimTrigPrims = hSkimTpDigis.product();
  err_ = false;
  if (skimTrigPrims->size() == 0)
    err("Skimmed TP collection is empty!");
  std::stringstream tpList;
  for (EcalTrigPrimDigiCollection::const_iterator skimTrigPrim = skimTrigPrims->begin();
       skimTrigPrim != skimTrigPrims->end();
       ++skimTrigPrim) {
    tpList << "\t- detid=" << skimTrigPrim->id().rawId() << " ieta=" << skimTrigPrim->id().ieta()
           << " iphi=" << skimTrigPrim->id().iphi() << "\n";
    EcalTrigPrimDigiCollection::const_iterator origTrigPrim = trigPrims->find(skimTrigPrim->id());
    if (origTrigPrim == trigPrims->end())
      err("Skimmed TP ") << skimTrigPrim->id() << " not found in original TP collection!\n";
    else {
      if (skimTrigPrim->size() != origTrigPrim->size()) {
        std::cout << "TP from skimmed colletion has " << skimTrigPrim->size() << " sample(s), "
                  << "while TP from original collection has " << origTrigPrim->size() << " sample(s)!\n";
      }
      bool oneSample = false;
      if (skimTrigPrim->size() == 1 || origTrigPrim->size() == 1) {
        std::cout << "\tComparing only the \"sample of interest\"!\n";
        oneSample = true;
      }
      bool eq = true;
      if (oneSample) {
        const int skimSample = skimTrigPrim->sampleOfInterest();
        const int origSample = origTrigPrim->sampleOfInterest();
        //masked unused bits (see flaw in unpacker that modifies them)
        eq = ((skimTrigPrim->sample(skimSample).raw() & 0x1FFF) == (origTrigPrim->sample(origSample).raw() & 0x1FFF));
      } else if (skimTrigPrim->size() == origTrigPrim->size()) {
        for (int iS = 0; iS < skimTrigPrim->size(); ++iS) {
          //masked unused bits (see flaw in unpacker that modifies them)
          if ((skimTrigPrim->sample(iS).raw() & 0x1FFF) != (origTrigPrim->sample(iS).raw() & 0x1FFF))
            eq = false;
        }
      } else {
        err_ = true;
      }
      if (!eq)
        err("Skimmed Trig prim differs from original one: ") << *skimTrigPrim << "  --  " << *origTrigPrim << "\n";
    }
  }
  std::cout << "List if TT in skimmed TP collection: " << tpList.str() << "\n";
  std::cout << "Validation of skimmed trigger primitive collection " << (err_ ? "failed" : "succeeded") << "\n";
}

std::ostream& EcalCompactTrigPrimProducerTest::err(const char* mess) {
  err_ = true;
  std::cout << mess;
  return std::cout;
}

EcalCompactTrigPrimProducerTest::~EcalCompactTrigPrimProducerTest() {
  std::cout << "# of non-null compressed Et: " << nCompressEt_ << "\n";
  std::cout << "# of non-null FGVB: " << nFineGrain_ << "\n";
  std::cout << "# of non-null TTF: " << nTTF_ << "\n";
  std::cout << "# of non-null L1ASpike: " << nL1aSpike_ << "\n";
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EcalCompactTrigPrimProducerTest);
