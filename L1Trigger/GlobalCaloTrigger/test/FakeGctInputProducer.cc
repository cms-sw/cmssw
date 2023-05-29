#include "L1Trigger/GlobalCaloTrigger/test/FakeGctInputProducer.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"

#include <memory>
#include <iostream>

FakeGctInputProducer::FakeGctInputProducer(const edm::ParameterSet& iConfig) {
  produces<L1CaloEmCollection>();
  produces<L1CaloRegionCollection>();

  rgnMode_ = iConfig.getUntrackedParameter<int>("regionMode", 0);
  iemMode_ = iConfig.getUntrackedParameter<int>("isoEmMode", 0);
  niemMode_ = iConfig.getUntrackedParameter<int>("nonIsoEmMode", 0);
}

void FakeGctInputProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;

  // containers
  std::unique_ptr<L1CaloEmCollection> emCands(new L1CaloEmCollection);
  std::unique_ptr<L1CaloRegionCollection> regions(new L1CaloRegionCollection);

  /// REGIONS ///

  // decide which region to fill for mode 1
  int nCrt = rand() % 18;
  int nCrd = rand() % 7;
  int nRgn = rand() % 2;

  //  std::cout << "Making region at crt=" << nCrt << " crd=" << nCrd << " rgn=" << nRgn << std::endl;

  // make regions
  for (int iCrt = 0; iCrt < 18; iCrt++) {
    for (int iCrd = 0; iCrd < 7; iCrd++) {
      for (int iRgn = 0; iRgn < 2; iRgn++) {
        unsigned et = 0;

        if (rgnMode_ == 1) {
          // throw random Et
          et = int(100 * rand() / (RAND_MAX + 1.));
        } else if (rgnMode_ == 2 && iCrt == nCrt && iCrd == nCrd && iRgn == nRgn) {
          et = 10;
        }

        regions->push_back(L1CaloRegion(et, false, false, false, false, iCrt, iCrd, iRgn));
      }
    }
  }

  /// ISO EM ///

  // decide where single candidate will go (mode 2)
  nCrt = rand() % 18;
  nCrd = rand() % 7;
  nRgn = rand() % 2;

  //  std::cout << "Making iso cand at crt=" << nCrt << " crd=" << nCrd << " rgn=" << nRgn << std::endl;

  // loop over crates and candidates
  for (int iCrt = 0; iCrt < 18; iCrt++) {
    for (int iCand = 0; iCand < 4; iCand++) {
      if (iemMode_ == 1) {
        // multiple EM candidates not yet implemented
      } else if (iemMode_ == 2) {
        if (iCand == 0 && iCrt == nCrt) {
          emCands->push_back(L1CaloEmCand(1, nRgn, nCrd, nCrt, true));
        } else {
          emCands->push_back(L1CaloEmCand(0, 0, iCand, iCrt, true));
        }
      }
    }
  }

  /// NON-ISO EM ///

  // decide where single candidate will go (mode 2)
  nCrt = rand() % 18;
  nCrd = rand() % 7;
  nRgn = rand() % 2;

  //  std::cout << "Making non-iso cand at crt=" << nCrt << " crd=" << nCrd << " rgn=" << nRgn << std::endl;

  // loop over crates and candidates
  for (int iCrt = 0; iCrt < 18; iCrt++) {
    for (int iCand = 0; iCand < 4; iCand++) {
      if (niemMode_ == 1) {
        // multiple EM candidates not yet implemented
      } else if (niemMode_ == 2) {
        if (iCand == 0 && iCrt == nCrt) {
          emCands->push_back(L1CaloEmCand(1, nRgn, nCrd, nCrt, false));
        } else {
          emCands->push_back(L1CaloEmCand(0, 0, iCand, iCrt, false));
        }
      }
    }
  }

  // put everything in the event
  iEvent.put(std::move(emCands));
  iEvent.put(std::move(regions));
}
