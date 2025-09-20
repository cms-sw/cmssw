//
// L1TZDCEtSumsProducer
//  EDProducer to compute the ZDC l1t::EtSums from HCAL trigger primitives
//
// Original author: Chris McGinn
// Contact: christopher.mc.ginn@cern.ch or
//          cfmcginn on github for bugs/issues
//
#include <algorithm>
#include <array>
#include <memory>
#include <vector>

#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

class L1TZDCEtSumsProducer : public edm::global::EDProducer<> {
public:
  explicit L1TZDCEtSumsProducer(edm::ParameterSet const&);

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

  edm::EDGetTokenT<HcalTrigPrimDigiCollection> const hcalTPDigisToken_;

  int const bxFirst_;
  int const bxLast_;

  static constexpr int kZDCAbsIEta = 42;
  static constexpr int kZDCiEtSumsIPhi = 99;
  static constexpr int kZDCiEtSumMaxValue = 1023;
};

L1TZDCEtSumsProducer::L1TZDCEtSumsProducer(edm::ParameterSet const& ps)
    : hcalTPDigisToken_{consumes(ps.getParameter<edm::InputTag>("hcalTPDigis"))},
      bxFirst_{ps.getParameter<int>("bxFirst")},
      bxLast_{ps.getParameter<int>("bxLast")} {
  produces<l1t::EtSumBxCollection>();
}

void L1TZDCEtSumsProducer::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
  auto outZDCEtSums = std::make_unique<l1t::EtSumBxCollection>(0, bxFirst_, bxLast_);

  auto const hcalTPs = iEvent.getHandle(hcalTPDigisToken_);

  if (not hcalTPs.isValid()) {
    edm::LogWarning("L1TZDCEtSumsProducer") << "Invalid handle to HcalTrigPrimDigiCollection collection"
                                            << ": returning empty l1t::EtSumBxCollection for ZDC EtSums !";
  } else if (bxFirst_ > bxLast_) {
    edm::LogWarning("L1TZDCEtSumsProducer")
        << "Invalid configuration parameters (bxFirst [" << bxFirst_ << "] > bxLast [" << bxLast_
        << "]): returning empty l1t::EtSumBxCollection for ZDC EtSums !";
  } else {
    // number of bunch crossings
    unsigned int const nBXs = (bxLast_ - bxFirst_) + 1;

    // iEtSums as taken directly from the ZDC iEtSum TPs (iphi == 99)
    std::vector<std::array<int, 2>> iEtSumsFromEtSumTPs{nBXs, {{0, 0}}};

    // iEtSums recomputed from the ZDC non-iEtSums TPs (iphi != 99)
    std::vector<std::array<int, 2>> iEtSumsFromOtherTPs{nBXs, {{0, 0}}};

    // unsigned integers indicating which iEtSums are available for a given BX
    //  - Bit #1 (0b01): iEtSums as taken directly from the ZDC iEtSum TPs
    //  - Bit #2 (0b10): iEtSums recomputed from the ZDC non-iEtSums TPs
    std::vector<std::array<unsigned int, 2>> iEtSumsFillFlags{nBXs, {{0, 0}}};

    for (auto const& hcalTp : *hcalTPs) {
      // absIEta position 42 is used for the ZDC (-42 for ZDCM, +42 for ZDCP)
      auto const ieta = hcalTp.id().ieta();
      auto const absIEta = std::abs(ieta);

      if (absIEta != kZDCAbsIEta) {
        continue;
      }

      // ZDC "index": 0 for ZDCM, 1 for ZDCP
      auto const zdcIndex = (ieta < 0) ? 0 : 1;

      // For ZDC, iphi position 99 is used for iEtSum TPs
      auto const iphi = hcalTp.id().iphi();
      auto const isZDCiEtSum = (iphi == kZDCiEtSumsIPhi);

      // Number of samples, and number of presamples (nPresamples is BX=0)
      int const nSamples = hcalTp.size();
      int const nPresamples = hcalTp.presamples();

      for (auto iSample = 0; iSample < nSamples; ++iSample) {
        auto const ibx = iSample - nPresamples;
        if (ibx >= bxFirst_ and ibx <= bxLast_) {
          auto const& hcalTpSample = hcalTp.sample(iSample);
          auto const ietIn = hcalTpSample.raw() & kZDCiEtSumMaxValue;
          auto const bxIndex = ibx - bxFirst_;
          if (isZDCiEtSum) {
            iEtSumsFromEtSumTPs[bxIndex][zdcIndex] = ietIn;
            iEtSumsFillFlags[bxIndex][zdcIndex] |= 0b01;
          } else {
            iEtSumsFromOtherTPs[bxIndex][zdcIndex] += ietIn;
            iEtSumsFillFlags[bxIndex][zdcIndex] |= 0b10;
          }
        }
      }
    }

    for (unsigned int bxIndex = 0; bxIndex < nBXs; ++bxIndex) {
      int const bx = bxIndex + bxFirst_;
      for (unsigned int zdcIndex = 0; zdcIndex < 2; ++zdcIndex) {
        int zdc_hwPt{0};
        // Option #1: take iEtSum from the ZDC iEtSum TP
        if (iEtSumsFillFlags[bxIndex][zdcIndex] & 0b01) {
          zdc_hwPt = iEtSumsFromEtSumTPs[bxIndex][zdcIndex];
        }
        // Option #2: take iEtSum recomputed from the ZDC non-iEtSums TPs
        else if (iEtSumsFillFlags[bxIndex][zdcIndex] & 0b10) {
          // recomputed sum cannot be higher than kZDCiEtSumMaxValue
          zdc_hwPt = std::min(iEtSumsFromOtherTPs[bxIndex][zdcIndex], kZDCiEtSumMaxValue);
        }
        // Skip if no iEtSum value is available for this BX
        else {
          continue;
        }

        int const zdc_hwEta = (zdcIndex == 0) ? -1 : 1;

        auto const zdc_type = (zdcIndex == 0) ? l1t::EtSum::EtSumType::kZDCM : l1t::EtSum::EtSumType::kZDCP;

        l1t::EtSum zdc_etSum{};
        zdc_etSum.setHwPt(zdc_hwPt);
        zdc_etSum.setHwEta(zdc_hwEta);
        zdc_etSum.setHwPhi(0);
        zdc_etSum.setType(zdc_type);

        outZDCEtSums->push_back(bx, l1t::CaloTools::etSumP4Demux(zdc_etSum));
      }
    }
  }

  iEvent.put(std::move(outZDCEtSums));
}

void L1TZDCEtSumsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("hcalTPDigis", edm::InputTag("simHcalTriggerPrimitiveDigis"));
  desc.add<int>("bxFirst", -2);
  desc.add<int>("bxLast", 2);
  descriptions.add("l1tZDCEtSumsProducer", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TZDCEtSumsProducer);
