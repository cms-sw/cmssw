/*
 * \file L1TStage2CaloLayer1.cc
 *
 * N. Smith <nick.smith@cern.ch>
 */
//Modified by Bhawna Gomber <bhawna.gomber@cern.ch>

#include "DQM/L1TMonitor/interface/L1TStage2CaloLayer1.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"

#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"

#include "EventFilter/L1TXRawToDigi/plugins/UCTDAQRawData.h"
#include "EventFilter/L1TXRawToDigi/plugins/UCTAMCRawData.h"

L1TStage2CaloLayer1::L1TStage2CaloLayer1(const edm::ParameterSet& ps)
    : ecalTPSourceRecd_(consumes<EcalTrigPrimDigiCollection>(ps.getParameter<edm::InputTag>("ecalTPSourceRecd"))),
      ecalTPSourceRecdLabel_(ps.getParameter<edm::InputTag>("ecalTPSourceRecd").label()),
      hcalTPSourceRecd_(consumes<HcalTrigPrimDigiCollection>(ps.getParameter<edm::InputTag>("hcalTPSourceRecd"))),
      hcalTPSourceRecdLabel_(ps.getParameter<edm::InputTag>("hcalTPSourceRecd").label()),
      ecalTPSourceSent_(consumes<EcalTrigPrimDigiCollection>(ps.getParameter<edm::InputTag>("ecalTPSourceSent"))),
      ecalTPSourceSentLabel_(ps.getParameter<edm::InputTag>("ecalTPSourceSent").label()),
      hcalTPSourceSent_(consumes<HcalTrigPrimDigiCollection>(ps.getParameter<edm::InputTag>("hcalTPSourceSent"))),
      hcalTPSourceSentLabel_(ps.getParameter<edm::InputTag>("hcalTPSourceSent").label()),
      fedRawData_(consumes<FEDRawDataCollection>(ps.getParameter<edm::InputTag>("fedRawDataLabel"))),
      histFolder_(ps.getParameter<std::string>("histFolder")),
      tpFillThreshold_(ps.getUntrackedParameter<int>("etDistributionsFillThreshold", 0)),
      ignoreHFfbs_(ps.getUntrackedParameter<bool>("ignoreHFfbs", false)) {
  ecalTPSentRecd_.reserve(28 * 2 * 72);
  hcalTPSentRecd_.reserve(41 * 2 * 72);
}

L1TStage2CaloLayer1::~L1TStage2CaloLayer1() {}

void L1TStage2CaloLayer1::analyze(const edm::Event& event, const edm::EventSetup& es) {
  // Monitorables stored in Layer 1 raw data but
  // not accessible from existing persistent data formats
  edm::Handle<FEDRawDataCollection> fedRawDataCollection;
  event.getByToken(fedRawData_, fedRawDataCollection);
  bool caloLayer1OutOfRun{true};
  if (fedRawDataCollection.isValid()) {
    caloLayer1OutOfRun = false;
    for (int iFed = 1354; iFed < 1360; iFed += 2) {
      const FEDRawData& fedRawData = fedRawDataCollection->FEDData(iFed);
      if (fedRawData.size() == 0) {
        caloLayer1OutOfRun = true;
        continue;  // In case one of 3 layer 1 FEDs not in
      }
      const uint64_t* fedRawDataArray = (const uint64_t*)fedRawData.data();
      UCTDAQRawData daqData(fedRawDataArray);
      for (uint32_t i = 0; i < daqData.nAMCs(); i++) {
        UCTAMCRawData amcData(daqData.amcPayload(i));
        int lPhi = amcData.layer1Phi();
        if (daqData.BXID() != amcData.BXID()) {
          bxidErrors_->Fill(lPhi);
        }
        if (daqData.L1ID() != amcData.L1ID()) {
          l1idErrors_->Fill(lPhi);
        }
        // AMC payload header has 16 bit orbit number, AMC13 header is full 32
        if ((daqData.orbitNumber() & 0xFFFF) != amcData.orbitNo()) {
          orbitErrors_->Fill(lPhi);
        }
      }
    }
  }

  edm::Handle<EcalTrigPrimDigiCollection> ecalTPsSent;
  event.getByToken(ecalTPSourceSent_, ecalTPsSent);
  edm::Handle<EcalTrigPrimDigiCollection> ecalTPsRecd;
  event.getByToken(ecalTPSourceRecd_, ecalTPsRecd);

  ecalTPSentRecd_.clear();

  ComparisonHelper::zip(ecalTPsSent->begin(),
                        ecalTPsSent->end(),
                        ecalTPsRecd->begin(),
                        ecalTPsRecd->end(),
                        std::inserter(ecalTPSentRecd_, ecalTPSentRecd_.begin()),
                        EcalTrigPrimDigiCollection::key_compare());

  int nEcalLinkErrors{0};
  int nEcalMismatch{0};

  for (const auto& tpPair : ecalTPSentRecd_) {
    auto sentTp = tpPair.first;
    if (sentTp.compressedEt() < 0) {
      // ECal zero-suppresses digis, and a default-constructed
      // digi has et=-1 apparently, but we know it should be zero
      EcalTriggerPrimitiveSample sample(0);
      EcalTriggerPrimitiveDigi tpg(sentTp.id());
      tpg.setSize(1);
      tpg.setSample(0, sample);
      swap(sentTp, tpg);
    }
    const auto& recdTp = tpPair.second;
    const int ieta = sentTp.id().ieta();
    const int iphi = sentTp.id().iphi();
    const bool towerMasked = recdTp.sample(0).raw() & (1 << 13);
    const bool linkMasked = recdTp.sample(0).raw() & (1 << 14);
    const bool linkError = recdTp.sample(0).raw() & (1 << 15);

    // Link status bits from layer 1
    if (towerMasked) {
      ecalOccTowerMasked_->Fill(ieta, iphi);
    }
    if (linkMasked) {
      ecalOccLinkMasked_->Fill(ieta, iphi);
    }

    if (sentTp.compressedEt() > tpFillThreshold_) {
      ecalTPRawEtSent_->Fill(sentTp.compressedEt());
      ecalOccSent_->Fill(ieta, iphi);
    }
    if (sentTp.fineGrain() == 1) {
      ecalOccSentFgVB_->Fill(ieta, iphi);
    }

    if (towerMasked || caloLayer1OutOfRun) {
      // Do not compare if we have a mask applied
      continue;
    }

    if (linkError) {
      ecalLinkError_->Fill(ieta, iphi);
      ecalLinkErrorByLumi_->Fill(event.id().luminosityBlock());
      nEcalLinkErrors++;
      // Don't compare anymore, we already know its bad
      continue;
    }

    ecalTPRawEtCorrelation_->Fill(sentTp.compressedEt(), recdTp.compressedEt());

    if (recdTp.compressedEt() > tpFillThreshold_) {
      ecalTPRawEtRecd_->Fill(recdTp.compressedEt());
      ecalOccupancy_->Fill(ieta, iphi);
      ecalOccRecdEtWgt_->Fill(ieta, iphi, recdTp.compressedEt());
    }
    if (recdTp.fineGrain() == 1) {
      ecalOccRecdFgVB_->Fill(ieta, iphi);
    }

    // Now for comparisons

    const bool EetAgreement = sentTp.compressedEt() == recdTp.compressedEt();
    const bool EfbAgreement = sentTp.fineGrain() == recdTp.fineGrain();
    if (EetAgreement && EfbAgreement) {
      // Full match
      if (sentTp.compressedEt() > tpFillThreshold_) {
        ecalOccSentAndRecd_->Fill(ieta, iphi);
        ecalTPRawEtSentAndRecd_->Fill(sentTp.compressedEt());
      }
    } else {
      // There is some issue
      ecalDiscrepancy_->Fill(ieta, iphi);
      ecalMismatchByLumi_->Fill(event.id().luminosityBlock());
      ECALmismatchesPerBx_->Fill(event.bunchCrossing());
      nEcalMismatch++;

      if (not EetAgreement) {
        ecalOccEtDiscrepancy_->Fill(ieta, iphi);
        ecalTPRawEtDiffNoMatch_->Fill(recdTp.compressedEt() - sentTp.compressedEt());
        updateMismatch(event, 0);

        if (sentTp.compressedEt() == 0)
          ecalOccRecdNotSent_->Fill(ieta, iphi);
        else if (recdTp.compressedEt() == 0)
          ecalOccSentNotRecd_->Fill(ieta, iphi);
        else
          ecalOccNoMatch_->Fill(ieta, iphi);
      }
      if (not EfbAgreement) {
        // occ for fine grain mismatch
        ecalOccFgDiscrepancy_->Fill(ieta, iphi);
        updateMismatch(event, 1);
      }
    }
  }

  if (nEcalLinkErrors > maxEvtLinkErrorsECALCurrentLumi_) {
    maxEvtLinkErrorsECALCurrentLumi_ = nEcalLinkErrors;
  }
  if (nEcalMismatch > maxEvtMismatchECALCurrentLumi_) {
    maxEvtMismatchECALCurrentLumi_ = nEcalMismatch;
  }

  edm::Handle<HcalTrigPrimDigiCollection> hcalTPsSent;
  event.getByToken(hcalTPSourceSent_, hcalTPsSent);
  edm::Handle<HcalTrigPrimDigiCollection> hcalTPsRecd;
  event.getByToken(hcalTPSourceRecd_, hcalTPsRecd);

  hcalTPSentRecd_.clear();

  ComparisonHelper::zip(hcalTPsSent->begin(),
                        hcalTPsSent->end(),
                        hcalTPsRecd->begin(),
                        hcalTPsRecd->end(),
                        std::inserter(hcalTPSentRecd_, hcalTPSentRecd_.begin()),
                        HcalTrigPrimDigiCollection::key_compare());

  int nHcalLinkErrors{0};
  int nHcalMismatch{0};

  for (const auto& tpPair : hcalTPSentRecd_) {
    const auto& sentTp = tpPair.first;
    const auto& recdTp = tpPair.second;
    const int ieta = sentTp.id().ieta();
    if (abs(ieta) > 28 && sentTp.id().version() != 1)
      continue;
    const int iphi = sentTp.id().iphi();
    const bool towerMasked = recdTp.sample(0).raw() & (1 << 13);
    const bool linkMasked = recdTp.sample(0).raw() & (1 << 14);
    const bool linkError = recdTp.sample(0).raw() & (1 << 15);

    if (towerMasked) {
      hcalOccTowerMasked_->Fill(ieta, iphi);
    }
    if (linkMasked) {
      hcalOccLinkMasked_->Fill(ieta, iphi);
    }

    if (sentTp.SOI_compressedEt() > tpFillThreshold_) {
      hcalTPRawEtSent_->Fill(sentTp.SOI_compressedEt());
      hcalOccSent_->Fill(ieta, iphi);
    }
    if (sentTp.SOI_fineGrain() == 1) {
      hcalOccSentFb_->Fill(ieta, iphi);
    }
    if (sentTp.t0().fineGrain(1) == 1) {
      hcalOccSentFb2_->Fill(ieta, iphi);
    }

    if (towerMasked || caloLayer1OutOfRun) {
      // Do not compare if we have a mask applied
      continue;
    }

    if (linkError) {
      hcalLinkError_->Fill(ieta, iphi);
      hcalLinkErrorByLumi_->Fill(event.id().luminosityBlock());
      nHcalLinkErrors++;
      // Don't compare anymore, we already know its bad
      continue;
    }

    if (recdTp.SOI_compressedEt() > tpFillThreshold_) {
      hcalTPRawEtRecd_->Fill(recdTp.SOI_compressedEt());
      hcalOccupancy_->Fill(ieta, iphi);
      hcalOccRecdEtWgt_->Fill(ieta, iphi, recdTp.SOI_compressedEt());
    }
    if (recdTp.SOI_fineGrain()) {
      hcalOccRecdFb_->Fill(ieta, iphi);
    }
    if (recdTp.t0().fineGrain(1)) {
      hcalOccRecdFb2_->Fill(ieta, iphi);
    }

    if (abs(ieta) > 29) {
      hcalTPRawEtCorrelationHF_->Fill(sentTp.SOI_compressedEt(), recdTp.SOI_compressedEt());
    } else {
      hcalTPRawEtCorrelationHBHE_->Fill(sentTp.SOI_compressedEt(), recdTp.SOI_compressedEt());
    }

    const bool HetAgreement = sentTp.SOI_compressedEt() == recdTp.SOI_compressedEt();
    const bool Hfb1Agreement = (abs(ieta) < 29) ? true
                                                : (recdTp.SOI_compressedEt() == 0 ||
                                                   (sentTp.SOI_fineGrain() == recdTp.SOI_fineGrain()) || ignoreHFfbs_);
    // Ignore minBias (FB2) bit if we receieve 0 ET, which means it is likely zero-suppressed on HCal readout side
    const bool Hfb2Agreement =
        (abs(ieta) < 29)
            ? true
            : (recdTp.SOI_compressedEt() == 0 || (sentTp.SOI_fineGrain(1) == recdTp.SOI_fineGrain(1)) || ignoreHFfbs_);
    if (HetAgreement && Hfb1Agreement && Hfb2Agreement) {
      // Full match
      if (sentTp.SOI_compressedEt() > tpFillThreshold_) {
        hcalOccSentAndRecd_->Fill(ieta, iphi);
        hcalTPRawEtSentAndRecd_->Fill(sentTp.SOI_compressedEt());
      }
    } else {
      // There is some issue
      hcalDiscrepancy_->Fill(ieta, iphi);
      hcalMismatchByLumi_->Fill(event.id().luminosityBlock());
      nHcalMismatch++;

      if (not HetAgreement) {
        if (abs(ieta) > 29) {
          HFmismatchesPerBx_->Fill(event.bunchCrossing());
        } else {
          HBHEmismatchesPerBx_->Fill(event.bunchCrossing());
        }
        hcalOccEtDiscrepancy_->Fill(ieta, iphi);
        hcalTPRawEtDiffNoMatch_->Fill(recdTp.SOI_compressedEt() - sentTp.SOI_compressedEt());
        updateMismatch(event, 2);

        // Handle HCal discrepancy debug
        if (sentTp.SOI_compressedEt() == 0)
          hcalOccRecdNotSent_->Fill(ieta, iphi);
        else if (recdTp.SOI_compressedEt() == 0)
          hcalOccSentNotRecd_->Fill(ieta, iphi);
        else
          hcalOccNoMatch_->Fill(ieta, iphi);
      }
      if (not Hfb1Agreement) {
        // Handle fine grain discrepancies
        hcalOccFbDiscrepancy_->Fill(ieta, iphi);
        updateMismatch(event, 3);
      }
      if (not Hfb2Agreement) {
        // Handle fine grain discrepancies
        hcalOccFb2Discrepancy_->Fill(ieta, iphi);
        updateMismatch(event, 3);
      }
    }
  }

  if (nHcalLinkErrors > maxEvtLinkErrorsHCALCurrentLumi_) {
    maxEvtLinkErrorsHCALCurrentLumi_ = nHcalLinkErrors;
  }
  if (nHcalMismatch > maxEvtMismatchHCALCurrentLumi_) {
    maxEvtMismatchHCALCurrentLumi_ = nHcalMismatch;
  }
}

void L1TStage2CaloLayer1::updateMismatch(const edm::Event& e, int mismatchType) {
  auto id = e.id();
  std::string eventString{std::to_string(id.run()) + ":" + std::to_string(id.luminosityBlock()) + ":" +
                          std::to_string(id.event())};
  if (last20MismatchArray_.at(lastMismatchIndex_).first == eventString) {
    // same event
    last20MismatchArray_.at(lastMismatchIndex_).second |= 1 << mismatchType;
  } else {
    // New event, advance
    lastMismatchIndex_ = (lastMismatchIndex_ + 1) % 20;
    last20MismatchArray_.at(lastMismatchIndex_) = {eventString, 1 << mismatchType};
  }
}

void L1TStage2CaloLayer1::beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) {
  // Ugly way to loop backwards through the last 20 mismatches
  auto h = last20Mismatches_;
  for (size_t ibin = 1, imatch = lastMismatchIndex_; ibin <= 20; ibin++, imatch = (imatch + 19) % 20) {
    h->setBinLabel(ibin, last20MismatchArray_.at(imatch).first, /* axis */ 2);
    for (int itype = 0; itype < h->getNbinsX(); ++itype) {
      int binContent = (last20MismatchArray_.at(imatch).second >> itype) & 1;
      last20Mismatches_->setBinContent(itype + 1, ibin, binContent);
    }
  }
}

void L1TStage2CaloLayer1::endLuminosityBlock(const edm::LuminosityBlock& lumi, const edm::EventSetup&) {
  auto id = static_cast<double>(lumi.id().luminosityBlock());  // uint64_t

  if (maxEvtLinkErrorsECALCurrentLumi_ > 0) {
    maxEvtLinkErrorsByLumiECAL_->Fill(id, maxEvtLinkErrorsECALCurrentLumi_);
  }
  if (maxEvtLinkErrorsHCALCurrentLumi_ > 0) {
    maxEvtLinkErrorsByLumiHCAL_->Fill(id, maxEvtLinkErrorsHCALCurrentLumi_);
  }
  if (maxEvtLinkErrorsECALCurrentLumi_ + maxEvtLinkErrorsHCALCurrentLumi_ > 0) {
    maxEvtLinkErrorsByLumi_->Fill(id, maxEvtLinkErrorsECALCurrentLumi_ + maxEvtLinkErrorsHCALCurrentLumi_);
  }
  maxEvtLinkErrorsECALCurrentLumi_ = 0;
  maxEvtLinkErrorsHCALCurrentLumi_ = 0;

  if (maxEvtMismatchECALCurrentLumi_ > 0) {
    maxEvtMismatchByLumiECAL_->Fill(id, maxEvtMismatchECALCurrentLumi_);
  }
  if (maxEvtMismatchHCALCurrentLumi_ > 0) {
    maxEvtMismatchByLumiHCAL_->Fill(id, maxEvtMismatchHCALCurrentLumi_);
  }
  if (maxEvtMismatchECALCurrentLumi_ + maxEvtMismatchHCALCurrentLumi_ > 0) {
    maxEvtMismatchByLumi_->Fill(id, maxEvtMismatchECALCurrentLumi_ + maxEvtMismatchHCALCurrentLumi_);
  }
  maxEvtMismatchECALCurrentLumi_ = 0;
  maxEvtMismatchHCALCurrentLumi_ = 0;

  // Simple way to embed current lumi to auto-scale axis limits in render plugin
  ecalLinkErrorByLumi_->setBinContent(0, id);
  ecalMismatchByLumi_->setBinContent(0, id);
  hcalLinkErrorByLumi_->setBinContent(0, id);
  hcalMismatchByLumi_->setBinContent(0, id);
  maxEvtLinkErrorsByLumiECAL_->setBinContent(0, id);
  maxEvtLinkErrorsByLumiHCAL_->setBinContent(0, id);
  maxEvtLinkErrorsByLumi_->setBinContent(0, id);
  maxEvtMismatchByLumiECAL_->setBinContent(0, id);
  maxEvtMismatchByLumiHCAL_->setBinContent(0, id);
  maxEvtMismatchByLumi_->setBinContent(0, id);
}

void L1TStage2CaloLayer1::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run& run, const edm::EventSetup& es) {
  auto bookEt = [&ibooker](std::string name, std::string title) {
    return ibooker.book1D(name, title + ";Raw ET;Counts", 256, -0.5, 255.5);
  };
  auto bookEtCorrelation = [&ibooker](std::string name, std::string title) {
    return ibooker.book2D(name, title, 256, -0.5, 255.5, 256, -0.5, 255.5);
  };
  auto bookEtDiff = [&ibooker](std::string name, std::string title) {
    return ibooker.book1D(name, title + ";#Delta Raw ET;Counts", 511, -255.5, 255.5);
  };
  auto bookEcalOccupancy = [&ibooker](std::string name, std::string title) {
    return ibooker.book2D(name, title + ";iEta;iPhi", 57, -28.5, 28.5, 72, 0.5, 72.5);
  };
  auto bookHcalOccupancy = [&ibooker](std::string name, std::string title) {
    return ibooker.book2D(name, title + ";iEta;iPhi", 83, -41.5, 41.5, 72, 0.5, 72.5);
  };

  ibooker.setCurrentFolder(histFolder_);

  ecalDiscrepancy_ = bookEcalOccupancy("ecalDiscrepancy", "ECAL Discrepancies between TCC and Layer1 Readout");
  ecalLinkError_ = bookEcalOccupancy("ecalLinkError", "ECAL Link Errors");
  ecalOccupancy_ = bookEcalOccupancy("ecalOccupancy", "ECAL TP Occupancy at Layer1");
  ecalOccRecdEtWgt_ = bookEcalOccupancy("ecalOccRecdEtWgt", "ECal TP ET-weighted Occupancy at Layer1");
  hcalDiscrepancy_ = bookHcalOccupancy("hcalDiscrepancy", "HCAL Discrepancies between uHTR and Layer1 Readout");
  hcalLinkError_ = bookHcalOccupancy("hcalLinkError", "HCAL Link Errors");
  hcalOccupancy_ = bookHcalOccupancy("hcalOccupancy", "HCAL TP Occupancy at Layer1");
  hcalOccRecdEtWgt_ = bookHcalOccupancy("hcalOccRecdEtWgt", "HCal TP ET-weighted Occupancy at Layer1");

  ibooker.setCurrentFolder(histFolder_ + "/ECalDetail");

  ecalOccEtDiscrepancy_ = bookEcalOccupancy("ecalOccEtDiscrepancy", "ECal Et Discrepancy Occupancy");
  ecalOccFgDiscrepancy_ = bookEcalOccupancy("ecalOccFgDiscrepancy", "ECal FG Veto Bit Discrepancy Occupancy");
  ecalOccLinkMasked_ = bookEcalOccupancy("ecalOccLinkMasked", "ECal Masked Links");
  ecalOccRecdFgVB_ = bookEcalOccupancy("ecalOccRecdFgVB", "ECal FineGrain Veto Bit Occupancy at Layer1");
  ecalOccSentAndRecd_ = bookEcalOccupancy("ecalOccSentAndRecd", "ECal TP Occupancy FULL MATCH");
  ecalOccSentFgVB_ = bookEcalOccupancy("ecalOccSentFgVB", "ECal FineGrain Veto Bit Occupancy at TCC");
  ecalOccSent_ = bookEcalOccupancy("ecalOccSent", "ECal TP Occupancy at TCC");
  ecalOccTowerMasked_ = bookEcalOccupancy("ecalOccTowerMasked", "ECal Masked towers");
  ecalTPRawEtCorrelation_ =
      bookEtCorrelation("ecalTPRawEtCorrelation", "Raw Et correlation TCC and Layer1;TCC Et;Layer1 Et");
  ecalTPRawEtDiffNoMatch_ = bookEtDiff("ecalTPRawEtDiffNoMatch", "ECal Raw Et Difference Layer1 - TCC");
  ecalTPRawEtRecd_ = bookEt("ecalTPRawEtRecd", "ECal Raw Et Layer1 Readout");
  ecalTPRawEtSentAndRecd_ = bookEt("ecalTPRawEtMatch", "ECal Raw Et FULL MATCH");
  ecalTPRawEtSent_ = bookEt("ecalTPRawEtSent", "ECal Raw Et TCC Readout");

  ibooker.setCurrentFolder(histFolder_ + "/ECalDetail/TCCDebug");
  ecalOccSentNotRecd_ = bookHcalOccupancy("ecalOccSentNotRecd", "ECal TP Occupancy sent by TCC, zero at Layer1");
  ecalOccRecdNotSent_ = bookHcalOccupancy("ecalOccRecdNotSent", "ECal TP Occupancy received by Layer1, zero at TCC");
  ecalOccNoMatch_ = bookHcalOccupancy("ecalOccNoMatch", "ECal TP Occupancy for TCC and Layer1 nonzero, not matching");

  ibooker.setCurrentFolder(histFolder_ + "/HCalDetail");

  hcalOccEtDiscrepancy_ = bookHcalOccupancy("hcalOccEtDiscrepancy", "HCal Et Discrepancy Occupancy");
  hcalOccFbDiscrepancy_ = bookHcalOccupancy("hcalOccFbDiscrepancy", "HCal Feature Bit Discrepancy Occupancy");
  hcalOccFb2Discrepancy_ = bookHcalOccupancy("hcalOccFb2Discrepancy", "HCal Second Feature Bit Discrepancy Occupancy");
  hcalOccLinkMasked_ = bookHcalOccupancy("hcalOccLinkMasked", "HCal Masked Links");
  hcalOccRecdFb_ = bookHcalOccupancy("hcalOccRecdFb", "HCal Feature Bit Occupancy at Layer1");
  hcalOccRecdFb2_ = bookHcalOccupancy("hcalOccRecdFb2", "HF Second Feature Bit Occupancy at Layer1");
  hcalOccSentAndRecd_ = bookHcalOccupancy("hcalOccSentAndRecd", "HCal TP Occupancy FULL MATCH");
  hcalOccSentFb_ = bookHcalOccupancy("hcalOccSentFb", "HCal Feature Bit Occupancy at uHTR");
  hcalOccSentFb2_ = bookHcalOccupancy("hcalOccSentFb2", "HF Second Feature Bit Occupancy at uHTR");
  hcalOccSent_ = bookHcalOccupancy("hcalOccSent", "HCal TP Occupancy at uHTR");
  hcalOccTowerMasked_ = bookHcalOccupancy("hcalOccTowerMasked", "HCal Masked towers");
  hcalTPRawEtCorrelationHBHE_ =
      bookEtCorrelation("hcalTPRawEtCorrelationHBHE", "HBHE Raw Et correlation uHTR and Layer1;uHTR Et;Layer1 Et");
  hcalTPRawEtCorrelationHF_ =
      bookEtCorrelation("hcalTPRawEtCorrelationHF", "HF Raw Et correlation uHTR and Layer1;uHTR Et;Layer1 Et");
  hcalTPRawEtDiffNoMatch_ = bookEtDiff("hcalTPRawEtDiffNoMatch", "HCal Raw Et Difference Layer1 - uHTR");
  hcalTPRawEtRecd_ = bookEt("hcalTPRawEtRecd", "HCal Raw Et Layer1 Readout");
  hcalTPRawEtSentAndRecd_ = bookEt("hcalTPRawEtMatch", "HCal Raw Et FULL MATCH");
  hcalTPRawEtSent_ = bookEt("hcalTPRawEtSent", "HCal Raw Et uHTR Readout");

  ibooker.setCurrentFolder(histFolder_ + "/HCalDetail/uHTRDebug");
  hcalOccSentNotRecd_ = bookHcalOccupancy("hcalOccSentNotRecd", "HCal TP Occupancy sent by uHTR, zero at Layer1");
  hcalOccRecdNotSent_ = bookHcalOccupancy("hcalOccRecdNotSent", "HCal TP Occupancy received by Layer1, zero at uHTR");
  hcalOccNoMatch_ = bookHcalOccupancy("hcalOccNoMatch", "HCal TP Occupancy for uHTR and Layer1 nonzero, not matching");

  ibooker.setCurrentFolder(histFolder_ + "/MismatchDetail");

  const int nMismatchTypes = 4;
  last20Mismatches_ = ibooker.book2D("last20Mismatches",
                                     "Log of last 20 mismatches (use json tool to copy/paste)",
                                     nMismatchTypes,
                                     0,
                                     nMismatchTypes,
                                     20,
                                     0,
                                     20);
  last20Mismatches_->setBinLabel(1, "Ecal TP Et Mismatch");
  last20Mismatches_->setBinLabel(2, "Ecal TP Fine Grain Bit Mismatch");
  last20Mismatches_->setBinLabel(3, "Hcal TP Et Mismatch");
  last20Mismatches_->setBinLabel(4, "Hcal TP Feature Bit Mismatch");
  for (size_t i = 0; i < last20MismatchArray_.size(); ++i)
    last20MismatchArray_[i] = {"-" + std::to_string(i), 0};
  for (size_t i = 1; i <= 20; ++i)
    last20Mismatches_->setBinLabel(i, "-" + std::to_string(i), /* axis */ 2);

  const int nLumis = 2000;
  ecalLinkErrorByLumi_ = ibooker.book1D(
      "ecalLinkErrorByLumi", "Link error counts per lumi section for ECAL;LumiSection;Counts", nLumis, .5, nLumis + 0.5);
  ecalMismatchByLumi_ = ibooker.book1D(
      "ecalMismatchByLumi", "Mismatch counts per lumi section for ECAL;LumiSection;Counts", nLumis, .5, nLumis + 0.5);
  hcalLinkErrorByLumi_ = ibooker.book1D(
      "hcalLinkErrorByLumi", "Link error counts per lumi section for HCAL;LumiSection;Counts", nLumis, .5, nLumis + 0.5);
  hcalMismatchByLumi_ = ibooker.book1D(
      "hcalMismatchByLumi", "Mismatch counts per lumi section for HCAL;LumiSection;Counts", nLumis, .5, nLumis + 0.5);

  ECALmismatchesPerBx_ =
      ibooker.book1D("ECALmismatchesPerBx", "Mismatch counts per bunch crossing for ECAL", 3564, -.5, 3563.5);
  HBHEmismatchesPerBx_ =
      ibooker.book1D("HBHEmismatchesPerBx", "Mismatch counts per bunch crossing for HBHE", 3564, -.5, 3563.5);
  HFmismatchesPerBx_ =
      ibooker.book1D("HFmismatchesPerBx", "Mismatch counts per bunch crossing for HF", 3564, -.5, 3563.5);

  maxEvtLinkErrorsByLumiECAL_ =
      ibooker.book1D("maxEvtLinkErrorsByLumiECAL",
                     "Max number of single-event ECAL link errors per lumi section;LumiSection;Counts",
                     nLumis,
                     .5,
                     nLumis + 0.5);
  maxEvtLinkErrorsByLumiHCAL_ =
      ibooker.book1D("maxEvtLinkErrorsByLumiHCAL",
                     "Max number of single-event HCAL link errors per lumi section;LumiSection;Counts",
                     nLumis,
                     .5,
                     nLumis + 0.5);

  maxEvtMismatchByLumiECAL_ =
      ibooker.book1D("maxEvtMismatchByLumiECAL",
                     "Max number of single-event ECAL discrepancies per lumi section;LumiSection;Counts",
                     nLumis,
                     .5,
                     nLumis + 0.5);
  maxEvtMismatchByLumiHCAL_ =
      ibooker.book1D("maxEvtMismatchByLumiHCAL",
                     "Max number of single-event HCAL discrepancies per lumi section;LumiSection;Counts",
                     nLumis,
                     .5,
                     nLumis + 0.5);

  ibooker.setCurrentFolder(histFolder_);
  maxEvtLinkErrorsByLumi_ = ibooker.book1D("maxEvtLinkErrorsByLumi",
                                           "Max number of single-event link errors per lumi section;LumiSection;Counts",
                                           nLumis,
                                           .5,
                                           nLumis + 0.5);
  maxEvtMismatchByLumi_ = ibooker.book1D("maxEvtMismatchByLumi",
                                         "Max number of single-event discrepancies per lumi section;LumiSection;Counts",
                                         nLumis,
                                         .5,
                                         nLumis + 0.5);

  ibooker.setCurrentFolder(histFolder_ + "/AMC13ErrorCounters");
  bxidErrors_ =
      ibooker.book1D("bxidErrors", "bxid mismatch between AMC13 and CTP Cards;Layer1 Phi;Counts", 18, -.5, 17.5);
  l1idErrors_ =
      ibooker.book1D("l1idErrors", "l1id mismatch between AMC13 and CTP Cards;Layer1 Phi;Counts", 18, -.5, 17.5);
  orbitErrors_ =
      ibooker.book1D("orbitErrors", "orbit mismatch between AMC13 and CTP Cards;Layer1 Phi;Counts", 18, -.5, 17.5);
}
