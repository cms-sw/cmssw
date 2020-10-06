/*
 * \file L1TStage2CaloLayer1Offline.cc
 *
 * N. Smith <nick.smith@cern.ch>
 */
//Modified by Bhawna Gomber <bhawna.gomber@cern.ch>
//Modified into offline version by Andrew Loeliger. <andrew.loeliger@cern.ch>

#include "DQMOffline/L1Trigger/interface/L1TStage2CaloLayer1Offline.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"

#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"

#include "EventFilter/L1TXRawToDigi/plugins/UCTDAQRawData.h"
#include "EventFilter/L1TXRawToDigi/plugins/UCTAMCRawData.h"

L1TStage2CaloLayer1Offline::L1TStage2CaloLayer1Offline(const edm::ParameterSet& ps)
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
      ignoreHFfbs_(ps.getUntrackedParameter<bool>("ignoreHFfbs", false)) {}

L1TStage2CaloLayer1Offline::~L1TStage2CaloLayer1Offline() {}

void L1TStage2CaloLayer1Offline::dqmAnalyze(const edm::Event& event,
                                            const edm::EventSetup& es,
                                            const CaloL1Information::data& data) const {
  //This must be moved here compared to online version to avoid a const function modifying a class variable
  //This unfortunately means that the variable is local and reallocated per event
  std::vector<std::pair<EcalTriggerPrimitiveDigi, EcalTriggerPrimitiveDigi> > ecalTPSentRecd_;
  std::vector<std::pair<HcalTriggerPrimitiveDigi, HcalTriggerPrimitiveDigi> > hcalTPSentRecd_;
  ecalTPSentRecd_.reserve(28 * 2 * 72);
  hcalTPSentRecd_.reserve(41 * 2 * 72);

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
          data.bxidErrors_->Fill(lPhi);
        }
        if (daqData.L1ID() != amcData.L1ID()) {
          data.l1idErrors_->Fill(lPhi);
        }
        // AMC payload header has 16 bit orbit number, AMC13 header is full 32
        if ((daqData.orbitNumber() & 0xFFFF) != amcData.orbitNo()) {
          data.orbitErrors_->Fill(lPhi);
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
      data.ecalOccTowerMasked_->Fill(ieta, iphi);
    }
    if (linkMasked) {
      data.ecalOccLinkMasked_->Fill(ieta, iphi);
    }

    if (sentTp.compressedEt() > tpFillThreshold_) {
      data.ecalTPRawEtSent_->Fill(sentTp.compressedEt());
      data.ecalOccSent_->Fill(ieta, iphi);
    }
    if (sentTp.fineGrain() == 1) {
      data.ecalOccSentFgVB_->Fill(ieta, iphi);
    }

    if (towerMasked || caloLayer1OutOfRun) {
      // Do not compare if we have a mask applied
      continue;
    }

    if (linkError) {
      data.ecalLinkError_->Fill(ieta, iphi);
      data.ecalLinkErrorByLumi_->Fill(event.id().luminosityBlock());
      nEcalLinkErrors++;
      // Don't compare anymore, we already know its bad
      continue;
    }

    data.ecalTPRawEtCorrelation_->Fill(sentTp.compressedEt(), recdTp.compressedEt());

    if (recdTp.compressedEt() > tpFillThreshold_) {
      data.ecalTPRawEtRecd_->Fill(recdTp.compressedEt());
      data.ecalOccupancy_->Fill(ieta, iphi);
      data.ecalOccRecdEtWgt_->Fill(ieta, iphi, recdTp.compressedEt());
    }
    if (recdTp.fineGrain() == 1) {
      data.ecalOccRecdFgVB_->Fill(ieta, iphi);
    }

    // Now for comparisons

    const bool EetAgreement = sentTp.compressedEt() == recdTp.compressedEt();
    const bool EfbAgreement = sentTp.fineGrain() == recdTp.fineGrain();
    if (EetAgreement && EfbAgreement) {
      // Full match
      if (sentTp.compressedEt() > tpFillThreshold_) {
        data.ecalOccSentAndRecd_->Fill(ieta, iphi);
        data.ecalTPRawEtSentAndRecd_->Fill(sentTp.compressedEt());
      }
    } else {
      // There is some issue
      data.ecalDiscrepancy_->Fill(ieta, iphi);
      data.ecalMismatchByLumi_->Fill(event.id().luminosityBlock());
      data.ECALmismatchesPerBx_->Fill(event.bunchCrossing());
      nEcalMismatch++;

      if (not EetAgreement) {
        data.ecalOccEtDiscrepancy_->Fill(ieta, iphi);
        data.ecalTPRawEtDiffNoMatch_->Fill(recdTp.compressedEt() - sentTp.compressedEt());

        if (sentTp.compressedEt() == 0)
          data.ecalOccRecdNotSent_->Fill(ieta, iphi);
        else if (recdTp.compressedEt() == 0)
          data.ecalOccSentNotRecd_->Fill(ieta, iphi);
        else
          data.ecalOccNoMatch_->Fill(ieta, iphi);
      }
      if (not EfbAgreement) {
        // occ for fine grain mismatch
        data.ecalOccFgDiscrepancy_->Fill(ieta, iphi);
      }
    }
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
      data.hcalOccTowerMasked_->Fill(ieta, iphi);
    }
    if (linkMasked) {
      data.hcalOccLinkMasked_->Fill(ieta, iphi);
    }

    if (sentTp.SOI_compressedEt() > tpFillThreshold_) {
      data.hcalTPRawEtSent_->Fill(sentTp.SOI_compressedEt());
      data.hcalOccSent_->Fill(ieta, iphi);
    }
    if (sentTp.SOI_fineGrain() == 1) {
      data.hcalOccSentFb_->Fill(ieta, iphi);
    }
    if (sentTp.t0().fineGrain(1) == 1) {
      data.hcalOccSentFb2_->Fill(ieta, iphi);
    }

    if (towerMasked || caloLayer1OutOfRun) {
      // Do not compare if we have a mask applied
      continue;
    }

    if (linkError) {
      data.hcalLinkError_->Fill(ieta, iphi);
      data.hcalLinkErrorByLumi_->Fill(event.id().luminosityBlock());
      nHcalLinkErrors++;
      // Don't compare anymore, we already know its bad
      continue;
    }

    if (recdTp.SOI_compressedEt() > tpFillThreshold_) {
      data.hcalTPRawEtRecd_->Fill(recdTp.SOI_compressedEt());
      data.hcalOccupancy_->Fill(ieta, iphi);
      data.hcalOccRecdEtWgt_->Fill(ieta, iphi, recdTp.SOI_compressedEt());
    }
    if (recdTp.SOI_fineGrain()) {
      data.hcalOccRecdFb_->Fill(ieta, iphi);
    }
    if (recdTp.t0().fineGrain(1)) {
      data.hcalOccRecdFb2_->Fill(ieta, iphi);
    }

    if (abs(ieta) > 29) {
      data.hcalTPRawEtCorrelationHF_->Fill(sentTp.SOI_compressedEt(), recdTp.SOI_compressedEt());
    } else {
      data.hcalTPRawEtCorrelationHBHE_->Fill(sentTp.SOI_compressedEt(), recdTp.SOI_compressedEt());
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
        data.hcalOccSentAndRecd_->Fill(ieta, iphi);
        data.hcalTPRawEtSentAndRecd_->Fill(sentTp.SOI_compressedEt());
      }
    } else {
      // There is some issue
      data.hcalDiscrepancy_->Fill(ieta, iphi);
      data.hcalMismatchByLumi_->Fill(event.id().luminosityBlock());
      nHcalMismatch++;

      if (not HetAgreement) {
        if (abs(ieta) > 29) {
          data.HFmismatchesPerBx_->Fill(event.bunchCrossing());
        } else {
          data.HBHEmismatchesPerBx_->Fill(event.bunchCrossing());
        }
        data.hcalOccEtDiscrepancy_->Fill(ieta, iphi);
        data.hcalTPRawEtDiffNoMatch_->Fill(recdTp.SOI_compressedEt() - sentTp.SOI_compressedEt());

        // Handle HCal discrepancy debug
        if (sentTp.SOI_compressedEt() == 0)
          data.hcalOccRecdNotSent_->Fill(ieta, iphi);
        else if (recdTp.SOI_compressedEt() == 0)
          data.hcalOccSentNotRecd_->Fill(ieta, iphi);
        else
          data.hcalOccNoMatch_->Fill(ieta, iphi);
      }
      if (not Hfb1Agreement) {
        // Handle fine grain discrepancies
        data.hcalOccFbDiscrepancy_->Fill(ieta, iphi);
      }
      if (not Hfb2Agreement) {
        // Handle fine grain discrepancies
        data.hcalOccFb2Discrepancy_->Fill(ieta, iphi);
      }
    }
  }

  //autoscales axis limits in render plugin
  auto id = static_cast<double>(event.id().luminosityBlock());
  data.ecalLinkErrorByLumi_->setBinContent(0, id);
  data.ecalMismatchByLumi_->setBinContent(0, id);
  data.hcalLinkErrorByLumi_->setBinContent(0, id);
  data.hcalMismatchByLumi_->setBinContent(0, id);
}

void L1TStage2CaloLayer1Offline::bookHistograms(DQMStore::IBooker& ibooker,
                                                const edm::Run& run,
                                                const edm::EventSetup& es,
                                                CaloL1Information::data& data) const {
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

  data.ecalDiscrepancy_ = bookEcalOccupancy("ecalDiscrepancy", "ECAL Discrepancies between TCC and Layer1 Readout");
  data.ecalLinkError_ = bookEcalOccupancy("ecalLinkError", "ECAL Link Errors");
  data.ecalOccupancy_ = bookEcalOccupancy("ecalOccupancy", "ECAL TP Occupancy at Layer1");
  data.ecalOccRecdEtWgt_ = bookEcalOccupancy("ecalOccRecdEtWgt", "ECal TP ET-weighted Occupancy at Layer1");
  data.hcalDiscrepancy_ = bookHcalOccupancy("hcalDiscrepancy", "HCAL Discrepancies between uHTR and Layer1 Readout");
  data.hcalLinkError_ = bookHcalOccupancy("hcalLinkError", "HCAL Link Errors");
  data.hcalOccupancy_ = bookHcalOccupancy("hcalOccupancy", "HCAL TP Occupancy at Layer1");
  data.hcalOccRecdEtWgt_ = bookHcalOccupancy("hcalOccRecdEtWgt", "HCal TP ET-weighted Occupancy at Layer1");

  ibooker.setCurrentFolder(histFolder_ + "/ECalDetail");

  data.ecalOccEtDiscrepancy_ = bookEcalOccupancy("ecalOccEtDiscrepancy", "ECal Et Discrepancy Occupancy");
  data.ecalOccFgDiscrepancy_ = bookEcalOccupancy("ecalOccFgDiscrepancy", "ECal FG Veto Bit Discrepancy Occupancy");
  data.ecalOccLinkMasked_ = bookEcalOccupancy("ecalOccLinkMasked", "ECal Masked Links");
  data.ecalOccRecdFgVB_ = bookEcalOccupancy("ecalOccRecdFgVB", "ECal FineGrain Veto Bit Occupancy at Layer1");
  data.ecalOccSentAndRecd_ = bookEcalOccupancy("ecalOccSentAndRecd", "ECal TP Occupancy FULL MATCH");
  data.ecalOccSentFgVB_ = bookEcalOccupancy("ecalOccSentFgVB", "ECal FineGrain Veto Bit Occupancy at TCC");
  data.ecalOccSent_ = bookEcalOccupancy("ecalOccSent", "ECal TP Occupancy at TCC");
  data.ecalOccTowerMasked_ = bookEcalOccupancy("ecalOccTowerMasked", "ECal Masked towers");
  data.ecalTPRawEtCorrelation_ =
      bookEtCorrelation("ecalTPRawEtCorrelation", "Raw Et correlation TCC and Layer1;TCC Et;Layer1 Et");
  data.ecalTPRawEtDiffNoMatch_ = bookEtDiff("ecalTPRawEtDiffNoMatch", "ECal Raw Et Difference Layer1 - TCC");
  data.ecalTPRawEtRecd_ = bookEt("ecalTPRawEtRecd", "ECal Raw Et Layer1 Readout");
  data.ecalTPRawEtSentAndRecd_ = bookEt("ecalTPRawEtMatch", "ECal Raw Et FULL MATCH");
  data.ecalTPRawEtSent_ = bookEt("ecalTPRawEtSent", "ECal Raw Et TCC Readout");

  ibooker.setCurrentFolder(histFolder_ + "/ECalDetail/TCCDebug");
  data.ecalOccSentNotRecd_ = bookHcalOccupancy("ecalOccSentNotRecd", "ECal TP Occupancy sent by TCC, zero at Layer1");
  data.ecalOccRecdNotSent_ =
      bookHcalOccupancy("ecalOccRecdNotSent", "ECal TP Occupancy received by Layer1, zero at TCC");
  data.ecalOccNoMatch_ =
      bookHcalOccupancy("ecalOccNoMatch", "ECal TP Occupancy for TCC and Layer1 nonzero, not matching");

  ibooker.setCurrentFolder(histFolder_ + "/HCalDetail");

  data.hcalOccEtDiscrepancy_ = bookHcalOccupancy("hcalOccEtDiscrepancy", "HCal Et Discrepancy Occupancy");
  data.hcalOccFbDiscrepancy_ = bookHcalOccupancy("hcalOccFbDiscrepancy", "HCal Feature Bit Discrepancy Occupancy");
  data.hcalOccFb2Discrepancy_ =
      bookHcalOccupancy("hcalOccFb2Discrepancy", "HCal Second Feature Bit Discrepancy Occupancy");
  data.hcalOccLinkMasked_ = bookHcalOccupancy("hcalOccLinkMasked", "HCal Masked Links");
  data.hcalOccRecdFb_ = bookHcalOccupancy("hcalOccRecdFb", "HCal Feature Bit Occupancy at Layer1");
  data.hcalOccRecdFb2_ = bookHcalOccupancy("hcalOccRecdFb2", "HF Second Feature Bit Occupancy at Layer1");
  data.hcalOccSentAndRecd_ = bookHcalOccupancy("hcalOccSentAndRecd", "HCal TP Occupancy FULL MATCH");
  data.hcalOccSentFb_ = bookHcalOccupancy("hcalOccSentFb", "HCal Feature Bit Occupancy at uHTR");
  data.hcalOccSentFb2_ = bookHcalOccupancy("hcalOccSentFb2", "HF Second Feature Bit Occupancy at uHTR");
  data.hcalOccSent_ = bookHcalOccupancy("hcalOccSent", "HCal TP Occupancy at uHTR");
  data.hcalOccTowerMasked_ = bookHcalOccupancy("hcalOccTowerMasked", "HCal Masked towers");
  data.hcalTPRawEtCorrelationHBHE_ =
      bookEtCorrelation("hcalTPRawEtCorrelationHBHE", "HBHE Raw Et correlation uHTR and Layer1;uHTR Et;Layer1 Et");
  data.hcalTPRawEtCorrelationHF_ =
      bookEtCorrelation("hcalTPRawEtCorrelationHF", "HF Raw Et correlation uHTR and Layer1;uHTR Et;Layer1 Et");
  data.hcalTPRawEtDiffNoMatch_ = bookEtDiff("hcalTPRawEtDiffNoMatch", "HCal Raw Et Difference Layer1 - uHTR");
  data.hcalTPRawEtRecd_ = bookEt("hcalTPRawEtRecd", "HCal Raw Et Layer1 Readout");
  data.hcalTPRawEtSentAndRecd_ = bookEt("hcalTPRawEtMatch", "HCal Raw Et FULL MATCH");
  data.hcalTPRawEtSent_ = bookEt("hcalTPRawEtSent", "HCal Raw Et uHTR Readout");

  ibooker.setCurrentFolder(histFolder_ + "/HCalDetail/uHTRDebug");
  data.hcalOccSentNotRecd_ = bookHcalOccupancy("hcalOccSentNotRecd", "HCal TP Occupancy sent by uHTR, zero at Layer1");
  data.hcalOccRecdNotSent_ =
      bookHcalOccupancy("hcalOccRecdNotSent", "HCal TP Occupancy received by Layer1, zero at uHTR");
  data.hcalOccNoMatch_ =
      bookHcalOccupancy("hcalOccNoMatch", "HCal TP Occupancy for uHTR and Layer1 nonzero, not matching");

  ibooker.setCurrentFolder(histFolder_ + "/MismatchDetail");

  const int nLumis = 2000;
  data.ecalLinkErrorByLumi_ = ibooker.book1D(
      "ecalLinkErrorByLumi", "Link error counts per lumi section for ECAL;LumiSection;Counts", nLumis, .5, nLumis + 0.5);
  data.ecalMismatchByLumi_ = ibooker.book1D(
      "ecalMismatchByLumi", "Mismatch counts per lumi section for ECAL;LumiSection;Counts", nLumis, .5, nLumis + 0.5);
  data.hcalLinkErrorByLumi_ = ibooker.book1D(
      "hcalLinkErrorByLumi", "Link error counts per lumi section for HCAL;LumiSection;Counts", nLumis, .5, nLumis + 0.5);
  data.hcalMismatchByLumi_ = ibooker.book1D(
      "hcalMismatchByLumi", "Mismatch counts per lumi section for HCAL;LumiSection;Counts", nLumis, .5, nLumis + 0.5);

  data.ECALmismatchesPerBx_ =
      ibooker.book1D("ECALmismatchesPerBx", "Mismatch counts per bunch crossing for ECAL", 3564, -.5, 3563.5);
  data.HBHEmismatchesPerBx_ =
      ibooker.book1D("HBHEmismatchesPerBx", "Mismatch counts per bunch crossing for HBHE", 3564, -.5, 3563.5);
  data.HFmismatchesPerBx_ =
      ibooker.book1D("HFmismatchesPerBx", "Mismatch counts per bunch crossing for HF", 3564, -.5, 3563.5);

  ibooker.setCurrentFolder(histFolder_ + "/AMC13ErrorCounters");
  data.bxidErrors_ =
      ibooker.book1D("bxidErrors", "bxid mismatch between AMC13 and CTP Cards;Layer1 Phi;Counts", 18, -.5, 17.5);
  data.l1idErrors_ =
      ibooker.book1D("l1idErrors", "l1id mismatch between AMC13 and CTP Cards;Layer1 Phi;Counts", 18, -.5, 17.5);
  data.orbitErrors_ =
      ibooker.book1D("orbitErrors", "orbit mismatch between AMC13 and CTP Cards;Layer1 Phi;Counts", 18, -.5, 17.5);
}

//Define this module as a framework plugin
DEFINE_FWK_MODULE(L1TStage2CaloLayer1Offline);
