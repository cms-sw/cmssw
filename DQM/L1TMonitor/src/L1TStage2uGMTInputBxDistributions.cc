#include "DQM/L1TMonitor/interface/L1TStage2uGMTInputBxDistributions.h"

L1TStage2uGMTInputBxDistributions::L1TStage2uGMTInputBxDistributions(const edm::ParameterSet& ps)
    : ugmtMuonToken_(consumes<l1t::MuonBxCollection>(ps.getParameter<edm::InputTag>("muonProducer"))),
      ugmtMuonShowerToken_(consumes<l1t::MuonShowerBxCollection>(ps.getParameter<edm::InputTag>("muonShowerProducer"))),
      monitorDir_(ps.getUntrackedParameter<std::string>("monitorDir")),
      emul_(ps.getUntrackedParameter<bool>("emulator")),
      verbose_(ps.getUntrackedParameter<bool>("verbose")),
      hadronicShowers_(ps.getUntrackedParameter<bool>("hadronicShowers")) {
  if (!emul_) {
    ugmtBMTFToken_ = consumes<l1t::RegionalMuonCandBxCollection>(ps.getParameter<edm::InputTag>("bmtfProducer"));
    ugmtOMTFToken_ = consumes<l1t::RegionalMuonCandBxCollection>(ps.getParameter<edm::InputTag>("omtfProducer"));
    ugmtEMTFToken_ = consumes<l1t::RegionalMuonCandBxCollection>(ps.getParameter<edm::InputTag>("emtfProducer"));
    ugmtEMTFShowerToken_ =
        consumes<l1t::RegionalMuonShowerBxCollection>(ps.getParameter<edm::InputTag>("emtfShowerProducer"));
  }
}

L1TStage2uGMTInputBxDistributions::~L1TStage2uGMTInputBxDistributions() {}

void L1TStage2uGMTInputBxDistributions::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("muonProducer")->setComment("uGMT output muons.");

  desc.add<edm::InputTag>("bmtfProducer")->setComment("RegionalMuonCands from BMTF.");
  desc.add<edm::InputTag>("omtfProducer")->setComment("RegionalMuonCands from OMTF.");
  desc.add<edm::InputTag>("emtfProducer")->setComment("RegionalMuonCands from EMTF.");
  desc.add<edm::InputTag>("muonShowerProducer")->setComment("uGMT output showers.");
  desc.add<edm::InputTag>("emtfShowerProducer")->setComment("RegionalMuonShowers from EMTF.");
  desc.addUntracked<std::string>("monitorDir", "")
      ->setComment("Target directory in the DQM file. Will be created if not existing.");
  desc.addUntracked<bool>("emulator", false)
      ->setComment("Create histograms for muonProducer input only. xmtfProducer inputs are ignored.");
  desc.addUntracked<bool>("verbose", false);
  desc.addUntracked<bool>("hadronicShowers", false);
  descriptions.add("l1tStage2uGMTInputBxDistributions", desc);
}

void L1TStage2uGMTInputBxDistributions::bookHistograms(DQMStore::IBooker& ibooker,
                                                       const edm::Run&,
                                                       const edm::EventSetup&) {
  if (!emul_) {
    // BMTF Input
    ibooker.setCurrentFolder(monitorDir_ + "/BMTFInput");

    ugmtBMTFBX = ibooker.book1D("ugmtBMTFBX", "uGMT BMTF Input BX", 7, -3.5, 3.5);
    ugmtBMTFBX->setAxisTitle("BX", 1);

    // OMTF Input
    ibooker.setCurrentFolder(monitorDir_ + "/OMTFInput");

    ugmtOMTFBX = ibooker.book1D("ugmtOMTFBX", "uGMT OMTF Input BX", 7, -3.5, 3.5);
    ugmtOMTFBX->setAxisTitle("BX", 1);

    // EMTF Input
    ibooker.setCurrentFolder(monitorDir_ + "/EMTFInput");

    ugmtEMTFBX = ibooker.book1D("ugmtEMTFBX", "uGMT EMTF Input BX", 7, -3.5, 3.5);
    ugmtEMTFBX->setAxisTitle("BX", 1);

    // EMTF muon showers
    if (hadronicShowers_) {
      ibooker.setCurrentFolder(monitorDir_ + "/EMTFInput/Muon showers");

      ugmtEMTFShowerTypeOccupancyPerBx =
          ibooker.book2D("ugmtEMTFShowerTypeOccupancyPerBx", "Shower type occupancy per BX", 7, -3.5, 3.5, 3, 1, 4);
      ugmtEMTFShowerTypeOccupancyPerBx->setAxisTitle("BX", 1);
      ugmtEMTFShowerTypeOccupancyPerBx->setAxisTitle("Shower type", 2);
      ugmtEMTFShowerTypeOccupancyPerBx->setBinLabel(IDX_LOOSE_SHOWER, "Loose", 2);
      ugmtEMTFShowerTypeOccupancyPerBx->setBinLabel(IDX_TIGHT_SHOWER, "Tight", 2);
      ugmtEMTFShowerTypeOccupancyPerBx->setBinLabel(IDX_NOMINAL_SHOWER, "Nominal", 2);

      ugmtEMTFShowerSectorOccupancyPerBx = ibooker.book2D(
          "ugmtEMTFShowerSectorOccupancyPerBx", "Shower BX occupancy per sector", 7, -3.5, 3.5, 12, 1, 13);
      ugmtEMTFShowerSectorOccupancyPerBx->setAxisTitle("BX", 1);
      ugmtEMTFShowerSectorOccupancyPerBx->setAxisTitle("Processor", 2);
      ugmtEMTFShowerSectorOccupancyPerBx->setBinLabel(12, "+6", 2);
      ugmtEMTFShowerSectorOccupancyPerBx->setBinLabel(11, "+5", 2);
      ugmtEMTFShowerSectorOccupancyPerBx->setBinLabel(10, "+4", 2);
      ugmtEMTFShowerSectorOccupancyPerBx->setBinLabel(9, "+3", 2);
      ugmtEMTFShowerSectorOccupancyPerBx->setBinLabel(8, "+2", 2);
      ugmtEMTFShowerSectorOccupancyPerBx->setBinLabel(7, "+1", 2);
      ugmtEMTFShowerSectorOccupancyPerBx->setBinLabel(6, "-6", 2);
      ugmtEMTFShowerSectorOccupancyPerBx->setBinLabel(5, "-5", 2);
      ugmtEMTFShowerSectorOccupancyPerBx->setBinLabel(4, "-4", 2);
      ugmtEMTFShowerSectorOccupancyPerBx->setBinLabel(3, "-3", 2);
      ugmtEMTFShowerSectorOccupancyPerBx->setBinLabel(2, "-2", 2);
      ugmtEMTFShowerSectorOccupancyPerBx->setBinLabel(1, "-1", 2);
    }
  }

  // Subsystem Monitoring and Muon Output
  ibooker.setCurrentFolder(monitorDir_);

  if (!emul_) {
    ugmtBMTFBXvsProcessor =
        ibooker.book2D("ugmtBXvsProcessorBMTF", "uGMT BMTF Input BX vs Processor", 12, -0.5, 11.5, 5, -2.5, 2.5);
    ugmtBMTFBXvsProcessor->setAxisTitle("Wedge", 1);
    for (int bin = 1; bin <= 12; ++bin) {
      ugmtBMTFBXvsProcessor->setBinLabel(bin, std::to_string(bin), 1);
    }
    ugmtBMTFBXvsProcessor->setAxisTitle("BX", 2);

    ugmtOMTFBXvsProcessor =
        ibooker.book2D("ugmtBXvsProcessorOMTF", "uGMT OMTF Input BX vs Processor", 12, -0.5, 11.5, 5, -2.5, 2.5);
    ugmtOMTFBXvsProcessor->setAxisTitle("Sector (Detector Side)", 1);
    for (int bin = 1; bin <= 6; ++bin) {
      ugmtOMTFBXvsProcessor->setBinLabel(bin, std::to_string(7 - bin) + " (-)", 1);
      ugmtOMTFBXvsProcessor->setBinLabel(bin + 6, std::to_string(bin) + " (+)", 1);
    }
    ugmtOMTFBXvsProcessor->setAxisTitle("BX", 2);

    ugmtEMTFBXvsProcessor =
        ibooker.book2D("ugmtBXvsProcessorEMTF", "uGMT EMTF Input BX vs Processor", 12, -0.5, 11.5, 5, -2.5, 2.5);
    ugmtEMTFBXvsProcessor->setAxisTitle("Sector (Detector Side)", 1);
    for (int bin = 1; bin <= 6; ++bin) {
      ugmtEMTFBXvsProcessor->setBinLabel(bin, std::to_string(7 - bin) + " (-)", 1);
      ugmtEMTFBXvsProcessor->setBinLabel(bin + 6, std::to_string(bin) + " (+)", 1);
    }
    ugmtEMTFBXvsProcessor->setAxisTitle("BX", 2);

    ugmtBXvsLink = ibooker.book2D("ugmtBXvsLink", "uGMT BX vs Input Links", 36, 35.5, 71.5, 5, -2.5, 2.5);
    ugmtBXvsLink->setAxisTitle("Link", 1);
    for (int bin = 1; bin <= 6; ++bin) {
      ugmtBXvsLink->setBinLabel(bin, Form("E+%d", bin), 1);
      ugmtBXvsLink->setBinLabel(bin + 6, Form("O+%d", bin), 1);
      ugmtBXvsLink->setBinLabel(bin + 12, Form("B%d", bin), 1);
      ugmtBXvsLink->setBinLabel(bin + 18, Form("B%d", bin + 6), 1);
      ugmtBXvsLink->setBinLabel(bin + 24, Form("O-%d", bin), 1);
      ugmtBXvsLink->setBinLabel(bin + 30, Form("E-%d", bin), 1);
    }
    ugmtBXvsLink->setAxisTitle("BX", 2);
  }
}

void L1TStage2uGMTInputBxDistributions::analyze(const edm::Event& e, const edm::EventSetup& c) {
  if (verbose_)
    edm::LogInfo("L1TStage2uGMTInputBxDistributions") << "L1TStage2uGMTInputBxDistributions: analyze..." << std::endl;

  if (!emul_) {
    edm::Handle<l1t::RegionalMuonCandBxCollection> BMTFBxCollection;
    e.getByToken(ugmtBMTFToken_, BMTFBxCollection);

    for (int itBX = BMTFBxCollection->getFirstBX(); itBX <= BMTFBxCollection->getLastBX(); ++itBX) {
      for (l1t::RegionalMuonCandBxCollection::const_iterator BMTF = BMTFBxCollection->begin(itBX);
           BMTF != BMTFBxCollection->end(itBX);
           ++BMTF) {
        ugmtBMTFBX->Fill(itBX);

        ugmtBMTFBXvsProcessor->Fill(BMTF->processor(), itBX);
        ugmtBXvsLink->Fill(BMTF->link(), itBX);
      }
    }

    edm::Handle<l1t::RegionalMuonCandBxCollection> OMTFBxCollection;
    e.getByToken(ugmtOMTFToken_, OMTFBxCollection);

    for (int itBX = OMTFBxCollection->getFirstBX(); itBX <= OMTFBxCollection->getLastBX(); ++itBX) {
      for (l1t::RegionalMuonCandBxCollection::const_iterator OMTF = OMTFBxCollection->begin(itBX);
           OMTF != OMTFBxCollection->end(itBX);
           ++OMTF) {
        ugmtOMTFBX->Fill(itBX);

        l1t::tftype trackFinderType = OMTF->trackFinderType();

        if (trackFinderType == l1t::omtf_neg) {
          ugmtOMTFBXvsProcessor->Fill(5 - OMTF->processor(), itBX);
        } else {
          ugmtOMTFBXvsProcessor->Fill(OMTF->processor() + 6, itBX);
        }

        ugmtBXvsLink->Fill(OMTF->link(), itBX);
      }
    }

    edm::Handle<l1t::RegionalMuonCandBxCollection> EMTFBxCollection;
    e.getByToken(ugmtEMTFToken_, EMTFBxCollection);

    for (int itBX = EMTFBxCollection->getFirstBX(); itBX <= EMTFBxCollection->getLastBX(); ++itBX) {
      for (l1t::RegionalMuonCandBxCollection::const_iterator EMTF = EMTFBxCollection->begin(itBX);
           EMTF != EMTFBxCollection->end(itBX);
           ++EMTF) {
        ugmtEMTFBX->Fill(itBX);

        l1t::tftype trackFinderType = EMTF->trackFinderType();

        if (trackFinderType == l1t::emtf_neg) {
          ugmtEMTFBXvsProcessor->Fill(5 - EMTF->processor(), itBX);
        } else {
          ugmtEMTFBXvsProcessor->Fill(EMTF->processor() + 6, itBX);
        }

        ugmtBXvsLink->Fill(EMTF->link(), itBX);
      }
    }

    // Fill shower plots
    if (hadronicShowers_) {
      edm::Handle<l1t::RegionalMuonShowerBxCollection> EMTFShowersBxCollection;
      e.getByToken(ugmtEMTFShowerToken_, EMTFShowersBxCollection);

      for (int itBX = EMTFShowersBxCollection->getFirstBX(); itBX <= EMTFShowersBxCollection->getLastBX(); ++itBX) {
        for (l1t::RegionalMuonShowerBxCollection::const_iterator shower = EMTFShowersBxCollection->begin(itBX);
             shower != EMTFShowersBxCollection->end(itBX);
             ++shower) {
          if (not shower->isValid()) {
            continue;
          }
          if (shower->isOneNominalInTime()) {
            ugmtEMTFShowerSectorOccupancyPerBx->Fill(
                itBX, shower->processor() + 1 + (shower->trackFinderType() == l1t::tftype::emtf_pos ? 6 : 0));
            ugmtEMTFShowerTypeOccupancyPerBx->Fill(itBX, IDX_NOMINAL_SHOWER);
          }
          if (shower->isOneTightInTime()) {
            ugmtEMTFShowerSectorOccupancyPerBx->Fill(
                itBX, shower->processor() + 1 + (shower->trackFinderType() == l1t::tftype::emtf_pos ? 6 : 0));
            ugmtEMTFShowerTypeOccupancyPerBx->Fill(itBX, IDX_TIGHT_SHOWER);
          }
          if (shower->isOneLooseInTime()) {
            ugmtEMTFShowerSectorOccupancyPerBx->Fill(
                itBX, shower->processor() + 1 + (shower->trackFinderType() == l1t::tftype::emtf_pos ? 6 : 0));
            ugmtEMTFShowerTypeOccupancyPerBx->Fill(itBX, IDX_LOOSE_SHOWER);
          }
        }
      }
    }
  }
}
