#include <string>

#include "DQM/L1TMonitor/interface/L1TdeStage2EMTF.h"

L1TdeStage2EMTF::L1TdeStage2EMTF(const edm::ParameterSet& ps)
    : dataToken(consumes<l1t::RegionalMuonCandBxCollection>(ps.getParameter<edm::InputTag>("dataSource"))),
      emulToken(consumes<l1t::RegionalMuonCandBxCollection>(ps.getParameter<edm::InputTag>("emulSource"))),
      monitorDir(ps.getUntrackedParameter<std::string>("monitorDir", "")),
      verbose(ps.getUntrackedParameter<bool>("verbose", false)) {}

L1TdeStage2EMTF::~L1TdeStage2EMTF() {}

void L1TdeStage2EMTF::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup&) {
  ibooker.setCurrentFolder(monitorDir);

  emtfComparenMuonsEvent =
      ibooker.book2D("emtfComparenMuonsEvent", "Number of EMTF Muon Cands per Event", 12, 0, 12, 12, 0, 12);
  for (int axis = 1; axis <= 2; ++axis) {
    std::string axisTitle = (axis == 1) ? "Data" : "Emulator";
    emtfComparenMuonsEvent->setAxisTitle(axisTitle, axis);
    for (int bin = 1; bin <= 12; ++bin) {
      std::string binLabel = (bin == 12) ? "Overflow" : std::to_string(bin - 1);
      emtfComparenMuonsEvent->setBinLabel(bin, binLabel, axis);
    }
  }

  emtfDataBX = ibooker.book1D("emtfDataBX", "EMTF Muon Cand BX", 7, -3, 4);
  emtfDataBX->setAxisTitle("BX", 1);

  emtfEmulBX = ibooker.book1D("emtfEmulBX", "EMTF Emulated Muon Cand BX", 7, -3, 4);
  emtfEmulBX->setAxisTitle("BX", 1);

  for (int bin = 1, bin_label = -3; bin <= 7; ++bin, ++bin_label) {
    emtfDataBX->setBinLabel(bin, std::to_string(bin_label), 1);
    emtfEmulBX->setBinLabel(bin, std::to_string(bin_label), 1);
  }

  emtfDatahwPt = ibooker.book1D("emtfDatahwPt", "EMTF Muon Cand p_{T}", 512, 0, 512);
  emtfDatahwPt->setAxisTitle("Hardware p_{T}", 1);

  emtfEmulhwPt = ibooker.book1D("emtfEmulhwPt", "EMTF Emulated Muon Cand p_{T}", 512, 0, 512);
  emtfEmulhwPt->setAxisTitle("Hardware p_{T}", 1);

  emtfDatahwEta = ibooker.book1D("emtfDatahwEta", "EMTF Muon Cand #eta", 460, -230, 230);
  emtfDatahwEta->setAxisTitle("Hardware #eta", 1);

  emtfEmulhwEta = ibooker.book1D("emtfEmulhwEta", "EMTF Emulated Muon Cand #eta", 460, -230, 230);
  emtfEmulhwEta->setAxisTitle("Hardware #eta", 1);

  emtfDatahwPhi = ibooker.book1D("emtfDatahwPhi", "EMTF Muon Cand #phi", 125, -20, 105);
  emtfDatahwPhi->setAxisTitle("Hardware #phi", 1);

  emtfEmulhwPhi = ibooker.book1D("emtfEmulhwPhi", "EMTF Emulated Muon Cand #phi", 125, -20, 105);
  emtfEmulhwPhi->setAxisTitle("Hardware #phi", 1);

  emtfDatahwQual = ibooker.book1D("emtfDatahwQual", "EMTF Muon Cand Quality", 16, 0, 16);
  emtfDatahwQual->setAxisTitle("Quality", 1);

  emtfEmulhwQual = ibooker.book1D("emtfEmulhwQual", "EMTF Emulated Muon Cand Quality", 16, 0, 16);
  emtfEmulhwQual->setAxisTitle("Quality", 1);

  for (int bin = 1; bin <= 16; ++bin) {
    emtfDatahwQual->setBinLabel(bin, std::to_string(bin - 1), 1);
    emtfEmulhwQual->setBinLabel(bin, std::to_string(bin - 1), 1);
  }

  // Comparison plots reserved for updated emulator.
  /*emtfComparehwPt = ibooker.book2D("emtfComparehwPt", "EMTF Muon Cand p_{T}", 512, 0, 512, 512, 0, 512);
  emtfComparehwPt->setAxisTitle("Hardware p_{T}", 1);
  emtfComparehwPt->setAxisTitle("Emulator Hardware p_{T}", 2);

  emtfComparehwEta = ibooker.book2D("emtfComparehwEta", "EMTF Muon Cand #eta", 460, -230, 230, 460, -230, 230);
  emtfComparehwEta->setAxisTitle("Hardware #eta", 1);
  emtfComparehwEta->setAxisTitle("Emulator Hardware #eta", 2);

  emtfComparehwPhi = ibooker.book2D("emtfComparehwPhi", "EMTF Muon Cand #phi", 125, -20, 105, 125, -20, 105);
  emtfComparehwPhi->setAxisTitle("Hardware #phi", 1);
  emtfComparehwPhi->setAxisTitle("Emulator Hardware #phi", 2);

  emtfComparehwQual = ibooker.book2D("emtfComparehwQual", "EMTF Muon Cand Quality", 16, 0, 16, 16, 0, 16);
  for (int axis = 1; axis <= 2; ++axis) {
    std::string axisTitle = (axis == 1) ? "Quality" : "Emulator Quality";
    emtfComparehwQual->setAxisTitle(axisTitle, axis);
    for (int bin = 1; bin <= 16; ++bin) {
      emtfComparehwQual->setBinLabel(bin, std::to_string(bin - 1), axis);
    }
  }*/
}

void L1TdeStage2EMTF::analyze(const edm::Event& e, const edm::EventSetup& c) {
  if (verbose)
    edm::LogInfo("L1TdeStage2EMTF") << "L1TdeStage2EMTF: analyze..." << std::endl;

  edm::Handle<l1t::RegionalMuonCandBxCollection> dataMuons;
  e.getByToken(dataToken, dataMuons);

  edm::Handle<l1t::RegionalMuonCandBxCollection> emulMuons;
  e.getByToken(emulToken, emulMuons);

  emtfComparenMuonsEvent->Fill(dataMuons->size(), emulMuons->size());

  for (int itBX = dataMuons->getFirstBX(); itBX <= dataMuons->getLastBX(); ++itBX) {
    for (l1t::RegionalMuonCandBxCollection::const_iterator dataMuon = dataMuons->begin(itBX);
         dataMuon != dataMuons->end(itBX);
         ++dataMuon) {
      emtfDataBX->Fill(itBX);
      emtfDatahwPt->Fill(dataMuon->hwPt());
      emtfDatahwEta->Fill(dataMuon->hwEta());
      emtfDatahwPhi->Fill(dataMuon->hwPhi());
      emtfDatahwQual->Fill(dataMuon->hwQual());
    }
  }

  for (int itBX = emulMuons->getFirstBX(); itBX <= emulMuons->getLastBX(); ++itBX) {
    for (l1t::RegionalMuonCandBxCollection::const_iterator emulMuon = emulMuons->begin(itBX);
         emulMuon != emulMuons->end(itBX);
         ++emulMuon) {
      emtfEmulBX->Fill(itBX);
      emtfEmulhwPt->Fill(emulMuon->hwPt());
      emtfEmulhwEta->Fill(emulMuon->hwEta());
      emtfEmulhwPhi->Fill(emulMuon->hwPhi());
      emtfEmulhwQual->Fill(emulMuon->hwQual());
    }
  }
}
