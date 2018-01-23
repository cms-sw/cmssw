#include <string>

#include "DQM/L1TMonitor/interface/L1TdeStage2EMTF.h"


L1TdeStage2EMTF::L1TdeStage2EMTF(const edm::ParameterSet& ps)
    : dataToken(consumes<l1t::RegionalMuonCandBxCollection>(ps.getParameter<edm::InputTag>("dataSource"))),
      emulToken(consumes<l1t::RegionalMuonCandBxCollection>(ps.getParameter<edm::InputTag>("emulSource"))),
      monitorDir(ps.getUntrackedParameter<std::string>("monitorDir", "")),
      verbose(ps.getUntrackedParameter<bool>("verbose", false)) {}

L1TdeStage2EMTF::~L1TdeStage2EMTF() {}

void L1TdeStage2EMTF::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c, emtfdedqm::Histograms& histograms) const {}

void L1TdeStage2EMTF::bookHistograms(DQMStore::ConcurrentBooker& booker, const edm::Run&, const edm::EventSetup&, emtfdedqm::Histograms& histograms) const {

  booker.setCurrentFolder(monitorDir);

  histograms.emtfComparenMuonsEvent = booker.book2D("emtfComparenMuonsEvent", "Number of EMTF Muon Cands per Event", 12, 0, 12, 12, 0, 12);
  for (int axis = 1; axis <= 2; ++axis) {
    std::string axisTitle = (axis == 1) ? "Data" : "Emulator";
    histograms.emtfComparenMuonsEvent.setAxisTitle(axisTitle, axis);
    for (int bin = 1; bin <= 12; ++bin) {
      std::string binLabel = (bin == 12) ? "Overflow" : std::to_string(bin - 1);
      histograms.emtfComparenMuonsEvent.setBinLabel(bin, binLabel, axis);
    }
  }

  histograms.emtfDataBX = booker.book1D("emtfDataBX", "EMTF Muon Cand BX", 7, -3, 4);
  histograms.emtfDataBX.setAxisTitle("BX", 1);

  histograms.emtfEmulBX = booker.book1D("emtfEmulBX", "EMTF Emulated Muon Cand BX", 7, -3, 4);
  histograms.emtfEmulBX.setAxisTitle("BX", 1);

  for (int bin = 1, bin_label = -3; bin <= 7; ++bin, ++bin_label) {
    histograms.emtfDataBX.setBinLabel(bin, std::to_string(bin_label), 1);
    histograms.emtfEmulBX.setBinLabel(bin, std::to_string(bin_label), 1);
  }

  histograms.emtfDatahwPt = booker.book1D("emtfDatahwPt", "EMTF Muon Cand p_{T}", 512, 0, 512);
  histograms.emtfDatahwPt.setAxisTitle("Hardware p_{T}", 1);

  histograms.emtfEmulhwPt = booker.book1D("emtfEmulhwPt", "EMTF Emulated Muon Cand p_{T}", 512, 0, 512);
  histograms.emtfEmulhwPt.setAxisTitle("Hardware p_{T}", 1);

  histograms.emtfDatahwEta = booker.book1D("emtfDatahwEta", "EMTF Muon Cand #eta", 460, -230, 230);
  histograms.emtfDatahwEta.setAxisTitle("Hardware #eta", 1);

  histograms.emtfEmulhwEta = booker.book1D("emtfEmulhwEta", "EMTF Emulated Muon Cand #eta", 460, -230, 230);
  histograms.emtfEmulhwEta.setAxisTitle("Hardware #eta", 1);

  histograms.emtfDatahwPhi = booker.book1D("emtfDatahwPhi", "EMTF Muon Cand #phi", 125, -20, 105);
  histograms.emtfDatahwPhi.setAxisTitle("Hardware #phi", 1);

  histograms.emtfEmulhwPhi = booker.book1D("emtfEmulhwPhi", "EMTF Emulated Muon Cand #phi", 125, -20, 105);
  histograms.emtfEmulhwPhi.setAxisTitle("Hardware #phi", 1);

  histograms.emtfDatahwQual = booker.book1D("emtfDatahwQual", "EMTF Muon Cand Quality", 16, 0, 16);
  histograms.emtfDatahwQual.setAxisTitle("Quality", 1);

  histograms.emtfEmulhwQual = booker.book1D("emtfEmulhwQual", "EMTF Emulated Muon Cand Quality", 16, 0, 16);
  histograms.emtfEmulhwQual.setAxisTitle("Quality", 1);

  for (int bin = 1; bin <= 16; ++bin) {
    histograms.emtfDatahwQual.setBinLabel(bin, std::to_string(bin - 1), 1);
    histograms.emtfEmulhwQual.setBinLabel(bin, std::to_string(bin - 1), 1);
  }

  // Comparison plots reserved for updated emulator.
  /*histograms.emtfComparehwPt = booker.book2D("emtfComparehwPt", "EMTF Muon Cand p_{T}", 512, 0, 512, 512, 0, 512);
  histograms.emtfComparehwPt.setAxisTitle("Hardware p_{T}", 1);
  histograms.emtfComparehwPt.setAxisTitle("Emulator Hardware p_{T}", 2);

  histograms.emtfComparehwEta = booker.book2D("emtfComparehwEta", "EMTF Muon Cand #eta", 460, -230, 230, 460, -230, 230);
  histograms.emtfComparehwEta.setAxisTitle("Hardware #eta", 1);
  histograms.emtfComparehwEta.setAxisTitle("Emulator Hardware #eta", 2);

  histograms.emtfComparehwPhi = booker.book2D("emtfComparehwPhi", "EMTF Muon Cand #phi", 125, -20, 105, 125, -20, 105);
  histograms.emtfComparehwPhi.setAxisTitle("Hardware #phi", 1);
  histograms.emtfComparehwPhi.setAxisTitle("Emulator Hardware #phi", 2);

  histograms.emtfComparehwQual = booker.book2D("emtfComparehwQual", "EMTF Muon Cand Quality", 16, 0, 16, 16, 0, 16);
  for (int axis = 1; axis <= 2; ++axis) {
    std::string axisTitle = (axis == 1) ? "Quality" : "Emulator Quality";
    histograms.emtfComparehwQual.setAxisTitle(axisTitle, axis);
    for (int bin = 1; bin <= 16; ++bin) {
      histograms.emtfComparehwQual.setBinLabel(bin, std::to_string(bin - 1), axis);
    }
  }*/
}

void L1TdeStage2EMTF::dqmAnalyze(const edm::Event& e, const edm::EventSetup& c, const emtfdedqm::Histograms& histograms) const {

  if (verbose) edm::LogInfo("L1TdeStage2EMTF") << "L1TdeStage2EMTF: analyze..." << std::endl;

  edm::Handle<l1t::RegionalMuonCandBxCollection> dataMuons;
  e.getByToken(dataToken, dataMuons);

  edm::Handle<l1t::RegionalMuonCandBxCollection> emulMuons;
  e.getByToken(emulToken, emulMuons);

  histograms.emtfComparenMuonsEvent.fill(dataMuons->size(), emulMuons->size());

  for (int itBX = dataMuons->getFirstBX(); itBX <= dataMuons->getLastBX(); ++itBX) {
    for (l1t::RegionalMuonCandBxCollection::const_iterator dataMuon = dataMuons->begin(itBX); dataMuon != dataMuons->end(itBX); ++dataMuon) {
      histograms.emtfDataBX.fill(itBX);
      histograms.emtfDatahwPt.fill(dataMuon->hwPt());
      histograms.emtfDatahwEta.fill(dataMuon->hwEta());
      histograms.emtfDatahwPhi.fill(dataMuon->hwPhi());
      histograms.emtfDatahwQual.fill(dataMuon->hwQual());
    }
  }

  for (int itBX = emulMuons->getFirstBX(); itBX <= emulMuons->getLastBX(); ++itBX) {
    for (l1t::RegionalMuonCandBxCollection::const_iterator emulMuon = emulMuons->begin(itBX); emulMuon != emulMuons->end(itBX); ++emulMuon) {
      histograms.emtfEmulBX.fill(itBX);
      histograms.emtfEmulhwPt.fill(emulMuon->hwPt());
      histograms.emtfEmulhwEta.fill(emulMuon->hwEta());
      histograms.emtfEmulhwPhi.fill(emulMuon->hwPhi());
      histograms.emtfEmulhwQual.fill(emulMuon->hwQual());
    }
  }
}

