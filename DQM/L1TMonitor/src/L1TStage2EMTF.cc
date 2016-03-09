#include "DQM/L1TMonitor/interface/L1TStage2EMTF.h"


L1TStage2EMTF::L1TStage2EMTF(const edm::ParameterSet& ps) 
    : emtfToken(consumes<l1t::EMTFOutputCollection>(ps.getParameter<edm::InputTag>("emtfSource"))),
      monitorDir(ps.getUntrackedParameter<std::string>("monitorDir", "")),
      verbose(ps.getUntrackedParameter<bool>("verbose", false)) {}

L1TStage2EMTF::~L1TStage2EMTF() {}

void L1TStage2EMTF::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c) {}

void L1TStage2EMTF::beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) {}

void L1TStage2EMTF::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup&) {

  ibooker.setCurrentFolder(monitorDir);

  emtfnTracks = ibooker.book1D("emtfnTracks", "Number of EMTF Tracks per Event", 4, 0, 4);
  
  emtfTrackPt = ibooker.book1D("emtfTrackPt", "EMTF Track p_{T}", 512, 0, 512);
  emtfTrackPt->setAxisTitle("Track p_{T} [GeV]", 1);

  emtfTrackEta = ibooker.book1D("emtfTrackEta", "EMTF Track #eta", 460, -230, 230);
  emtfTrackEta->setAxisTitle("Track #eta", 1);

  emtfTrackPhi = ibooker.book1D("emtfTrackPhi", "EMTF Track #phi", 116, -16, 100);
  emtfTrackPhi->setAxisTitle("Track #phi", 1);

  emtfTrackPhiFull = ibooker.book1D("emtfTrackPhiFull", "EMTF Full Precision Track #phi", 4096, 0, 4096);
  emtfTrackPhiFull->setAxisTitle("Full Precision Track #phi", 1);

  emtfTrackBX = ibooker.book1D("emtfTrackBX", "EMTF Track Bunch Crossings", 4, 0, 4);
  emtfTrackBX->setAxisTitle("Track BX", 1);
}

void L1TStage2EMTF::analyze(const edm::Event& e, const edm::EventSetup& c) {

  if (verbose) edm::LogInfo("L1TStage2EMTF") << "L1TStage2EMTF: analyze..." << std::endl;

  int nTracks = 0;

  edm::Handle<l1t::EMTFOutputCollection> EMTF;
  e.getByToken(emtfToken, EMTF);
 
  for (std::vector<l1t::EMTFOutput>::const_iterator itEMTF = EMTF->begin(); itEMTF != EMTF->end(); ++itEMTF) {

    l1t::emtf::SPCollection SP = itEMTF->GetSPCollection();

    for (std::vector<l1t::emtf::SP>::const_iterator itSP = SP.begin(); itSP != SP.end(); ++itSP) {
      emtfTrackBX->Fill(itSP->BX());
      emtfTrackPt->Fill(itSP->Pt());
      emtfTrackEta->Fill(itSP->Eta_GMT());
      emtfTrackPhi->Fill(itSP->Phi_GMT());
      emtfTrackPhiFull->Fill(itSP->Phi_full());
      nTracks++;
    }
  }
  emtfnTracks->Fill(nTracks);
}

