#include "CSCTFanalyzer.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/TrackStub.h"
#include <DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h>
#include <DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h>

CSCTFanalyzer::CSCTFanalyzer(edm::ParameterSet const& pset) {
  verbose = pset.getUntrackedParameter<unsigned int>("verbose", 0);
  dataTrackProducer = pset.getUntrackedParameter<edm::InputTag>("dataTrackProducer", edm::InputTag("csctfDigis"));
  emulTrackProducer = pset.getUntrackedParameter<edm::InputTag>("emulTrackProducer", edm::InputTag("csctfTrackDigis"));
  lctProducer = pset.getUntrackedParameter<edm::InputTag>("lctProducer", edm::InputTag("csctfDigis"));
  mbProducer = pset.getUntrackedParameter<edm::InputTag>("mbProducer", edm::InputTag("csctfDigis"));
  scalesToken = esConsumes<L1MuTriggerScales, L1MuTriggerScalesRcd>();
  file = new TFile("qwe.root", "RECREATE");
  tree = new TTree("dy", "QWE");
  tree->Branch("nDataMuons", &nDataMuons, "nDataMuons/I");
  tree->Branch("dphi1", &dphi1, "dphi1/D");
  tree->Branch("dphi2", &dphi2, "dphi2/D");
  tree->Branch("dphi3", &dphi3, "dphi3/D");
  tree->Branch("deta1", &deta1, "deta1/D");
  tree->Branch("deta2", &deta2, "deta2/D");
  tree->Branch("deta3", &deta3, "deta3/D");
  tree->Branch("dpt1", &dpt1, "dpt1/I");
  tree->Branch("dpt2", &dpt2, "dpt2/I");
  tree->Branch("dpt3", &dpt3, "dpt3/I");
  tree->Branch("dch1", &dch1, "dch1/I");
  tree->Branch("dch2", &dch2, "dch2/I");
  tree->Branch("dch3", &dch3, "dch3/I");
  tree->Branch("dbx1", &dbx1, "dbx1/I");
  tree->Branch("dbx2", &dbx2, "dbx2/I");
  tree->Branch("dbx3", &dbx3, "dbx3/I");
  tree->Branch("drank1", &drank1, "drank1/I");
  tree->Branch("drank2", &drank2, "drank2/I");
  tree->Branch("drank3", &drank3, "drank3/I");
  tree->Branch("dmode1", &dmode1, "dmode1/I");
  tree->Branch("dmode2", &dmode2, "dmode2/I");
  tree->Branch("dmode3", &dmode3, "dmode3/I");
  tree->Branch("dlcts1", &dlcts1, "dlcts1/I");
  tree->Branch("dlcts2", &dlcts2, "dlcts2/I");
  tree->Branch("dlcts3", &dlcts3, "dlcts3/I");

  tree->Branch("nEmulMuons", &nEmulMuons, "nEmulMuons/I");
  tree->Branch("ephi1", &ephi1, "ephi1/D");
  tree->Branch("ephi2", &ephi2, "ephi2/D");
  tree->Branch("ephi3", &ephi3, "ephi3/D");
  tree->Branch("eeta1", &eeta1, "eeta1/D");
  tree->Branch("eeta2", &eeta2, "eeta2/D");
  tree->Branch("eeta3", &eeta3, "eeta3/D");
  tree->Branch("ept1", &ept1, "ept1/I");
  tree->Branch("ept2", &ept2, "ept2/I");
  tree->Branch("ept3", &ept3, "ept3/I");
  tree->Branch("ech1", &ech1, "ech1/I");
  tree->Branch("ech2", &ech2, "ech2/I");
  tree->Branch("ech3", &ech3, "ech3/I");
  tree->Branch("ebx1", &ebx1, "ebx1/I");
  tree->Branch("ebx2", &ebx2, "ebx2/I");
  tree->Branch("ebx3", &ebx3, "ebx3/I");
  tree->Branch("erank1", &erank1, "erank1/I");
  tree->Branch("erank2", &erank2, "erank2/I");
  tree->Branch("erank3", &erank3, "erank3/I");
  tree->Branch("emode1", &emode1, "emode1/I");
  tree->Branch("emode2", &emode2, "emode2/I");
  tree->Branch("emode3", &emode3, "emode3/I");

  ts = nullptr;
}

void CSCTFanalyzer::endJob(void) {
  tree->Write();
  file->Write();
  file->Close();
}

void CSCTFanalyzer::analyze(edm::Event const& e, edm::EventSetup const& es) {
  if (!ts) {
    edm::ESHandle<L1MuTriggerScales> scales = es.getHandle(scalesToken);
    ts = scales.product();
  }

  if (lctProducer.label() != "null") {
    edm::Handle<CSCCorrelatedLCTDigiCollection> LCTs;
    e.getByLabel(lctProducer.label(), lctProducer.instance(), LCTs);

    for (CSCCorrelatedLCTDigiCollection::DigiRangeIterator csc = LCTs.product()->begin(); csc != LCTs.product()->end();
         csc++) {
      int lctId = 0;

      CSCCorrelatedLCTDigiCollection::Range range1 = LCTs.product()->get((*csc).first);
      for (CSCCorrelatedLCTDigiCollection::const_iterator lct = range1.first; lct != range1.second; lct++, lctId++) {
        int station = (*csc).first.station() - 1;
        int cscId = (*csc).first.triggerCscId() - 1;
        int sector = (*csc).first.triggerSector() - 1 + ((*csc).first.endcap() == 1 ? 0 : 6);
        //int subSector = CSCTriggerNumbering::triggerSubSectorFromLabels((*csc).first);
        int tbin = lct->getBX();
        //int fpga    = ( subSector ? subSector-1 : station+1 );
        if ((verbose & 1) == 1)
          std::cout << "LCT in station=" << (station + 1) << " sector=" << (sector + 1) << " cscId=" << (cscId + 1)
                    << " bx=" << tbin << std::endl;
      }
    }
  }

  if (mbProducer.label() != "null") {
    edm::Handle<CSCTriggerContainer<csctf::TrackStub> > dtStubs;
    e.getByLabel(mbProducer.label(), mbProducer.instance(), dtStubs);
    if (dtStubs.isValid()) {
      std::vector<csctf::TrackStub> vstubs = dtStubs->get();
      for (std::vector<csctf::TrackStub>::const_iterator stub = vstubs.begin(); stub != vstubs.end(); stub++) {
        //int dtSector =(stub->sector()-1)*2 + stub->subsector()-1;
        //int dtEndcap = stub->endcap()-1;
        std::cout << "   DT data: tbin=" << stub->BX() << " (CSC) sector=" << stub->sector()
                  << " (CSC) subsector=" << stub->subsector() << " station=" << stub->station()
                  << " endcap=" << stub->endcap() << " phi=" << stub->phiPacked() << " phiBend=" << stub->getBend()
                  << " quality=" << stub->getQuality() << " id=" << stub->getMPCLink() << " mb_bxn=" << stub->cscid()
                  << std::endl;
      }

    } else
      edm::LogInfo("CSCTFAnalyzer") << "  No valid CSCTriggerContainer<csctf::TrackStub> products found";
  }

  nDataMuons = 0;
  nEmulMuons = 0;
  dphi1 = -1;
  deta1 = -1;
  dpt1 = -1;
  dch1 = -1, dbx1 = -10;
  dphi2 = -1;
  deta2 = -1;
  dpt2 = -1;
  dch2 = -1, dbx2 = -10;
  dphi3 = -1;
  deta3 = -1;
  dpt3 = -1;
  dch3 = -1, dbx3 = -10;
  drank1 = -1;
  drank2 = -1;
  drank3 = -1;
  dmode1 = -1;
  dmode2 = -1;
  dmode3 = -1;
  dlcts1 = 0;
  dlcts2 = 0;
  dlcts3 = 0;
  ephi1 = -1;
  eeta1 = -1;
  ept1 = -1;
  ech1 = -1, ebx1 = -10;
  ephi2 = -1;
  eeta2 = -1;
  ept2 = -1;
  ech2 = -1, ebx2 = -10;
  ephi3 = -1;
  eeta3 = -1;
  ept3 = -1;
  ech3 = -1, ebx3 = -10;
  erank1 = -1;
  erank2 = -1;
  erank3 = -1;
  emode1 = -1;
  emode2 = -1;
  emode3 = -1;

  if (dataTrackProducer.label() != "null") {
    edm::Handle<L1CSCTrackCollection> tracks;
    e.getByLabel(dataTrackProducer.label(), dataTrackProducer.instance(), tracks);
    // Muon sorter emulation:
    std::vector<csc::L1Track> result;
    CSCTriggerContainer<csc::L1Track> stripped_tracks;
    for (L1CSCTrackCollection::const_iterator tmp_trk = tracks->begin(); tmp_trk != tracks->end(); tmp_trk++) {
      csc::L1Track qqq(tmp_trk->first);
      qqq.setOutputLink(0);
      CSCCorrelatedLCTDigiCollection qwe = tmp_trk->second;
      for (CSCCorrelatedLCTDigiCollection::DigiRangeIterator csc = qwe.begin(); csc != qwe.end(); csc++) {
        CSCCorrelatedLCTDigiCollection::Range range1 = qwe.get((*csc).first);
        for (CSCCorrelatedLCTDigiCollection::const_iterator lct = range1.first; lct != range1.second; lct++)
          qqq.setOutputLink(qqq.outputLink() | (1 << (*csc).first.station()));
      }
      stripped_tracks.push_back(qqq);
    }
    // First we sort and crop the incoming tracks based on their rank.
    for (int bx = -3; bx <= 3; ++bx) {  // switch back into signed BX
      std::vector<csc::L1Track> tks = stripped_tracks.get(bx);
      std::sort(tks.begin(), tks.end(), std::greater<csc::L1Track>());
      if (tks.size() > 4)
        tks.resize(4);  // resize to max number of muons the MS can output
      for (std::vector<csc::L1Track>::iterator itr = tks.begin(); itr != tks.end(); itr++) {
        unsigned gbl_phi =
            itr->localPhi() + ((itr->sector() - 1) * 24) + 6;  // for now, convert using this.. LUT in the future
        if (gbl_phi > 143)
          gbl_phi -= 143;
        itr->setPhiPacked(gbl_phi & 0xff);
        unsigned eta_sign = (itr->endcap() == 1 ? 0 : 1);
        int gbl_eta = itr->eta_packed() | eta_sign << (L1MuRegionalCand::ETA_LENGTH - 1);
        itr->setEtaPacked(gbl_eta & 0x3f);
        itr->setQualityPacked((itr->rank() >> 4) & 0x3);
        itr->setPtPacked(itr->rank() & 0x1f);
        if (!itr->empty())
          result.push_back(*itr);
      }
    }
    //		for(std::vector<csc::L1Track>::const_iterator trk=result.begin(); trk!=result.end(); trk++){
    for (L1CSCTrackCollection::const_iterator _trk = tracks->begin(); _trk != tracks->end(); _trk++) {
      const csc::L1Track* trk = &(_trk->first);
      switch (nDataMuons) {
        case 0:
          dphi1 = ts->getPhiScale()->getLowEdge(trk->phi_packed());
          deta1 = ts->getRegionalEtaScale(2)->getCenter(trk->eta_packed());
          dpt1 = trk->pt_packed();
          dch1 = trk->charge_packed();
          dbx1 = trk->BX();
          dmode1 = trk->mode();
          drank1 = trk->rank();
          dlcts1 = trk->outputLink();
          break;
        case 1:
          dphi2 = ts->getPhiScale()->getLowEdge(trk->phi_packed());
          deta2 = ts->getRegionalEtaScale(2)->getCenter(trk->eta_packed());
          dpt2 = trk->pt_packed();
          dch2 = trk->charge_packed();
          dbx2 = trk->BX();
          dmode2 = trk->mode();
          drank2 = trk->rank();
          dlcts2 = trk->outputLink();
          break;
        case 2:
          dphi3 = ts->getPhiScale()->getLowEdge(trk->phi_packed());
          deta3 = ts->getRegionalEtaScale(2)->getCenter(trk->eta_packed());
          dpt3 = trk->pt_packed();
          dch3 = trk->charge_packed();
          dbx3 = trk->BX();
          dmode3 = trk->mode();
          drank3 = trk->rank();
          dlcts3 = trk->outputLink();
          break;
        default:
          break;
      }
      if ((verbose & 2) == 2)
        std::cout << "Data: TRK in endcap=" << trk->endcap() << " sector=" << trk->sector() << " bx=" << trk->BX()
                  << " (rank=" << trk->rank() << " localPhi=" << trk->localPhi() << " etaPacked=" << trk->eta_packed()
                  << " me1D=" << trk->me1ID() << " me2D=" << trk->me2ID() << " me3D=" << trk->me3ID()
                  << " me4D=" << trk->me4ID() << " mb1D=" << trk->mb1ID() << " pTaddr=" << std::hex
                  << trk->ptLUTAddress() << std::dec << ")" << std::endl;
      nDataMuons++;
    }
  }

  if (emulTrackProducer.label() != "null") {
    edm::Handle<L1CSCTrackCollection> tracks;
    e.getByLabel(emulTrackProducer.label(), emulTrackProducer.instance(), tracks);
    for (L1CSCTrackCollection::const_iterator trk = tracks.product()->begin(); trk != tracks.product()->end(); trk++) {
      switch (nEmulMuons) {
        case 0:
          ephi1 = trk->first.localPhi();
          eeta1 = trk->first.eta_packed();
          ept1 = trk->first.pt_packed();
          ech1 = trk->first.charge_packed();
          ebx1 = trk->first.BX();
          break;
        case 1:
          ephi2 = trk->first.localPhi();
          eeta2 = trk->first.eta_packed();
          ept2 = trk->first.pt_packed();
          ech2 = trk->first.charge_packed();
          ebx2 = trk->first.BX();
          break;
        case 2:
          ephi3 = trk->first.localPhi();
          eeta3 = trk->first.eta_packed();
          ept3 = trk->first.pt_packed();
          ech3 = trk->first.charge_packed();
          ebx3 = trk->first.BX();
          break;
        default:
          break;
      }
      if ((verbose & 2) == 2)
        std::cout << "Emulator: TRK in endcap=" << trk->first.endcap() << " sector=" << trk->first.sector()
                  << " bx=" << trk->first.BX() << " (rank=" << trk->first.rank()
                  << " localPhi=" << trk->first.localPhi() << " etaPacked=" << trk->first.eta_packed()
                  << " me1D=" << trk->first.me1ID() << " me2D=" << trk->first.me2ID() << " me3D=" << trk->first.me3ID()
                  << " me4D=" << trk->first.me4ID() << " mb1D=" << trk->first.mb1ID() << " pTaddr=" << std::hex
                  << trk->first.ptLUTAddress() << std::dec << ")" << std::endl;
      nEmulMuons++;
    }
  }

  tree->Fill();
}
