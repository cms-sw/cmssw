#include <string>
#include <vector>
#include "DQM/L1TMonitor/interface/L1TdeStage2EMTF.h"


L1TdeStage2EMTF::L1TdeStage2EMTF(const edm::ParameterSet& ps)
    : dataToken(consumes<l1t::RegionalMuonCandBxCollection>(ps.getParameter<edm::InputTag>("dataSource"))),
      emulToken(consumes<l1t::RegionalMuonCandBxCollection>(ps.getParameter<edm::InputTag>("emulSource"))),
      datahitToken(consumes<l1t::EMTFHitCollection>(ps.getParameter<edm::InputTag>("dataSource"))),
      emulhitToken(consumes<l1t::EMTFHitCollection>(ps.getParameter<edm::InputTag>("emulSource"))),
      datatrackToken(consumes<l1t::EMTFTrackCollection>(ps.getParameter<edm::InputTag>("dataSource"))),
      emultrackToken(consumes<l1t::EMTFTrackCollection>(ps.getParameter<edm::InputTag>("emulSource"))),
      monitorDir(ps.getUntrackedParameter<std::string>("monitorDir", "")),
      verbose(ps.getUntrackedParameter<bool>("verbose", false)) {}

L1TdeStage2EMTF::~L1TdeStage2EMTF() {}

void L1TdeStage2EMTF::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c) {}

void L1TdeStage2EMTF::beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) {}

void L1TdeStage2EMTF::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup&) {

  ibooker.setCurrentFolder(monitorDir);

  //RegionalMuonCand Output hw Plots
  emtfMuonMatchhwEta = ibooker.book1D("emtfMuonMatchhwEta", "EMTF Muon Match hw #eta", 920, -460, 460);
  emtfMuonMatchhwPhi= ibooker.book1D("emtfMuonMatchhwPhi", "EMTF Muon Match hw #phi", 250, -125, 125);
  emtfMuonMatchhwPt= ibooker.book1D("emtfMuonMatchhwPt", "EMTF Muon Match hw p_{T}", 1024, -512, 512);
  emtfMuonMatchhwQual= ibooker.book1D("emtfMuonMatchhwQual", "EMTF Muon Match hw Quality",32, -16, 16);

  emtfComparenMuonsEvent = ibooker.book2D("emtfComparenMuonsEvent", "Number of EMTF Muon Cands per Event", 12, 0, 12, 12, 0, 12);
  for (int axis = 1; axis <= 2; ++axis) {
    std::string axisTitle = (axis == 1) ? "Data" : "Emulator";
    emtfComparenMuonsEvent->setAxisTitle(axisTitle, axis);
    for (int bin = 1; bin <= 12; ++bin) {
      std::string binLabel = (bin == 12) ? "Overflow" : std::to_string(bin - 1);
      emtfComparenMuonsEvent->setBinLabel(bin, binLabel, axis);
    }
  }
 

  //Track Plots
  emtfTrackMatchEta = ibooker.book2D("emtfTrackMatchEta","EMTF Track Match #eta", 100, -2.5, 2.5, 100, -2.5, 2.5);
  emtfTrackMatchEta->setAxisTitle("Data #eta",1);
  emtfTrackMatchEta->setAxisTitle("Emul #eta", 2);

  emtfTrackMatchPhi = ibooker.book2D("emtfTrackMatchPhi","EMTF Track Match #phi", 126, -3.15, 3.15, 126, -3.15, 3.15);
  emtfTrackMatchPhi->setAxisTitle("Data #phi",1);
  emtfTrackMatchPhi->setAxisTitle("Emul #phi", 2);

  emtfTrackMatchPt = ibooker.book2D("emtfTrackMatchPt","EMTF Track Match p_{T}", 256, 1, 257, 256, 1, 257);
  emtfTrackMatchPt->setAxisTitle("Data p_{T}",1);
  emtfTrackMatchPt->setAxisTitle("Emul p_{T}", 2);

  emtfTrackMatchBx = ibooker.book2D("emtfTrackMatchBx","EMTF Track Match Bx", 9, -4, 4, 9, -4 ,4);
  emtfTrackMatchBx->setAxisTitle("Data Track Bx", 1);
  emtfTrackMatchBx->setAxisTitle("Emul Track Bx", 2);

  emtfTrackMatchQuality = ibooker.book2D("emtfTrackMatchQuality","EMTF Track Match Quality", 16, 0, 16, 16, 0, 16);
  emtfTrackMatchQuality->setAxisTitle("Data Quality",1);
  emtfTrackMatchQuality->setAxisTitle("Emul Quality", 2);

  emtfTrackMatchMode = ibooker.book2D("emtfTrackMatchMode","EMTF Track Match Mode", 16, 0, 16, 16, 0, 16);
  emtfTrackMatchMode->setAxisTitle("Data Mode",1);
  emtfTrackMatchMode->setAxisTitle("Emul Mode", 2);
  for (int bin = 1; bin <= 16; ++bin) {
    emtfTrackMatchQuality->setBinLabel(bin, std::to_string(bin - 1), 1);
    emtfTrackMatchMode->setBinLabel(bin, std::to_string(bin - 1), 1);
    emtfTrackMatchQuality->setBinLabel(bin, std::to_string(bin - 1), 2);
    emtfTrackMatchMode->setBinLabel(bin, std::to_string(bin - 1), 2);
  }

  //Difference plots
 
  emtfTrackEtaDif=ibooker.book1D("emtfTrackEtaDif", "EMTF Track Data #eta - Emu #eta",200, -5, 5);
  emtfTrackPhiDif=ibooker.book1D("emtfTrackPhiDif", "EMTF Track Data #phi - Emu #phi", 128, -3.2, 3.2);
  emtfTrackPtDif=ibooker.book1D("emtfTrackPtDif", "EMTF Track Data p_{T} - Emu p_{T}", 512, -256, 256);
  emtfTrackQualDif=ibooker.book1D("emtfTrackQualDif", "EMTF Track Data Quality - Emu Quality", 32, -16, 16);

//  emtfCollectionSizes = ibooker.book1D("emtfCollectionSizes","EMTF CollectionSizes", 3,0,3);
//  emtfCollectionSizes->setBinLabel(1,"Data Hits",1);
//  emtfCollectionSizes->setBinLabel(2,"Emul Hits",1);
//  emtfCollectionSizes->setBinLabel(1,"Data Tracks",1);
//  emtfCollectionSizes->setBinLabel(2,"Emul Tracks",1);
//  emtfCollectionSizes->setBinLabel(5,"Data Output" ,1);
//  emtfCollectionSizes->setBinLabel(6,"Emul output", 1);
}

void L1TdeStage2EMTF::analyze(const edm::Event& e, const edm::EventSetup& c) {

  if (verbose) edm::LogInfo("L1TdeStage2EMTF") << "L1TdeStage2EMTF: analyze..." << std::endl;

  //This module is only for direct by-event comparisons between data and emul
  //Ratio plots are made in the client L1TStage2EMTFDEClient.cc
  //Restrict BX to -1, 0, or 1
  edm::Handle<l1t::RegionalMuonCandBxCollection> dataMuons;
  e.getByToken(dataToken, dataMuons);

  edm::Handle<l1t::RegionalMuonCandBxCollection> emulMuons;
  e.getByToken(emulToken, emulMuons);

  emtfComparenMuonsEvent->Fill(dataMuons->size(), emulMuons->size());

  //data Hits
  edm::Handle<l1t::EMTFHitCollection> dataHitCollection;
  e.getByToken(datahitToken, dataHitCollection);
  //emul Hits
  edm::Handle<l1t::EMTFHitCollection> emulHitCollection;
  e.getByToken(emulhitToken, emulHitCollection);
 //data Tracks
    edm::Handle<l1t::EMTFTrackCollection> dataTrackCollection;
  e.getByToken(datatrackToken, dataTrackCollection);
  //emul Tracks
    edm::Handle<l1t::EMTFTrackCollection> emulTrackCollection;
  e.getByToken(emultrackToken, emulTrackCollection);


  //Best Match Regional Muon Cand plots
  float minhwdR = 999999;
  l1t::RegionalMuonCand closesthwData;
  l1t::RegionalMuonCand closesthwEmul;
 
  for (int itBXD = dataMuons->getFirstBX(); itBXD <= dataMuons->getLastBX(); ++itBXD) {
    if  (itBXD > 1 || itBXD < -1) continue;
    for (l1t::RegionalMuonCandBxCollection::const_iterator dataMuon = dataMuons->begin(itBXD); dataMuon != dataMuons->end(itBXD); ++dataMuon) {
      for (int itBXE = emulMuons->getFirstBX(); itBXE <= emulMuons->getLastBX(); ++itBXE) {
        if (itBXE > 1 || itBXE < -1) continue;
        for (l1t::RegionalMuonCandBxCollection::const_iterator emulMuon = emulMuons->begin(itBXE); emulMuon != emulMuons->end(itBXE); ++emulMuon) {
          int hwdPhi = dataMuon->hwPhi() - emulMuon->hwPhi();
          float hwdEta = dataMuon->hwEta() - emulMuon->hwEta();
          if((dataMuon->link() == emulMuon->link()) && (hwdPhi*hwdPhi + hwdEta*hwdEta) < minhwdR){
            minhwdR = hwdPhi*hwdPhi + hwdEta*hwdEta;
            closesthwData = *dataMuon;
            closesthwEmul = *emulMuon;
          }
        }
       }
     }
    }
  if(minhwdR != 999999){
    emtfMuonMatchhwEta->Fill(closesthwData.hwEta() - closesthwEmul.hwEta());
    emtfMuonMatchhwPhi->Fill(closesthwData.hwPhi() - closesthwEmul.hwPhi());
    emtfMuonMatchhwPt->Fill(closesthwData.hwPt() - closesthwEmul.hwPt());
    emtfMuonMatchhwQual->Fill(closesthwData.hwQual() - closesthwEmul.hwQual());
  }

    //Best Match Track plots

  float mindR = 999;

  l1t::EMTFTrack closestData; 
  l1t::EMTFTrack closestEmul;

  //Find the pair of data/emulator tracks that minimizes d(phi)^2 + d(eta)^2 
  for (std::vector<l1t::EMTFTrack>::const_iterator dataTrack = dataTrackCollection->begin(); dataTrack != dataTrackCollection->end(); ++dataTrack){
    if (dataTrack->BX() > 1 || dataTrack->BX() < -1) continue;
    for (std::vector<l1t::EMTFTrack>::const_iterator emulTrack = emulTrackCollection->begin(); emulTrack !=emulTrackCollection->end(); ++emulTrack){
      if (emulTrack->BX() > 1 || emulTrack->BX() < -1) continue;
      float dPhi = dataTrack->Phi_glob_rad() - emulTrack->Phi_glob_rad();
      if (dPhi > 3.14159) dPhi -= 3.14159;//ensure dPhi falls between -pi and +pi
      if (dPhi < -3.14159) dPhi += 3.14159;
      float dEta = dataTrack->Eta() - emulTrack->Eta();
      if (dPhi*dPhi + dEta*dEta < mindR){
        mindR = dPhi*dPhi + dEta*dEta;
        closestData = *dataTrack;
        closestEmul = *emulTrack;
      }
    }
  }
  if (mindR != 999){
    emtfTrackMatchEta->Fill(closestData.Eta(), closestEmul.Eta());
    emtfTrackEtaDif->Fill(closestData.Eta() - closestEmul.Eta());  
    emtfTrackMatchPhi->Fill(closestData.Phi_glob_rad(), closestEmul.Phi_glob_rad());
    emtfTrackPhiDif->Fill(closestData.Phi_glob_rad() - closestEmul.Phi_glob_rad());
    emtfTrackMatchPt->Fill(closestData.Pt(), closestEmul.Pt());
    emtfTrackPtDif->Fill(closestData.Pt() - closestEmul.Pt());
    emtfTrackMatchBx->Fill(closestData.BX(), closestEmul.BX()); 
    
    emtfTrackMatchMode->Fill(closestData.Mode(), closestEmul.Mode());
    
    emtfTrackMatchQuality->Fill(closestData.Quality(), closestEmul.Quality());
    emtfTrackQualDif->Fill(closestData.Quality() - closestEmul.Quality());
  } 

} 
