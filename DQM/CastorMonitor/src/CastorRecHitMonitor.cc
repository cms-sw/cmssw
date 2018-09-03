//***************************************************
//		 CastorRecHitMonitor
//	 Author: Dmytro Volyanskyy
//	 Date  : 23.09.2008 (first version)
// last modification: Pedro Cipriano 09.07.2013
//----------------------------------------------
//	critical revision 26.06.2014 (Vladimir Popov)
//***************************************************

#include "DQM/CastorMonitor/interface/CastorRecHitMonitor.h"
#include <string>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

using namespace std;

CastorRecHitMonitor::CastorRecHitMonitor(const edm::ParameterSet& ps) {
  fVerbosity = ps.getUntrackedParameter<int>("debug", 0);
  if (fVerbosity > 0)
    std::cout << "CastorRecHitMonitor Constructor: " << this << std::endl;
  subsystemname =
      ps.getUntrackedParameter<std::string>("subSystemFolder", "Castor");
  ievt_ = 0;
}

CastorRecHitMonitor::~CastorRecHitMonitor() {}

void CastorRecHitMonitor::bookHistograms(DQMStore::IBooker& ibooker,
                                         const edm::Run& iRun,
                                         const edm::EventSetup& iSetup) {
  char s[60];
  if (fVerbosity > 0)
    std::cout << "CastorRecHitMonitor::bookHistograms" << std::endl;
  ibooker.setCurrentFolder(subsystemname + "/CastorRecHitMonitor");

  const int N_Sec = 16;
  const int nySec = 20;
  float ySec[nySec + 1];
  float xSec[N_Sec + 1];
  double E0sec = 1. / 1024.;
  ySec[0] = 0.;
  ySec[1] = E0sec;
  double lnBsec = log(2.);
  for (int j = 1; j < nySec; j++) ySec[j + 1] = E0sec * exp(j * lnBsec);
  for (int i = 0; i <= N_Sec; i++) xSec[i] = i;

  sprintf(s, "Castor Energy by Sectors #Phi");
  h2RHvsSec = ibooker.book2D(s, s, N_Sec, xSec, nySec, ySec);
  h2RHvsSec->getTH2F()->GetXaxis()->SetTitle("sector #Phi");
  h2RHvsSec->getTH2F()->GetYaxis()->SetTitle("RecHit / GeV");
  h2RHvsSec->getTH2F()->SetOption("colz");

  const int nxCh = 224;
  const int nyE = 18;
  float xCh[nxCh + 1];
  float yErh[nyE + 1];
  for (int i = 0; i <= nxCh; i++) xCh[i] = i;
  double E0 = 1. / 1024.;
  double lnA = log(2.);
  yErh[0] = 0.;
  yErh[1] = E0;
  for (int j = 1; j < nyE; j++) yErh[j + 1] = E0 * exp(j * lnA);

  string st = "Castor Cell Energy Map (cell-wise)";
  h2RHchan = ibooker.book2D(st, st + ";moduleZ*16 + sector #Phi;RecHit / GeV",
                            nxCh, xCh, nyE, yErh);
  h2RHchan->getTH2F()->SetOption("colz");

  sprintf(s, "Castor Cell Energy");
  hallchan = ibooker.book1D(s, s, nyE, yErh);
  hallchan->getTH1F()->GetXaxis()->SetTitle("GeV");

  st = "Castor cell avr Energy per event Map Z-Phi";
  h2RHoccmap = ibooker.bookProfile2D(st, st + ";module Z;sector Phi", 14, 0, 14,
                                     16, 0, 16, 0., 1.e10, "");
  h2RHoccmap->getTProfile2D()->SetOption("colz");

  sprintf(s, "CastorRecHitEntriesMap");
  h2RHentriesMap = ibooker.book2D(s, s, 14, 0, 14, 16, 0, 16);
  h2RHentriesMap->getTH2F()->GetXaxis()->SetTitle("moduleZ");
  h2RHentriesMap->getTH2F()->GetYaxis()->SetTitle("sector #Phi");
  h2RHentriesMap->getTH2F()->SetOption("colz");

  sprintf(s, "CastorRecHitTime");
  hRHtime = ibooker.book1D(s, s, 301, -101., 200.);

  sprintf(s, "CASTORTowerDepth");
  hTowerDepth = ibooker.book1D(s, s, 130, -15500., -14200.);
  hTowerDepth->getTH1F()->GetXaxis()->SetTitle("mm");

  sprintf(s, "CASTORTowerMultiplicity");
  hTowerMultipl = ibooker.book1D(s, s, 20, 0., 20.);

  const int NEtow = 20;
  float EhadTow[NEtow + 1];
  float EMTow[NEtow + 1];
  float ETower[NEtow + 2];
  double E0tow = 1. / 1024.;
  EMTow[0] = 0.;
  EMTow[1] = E0tow;
  EhadTow[0] = 0.;
  EhadTow[1] = E0tow;
  ETower[0] = 0.;
  ETower[1] = E0tow;
  double lnBtow = log(2.);
  for (int j = 1; j < NEtow; j++) EMTow[j + 1] = E0tow * exp(j * lnBtow);
  for (int j = 1; j < NEtow; j++) EhadTow[j + 1] = E0tow * exp(j * lnBtow);
  for (int j = 1; j <= NEtow; j++) ETower[j + 1] = E0tow * exp(j * lnBtow);

  sprintf(s, "CASTORTowerEMvsEhad");
  h2TowerEMhad = ibooker.book2D(s, s, NEtow, EhadTow, NEtow, EMTow);
  h2TowerEMhad->getTH2F()->GetXaxis()->SetTitle("Ehad / GeV");
  h2TowerEMhad->getTH2F()->GetYaxis()->SetTitle("EM / GeV");
  h2TowerEMhad->getTH2F()->SetOption("colz");

  sprintf(s, "CASTORTowerTotalEnergy");
  hTowerE = ibooker.book1D(s, s, NEtow + 1, ETower);
  hTowerE->getTH1F()->GetXaxis()->SetTitle("GeV");

  sprintf(s, "CASTORJetsMultiplicity");
  hJetsMultipl = ibooker.book1D(s, s, 16, 0., 16.);

  sprintf(s, "CASTORJetEnergy");
  hJetEnergy = ibooker.book1D(s, s, 5000, 0., 500.);

  sprintf(s, "CASTORJetEta");
  hJetEta = ibooker.book1D(s, s, 126, -6.3, 6.3);

  sprintf(s, "CASTORJetPhi");
  hJetPhi = ibooker.book1D(s, s, 63, -3.15, 3.15);

  if (fVerbosity > 0)
    std::cout << "CastorRecHitMonitor::bookHistograms(end)" << std::endl;
  return;
}

void CastorRecHitMonitor::processEventTowers(
    const reco::CastorTowerCollection& castorTowers) {
  if (castorTowers.empty()) return;
  int nTowers = 0;

  for (reco::CastorTowerCollection::const_iterator iTower =
           castorTowers.begin();
       iTower != castorTowers.end(); iTower++) {
    hTowerE->Fill(iTower->energy() * 0.001);
    h2TowerEMhad->Fill(iTower->hadEnergy() * 0.001, iTower->emEnergy() * 0.001);
    hTowerDepth->Fill(iTower->depth());
    nTowers++;
  }
  hTowerMultipl->Fill(nTowers);
}

void CastorRecHitMonitor::processEvent(
    const CastorRecHitCollection& castorHits) {
  if (fVerbosity > 0)
    std::cout << "CastorRecHitMonitor::processEvent (begin)" << std::endl;
  ievt_++;
  for (int z = 0; z < 14; z++)
    for (int phi = 0; phi < 16; phi++) energyInEachChannel[z][phi] = 0.;

  CastorRecHitCollection::const_iterator CASTORiter;
  // if (showTiming)  { cpu_timer.reset(); cpu_timer.start(); }

  if (castorHits.empty()) return;

  for (CASTORiter = castorHits.begin(); CASTORiter != castorHits.end();
       ++CASTORiter) {
    float energy = CASTORiter->energy();
    float time = CASTORiter->time();
    float time2 = time;
    if (time < -100.) time2 = -100.;
    hRHtime->Fill(time2);

    HcalCastorDetId id(CASTORiter->detid().rawId());
    // float zside  = id.zside();
    int module = (int)id.module();  //-- get module
    int sector = (int)id.sector();  //-- get sector

    energyInEachChannel[module - 1][sector - 1] += energy;

    h2RHentriesMap->Fill(module - 1, sector - 1);
  }  // end for(CASTORiter=castorHits.begin(); CASTORiter!= ...

  double etot = 0.;
  for (int phi = 0; phi < 16; phi++) {
    double es = 0.;
    for (int z = 0; z < 14; z++) {
      float rh = energyInEachChannel[z][phi] * 0.001;
      int ind = z * 16 + phi + 1;
      //    int ind = phi*14 + z +1;
      h2RHchan->Fill(ind, rh);
      hallchan->Fill(rh);
      if (rh < 0.) continue;
      h2RHoccmap->Fill(z, phi, rh);
      es += rh;
    }
    h2RHvsSec->Fill(phi, es);
    etot += es;
  }  // end for(int phi=0;

  if (fVerbosity > 0)
    std::cout << "CastorRecHitMonitor::processEvent (end)" << std::endl;
  return;
}

void CastorRecHitMonitor::processEventJets(
    const reco::BasicJetCollection& Jets) {
  int nJets = 0;
  for (reco::BasicJetCollection::const_iterator ibegin = Jets.begin(),
                                                iend = Jets.end(),
                                                ijet = ibegin;
       ijet != iend; ++ijet) {
    nJets++;
    float energy = ijet->energy() * 0.001;
    hJetEnergy->Fill(energy);
    hJetEta->Fill(ijet->eta());
    hJetPhi->Fill(ijet->phi());
  }
  hJetsMultipl->Fill(nJets);
}
