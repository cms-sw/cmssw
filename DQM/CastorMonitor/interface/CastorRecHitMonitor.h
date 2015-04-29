#ifndef DQM_CASTORMONITOR_CASTORRECHITMONITOR_H
#define DQM_CASTORMONITOR_CASTORRECHITMONITOR_H

#include "DQM/CastorMonitor/interface/CastorBaseMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
//#include "DataFormats/HcalRecHit/interface/CastorRecHit.h"

#include "DataFormats/CastorReco/interface/CastorTower.h"
#include "DataFormats/CastorReco/interface/CastorCluster.h"
#include "DataFormats/CastorReco/interface/CastorJet.h"
#include "DataFormats/JetReco/interface/CastorJetID.h"
#include "RecoJets/JetProducers/interface/CastorJetIDHelper.h"
#include "RecoJets/JetProducers/plugins/CastorJetIDProducer.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "DataFormats/JetReco/interface/Jet.h"

#include "DataFormats/Common/interface/TriggerResults.h"

class CastorRecHitMonitor: public CastorBaseMonitor {
public:
  CastorRecHitMonitor(const edm::ParameterSet& ps); 
  ~CastorRecHitMonitor(); 

  void setup(const edm::ParameterSet& ps);
 void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &);
  void processEvent(const CastorRecHitCollection& castorHits);
 void processEventTowers(const reco::CastorTowerCollection& castorTowers);
 void processEventJets(const reco::BasicJetCollection& Jets);
private:  
 int ievt_;
 float energyInEachChannel[14][16];
 std::string subsystemname;

 MonitorElement *hTowerE;
 MonitorElement *hTowerDepth;
 MonitorElement *h2TowerEMhad;
 MonitorElement *hTowerMultipl;
 MonitorElement *hJetsMultipl;
 MonitorElement *hJetEnergy;
 MonitorElement *hJetEta;
 MonitorElement *hJetPhi;

 TH2F *h2RecHitMap;
 MonitorElement* h2RHchan;
 MonitorElement* h2RHvsSec;
 MonitorElement* h2RHmap;
 MonitorElement* h2RHoccmap;
 MonitorElement* h2RHentriesMap;
 MonitorElement* hRHtime;
 MonitorElement* hallchan;
};

#endif
