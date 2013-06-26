#ifndef DQM_CASTORMONITOR_CASTORTOWERJETMONITOR_H
#define DQM_CASTORMONITOR_CASTORTOWERJETMONITOR_H

#include "DQM/CastorMonitor/interface/CastorBaseMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/CastorReco/interface/CastorTower.h"
#include "DataFormats/CastorReco/interface/CastorCluster.h"
#include "DataFormats/CastorReco/interface/CastorJet.h"
#include "DataFormats/JetReco/interface/CastorJetID.h"
#include "RecoJets/JetProducers/interface/CastorJetIDHelper.h"
#include "RecoJets/JetProducers/plugins/CastorJetIDProducer.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "DataFormats/JetReco/interface/Jet.h"


class CastorTowerJetMonitor: public CastorBaseMonitor {

public:
  CastorTowerJetMonitor(); 
  ~CastorTowerJetMonitor(); 

  void setup(const edm::ParameterSet& ps, DQMStore* dbe);

  void processEventTowers(const reco::CastorTowerCollection& castorTowers);
  void processEventJets(const  reco::BasicJet& castorBasicJets);
  void processEventJetIDs(const reco::CastorJetIDValueMap& castorJetIDs);


  void reset();

  void done();

private: 
  
  MonitorElement* meEVT_;

  int ievt_;
 
  //=================== TOWERS =============//
 
  ////--- total energy of tower
  double energyTower;
   
  ////--- tower em energy
  double emEnergyTower;      
  ////--- tower had energy
  double hadEnergyTower;
  
  ////---- em to total fracton
  double femTower;
  
  ////--- pseudorapidity of tower centroid
  double etaTower; 
  /////---- azimuthal angle of tower centroid
  double phiTower;
  
  /////---- depth in z
  double depthTower;    
 
 ////--- number of towers per event
  int nTowers;

  //===================== JETS =================//
 
  unsigned int idx;

  ////--- total energy of jet
  double energyJet;
  ////--- eta of Jet
  double etaJet;      
  ////--- phi of Jet
  double phiJet;
  
   ////--- number of jets per event
  int nJets;
  
    ////--- number of jet IDs per event
  int nJetIDs;
  
  
  //--- add more here....

       ////---- define Monitor Elements  
       
       //-- tower
       MonitorElement* meCastorTowerEnergy;
       MonitorElement* meCastorTowerEMEnergy;
       MonitorElement* meCastorTowerHADEnergy;
       MonitorElement* meCastorTowerFEM;
       MonitorElement* meCastorTowerEta;
       MonitorElement* meCastorTowerPhi;
       MonitorElement* meCastorTowerDepth;
       MonitorElement* meCastorTowerMultiplicity; 
  
       //-- jet
       MonitorElement* meCastorJetEnergy;
       MonitorElement* meCastorJetEta;
       MonitorElement* meCastorJetPhi;
       MonitorElement* meCastorJetMultiplicity;
       MonitorElement* meCastorJetIDMultiplicity;
      //-- add more here....
 
 
};

#endif
