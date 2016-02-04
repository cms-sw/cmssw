#include "DQM/CastorMonitor/interface/CastorTowerJetMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//****************************************************//
//********** CastorTowerJetMonitor: ******************//
//********** Author: Dmytro Volyanskyy   *************//
//********** Date  : 18.03.2011 (first version) ******// 
//****************************************************//


//==================================================================//
//======================= Constructor ==============================//
//==================================================================//
CastorTowerJetMonitor::CastorTowerJetMonitor() {  
}


//==================================================================//
//======================= Destructor ===============================//
//==================================================================//
CastorTowerJetMonitor::~CastorTowerJetMonitor() {
}


//==================================================================//
//=========================== reset  ===============================//
//==================================================================//
void CastorTowerJetMonitor::reset(){
}


//==================================================================//
//=========================== done  ===============================//
//==================================================================//
void CastorTowerJetMonitor::done(){
}


//==================================================================//
//=========================== setup  ===============================//
//==================================================================//
void CastorTowerJetMonitor::setup(const edm::ParameterSet& ps, DQMStore* dbe){

  CastorBaseMonitor::setup(ps,dbe);
  baseFolder_ = rootFolder_+"CastorTowerJetMonitor";
   
  if(fVerbosity>0) std::cout << "CastorTowerJetMonitor::setup (start)" << std::endl;
  
   ////---- initialize these here
  ievt_=0;  nTowers=0; energyTower =0;  emEnergyTower = 0; hadEnergyTower=0;  femTower=0; etaTower=0; 
  phiTower=0; depthTower=0; nJets=0; energyJet =0; etaJet=0; phiJet=0;
      
  
  if ( m_dbe !=NULL ) {
    
    m_dbe->setCurrentFolder(baseFolder_);
    meEVT_ = m_dbe->bookInt("TowerJet Event Number");
    meEVT_->Fill(ievt_);
        
    m_dbe->setCurrentFolder(baseFolder_);
  
      ////---- book the following histograms for Towers 
      meCastorTowerEnergy    =  m_dbe->book1D("CASTOR Tower Total Energy","CASTOR Tower Total Energy",200,0,1000);
      meCastorTowerEMEnergy  =  m_dbe->book1D("CASTOR Tower EM Energy","CASTOR Tower EM Energy",200,0,1000);
      meCastorTowerHADEnergy  =  m_dbe->book1D("CASTOR Tower HAD Energy","CASTOR Tower HAD Energy",200,0,1000);
      meCastorTowerFEM   =  m_dbe->book1D("CASTOR Tower fem - EM to Total Energy Fraction","CASTOR fem - EM to Total Energy Fraction",20,0,1.2);
      meCastorTowerEta   =  m_dbe->book1D("CASTOR Tower Eta","CASTOR Tower Eta",42,-7,7);
      meCastorTowerPhi   =  m_dbe->book1D("CASTOR Tower Phi","CASTOR Tower Phi",35,-3.5,3.5);
      meCastorTowerDepth =  m_dbe->book1D("CASTOR Tower Depth","CASTOR Tower Depth",200,0,1000); 
      meCastorTowerMultiplicity =  m_dbe->book1D("CASTOR Tower Multiplicity","CASTOR Tower Multiplicity",20,0,20); 
      
      
      ////---- book the following histograms for Jets 
      meCastorJetEnergy    =  m_dbe->book1D("CASTOR BasicJet Total Energy","CASTOR BasicJet Total Energy",200,0,2000);
      meCastorJetEta   =  m_dbe->book1D("CASTOR BasicJet Eta","CASTOR BasicJet Eta",42,-7,7);
      meCastorJetPhi   =  m_dbe->book1D("CASTOR BasicJet Phi","CASTOR BasicJet Phi",35,-3.5,3.5);
      meCastorJetMultiplicity =  m_dbe->book1D("CASTOR BasicJet Multiplicity","CASTOR BasicJet Multiplicity",16,0,16); 
      meCastorJetIDMultiplicity =  m_dbe->book1D("CASTOR JetID Multiplicity","CASTOR JetID Multiplicity",16,0,16); 
      //-- add more here.....
 
   }
  
  
  else{ 
   if(fVerbosity>0) std::cout << "CastorTowerJetMonitor::setup - NO DQMStore service" << std::endl; 
  }

  if(fVerbosity>0) std::cout << "CastorTowerJetMonitor::setup (end)" << std::endl;
  

  return;
}




//=============================================================================//
//=========================== processEvent for Towers  ========================//
//============================================================================//

void  CastorTowerJetMonitor::processEventTowers(const reco::CastorTowerCollection& castorTowers)
{
  
  
 if(fVerbosity>0) 
   std::cout << "==>CastorTowerJetMonitor::processEventTowers !!!"<< std::endl;


  if(!m_dbe) { 
    if(fVerbosity>0) std::cout<<"CastorTowerJetMonitor::processEventTowers DQMStore is not instantiated!!!"<<std::endl;  
    return; 
  }

  meEVT_->Fill(ievt_);

  ////---- initialize these for every event
  nTowers=0; 

  ////----------------------------------////  
  ////---- look at CastorTowers --------////
  ////----------------------------------////
   
   if(castorTowers.size()>0){
 
  //for (size_t l=0; l<castorTowers.size() ; l++) 


  for(reco::CastorTowerCollection::const_iterator iTower= castorTowers.begin();  iTower!= castorTowers.end(); iTower++) {

     ////---- get total tower energy
      energyTower = iTower->energy();    
     ////--- tower em energy
     emEnergyTower = iTower->emEnergy();  
     ////--- tower had energy
      hadEnergyTower=iTower->hadEnergy();  
     ////---- em to total fracton
      femTower=iTower->fem();
     ////--- pseudorapidity of tower centroid
      etaTower=iTower->eta(); 
     /////---- azimuthal angle of tower centroid
      phiTower=iTower->phi();
      /////---- depth in z
      depthTower=iTower->depth(); 
      
      
      if (energyTower>0) {  //-- just a check
	
      ////---- fill histograms 
      meCastorTowerEnergy->Fill(energyTower);
      meCastorTowerEMEnergy->Fill(emEnergyTower);
      meCastorTowerHADEnergy->Fill(hadEnergyTower);
      meCastorTowerFEM->Fill(femTower);
      meCastorTowerEta->Fill(etaTower);
      meCastorTowerPhi->Fill(phiTower);
      meCastorTowerDepth->Fill(depthTower);
	
      nTowers++;
     }
    }
  meCastorTowerMultiplicity->Fill(nTowers);
   }
   
   
     else {
    if(fVerbosity>0) std::cout << "CastorTowerJetMonitor::processEvent NO Castor Towers !!!" << std::endl;
  }
     
   ievt_++;

  return;
}



//=============================================================================//
//=========================== processEvent for BasicJets  =====================//
//============================================================================//

void  CastorTowerJetMonitor::processEventJets(const  reco::BasicJet& castorBasicJets){
  
 if(fVerbosity>0) 
   std::cout << "==>CastorTowerJetMonitor::processEventJets !!!"<< std::endl;
   
  if(!m_dbe) { 
    if(fVerbosity>0) std::cout<<"CastorTowerJetMonitor::processEventJets DQMStore is not instantiated!!!"<<std::endl;  
    return; 
  }

   nJets=0; 
    ////---- loop over Castor Basic Jets
   for(reco::BasicJet::const_iterator ibegin = castorBasicJets.begin(), iend = castorBasicJets.end(), ijet = ibegin; ijet!= iend; ++ijet) {
     idx = ijet - ibegin;
     nJets++;
     //-- leave it empty for now - no time now...
     //-- add new stuff here...
    }
   meCastorJetMultiplicity->Fill(nJets);

    return;
}  


//============================================================================//
//=========================== processEvent for JetIDs  =======================//
//============================================================================//

void  CastorTowerJetMonitor::processEventJetIDs(const reco::CastorJetIDValueMap& castorJetIDs){


 if(fVerbosity>0) 
   std::cout << "==>CastorTowerJetMonitor::processEventJetIDs !!!"<< std::endl;   

 if(!m_dbe) { 
    if(fVerbosity>0) std::cout<<"CastorTowerJetMonitor::processEventJetIDs DQMStore is not instantiated!!!"<<std::endl;  
    return; 
  }  

  nJetIDs=0;

 ////---- loop over Castor JetIDs
   for(reco::CastorJetIDValueMap::const_iterator iJetID= castorJetIDs.begin();   iJetID!= castorJetIDs.end(); iJetID++) {
    //-- leave it empty for now - no time now...  
    //-- add new stuff here
    nJetIDs++;
   }
  meCastorJetIDMultiplicity->Fill(nJetIDs);
    
  
  return;
}
  


