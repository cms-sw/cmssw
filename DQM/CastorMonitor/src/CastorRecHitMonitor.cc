#include "DQM/CastorMonitor/interface/CastorRecHitMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//***************************************************//
//********** CastorRecHitMonitor: *******************//
//********** Author: Dmytro Volyanskyy   ************//
//********** Date  : 23.09.2008 (first version) ******// 
//***************************************************//
////---- energy and time of Castor RecHits 
////---- last revision: 05.03.2010 

//==================================================================//
//======================= Constructor ==============================//
//==================================================================//
CastorRecHitMonitor::CastorRecHitMonitor() {
  doPerChannel_ = true;
  //  occThresh_ = 1;
  ievt_=0; 
}

//==================================================================//
//======================= Destructor ==============================//
//==================================================================//
CastorRecHitMonitor::~CastorRecHitMonitor(){
}

void CastorRecHitMonitor::reset(){
}


//==========================================================//
//========================= setup ==========================//
//==========================================================//

void CastorRecHitMonitor::setup(const edm::ParameterSet& ps, DQMStore* dbe){
  
  CastorBaseMonitor::setup(ps,dbe);
  baseFolder_ = rootFolder_+"CastorRecHitMonitor";

   if(fVerbosity>0) std::cout << "CastorRecHitMonitor::setup (start)" << std::endl;
  
  if ( ps.getUntrackedParameter<bool>("RecHitsPerChannel", false) ){
    doPerChannel_ = true;
  }
    
  ievt_=0; module=0; sector=0; zside=0; totEnergy=0.;

  
   ////---- initialize the array energyInEachChannel  for each new event
   for (int mod=0; mod<14; mod++){
     for (int sec=0; sec<16; sec++){
          energyInEachChannel[mod][sec] = 0.;
          allEnergyModule[mod]=0;
          allEnergySector[sec]=0;
      }
    }
  
  if ( m_dbe !=NULL ) {    
 m_dbe->setCurrentFolder(baseFolder_);
////---- book MonitorElements
meEVT_ = m_dbe->bookInt("RecHit Event Number"); // meEVT_->Fill(ievt_);
////---- energy and time of all RecHits
castorHists.meRECHIT_E_all = m_dbe->book1D("CastorRecHit Energies- above threshold on RecHitEnergy","CastorRecHit Energies- above threshold on RecHitEnergy",150,0,150);
castorHists.meRECHIT_T_all = m_dbe->book1D("CastorRecHit Times- above threshold on RecHitEnergy","CastorRecHit Times- above threshold on RecHitEnergy",300,-100,100);    
////---- 1D energy map
castorHists.meRECHIT_MAP_CHAN_E = m_dbe->book1D("CastorRecHit Energy in each channel- above threshold","CastorRecHit Energy in each channel- above threshold",224,0,224);
////---- energy in modules    
castorHists.meRECHIT_E_modules = m_dbe->book1D("CastorRecHit Energy in modules- above threshold","CastorRecHit Energy in modules- above threshold", 14, 0, 14);
////---- energy in sectors    
castorHists.meRECHIT_E_sectors = m_dbe->book1D("CastorRecHit Energy in sectors- above threshold","CastorRecHit Energy in sectors- above threshold", 16, 0, 16);

////---- number of rechits in modules    
castorHists.meRECHIT_N_modules = m_dbe->book1D("Number of CastorRecHits in modules- above threshold","Number of CastorRecHits in modules- above threshold", 14, 0, 14);
////---- number of rechist in sectors    
castorHists.meRECHIT_N_sectors = m_dbe->book1D("Number of CastorRecHits in sectors- above threshold","Number of CastorRecHits in sectors- above threshold", 16, 0, 16);
////---- occupancy
castorHists.meCastorRecHitsOccupancy = m_dbe->book2D("CastorRecHits Occupancy Map", "CastorRecHits Occupancy Map", 14, 0,14, 16, 0,16);
////---- number of rechits per event 
castorHists.meRECHIT_N_event = m_dbe->book1D("Number of CASTOR RecHits per event- above threshold","Number of CASTOR RecHits per event- above threshold", 23, 0, 230);

 m_dbe->setCurrentFolder(baseFolder_+"/EnergyFraction");
////---- fraction of the energy deposted in each module    
castorHists.meRECHIT_E_relative_modules = m_dbe->book1D("Fraction of the total energy in CASTOR modules","Fraction of the total energy in CASTOR modules", 14, 0, 14);
////---- fraction of the energy deposted in each sector    
castorHists.meRECHIT_E_relative_sectors = m_dbe->book1D("Fraction of the total energy in CASTOR sectors","Fraction of the total energy in CASTOR sectors", 16, 0, 16);
/*
castorHists.meRECHIT_E_relative_modules->getTH1F()->GetYaxis()->SetTitle("fraction of the total energy");
castorHists.meRECHIT_E_relative_sectors->getTH1F()->GetYaxis()->SetTitle("fraction of the total energy");
 
   char ibin1[1024];
    for(int i=0; i<14; i++){
     sprintf(ibin1,"%i", i+1);
     castorHists.meRECHIT_E_relative_modules->getTH1F()->GetXaxis()->SetBinLabel(i+1,ibin1);
    }
   char ibin2[1024];
    for(int i=0; i<16; i++){
     sprintf(ibin2,"%i", i+1);
     castorHists.meRECHIT_E_relative_sectors->getTH1F()->GetXaxis()->SetBinLabel(i+1,ibin2);
    }
*/

  } 



  else{
  if(fVerbosity>0) std::cout << "CastorRecHitMonitor::setup - NO DQMStore service" << std::endl; 
 }

  if(fVerbosity>0) std::cout << "CastorRecHitMonitor::setup (end)" << std::endl;

  return;
}

//==========================================================//
//=============== do histograms for every channel ==========//
//==========================================================//

namespace CastorRecHitPerChan{

  template<class RecHit>

  inline void perChanHists(const RecHit& rhit, 
			   std::map<HcalCastorDetId, MonitorElement*> &toolE, 
			   std::map<HcalCastorDetId, MonitorElement*> &toolT,
			   DQMStore* dbe, std::string baseFolder) {
    
    std::map<HcalCastorDetId,MonitorElement*>::iterator _mei;

    std::string type = "CastorRecHitPerChannel";
    if(dbe) dbe->setCurrentFolder(baseFolder+"/"+type);
    
    ////---- energies by channel  
    _mei=toolE.find(rhit.id()); //-- look for a histogram with this hit's id !!!
    if (_mei!=toolE.end()){
      if (_mei->second==0) return;
      else _mei->second->Fill(rhit.energy()); //-- if it's there, fill it with energy
    }
    else{
       if(dbe){
	 char name[1024];
	 sprintf(name,"CastorRecHit Energy zside=%d module=%d sector=%d", rhit.id().zside(), rhit.id().module(), rhit.id().sector());
         toolE[rhit.id()] =  dbe->book1D(name,name,60,-10,20); 
	 toolE[rhit.id()]->Fill(rhit.energy());
      }
    }
    
    ////---- times by channel
    _mei=toolT.find(rhit.id()); //-- look for a histogram with this hit's id
    if (_mei!=toolT.end()){
      if (_mei->second==0) return;
      else _mei->second->Fill(rhit.time()); //-- if it's there, fill it with time
    }
    else{
      if(dbe){
	char name[1024];
	sprintf(name,"CastorRecHit Time zside=%d module=%d sector=%d", rhit.id().zside(), rhit.id().module(), rhit.id().sector());
	toolT[rhit.id()] =  dbe->book1D(name,name,200,-100,100); 
	toolT[rhit.id()]->Fill(rhit.time());
      }
    }
    
    
  }
}

//==========================================================//
//================== processEvent ==========================//
//==========================================================//

void CastorRecHitMonitor::processEvent(const CastorRecHitCollection& castorHits ){

  if(fVerbosity>0) std::cout << "==>CastorRecHitMonitor::processEvent !!!"<< std::endl;
       

  ////--- increment here
  ievt_++; totEnergy=0.;

  ////---- fill the event number
   meEVT_->Fill(ievt_);



  if(!m_dbe) { 
    if(fVerbosity>0) std::cout <<"CastorRecHitMonitor::processEvent => DQMStore is not instantiated !!!"<<std::endl;  
    return; 
  }

  CastorRecHitCollection::const_iterator CASTORiter;
  if (showTiming)  { cpu_timer.reset(); cpu_timer.start(); } 

  //castorHists.meRECHIT_N_event->Fill(castorHits.size());
   int iHit=0;

     if(castorHits.size()>0)
    {    
       if(fVerbosity>0) std::cout << "==>CastorRecHitMonitor::processEvent: castorHits.size()>0 !!!" << std::endl; 

      //////////////////////---------------------- loop over all hits
      for (CASTORiter=castorHits.begin(); CASTORiter!=castorHits.end(); ++CASTORiter) { 
  
     ////---- get energy and time for every hit:
      energy = CASTORiter->energy();    
      time = CASTORiter->time();
      
      ////---- plot energy vs channel 
      HcalCastorDetId id(CASTORiter->detid().rawId());
      //float zside  = id.zside(); 
      module = (int)id.module(); //-- get module
      sector = (int)id.sector(); //-- get sector 
      channel = 16*(module-1)+sector; //-- define channel

      if (energy>0.) { 
      ////---- fill histograms with energy and time for every hit:
      castorHists.meRECHIT_E_all->Fill(energy);
      castorHists.meRECHIT_T_all->Fill(time);
      ////---- fill energy vs channel     
      castorHists.meRECHIT_MAP_CHAN_E->Fill(channel,energy);
      ////---- fill energy in modules
      castorHists.meRECHIT_E_modules->Fill(module-1, energy);
      ////---- fill energy in sectors
      castorHists.meRECHIT_E_sectors->Fill(sector-1, energy);
      ////---- fill number of rechits in modules
      castorHists.meRECHIT_N_modules->Fill(module-1);
      ////---- fill number of rechits in sectors
      castorHists.meRECHIT_N_sectors->Fill(sector-1);     
      iHit++;
     }

      ////---- do not deal with negative values
      if(energy<0) energy=0;

     ////--- fill the array energyTotalInEachChannel for a particular event
      energyInEachChannel[module-1][sector-1] += energy; 
     
      ////---- fill the occupancy
      castorHists.meCastorRecHitsOccupancy->Fill(module-1,sector-1, energy); 

      ////---- fill the number of rechits per event
        castorHists.meRECHIT_N_event->Fill(iHit);

      ////---- do histograms per channel once per 100 events     
      if( doPerChannel_) 
         CastorRecHitPerChan::perChanHists<CastorRecHit>(*CASTORiter, castorHists.meRECHIT_E, castorHists.meRECHIT_T, m_dbe, baseFolder_); 
     }///////////////////////////------------------- end of loop over all hits

      ////---- get the total energy deposited over all events
      for(int mod=0; mod<14;mod++)
	for(int sec=0; sec<16;sec++)
          totEnergy+=energyInEachChannel[mod][sec];

       ////---- get the energy in each module over all events
       for(int mod=0; mod<14;mod++)
       allEnergyModule[mod]= energyInEachChannel[mod][0]+energyInEachChannel[mod][1]+energyInEachChannel[mod][2]+energyInEachChannel[mod][3]+
                          energyInEachChannel[mod][4]+energyInEachChannel[mod][5]+energyInEachChannel[mod][6]+energyInEachChannel[mod][7]+
                          energyInEachChannel[mod][8]+energyInEachChannel[mod][9]+energyInEachChannel[mod][10]+energyInEachChannel[mod][11]+
                           energyInEachChannel[mod][12]+energyInEachChannel[mod][13]+energyInEachChannel[mod][14]+energyInEachChannel[mod][15];
                         
     ////---- get the energy in each sector over all events
       for(int sec=0; sec<16;sec++)
       allEnergySector[sec]= energyInEachChannel[0][sec]+energyInEachChannel[1][sec]+energyInEachChannel[2][sec]+energyInEachChannel[3][sec]+
                          energyInEachChannel[4][sec]+energyInEachChannel[5][sec]+energyInEachChannel[6][sec]+energyInEachChannel[7][sec]+
                          energyInEachChannel[8][sec]+energyInEachChannel[9][sec]+energyInEachChannel[10][sec]+energyInEachChannel[11][sec]+
	                  energyInEachChannel[12][sec]+energyInEachChannel[13][sec];


   ////---- fill relative energy in modules and sectors 
       for(int mod=0; mod<14;mod++)
      castorHists.meRECHIT_E_relative_modules->getTH1F()->SetBinContent(mod+1,allEnergyModule[mod]/totEnergy);
      for(int sec=0; sec<16;sec++)
	castorHists.meRECHIT_E_relative_sectors->getTH1F()->SetBinContent(sec+1,allEnergySector[sec]/totEnergy); 

  

   }

  else { if(fVerbosity>0) std::cout<<"CastorRecHitMonitor::processEvent NO Castor RecHits !!!"<<std::endl; }

  if (showTiming) { 
      cpu_timer.stop(); std::cout << " TIMER::CastorRecHit -> " << cpu_timer.cpuTime() << std::endl; 
      cpu_timer.reset(); cpu_timer.start();  
    }
    
 
  return;
}


