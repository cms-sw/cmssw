#include "DQM/CastorMonitor/interface/CastorRecHitMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//***************************************************//
//********** CastorRecHitMonitor: *******************//
//********** Author: Dmytro Volyanskyy   ************//
//********** Date  : 23.09.2008 (first version) ******// 
//***************************************************//
///// energy and time of Castor RecHits 

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

   if(fVerbosity>0) cout << "CastorRecHitMonitor::setup (start)" << endl;
  

  if ( ps.getUntrackedParameter<bool>("RecHitsPerChannel", false) ){
    doPerChannel_ = true;
  }
    
  ievt_=0;
  
  if ( m_dbe !=NULL ) {    
    m_dbe->setCurrentFolder(baseFolder_);
    ////---- book MonitorElements
    meEVT_ = m_dbe->bookInt("RecHit Event Number"); // meEVT_->Fill(ievt_);
    castorHists.meRECHIT_E_all = m_dbe->book1D("Castor RecHit Energies","Castor RecHit Energies",150,0,150);
    castorHists.meRECHIT_T_all = m_dbe->book1D("Castor RecHit Times","Castor RecHit Times",300,-100,200);     
    castorHists.meRECHIT_MAP_CHAN_E = m_dbe->book1D("RecHit Channel Energy Map","RecHit Channel Energy Map",224,0,224);
  } 

  else{
  if(fVerbosity>0) cout << "CastorRecHitMonitor::setup - NO DQMStore service" << endl; 
 }

  if(fVerbosity>0) cout << "CastorRecHitMonitor::setup (end)" << endl;

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
			   DQMStore* dbe, string baseFolder) {
    
    std::map<HcalCastorDetId,MonitorElement*>::iterator _mei;

    string type = "CastorRecHitPerChannel";
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
	 sprintf(name,"Castor RecHit Energy zside=%d module=%d sector=%d", rhit.id().zside(), rhit.id().module(), rhit.id().sector());
         toolE[rhit.id()] =  dbe->book1D(name,name,200,-10,20); 
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
	sprintf(name,"Castor RecHit Time zside=%d module=%d sector=%d", rhit.id().zside(), rhit.id().module(), rhit.id().sector());
	toolT[rhit.id()] =  dbe->book1D(name,name,300,-100,200); 
	toolT[rhit.id()]->Fill(rhit.time());
      }
    }
    
    
  }
}

//==========================================================//
//================== processEvent ==========================//
//==========================================================//

void CastorRecHitMonitor::processEvent(const CastorRecHitCollection& castorHits ){

 cout << "==>CastorRecHitMonitor::processEvent !!!" << endl;

  if(!m_dbe) { 
    if(fVerbosity>0) cout <<"CastorRecHitMonitor::processEvent => DQMStore not instantiated !!!"<<endl;  
    return; 
  }

  ievt_++;  meEVT_->Fill(ievt_);

  CastorRecHitCollection::const_iterator CASTORiter;
  if (showTiming)  { cpu_timer.reset(); cpu_timer.start(); } 

  try
  {
     if(castorHits.size()>0)
    {    
       cout << "==>CastorRecHitMonitor::processEvent: castorHits.size()>0 !!!" << endl; 

      ////---- loop over all hits
      for (CASTORiter=castorHits.begin(); CASTORiter!=castorHits.end(); ++CASTORiter) { 
  
     ////---- get energy and time for every hit:
      float energy = CASTORiter->energy();    
      float time = CASTORiter->time();
      ////---- fill histograms with them:
      castorHists.meRECHIT_E_all->Fill(energy);
      castorHists.meRECHIT_T_all->Fill(time);
     
      ////---- plot energy vs channel 
      HcalCastorDetId id(CASTORiter->detid().rawId());
      //float zside  = id.zside(); 
      float module = id.module(); float sector = id.sector(); //get module & sector from id
      float channel = 16*(module-1)+sector; // define channel
      castorHists.meRECHIT_MAP_CHAN_E->Fill(channel,energy);

      ////---- do histograms per channel     
      if(doPerChannel_) 
         CastorRecHitPerChan::perChanHists<CastorRecHit>(*CASTORiter, castorHists.meRECHIT_E, castorHists.meRECHIT_T, m_dbe, baseFolder_); 
      }
	
   }
  }
  catch (...) { if(fVerbosity>0) cout<<"CastorRecHitMonitor::Error in processEvent !!!"<<endl; }

  if (showTiming)
    { 
      cpu_timer.stop(); cout << " TIMER::CastorRecHit -> " << cpu_timer.cpuTime() << endl; 
      cpu_timer.reset(); cpu_timer.start();  
    }

  return;
}


