#include "DQM/CastorMonitor/interface/CastorRecHitMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//***************************************************//
//********** CastorRecHitMonitor: *******************//
//********** Author: Dmytro Volyanskyy   ************//
//********** Date  : 23.09.2008 (first version) ******// 
////---- energy and time of Castor RecHits 
////---- last revision: Pedro Cipriano 09.07.2013 
//***************************************************//
//---- critical revision 26.06.2014 (Vladimir Popov)
//==================================================================//
//======================= Constructor ==============================//
CastorRecHitMonitor::CastorRecHitMonitor(const edm::ParameterSet& ps)
{
 subsystemname =
	ps.getUntrackedParameter<std::string>("subSystemFolder","Castor");
 ievt_=0; 
}

//======================= Destructor ==============================//
CastorRecHitMonitor::~CastorRecHitMonitor() { }

//=================== setup ===============//

void CastorRecHitMonitor::setup(const edm::ParameterSet& ps)
{
//  CastorBaseMonitor::setup(ps);
  return;
}


//============== boolHistograms  ==============//
void CastorRecHitMonitor::bookHistograms(DQMStore::IBooker& ibooker,
	const edm::Run& iRun, const edm::EventSetup& iSetup)
{
 char s[60];
 if(fVerbosity>0) 
  std::cout<<"CastorRecHitMonitor::bookHistograms"<<std::endl;
 ibooker.setCurrentFolder(subsystemname + "/CastorRecHitMonitor");

 sprintf(s,"CastorRecHitSumInSectors");
    //h2RHvsSec = ibooker.book2D(s,s, 16, 0,16, 20000, 0.,200000.);
    //h2RHvsSec->getTH2F()->GetXaxis()->SetTitle("sectorPhi");
    //h2RHvsSec->getTH2F()->GetYaxis()->SetTitle("RecHit");
    //h2RHvsSec->getTH2F()->SetOption("colz");

  sprintf(s,"CastorTileRecHit");
    //h2RHchan = ibooker.book2D(s,s, 224, 0,224, 5100, -1000,50000.);
    //h2RHchan->getTH2F()->GetXaxis()->SetTitle("sector*14+module");
    //h2RHchan->getTH2F()->GetYaxis()->SetTitle("RecHit");
    //h2RHchan->getTH2F()->SetOption("colz");  

  sprintf(s,"CastorRecHitMap(cumulative)");
    h2RHmap = ibooker.book2D(s,s,14, 0,14, 16, 0,16);
    h2RHmap->getTH2F()->GetXaxis()->SetTitle("moduleZ");  
    h2RHmap->getTH2F()->GetYaxis()->SetTitle("sectorPhi");
    h2RHmap->getTH2F()->SetOption("colz");

  sprintf(s,"CastorRecHitOccMap");
    h2RHoccmap = ibooker.book2D(s,s,14, 0,14, 16, 0,16);
    h2RHoccmap->getTH2F()->GetXaxis()->SetTitle("moduleZ");
    h2RHoccmap->getTH2F()->GetYaxis()->SetTitle("sectorPhi");
    h2RHoccmap->getTH2F()->SetOption("colz");

  sprintf(s,"CastorRecHitEntriesMap");
    h2RHentriesMap = ibooker.book2D(s,s,14, 0,14, 16, 0,16);
    h2RHentriesMap->getTH2F()->GetXaxis()->SetTitle("moduleZ");
    h2RHentriesMap->getTH2F()->GetYaxis()->SetTitle("sectorPhi");
    h2RHentriesMap->getTH2F()->SetOption("colz");

  sprintf(s,"CastorRecHitTime");
    hRHtime = ibooker.book1D(s,s,301, -101.,200.);

  sprintf(s,"Reco all tiles");
   hallchan = ibooker.book1D(s,s,22000,-20000.,200000.);
 
  if(fVerbosity>0) 
    std::cout<<"CastorRecHitMonitor::bookHistograms(end)"<<std::endl;
  return;
}

//================== processEvent ==========================//
void CastorRecHitMonitor::processEvent(const CastorRecHitCollection& castorHits )
{
 if(fVerbosity>0) std::cout << "CastorRecHitMonitor::processEvent (begin)"<< std::endl;
 ievt_++; 
 for (int z=0; z<14; z++) for (int phi=0; phi<16; phi++)
	energyInEachChannel[z][phi] = 0.;

 CastorRecHitCollection::const_iterator CASTORiter;
 if (showTiming)  { cpu_timer.reset(); cpu_timer.start(); } 

 if(castorHits.size() <= 0) return;

 for(CASTORiter=castorHits.begin(); CASTORiter!=castorHits.end(); ++CASTORiter)
 { 
   float energy = CASTORiter->energy();    
   float time = CASTORiter->time();
   float time2 = time;
   if(time < -100.) time2 = -100.;
   hRHtime->Fill(time2);

   HcalCastorDetId id(CASTORiter->detid().rawId());
      //float zside  = id.zside(); 
   int module = (int)id.module(); //-- get module
   int sector = (int)id.sector(); //-- get sector 

   energyInEachChannel[module-1][sector-1] += energy;

   h2RHentriesMap->Fill(module-1,sector-1);
 } // end for(CASTORiter=castorHits.begin(); CASTORiter!= ...

  for(int phi=0; phi<16; phi++) {
    double es = 0.;
    for (int z=0; z<14; z++) {
      //int ind = phi*14 + z +1;
      float rh = energyInEachChannel[z][phi];
      //h2RHchan->Fill(ind,rh);
      hallchan->Fill(rh);
      if(rh < 0.) continue;      
      h2RHmap->Fill(z,phi,rh); 
      es += rh;
    }
    //h2RHvsSec->Fill(phi,es);
  } // end for(int phi=0;

 if(ievt_ %100 == 0) 
  for(int mod=1; mod<=14; mod++) for(int sec=1; sec<=16;sec++) {
    double a= h2RHmap->getTH2F()->GetBinContent(mod,sec);
    h2RHoccmap->getTH2F()->SetBinContent(mod,sec,a/double(ievt_));
  }

  if(fVerbosity>0) std::cout << "CastorRecHitMonitor::processEvent (end)"<< std::endl;
  return;
}
