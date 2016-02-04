#include "DQM/CastorMonitor/interface/CastorChannelQualityMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//***************************************************//
//********** CastorChannelQualityMonitor *******************//
//********** Author: Dmytro Volyanskyy   ************//
//********** Date  : 04.03.2010 (first version) ******// 
//***************************************************//
////---- energy and time of Castor RecHits 
////---- revision: 05.03.2010 (Dima Volyanskyy)
////----- last revision 31.05.2011 (Panos Katsas)

//==================================================================//
//======================= Constructor ==============================//
//==================================================================//
CastorChannelQualityMonitor::CastorChannelQualityMonitor(){
 ievt_=0;
 counter1=0;
 counter2=0;
}

//==================================================================//
//======================= Destructor ==============================//
//==================================================================//

CastorChannelQualityMonitor::~CastorChannelQualityMonitor(){
}


void CastorChannelQualityMonitor::reset(){
}


//==========================================================//
//========================= setup ==========================//
//==========================================================//

void CastorChannelQualityMonitor::setup(const edm::ParameterSet& ps, DQMStore* dbe){

  CastorBaseMonitor::setup(ps,dbe);
   averageEnergyMethod_ = ps.getUntrackedParameter<bool>("averageEnergyMethod", true); 
   offline_             = ps.getUntrackedParameter<bool>("OfflineMode", false); 
   nThreshold_          = ps.getUntrackedParameter<double>("nThreshold", 0);
   dThreshold_          = ps.getUntrackedParameter<double>("dThreshold", 0);

  if ( m_dbe !=NULL ) {    
    ////---- create ReportSummary Map 
    m_dbe->setCurrentFolder(rootFolder_+"CastorChannelQuality");
    reportSummary    = m_dbe->bookFloat("RecHit Energy based reportSummary");
    reportSummaryMap = m_dbe->book2D("RecHitEnergyBasedSummaryMap","RecHitEnergyBasedSummaryMap",14,0.0,14.0,16,0.0,16.0);
    if(offline_){
      h_reportSummaryMap =reportSummaryMap->getTH2F();
      h_reportSummaryMap->SetOption("textcolz");
      h_reportSummaryMap->GetXaxis()->SetTitle("module");
      h_reportSummaryMap->GetYaxis()->SetTitle("sector");
    }
    
    overallStatus = m_dbe->bookFloat("RecHit Energy based fraction of good channels");
   } 

  else{
  if(fVerbosity>0) std::cout << "CastorChannelQualityMonitor::setup - NO DQMStore service" << std::endl; 
 }

  // baseFolder_ = rootFolder_+"CastorChannelQualityMonitor";

  if(fVerbosity>0) std::cout << "CastorChannelQualityMonitor::setup (start)" << std::endl;

  ////---- initialize array aboveNoisyThreshold and belowDThreshold
  for (int row=0; row<14; row++) {
    for (int col=0; col<16; col++){
        aboveNoisyThreshold[row][col] = 0;
        belowDThreshold[row][col]     = 0;
        energyArray[row][col]         = 0;
        aboveThr[row][col]         = 0;
    }
  }

  ////---- initialize the event counter and other counters  
  ievt_=0; counter1=0; counter2=0;  wcounter1=0.; wcounter2=0.; 
  ////---- initialize the fraction of good channels
  fraction=0.; 
  overallStatus->Fill(fraction); reportSummary->Fill(fraction); 
  ////---- initialize status and number of good channels
  status = -99; numOK = 0; 
  ////---- initialize recHitPresent
  iRecHit=false;
  ////---- initialize averageEnergy
  averageEnergy=0.;

  if(fVerbosity>0) std::cout << "CastorChannelQualityMonitor::setup (end)" << std::endl;

  return;
}




//==========================================================//
//================== processEvent ==========================//
//==========================================================//
void CastorChannelQualityMonitor::processEvent(const CastorRecHitCollection& castorHits){

  if(fVerbosity>0) std::cout << "CastorChannelQualityMonitor::processEvent !!!" << std::endl;

  if(!m_dbe) { 
    if(fVerbosity>0) std::cout <<"CastorChannelQualityMonitor::processEvent => DQMStore is not instantiated !!!"<<std::endl;  
    return; 
  }

  if(fVerbosity>0){
     std::cout << "CastorChannelQualityMonitor: Noisy Threshold is set to: "<< nThreshold_ << std::endl;
     std::cout << "CastorChannelQualityMonitor: Dead Threshold is set to: " << dThreshold_ << std::endl;
    }

  castorModule = -1;  castorSector = -1; castorEnergy = -1.; 
  
  if (castorHits.size()>0) iRecHit=true;
  else iRecHit=false;
 
 
 ////---- loop over RecHits 
 for(CastorRecHitCollection::const_iterator recHit = castorHits.begin(); recHit != castorHits.end(); ++recHit){

    HcalCastorDetId CastorID = HcalCastorDetId(recHit->id());
    if(fVerbosity>0) std::cout << "Castor ID = " << CastorID << std::endl;
    CastorRecHitCollection::const_iterator rh = castorHits.find(CastorID);
    ////---- obtain module, sector and energy of a rechit
    castorModule = CastorID.module();
    castorSector = CastorID.sector();
    castorEnergy = rh->energy();
    //if(ievt_ % 1000 == 0) std::cout << "==> module=" << module << " sector=" << sector << " energy= "<< energy << std::endl;
    iRecHit=true; 
   
    ////----  fill the arrays 
    ////----  need to substruct 1 since module and sector start from 1
    if(castorEnergy > nThreshold_) ++aboveNoisyThreshold[castorModule-1][castorSector-1];
    if(castorEnergy < dThreshold_) ++belowDThreshold[castorModule-1][castorSector-1];
    ////---- use this threshold for the moment
    if(castorEnergy>1) { //if(castorEnergy>0) {
      ++aboveThr[castorModule-1][castorSector-1]; 
      energyArray[castorModule-1][castorSector-1]=energyArray[castorModule-1][castorSector-1]+castorEnergy;
    }

   }////---- end of the loop




 ////---- increment here 
  ievt_++;


 ////---------------------------------------------
 ////---- update reportSummarymap every 500 events
 ////----------------------------------------------

  // if( (ievt_ == 25 || ievt_ % 500 == 0) && iRecHit ) {
  // no particular event selection done 
   if( iRecHit ) {
   
   status = -99;  numOK = 0; 

     ////---- reset the reportSummaryMap 
    // if(offline_) reportSummaryMap->Reset();


 ////---- check each channel 
 for (int sector=0; sector<16; sector++){
  for (int module=0; module<14; module++){
  

   if(averageEnergyMethod_){

     if(aboveThr[module][sector] >0)
     averageEnergy= energyArray[module][sector]/double(aboveThr[module][sector]); // calculate the average energy in each channel
     else averageEnergy=0;
 
     ////---- evaluation
     if( averageEnergy >  nThreshold_ )  status= 0;   ////--- channel is noisy 
     if( averageEnergy < dThreshold_  ) { status= -1;  ////--- channel is dead 
          if(fVerbosity>0) std::cout << "!!! dChannels ===> module="<< module+1 << " sector="<< sector+1 << std::endl;
	  }
     if( averageEnergy <  nThreshold_ && averageEnergy > dThreshold_ )  status= 1; ////---- channel is good

     if(fVerbosity>0)
      std::cout << "===> module="<< module+1 << " sector="<< sector+1 <<" *** average Energy=" 
	   << averageEnergy << " => energy=" << energyArray[module][sector] << " events="
           << ievt_ << " aboveThr="<< double(aboveThr[module][sector]) <<std::endl;
    }


   else{
      //-- look at the arrays
      counter1= aboveNoisyThreshold[module][sector]; //counter1 defines how many times the energy was above a noisythreshold
      counter2= belowDThreshold[module][sector];  //counter2 defines how many times the energy was below a dthreshold
      wcounter1= double(counter1)/double(ievt_);
      wcounter2= double(counter2)/double(ievt_);
      ////---- evaluation
      if( wcounter1 > 0.85 )  status= 0; ////--- channel is noisy (85% of cases energy was above NoisyThreshold) 
      if( wcounter2 > 0.85 ) {status= -1;  ////--- channel is dead (85% of cases energy was below dThreshold)
         if(fVerbosity>0) std::cout << "!!! dChannels ===> module="<< module+1 << " sector="<< sector+1 << std::endl; 
	}
      if( wcounter1 < 0.85 && wcounter2 <  0.85 ) status= 1; ////---- channel is good

      if(fVerbosity>0)
        std::cout << "===> module="<< module+1 << " sector="<< sector+1 <<" *** counter1=" 
        << counter1 << " => counter2=" << counter2 << " events="<< ievt_ 
        << " wcounter1=" << wcounter1  << " wcounter2=" <<  wcounter2
        << " *** ==> STATUS=" << status <<std::endl;      
    }

   ////---- fill reportSummaryMap  
   // reportSummaryMap->Fill(module,sector,status);
   reportSummaryMap->getTH2F()->SetBinContent(module+1,sector+1,double(status));
   if (status == 1) numOK++;

      }
    }
 ////--- calculate the fraction of good channels and fill it in
  fraction=double(numOK)/224;
  overallStatus->Fill(fraction); reportSummary->Fill(fraction); 
  }
  
  
   return;
} 
 
