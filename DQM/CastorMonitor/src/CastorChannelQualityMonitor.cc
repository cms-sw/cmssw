#include "DQM/CastorMonitor/interface/CastorChannelQualityMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//***************************************************//
//********** CastorChannelQualityMonitor *******************//
//********** Author: Dmytro Volyanskyy   ************//
//********** Date  : 04.03.2010 (first version) ******// 
//***************************************************//
////---- energy and time of Castor RecHits 
////---- last revision: 05.03.2010 


//==================================================================//
//======================= Constructor ==============================//
//==================================================================//
CastorChannelQualityMonitor::CastorChannelQualityMonitor(){
 ievt_=0;

 
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

   offline_ = ps.getUntrackedParameter<bool>("OfflineMode", false); 
   nThreshold_ = ps.getUntrackedParameter<double>("nThreshold", 100.0);
   dThreshold_ = ps.getUntrackedParameter<double>("dThreshold", 1.0);

  if ( m_dbe !=NULL ) {    
    ////---- create ReportSummary Map 
    m_dbe->setCurrentFolder(rootFolder_+"EventInfo");
    reportSummary    = m_dbe->bookFloat("reportSummary");
    reportSummaryMap = m_dbe->book2D("reportSummaryMap","reportSummaryMap",14,0.0,14.0,16,0.0,16.0);
    if(offline_){
      h_reportSummaryMap =reportSummaryMap->getTH2F();
      h_reportSummaryMap->SetOption("textcolz");
      h_reportSummaryMap->GetXaxis()->SetTitle("module");
      h_reportSummaryMap->GetYaxis()->SetTitle("sector");
    }
    
    m_dbe->setCurrentFolder(rootFolder_+"EventInfo/reportSummaryContents");
    overallStatus = m_dbe->bookFloat("fraction of good channels");
   } 

  else{
  if(fVerbosity>0) cout << "CastorChannelQualityMonitor::setup - NO DQMStore service" << endl; 
 }

  // baseFolder_ = rootFolder_+"CastorChannelQualityMonitor";

  if(fVerbosity>0) cout << "CastorChannelQualityMonitor::setup (start)" << endl;

  ////---- initialize array aboveNoisyThreshold and aboveDThreshold
  for (int row=0; row<14; row++) {
    for (int col=0; col<16; col++){
        aboveNoisyThreshold[row][col] = 0;
         aboveDThreshold[row][col] = 0;
    }
  }

  ////---- initialize the event counter  
  ievt_=0;
  ////---- initialize the fraction of good channels
  fraction=0; 
  overallStatus->Fill(fraction); reportSummary->Fill(fraction); 
  ////---- initialize status and number of good channels
  status = -99; numOK = 0; 
  ////---- initialize recHitPresent
  iRecHit=false;
  
  if(fVerbosity>0) cout << "CastorChannelQualityMonitor::setup (end)" << endl;

  return;
}




//==========================================================//
//================== processEvent ==========================//
//==========================================================//
void CastorChannelQualityMonitor::processEvent(const CastorRecHitCollection& castorHits){

  if(fVerbosity>0) cout << "CastorChannelQualityMonitor::processEvent !!!" << endl;

  if(!m_dbe) { 
    if(fVerbosity>0) cout <<"CastorChannelQualityMonitor::processEvent => DQMStore is not instantiated !!!"<<endl;  
    return; 
  }

  if(fVerbosity>0){
     cout << "CastorChannelQualityMonitor: Noisy Threshold is set to: "<< nThreshold_ << endl;
     cout << "CastorChannelQualityMonitor: Dead Threshold is set to: " << dThreshold_ << endl;
    }

  module = -1;  sector = -1; energy = -1.; 
  
  if (castorHits.size()>0) iRecHit=true;
  else iRecHit=false;
  
 ////---- loop over RecHits 
 for(CastorRecHitCollection::const_iterator recHit = castorHits.begin(); recHit != castorHits.end(); ++recHit){

    HcalCastorDetId CastorID = HcalCastorDetId(recHit->id());
    if(fVerbosity>0) cout << "Castor ID = " << CastorID << std::endl;
    CastorRecHitCollection::const_iterator rh = castorHits.find(CastorID);
    ////---- obtain module, sector and energy of a rechit
    module = CastorID.module();
    sector = CastorID.sector();
    energy = rh->energy();
    //cout<< "energy= "<< energy << endl;
    iRecHit=true;
 
    int iN=0; int iD=0;
   ////----  fill the arrays
    if(energy > nThreshold_){ 
       //++aboveNoisyThreshold[module-1][sector-1];
       iN=aboveNoisyThreshold[module-1][sector-1]; aboveNoisyThreshold[module-1][sector-1]=iN+1;
    }

  if(energy < dThreshold_){
      // ++aboveDThreshold[module-1][sector-1];
       iD=aboveDThreshold[module-1][sector-1]; aboveDThreshold[module-1][sector-1]=iD+1;
    }
 
  
  }

 ////---- increment here 
  ievt_++;

 ////---- update reportSummarymap every 200 events
 if( (ievt_ == 25 || ievt_ % 200 == 0) && iRecHit ) {
   
   cout<< "===> start filling the reportSummaryMap!" << " dThreshold is set to " 
       << dThreshold_ << " nThreshold is set to "<< nThreshold_ << endl;
   
   ////--- initialize these here
   status = -99;  numOK = 0;  counter1=0; counter2=0; 
   counter1_=0.; counter2_=0.;

   ////---- reset the reportSummaryMap 
   // if(offline_) reportSummaryMap->Reset();
  
   ////---- look at the values in the arrays
  for (int module=0; module<14; module++){
    for (int sector=0; sector<16; sector++){
   //-- look at the arrays
   counter1= aboveNoisyThreshold[module][sector]; //counter1 defines how many times the energy was above a noisythreshold
   counter2= aboveDThreshold[module][sector];  //counter2 defines how many times the energy was below a dthreshold
   counter1_ = double(counter1)/double(ievt_); // get the fraction of events when it was above a noisythreshold
   counter2_ = double(counter2)/double(ievt_); // get the fraction of events when it was below a dThreshold
  
    if( counter1_ > 0.85 ) ////--- channel is noisy (85% of cases energy was above NoisyThreshold)
      status= 0;   
  
    if( counter2_ > 0.85 )   ////--- channel is dead (85% of cases energy was below dThreshold)
      {status= -1; cout << "!!! dChannels ===> module="<< module << " sector="<< sector << endl;}
 
   if( counter1_ < 0.85 && counter2_ <  0.85 ) ////---- channel is good
        status= 1;

   if(fVerbosity>0)
   cout << "===> module="<< module << " sector="<< sector <<" *** < counter1=" 
        << counter1 << " => counter2=" << counter2 << " events="<< ievt_ 
        << " > ==> STATUS=" << status <<endl;

   ////---- fill reportSummaryMap  
   // reportSummaryMap->Fill(module,sector,status);
   reportSummaryMap->getTH2F()->SetBinContent(module+1,sector+1,status);
   if (status == 1) numOK++;
      }
    }
 ////--- calculate the fraction of good channels and fill it in
  fraction=double(numOK)/224;
  overallStatus->Fill(fraction); reportSummary->Fill(fraction); 
  }
  
  
   return;
} 
 
