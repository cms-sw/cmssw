#include "DQM/CastorMonitor/interface/CastorPSMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "CondFormats/CastorObjects/interface/CastorPedestals.h"
#include "CondFormats/CastorObjects/interface/CastorPedestalWidths.h"
#include "CondFormats/CastorObjects/interface/CastorQIECoder.h"
#include "CondFormats/CastorObjects/interface/CastorQIEData.h"
#include "CondFormats/CastorObjects/interface/CastorQIEShape.h"
#include "CondFormats/CastorObjects/interface/CastorElectronicsMap.h"
#include "CondFormats/CastorObjects/interface/AllObjects.h"

#include "CalibFormats/CastorObjects/interface/CastorDbRecord.h"
#include "CalibFormats/CastorObjects/interface/CastorDbService.h"
#include "CalibFormats/CastorObjects/interface/CastorCalibrations.h"
#include "CalibFormats/CastorObjects/interface/CastorCalibrationWidths.h"

#include "CalibCalorimetry/CastorCalib/interface/CastorDbASCIIIO.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"

#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"
#include "DataFormats/HcalRecHit/interface/CastorRecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h" //-- CastorRecHitCollection

//***************************************************//
//********** CastorPSMonitor:   *******************//
//********** Author: D.Volyanskyy/I.Katkov  *********//
//********** Date  : 03.03.2010 (first version) ******// 
//***************************************************//
////---- pulse shape monitor + digi occupancy and quality
////---- last revision: 01.06.2010 

//==================================================================//
//======================= Constructor ==============================//
//==================================================================//
CastorPSMonitor::CastorPSMonitor() {
  doPerChannel_ = true;
  ievt_=0;
  firstRegionThreshold_=0.;
  secondRegionThreshold_=0.;
  status=-99;

}

//==================================================================//
//======================= Destructor ==============================//
//==================================================================//
CastorPSMonitor::~CastorPSMonitor(){
}

void CastorPSMonitor::reset(){
}

//==========================================================//
//========================= setup ==========================//
//==========================================================//

void CastorPSMonitor::setup(const edm::ParameterSet& ps, DQMStore* dbe){
  
  CastorBaseMonitor::setup(ps,dbe);
  baseFolder_ = rootFolder_+"CastorPSMonitor";

  ////---- get these parameters from python cfg  
  numberSigma_ = ps.getUntrackedParameter<double>("numberSigma", 1.5);
  thirdRegionThreshold_ = ps.getUntrackedParameter<double>("thirdRegionThreshold", 100);
  offline_             = ps.getUntrackedParameter<bool>("OfflineMode", false); 

  if(fVerbosity>0) cout << "CastorPSMonitor::setup (start)" << endl;
    
  ievt_=0;  firstTime_ = true;
  firstRegionThreshold_=0.;
  secondRegionThreshold_=0.;
  numOK = 0; 
  fraction=0.; 

  ////---- initialize the array sumDigiForEachChannel
  for (int row=0; row<14; row++) 
    for (int col=0; col<16; col++)
        sumDigiForEachChannel[row][col] = 0;


  if ( m_dbe !=NULL ) {    
    m_dbe->setCurrentFolder(baseFolder_);
    
    ////---- book MonitorElements
    meEvt_ = m_dbe->bookInt("PS Event Number"); 
    castorDigiHists.meDigi_pulseBX = m_dbe->book1D("CASTOR average pulse in bunch crossings","CASTOR average pulse in bunch crossings", 3600,  -0.5, 3600);
    TH1F* h_meDigi_pulseBX = castorDigiHists.meDigi_pulseBX->getTH1F();
    h_meDigi_pulseBX->GetXaxis()->SetTitle("orbit");

    //---- Pulse Shape per sector
    char name[1024];
    for(int i=0; i<16; i++){
    sprintf(name,"Castor Pulse Shape for sector=%d (in all 14 modules)",i);      
    PSsector[i] =  m_dbe->book1D(name,name,140,-0.5,139.5);
    }

    ////---- digi occupancy map
    DigiOccupancyMap = m_dbe->book2D("CASTOR Digi Occupancy Map","CASTOR Digi Occupancy Map",14,0.0,14.0,16,0.0,16.0);
   
    ////---- create Digi based reportSummaryMap
    m_dbe->setCurrentFolder(rootFolder_+"EventInfo");
    reportSummary    = m_dbe->bookFloat("reportSummary");
    reportSummaryMap = m_dbe->book2D("reportSummaryMap","ChannelSummaryMap",14,0.0,14.0,16,0.0,16.0);
    if(offline_){
      h_reportSummaryMap =reportSummaryMap->getTH2F();
      h_reportSummaryMap->SetOption("textcolz");
      h_reportSummaryMap->GetXaxis()->SetTitle("module");
      h_reportSummaryMap->GetYaxis()->SetTitle("sector");
    }
    m_dbe->setCurrentFolder(rootFolder_+"EventInfo/reportSummaryContents");
    overallStatus = m_dbe->bookFloat("fraction of good channels");
    overallStatus->Fill(fraction); reportSummary->Fill(fraction); 
  } 

  else{
    if(fVerbosity>0) cout << "CastorPSMonitor::setup - NO DQMStore service" << endl; 
  }

  if(fVerbosity>0) cout << "CastorPSMonitor::setup (end)" << endl;

  return;
}

//==========================================================//
//================== processEvent ==========================//
//==========================================================//

void CastorPSMonitor::processEvent(const CastorDigiCollection& castorDigis, const CastorDbService& conditions, vector<HcalGenericDetId> listEMap, int iBunch, float PedSigmaInChannel[14][16])
 {
  
     
  if(fVerbosity>0) cout << "==>CastorPSMonitor::processEvent !!!" << endl;
 
  if(!m_dbe) { 
    if(fVerbosity>0) cout <<"CastorPSMonitor::processEvent => DQMStore not instantiated !!!"<<endl;  
    return; 
  }
 
  meEvt_->Fill(ievt_); 

  ////---- increment here
  ievt_++;

  status = -99;

  ////---- get Castor Shape from the Conditions
  const CastorQIEShape* shape = conditions.getCastorShape();
     
  if(firstTime_)     
    {
      //===> show the array of sigmas
      for (int i=0; i<14; i++)
        for (int k=0; k<16; k++)
	  cout<< "module:"<<i+1<< " sector:"<<k+1<< " Sigma=" <<   PedSigmaInChannel[i][k] << endl;   
      
      for (std::vector<HcalGenericDetId>::const_iterator it = listEMap.begin(); it != listEMap.end(); it++)
	{     
	  HcalGenericDetId mygenid(it->rawId());
	  if(mygenid.isHcalCastorDetId())
	    {
	      NewBunch myBunch;
	      HcalCastorDetId chanid(mygenid.rawId());
	      myBunch.detid = chanid;
	      myBunch.usedflag = false;
	      std::string type;
	      type = "CASTOR";
	      for(int i = 0; i != 20; i++)
		{
		  myBunch.tsCapId[i] = 0;
		  myBunch.tsAdc[i] = 0;
		  myBunch.tsfC[i] = 0.0;
		}
	      Bunches_.push_back(myBunch);
	    }
	}

      firstTime_ = false;
    }
  
  
      if(castorDigis.size()>0) {
	////---- consider time slices between 0 and 9
	int firstTS = 0; 
	int lastTS  = 9;   
	std::vector<NewBunch>::iterator bunch_it;
	int numBunches = 0;
	bool firstDigi = true;

	
	////---- loop over Digi Colllection
	for(CastorDigiCollection::const_iterator j = castorDigis.begin(); j != castorDigis.end(); j++)
	  {
	    
	    const CastorDataFrame digi = (const CastorDataFrame)(*j);
	    if ( lastTS+1 > digi.size() ) lastTS = digi.size()-1;
	    for(bunch_it = Bunches_.begin(); bunch_it != Bunches_.end(); bunch_it++)
	      if(bunch_it->detid.rawId() == digi.id().rawId()) break;
	    bunch_it->usedflag = true;
	    
	    numBunches++;
	    //
	    //---- Skip noisy channels present in 2009 beam data:
	    // if ( (bunch_it->detid.sector() == 16 && bunch_it->detid.module() == 6)  ||
	    //	 (bunch_it->detid.sector() ==  3 && bunch_it->detid.module() == 8)  ||
	    //	 (bunch_it->detid.sector() ==  8 && bunch_it->detid.module() == 8) ) continue;
	    //

            if ( lastTS+1 > digi.size() ) lastTS = digi.size()-1;
	    
	    ////---- get Conditions from the CondDB
	    const CastorCalibrations& calibrations=conditions.getCastorCalibrations(digi.id().rawId());

	    ////---- intialize these here
            double sumDigi=0.; int bxTS=-9999;

	    ////---- loop over Time Sclices
	    for(int ts = firstTS; ts != lastTS+1; ts++)
	      {
	         if (firstDigi) {
		    bxTS = (iBunch+ts-1-digi.presamples());
		    if ( bxTS < 0 ) bxTS += 3563;
		    bxTS = ( bxTS % 3563 ) + 1;
		    if(fVerbosity>0) cout << "!!! " << bxTS << " " << iBunch <<" "<< ts << " " << digi.presamples() << endl;
		  }
		
                  const CastorQIECoder* coder = conditions.getCastorCoder(digi.id().rawId());
		  bunch_it->tsCapId[ts] = digi.sample(ts).capid();
		  bunch_it->tsAdc[ts] = digi.sample(ts).adc();
		
                  ////---- get charge in fC in each channel and CapID
                  double charge_fC = coder->charge(*shape, digi.sample(ts).adc(), digi.sample(ts).capid());
                
		  ////---- substract pedestal value
                  bunch_it->tsfC[ts] = charge_fC - calibrations.pedestal(digi.sample(ts).capid());
		  ////---- fill pulse shape vs BX number 
		  castorDigiHists.meDigi_pulseBX->Fill(static_cast<double>(bxTS),(bunch_it->tsfC[ts])/224.);
		  ////---- fill pulse shape in sectors 
                  PSsector[bunch_it->detid.sector()-1]->Fill(10*(bunch_it->detid.module()-1)+ts, bunch_it->tsfC[ts]); 

                  ////---- sum the signal over all TS
                  sumDigi +=  bunch_it->tsfC[ts]; //cout<< " signal in TS:"<<ts << " =" << bunch_it->tsfC[ts] << endl; 	   
	     
	      } //-- end of the loop for time slices
    

            ////---- fill the digi occupancy map 
	    DigiOccupancyMap->Fill(bunch_it->detid.module()-1,bunch_it->detid.sector()-1, double(sumDigi/10)); //cout<< "=====> sumDigi=" << sumDigi << endl;

             ////---- fill  the array  sumDigiForEachChannel
            sumDigiForEachChannel[bunch_it->detid.module()-1][bunch_it->detid.sector()-1] += sumDigi;

	 
	   firstDigi = false;  
    } //-- end of the loop over digis 





  ////--------------------------------------------------------------------
 ////---- define and update digi based reportSummarymap every 500 events
 ////--------------------------------------------------------------------
 if( ievt_ == 25 || ievt_ % 500 == 0 ) {

    numOK = 0;

    ////---- check each channel 
     for (int sector=0; sector<16; sector++){
       for (int module=0; module<14; module++){
  
     ////---- get the thresholds
     firstRegionThreshold_ = (-1)*(numberSigma_*PedSigmaInChannel[module][sector]);
     secondRegionThreshold_ = numberSigma_*PedSigmaInChannel[module][sector] ;
    

 ////---- channel is dead => no DAQ
 if(double(sumDigiForEachChannel[module][sector]/(10*ievt_)) < firstRegionThreshold_ )  
    status = -1.;
              
 ////---- channel is OK => just pedestal in it
 if(double(sumDigiForEachChannel[module][sector]/(10*ievt_)) > firstRegionThreshold_ && double(sumDigiForEachChannel[module][sector]/(10*ievt_)) < secondRegionThreshold_ ) 
    status = 0.25;
  
 ////---- channel is OK -> signal in it
 if(double(sumDigiForEachChannel[module][sector]/(10*ievt_)) > secondRegionThreshold_ && double(sumDigiForEachChannel[module][sector]/(10*ievt_))< thirdRegionThreshold_ ) 
    status = 1.;

 ////---- channel is noisy 
 if(double(sumDigiForEachChannel[module][sector]/(10*ievt_)) > thirdRegionThreshold_ ) 
    status = -0.25;   	      
    
   ////---- fill the reportSummaryMap
   reportSummaryMap->getTH2F()->SetBinContent(module+1,sector+1,status);
   if ( status > 0) numOK++;
       }
    }
    ////--- calculate the fraction of good channels and fill it in
    fraction=double(numOK)/224;
    overallStatus->Fill(fraction); reportSummary->Fill(fraction); 
  } //

 } //-- end of the if castDigi
       
  else { if(fVerbosity>0) cout<<"CastorPSMonitor::processEvent NO Castor Digis !!!"<<endl; }

      
   if (showTiming) { 
      cpu_timer.stop(); cout << " TIMER::CastorPS -> " << cpu_timer.cpuTime() << endl; 
      cpu_timer.reset(); cpu_timer.start();  
    }
 
  
  return;
}











