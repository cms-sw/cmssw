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
////---- revision: 01.06.2010 (Dima Volyanskyy) 
////----- last revision: 31.05.2011 (Panos Katsas)  


//==================================================================//
//======================= Constructor ==============================//
//==================================================================//
CastorPSMonitor::CastorPSMonitor() {
  doPerChannel_ = true;
  ievt_=0;
  firstRegionThreshold_=0.;
  secondRegionThreshold_=0.;
  status=-99;
  statusRS=-99;
  statusSaturated=-99;
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
  thirdRegionThreshold_ = ps.getUntrackedParameter<double>("thirdRegionThreshold", 300);
  saturatedThreshold_   = ps.getUntrackedParameter<double>("saturatedThreshold", 0.05); //-- fraction of events in which chargeTS > 127 ADC
  offline_              = ps.getUntrackedParameter<bool>("OfflineMode", false); 

  if(fVerbosity>0) std::cout << "CastorPSMonitor::setup (start)" << std::endl;
    
  ievt_=0;  firstTime_ = true;
  firstRegionThreshold_=0.;
  secondRegionThreshold_=0.;
  numOK = 0; 
  fraction=0.; 

  ////---- initialize the arrays sumDigiForEachChannel, saturatedMap
  for (int row=0; row<14; row++){
    for (int col=0; col<16; col++){
        sumDigiForEachChannel[row][col] = 0;
        saturatedMap [row][col] = 0;
    }
  }

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
    sprintf(name,"Castor Pulse Shape for sector=%d (in all 14 modules)",i+1);      
    PSsector[i] =  m_dbe->book1D(name,name,140,-0.5,139.5);
    }

    ////---- digi occupancy map
    DigiOccupancyMap = m_dbe->book2D("CASTOR Digi Occupancy Map","CASTOR Digi Occupancy Map",14,0.0,14.0,16,0.0,16.0);
    ////---- channel summary map
    ChannelSummaryMap = m_dbe->book2D("CASTOR Digi ChannelSummaryMap","CASTOR Digi ChannelSummaryMap",14,0.0,14.0,20,0.0,20.0); // put 20 instead of 16 to get some space for the legend

    ////---- saturation summary map: 
    ////---- distinguish between channels with no saturation, saturated in more than 5% events, saturated in less than 5% of events 
    SaturationSummaryMap = m_dbe->book2D("CASTOR Digi SaturationSummaryMap","CASTOR Digi SaturationSummaryMap",14,0.0,14.0,20,0.0,20.0); // put 20 instead of 16 to get some space for the legend



    ////---- create Digi based reportSummaryMap
    m_dbe->setCurrentFolder(rootFolder_+"EventInfo");
    reportSummary    = m_dbe->bookFloat("reportSummary");
    reportSummaryMap = m_dbe->book2D("reportSummaryMap","CASTOR reportSummaryMap",14,0.0,14.0,16,0.0,16.0);
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
    if(fVerbosity>0) std::cout << "CastorPSMonitor::setup - NO DQMStore service" << std::endl; 
  }

  if(fVerbosity>0) std::cout << "CastorPSMonitor::setup (end)" << std::endl;

  return;
}

//==========================================================//
//================== processEvent ==========================//
//==========================================================//

void CastorPSMonitor::processEvent(const CastorDigiCollection& castorDigis, const CastorDbService& conditions, std::vector<HcalGenericDetId> listEMap, int iBunch, float PedSigmaInChannel[14][16])
 {
  
     
  if(fVerbosity>0) std::cout << "==>CastorPSMonitor::processEvent !!!" << std::endl;
 
  if(!m_dbe) { 
    if(fVerbosity>0) std::cout <<"CastorPSMonitor::processEvent => DQMStore not instantiated !!!"<<std::endl;  
    return; 
  }
 
  meEvt_->Fill(ievt_); 

  ////---- increment here
  ievt_++;

  status = -99; statusRS = -99; statusSaturated=-99;

  ////---- get Castor Shape from the Conditions
  const CastorQIEShape* shape = conditions.getCastorShape();
     
  if(firstTime_)     
    {
      //===> show the array of sigmas
      for (int i=0; i<14; i++){
        for (int k=0; k<16; k++){
	if(fVerbosity>0)  std::cout<< "module:"<<i+1<< " sector:"<<k+1<< " Sigma=" <<   PedSigmaInChannel[i][k] << std::endl;   
	}
      }
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
        bool saturated = false;
	
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
            double sumDigi=0.;  double sumDigiADC=0.; int bxTS=-9999; saturated = false;

	    ////---- loop over Time Sclices
	    for(int ts = firstTS; ts != lastTS+1; ts++)
	      {
	         if (firstDigi) {
		    bxTS = (iBunch+ts-1-digi.presamples());
		    if ( bxTS < 0 ) bxTS += 3563;
		    bxTS = ( bxTS % 3563 ) + 1;
		    if(fVerbosity>0) std::cout << "!!! " << bxTS << " " << iBunch <<" "<< ts << " " << digi.presamples() << std::endl;
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

		  // PK: do not normalize histograms
		  //  PSsector[bunch_it->detid.sector()-1]->Fill(10*(bunch_it->detid.module()-1)+ts, bunch_it->tsfC[ts]/double(ievt_)); 
                  PSsector[bunch_it->detid.sector()-1]->Fill(10*(bunch_it->detid.module()-1)+ts, bunch_it->tsfC[ts]);
 
                  ////---- sum the signal over all TS in fC
                  sumDigi +=  bunch_it->tsfC[ts]; //std::cout<< " signal(fC) in TS:"<<ts << " =" << bunch_it->tsfC[ts] << std::endl; 	   
		  ////---- sum the signal over all TS in ADC
                  sumDigiADC +=   bunch_it->tsAdc[ts]; //std::cout<< " signal(ADC) in TS:"<<ts << " =" << bunch_it->tsAdc[ts] << std::endl; 

		  ////---- check whether the channel is saturated
		  if(bunch_it->tsAdc[ts]>126.95) {
                    saturated = true;
		  if(fVerbosity>0) 
                  std::cout<< "WARNING: ==> Module:" << bunch_it->detid.module() << " Sector:" << bunch_it->detid.sector() << " SATURATED !!! in TS:"<< ts <<std::endl;   
                 }

	      } //-- end of the loop for time slices
          
            ////---- fill  the array  sumDigiForEachChannel
            sumDigiForEachChannel[bunch_it->detid.module()-1][bunch_it->detid.sector()-1] += sumDigi;

	    ////---- fill the array saturatedMap
            if(saturated) saturatedMap[bunch_it->detid.module()-1][bunch_it->detid.sector()-1] += 1;

            ////---- fill the digi occupancy map 
	    DigiOccupancyMap->Fill(bunch_it->detid.module()-1,bunch_it->detid.sector()-1, double(sumDigi/10)); //std::cout<< "=====> sumDigi=" << sumDigi << std::endl;

            
            if(fVerbosity>0){
             std::cout<< "==> Module:" << bunch_it->detid.module() << " Sector:" << bunch_it->detid.sector() << std::endl; 
             std::cout<< "==> Total charge in fC:" << sumDigi << std::endl;  
             std::cout<< "==> Total charge in ADC:" << sumDigiADC << std::endl;  
	    }
	 
	   firstDigi = false;  
    } //-- end of the loop over digis 





  ////--------------------------------------------------------------------
 ////---- define and update digi based reportSummarymap every 500 events
 ////--------------------------------------------------------------------


	// if( ievt_ == 25 || ievt_ % 500 == 0 ) {  // no event selection - get all events

    numOK = 0;

    ////---- check each channel 
     for (int sector=0; sector<16; sector++){
       for (int module=0; module<14; module++){
  
     ////---- get the thresholds
     firstRegionThreshold_ = (-1)*(numberSigma_*PedSigmaInChannel[module][sector]);
     secondRegionThreshold_ = numberSigma_*PedSigmaInChannel[module][sector] ;
    

 ////---- channel is dead => no DAQ
 if(double(sumDigiForEachChannel[module][sector]/(10*ievt_)) < firstRegionThreshold_ )  
   { status = -1.; statusRS=0.; }
              
 ////---- channel is OK => just pedestal in it
 if(double(sumDigiForEachChannel[module][sector]/(10*ievt_)) > firstRegionThreshold_ && double(sumDigiForEachChannel[module][sector]/(10*ievt_)) < secondRegionThreshold_ ) 
   { status = 0.25; statusRS=0.95; }
  
 ////---- channel is OK -> signal in it
 if(double(sumDigiForEachChannel[module][sector]/(10*ievt_)) > secondRegionThreshold_ && double(sumDigiForEachChannel[module][sector]/(10*ievt_))< thirdRegionThreshold_ ) 
   { status = 1.; statusRS=1.0; }

 //---- leave it out for the time being
 ////---- channel is noisy
 // if(double(sumDigiForEachChannel[module][sector]/(10*ievt_)) > thirdRegionThreshold_ ) 
 // { status = -0.25; statusRS=0.88 ; }
   
 //-- define the fraction of saturated events for a particular channel
 double fractionSaturated =  double(saturatedMap[module][sector])/double(ievt_) ;
 ////---- channel is saturated (in more than saturatedThreshold_ events) 
 if(fVerbosity>0) std::cout<< "==> module: " << module << " sector: " << sector << " ==> N_saturation:" << saturatedMap[module][sector] << " events:"<< ievt_ << " fraction:" << fractionSaturated << std::endl;

 if( fractionSaturated > saturatedThreshold_ ) 
   { status = -0.25; statusRS=0.88 ; statusSaturated=-1.0; }

 ////---- channels is saturated at least once
 if( saturatedMap[module][sector] > 0 && fractionSaturated <  saturatedThreshold_  ) 
   { statusSaturated= 0; }

 ////---- channel was not in saturation at all
 if( saturatedMap[module][sector] == 0 ) 
   { statusSaturated= 1; }

   ////---- fill the ChannelSummaryMap
   ChannelSummaryMap->getTH2F()->SetBinContent(module+1,sector+1,status);

   ////---- fill the reportSummaryMap
   reportSummaryMap->getTH2F()->SetBinContent(module+1,sector+1,statusRS);

  ////---- fill the SaturationSummaryMap
   SaturationSummaryMap->getTH2F()->SetBinContent(module+1,sector+1,double(statusSaturated));

  ////---- calculate the number of good channels
   if ( statusRS > 0.9) numOK++;

       } //-- end of the loop over the modules
     } //-- end of the loop over the sectors

    ////--- calculate the fraction of good channels and fill it in
    fraction=double(numOK)/224;
    overallStatus->Fill(fraction); reportSummary->Fill(fraction); 

    //  } //-- end of if for the number of events // update ( PK ):   


    ////---- set 99 for these (space used for the legend)
      for (int sector=16; sector<20; sector++){
        for (int module=0; module<14; module++){
      ChannelSummaryMap->getTH2F()->SetBinContent(module+1,sector+1,99);
        }
    }
    ////---- set 99 for these (space used for the legend)
      for (int sector=16; sector<20; sector++){
        for (int module=0; module<14; module++){
	  SaturationSummaryMap->getTH2F()->SetBinContent(module+1,sector+1, 99);
        }
    }





 } //-- end of the if castDigi
       
  else { if(fVerbosity>0) std::cout<<"CastorPSMonitor::processEvent NO Castor Digis !!!"<<std::endl; }

      
   if (showTiming) { 
      cpu_timer.stop(); std::cout << " TIMER::CastorPS -> " << cpu_timer.cpuTime() << std::endl; 
      cpu_timer.reset(); cpu_timer.start();  
    }
 
  
  return;
}











