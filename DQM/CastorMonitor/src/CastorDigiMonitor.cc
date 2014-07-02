#include "DQM/CastorMonitor/interface/CastorDigiMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//****************************************************//
//********** CastorDigiMonitor: ******************//
//********** Author: Dmytro Volyanskyy   *************//
//********** Date  : 29.08.2008 (first version) ******// 
//****************************************************//
////---- digi values in Castor r/o channels 
////      revision: 31.05.2011 (Panos Katsas) to remove selecting N events for filling the histograms
//// last revision: 13.03.2014 (Vladimir Popov) QIE validation 2d-histogram
//==================================================================//
//======================= Constructor ==============================//
//==================================================================//
CastorDigiMonitor::CastorDigiMonitor()
  {

  doPerChannel_ = false;

  }


//==================================================================//
//======================= Destructor ===============================//
//==================================================================//
CastorDigiMonitor::~CastorDigiMonitor()
  {

  }


//==================================================================//
//=========================== reset  ===============================//
//==================================================================//
void CastorDigiMonitor::reset()
  {
  
  }

//==================================================================//
//=========================== setup  ===============================//
//==================================================================//
void CastorDigiMonitor::setup(const edm::ParameterSet& ps, DQMStore* dbe)
  {

  CastorBaseMonitor::setup(ps,dbe);

  if(fVerbosity>0) { std::cout << "CastorDigiMonitor::setup (start)" << std::endl; }

  //set base folder
  baseFolder_ = rootFolder_+"CastorDigiMonitor";

  doPerChannel_ = ps.getUntrackedParameter<bool>("DigiPerChannel", false);
  doFCpeds_ = ps.getUntrackedParameter<bool>("DigiInFC", true);

  ievt_=0;

  outputFile_ = ps.getUntrackedParameter<std::string>("PedestalFile", "");
  if ( outputFile_.size() != 0 )
  	{
	if(fVerbosity>0) { std::cout << "Castor Pedestal Calibrations will be saved to " << outputFile_.c_str() << std::endl; }
	}


  if(fVerbosity>0) { std::cout << "CastorDigiMonitor::setup (end)" << std::endl; }

  return;
}


//==================================================================//
//=========================== beginRun =============================//
//==================================================================//
void CastorDigiMonitor::beginRun(const edm::EventSetup& iSetup)
  {
  if(fVerbosity>0) std::cout << "CastorDigiMonitor::beginRun (start)" << std::endl;

  if ( m_dbe !=NULL )
	{
    	m_dbe->setCurrentFolder(baseFolder_);
    	meEVT_ = m_dbe->bookInt("Digi Task Event Number");
    	meEVT_->Fill(ievt_);
        
    	m_dbe->setCurrentFolder(baseFolder_);
  
    	////---- book the following histograms 
    	std::string type = "Castor Digis ADC counts";
    	castHists.ALLPEDS =  m_dbe->book1D(type,type,130,0,130);
    
    	////---- LEAVE IT OUT FOR THE MOMENT
    	//type = "Castor Pedestal Mean Reference Values - from CondDB";
    	//castHists.PEDESTAL_REFS = m_dbe->book1D(type,type,50,0,50);
    	//type = "Castor Pedestal RMS Reference Values - from CondDB";
    	//castHists.WIDTH_REFS = m_dbe->book1D(type,type,20,0,10); 
    	///// castHists.PEDRMS  =  m_dbe->book1D("Castor Pedestal RMS Values","Castor Pedestal RMS Values",100,0,3);
    	///// castHists.SUBMEAN =  m_dbe->book1D("Castor Subtracted Mean Values","Castor Subtracted Mean Values",100,-2.5,2.5);
    	/////  castHists.PEDMEAN =  m_dbe->book1D("Castor Pedestal Mean Values","Castor Pedestal Mean Values",100,0,9);
    	///// castHists.QIERMS  =  m_dbe->book1D("Castor QIE RMS Values","Castor QIE RMS Values",50,0,3);
    	///// castHists.QIEMEAN =  m_dbe->book1D("Castor QIE Mean Values","Castor QIE Mean Values",50,0,10);

	std::string s2 = "QIE_capID+er+dv";
	std::string s3 = "qieError2D";
    	h2digierr = m_dbe->book2DD(s3,s2,16,0.5,16.5, 14,0.5,14.5);
	}
  else
	{ 
   	if(fVerbosity>0) std::cout << "CastorDigiMonitor::setup - NO DQMStore service" << std::endl; 
  	}
 
 
 if(fVerbosity>0) std::cout << "CastorDigiMonitor::beginRun (end)" << std::endl;

 return;
}


//==================================================================//
//=========================== processEvent  ========================//
//==================================================================//
void CastorDigiMonitor::processEvent(const CastorDigiCollection& castorDigis, const CastorDbService& cond)
  {
  if(fVerbosity>0) std::cout << "CastorDigiMonitor::processEvent (begin)"<< std::endl;

  if(!m_dbe) { 
    if(fVerbosity>0) std::cout<<"CastorDigiMonitor::processEvent DQMStore is not instantiated!!!"<<std::endl;  
    return; 
  }

  //if(!shape_) shape_ = cond.getCastorShape(); // this one is generic

  meEVT_->Fill(ievt_);

  CaloSamples tool;  
 
  if(castorDigis.size()>0) {

   for (CastorDigiCollection::const_iterator j=castorDigis.begin(); j!=castorDigis.end(); j++){
      const CastorDataFrame digi = (const CastorDataFrame)(*j);	
 

       detID_.clear(); capID_.clear(); pedVals_.clear();


     ////---- LEAVE THE DB STUFF OUT FOR THE MOMENT

      // const CastorCalibrations& calibrations = cond.getCastorCalibrations(digi.id().rawId());
      // const CastorPedestal* ped              = cond.getPedestal(digi.id()); 
      // const CastorPedestalWidth* pedw        = cond.getPedestalWidth(digi.id());
       ////---- get access to Castor Pedestal in the CONDITION DATABASE
       /////// calibs_= cond.getCastorCalibrations(digi.id());  //-- in HCAL code 
       // const CastorPedestal* ped = cond.getPedestal(digi.id()); 
       // const CastorPedestalWidth* pedw = cond.getPedestalWidth(digi.id());
      
       /*
       ////---- if to convert ADC to fC 
      if(doFCpeds_){
	channelCoder_ = cond.getCastorCoder(digi.id());
	CastorCoderDb coderDB(*channelCoder_, *shape_);
	coderDB.adc2fC(digi,tool);
      }
     
     
       ////---- fill Digi Mean and RMS values from the CONDITION DATABASE
       for(int capID=0; capID<4; capID++){
            ////---- pedestal Mean from the Condition Database
	    float pedvalue=0; 	 
	    if(ped) pedvalue=ped->getValue(capID);
            castHists.PEDESTAL_REFS->Fill(pedvalue);
            PEDESTAL_REFS->Fill(pedvalue);
          ////////// castHists.PEDESTAL_REFS->Fill(calibs_.pedestal(capID)); //In HCAL code
	  /////////   PEDESTAL_REFS->Fill(calibs_.pedestal(capID));  // In HCAL code
          ////---- pedestal RMS from the Condition Database 
           float width=0;
	   if(pedw) width = pedw->getWidth(capID);
           castHists.WIDTH_REFS->Fill(width);
           WIDTH_REFS->Fill(width);
    }
     */
     
 
      ////---- fill ALL Digi Values each 1000 events
       
       //      if(ievt_ %1000 == 0 )           // PK: skip limited number of events
      //   { 
      for (int i=0; i<digi.size(); i++) {
	if(doFCpeds_) pedVals_.push_back(tool[i]); // default is FALSE
	else pedVals_.push_back(digi.sample(i).adc());
	detID_.push_back(digi.id());
	capID_.push_back(digi.sample(i).capid());
	castHists.ALLPEDS->Fill(pedVals_[i]);
      }
      
      //      }      

      ////---- do histograms for every channel once per 100 events
      //      if( ievt_%100 == 0 && doPerChannel_) perChanHists(detID_,capID_,pedVals_,castHists.PEDVALS, baseFolder_);
      if( doPerChannel_) perChanHists(detID_,capID_,pedVals_,castHists.PEDVALS, baseFolder_); // PK: no special event selection done
   int capid1 = digi.sample(0).capid();
   for (int i=1; i<digi.size(); i++) {
     int module = digi.id().module();
     int sector = digi.id().sector();
     if(capid1 < 3) capid1++;
     else capid1 = 0;
     int capid = digi.sample(i).capid();
     int dv = digi.sample(i).dv();
     int er = digi.sample(i).er();
     int err = (capid != capid1) | er<<1 | (!dv)<<2; // =0
     if(err !=0) h2digierr->Fill(sector,module);
//     if(err != 0 && fVerbosity>0)
//     std::cout<<"event/idigi=" <<ievt_<<"/" <<i<< " cap_cap1_dv_er: " <<
//	capid <<"="<< capid1 <<" "<< dv <<" "<< er<<" "<< err << std::endl;
     capid1 = capid;
   }
    }
  } 
   else {
    if(fVerbosity>0) std::cout << "CastorPSMonitor::processEvent NO Castor Digis !!!" << std::endl;
  }

 if (showTiming) { 
      cpu_timer.stop(); std::cout << " TIMER::CastorDigi -> " << cpu_timer.cpuTime() << std::endl; 
      cpu_timer.reset(); cpu_timer.start();  
    }

  ievt_++;

  if(fVerbosity>0) std::cout << "CastorDigiMonitor::processEvent (end)"<< std::endl;


  return;
  }


//==================================================================//
//======================= done =====================================//
//==================================================================//
void CastorDigiMonitor::done()
{
  if(m_dbe!=NULL && h2digierr!=NULL)
    {
      if(fVerbosity>0) 
	{
	  long int hdigierrEntr = h2digierr->getEntries();
	  std::cout << "CastorDigiMonitor: capId,er,dv summary (entries=" << hdigierrEntr << "):" << std::endl;
	}
    }
  else
    edm::LogWarning("CastorDigiMonitor") << "DQMStore or histogram not available";
  
  return;
}


//==================================================================//
//======================= perChanHists  ============================//
//==================================================================//
////---- do histograms per channel
void CastorDigiMonitor::perChanHists( const std::vector<HcalCastorDetId>& detID, const std::vector<int>& capID, const std::vector<float>& peds,
				          std::map<HcalCastorDetId, std::map<int, MonitorElement*> > &toolP,  
				          ////// std::map<HcalCastorDetId, std::map<int, MonitorElement*> > &toolS, 
                                          std::string baseFolder) 
 {
  
  if(m_dbe) m_dbe->setCurrentFolder(baseFolder);

  ////---- loop over all channels 
  for(unsigned int d=0; d<detID.size(); d++){
    HcalCastorDetId detid = detID[d];
    int capid = capID[d];
    float pedVal = peds[d];
    ////---- outer iteration
    bool gotit=false;
    if(REG[detid]) gotit=true;
    
    if(gotit){
      ////---- inner iteration
      std::map<int, MonitorElement*> _mei = toolP[detid];
      if(_mei[capid]==NULL){
	if(fVerbosity>0) std::cout<<"CastorDigiMonitor::perChanHists  This histo is NULL!!??"<< std::endl;
      }
      else _mei[capid]->Fill(pedVal);
      
      ///////// _mei = toolS[detid];
      ////////  if(_mei[capid]==NULL){
      ////////	if(fVerbosity>0) std::cout<<"CastorPedestalAnalysis::perChanHists  This histo is NULL!!??\n"<<std::endl;
      ////////  }
      //////// else _mei[capid]->Fill(pedVal-calibs_.pedestal(capid));
    }
    else{
      if(m_dbe){
	std::map<int,MonitorElement*> insertP; //-- Pedestal values in ADC
         //////// std::map<int,MonitorElement*> insertS; // Pedestal values (substracted) 
	
        ////---- Loop over capID 
	for(int i=0; i<4; i++){
	  char name[1024];
	  sprintf(name,"Castor Digi Value (ADC) zside=%d module=%d sector=%d CAPID=%d",
		  detid.zside(),detid.module(),detid.sector(),i);      
	  insertP[i] =  m_dbe->book1D(name,name,10,-0.5,9.5);
	  
	  ////////// sprintf(name," Pedestal Value (Subtracted) zside=%d module=%d sector=%d CAPID=%d",
	  /////////  detid.zside(),detid.module(),detid.sector(),i);      
	  /////////  insertS[i] =  m_dbe->book1D(name,name,10,-5,5);	
	}
	
	insertP[capid]->Fill(pedVal);
	//////// insertS[capid]->Fill(pedVal-calibs_.pedestal(capid));
	toolP[detid] = insertP;
	//////// toolS[detid] = insertS;
      }
      REG[detid] = true;
    }
  }
}

