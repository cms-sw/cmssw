#include "DQM/CastorMonitor/interface/CastorLEDMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//***************************************************//
//********** CastorLEDMonitor ***********************//
//********** Author: Dmytro Volyanskyy   ************//
//********** Date  : 20.11.2008 (first version) ******// 
//---------- last revision: 31.05.2011 (Panos Katsas) 
//***************************************************//
//---- critical revision 26.06.2014 (Vladimir Popov)
//==================================================================//
//======================= Constructor ==============================//
//==================================================================//
CastorLEDMonitor::CastorLEDMonitor(const edm::ParameterSet& ps)
{
 subsystemname =
        ps.getUntrackedParameter<std::string>("subSystemFolder","Castor");
 ievt_=0;

//  doPerChannel_ = false;
//  sigS0_=0;
//  sigS1_=9;
}

//==================================================================//
//======================= Destructor ==============================//
//==================================================================//
CastorLEDMonitor::~CastorLEDMonitor() {
}
  
//==========================================================//
//========================= setup ==========================//
//==========================================================//
void CastorLEDMonitor::setup(const edm::ParameterSet& ps, DQMStore* dbe)
{
 return;
/*
  CastorBaseMonitor::setup(ps,dbe);
  if(fVerbosity>0) std::cout<<"CastorLEDMonitor::setup (start)"<<std::endl;  
  baseFolder_ = rootFolder_+"CastorLEDMonitor";
//  if(ps.getUntrackedParameter<bool>("LEDPerChannel",false)) doPerChannel_ = true;
//  sigS0_ = ps.getUntrackedParameter<int>("FirstSignalBin", 0);
//  sigS1_ = ps.getUntrackedParameter<int>("LastSignalBin", 9);
//  adcThresh_ = ps.getUntrackedParameter<double>("LED_ADC_Thresh", 0);
  if(sigS0_<0)
	{
    	if(fVerbosity>1) { std::cout << "CastorLEDMonitor::setup, illegal range for first sample: " << sigS0_ << std::endl; }
    	sigS0_=0;
  	}
  if(sigS1_>9)
	{
    	if(fVerbosity>1) { std::cout << "CastorLEDMonitor::setup, illegal range for last sample: " << sigS1_ << std::endl; }
    	sigS1_=9;
  	}
  if(sigS0_ > sigS1_)
	{ 
    	if(fVerbosity>1) { std::cout<< "CastorLEDMonitor::setup, illegal range for first: "<< sigS0_ << " and last sample: " << sigS1_ << std::endl; }
    	sigS0_=0; sigS1_=9;
  	}
*/
  ievt_=0;
  if(fVerbosity>0) std::cout<<"CastorLEDMonitor::setup (end)"<<std::endl;  
  return;
}

//=================================================================//
//========================== beginRun =============================//
//================================================================//
//void CastorLEDMonitor::beginRun(const edm::EventSetup& iSetup)
void CastorLEDMonitor::bookHistograms(DQMStore::IBooker& ibooker,
	const edm::Run& iRun, const edm::EventSetup& iSetup)
{
  char s[60];
  if(fVerbosity>0) std::cout<<"CastorLEDMonitor::bookHistograms"<<std::endl;  
/*
 if (m_dbe ==NULL) {
  if(fVerbosity>0) 
	std::cout <<"CastorLEDMonitor::beginRun - NO DQMStore service"<<std::endl; 
  return;
 }
  m_dbe->setCurrentFolder(baseFolder_);
*/
  ibooker.setCurrentFolder(subsystemname + "/CastorLEDMonitor");
  sprintf(s,"CastorLED_qVsTS(allPMT)");
//   h2qts = m_dbe->book2D(s,s, 10,0,10., 5000,0.,10000.);
   h2qts = ibooker.book2D(s,s, 10,0,10., 5000,0.,10000.);
   h2qts->getTH2F()->GetXaxis()->SetTitle("TS");
   h2qts->getTH2F()->GetYaxis()->SetTitle("Q(fC)");
   h2qts->getTH2F()->SetOption("colz");

  sprintf(s,"CastorLED_qVsPMT");
//   h2QvsPMT = m_dbe->book2D(s,s, 224,0,224, 5000,0.,50000.);    
   h2QvsPMT = ibooker.book2D(s,s, 224,0,224, 5000,0.,50000.);    
   h2QvsPMT->getTH2F()->GetXaxis()->SetTitle("sector*14+module");
   h2QvsPMT->getTH2F()->GetYaxis()->SetTitle("RecHit");
   h2QvsPMT->getTH2F()->SetOption("colz");

  sprintf(s,"CastorLEDqMap(cumulative)");
//    h2qMap = m_dbe->book2D(s,s,14, 0,14, 16, 0,16);
    h2qMap = ibooker.book2D(s,s,14, 0,14, 16, 0,16);
    h2qMap->getTH2F()->SetOption("colz");
  sprintf(s,"CastorLED_QmeanMap");
//    h2meanMap = m_dbe->book2D(s,s,14, 0,14, 16, 0,16);
    h2meanMap = ibooker.book2D(s,s,14, 0,14, 16, 0,16);
    h2meanMap->getTH2F()->GetXaxis()->SetTitle("moduleZ");
    h2meanMap->getTH2F()->GetYaxis()->SetTitle("sectorPhi");
    h2meanMap->getTH2F()->SetOption("colz");

 ievt_=0;
 return;
/*
//    	meEVT_ = m_dbe->bookInt("LED Task Event Number");    
//    	meEVT_->Fill(ievt_);
       	castHists.shapePED =  m_dbe->book1D("Castor Ped Subtracted Pulse Shape","Castor Ped Subtracted Pulse Shape",10,-0.5,9.5);
    	castHists.shapeALL =  m_dbe->book1D("Castor Average Pulse Shape","Castor Average Pulse Shape",10,-0.5,9.5);
    	castHists.energyALL =  m_dbe->book1D("Castor Average Pulse Energy","Castor Average Pulse Energy",500,0,500);
    	castHists.timeALL =  m_dbe->book1D("Castor Average Pulse Time","Castor Average Pulse Time",200,-1,10);
    	castHists.rms_shape =  m_dbe->book1D("Castor LED Shape RMS Values","Castor LED Shape RMS Values",100,0,5);
    	castHists.mean_shape =  m_dbe->book1D("Castor LED Shape Mean Values","Castor LED Shape Mean Values",100,-0.5,9.5);
    	castHists.rms_time =  m_dbe->book1D("Castor LED Time RMS Values","Castor LED Time RMS Values",100,0,5);
    	castHists.mean_time =  m_dbe->book1D("Castor LED Time Mean Values","Castor LED Time Mean Values",100,-1,10);
    	castHists.rms_energy =  m_dbe->book1D("Castor LED Energy RMS Values","Castor LED Energy RMS Values",100,0,500);
    	castHists.mean_energy =  m_dbe->book1D("Castor LED Energy Mean Values","Castor LED Energy Mean Values",100,0,1000); 
*/

 if(fVerbosity>0) std::cout<<"CastorLEDMonitor::beginRun(end)"<<std::endl; 
}

//==========================================================//
//=================== processEvent  ========================//
//==========================================================//

void CastorLEDMonitor::processEvent( const CastorDigiCollection& castorDigis, const CastorDbService& cond)
  {
  if(fVerbosity>0) std::cout<<"CastorLEDMonitor::processEvent (start)"<<std::endl;  
/*
//  meEVT_->Fill(ievt_);
  if(!m_dbe){ 
    if(fVerbosity>0) std::cout<<"CastorLEDMonitor::processEvent DQMStore not instantiated!!!"<<std::endl;  
    return; 
  }
*/

/* be implemented
 edm::Handle<HcalTBTriggerData> trigger_data;
 iEvent.getByToken(tok_tb_, trigger_data);
 if(trigger_data.isValid()) 
  if(trigger_data->triggerWord()==6) LEDevent=true; 
*/
  
 if(castorDigis.size() <= 0) {
  if(fVerbosity > 0) 
	std::cout <<"CastorLEDMonitor::processEvent NO Castor Digis"<<std::endl;
  return;
 }

 for(CastorDigiCollection::const_iterator j=castorDigis.begin(); j!=castorDigis.end(); j++)
 {
   const CastorDataFrame digi = (const CastorDataFrame)(*j);
   int module = digi.id().module()-1;
   int sector = digi.id().sector()-1;
   double qsum=0.;
//   for(int i=sigS0_; i<=sigS1_; i++) {
   for(int i=0; i<digi.size(); i++) {
     int dig=digi.sample(i).adc() & 0x7f;
     float ets = LedMonAdc2fc[dig] + 0.5;
     h2qts->Fill(i,ets);
     qsum += ets;
   }
   int ind = sector*14 + module;
   h2QvsPMT->Fill(ind,qsum);
   h2qMap->Fill(module,sector,qsum);

/*
// float vals[10];
      /////---- No getCastorCalibrations method at the moment
      //// calibs_= cond.getCastorCalibrations(digi.id());  // Old method was made private. 
      ////---- leave calibs_ empty for the moment:
      float energy=0;
      float ts =0; float bs=0;
      int maxi=0; float maxa=0;
      for(int i=sigS0_; i<=sigS1_; i++){
	if(digi.sample(i).adc()>maxa){maxa=digi.sample(i).adc(); maxi=i;}
      }
      for(int i=sigS0_; i<=sigS1_; i++){	  
	float tmp1 =0;   
        int j1=digi.sample(i).adc();
        tmp1 = (LedMonAdc2fc[j1]+0.5);   	  
	energy += tmp1-calibs_.pedestal(digi.sample(i).capid());
	if(i>=(maxi-1) && i<=maxi+1){
	  ts += i*(tmp1-calibs_.pedestal(digi.sample(i).capid()));
	  bs += tmp1-calibs_.pedestal(digi.sample(i).capid());
	}
      }
      if(energy<adcThresh_) continue;
     //      castHists.energyALL->Fill(energy);
      if(bs!=0) castHists.timeALL->Fill(ts/bs);
      //     if(ievt_%1000 == 0 ){
       for (int i=0; i<digi.size(); i++) {
	float tmp =0;
        int j=digi.sample(i).adc();
        tmp = (LedMonAdc2fc[j]+0.5);
        castHists.shapeALL->Fill(i,tmp);
        castHists.shapePED->Fill(i,tmp-calibs_.pedestal(digi.sample(i).capid()));
//	vals[i] = tmp-calibs_.pedestal(digi.sample(i).capid());
       }
       //      }
      //do per channel histograms once for each 100 events
//if(doPerChannel_) perChanHists(digi.id(),vals,castHists.shape,castHists.time,castHists.energy,baseFolder_);
*/
 } // end for(CastorDigiCollection::const_iterator j=castorDigis...

  ievt_++; 
  if(ievt_%100 == 0) {
//   printf("CastorLEDMonitor(ievt=%d)",ievt_);
   for(int mod=1; mod<=14; mod++) for(int sec=1; sec<=16;sec++) {
    double a= h2qMap->getTH2F()->GetBinContent(mod,sec);
//    printf(" [%d,%d]%e",mod,sec,a); 
    h2meanMap->getTH2F()->SetBinContent(mod,sec,a/double(ievt_));
   }
//   printf("\n");
  }

//if(fVerbosity>0) std::cout<<"CastorLEDMonitor::processEvent(end)"<<std::endl;
  return;
}


//==========================================================//
//=================== done  ================================//
//==========================================================//

void CastorLEDMonitor::done() {
  return;
}


//==========================================================//
//========================= reset ==========================//
//==========================================================//

void CastorLEDMonitor::reset(){
/*
  MonitorElement* unpackedFEDS = m_dbe->get("Castor/FEDs Unpacked");
  if(unpackedFEDS){
    for(int b=1; b<=unpackedFEDS->getNbinsX(); b++) 
      if(unpackedFEDS->getBinContent(b)>0)
	createFEDmap(700+(b-1));  
  }
*/
}

/*
//==========================================================//
//=================== perChanHists  ========================//
//==========================================================//
void CastorLEDMonitor::perChanHists(const HcalCastorDetId DetID, float* vals, 
				  std::map<HcalCastorDetId, MonitorElement*> &tShape, 
				  std::map<HcalCastorDetId, MonitorElement*> &tTime,
				  std::map<HcalCastorDetId, MonitorElement*> &tEnergy,
				  std::string baseFolder){
  
  std::string type = "CastorLEDPerChannel";
  if(m_dbe) m_dbe->setCurrentFolder(baseFolder+"/"+type);

  MonitorElement* me;
  if(m_dbe==NULL) return;
  meIter=tShape.begin();
  meIter = tShape.find(DetID);
 
 if (meIter!=tShape.end()){
    me= meIter->second;
    if(me==NULL && fVerbosity>0) printf("CastorLEDAnalysis::perChanHists  This histo is NULL!!??\n");
    else{
      float energy=0;
      float ts =0; float bs=0;
      int maxi=0; float maxa=0;
      for(int i=sigS0_; i<=sigS1_; i++){
	if(vals[i]>maxa){maxa=vals[i]; maxi=i;}
      }
      for(int i=sigS0_; i<=sigS1_; i++){	  
	energy += vals[i];
	if(i>=(maxi-1) && i<=maxi+1){
	  ts += i*vals[i];
	  bs += vals[i];
	}
	me->Fill(i,vals[i]);
      }
      me = tTime[DetID];      
      if(bs!=0) me->Fill(ts/bs);
      me = tEnergy[DetID];  
      me->Fill(energy); 
    }
  }
  else{
    char name[1024];
    sprintf(name,"Castor LED Shape zside=%d  module=%d  sector=%d",DetID.zside(),DetID.module(),DetID.sector());      
    MonitorElement* insert1;
    insert1 =  m_dbe->book1D(name,name,10,-0.5,9.5);
    float energy=0;
    float ts =0; float bs=0;
    int maxi=0; float maxa=0;
    for(int i=sigS0_; i<=sigS1_; i++){
      if(vals[i]>maxa){maxa=vals[i]; maxi=i;}
      insert1->Fill(i,vals[i]); 
    }
    for(int i=sigS0_; i<=sigS1_; i++){	  
      energy += vals[i];
      if(i>=(maxi-1) && i<=maxi+1){
	ts += i*vals[i];
	bs += vals[i];
      }
    }
    tShape[DetID] = insert1;
    
    sprintf(name,"Castor LED Time  zside=%d  module=%d  sector=%d",DetID.zside(),DetID.module(),DetID.sector());      
    MonitorElement* insert2 =  m_dbe->book1D(name,name,100,0,10);
    if(bs!=0) insert2->Fill(ts/bs); 
    tTime[DetID] = insert2;	


    sprintf(name,"Castor LED Energy zside=%d  module=%d  sector=%d",DetID.zside(),DetID.module(),DetID.sector());      
    MonitorElement* insert3 =  m_dbe->book1D(name,name,500,0,500);
    insert3->Fill(energy); 
    tEnergy[DetID] = insert3;	
    
  } 
  return;
}  
*/
/*
//==========================================================//
//================== createFEDmap ==========================//
//==========================================================//
void CastorLEDMonitor::createFEDmap(unsigned int fed){
  fedIter = MEAN_MAP_SHAPE_DCC.find(fed);
  
  if(fedIter==MEAN_MAP_SHAPE_DCC.end()){
    m_dbe->setCurrentFolder(baseFolder_);
    char name[256];

    sprintf(name,"DCC %d Mean Shape Map",fed);
    MonitorElement* mean_shape = m_dbe->book2D(name,name,24,0.5,24.5,15,0.5,15.5);
    sprintf(name,"DCC %d RMS Shape Map",fed);
    MonitorElement* rms_shape = m_dbe->book2D(name,name,24,0.5,24.5,15,0.5,15.5);

    MEAN_MAP_SHAPE_DCC[fed] = mean_shape;
    RMS_MAP_SHAPE_DCC[fed] = rms_shape;

    sprintf(name,"DCC %d Mean Time Map",fed);
    MonitorElement* mean_time = m_dbe->book2D(name,name,24,0.5,24.5,15,0.5,15.5);
    sprintf(name,"DCC %d RMS Time Map",fed);
    MonitorElement* rms_time = m_dbe->book2D(name,name,24,0.5,24.5,15,0.5,15.5);
    MEAN_MAP_TIME_DCC[fed] = mean_time;
    RMS_MAP_TIME_DCC[fed] = rms_time;
    
    sprintf(name,"DCC %d Mean Energy Map",fed);
    MonitorElement* mean_energy = m_dbe->book2D(name,name,24,0.5,24.5,15,0.5,15.5);
    sprintf(name,"DCC %d RMS Energy Map",fed);
    MonitorElement* rms_energy = m_dbe->book2D(name,name,24,0.5,24.5,15,0.5,15.5);
    MEAN_MAP_ENERGY_DCC[fed] = mean_energy;
    RMS_MAP_ENERGY_DCC[fed] = rms_energy;
  }   
}
*/

