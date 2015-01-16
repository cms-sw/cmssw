#include "DQM/CastorMonitor/interface/CastorDigiMonitor.h"
#include "DQM/CastorMonitor/interface/CastorLEDMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//****************************************************//
//********** CastorDigiMonitor: ******************//
//********** Author: Dmytro Volyanskyy   *************//
//********** Date  : 29.08.2008 (first version) ******// 
////---- digi values in Castor r/o channels 
//// last revision: 31.05.2011 (Panos Katsas) to remove selecting N events for filling the histograms
//****************************************************//
//---- critical revision 26.06.2014 (Vladimir Popov)
//==================================================================//
//======================= Constructor ==============================//
//==================================================================//
CastorDigiMonitor::CastorDigiMonitor(const edm::ParameterSet& ps)
{
subsystemname_=ps.getUntrackedParameter<std::string>("subSystemFolder","Castor");
 fVerbosity = ps.getUntrackedParameter<int>("debug",0);
//  doPerChannel_ = false;
}


//==================================================================//
//======================= Destructor ===============================//
//==================================================================//
CastorDigiMonitor::~CastorDigiMonitor() { }


//==================================================================//
//=========================== reset  ===============================//
//==================================================================//
void CastorDigiMonitor::reset() { }

//==================================================================//
//=========================== setup  ===============================//
//==================================================================//
//void CastorDigiMonitor::setup(const edm::ParameterSet& ps, DQMStore* dbe)
void CastorDigiMonitor::setup(const edm::ParameterSet& ps)
{
 subsystemname_=
   ps.getUntrackedParameter<std::string>("subSystemFolder","Castor");
 return;
//  CastorBaseMonitor::setup(ps,dbe);
/*
  if(fVerbosity>0) { std::cout << "CastorDigiMonitor::setup (start)" << std::endl; }

//  baseFolder_ = rootFolder_+"CastorDigiMonitor";
 
//  doPerChannel_ = ps.getUntrackedParameter<bool>("DigiPerChannel", false);
//  doFCpeds_ = ps.getUntrackedParameter<bool>("DigiInFC", true);
//  ievt_=0;
*/
/*
  outputFile_ = ps.getUntrackedParameter<std::string>("PedestalFile", "");
  if ( outputFile_.size() != 0 )
  	{
	if(fVerbosity>0) { std::cout << "Castor Pedestal Calibrations will be saved to " << outputFile_.c_str() << std::endl; }
	}

*/
//  if(fVerbosity>0) std::cout << "CastorDigiMonitor::setup (end)"<<std::endl;
  return;
}


//==================================================================//
//=========================== beginRun =============================//
//==================================================================//
//void CastorDigiMonitor::beginRun(const edm::EventSetup& iSetup, DQMStore* pdbe)
//void CastorDigiMonitor::beginRun(const edm::EventSetup& iSetup)
void CastorDigiMonitor::bookHistograms(DQMStore::IBooker& ibooker,
	const edm::Run& iRun, const edm::EventSetup& iSetup)
{
  char s[60];
  if(fVerbosity>0) std::cout << "CastorDigiMonitor::beginRun (start)" << std::endl;
  ievt_=0;
//   pset_.getUntrackedParameter<std::string>("subSystemFolder","Castor");
//  baseFolder_ = rootFolder_+"CastorDigiMonitor";
/*
  dbe = edm::Service<DQMStore>().operator->();
// if( m_dbe !=NULL ) {
 if(dbe ==NULL ) {
   if(fVerbosity>0) 
	std::cout<<"CastorDigiMonitor::beginRun -NO DQMStore service"<<std::endl; 
   return;
 }
//    	m_dbe->setCurrentFolder(baseFolder_);
//  dbe->setCurrentFolder(subsystemname + "/CastorDigiMonitor");
*/

  ibooker.setCurrentFolder(subsystemname_ + "/CastorDigiMonitor");
	std::string s2 = "QIE_capID+er+dv";
//    	h2digierr = m_dbe->book2D(s2,s2,14,0.,14., 16,0.,16.);
//    	h2digierr = dbe->book2D(s2,s2,14,0.,14., 16,0.,16.);
    	h2digierr = ibooker.book2D(s2,s2,14,0.,14., 16,0.,16.);
	h2digierr->getTH2F()->GetXaxis()->SetTitle("ModuleZ");
	h2digierr->getTH2F()->GetYaxis()->SetTitle("SectorPhi");
	h2digierr->getTH2F()->SetOption("colz");

  sprintf(s,"BunchOccupancy(fC)_all_TS");
//    hBunchOcc = m_dbe->bookProfile(s,s,4000,0,4000, 100,0,1.e10,"");
//    hBunchOcc = dbe->bookProfile(s,s,4000,0,4000, 100,0,1.e10,"");
    hBunchOcc = ibooker.bookProfile(s,s,4000,0,4000, 100,0,1.e10,"");
    hBunchOcc->getTProfile()->GetXaxis()->SetTitle("BX");
    hBunchOcc->getTProfile()->GetYaxis()->SetTitle("QIE(fC)");

    sprintf(s,"DigiSize");
//        hdigisize = m_dbe->book1D(s,s,20,0.,20.);
//        hdigisize = dbe->book1D(s,s,20,0.,20.);
        hdigisize = ibooker.book1D(s,s,20,0.,20.);
    sprintf(s,"Module(fC)_allTS");
//        hModule = m_dbe->book1D(s,s,14,0.,14.);
//        hModule = dbe->book1D(s,s,14,0.,14.);
        hModule = ibooker.book1D(s,s,14,0.,14.);
	hModule->getTH1F()->GetXaxis()->SetTitle("ModuleZ");
	hModule->getTH1F()->GetYaxis()->SetTitle("QIE(fC)");
    sprintf(s,"Sector(fC)_allTS");
//        hSector = m_dbe->book1D(s,s,16,0.,16.);
//        hSector = dbe->book1D(s,s,16,0.,16.);
        hSector = ibooker.book1D(s,s,16,0.,16.);
	hSector->getTH1F()->GetXaxis()->SetTitle("SectorPhi");
	hSector->getTH1F()->GetYaxis()->SetTitle("QIE(fC)");
    sprintf(s,"QfC=f(Tile TS) (cumulative)");
//       h2QtsvsCh = m_dbe->book2D(s,s,224,0.,224., 10,0.,10.);
//       h2QtsvsCh = dbe->book2D(s,s,224,0.,224., 10,0.,10.);
       h2QtsvsCh = ibooker.book2D(s,s,224,0.,224., 10,0.,10.);
  h2QtsvsCh->getTH2F()->GetXaxis()->SetTitle("Tile(=sector*14+module)");
       h2QtsvsCh->getTH2F()->GetYaxis()->SetTitle("TS");
//       h2QtsvsCh->getTH2F()->SetOption("colz");
    sprintf(s,"QmeanfC=f(Tile TS)");
//      h2QmeantsvsCh = m_dbe->book2D(s,s,224,0.,224., 10,0.,10.);
//      h2QmeantsvsCh = dbe->book2D(s,s,224,0.,224., 10,0.,10.);
      h2QmeantsvsCh = ibooker.book2D(s,s,224,0.,224., 10,0.,10.);
  h2QmeantsvsCh->getTH2F()->GetXaxis()->SetTitle("Tile(=sector*14+module)");
      h2QmeantsvsCh->getTH2F()->GetYaxis()->SetTitle("TS");
      h2QmeantsvsCh->getTH2F()->SetOption("colz");
    sprintf(s,"QmeanfC_map(allTS)");
//      h2QmeanMap = m_dbe->book2D(s,s,14,0.,14., 16,0.,16.);
//      h2QmeanMap = dbe->book2D(s,s,14,0.,14., 16,0.,16.);
      h2QmeanMap = ibooker.book2D(s,s,14,0.,14., 16,0.,16.);
      h2QmeanMap->getTH2F()->GetXaxis()->SetTitle("ModuleZ");
      h2QmeanMap->getTH2F()->GetYaxis()->SetTitle("SectorPhi");
      h2QmeanMap->getTH2F()->SetOption("textcolz");

/*
//    	meEVT_ = m_dbe->bookInt("Digi Task Event Number");
//    	meEVT_->Fill(ievt_);  
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
*/
 
 if(fVerbosity>0) std::cout<<"CastorDigiMonitor::beginRun(end)"<<std::endl;
 return;
}


//==================================================================//
//=========================== processEvent  ========================//
//==================================================================//
void CastorDigiMonitor::processEvent(const CastorDigiCollection& castorDigis,
	const CastorDbService& cond, int iBunch)
 {
//  printf("CastorDigiMonitor::processEvent() is called\n");
  if(fVerbosity>0) std::cout << "CastorDigiMonitor::processEvent (begin)"<< std::endl;
/*
  if(!m_dbe) { 
    if(fVerbosity>0) std::cout<<"CastorDigiMonitor::processEvent DQMStore is not instantiated!!!"<<std::endl;  
    return; 
  }
*/
  //if(!shape_) shape_ = cond.getCastorShape(); // this one is generic

//  meEVT_->Fill(ievt_);

  CaloSamples tool;  
 
  if(castorDigis.size()>0) {

   for (CastorDigiCollection::const_iterator j=castorDigis.begin(); j!=castorDigis.end(); j++){
      const CastorDataFrame digi = (const CastorDataFrame)(*j);	
 

//       detID_.clear(); capID_.clear(); pedVals_.clear();


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
*/

   int capid1 = digi.sample(0).capid();
   hdigisize->Fill(digi.size());
   double sum = 0.;
//   for (int i=1; i<digi.size(); i++) {
   for (int i=0; i<digi.size(); i++) {
     int module = digi.id().module()-1;
     int sector = digi.id().sector()-1;
     if(capid1 < 3) capid1++;
     else capid1 = 0;
     int capid = digi.sample(i).capid();
     int dv = digi.sample(i).dv();
     int er = digi.sample(i).er();
     int rawd = digi.sample(i).adc();
     rawd = rawd&0x7F;
     int err = (capid != capid1) | er<<1 | (!dv)<<2; // =0
     if(err !=0) h2digierr->Fill(module,sector);
     int ind = sector*14 + module;
     h2QtsvsCh->Fill(ind,i,LedMonAdc2fc[rawd]);
     sum += LedMonAdc2fc[rawd];
//     if(err != 0 && fVerbosity>0)
//     std::cout<<"event/idigi=" <<ievt_<<"/" <<i<< " cap_cap1_dv_er: " <<
//	capid <<"="<< capid1 <<" "<< dv <<" "<< er<<" "<< err << std::endl;
     capid1 = capid;
   }
   hBunchOcc->Fill(iBunch,sum);
  } //end for(CastorDigiCollection::const_iterator ...
} 
   else {
    if(fVerbosity>0) std::cout << "CastorPSMonitor::processEvent NO Castor Digis !!!" << std::endl;
  }
// if(castorDigis.size()>0) {
/*
 if (showTiming) { 
      cpu_timer.stop(); std::cout << " TIMER::CastorDigi -> " << cpu_timer.cpuTime() << std::endl; 
      cpu_timer.reset(); cpu_timer.start();  
    }
*/
  ievt_++;
  if(ievt_ %100 == 0) {
   float ModuleSum[14], SectorSum[16];
   for(int m=0; m<14; m++) ModuleSum[m]=0.;
   for(int s=0; s<16; s++) SectorSum[s]=0.;
   for(int mod=0; mod<14; mod++) for(int sec=0; sec<16; sec++) {
     double sum=0.;
     for(int ts=1; ts<=10; ts++) {
       int ind = sec*14 + mod +1;
       double a=h2QtsvsCh->getTH2F()->GetBinContent(ind,ts);
  h2QmeantsvsCh->getTH2F()->SetBinContent(ind,ts,a/double(ievt_));
       sum += a;
     }
     sum /= double(ievt_);
     ModuleSum[mod] += sum;
     SectorSum[sec] += sum;
     float isum = float(int(sum*10.+0.5))/10.;
     h2QmeanMap->getTH2F()->SetBinContent(mod+1,sec+1,isum);
   } // end for(int mod=0; mod<14; mod++) for(int sec=0; sec<16; sec++) 
   for(int mod=0; mod<14; mod++) 
	hModule->getTH1F()->SetBinContent(mod+1,ModuleSum[mod]);
   for(int sec=0; sec<16; sec++) 
	hSector->getTH1F()->SetBinContent(sec+1,SectorSum[sec]);
  } //end if(ievt_ %500 == 0) {

  if(fVerbosity>0) std::cout << "CastorDigiMonitor::processEvent (end)"<< std::endl;


  return;
  }


//==================================================================//
//======================= done =====================================//
//==================================================================//
void CastorDigiMonitor::done()
  {
  long int hdigierrEntr = h2digierr->getEntries();
  if(fVerbosity>0) std::cout << "CastorDigiMonitor: capId,er,dv summary (entries="
	<<hdigierrEntr<<"):"<<std::endl;
  return;
  }


//==================================================================//
//======================= perChanHists  ============================//
//==================================================================//
////---- do histograms per channel
/*
void CastorDigiMonitor::perChanHists( std::vector<HcalCastorDetId> detID, std::vector<int> capID, std::vector<float> peds,
				          std::map<HcalCastorDetId, std::map<int, MonitorElement*> > &toolP,  
				          ////// std::map<HcalCastorDetId, std::map<int, MonitorElement*> > &toolS, 
                                          std::string baseFolder) 
 {
  return;
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
*/
