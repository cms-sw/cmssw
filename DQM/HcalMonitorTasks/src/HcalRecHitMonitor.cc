#include "DQM/HcalMonitorTasks/interface/HcalRecHitMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

HcalRecHitMonitor::HcalRecHitMonitor() {
  doPerChannel_ = false;
  occThresh_ = 1;
  ievt_=0;
}

HcalRecHitMonitor::~HcalRecHitMonitor() {
}

void HcalRecHitMonitor::reset(){}


namespace HcalRecHitPerChan{
  template<class RecHit>
  inline void perChanHists(int id, const RecHit& rhit, 
			   std::map<HcalDetId, MonitorElement*> &toolE, 
			   std::map<HcalDetId, MonitorElement*> &toolT,
			   DQMStore* dbe, string baseFolder) {
    
    std::map<HcalDetId,MonitorElement*>::iterator _mei;

    string type = "HB";
    if(id==1)type = "HE"; 
    else if(id==2) type = "HO"; 
    else if(id==3) type = "HF"; 

    if(dbe) dbe->setCurrentFolder(baseFolder+"/"+type);

    
    ///energies by channel
    _mei=toolE.find(rhit.id()); // look for a histogram with this hit's id
    if (_mei!=toolE.end()){
      if (_mei->second==0) return;
      else _mei->second->Fill(rhit.energy()); // if it's there, fill it with energy
    }
    else{
      if(dbe){
	char name[1024];
	sprintf(name,"%s RecHit Energy ieta=%d iphi=%d depth=%d",type.c_str(),rhit.id().ieta(),rhit.id().iphi(),rhit.id().depth());
	//changed for GRUMM cosmics:
	//	toolE[rhit.id()] =  dbe->book1D(name,name,200,0,200); 
       	toolE[rhit.id()] =  dbe->book1D(name,name,200,-10,20); 
	toolE[rhit.id()]->Fill(rhit.energy());
      }
    }
    
    ///times by channel
    _mei=toolT.find(rhit.id()); // look for a histogram with this hit's id
    if (_mei!=toolT.end()){
      if (_mei->second==0) return;
      else _mei->second->Fill(rhit.time()); // if it's there, fill it with time
    }
    else{
      if(dbe){
	char name[1024];
	sprintf(name,"%s RecHit Time ieta=%d iphi=%d depth=%d",type.c_str(),rhit.id().ieta(),rhit.id().iphi(),rhit.id().depth());
	toolT[rhit.id()] =  dbe->book1D(name,name,300,-100,200); 
	toolT[rhit.id()]->Fill(rhit.time());
      }
    }
  }
}


void HcalRecHitMonitor::setup(const edm::ParameterSet& ps, DQMStore* dbe){
  HcalBaseMonitor::setup(ps,dbe);
  baseFolder_ = rootFolder_+"RecHitMonitor";

  if ( ps.getUntrackedParameter<bool>("RecHitsPerChannel", false) ){
    doPerChannel_ = true;
  }
  
  etaMax_ = ps.getUntrackedParameter<double>("MaxEta", 41.5);
  etaMin_ = ps.getUntrackedParameter<double>("MinEta", -41.5);
  etaBins_ = (int)(etaMax_ - etaMin_);
  if (fVerbosity) cout << "RecHit eta min/max set to " << etaMin_ << "/" << etaMax_ << endl;
  
  phiMax_ = ps.getUntrackedParameter<double>("MaxPhi", 73);
  phiMin_ = ps.getUntrackedParameter<double>("MinPhi", 0);
  phiBins_ = (int)(phiMax_ - phiMin_);
  if (fVerbosity) cout << "RecHit phi min/max set to " << phiMin_ << "/" << phiMax_ << endl;
  
  occThresh_ = ps.getUntrackedParameter<double>("RecHitOccThresh", 1.0);
  if (fVerbosity) cout << "RecHit occupancy threshold set to " << occThresh_ << endl;
  
  ievt_=0;
  
  if ( m_dbe !=NULL ) {    

    m_dbe->setCurrentFolder(baseFolder_);
    //changed for cosmics
    //    meRECHIT_E_all =  m_dbe->book1D("RecHit Total Energy","RecHit Total Energy",100,0,400);
    meRECHIT_E_all =  m_dbe->book1D("RecHit Total Energy","RecHit Total Energy",100,-20,400);
    meRECHIT_Ethresh_all =  m_dbe->book1D("RecHit Total Energy - Threshold","RecHit Total Energy - Threshold",100,0,400);

    meEVT_ = m_dbe->bookInt("RecHit Event Number");    
    meEVT_->Fill(ievt_);
    meOCC_MAP_L1 = m_dbe->book2D("RecHit Depth 1 Occupancy Map","RecHit Depth 1 Occupancy Map",
				 etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    meOCC_MAP_L1_E = m_dbe->book2D("RecHit Depth 1 Energy Map","RecHit Depth 1 Energy Map",
				   etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);

    meOCC_MAP_L2 = m_dbe->book2D("RecHit Depth 2 Occupancy Map","RecHit Depth 2 Occupancy Map",
				 etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    meOCC_MAP_L2_E = m_dbe->book2D("RecHit Depth 2 Energy Map","RecHit Depth 2 Energy Map",
				   etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);

    meOCC_MAP_L3 = m_dbe->book2D("RecHit Depth 3 Occupancy Map","RecHit Depth 3 Occupancy Map",
				 etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    meOCC_MAP_L3_E = m_dbe->book2D("RecHit Depth 3 Energy Map","RecHit Depth 3 Energy Map",
				   etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);

    meOCC_MAP_L4 = m_dbe->book2D("RecHit Depth 4 Occupancy Map","RecHit Depth 4 Occupancy Map",
				 etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    meOCC_MAP_L4_E = m_dbe->book2D("RecHit Depth 4 Energy Map","RecHit Depth 4 Energy Map",
				   etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    
    meOCC_MAP_ETA = m_dbe->book1D("RecHit Eta Occupancy Map","RecHit Eta Occupancy Map",etaBins_,etaMin_,etaMax_);
    meOCC_MAP_PHI = m_dbe->book1D("RecHit Phi Occupancy Map","RecHit Phi Occupancy Map",phiBins_,phiMin_,phiMax_);

    meOCC_MAP_ETA_E = m_dbe->book1D("RecHit Eta Energy Map","RecHit Eta Energy Map",etaBins_,etaMin_,etaMax_);
    meOCC_MAP_PHI_E = m_dbe->book1D("RecHit Phi Energy Map","RecHit Phi Energy Map",phiBins_,phiMin_,phiMax_);

    m_dbe->setCurrentFolder(baseFolder_+"/HB");
    //changed for cosmics
    //    hbHists.meRECHIT_E_tot = m_dbe->book1D("HB RecHit Total Energy","HB RecHit Total Energy",100,0,400);
    hbHists.meRECHIT_E_tot = m_dbe->book1D("HB RecHit Total Energy","HB RecHit Total Energy",100,-200,200);
    //    hbHists.meRECHIT_E_all = m_dbe->book1D("HB RecHit Energies","HB RecHit Energies",200,0,200);
    hbHists.meRECHIT_E_all = m_dbe->book1D("HB RecHit Energies","HB RecHit Energies",200,-2,2);

    hbHists.meRECHIT_E_low = m_dbe->book1D("HB RecHit Energies - Low Region","HB RecHit Energies - Low Region",200,0,10);
    hbHists.meRECHIT_T_all = m_dbe->book1D("HB RecHit Times","HB RecHit Times",300,-100,200);
    hbHists.meOCC_MAP_GEO = m_dbe->book2D("HB RecHit Geo Occupancy Map","HB RecHit Geo Occupancy Map",
					  etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    
    hbHists.meRECHIT_Ethresh_tot = m_dbe->book1D("HB RecHit Total Energy - Threshold","HB RecHit Total Energy - Threshold",100,0,400);
    hbHists.meRECHIT_Tthresh_all = m_dbe->book1D("HB RecHit Times - Threshold","HB RecHit Times - Threshold",300,-100,200);
    hbHists.meOCC_MAPthresh_GEO = m_dbe->book2D("HB RecHit Geo Occupancy Map - Threshold",
						"HB RecHit Geo Occupancy Map - Threshold",
						etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);


    m_dbe->setCurrentFolder(baseFolder_+"/HE");
    //changed for cosmics
    //     heHists.meRECHIT_E_tot = m_dbe->book1D("HE RecHit Total Energy","HE RecHit Total Energy",100,0,400);
    heHists.meRECHIT_E_tot = m_dbe->book1D("HE RecHit Total Energy","HE RecHit Total Energy",100,-200,200);
    //    heHists.meRECHIT_E_all = m_dbe->book1D("HE RecHit Energies","HE RecHit Energies",200,0,200);
    heHists.meRECHIT_E_all = m_dbe->book1D("HE RecHit Energies","HE RecHit Energies",200,-2,2);
    heHists.meRECHIT_E_low = m_dbe->book1D("HE RecHit Energies - Low Region","HE RecHit Energies - Low Region",200,0,10);
    heHists.meRECHIT_T_all = m_dbe->book1D("HE RecHit Times","HE RecHit Times",300,-100,200);
    heHists.meOCC_MAP_GEO = m_dbe->book2D("HE RecHit Geo Occupancy Map","HE RecHit Geo Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    heHists.meRECHIT_Ethresh_tot = m_dbe->book1D("HE RecHit Total Energy - Threshold","HE RecHit Total Energy - Threshold",100,0,400);
    heHists.meRECHIT_Tthresh_all = m_dbe->book1D("HE RecHit Times - Threshold","HE RecHit Times - Threshold",300,-100,200);
    heHists.meOCC_MAPthresh_GEO = m_dbe->book2D("HE RecHit Geo Occupancy Map - Threshold",
						"HE RecHit Geo Occupancy Map - Threshold",
						etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);

    m_dbe->setCurrentFolder(baseFolder_+"/HF");
    //changed for cosmics
    // hfHists.meRECHIT_E_tot = m_dbe->book1D("HF RecHit Total Energy","HF RecHit Total Energy",100,0,400);
    hfHists.meRECHIT_E_tot = m_dbe->book1D("HF RecHit Total Energy","HF RecHit Total Energy",100,-200,200);

    //    hfHists.meRECHIT_E_all = m_dbe->book1D("HF Long, RecHit Energies","HF Long, RecHit Energies",200,0,200);
    hfHists.meRECHIT_E_all = m_dbe->book1D("HF Long RecHit Energies","HF Long RecHit Energies",200,-5,5);
    hfHists.meRECHIT_E_low = m_dbe->book1D("HF Long RecHit Energies - Low Region","HF Long RecHit Energies - Low Region",200,0,10);
    hfHists.meRECHIT_T_all = m_dbe->book1D("HF Long RecHit Times","HF Long RecHit Times",300,-100,200);

    
    //need to see Long (depth1) and Short (depth2) fibers separately:
    //hfHists.meRECHIT_E_all_L = m_dbe->book1D("HF Long, RecHit Energies","HF Long, RecHit Energies",200,-5,5);
    //hfHists.meRECHIT_E_low_L = m_dbe->book1D("HF Long, RecHit Energies - Low Region","HF Long, RecHit Energies - Low Region",200,0,10);
    //hfHists.meRECHIT_T_all_L = m_dbe->book1D("HF Long, RecHit Times","HF Long, RecHit Times",300,-100,200);
    //--
    //hfHists.meRECHIT_E_all_S = m_dbe->book1D("HF Short, RecHit Energies","HF Short, RecHit Energies",200,-5,5);
    //hfHists.meRECHIT_E_low_S = m_dbe->book1D("HF Short, RecHit Energies - Low Region","HF Short, RecHit Energies - Low Region",200,0,10);
    //hfHists.meRECHIT_T_all_S = m_dbe->book1D("HF Short, RecHit Times","HF Short, RecHit Times",300,-100,200);

    //--but above is in a map...instead I just add 3 histos for Short:
    hfshort_meRECHIT_E_all = m_dbe->book1D("HF Short RecHit Energies","HF Short RecHit Energies",200,-5,5);
    hfshort_meRECHIT_E_low = m_dbe->book1D("HF Short RecHit Energies - Low Region","HF Short RecHit Energies - Low Region",200,0,10);
    hfshort_meRECHIT_T_all = m_dbe->book1D("HF Short RecHit Times","HF Short RecHit Times",300,-100,200);



    hfHists.meOCC_MAP_GEO = m_dbe->book2D("HF RecHit Geo Occupancy Map","HF RecHit Geo Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.meRECHIT_Ethresh_tot = m_dbe->book1D("HF RecHit Total Energy - Threshold","HF RecHit Total Energy - Threshold",100,0,400);
    hfHists.meRECHIT_Tthresh_all = m_dbe->book1D("HF RecHit Times - Threshold","HF RecHit Times - Threshold",300,-100,200);
    hfHists.meOCC_MAPthresh_GEO = m_dbe->book2D("HF RecHit Geo Occupancy Map - Threshold",
						"HF RecHit Geo Occupancy Map - Threshold",
						etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);


    m_dbe->setCurrentFolder(baseFolder_+"/HO");
    //changed for cosmics
    //hoHists.meRECHIT_E_tot = m_dbe->book1D("HO RecHit Total Energy","HO RecHit Total Energy",100,0,400);
    hoHists.meRECHIT_E_tot = m_dbe->book1D("HO RecHit Total Energy","HO RecHit Total Energy",100,-200,200);
    //    hoHists.meRECHIT_E_all = m_dbe->book1D("HO RecHit Energies","HO RecHit Energies",200,0,200);
    hoHists.meRECHIT_E_all = m_dbe->book1D("HO RecHit Energies","HO RecHit Energies",200,-2,2);
    hoHists.meRECHIT_E_low = m_dbe->book1D("HO RecHit Energies - Low Region","HO RecHit Energies - Low Region",200,0,10);
    hoHists.meRECHIT_T_all = m_dbe->book1D("HO RecHit Times","HO RecHit Times",300,-100,200);
    hoHists.meOCC_MAP_GEO = m_dbe->book2D("HO RecHit Geo Occupancy Map","HO RecHit Geo Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoHists.meRECHIT_Ethresh_tot = m_dbe->book1D("HO RecHit Total Energy - Threshold","HO RecHit Total Energy - Threshold",100,0,400);
    hoHists.meRECHIT_Tthresh_all = m_dbe->book1D("HO RecHit Times - Threshold","HO RecHit Times - Threshold",300,-100,200);
    hoHists.meOCC_MAPthresh_GEO = m_dbe->book2D("HO RecHit Geo Occupancy Map - Threshold",
						"HO RecHit Geo Occupancy Map - Threshold",
						etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);

    // ZDC histograms
    m_dbe->setCurrentFolder(baseFolder_+"/ZDC");
    ZDCtanAlpha = m_dbe->book1D("ZDC_EM_tan_alpha","ZDC EM tan #alpha",100,-0.0003,0.0003);
    ZDCaverageX = m_dbe->book1D("ZDC_EM_avg_weighted_X_position","ZDC EM avg weighted X position",
				147,-73,74);
    ZDCxplusVSxminus = m_dbe->book2D("ZDC_EM_Xplus_Vs_Xminus","ZDC EM <X+> vs <X->",
				     147,-73,74,147,-73,74);
    // Adjust these limits later
    ZDChadVSem_plus = m_dbe->book2D("ZDCplus_HAD_vs_EM","ZDC+ HAD vs EM",400,-100,100,400,-100,100);
    ZDChadVSem_minus = m_dbe->book2D("ZDCminus_HAD_vs_EM","ZDC- HAD vs EM",400,-100,100,400,-100,100);
    ZDCenergy_plusVSminus = m_dbe->book2D("ZDC_energy_plus_vs_minus","ZDC Energy Plus vs Minus",
					  400,-200,200,400,-200,200);
    ZDCenergyVSlayer_plus = m_dbe->bookProfile("ZDC_plus_energy_vs_layer", "ZDC Plus Energy vs Layer",5,0,5,400,-200,200);
    ZDCenergyVSlayer_plus->setBinLabel(1,"EM");
    ZDCenergyVSlayer_plus->setBinLabel(2,"HAD0");
    ZDCenergyVSlayer_plus->setBinLabel(3,"HAD1");
    ZDCenergyVSlayer_plus->setBinLabel(4,"HAD2");
    ZDCenergyVSlayer_plus->setBinLabel(5,"HAD3");
    
    ZDCenergyVSlayer_minus = m_dbe->bookProfile("ZDC_minus_energy_vs_layer", "ZDC Minus Energy vs Layer",5,0,5,400,-200,200);
    ZDCenergyVSlayer_minus->setBinLabel(1,"EM");
    ZDCenergyVSlayer_minus->setBinLabel(2,"HAD0");
    ZDCenergyVSlayer_minus->setBinLabel(3,"HAD1");
    ZDCenergyVSlayer_minus->setBinLabel(4,"HAD2");
    ZDCenergyVSlayer_minus->setBinLabel(5,"HAD3");
    

  } // if (m_dbe) !=NULL

  return;
}

void HcalRecHitMonitor::processEvent(const HBHERecHitCollection& hbHits, 
				     const HORecHitCollection& hoHits, 
				     const HFRecHitCollection& hfHits
				     //const ZDCRecHitCollection& zdcHits // not yet ready for this withing main routine
				     ){

  if(!m_dbe) { 
    if(fVerbosity) cout <<"HcalRecHitMonitor::processEvent   DQMStore not instantiated!!!"<<endl;  
    return; 
  }

  ievt_++;
  meEVT_->Fill(ievt_);


  HBHERecHitCollection::const_iterator HBHEiter;
  HORecHitCollection::const_iterator HOiter;
  HFRecHitCollection::const_iterator HFiter;

  float tot = 0, tot2=0, all =0;
  float totThr = 0, tot2Thr=0, allThr =0;

  if (showTiming) 
    { 
      cpu_timer.reset(); cpu_timer.start();  
    } 


  try{
    if(hbHits.size()>0){    
      for (HBHEiter=hbHits.begin(); HBHEiter!=hbHits.end(); ++HBHEiter) { // loop over all hits
	float en = HBHEiter->energy();    
	float ti = HBHEiter->time();
	
	HcalDetId id(HBHEiter->detid().rawId());

	float ieta = id.ieta(); 
	float iphi = id.iphi();
	float depth = id.depth();
	
	//for cosmics, want to see whole distribution, changed to -100
	//	if(en>0.0){
	if(en>-100){
	  if((HcalSubdetector)(id.subdet())==HcalBarrel){
	    hbHists.meRECHIT_E_all->Fill(en);
	    hbHists.meRECHIT_E_low->Fill(en);
	    hbHists.meRECHIT_T_all->Fill(ti);
	    //NON-threshold occupancy map:
	    hbHists.meOCC_MAP_GEO->Fill(ieta,iphi);
	    tot += en;

	    if(en>occThresh_){
	      totThr += en;
	      hbHists.meOCC_MAPthresh_GEO->Fill(ieta,iphi);
	      hbHists.meRECHIT_Tthresh_all->Fill(ti);

	      meOCC_MAP_ETA->Fill(ieta);
	      meOCC_MAP_PHI->Fill(iphi);	      
	      meOCC_MAP_ETA_E->Fill(ieta,en);
	      meOCC_MAP_PHI_E->Fill(iphi,en);
	      
	      if(depth==1){ 
		meOCC_MAP_L1->Fill(ieta,iphi);
		meOCC_MAP_L1_E->Fill(ieta,iphi, en);
	      }
	      else if(depth==2){ 
		meOCC_MAP_L2->Fill(ieta,iphi);
		meOCC_MAP_L2_E->Fill(ieta,iphi, en);
	      }
	      else if(depth==3){ 
		meOCC_MAP_L3->Fill(ieta,iphi);
		meOCC_MAP_L3_E->Fill(ieta,iphi, en);
	      }
	      if(depth==4){ 
		meOCC_MAP_L4->Fill(ieta,iphi);
		meOCC_MAP_L4_E->Fill(ieta,iphi, en);
	      }
	    }      
	    if(doPerChannel_) 
	      HcalRecHitPerChan::perChanHists<HBHERecHit>(0,*HBHEiter,hbHists.meRECHIT_E,hbHists.meRECHIT_T,m_dbe,baseFolder_);
	  }
	  else if((HcalSubdetector)(id.subdet())==HcalEndcap){
	    heHists.meRECHIT_E_all->Fill(en);
	    heHists.meRECHIT_E_low->Fill(en);
	    heHists.meRECHIT_T_all->Fill(ti);
	    //NON-threshold occupancy map:
	    heHists.meOCC_MAP_GEO->Fill(ieta,iphi);

	    tot2 += en;
	    if(en>occThresh_){
	      tot2Thr += en;
	      meOCC_MAP_ETA->Fill(ieta);
	      meOCC_MAP_PHI->Fill(iphi);
	      meOCC_MAP_ETA_E->Fill(ieta,en);
	      meOCC_MAP_PHI_E->Fill(iphi,en);
	      	      
	      heHists.meOCC_MAPthresh_GEO->Fill(ieta,iphi);
	      heHists.meRECHIT_Tthresh_all->Fill(ti);

	      if(depth==1){ 
		meOCC_MAP_L1->Fill(ieta,iphi);
		meOCC_MAP_L1_E->Fill(ieta,iphi, en);
	      }
	      else if(depth==2){ 
		meOCC_MAP_L2->Fill(ieta,iphi);
		meOCC_MAP_L2_E->Fill(ieta,iphi, en);
	      }
	      else if(depth==3){ 
		meOCC_MAP_L3->Fill(ieta,iphi);
		meOCC_MAP_L3_E->Fill(ieta,iphi, en);
	      }
	      if(depth==4){ 
		meOCC_MAP_L4->Fill(ieta,iphi);
		meOCC_MAP_L4_E->Fill(ieta,iphi, en);
	      }
	    }      
	    if(doPerChannel_) 
	      HcalRecHitPerChan::perChanHists<HBHERecHit>(1,*HBHEiter,heHists.meRECHIT_E,heHists.meRECHIT_T,m_dbe, baseFolder_);
	  }
	}
	
      }
      //      if(tot>0) hbHists.meRECHIT_E_tot->Fill(tot);
      if(tot>-100) hbHists.meRECHIT_E_tot->Fill(tot);
      if(totThr>0) hbHists.meRECHIT_Ethresh_tot->Fill(totThr);
      //if(tot2>0) heHists.meRECHIT_E_tot->Fill(tot2);
      if(tot2>-100) heHists.meRECHIT_E_tot->Fill(tot2);
      if(tot2Thr>0) heHists.meRECHIT_Ethresh_tot->Fill(tot2Thr);
      all += tot;
      all += tot2;
      allThr += totThr;
      allThr += tot2Thr;
    }
  } catch (...) {    
    if(fVerbosity) cout<<"HcalRecHitMonitor::processEvent  Error in HBHE RecHit loop"<<endl;
  }

  if (showTiming)
    { 
      cpu_timer.stop(); std::cout << " TIMER::HcalRecHit RECHIT HBHE-> " << cpu_timer.cpuTime() << std::endl; 
      cpu_timer.reset(); cpu_timer.start();  
    } 

  try{
    tot = 0; totThr = 0;
    if(hoHits.size()>0){
      for (HOiter=hoHits.begin(); HOiter!=hoHits.end(); ++HOiter) { // loop over all hits
	//changed to -100 for cosmics
	//	if(HOiter->energy()>0.0){
	if(HOiter->energy()>-100){
	  hoHists.meRECHIT_E_all->Fill(HOiter->energy());
	  hoHists.meRECHIT_E_low->Fill(HOiter->energy());
	  hoHists.meRECHIT_T_all->Fill(HOiter->time());
	  //HO for some reason DOES NOT have a non-threshold occupancy map
	  
	  tot += HOiter->energy();
	  if(HOiter->energy()>occThresh_){
	    totThr += HOiter->energy();
	    HcalDetId id(HOiter->detid().rawId());

	    hoHists.meOCC_MAPthresh_GEO->Fill(id.ieta(),id.iphi());
	    hoHists.meRECHIT_Tthresh_all->Fill(HOiter->time());

	    meOCC_MAP_ETA->Fill(id.ieta());
	    meOCC_MAP_PHI->Fill(id.iphi());
	    meOCC_MAP_ETA_E->Fill(id.ieta(),HOiter->energy());
	    meOCC_MAP_PHI_E->Fill(id.iphi(),HOiter->energy());
	    
	    hoHists.meOCC_MAP_GEO->Fill(id.ieta(),id.iphi());
	    if(id.depth()==1){ 
	      meOCC_MAP_L1->Fill(id.ieta(),id.iphi());
	      meOCC_MAP_L1_E->Fill(id.ieta(),id.iphi(), HOiter->energy());
	    }
	    else if(id.depth()==2){ 
	      meOCC_MAP_L2->Fill(id.ieta(),id.iphi());
	      meOCC_MAP_L2_E->Fill(id.ieta(),id.iphi(), HOiter->energy());
	    }
	    else if(id.depth()==3){ 
	      meOCC_MAP_L3->Fill(id.ieta(),id.iphi());
	      meOCC_MAP_L3_E->Fill(id.ieta(),id.iphi(), HOiter->energy());
	    }
	    if(id.depth()==4){ 
	      meOCC_MAP_L4->Fill(id.ieta(),id.iphi());
	      meOCC_MAP_L4_E->Fill(id.ieta(),id.iphi(), HOiter->energy());
	    }
	  }
	  if(doPerChannel_) HcalRecHitPerChan::perChanHists<HORecHit>(2,*HOiter,hoHists.meRECHIT_E,hoHists.meRECHIT_T,m_dbe, baseFolder_);
	}
      }
      //      if(tot>0) hoHists.meRECHIT_E_tot->Fill(tot);
      if(tot>-100) hoHists.meRECHIT_E_tot->Fill(tot);
      if(totThr>0) hoHists.meRECHIT_Ethresh_tot->Fill(totThr);
      all += tot;
      allThr += totThr;
    }
  } catch (...) {    
    if(fVerbosity) printf("HcalRecHitMonitor::processEvent  Error in HO RecHit loop\n");
  }
   if (showTiming)
    { 
      cpu_timer.stop(); std::cout << " TIMER::HcalRecHit RECHIT HO-> " << cpu_timer.cpuTime() << std::endl; 
      cpu_timer.reset(); cpu_timer.start();  
    } 

  try{
    tot=0;  totThr=0;
    if(hfHits.size()>0){
      for (HFiter=hfHits.begin(); HFiter!=hfHits.end(); HFiter++) { // loop over all hits
	//changed to -100 for cosmics
	//	if(HFiter->energy()>0.0){
	if(HFiter->energy()>-100){
	  //Want to see these 3 histos for Long fibers:
	  if (HFiter->id().depth()==1){
	    hfHists.meRECHIT_E_all->Fill(HFiter->energy());
	    hfHists.meRECHIT_E_low->Fill(HFiter->energy());
	    hfHists.meRECHIT_T_all->Fill(HFiter->time());
	  }

	  HcalDetId id(HFiter->detid().rawId());
	  //Fill 3 histos for Short Fibers :
	  if (id.depth()==2){
	    hfshort_meRECHIT_E_all->Fill(HFiter->energy());
	    hfshort_meRECHIT_E_low->Fill(HFiter->energy());
	    hfshort_meRECHIT_T_all->Fill(HFiter->time());
	  }

	  //HF: no non-threshold occupancy map is filled?
	  
	  tot += HFiter->energy();
	  if(HFiter->energy()>occThresh_){
	    totThr += HFiter->energy();
	    hfHists.meOCC_MAPthresh_GEO->Fill(id.ieta(),id.iphi());
	    hfHists.meRECHIT_Tthresh_all->Fill(HFiter->time());
	    
	    meOCC_MAP_ETA->Fill(id.ieta());
	    meOCC_MAP_PHI->Fill(id.iphi());
	    meOCC_MAP_ETA_E->Fill(id.ieta(),HFiter->energy());
	    meOCC_MAP_PHI_E->Fill(id.iphi(),HFiter->energy());
	    
	    hfHists.meOCC_MAP_GEO->Fill(id.ieta(),id.iphi());
	    if(id.depth()==1){ 
	      meOCC_MAP_L1->Fill(id.ieta(),id.iphi());
	      meOCC_MAP_L1_E->Fill(id.ieta(),id.iphi(), HFiter->energy());
	    }
	    else if(id.depth()==2){ 
	      meOCC_MAP_L2->Fill(id.ieta(),id.iphi());
	      meOCC_MAP_L2_E->Fill(id.ieta(),id.iphi(), HFiter->energy());
	    }
	    else if(id.depth()==3){ 
	      meOCC_MAP_L3->Fill(id.ieta(),id.iphi());
	      meOCC_MAP_L3_E->Fill(id.ieta(),id.iphi(), HFiter->energy());
	    }
	    if(id.depth()==4){ 
	      meOCC_MAP_L4->Fill(id.ieta(),id.iphi());
	      meOCC_MAP_L4_E->Fill(id.ieta(),id.iphi(), HFiter->energy());
	    }
	  }
	  if(doPerChannel_) HcalRecHitPerChan::perChanHists<HFRecHit>(3,*HFiter,hfHists.meRECHIT_E,hfHists.meRECHIT_T,m_dbe, baseFolder_);
	}
      }
      //      if(tot>0) hfHists.meRECHIT_E_tot->Fill(tot)
      if(tot>-100) hfHists.meRECHIT_E_tot->Fill(tot);
      if(totThr>0) hfHists.meRECHIT_Ethresh_tot->Fill(totThr);
      all += tot;
      allThr += totThr;
    }
  } catch (...) {    
    if(fVerbosity) cout<<"HcalRecHitMonitor::processEvent  Error in HF RecHit loop"<<endl;
  }

  if (showTiming)
    { 
      cpu_timer.stop(); std::cout << " TIMER::HcalRecHit RECHIT HF-> " << cpu_timer.cpuTime() << std::endl; 
    } 

  //  if(all>0) meRECHIT_E_all->Fill(all);
  if(all>-100) meRECHIT_E_all->Fill(all);
  if(allThr>0) meRECHIT_Ethresh_all->Fill(allThr);

  return;
}


void HcalRecHitMonitor::processZDC(const ZDCRecHitCollection& zdcHits)
{

  if (showTiming)  cpu_timer.reset(); cpu_timer.start();  

  ZDCRecHitCollection::const_iterator ZDCiter;

  //cout <<"ZDC SIZE = "<<zdcHits.size()<<endl;
  try
    {
      if(zdcHits.size()>0)
	{
	  // Each EM channel has a width of ~18 mm in X, extending from -46 -> +46 mm  
	  // There are 5 EM channels.  Assume the mapping is:
	  // channel 0:  -36.8 mm
	  // channel 1:  -18.4 mm
	  // channel 2:  0 mm
	  // channel 3:  +18.4 mm
	  // channel 4:  +36.8 mm
	  // Change this mapping in the future if I've gotten it wrong.

	  float EMmapping[5]={-36.8,-18.4,0.,18.4,+36.8};
	  float weightedXplus=0;
	  float weightedXminus=0;
	  float sumEemplus=0;
	  float sumEemminus=0;
	  float sumEhadplus=0;
	  float sumEhadminus=0;
	  float sumEhadlayerplus[4]={0.};
	  float sumEhadlayerminus[4]={0.};

	  for (ZDCiter=zdcHits.begin(); ZDCiter!=zdcHits.end(); ++ZDCiter) 
	    { // loop over all hits
	      if (ZDCiter->id().section()==0 || ZDCiter->id().section()==3) continue;

	      else if (ZDCiter->id().section()==1)  // found EM hit
		{
		  // positive side first
		  if (ZDCiter->id().zside()==1)
		    {
		      // assume EM depth runs from 0-4; convert to distance using above mapping
		      weightedXplus+=EMmapping[ZDCiter->id().depth()]*ZDCiter->energy();
		      sumEemplus+=ZDCiter->energy();
		    }
		  else // negative side
		    {
		      // assume EM depth runs from 0-4; convert to distance using above mapping
		      weightedXminus+=EMmapping[ZDCiter->id().depth()]*ZDCiter->energy();
		      sumEemminus+=ZDCiter->energy(); 
		    }
		} // else if (ZDCiter->id().section==1)

	      else if (ZDCiter->id().section()==2) // found HAD hit
		{
		  // positive side first
		  if (ZDCiter->id().zside()==1)
		    {
		      sumEhadplus+=ZDCiter->energy();
		      sumEhadlayerplus[ZDCiter->id().depth()]+=ZDCiter->energy();
		    }
		  else // negative side
		    {
		      sumEhadminus+=ZDCiter->energy();
		      sumEhadlayerminus[ZDCiter->id().depth()]+=ZDCiter->energy();
		    }
		}
	      else 
		{ if (fVerbosity) cout <<"HcalRecHitMonitor::processEvent:  Unrecognized ZDC section:  "<<ZDCiter->id().section()<<endl;}

	    } // for (ZDCiter=zdcHits.begin()...)

	  // Finished looping on ZDC hits.  Now form weighted X positions
	  weightedXplus/=sumEemplus;
	  weightedXminus/=sumEemminus;

	  // Fill histograms
	  ZDCtanAlpha->Fill((weightedXplus-weightedXminus)/280000.0); // distance in mm
	  ZDCaverageX->Fill((weightedXplus-weightedXminus)/2.);
	  ZDCxplusVSxminus->Fill(weightedXminus, weightedXplus);
	  ZDChadVSem_plus->Fill(sumEhadplus, sumEemplus);
	  ZDChadVSem_minus->Fill(sumEhadminus, sumEemminus);
	  ZDCenergy_plusVSminus->Fill((sumEemminus+sumEhadminus),(sumEemplus+sumEhadplus));
	  
	  // Fill dEdz plots as TProfiles with weight 1
	  ZDCenergyVSlayer_plus->Fill(0,sumEemplus,1);
	  ZDCenergyVSlayer_plus->Fill(1,sumEhadlayerplus[0],1);
	  ZDCenergyVSlayer_plus->Fill(2,sumEhadlayerplus[1],1);
	  ZDCenergyVSlayer_plus->Fill(3,sumEhadlayerplus[2],1);
	  ZDCenergyVSlayer_plus->Fill(4,sumEhadlayerplus[3],1);
	  	  
	  ZDCenergyVSlayer_minus->Fill(0,sumEemminus,1);
	  ZDCenergyVSlayer_minus->Fill(1,sumEhadlayerminus[0],1);
	  ZDCenergyVSlayer_minus->Fill(2,sumEhadlayerminus[1],1);
	  ZDCenergyVSlayer_minus->Fill(3,sumEhadlayerminus[2],1);
	  ZDCenergyVSlayer_minus->Fill(4,sumEhadlayerminus[3],1);
	  	
	} // if (zdcHits.size()>0)
    } // try
  catch (...)
    { if (fVerbosity) cout <<"HcalRecHitMonitor::processEvent Error in ZDC RecHit loop"<<endl;}




  if (showTiming)
    { 
      cpu_timer.stop(); std::cout << " TIMER::HcalRecHit RECHIT ZDC-> " << cpu_timer.cpuTime() << std::endl; 
    } 

}
