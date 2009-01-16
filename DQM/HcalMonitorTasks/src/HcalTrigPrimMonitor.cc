#include "DQM/HcalMonitorTasks/interface/HcalTrigPrimMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"

HcalTrigPrimMonitor::HcalTrigPrimMonitor() {
  ievt_=0;
  occThresh_=0;
}

HcalTrigPrimMonitor::~HcalTrigPrimMonitor() {
}

void HcalTrigPrimMonitor::reset(){}

void HcalTrigPrimMonitor::clearME(){

  if(m_dbe){
    m_dbe->setCurrentFolder(baseFolder_);
    m_dbe->removeContents();
    meEVT_= 0;
  }
} // void HcalTrigPrimMonitor::clearME()


void HcalTrigPrimMonitor::setup(const edm::ParameterSet& ps, DQMStore* dbe){
  HcalBaseMonitor::setup(ps,dbe);
  baseFolder_ = rootFolder_+"TrigPrimMonitor";

  
  occThresh_ = ps.getUntrackedParameter<double>("TrigPrimMonitor_OccThresh", 1.0);
  TPThresh_ = ps.getUntrackedParameter<double>("TrigPrimMonitor_Threshold", 1.0);

  TPdigi_ = ps.getUntrackedParameter<int>("TrigPrimMonitor_TPdigiTS", 1);
  ADCdigi_ = ps.getUntrackedParameter<int>("TrigPrimMonitor_ADCdigiTS", 3);

  tp_checkNevents_=ps.getUntrackedParameter<int>("TrigPrimMonitor_checkNevents",checkNevents_);
  tp_makeDiagnostics_=ps.getUntrackedParameter<bool>("TrigPrimMonitor_makeDiagnostics",makeDiagnostics);

  ievt_=0;
  
  if ( m_dbe !=NULL ) {    

    char* type;
    //    char name[128];
    m_dbe->setCurrentFolder(baseFolder_);

    //ZZ Expert Plots
    m_dbe->setCurrentFolder(baseFolder_ + "/ZZ Expert Plots/ZZ DQM Expert Plots");
    type = "TrigPrim Event Number";
    meEVT_ = m_dbe->bookInt(type);
    
    // 00 TP Occupancy
    m_dbe->setCurrentFolder(baseFolder_);
    type = "00 TP Occupancy";
    TPOcc_          = m_dbe->book2D(type,type,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    //setupDepthHists2D(TPOcc_,type,"");
    //Timing Plots
    m_dbe->setCurrentFolder(baseFolder_+"/Timing Plots");
    type = "TP Size";
    tpSize_ = m_dbe->book1D(type,type,20,-0.5,19.5);
    type = "TP Timing";
    TPTiming_       = m_dbe->book1D(type,type,10,0,10);
    type = "TP Timing (Top wedges)";
    TPTimingTop_    = m_dbe->book1D(type,type,10,0,10);
    type = "TP Timing (Bottom wedges)";
    TPTimingBot_    = m_dbe->book1D(type,type,10,0,10);
    type = "TS with max ADC";
    TS_MAX_         = m_dbe->book1D(type,type,10,0,10);

    //Energy Plots
    m_dbe->setCurrentFolder(baseFolder_+"/Energy Plots");
    type = "# TP Digis";
    tpCount_ = m_dbe->book1D(type,type,500,-0.5,4999.5);
    type = "# TP Digis over Threshold";
    tpCountThr_ = m_dbe->book1D(type,type,100,-0.5,999.5);
    type = "ADC spectrum positive TP";
    TP_ADC_         = m_dbe->book1D(type,type,200,-0.5,199.5);
    type = "Max ADC in TP";
    MAX_ADC_        = m_dbe->book1D(type,type,20,-0.5,19.5);
    type = "Full TP Spectrum";
    tpSpectrumAll_ = m_dbe->book1D(type,type,200,-0.5,199.5);
    type = "TP ET Sum";
    tpETSumAll_ = m_dbe->book1D(type,type,200,-0.5,199.5);
    type = "TP SOI ET";
    tpSOI_ET_ = m_dbe->book1D(type,type,100,-0.5,99.5);
    
    m_dbe->setCurrentFolder(baseFolder_+"/Energy Plots/TP Spectra by TS");
    for (int i=0; i<10; ++i) {
      type = "TP Spectrum sample ";
      std::stringstream samp;
      std::string teststr;
      samp << i;
      samp >> teststr;
      teststr = type + teststr;
      tpSpectrum_[i]= m_dbe->book1D(teststr,teststr,100,-0.5,99.5);      
    }

    //Electronics Plots
    m_dbe->setCurrentFolder(baseFolder_+"/Electronics Plots");
    type = "HBHE Sliding Pair Sum Maxes";
    me_HBHE_ZS_SlidingSum = m_dbe->book1D(type,type,128,0,128);
    type = "HF Sliding Pair Sum Maxes";
    me_HF_ZS_SlidingSum   = m_dbe->book1D(type,type,128,0,128);
    type = "HO Sliding Pair Sum Maxes";
    me_HO_ZS_SlidingSum   = m_dbe->book1D(type,type,128,0,128);

    type = "TP vs Digi";
    TPvsDigi_       = m_dbe->book2D(type,type,128,0,128,200,0,200);
    TPvsDigi_->setAxisTitle("lin ADC digi",1);
    TPvsDigi_->setAxisTitle("TP digi",2);
    type = "TrigPrim VME Occupancy Map";
    OCC_ELEC_VME = m_dbe->book2D(type,type,40,-0.25,19.75,18,-0.5,17.5);
    type = "TrigPrim Spigot Occupancy Map";
    OCC_ELEC_DCC = m_dbe->book2D(type,type,HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,36,-0.5,35.5);
    type = "TrigPrim VME Energy Map";
    EN_ELEC_VME = m_dbe->book2D(type,type,40,-0.25,19.75,18,-0.5,17.5);
    type = "TrigPrim Spigot Energy Map";
    EN_ELEC_DCC = m_dbe->book2D(type,type,HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,36,-0.5,35.5);

    //Geometry Plots
    m_dbe->setCurrentFolder(baseFolder_+"/Geometry Plots");
    type = "TrigPrim Eta Occupancy Map";
    OCC_ETA = m_dbe->book1D(type,type,etaBins_,etaMin_,etaMax_);
    type = "TrigPrim Phi Occupancy Map";
    OCC_PHI = m_dbe->book1D(type,type,phiBins_,phiMin_,phiMax_);
    type = "TrigPrim Geo Occupancy Map";
    OCC_MAP_ETAPHI = m_dbe->book2D(type,type,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    //setupDepthHists2D(OCC_MAP_ETAPHI,type,"");
    //type = "old TrigPrim Geo Threshold Map";
    //OCC_MAP_GEO = m_dbe->book2D(type,type,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    type = "TrigPrim Geo Threshold Map";
    OCC_MAP_ETAPHI_THR = m_dbe->book2D(type,type,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    //setupDepthHists2D(OCC_MAP_ETAPHI_THR,type,"");
    type = "TrigPrim Eta Energy Map";
    EN_ETA = m_dbe->book1D(type,type,etaBins_,etaMin_,etaMax_);
    type = "TrigPrim Phi Energy Map";
    EN_PHI = m_dbe->book1D(type,type,phiBins_,phiMin_,phiMax_);
    type = "TrigPrim Geo Energy Map";
    EN_MAP_ETAPHI = m_dbe->book2D(type,type,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    //setupDepthHists2D(EN_MAP_ETAPHI,type,"");
    meEVT_->Fill(ievt_);

  } // if (m_dbe !=NULL)

  return;
}

void HcalTrigPrimMonitor::processEvent(const HBHERecHitCollection& hbHits, 
				       const HORecHitCollection& hoHits, 
				       const HFRecHitCollection& hfHits,
				       const HBHEDigiCollection& hbhedigi,
				       const HODigiCollection& hodigi,
				       const HFDigiCollection& hfdigi,
                                       const HcalTrigPrimDigiCollection& tpDigis,
				       const HcalElectronicsMap& emap
				       )
{

  if(!m_dbe) { 
    if (fVerbosity>0) cout <<"HcalTrigPrimMonitor::processEvent   DQMStore not instantiated!!!"<<endl;  
    return; 
  }

  ++ievt_;
  meEVT_->Fill(ievt_);
  
  //XXX// tpCount_->Fill(tpDigis.size()*1.0);  // number of TPGs collected per event
  ++val_tpCount_[tpDigis.size()];

  float data[10];
  ClearEvent();

  edm::Handle<HcalTrigPrimDigiCollection> Tr_hbhe;
    
    
  int TPGsOverThreshold = 0;
  //int iDepth;
  int iEta;
  int iPhi;
  for (HcalTrigPrimDigiCollection::const_iterator j=tpDigis.begin(); j!=tpDigis.end(); ++j)
    {
      const HcalTriggerPrimitiveDigi digi = (const HcalTriggerPrimitiveDigi)(*j);
      
      
      // find corresponding rechit and digis
      HcalTrigTowerDetId tpid=digi.id();
      //iDepth=digi.id().depth()-1;
      /*
	iDepth=0; // HcalTrigTowerDetId only has one depth?
	if (iDepth<2 && (HcalSubdetector)(digi.id().subdet())==HcalEndcap)
	iDepth+=4; // shift HE depths 1 and 2
      */
      iEta=tpid.ieta();
      iPhi=tpid.iphi();
      HcalElectronicsId eid = emap.lookupTrigger(tpid);
      
      //XXX// tpSOI_ET_->Fill(digi.SOI_compressedEt());
      ++val_tpSOI_ET_[static_cast<int>(digi.SOI_compressedEt())];
      //if(digi.SOI_compressedEt()>0 || true) // digi.SOI_compressedEt() check does nothing here -- do we only want to fill when SOI_compressedEt >0?
      {	
	//XXX// tpSize_->Fill(digi.size());	
	++val_tpSize_[static_cast<int>(digi.size())];
	
	//XXX//OCC_ETA->Fill(tpid.ieta());
	++val_OCC_ETA[static_cast<int>(iEta+(etaBins_-2)/2)];
	
	//XXX//OCC_PHI->Fill(tpid.iphi());
	++val_OCC_PHI[iPhi-1];
	//XXX//OCC_MAP_ETAPHI->Fill(tpid.ieta(), tpid.iphi());
	++val_OCC_MAP_ETAPHI[static_cast<int>(iEta+(etaBins_-2)/2)][iPhi-1];
	
	
	//XXX//EN_ETA->Fill(tpid.ieta(),digi.SOI_compressedEt());
	//XXX//EN_PHI->Fill(tpid.iphi(),digi.SOI_compressedEt());
	//XXX//EN_MAP_ETAPHI->Fill(tpid.ieta(), tpid.iphi(),digi.SOI_compressedEt());
	val_EN_ETA[static_cast<int>(iEta+(etaBins_-2)/2)]+=digi.SOI_compressedEt();
	val_EN_PHI[iPhi-1]+=digi.SOI_compressedEt();
	val_EN_MAP_ETAPHI[static_cast<int>(iEta+(etaBins_-2)/2)][iPhi-1]+=digi.SOI_compressedEt();
	
	float slotnum = eid.htrSlot() + 0.5*eid.htrTopBottom();	
	//XXX//OCC_ELEC_VME->Fill(slotnum,eid.readoutVMECrateId());
	//XXX//OCC_ELEC_DCC->Fill(eid.spigot(),eid.dccid());
	
	++val_OCC_ELEC_VME[static_cast<int>(2*(slotnum))][static_cast<int>(eid.readoutVMECrateId())];
	++val_OCC_ELEC_DCC[static_cast<int>(eid.spigot())][static_cast<int>(eid.dccid())];
	
	//XXX//EN_ELEC_VME->Fill(slotnum,eid.readoutVMECrateId(),digi.SOI_compressedEt());
	//XXX//EN_ELEC_DCC->Fill(eid.spigot(),eid.dccid(),digi.SOI_compressedEt());
	val_EN_ELEC_VME[static_cast<int>(2*(slotnum))][static_cast<int>(eid.readoutVMECrateId())]+=digi.SOI_compressedEt();
	val_EN_ELEC_DCC[static_cast<int>(eid.spigot())][static_cast<int>(eid.dccid())]+=digi.SOI_compressedEt();
	  
	double etSum = 0;
	bool threshCond = false;
	
	for (int j=0; j<digi.size(); ++j) 
	  {
	    etSum += digi.sample(j).compressedEt();
	    if (digi.sample(j).compressedEt()>occThresh_) threshCond = true;
	  }
	
	//XXX//tpETSumAll_->Fill(etSum);
	++val_tpETSumAll_[static_cast<int>(etSum)];
	if (threshCond)
	  {
	    //OCC_MAP_GEO->Fill(tpid.ieta(),tpid.iphi());  // which ieta and iphi positions the TPGs have for overThreshold cut
	    ++val_OCC_MAP_ETAPHI_THR[static_cast<int>(iEta+(etaBins_-2)/2)][iPhi-1];
	    ++TPGsOverThreshold;
	  }
      } // if (digi.SOI_compressedEt() || true) -- defunct loop condition
      
	/**************/
      for (int i=0; i<digi.size(); ++i) 
	{
	  data[i]=digi.sample(i).compressedEt();
	  if(digi.sample(i).compressedEt()>TPThresh_)
	    {
	      //XXX//tpSpectrum_[i]->Fill(digi.sample(i).compressedEt());
	      //XXX//tpSpectrumAll_->Fill(digi.sample(i).compressedEt());
	      ++val_tpSpectrum_[i][static_cast<int>(digi.sample(i).compressedEt())];
	      ++val_tpSpectrumAll_[static_cast<int>(digi.sample(i).compressedEt())];
	      
	      //XXX//TPTiming_->Fill(i);
	      ++val_TPTiming_[i];
	      if(digi.id().iphi()>1  && digi.id().iphi()<36) 
		{
		  //TPTimingTop_->Fill(i);
		  ++val_TPTimingTop_[i];
		}
	      if(digi.id().iphi()>37 && digi.id().iphi()<72) 
		{
		  //TPTimingBot_->Fill(i);
		  ++val_TPTimingBot_[i];
		}
	      //TPOcc_->Fill(digi.id().ieta(),digi.id().iphi());
	      ++val_TPOcc_[static_cast<int>(iEta+(etaBins_-2)/2)][iPhi-1];
	    }
	  //XXX//TPOcc_->Fill(digi.id().ieta(),digi.id().iphi());
	}
      set_tp(digi.id().ieta(),digi.id().iphi(),1,data);
      /*************/
      
    } // for (HcalTrigPrimDigiCollection...)
  
  //XXX//tpCountThr_->Fill(TPGsOverThreshold*1.0);  // number of TPGs collected per event
  ++val_tpCountThr_[TPGsOverThreshold];

  for(HBHEDigiCollection::const_iterator j=hbhedigi.begin(); j!=hbhedigi.end(); ++j)
    {
      HBHEDataFrame digi = (const HBHEDataFrame)(*j);
      for(int i=0; i<digi.size(); ++i) {
	data[i]=digi.sample(i).adc();
	if (i==0) maxsum = data[i];
	else if ((data[i] + data[i-1] ) > maxsum) 
	  maxsum = data[i] + data[i-1];
      }
      set_adc(digi.id().ieta(),digi.id().iphi(),digi.id().depth(),data);
      //XXX//me_HBHE_ZS_SlidingSum->Fill(maxsum);
      if (maxsum>=0 && maxsum<128)
	++val_HBHE_ZS_SlidingSum[static_cast<int>(maxsum)];
    } // for (HBHEDigiCollection...)
  

  for(HFDigiCollection::const_iterator j=hfdigi.begin(); j!=hfdigi.end(); ++j){
    HFDataFrame digi = (const HFDataFrame)(*j);
    for(int i=0; i<digi.size(); ++i) {
      data[i]=digi.sample(i).adc();
      if (i==0) maxsum = data[i];
      else if ((data[i] + data[i-1] ) > maxsum) 
	maxsum = data[i] + data[i-1];
    }
    //set_adc(digi.id().ieta(),digi.id().iphi(),digi.id().depth(),data);
    //XXX//me_HF_ZS_SlidingSum->Fill(maxsum);
    if (maxsum>=0 && maxsum<128)
      ++val_HF_ZS_SlidingSum[static_cast<int>(maxsum)];
  } // for (HFDigiCollection...)

  for(HODigiCollection::const_iterator j=hodigi.begin(); j!=hodigi.end(); ++j){
    HODataFrame digi = (const HODataFrame)(*j);
    for(int i=0; i<digi.size(); ++i) {
      data[i]=digi.sample(i).adc();
      if (i==0) maxsum = data[i];
      else if ((data[i] + data[i-1] ) > maxsum) 
	maxsum = data[i] + data[i-1];
    }
    //set_adc(digi.id().ieta(),digi.id().iphi(),digi.id().depth(),data);
    //XXX//me_HO_ZS_SlidingSum->Fill(maxsum);
    if (maxsum>=0 && maxsum<128)
      ++val_HO_ZS_SlidingSum[static_cast<int>(maxsum)];
  } // for (HODigiCollection...)
  
  // Correlation plots...
  int eta,phi;
  int tempsum;
  float tpval;
  for(eta=-16;eta<=16;++eta) 
    {for(phi=1;phi<=72;++phi)
	{
	  //for(int i=1;i<10;++i){ // not sure what this loop was supposed to accomplish
	  int j1=(int)get_adc(eta,phi,1)[ADCdigi_];
	  float tmp11 = (TrigMonAdc2fc[j1]+0.5);
	  int j2=(int)get_adc(eta,phi,1)[ADCdigi_+1];
	  float tmp21 = (TrigMonAdc2fc[j2]+0.5);
	  int j3=(int)get_adc(eta,phi,2)[ADCdigi_];
	  float tmp12 = (TrigMonAdc2fc[j3]+0.5);
	  int j4=(int)get_adc(eta,phi,2)[ADCdigi_+1];
	  float tmp22 = (TrigMonAdc2fc[j4]+0.5);
	  if(IsSet_adc(eta,phi,1) && IsSet_tp(eta,phi,1))
	    {
	      if(get_tp(eta,phi)[TPdigi_]>TPThresh_)
		{ 
		  tpval=get_tp(eta,phi)[TPdigi_];
		  //XXX//TPvsDigi_->Fill(tmp11+tmp21+tmp12+tmp22,get_tp(eta,phi)[TPdigi_]);
		  tempsum=static_cast<int>(tmp11+tmp21+tmp12+tmp22);
		  if (tempsum>=0 && tempsum<128 && tpval>=0 && tpval<200)
		    ++val_TPvsDigi_[static_cast<int>(tmp11+tmp21+tmp12+tmp22)][static_cast<int>(tpval)];
		  float Energy=0;
		  int TS = 0;
		  for(int j=0;j<10;++j)
		    {
		      if (get_adc(eta,phi,1)[j]>Energy)
			{
			  Energy=get_adc(eta,phi,1)[j];
			  TS = j;
			}
		    } // for (int j=0;j<10;++j)
		  //XXX//MAX_ADC_->Fill(Energy);
		  ++val_MAX_ADC_[static_cast<int>(Energy)];
		  //XXX//TS_MAX_->Fill(TS);
		  ++val_MAX_ADC_[static_cast<int>(TS)];
		  //This may need to continue?
		  Energy=0; 
		  for(int j=0;j<10;++j) 
		    Energy+=get_adc(eta,phi,1)[j]; 
		  //XXX//TP_ADC_->Fill(Energy);
		  ++val_TP_ADC_[static_cast<int>(Energy)];
		} // if (get_tp(eta,phi)[TPdigi_]>TPThresh_)
	    } // if (IsSet_adc(...))
	} //for (eta=-16;...)
    } // for (phi=1;...)


  if (ievt_%tp_checkNevents_==0)
    fill_Nevents();
  return;
}


void HcalTrigPrimMonitor::fill_Nevents()
{

  for (int i=0;i<200;++i)
    {
      if (val_tpSpectrumAll_[i])
	tpSpectrumAll_->setBinContent(i+1,val_tpSpectrumAll_[i]);
      if (val_tpETSumAll_[i])
	tpETSumAll_->setBinContent(i+1,val_tpETSumAll_[i]);
      if (val_TP_ADC_[i])
	TP_ADC_->setBinContent(i+1,val_TP_ADC_[i]);
      for (int j=0;j<10;++j)
	{
	  if (val_tpSpectrum_[j][i])
	    tpSpectrum_[j]->setBinContent(i+1,val_tpSpectrum_[j][i]);
	}
    }

  for (int i=0;i<5000;++i)
    {
      if ( val_tpCount_[i])
	tpCount_->setBinContent(i+1, val_tpCount_[i]);
    }

  for (int i=0;i<1000;++i)
    {
      if ( val_tpCountThr_[i])
	tpCountThr_->setBinContent(i+1, val_tpCountThr_[i]);
    }

  for (int i=0;i<20;++i)
    {
      if (val_tpSize_[i])
	tpSize_->setBinContent(i+1,val_tpSize_[i]);
      if ( val_MAX_ADC_[i])
	MAX_ADC_->setBinContent(i+1,val_MAX_ADC_[i]);
    }

  for (int i=0;i<100;++i)
    {
      if (val_tpSOI_ET_[i])
	tpSOI_ET_->setBinContent(i+1,val_tpSOI_ET_[i]);
    }

  for (int i=0;i<10;++i)
    {
      if (val_TPTiming_[i])
	TPTiming_->setBinContent(i+1,val_TPTiming_[i]);
      if (val_TPTimingTop_[i])
	TPTimingTop_->setBinContent(i+1,val_TPTimingTop_[i]);
      if (val_TPTimingBot_[i])
	TPTimingBot_->setBinContent(i+1,val_TPTimingBot_[i]);
      if (val_TS_MAX_[i])
	TS_MAX_->setBinContent(i+1,val_TS_MAX_[i]);
    }

  for (int i=0;i<128;++i)
    {
      if (val_HBHE_ZS_SlidingSum[i])
	me_HBHE_ZS_SlidingSum->setBinContent(i+1,val_HBHE_ZS_SlidingSum[i]);
      if (val_HO_ZS_SlidingSum[i])
	me_HO_ZS_SlidingSum->setBinContent(i+1,val_HO_ZS_SlidingSum[i]);
      if (val_HF_ZS_SlidingSum[i])
	me_HF_ZS_SlidingSum->setBinContent(i+1,val_HF_ZS_SlidingSum[i]);
      for (int j=0;j<200;++j)
	{
	  if (val_TPvsDigi_[i][j])
	    TPvsDigi_->setBinContent(i+1,j+1,val_TPvsDigi_[i][j]);
	}
    }

  // Eta, phi histograms have extra empty bin on either side, so we
  // need to add +2 (rather than +1) when setting bin content
  for (int i=0;i<87;++i)
    {
      if (val_OCC_ETA[i])
	OCC_ETA->setBinContent(i+2,val_OCC_ETA[i]);
      if (val_EN_ETA[i])
	EN_ETA->setBinContent(i+2,val_EN_ETA[i]);

      for (int j=0;j<72;++j)
	{
	  if (i==0)
	    {
	      if (val_OCC_PHI[j])
		OCC_PHI->setBinContent(j+2,val_OCC_PHI[j]);
	      if (val_EN_PHI[j])
		EN_PHI->setBinContent(j+2,val_EN_PHI[j]);
	    }
	  if (val_TPOcc_[i][j])
	    TPOcc_->setBinContent(i+2,j+2,val_TPOcc_[i][j]);
	  if (val_EN_MAP_ETAPHI[i][j])
	    EN_MAP_ETAPHI->setBinContent(i+2,j+2,val_EN_MAP_ETAPHI[i][j]);
	  if (val_OCC_MAP_ETAPHI[i][j])
	    OCC_MAP_ETAPHI->setBinContent(i+2,j+2,val_OCC_MAP_ETAPHI[i][j]);
	  if (val_OCC_MAP_ETAPHI_THR[i][j])
	    OCC_MAP_ETAPHI_THR->setBinContent(i+2,j+2,val_OCC_MAP_ETAPHI_THR[i][j]);
	}
    }

  for (int i=0;i<40;++i)
    {
      for (int j=0;j<18;++j)
	{
	  if (val_OCC_ELEC_VME[i][j])
	    OCC_ELEC_VME->setBinContent(i+1,j+1,val_OCC_ELEC_VME[i][j]);
	  if (val_EN_ELEC_VME[i][j])
	    EN_ELEC_VME->setBinContent(i+1,j+1,val_EN_ELEC_VME[i][j]);
	}
    }


  // Fill VME plots
  for (int i=0;i<HcalDCCHeader::SPIGOT_COUNT;++i)
    {
        for (int j=0;j<36;++j)
	{
	  if (val_OCC_ELEC_DCC[i][j])
	    OCC_ELEC_DCC->setBinContent(i+1,j+1,val_OCC_ELEC_DCC[i][j]);
	  if (val_EN_ELEC_DCC[i][j])
	    EN_ELEC_DCC->setBinContent(i+1,j+1,val_EN_ELEC_DCC[i][j]);
	}
    }
} //void HcalTrigPrimMonitor::Fill_Nevents()
