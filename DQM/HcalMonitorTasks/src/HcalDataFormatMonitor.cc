#include "DQM/HcalMonitorTasks/interface/HcalDataFormatMonitor.h"

HcalDataFormatMonitor::HcalDataFormatMonitor() {}

HcalDataFormatMonitor::~HcalDataFormatMonitor() {
}

void HcalDataFormatMonitor::clearME(){
  if(m_dbe){
    m_dbe->setCurrentFolder("HcalMonitor/DataFormatMonitor");
    m_dbe->removeContents();
  }
  return;
}

void HcalDataFormatMonitor::setup(const edm::ParameterSet& ps,
				  DaqMonitorBEInterface* dbe){
  HcalBaseMonitor::setup(ps,dbe);
  
  ievt_=0;
  fedUnpackList_ = ps.getParameter<vector<int> >("FEDs");
  firstFED_ = ps.getParameter<int>("HcalFirstFED");
  cout << "HcalDataFormatMonitor::setup  Will unpack FEDs ";
  for (unsigned int i=0; i<fedUnpackList_.size(); i++)
    cout << fedUnpackList_[i] << " ";
  cout << endl;
  
  if ( m_dbe ) {
    m_dbe->setCurrentFolder("HcalMonitor/DataFormatMonitor");
    
    meEVT_ = m_dbe->bookInt("Data Format Task Event Number");
    meEVT_->Fill(ievt_);
    
    char* type = "Spigot Format Errors";
    meSpigotFormatErrors_=  m_dbe->book1D(type,type,500,0,1000);
    type = "Bad Quality Digis";
    meBadQualityDigis_=  m_dbe->book1D(type,type,500,0,1000);
    type = "Unmapped Digis";
    meUnmappedDigis_=  m_dbe->book1D(type,type,500,0,1000);
    type = "Unmapped Trigger Primitive Digis";
    meUnmappedTPDigis_=  m_dbe->book1D(type,type,500,0,1000);
    type = "FED Error Map";
    meFEDerrorMap_ = m_dbe->book1D(type,type,33,699.5,732.5);
    type = "Evt Number Out-of-Synch";
    meEvtNumberSynch_= m_dbe->book2D(type,type,
				     HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				     33,699.5,732.5);

    
    type = "HB Data Format Error Word";
    hbHists.DCC_ErrWd =  m_dbe->book1D(type,type,16,-0.5,15.5);
    type = "HB ExtHeader5";
    hbHists.ExtHeader5 =  m_dbe->book1D(type,type,16,-0.5,15.5);
    type = "HB ExtHeader7";
    hbHists.ExtHeader7 =  m_dbe->book1D(type,type,16,-0.5,15.5);
    type = "HB Data Format Crate Error Map";
    hbHists.CrateMap = m_dbe->book2D(type,type,21,-0.5,20.5,21,-0.5,20.5);
    type = "HB Data Format Spigot Error Map";
    hbHists.SpigotMap = m_dbe->book2D(type,type,				      
				      HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				      33,699.5,732.5);
    type = "HB LE Error Map";
    hbHists.LEMap = m_dbe->book2D(type,type,				  
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);
    type = "HB LW Error Map";
    hbHists.LWMap = m_dbe->book2D(type,type,				  
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);
    type = "HB OD Error Map";
    hbHists.ODMap = m_dbe->book2D(type,type,				  
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);    
    type = "HB CK Error Map";
    hbHists.CKMap = m_dbe->book2D(type,type,				  
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);
    type = "HB OW Error Map";
    hbHists.OWMap = m_dbe->book2D(type,type,      
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);
    type = "HB BZ Error Map";
    hbHists.BZMap = m_dbe->book2D(type,type,      
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);
    type = "HB EE Error Map";
    hbHists.EEMap = m_dbe->book2D(type,type,      
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);
    type = "HB RL Error Map";
    hbHists.RLMap = m_dbe->book2D(type,type,      
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);
    type = "HB BE Error Map";
    hbHists.BEMap = m_dbe->book2D(type,type,      
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);

    type = "HE Data Format Error Word";
    heHists.DCC_ErrWd =  m_dbe->book1D(type,type,16,-0.5,15.5);
    type = "HE ExtHeader5";
    heHists.ExtHeader5 =  m_dbe->book1D(type,type,16,-0.5,15.5);
    type = "HE ExtHeader7";
    heHists.ExtHeader7 =  m_dbe->book1D(type,type,16,-0.5,15.5);
    type = "HE Data Format Crate Error Map";
    heHists.CrateMap = m_dbe->book2D(type,type,21,-0.5,20.5,21,-0.5,20.5);
    type = "HE Data Format Spigot Error Map";
    heHists.SpigotMap = m_dbe->book2D(type,type,				      
				      HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				      33,699.5,732.5);
    type = "HE LE Error Map";
    heHists.LEMap = m_dbe->book2D(type,type,				  
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);
    type = "HE LW Error Map";
    heHists.LWMap = m_dbe->book2D(type,type,				  
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);
    type = "HE OD Error Map";
    heHists.ODMap = m_dbe->book2D(type,type,				  
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);    
    type = "HE CK Error Map";
    heHists.CKMap = m_dbe->book2D(type,type,				  
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);
    type = "HE OW Error Map";
    heHists.OWMap = m_dbe->book2D(type,type,      
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);
    type = "HE BZ Error Map";
    heHists.BZMap = m_dbe->book2D(type,type,      
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);
    type = "HE EE Error Map";
    heHists.EEMap = m_dbe->book2D(type,type,      
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);
    type = "HE RL Error Map";
    heHists.RLMap = m_dbe->book2D(type,type,      
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);
    type = "HE BE Error Map";
    heHists.BEMap = m_dbe->book2D(type,type,      
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);

    type = "HF Data Format Error Word";
    hfHists.DCC_ErrWd =  m_dbe->book1D(type,type,16,-0.5,15.5);
    type = "HF ExtHeader5";
    hfHists.ExtHeader5 =  m_dbe->book1D(type,type,16,-0.5,15.5);
    type = "HF ExtHeader7";
    hfHists.ExtHeader7 =  m_dbe->book1D(type,type,16,-0.5,15.5);
    type = "HF Data Format Crate Error Map";
    hfHists.CrateMap = m_dbe->book2D(type,type,21,-0.5,20.5,21,-0.5,20.5);
    type = "HF Data Format Spigot Error Map";
    hfHists.SpigotMap = m_dbe->book2D(type,type,				      
				      HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				      33,699.5,732.5);
    type = "HF LE Error Map";
    hfHists.LEMap = m_dbe->book2D(type,type,				  
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);
    type = "HF LW Error Map";
    hfHists.LWMap = m_dbe->book2D(type,type,				  
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);
    type = "HF OD Error Map";
    hfHists.ODMap = m_dbe->book2D(type,type,				  
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);    
    type = "HF CK Error Map";
    hfHists.CKMap = m_dbe->book2D(type,type,				  
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);    
    type = "HF OW Error Map";
    hfHists.OWMap = m_dbe->book2D(type,type,      
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);
    type = "HF BZ Error Map";
    hfHists.BZMap = m_dbe->book2D(type,type,      
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);
    type = "HF EE Error Map";
    hfHists.EEMap = m_dbe->book2D(type,type,      
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);
    type = "HF RL Error Map";
    hfHists.RLMap = m_dbe->book2D(type,type,      
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);
    type = "HF BE Error Map";
    hfHists.BEMap = m_dbe->book2D(type,type,      
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);


    
    type = "HO ExtHeader5";
    hoHists.ExtHeader5 = m_dbe->book1D(type,type,16,-0.5,15.5);

    type = "HO ExtHeader7";
    hoHists.ExtHeader7 = m_dbe->book1D(type,type,16,-0.5,15.5);

    type = "HO Data Format Error Word";
    hoHists.DCC_ErrWd = m_dbe->book1D(type,type,16,-0.5,15.5);

    type = "HO Data Format Crate Error Map";
    hoHists.CrateMap = m_dbe->book2D(type,type,21,-0.5,20.5,21,-0.5,20.5);

    type = "HO Data Format Spigot Error Map";
    hoHists.SpigotMap = m_dbe->book2D(type,type,				      
				      HcalDCCHeader::SPIGOT_COUNT,0,HcalDCCHeader::SPIGOT_COUNT-1,
				      33,699.5,732.5);

    type = "HO LE Error Map";
    hoHists.LEMap = m_dbe->book2D(type,type,				  
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);
    type = "HO LW Error Map";
    hoHists.LWMap = m_dbe->book2D(type,type,				  
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);
    type = "HO OD Error Map";
    hoHists.ODMap = m_dbe->book2D(type,type,				  
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);    
    type = "HO CK Error Map";
    hoHists.CKMap = m_dbe->book2D(type,type,				  
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);
    type = "HO OW Error Map";
    hoHists.OWMap = m_dbe->book2D(type,type,      
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);
    type = "HO BZ Error Map";
    hoHists.BZMap = m_dbe->book2D(type,type,      
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);
    type = "HO EE Error Map";
    hoHists.EEMap = m_dbe->book2D(type,type,      
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);
    type = "HO RL Error Map";
    hoHists.RLMap = m_dbe->book2D(type,type,      
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);
    type = "HO BE Error Map";
    hoHists.BEMap = m_dbe->book2D(type,type,      
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  33,699.5,732.5);
    
   }

   return;
}

void HcalDataFormatMonitor::processEvent(const FEDRawDataCollection&
					 rawraw, const HcalUnpackerReport& report, 
					 const HcalElectronicsMap& emap){
  
  if(!m_dbe) { 
    printf("HcalDataFormatMonitor::processEvent DaqMonitorBEInterface not instantiated!!!\n");  
    return;
  }
  
  ievt_++;
  meEVT_->Fill(ievt_);
  
  lastEvtN_ = -1;
  for (vector<int>::const_iterator i=fedUnpackList_.begin();
       i!=fedUnpackList_.end(); i++) {
    const FEDRawData& fed = rawraw.FEDData(*i);
    unpack(fed,emap);
  }
  
  meSpigotFormatErrors_->Fill(report.spigotFormatErrors());
  meBadQualityDigis_->Fill(report.badQualityDigis());
  meUnmappedDigis_->Fill(report.unmappedDigis());
  meUnmappedTPDigis_->Fill(report.unmappedTPDigis());
  
  for(unsigned int i=0; i<report.getFedsError().size(); i++){
    const int m = report.getFedsError()[i];
    const FEDRawData& fed = rawraw.FEDData(m);
    const HcalDCCHeader* dccHeader=(const HcalDCCHeader*)(fed.data());
    if(!dccHeader) continue;
    int dccid=dccHeader->getSourceId();
    meFEDerrorMap_->Fill(dccid);
  }
  
   return;
}

void HcalDataFormatMonitor::unpack(const FEDRawData& raw, const
				   HcalElectronicsMap& emap){
  
  // get the DataFormat header
  const HcalDCCHeader* dccHeader=(const HcalDCCHeader*)(raw.data());
  // check the summary status
  if(!dccHeader) return;
  int dccid=dccHeader->getSourceId();

  // walk through the HTR data...
  HcalHTRData htr;
  
  for (int spigot=0; spigot<HcalDCCHeader::SPIGOT_COUNT; spigot++) {
    
    if (!dccHeader->getSpigotPresent(spigot)) continue;
    dccHeader->getSpigotData(spigot,htr);
    
    // check
    if (!htr.check() || htr.isHistogramEvent()) continue;
    
    ///check that all HTRs have the same L1A number
    if(lastEvtN_==-1) lastEvtN_ = htr.getL1ANumber();  ///the first one will be the reference
    else{
      if(htr.getL1ANumber()!=lastEvtN_) meEvtNumberSynch_->Fill(spigot,dccid);
    }
    
    MonitorElement* tmpErr = 0;
    MonitorElement* tmpHead5 = 0; MonitorElement* tmpHead7 = 0;
    MonitorElement* tmpMapCr = 0; MonitorElement* tmpMapSpig = 0;

    MonitorElement* tmpMapOW = 0; MonitorElement* tmpMapBZ = 0;
    MonitorElement* tmpMapEE = 0; MonitorElement* tmpMapRL = 0;
    MonitorElement* tmpMapLE = 0; MonitorElement* tmpMapLW = 0;
    MonitorElement* tmpMapOD = 0; MonitorElement* tmpMapCK = 0;
    MonitorElement* tmpMapBE = 0;


    bool valid = false;
    for(int fchan=0; fchan<3 && !valid; fchan++){
      for(int fib=0; fib<9 && !valid; fib++){
	HcalElectronicsId eid(fchan,fib,spigot,dccid-firstFED_);
	eid.setHTR(htr.readoutVMECrateId(),htr.htrSlot(),htr.htrTopBottom());
	DetId did=emap.lookup(eid);
	if (!did.null()) {
	  switch (((HcalSubdetector)did.subdetId())) {
	  case (HcalBarrel): {
	    tmpErr = hbHists.DCC_ErrWd; tmpMapCr = hbHists.CrateMap;
	    tmpMapSpig = hbHists.SpigotMap;
	    tmpHead5 = hbHists.ExtHeader5;  tmpHead7 = hbHists.ExtHeader7;
	    tmpMapLW = hbHists.LWMap;    tmpMapLE = hbHists.LEMap; 
	    tmpMapOD = hbHists.ODMap;    tmpMapCK = hbHists.CKMap; 
	    tmpMapOW = hbHists.OWMap;    tmpMapBZ = hbHists.BZMap;
	    tmpMapEE = hbHists.EEMap;    tmpMapRL = hbHists.RLMap;
	    tmpMapBE = hbHists.BEMap;
	    valid = true;
	  } break;
	  case (HcalEndcap): {
	    tmpErr = heHists.DCC_ErrWd; tmpMapCr = heHists.CrateMap;
	    tmpMapSpig = heHists.SpigotMap;
	    tmpHead5 = heHists.ExtHeader5;  tmpHead7 = heHists.ExtHeader7;
	    tmpMapLW = heHists.LWMap;    tmpMapLE = heHists.LEMap; 
	    tmpMapOD = heHists.ODMap;    tmpMapCK = heHists.CKMap; 
	    tmpMapOW = heHists.OWMap;    tmpMapBZ = heHists.BZMap;
	    tmpMapEE = heHists.EEMap;    tmpMapRL = heHists.RLMap;
	    tmpMapBE = heHists.BEMap;
	    valid = true;
	  } break;
	  case (HcalOuter): {
	    tmpErr = hoHists.DCC_ErrWd; tmpMapCr = hoHists.CrateMap;
	    tmpMapSpig = hoHists.SpigotMap;
	    tmpHead5 = hoHists.ExtHeader5;  tmpHead7 = hoHists.ExtHeader7;
	    tmpMapLW = hoHists.LWMap;    tmpMapLE = hoHists.LEMap; 
	    tmpMapOD = hoHists.ODMap;    tmpMapCK = hoHists.CKMap; 
	    tmpMapOW = hoHists.OWMap;    tmpMapBZ = hoHists.BZMap;
	    tmpMapEE = hoHists.EEMap;    tmpMapRL = hoHists.RLMap;
	    tmpMapBE = hoHists.BEMap;
	    valid = true;
	  } break;
	  case (HcalForward): {
	    tmpErr = hfHists.DCC_ErrWd; 
	    tmpMapCr = hfHists.CrateMap;    tmpMapSpig = hfHists.SpigotMap;
	    tmpHead5 = hfHists.ExtHeader5;  tmpHead7 = hfHists.ExtHeader7;
	    tmpMapLW = hfHists.LWMap;    tmpMapLE = hfHists.LEMap; 
	    tmpMapOD = hfHists.ODMap;    tmpMapCK = hfHists.CKMap; 
	    tmpMapOW = hfHists.OWMap;    tmpMapBZ = hfHists.BZMap;
	    tmpMapEE = hfHists.EEMap;    tmpMapRL = hfHists.RLMap;
	    tmpMapBE = hfHists.BEMap;
	    valid = true;
	  } break;
	  default: break;
	  }
	}
      }
    }
     
     int errWord = htr.getErrorsWord() & 0x1FFF;
     if(tmpErr!=NULL){
       for(int i=0; i<16; i++){
	 int errbit = errWord&(0x01<<i);
	 if (errbit !=0) tmpErr->Fill(i);
       }
     }

     int exthead5 = htr.getExtHeader5();
     if(tmpHead5!=NULL){
       for(int i=0; i<16; i++){
	 int errbit = exthead5&(0x01<<i);
	 if (errbit !=0) tmpHead5->Fill(i);
       }
     }

     int exthead7 = htr.getExtHeader7();
     if(tmpHead7!=NULL){
       for(int i=0; i<16; i++){
	 int errbit = exthead7&(0x01<<i);
	 if (errbit !=0) tmpHead7->Fill(i);
       }
     }
     
     if(errWord>0 && tmpMapCr!=NULL){
       tmpMapCr->Fill(htr.readoutVMECrateId(),htr.htrSlot());
       tmpMapSpig->Fill(spigot,dccid);  
       if ((errWord&0x01) != 0) tmpMapOW->Fill(spigot,dccid);
       if ((errWord&0x02) != 0) tmpMapBZ->Fill(spigot,dccid);
       if ((errWord&0x04) != 0) tmpMapEE->Fill(spigot,dccid);
       if ((errWord&0x08) != 0) tmpMapRL->Fill(spigot,dccid);
       if ((errWord&0x10) != 0) tmpMapLE->Fill(spigot,dccid);
       if ((errWord&0x20) != 0) tmpMapLW->Fill(spigot,dccid);
       if ((errWord&0x40) != 0) tmpMapOD->Fill(spigot,dccid);
       if ((errWord&0x80) != 0) tmpMapCK->Fill(spigot,dccid);
       if ((errWord&0x100) != 0) tmpMapBE->Fill(spigot,dccid);
     }

   }

   return;
}



