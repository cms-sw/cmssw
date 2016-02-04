#include "DCCDataMapper.h"
 
/*--------------------------------------------*/
/* DCCTBDataMapper::DCCTBDataMapper               */
/* class constructor                          */
/*--------------------------------------------*/
DCCTBDataMapper::DCCTBDataMapper( DCCTBDataParser * myParser)
: parser_(myParser){
  
  dccFields_        = new std::set<DCCTBDataField  * , DCCTBDataFieldComparator>;
  emptyEventFields_ = new std::set<DCCTBDataField  * , DCCTBDataFieldComparator>;
  
  tcc68Fields_    = new std::set<DCCTBDataField  * , DCCTBDataFieldComparator>; 
  tcc32Fields_    = new std::set<DCCTBDataField  * , DCCTBDataFieldComparator>;
  tcc16Fields_    = new std::set<DCCTBDataField  * , DCCTBDataFieldComparator>;

  srp68Fields_    = new std::set<DCCTBDataField  * , DCCTBDataFieldComparator>;
  srp32Fields_    = new std::set<DCCTBDataField  * , DCCTBDataFieldComparator>;
  srp16Fields_    = new std::set<DCCTBDataField  * , DCCTBDataFieldComparator>;
  
  towerFields_  = new std::set<DCCTBDataField  * , DCCTBDataFieldComparator>;   
  xtalFields_   = new std::set<DCCTBDataField  * , DCCTBDataFieldComparator>;
  trailerFields_= new std::set<DCCTBDataField  * , DCCTBDataFieldComparator>;
	
  buildDCCFields();
  buildTCCFields(); 
  buildSRPFields(); 
  buildTowerFields();
  buildXtalFields();
  buildTrailerFields();	
}

/*---------------------------------------------*/
/* DCCTBDataMapper::~DCCTBDataMapper               */
/* class destructor (free memory)              */
/*---------------------------------------------*/
DCCTBDataMapper::~DCCTBDataMapper(){
  
  std::set<DCCTBDataField *,DCCTBDataFieldComparator>::iterator it;
  
  for(it = dccFields_->begin()    ;it != dccFields_->end();     it++){ delete (*it);}
  for(it = emptyEventFields_->begin()    ;it != emptyEventFields_->end();     it++){ delete (*it);}
  
  for(it = tcc68Fields_->begin()  ;it != tcc68Fields_->end();     it++){ delete (*it);}		
  for(it = tcc32Fields_->begin()  ;it != tcc32Fields_->end();     it++){ delete (*it);}
  for(it = tcc16Fields_->begin()  ;it != tcc16Fields_->end();     it++){ delete (*it);}
  
  for(it = srp68Fields_->begin()  ;it != srp68Fields_->end();     it++){ delete (*it);}
  for(it = srp32Fields_->begin()  ;it != srp32Fields_->end();     it++){ delete (*it);}
  for(it = srp16Fields_->begin()  ;it != srp16Fields_->end();     it++){ delete (*it);}
	
  for(it = towerFields_->begin()  ;it != towerFields_->end();   it++){ delete (*it);}
  for(it = xtalFields_->begin()   ;it != xtalFields_->end();    it++){ delete (*it);}
  for(it = trailerFields_->begin();it != trailerFields_->end(); it++){ delete (*it);}
	
  delete dccFields_;
  delete emptyEventFields_;

  delete tcc68Fields_;
  delete tcc32Fields_;
  delete tcc16Fields_;
  
  delete srp68Fields_;
  delete srp32Fields_;
  delete srp16Fields_;
  
  delete towerFields_;
  delete xtalFields_;
  delete trailerFields_;
}


/*-------------------------------------------------*/
/* DCCTBDataMapper::buildDccFields                   */
/* builds raw data header fields                   */
/*-------------------------------------------------*/
void DCCTBDataMapper::buildDCCFields(){

  //32 Bit word numb 0
  dccFields_->insert( new DCCTBDataField("H",H_WPOSITION,H_BPOSITION,H_MASK));
  emptyEventFields_->insert( new DCCTBDataField("H",H_WPOSITION,H_BPOSITION,H_MASK));
  
  dccFields_->insert( new DCCTBDataField("FOV",FOV_WPOSITION,FOV_BPOSITION,FOV_MASK));
  emptyEventFields_->insert( new DCCTBDataField("FOV",FOV_WPOSITION,FOV_BPOSITION,FOV_MASK));
  
  dccFields_->insert( new DCCTBDataField("FED/DCC ID",DCCID_WPOSITION,DCCID_BPOSITION,DCCID_MASK));
  emptyEventFields_->insert( new DCCTBDataField("FED/DCC ID",DCCID_WPOSITION,DCCID_BPOSITION,DCCID_MASK));
	
  dccFields_->insert( new DCCTBDataField("BX",DCCBX_WPOSITION,DCCBX_BPOSITION,DCCBX_MASK));
  emptyEventFields_->insert( new DCCTBDataField("BX",DCCBX_WPOSITION,DCCBX_BPOSITION,DCCBX_MASK));

  //32Bit word numb 1
  dccFields_->insert( new DCCTBDataField("LV1",DCCL1_WPOSITION ,DCCL1_BPOSITION,DCCL1_MASK));
  emptyEventFields_->insert( new DCCTBDataField("LV1",DCCL1_WPOSITION ,DCCL1_BPOSITION,DCCL1_MASK));
	
  dccFields_->insert( new DCCTBDataField("TRIGGER TYPE",TRIGGERTYPE_WPOSITION,TRIGGERTYPE_BPOSITION,TRIGGERTYPE_MASK));
  emptyEventFields_->insert( new DCCTBDataField("TRIGGER TYPE",TRIGGERTYPE_WPOSITION,TRIGGERTYPE_BPOSITION,TRIGGERTYPE_MASK));
	
  dccFields_->insert( new DCCTBDataField("BOE",BOE_WPOSITION,BOE_BPOSITION,BOE_MASK));
  emptyEventFields_->insert( new DCCTBDataField("BOE",BOE_WPOSITION,BOE_BPOSITION,BOE_MASK));

  //32Bit word numb 2
  dccFields_->insert( new DCCTBDataField("EVENT LENGTH",EVENTLENGTH_WPOSITION,EVENTLENGTH_BPOSITION,EVENTLENGTH_MASK));
  emptyEventFields_->insert( new DCCTBDataField("EVENT LENGTH",EVENTLENGTH_WPOSITION,EVENTLENGTH_BPOSITION,EVENTLENGTH_MASK));
  
  dccFields_->insert( new DCCTBDataField("DCC ERRORS",DCCERRORS_WPOSITION  ,DCCERRORS_BPOSITION,DCCERRORS_MASK));
  emptyEventFields_->insert( new DCCTBDataField("DCC ERRORS",DCCERRORS_WPOSITION  ,DCCERRORS_BPOSITION,DCCERRORS_MASK));
  
  //32Bit word numb 3 
  dccFields_->insert( new DCCTBDataField("RUN NUMBER",RNUMB_WPOSITION,RNUMB_BPOSITION,RNUMB_MASK));
  emptyEventFields_->insert( new DCCTBDataField("RUN NUMBER",RNUMB_WPOSITION,RNUMB_BPOSITION,RNUMB_MASK));
	
  //32 Bit word numb 4
  dccFields_->insert( new DCCTBDataField("RUN TYPE",RUNTYPE_WPOSITION,RUNTYPE_BPOSITION,RUNTYPE_MASK));	
  emptyEventFields_->insert( new DCCTBDataField("RUN TYPE",RUNTYPE_WPOSITION,RUNTYPE_BPOSITION,RUNTYPE_MASK));	
  
  //32Bit word numb 5 
  dccFields_->insert( new DCCTBDataField("DETAILED TRIGGER TYPE",DETAILEDTT_WPOSITION,DETAILEDTT_BPOSITION,DETAILEDTT_MASK));
  emptyEventFields_->insert( new DCCTBDataField("DETAILED TRIGGER TYPE",DETAILEDTT_WPOSITION,DETAILEDTT_BPOSITION,DETAILEDTT_MASK));

  //32 Bit word numb 6
  dccFields_->insert( new DCCTBDataField("ORBIT COUNTER",ORBITCOUNTER_WPOSITION,ORBITCOUNTER_BPOSITION,ORBITCOUNTER_MASK));

  //32 Bit word numb 7
  dccFields_->insert( new DCCTBDataField("SR",SR_WPOSITION,SR_BPOSITION,SR_MASK));
  dccFields_->insert( new DCCTBDataField("ZS",ZS_WPOSITION,ZS_BPOSITION,ZS_MASK));
  dccFields_->insert( new DCCTBDataField("TZS",TZS_WPOSITION,TZS_BPOSITION,TZS_MASK));
	
  dccFields_->insert( new DCCTBDataField("SR_CHSTATUS",SR_CHSTATUS_WPOSITION,SR_CHSTATUS_BPOSITION,SR_CHSTATUS_MASK));	
  dccFields_->insert( new DCCTBDataField("TCC_CHSTATUS#1",TCC_CHSTATUS_WPOSITION,TCC_CHSTATUS_BPOSITION,TCC_CHSTATUS_MASK));	
  dccFields_->insert( new DCCTBDataField("TCC_CHSTATUS#2",TCC_CHSTATUS_WPOSITION,TCC_CHSTATUS_BPOSITION+4,TCC_CHSTATUS_MASK));
  dccFields_->insert( new DCCTBDataField("TCC_CHSTATUS#3",TCC_CHSTATUS_WPOSITION,TCC_CHSTATUS_BPOSITION+8,TCC_CHSTATUS_MASK));	
  dccFields_->insert( new DCCTBDataField("TCC_CHSTATUS#4",TCC_CHSTATUS_WPOSITION,TCC_CHSTATUS_BPOSITION+12,TCC_CHSTATUS_MASK));
  

  //add Headers Qualifiers: 8 words with 6 bits each written on the 2nd 32bit words
  for(uint32_t i=1;i<=8;i++){
    std::string header = std::string("H") + parser_->getDecString(i);
    dccFields_->insert( new DCCTBDataField(header,HD_WPOSITION + (i-1)*2 ,HD_BPOSITION,HD_MASK));		

    //fill only for empty events
    if(i<3){ emptyEventFields_->insert( new DCCTBDataField(header,HD_WPOSITION + (i-1)*2 ,HD_BPOSITION,HD_MASK));	}	
  }


  //add FE_CHSTATUS: 5 words each having 14 FE_CHSTATUS
  for(uint32_t wcount = 1; wcount<=5; wcount++){

    //1st word 32 bit
    for(uint32_t i=1;i<=8;i++){
      std::string chStatus = std::string("FE_CHSTATUS#") + parser_->getDecString( (wcount-1)*14 + i );
      dccFields_->insert( new DCCTBDataField(chStatus, FE_CHSTATUS_WPOSITION +(wcount-1)*2, 4*(i-1),FE_CHSTATUS_MASK));	
    }

    //2nd word 32 bit
    for(uint32_t i=9;i<=14;i++){
      std::string chStatus = std::string("FE_CHSTATUS#") + parser_->getDecString((wcount-1)*14 + i);
      dccFields_->insert( new DCCTBDataField(chStatus, FE_CHSTATUS_WPOSITION + (wcount-1)*2 + 1,4*(i-9),FE_CHSTATUS_MASK));	
    }
    
  }

}

/*-------------------------------------------------*/
/* DCCTBDataMapper::buildTCCFields                   */
/* builds raw data TCC block fields                */
/*-------------------------------------------------*/
void DCCTBDataMapper::buildTCCFields(){
	
  std::vector<std::set<DCCTBDataField *, DCCTBDataFieldComparator> *> pVector;
  pVector.push_back(tcc16Fields_);
  pVector.push_back(tcc32Fields_);
  pVector.push_back(tcc68Fields_);
	
  for(int i=0; i< ((int)(pVector.size())) ;i++){
    (pVector[i])->insert( new DCCTBDataField("TCC ID",TCCID_WPOSITION ,TCCID_BPOSITION,TCCID_MASK));
    (pVector[i])->insert( new DCCTBDataField("BX",TCCBX_WPOSITION ,TCCBX_BPOSITION,TCCBX_MASK));	
    (pVector[i])->insert( new DCCTBDataField("E0",TCCE0_WPOSITION ,TCCE0_BPOSITION,TCCE0_MASK));
    (pVector[i])->insert( new DCCTBDataField("LV1",TCCL1_WPOSITION ,TCCL1_BPOSITION,TCCL1_MASK));
    (pVector[i])->insert( new DCCTBDataField("E1", TCCE1_WPOSITION, TCCE1_BPOSITION, TCCE1_MASK));	
    (pVector[i])->insert( new DCCTBDataField("#TT", NTT_WPOSITION, NTT_BPOSITION, NTT_MASK));
    (pVector[i])->insert( new DCCTBDataField("#TIME SAMPLES",TCCTSAMP_WPOSITION, TCCTSAMP_BPOSITION,TCCTSAMP_MASK));	
    (pVector[i])->insert( new DCCTBDataField("LE0",TCCLE0_WPOSITION, TCCLE0_BPOSITION, TCCLE0_MASK));	
    (pVector[i])->insert( new DCCTBDataField("LE1",TCCLE1_WPOSITION, TCCLE1_BPOSITION, TCCLE1_MASK));	
  }
  
  uint32_t nTSamples = parser_->numbTriggerSamples();
	
  uint32_t totalTT   = 68*nTSamples; 
  
  uint32_t filter1 = 16*nTSamples;
  uint32_t filter2 = 32*nTSamples;
	
  uint32_t count(2) ;
	
  // Fill block with TT definition 
  for(uint32_t tt=1; tt<=totalTT; tt++){
    std::string tpg    = std::string("TPG#") + parser_->getDecString(tt);
    std::string ttFlag = std::string("TTF#") + parser_->getDecString(tt);

    if(tt<=filter1){ 
      tcc16Fields_->insert( new DCCTBDataField(tpg, TPG_WPOSITION -1 + count/2, TPG_BPOSITION + 16*( (count+2)%2 ),TPG_MASK));
      tcc16Fields_->insert( new DCCTBDataField(ttFlag, TTF_WPOSITION -1 + count/2, TTF_BPOSITION + 16*( (count+2)%2 ),TTF_MASK));
    }
    if(tt<=filter2){
      tcc32Fields_->insert( new DCCTBDataField(tpg, TPG_WPOSITION -1 + count/2, TPG_BPOSITION + 16*( (count+2)%2 ),TPG_MASK));
      tcc32Fields_->insert( new DCCTBDataField(ttFlag, TTF_WPOSITION -1 + count/2, TTF_BPOSITION + 16*( (count+2)%2 ),TTF_MASK));
    }
    
    tcc68Fields_->insert( new DCCTBDataField(tpg, TPG_WPOSITION -1 + count/2, TPG_BPOSITION + 16*( (count+2)%2 ),TPG_MASK));
    tcc68Fields_->insert( new DCCTBDataField(ttFlag, TTF_WPOSITION -1 + count/2, TTF_BPOSITION + 16*( (count+2)%2 ),TTF_MASK));
    count++;
  }
		
}

// ---> update with the correct number of SRP fields
void DCCTBDataMapper::buildSRPFields(){
  std::vector<std::set<DCCTBDataField *, DCCTBDataFieldComparator> * > pVector;
  pVector.push_back(srp68Fields_);
  pVector.push_back(srp32Fields_);
  pVector.push_back(srp16Fields_);
  
  for(int i=0; i< ((int)(pVector.size())) ;i++){
    // This method must be modified to take into account the different SRP blocks : 68 SRF in the barrel, 34 ,35 or 36 in the EE
    (pVector[i])->insert( new DCCTBDataField("SRP ID",SRPID_WPOSITION ,SRPID_BPOSITION,SRPID_MASK));
    (pVector[i])->insert( new DCCTBDataField("BX",SRPBX_WPOSITION ,SRPBX_BPOSITION,SRPBX_MASK));	
    (pVector[i])->insert( new DCCTBDataField("E0",SRPE0_WPOSITION ,SRPE0_BPOSITION,SRPE0_MASK));
    
    (pVector[i])->insert( new DCCTBDataField("LV1",SRPL1_WPOSITION ,SRPL1_BPOSITION,SRPL1_MASK));
    (pVector[i])->insert( new DCCTBDataField("E1", SRPE1_WPOSITION, SRPE1_BPOSITION, SRPE1_MASK));	
    (pVector[i])->insert( new DCCTBDataField("#SR FLAGS",NSRF_WPOSITION, NSRF_BPOSITION,NSRF_MASK));
    (pVector[i])->insert( new DCCTBDataField("LE0",SRPLE0_WPOSITION, SRPLE0_BPOSITION, SRPLE0_MASK));	
    (pVector[i])->insert( new DCCTBDataField("LE1",SRPLE1_WPOSITION, SRPLE1_BPOSITION, SRPLE1_MASK));	
  }
  
  uint32_t srpFlags(68); 
  
  uint32_t count1(1), count2(1), srSize(3), factor(0), wcount(0);
  for(uint32_t nsr =1; nsr<=srpFlags; nsr++){
    
    std::string sr = std::string("SR#") + parser_->getDecString(nsr);
    
    srp68Fields_->insert( new DCCTBDataField(sr,SRF_WPOSITION + wcount, SRF_BPOSITION + SRPBOFFSET*factor + (count2-1)*srSize,SRF_MASK));
    if( nsr<=32 ){ srp32Fields_->insert( new DCCTBDataField(sr,SRF_WPOSITION + wcount, SRF_BPOSITION + SRPBOFFSET*factor + (count2-1)*srSize,SRF_MASK));}
    if( nsr<=16 ){ srp16Fields_->insert( new DCCTBDataField(sr,SRF_WPOSITION + wcount, SRF_BPOSITION + SRPBOFFSET*factor + (count2-1)*srSize,SRF_MASK));}
    
    count1++; count2++; 
    
    //update word count
    if( count1 > 8){ wcount++; count1=1;}	
    
    //update bit offset
    if(count1 > 4){ factor = 1;}
    else{factor = 0;}
    
    //update bit shift
    if( count2 > 4){ count2 = 1;}
    
  }
}


/*-------------------------------------------------*/
/* DCCTBDataMapper::buildTowerFields                 */
/* builds raw data Tower Data fields               */
/*-------------------------------------------------*/
void DCCTBDataMapper::buildTowerFields(){
  //32bit word numb 1
  towerFields_->insert( new DCCTBDataField("TT/SC ID",TOWERID_WPOSITION ,TOWERID_BPOSITION,TOWERID_MASK));
  towerFields_->insert( new DCCTBDataField("#TIME SAMPLES",XSAMP_WPOSITION ,XSAMP_BPOSITION,XSAMP_MASK));
  towerFields_->insert( new DCCTBDataField("BX", TOWERBX_WPOSITION ,TOWERBX_BPOSITION,TOWERBX_MASK));	
  towerFields_->insert( new DCCTBDataField("E0",TOWERE0_WPOSITION ,TOWERE0_BPOSITION,TOWERE0_MASK));
  
  //32 bit word numb 2
  towerFields_->insert( new DCCTBDataField("LV1",TOWERL1_WPOSITION ,TOWERL1_BPOSITION, TOWERL1_MASK));
  towerFields_->insert( new DCCTBDataField("E1", TOWERE1_WPOSITION, TOWERE1_BPOSITION, TOWERE1_MASK));	
  towerFields_->insert( new DCCTBDataField("BLOCK LENGTH",TOWERLENGTH_WPOSITION, TOWERLENGTH_BPOSITION,TOWERLENGTH_MASK));
}


/*-------------------------------------------------*/
/* DCCTBDataMapper::buildXtalFields                  */
/* builds raw data Crystal Data fields             */
/*-------------------------------------------------*/
void DCCTBDataMapper::buildXtalFields(){
	
  //32bit word numb 1	
  xtalFields_->insert(new DCCTBDataField("STRIP ID",STRIPID_WPOSITION,STRIPID_BPOSITION,STRIPID_MASK));
  xtalFields_->insert(new DCCTBDataField("XTAL ID",XTALID_WPOSITION,XTALID_BPOSITION,XTALID_MASK));
  xtalFields_->insert(new DCCTBDataField("M",M_WPOSITION,M_BPOSITION,M_MASK));
  xtalFields_->insert(new DCCTBDataField("SMF",SMF_WPOSITION,SMF_BPOSITION,SMF_MASK));
  xtalFields_->insert(new DCCTBDataField("GMF",GMF_WPOSITION,GMF_BPOSITION,GMF_MASK));

  //first ADC is still on 1st word
  xtalFields_->insert(new DCCTBDataField("ADC#1",ADC_WPOSITION,ADCBOFFSET,ADC_MASK));
	
  //add the rest of the ADCs 
  for(uint32_t i=2; i <= parser_->numbXtalSamples();i++){
    std::string adc = std::string("ADC#") + parser_->getDecString(i);
    if(i%2){ xtalFields_->insert(new DCCTBDataField(adc,ADC_WPOSITION + i/2, ADCBOFFSET,ADC_MASK)); }
    else   { xtalFields_->insert(new DCCTBDataField(adc,ADC_WPOSITION + i/2, 0,ADC_MASK)); }
  }

  //the last word has written the test zero suppression flag and the gain decision bit
  uint32_t tzsOffset_ = parser_->numbXtalSamples()/2;
  xtalFields_->insert(new DCCTBDataField("TZS",XTAL_TZS_WPOSITION+tzsOffset_,XTAL_TZS_BPOSITION,XTAL_TZS_MASK));
  xtalFields_->insert(new DCCTBDataField("GDECISION",XTAL_GDECISION_WPOSITION+tzsOffset_,XTAL_GDECISION_BPOSITION,XTAL_GDECISION_MASK));
}


/*-------------------------------------------------*/
/* DCCTBDataMapper::buildTrailerFields               */
/* builds raw data Trailer words                   */
/*-------------------------------------------------*/
void DCCTBDataMapper::buildTrailerFields(){
  //32bit word numb 1
  trailerFields_->insert(new DCCTBDataField("T",T_WPOSITION,T_BPOSITION,T_MASK));
  trailerFields_->insert(new DCCTBDataField("TTS",TTS_WPOSITION,TTS_BPOSITION,TTS_MASK));
  trailerFields_->insert(new DCCTBDataField("EVENT STATUS",ESTAT_WPOSITION,ESTAT_BPOSITION,ESTAT_MASK));
  trailerFields_->insert(new DCCTBDataField("CRC",CRC_WPOSITION,CRC_BPOSITION,CRC_MASK));

  //32bit word numb 2
  trailerFields_->insert(new DCCTBDataField("EVENT LENGTH",TLENGTH_WPOSITION,TLENGTH_BPOSITION,TLENGTH_MASK));
  trailerFields_->insert(new DCCTBDataField("EOE",EOE_WPOSITION,EOE_BPOSITION,EOE_MASK));
	
}
