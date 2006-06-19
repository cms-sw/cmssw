#include "EventFilter/EcalRawToDigi/src/DCCDataMapper.h"

DCCDataMapper::DCCDataMapper( DCCDataParser * myParser)
: parser_(myParser){
	
	
	dccFields_        = new set<DCCDataField  * , DCCDataFieldComparator>;
	emptyEventFields_ = new set<DCCDataField  * , DCCDataFieldComparator>;
	
	tcc68Fields_    = new set<DCCDataField  * , DCCDataFieldComparator>; 
	tcc32Fields_    = new set<DCCDataField  * , DCCDataFieldComparator>;
	tcc16Fields_    = new set<DCCDataField  * , DCCDataFieldComparator>;

	srp68Fields_    = new set<DCCDataField  * , DCCDataFieldComparator>;
	srp32Fields_    = new set<DCCDataField  * , DCCDataFieldComparator>;
	srp16Fields_    = new set<DCCDataField  * , DCCDataFieldComparator>;
	
	towerFields_  = new set<DCCDataField  * , DCCDataFieldComparator>;   
	xtalFields_   = new set<DCCDataField  * , DCCDataFieldComparator>;
	trailerFields_= new set<DCCDataField  * , DCCDataFieldComparator>;

	
	buildDCCFields();
	buildTCCFields(); 
	buildSRPFields(); 
	buildTowerFields();
	buildXtalFields();
	buildTrailerFields();
		
}


DCCDataMapper::~DCCDataMapper(){

	
	set<DCCDataField *,DCCDataFieldComparator>::iterator it;
	
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


void DCCDataMapper::buildDCCFields(){

	
	//32 Bit word numb 0
	dccFields_->insert( new DCCDataField("H",H_WPOSITION,H_BPOSITION,H_MASK));
	emptyEventFields_->insert( new DCCDataField("H",H_WPOSITION,H_BPOSITION,H_MASK));
	
	dccFields_->insert( new DCCDataField("FOV",FOV_WPOSITION,FOV_BPOSITION,FOV_MASK));
	emptyEventFields_->insert( new DCCDataField("FOV",FOV_WPOSITION,FOV_BPOSITION,FOV_MASK));
	
	dccFields_->insert( new DCCDataField("FED/DCC ID",DCCID_WPOSITION,DCCID_BPOSITION,DCCID_MASK));
	emptyEventFields_->insert( new DCCDataField("FED/DCC ID",DCCID_WPOSITION,DCCID_BPOSITION,DCCID_MASK));
	
	dccFields_->insert( new DCCDataField("BX",DCCBX_WPOSITION,DCCBX_BPOSITION,DCCBX_MASK));
	emptyEventFields_->insert( new DCCDataField("BX",DCCBX_WPOSITION,DCCBX_BPOSITION,DCCBX_MASK));

	//32Bit word numb 1
	dccFields_->insert( new DCCDataField("LV1",DCCL1_WPOSITION ,DCCL1_BPOSITION,DCCL1_MASK));
	emptyEventFields_->insert( new DCCDataField("LV1",DCCL1_WPOSITION ,DCCL1_BPOSITION,DCCL1_MASK));
	
	dccFields_->insert( new DCCDataField("TRIGGER TYPE",TRIGGERTYPE_WPOSITION,TRIGGERTYPE_BPOSITION,TRIGGERTYPE_MASK));
	emptyEventFields_->insert( new DCCDataField("TRIGGER TYPE",TRIGGERTYPE_WPOSITION,TRIGGERTYPE_BPOSITION,TRIGGERTYPE_MASK));
	
	dccFields_->insert( new DCCDataField("BOE",BOE_WPOSITION,BOE_BPOSITION,BOE_MASK));
	emptyEventFields_->insert( new DCCDataField("BOE",BOE_WPOSITION,BOE_BPOSITION,BOE_MASK));

	//32Bit word numb 2
	dccFields_->insert( new DCCDataField("EVENT LENGTH",EVENTLENGTH_WPOSITION,EVENTLENGTH_BPOSITION,EVENTLENGTH_MASK));
	emptyEventFields_->insert( new DCCDataField("EVENT LENGTH",EVENTLENGTH_WPOSITION,EVENTLENGTH_BPOSITION,EVENTLENGTH_MASK));

	dccFields_->insert( new DCCDataField("DCC ERRORS",DCCERRORS_WPOSITION  ,DCCERRORS_BPOSITION,DCCERRORS_MASK));
	emptyEventFields_->insert( new DCCDataField("DCC ERRORS",DCCERRORS_WPOSITION  ,DCCERRORS_BPOSITION,DCCERRORS_MASK));
	
	//32Bit word numb 3 
	dccFields_->insert( new DCCDataField("RUN NUMBER",RNUMB_WPOSITION,RNUMB_BPOSITION,RNUMB_MASK));
	emptyEventFields_->insert( new DCCDataField("RUN NUMBER",RNUMB_WPOSITION,RNUMB_BPOSITION,RNUMB_MASK));
	
	//32 Bit word numb 4
	dccFields_->insert( new DCCDataField("RUN TYPE",RUNTYPE_WPOSITION,RUNTYPE_BPOSITION,RUNTYPE_MASK));	
	emptyEventFields_->insert( new DCCDataField("RUN TYPE",RUNTYPE_WPOSITION,RUNTYPE_BPOSITION,RUNTYPE_MASK));	
	
	//32Bit word numb 5 
	dccFields_->insert( new DCCDataField("DETAILED TRIGGER TYPE",DETAILEDTT_WPOSITION,DETAILEDTT_BPOSITION,DETAILEDTT_MASK));
	emptyEventFields_->insert( new DCCDataField("DETAILED TRIGGER TYPE",DETAILEDTT_WPOSITION,DETAILEDTT_BPOSITION,DETAILEDTT_MASK));


	
	//32 Bit word numb 6
	dccFields_->insert( new DCCDataField("ORBIT COUNTER",ORBITCOUNTER_WPOSITION,ORBITCOUNTER_BPOSITION,ORBITCOUNTER_MASK));

	//32 Bit word numb 7
	dccFields_->insert( new DCCDataField("SR",SR_WPOSITION,SR_BPOSITION,SR_MASK));
	dccFields_->insert( new DCCDataField("ZS",ZS_WPOSITION,ZS_BPOSITION,ZS_MASK));
	dccFields_->insert( new DCCDataField("TZS",TZS_WPOSITION,TZS_BPOSITION,TZS_MASK));
	
	dccFields_->insert( new DCCDataField("SR_CHSTATUS",SR_CHSTATUS_WPOSITION,SR_CHSTATUS_BPOSITION,SR_CHSTATUS_MASK));	
	dccFields_->insert( new DCCDataField("TCC_CHSTATUS#1",TCC_CHSTATUS_WPOSITION,TCC_CHSTATUS_BPOSITION,TCC_CHSTATUS_MASK));	
	dccFields_->insert( new DCCDataField("TCC_CHSTATUS#2",TCC_CHSTATUS_WPOSITION,TCC_CHSTATUS_BPOSITION+4,TCC_CHSTATUS_MASK));
	dccFields_->insert( new DCCDataField("TCC_CHSTATUS#3",TCC_CHSTATUS_WPOSITION,TCC_CHSTATUS_BPOSITION+8,TCC_CHSTATUS_MASK));	
	dccFields_->insert( new DCCDataField("TCC_CHSTATUS#4",TCC_CHSTATUS_WPOSITION,TCC_CHSTATUS_BPOSITION+12,TCC_CHSTATUS_MASK));
	

	//////////////////////////////////////////////////////////////////////////////////////////////////
	

	//Add Headers Qualifiers //////////////////////////////////////////////////////////////////
	for(ulong i=1;i<=8;i++){
		string header = string("H") + parser_->getDecString(i);
		dccFields_->insert( new DCCDataField(header,HD_WPOSITION + (i-1)*2 ,HD_BPOSITION,HD_MASK));		
		if(i<3){ emptyEventFields_->insert( new DCCDataField(header,HD_WPOSITION + (i-1)*2 ,HD_BPOSITION,HD_MASK));	}	
	}
	///////////////////////////////////////////////////////////////////////////////////////////
	

	//Add FE_CHSTATUS ///////////////////////////////////////////////////////////////////////////////////////////////////
	for(ulong wcount = 1; wcount<=5; wcount++){
		
		for(ulong i=1;i<=8;i++){
			string chStatus = string("FE_CHSTATUS#") + parser_->getDecString( (wcount-1)*14 + i );
			dccFields_->insert( new DCCDataField(chStatus, FE_CHSTATUS_WPOSITION +(wcount-1)*2, 4*(i-1),FE_CHSTATUS_MASK));	
		}
		for(ulong i=9;i<=14;i++){
			string chStatus = string("FE_CHSTATUS#") + parser_->getDecString((wcount-1)*14 + i);
			dccFields_->insert( new DCCDataField(chStatus, FE_CHSTATUS_WPOSITION + (wcount-1)*2 + 1,4*(i-9),FE_CHSTATUS_MASK));	
		}
	
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////	

}


void DCCDataMapper::buildTCCFields(){
	
	vector<	set<DCCDataField *, DCCDataFieldComparator> * > pVector;
	pVector.push_back(tcc16Fields_);
	pVector.push_back(tcc32Fields_);
	pVector.push_back(tcc68Fields_);
	
	for(unsigned int i=0; i<pVector.size();i++){
		(pVector[i])->insert( new DCCDataField("TCC ID",TCCID_WPOSITION ,TCCID_BPOSITION,TCCID_MASK));
		(pVector[i])->insert( new DCCDataField("BX",TCCBX_WPOSITION ,TCCBX_BPOSITION,TCCBX_MASK));	
		(pVector[i])->insert( new DCCDataField("E0",TCCE0_WPOSITION ,TCCE0_BPOSITION,TCCE0_MASK));
		(pVector[i])->insert( new DCCDataField("LV1",TCCL1_WPOSITION ,TCCL1_BPOSITION,TCCL1_MASK));
		(pVector[i])->insert( new DCCDataField("E1", TCCE1_WPOSITION, TCCE1_BPOSITION, TCCE1_MASK));	
		(pVector[i])->insert( new DCCDataField("#TT", NTT_WPOSITION, NTT_BPOSITION, NTT_MASK));
		(pVector[i])->insert( new DCCDataField("#TIME SAMPLES",TCCTSAMP_WPOSITION, TCCTSAMP_BPOSITION,TCCTSAMP_MASK));	
		(pVector[i])->insert( new DCCDataField("LE0",TCCLE0_WPOSITION, TCCLE0_BPOSITION, TCCLE0_MASK));	
		(pVector[i])->insert( new DCCDataField("LE1",TCCLE1_WPOSITION, TCCLE1_BPOSITION, TCCLE1_MASK));	
	}

	
	ulong nTSamples = parser_->numbTriggerSamples();
	
	ulong totalTT   = 68*nTSamples; 
	
	ulong filter1 = 16*nTSamples;
	ulong filter2 = 32*nTSamples;
	
	ulong count(2) ;
	
	// Fill block with TT definition //////////////////////////////////////////////////////////////////////////////////////////////////////////
	for(ulong tt=1; tt<=totalTT; tt++){
		string tpg    = string("TPG#") + parser_->getDecString(tt);
		string ttFlag = string("TTF#") + parser_->getDecString(tt);

		
		if(tt<=filter1){ 
			tcc16Fields_->insert( new DCCDataField(tpg, TPG_WPOSITION -1 + count/2, TPG_BPOSITION + 18*( (count+2)%2 ),TPG_MASK));
			tcc16Fields_->insert( new DCCDataField(ttFlag, TTF_WPOSITION -1 + count/2, TTF_BPOSITION + 18*( (count+2)%2 ),TTF_MASK));
		}
		if(tt<=filter2){
			tcc32Fields_->insert( new DCCDataField(tpg, TPG_WPOSITION -1 + count/2, TPG_BPOSITION + 18*( (count+2)%2 ),TPG_MASK));
			tcc32Fields_->insert( new DCCDataField(ttFlag, TTF_WPOSITION -1 + count/2, TTF_BPOSITION + 18*( (count+2)%2 ),TTF_MASK));
		}
		
		tcc68Fields_->insert( new DCCDataField(tpg, TPG_WPOSITION -1 + count/2, TPG_BPOSITION + 18*( (count+2)%2 ),TPG_MASK));
		tcc68Fields_->insert( new DCCDataField(ttFlag, TTF_WPOSITION -1 + count/2, TTF_BPOSITION + 18*( (count+2)%2 ),TTF_MASK));
		count++;
		
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	
}


void DCCDataMapper::buildSRPFields(){
	vector<	set<DCCDataField *, DCCDataFieldComparator> * > pVector;
	pVector.push_back(srp68Fields_);
	pVector.push_back(srp32Fields_);
	pVector.push_back(srp16Fields_);
	
	for(unsigned int i=0; i<pVector.size();i++){
		// This methid must be modified to take into account the different SRP blocks : 68 SRF in the barrel, 34 ,35 or 36 in the EE ///////////////////////
		(pVector[i])->insert( new DCCDataField("SRP ID",SRPID_WPOSITION ,SRPID_BPOSITION,SRPID_MASK));
		(pVector[i])->insert( new DCCDataField("BX",SRPBX_WPOSITION ,SRPBX_BPOSITION,SRPBX_MASK));	
		(pVector[i])->insert( new DCCDataField("E0",SRPE0_WPOSITION ,SRPE0_BPOSITION,SRPE0_MASK));
	
		(pVector[i])->insert( new DCCDataField("LV1",SRPL1_WPOSITION ,SRPL1_BPOSITION,SRPL1_MASK));
		(pVector[i])->insert( new DCCDataField("E1", SRPE1_WPOSITION, SRPE1_BPOSITION, SRPE1_MASK));	
		(pVector[i])->insert( new DCCDataField("#SR FLAGS",NSRF_WPOSITION, NSRF_BPOSITION,NSRF_MASK));
		(pVector[i])->insert( new DCCDataField("LE0",SRPLE0_WPOSITION, SRPLE0_BPOSITION, SRPLE0_MASK));	
		(pVector[i])->insert( new DCCDataField("LE1",SRPLE1_WPOSITION, SRPLE1_BPOSITION, SRPLE1_MASK));	
	}
	
	ulong srpFlags(68); 
	
	ulong count1(1), count2(1), srSize(3), factor(0), wcount(0);
	for(ulong nsr =1; nsr<=srpFlags; nsr++){
	
		string sr = string("SR#") + parser_->getDecString(nsr);
		
		srp68Fields_->insert( new DCCDataField(sr,SRF_WPOSITION + wcount, SRF_BPOSITION + SRPBOFFSET*factor + (count2-1)*srSize,SRF_MASK));
		if( nsr<=32 ){ srp32Fields_->insert( new DCCDataField(sr,SRF_WPOSITION + wcount, SRF_BPOSITION + SRPBOFFSET*factor + (count2-1)*srSize,SRF_MASK));}
		if( nsr<=16 ){ srp16Fields_->insert( new DCCDataField(sr,SRF_WPOSITION + wcount, SRF_BPOSITION + SRPBOFFSET*factor + (count2-1)*srSize,SRF_MASK));}
		
		count1++; count2++; 
		
		//update word count
		if( count1 > 8){ wcount++; count1=1;}	
		
		//update bit offset
		if(count1 > 4){ factor = 1;}
		else{factor = 0;}
		
		//update bit shift
		if( count2 > 4){ count2 = 1;}
		
	}
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
}


void DCCDataMapper::buildTowerFields(){
	
	towerFields_->insert( new DCCDataField("TT/SC ID",TOWERID_WPOSITION ,TOWERID_BPOSITION,TOWERID_MASK));
	towerFields_->insert( new DCCDataField("#TIME SAMPLES",XSAMP_WPOSITION ,XSAMP_BPOSITION,XSAMP_MASK));
		
	towerFields_->insert( new DCCDataField("BX", TOWERBX_WPOSITION ,TOWERBX_BPOSITION,TOWERBX_MASK));	
	towerFields_->insert( new DCCDataField("E0",TOWERE0_WPOSITION ,TOWERE0_BPOSITION,TOWERE0_MASK));

	towerFields_->insert( new DCCDataField("LV1",TOWERL1_WPOSITION ,TOWERL1_BPOSITION, TOWERL1_MASK));
	towerFields_->insert( new DCCDataField("E1", TOWERE1_WPOSITION, TOWERE1_BPOSITION, TOWERE1_MASK));	
	
	towerFields_->insert( new DCCDataField("BLOCK LENGTH",TOWERLENGTH_WPOSITION, TOWERLENGTH_BPOSITION,TOWERLENGTH_MASK));
}



void DCCDataMapper::buildXtalFields(){
	
	
	xtalFields_->insert(new DCCDataField("STRIP ID",STRIPID_WPOSITION,STRIPID_BPOSITION,STRIPID_MASK));

	xtalFields_->insert(new DCCDataField("XTAL ID",XTALID_WPOSITION,XTALID_BPOSITION,XTALID_MASK));
	xtalFields_->insert(new DCCDataField("M",M_WPOSITION,M_BPOSITION,M_MASK));
	
	xtalFields_->insert(new DCCDataField("SMF",SMF_WPOSITION,SMF_BPOSITION,SMF_MASK));
	xtalFields_->insert(new DCCDataField("GMF",GMF_WPOSITION,GMF_BPOSITION,GMF_MASK));
	xtalFields_->insert(new DCCDataField("TZS",XTAL_TZS_WPOSITION,XTAL_TZS_BPOSITION,XTAL_TZS_MASK));
	
	xtalFields_->insert(new DCCDataField("ADC#1",ADC_WPOSITION,ADCBOFFSET,ADC_MASK));
	
	// Add ADCs //////////////////////////////////////////////////////////////////////////////////////////////
	for(ulong i=2; i <= parser_->numbXtalSamples();i++){
		string adc = string("ADC#") + parser_->getDecString(i);
		if(i%2){ xtalFields_->insert(new DCCDataField(adc,ADC_WPOSITION + i/2, ADCBOFFSET,ADC_MASK)); }
		else   { xtalFields_->insert(new DCCDataField(adc,ADC_WPOSITION + i/2, 0,ADC_MASK)); }
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	
		
}


void DCCDataMapper::buildTrailerFields(){
	
	trailerFields_->insert(new DCCDataField("T",T_WPOSITION,T_BPOSITION,T_MASK));
	trailerFields_->insert(new DCCDataField("TTS",TTS_WPOSITION,TTS_BPOSITION,TTS_MASK));
	trailerFields_->insert(new DCCDataField("EVENT STATUS",ESTAT_WPOSITION,ESTAT_BPOSITION,ESTAT_MASK));
	trailerFields_->insert(new DCCDataField("CRC",CRC_WPOSITION,CRC_BPOSITION,CRC_MASK));
	trailerFields_->insert(new DCCDataField("EVENT LENGTH",TLENGTH_WPOSITION,TLENGTH_BPOSITION,TLENGTH_MASK));
	trailerFields_->insert(new DCCDataField("EOE",EOE_WPOSITION,EOE_BPOSITION,EOE_MASK));
	
}
