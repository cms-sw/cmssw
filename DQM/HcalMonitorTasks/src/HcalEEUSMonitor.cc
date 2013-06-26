#include "DQM/HcalMonitorTasks/interface/HcalEEUSMonitor.h"
// define sizes of ieta arrays for each subdetector

#define PI        3.1415926535897932

/*  

    v1.0
    11 February 2009
    by Jared Sturdy (from Jeff's HcalExpertMonitor)

*/


// constructor
HcalEEUSMonitor::HcalEEUSMonitor()
{
      for (int f=0; f<NUMFEDS;f++){
	for (int s=0; s<NUMSPIGS; s++) {
	  consecutiveEETriggers[f][s] = 0;
	  consecutiveNETriggers[f][s] = 0;
	  consecutiveTriggers[f][s]   = 0;
	  prevWasEE[f][s]             = 0;}}

//  std::cout << (int)sizeof(UScount) << std::endl;
  for (int f=0; f<NUMFEDS; f++) {
    for (int s=0; s<NUMSPIGS; s++) {
      UScount[f][s] = 0;
    }
  }

  //Francesco
//  std::cout << (int)sizeof(US0EE0count) << std::endl;
  for (int f=0; f<NUMFEDS; f++) {
    for (int s=0; s<NUMSPIGS; s++) {
      US0EE0count[f][s] = 0;
    }
  }

//  std::cout << (int)sizeof(US0EE1count) << std::endl;
  for (int f=0; f<NUMFEDS; f++) {
    for (int s=0; s<NUMSPIGS; s++) {
      US0EE1count[f][s] = 0;
    }
  }

//  std::cout << (int)sizeof(US1EE0count) << std::endl;
  for (int f=0; f<NUMFEDS; f++) {
    for (int s=0; s<NUMSPIGS; s++) {
      US1EE0count[f][s] = 0;
    }
  }

//  std::cout << (int)sizeof(US1EE1count) << std::endl;
  for (int f=0; f<NUMFEDS; f++) {
    for (int s=0; s<NUMSPIGS; s++) {
      US1EE1count[f][s] = 0;
    }
  }
  //----------

}

// destructor
HcalEEUSMonitor::~HcalEEUSMonitor() {}

void HcalEEUSMonitor::reset() {}

void HcalEEUSMonitor::clearME()
{
  if (m_dbe) 
    {
      m_dbe->setCurrentFolder(baseFolder_);
      m_dbe->removeContents();
    } // if (m_dbe)
  meEVT_=0;
} // void HcalEEUSMonitor::clearME()


void HcalEEUSMonitor::setup(const edm::ParameterSet& ps, DQMStore* dbe)
{
  HcalBaseMonitor::setup(ps,dbe);  // perform setups of base class

  ievt_=0; // event counter
  baseFolder_ = rootFolder_ + "EEUSMonitor"; // Will create an "EEUSMonitor" subfolder in .root output
  if (fVerbosity) std::cout <<"<HcalEEUSMonitor::setup> Setup in progress"<<std::endl;
  
  
  if(fVerbosity) std::cout << "About to pushback fedUnpackList_" << std::endl;
  firstFED_ = FEDNumbering::MINHCALFEDID;
  if (fVerbosity>0) std::cout <<"FIRST FED = "<<firstFED_<<std::endl;

  for (int i=FEDNumbering::MINHCALFEDID;
       i<=FEDNumbering::MAXHCALFEDID; ++i) 
    {
      if(fVerbosity) std::cout << "<HcalEEUSMonitor::setup>:Pushback for fedUnpackList_: " << i <<std::endl;
      fedUnpackList_.push_back(i);
    } // for (int i=FEDNumbering::MINHCALFEDID
  
  
  
  if (m_dbe)
    {
      std::string type;
      m_dbe->setCurrentFolder(baseFolder_);
      meEVT_ = m_dbe->bookInt("EEUSMonitor Event Number"); // store event number

      char label[10];
      //Francesco
  
      //fraction of X-Type events
  
      type = "Fraction Normal Events - US0 EE0";
      meNormFractSpigs_US0_EE0_ = m_dbe->book1D(type,type,481,0,481);
      for(int f=0; f<NUMFEDS; f++) {
	sprintf(label, "FED 7%02d", f);
	meNormFractSpigs_US0_EE0_->setBinLabel(1+(HcalDCCHeader::SPIGOT_COUNT*f), label);
	for(int s=1; s<HcalDCCHeader::SPIGOT_COUNT; s++) {
	  sprintf(label, "sp%02d", s-1);
	  meNormFractSpigs_US0_EE0_->setBinLabel(1+(HcalDCCHeader::SPIGOT_COUNT*f)+s, label);}}
  
      type = "Fraction Empty Events - US0 EE1";
      meEEFractSpigs_US0_EE1_ = m_dbe->book1D(type,type,481,0,481);
      for(int f=0; f<NUMFEDS; f++) {
	sprintf(label, "FED 7%02d", f);
	meEEFractSpigs_US0_EE1_->setBinLabel(1+(HcalDCCHeader::SPIGOT_COUNT*f), label);
	for(int s=1; s<HcalDCCHeader::SPIGOT_COUNT; s++) {
	  sprintf(label, "sp%02d", s-1);
	  meEEFractSpigs_US0_EE1_->setBinLabel(1+(HcalDCCHeader::SPIGOT_COUNT*f)+s, label);}}

      type = "Fraction UnSuppressed Events - US1 EE0";
      meUSFractSpigs_US1_EE0_ = m_dbe->book1D(type,type,481,0,481);
      for(int f=0; f<NUMFEDS; f++) {
	sprintf(label, "FED 7%02d", f);
	meUSFractSpigs_US1_EE0_->setBinLabel(1+(HcalDCCHeader::SPIGOT_COUNT*f), label);
	for(int s=1; s<HcalDCCHeader::SPIGOT_COUNT; s++) {
	  sprintf(label, "sp%02d", s-1);
	  meUSFractSpigs_US1_EE0_->setBinLabel(1+(HcalDCCHeader::SPIGOT_COUNT*f)+s, label);}}

      type = "Fraction UnSuppressed Events - US1 EE1";
      meUSFractSpigs_US1_EE1_ = m_dbe->book1D(type,type,481,0,481);
      for(int f=0; f<NUMFEDS; f++) {
	sprintf(label, "FED 7%02d", f);
	meUSFractSpigs_US1_EE1_->setBinLabel(1+(HcalDCCHeader::SPIGOT_COUNT*f), label);
	for(int s=1; s<HcalDCCHeader::SPIGOT_COUNT; s++) {
	  sprintf(label, "sp%02d", s-1);
	  meUSFractSpigs_US1_EE1_->setBinLabel(1+(HcalDCCHeader::SPIGOT_COUNT*f)+s, label);}}

      //raw data length for X-type events

      type = "Length of raw data - US0 EE0";
      meRawDataLength2_US0_EE0_ = m_dbe->book2D(type,type,481,0,481,600,0,1200);
      for(int f=0; f<NUMFEDS; f++) {
	sprintf(label, "FED 7%02d", f);
	meRawDataLength2_US0_EE0_->setBinLabel(1+(HcalDCCHeader::SPIGOT_COUNT*f), label);
	for(int s=1; s<HcalDCCHeader::SPIGOT_COUNT; s++) {
	  sprintf(label, "sp%02d", s-1);
	  meRawDataLength2_US0_EE0_->setBinLabel(1+(HcalDCCHeader::SPIGOT_COUNT*f)+s, label);}}

      type = "Length of raw data - US0 EE1";
      meRawDataLength2_US0_EE1_ = m_dbe->book2D(type,type,481,0,481,600,0,1200);
      for(int f=0; f<NUMFEDS; f++) {
	sprintf(label, "FED 7%02d", f);
	meRawDataLength2_US0_EE1_->setBinLabel(1+(HcalDCCHeader::SPIGOT_COUNT*f), label);
	for(int s=1; s<HcalDCCHeader::SPIGOT_COUNT; s++) {
	  sprintf(label, "sp%02d", s-1);
	  meRawDataLength2_US0_EE1_->setBinLabel(1+(HcalDCCHeader::SPIGOT_COUNT*f)+s, label);}}

      type = "Length of raw data - US1 EE0";
      meRawDataLength2_US1_EE0_ = m_dbe->book2D(type,type,481,0,481,600,0,1200);
      for(int f=0; f<NUMFEDS; f++) {
	sprintf(label, "FED 7%02d", f);
	meRawDataLength2_US1_EE0_->setBinLabel(1+(HcalDCCHeader::SPIGOT_COUNT*f), label);
	for(int s=1; s<HcalDCCHeader::SPIGOT_COUNT; s++) {
	  sprintf(label, "sp%02d", s-1);
	  meRawDataLength2_US1_EE0_->setBinLabel(1+(HcalDCCHeader::SPIGOT_COUNT*f)+s, label);}}

      type = "Length of raw data - US1 EE1";
      meRawDataLength2_US1_EE1_ = m_dbe->book2D(type,type,481,0,481,600,0,1200);
      for(int f=0; f<NUMFEDS; f++) {
	sprintf(label, "FED 7%02d", f);
	meRawDataLength2_US1_EE1_->setBinLabel(1+(HcalDCCHeader::SPIGOT_COUNT*f), label);
	for(int s=1; s<HcalDCCHeader::SPIGOT_COUNT; s++) {
	  sprintf(label, "sp%02d", s-1);
	  meRawDataLength2_US1_EE1_->setBinLabel(1+(HcalDCCHeader::SPIGOT_COUNT*f)+s, label);}}

      //---------

      //EECorrels Jason
  
      type="EE Spigot Correlation";
      meEECorrel_ = m_dbe->book2D(type, type,
				  (NUMSPIGS * NUMFEDS), 0, (NUMSPIGS * NUMFEDS),
				  (NUMSPIGS * NUMFEDS), 0, (NUMSPIGS * NUMFEDS));
      type="EE per Spigot";
      meEEPerSpigot_ = m_dbe->book1D(type, type,
				  (NUMSPIGS * NUMFEDS), 0, (NUMSPIGS * NUMFEDS));
      type="EE Spigots per Event";
      meEEThisEvent_ = m_dbe->book1D(type, type,500,-0.5,499.5);

      //EE/NE Triggers Jared
  
      char title[128];
      sprintf(title, "EE Triggers");  // DCC FED number 700:731
      meNumberEETriggered_ = m_dbe->book2D(title,title,(NUMSPIGS * NUMFEDS),0,(NUMSPIGS * NUMFEDS),100,-0.5,99.5);
      sprintf(title, "NE Triggers");  // DCC FED number 700:731
      meNumberNETriggered_ = m_dbe->book2D(title,title,(NUMSPIGS * NUMFEDS),0,(NUMSPIGS * NUMFEDS),100,-0.5,99.5);
      sprintf(title, "Triggers");  // DCC FED number 700:731
      meNumberTriggered_ = m_dbe->book2D(title,title,(NUMSPIGS * NUMFEDS),0,(NUMSPIGS * NUMFEDS),100,-0.5,99.5);
      
      for (int f=0; f<NUMFEDS;f++){
	sprintf(label, "DCC 7%02d", f);  // DCC FED number 700:731
	meEECorrel_->setBinLabel(1+(f*NUMSPIGS), label, 1);
	meEECorrel_->setBinLabel(1+(f*NUMSPIGS), label, 2);
	meEEPerSpigot_->setBinLabel(1+(f*NUMSPIGS), label, 1);
	meNumberEETriggered_->setBinLabel(1+(f*NUMSPIGS), label, 1);
	meNumberNETriggered_->setBinLabel(1+(f*NUMSPIGS), label, 1);
	meNumberTriggered_->setBinLabel(1+(f*NUMSPIGS), label, 1);
	for (int s=1; s<NUMSPIGS; s+=2) {
	  sprintf(label, "Spgt %02d", s-1);  // DCC Spigots
	  meEECorrel_->setBinLabel((f*NUMSPIGS)+s+1, label, 1);
	  meEECorrel_->setBinLabel((f*NUMSPIGS)+s+1, label, 2);
	  meEEPerSpigot_->setBinLabel((f*NUMSPIGS)+s+1, label, 1);
	  meNumberEETriggered_->setBinLabel((f*NUMSPIGS)+s+1, label, 1);
	  meNumberNETriggered_->setBinLabel((f*NUMSPIGS)+s+1, label, 1);
	  meNumberTriggered_->setBinLabel((f*NUMSPIGS)+s+1, label, 1);}}
      
      prevOrN = -1;
    } // if (m_dbe)
  
  return;
  
} // void HcalEEUSMonitor::setup()


void HcalEEUSMonitor::processEvent( const FEDRawDataCollection& rawraw,
				    const HcalUnpackerReport& report,
				    const HcalElectronicsMap& emap
				  )
  
{
  if (!m_dbe)
    {
      if (fVerbosity) std::cout <<"HcalEEUSMonitor::processEvent   DQMStore not instantiated!!!"<<std::endl;
      return;
    }

  // Fill Event Number
  ievt_++;
  meEVT_->Fill(ievt_);


  //EECorrels Assume the best, before unpack() 
  for (int i=0; i<(NUMSPIGS*NUMFEDS); i++) 
    EEthisEvent[i]=false;

  processEvent_RawData(rawraw, report, emap);
  prevOrN=dccOrN;
  return;
} // void HcalEEUSMonitor::processEvent


void HcalEEUSMonitor::processEvent_RawData(const FEDRawDataCollection& rawraw,
					     const HcalUnpackerReport& report,
					     const HcalElectronicsMap& emap)
{
  /*
    This processes Raw Data.
    Additional info on working with Raw Data can be found in 
    HcalDataFormatMonitor.cc
  */
  

  // Should not see this error
  if(!m_dbe) 
    {
      std::cout <<"HcalEEUSMonitor::processEvent_RawData:  DQMStore not instantiated!!!\n"<<std::endl;
      return;
    }
  numEEthisEvent = 0;
  // Loop over all FEDs reporting the event, unpacking if good.
  for (std::vector<int>::const_iterator i=fedUnpackList_.begin();i!=fedUnpackList_.end(); i++) 
    {
      const FEDRawData& fed = rawraw.FEDData(*i);
      if (fed.size()<12) continue; // Was 16.
      unpack(fed,emap);
    } // for (std::vector<int>::const_iterator i=fedUnpackList_.begin();...

  prevOrN=dccOrN;
  meEEThisEvent_->Fill(numEEthisEvent);
  numEEthisEvent = 0;

  //EECorrels :Fill a 2D histo where two each pair of 
  //spigots have EE in the same event.
  for (int outer=0; outer<(NUMSPIGS*NUMFEDS); outer++)
    if (EEthisEvent[outer]) 
      for (int inner=0; inner<(NUMSPIGS*NUMFEDS); inner++)
	if (EEthisEvent[inner]) 
	  meEECorrel_->Fill(outer, inner);
  
  return;

} // void HcalEEUSMonitor::processEvent_RawData(const FEDRawDataCollection& rawraw,



// Process one FED's worth (one DCC's worth) of the event data.
void HcalEEUSMonitor::unpack(const FEDRawData& raw, 
			       const HcalElectronicsMap& emap)
{
  /* 
     This unpacks raw data info.  Additional info on working with the raw data can be found in the unpack method of HcalDataFormatMonitor.cc
  */


  // get the DCC header
  const HcalDCCHeader* dccHeader=(const HcalDCCHeader*)(raw.data());
  if(!dccHeader) return;

  // get the DCC trailer 
  unsigned char* trailer_ptr = (unsigned char*) (raw.data()+raw.size()-sizeof(uint64_t));
  FEDTrailer trailer = FEDTrailer(trailer_ptr);

  //DCC Event Fragment sizes distribution, in bytes.
  //  int rawsize = raw.size();

  int dccid  = dccHeader->getSourceId();
  //  int dccBCN = dccHeader->getBunchId();
  dccOrN = dccHeader->getOrbitNumber();
  //  unsigned long dccEvtNum = dccHeader->getDCCEventNumber();

  //  uint64_t* lastDataWord = (uint64_t*) ( raw.data()+raw.size()-(2*sizeof(uint64_t)) );
  //  int EvFragLength = ((*lastDataWord>>32)*8);
  //  EvFragLength = raw.size();


  unsigned char WholeErrorList=0; 
  for(int j=0; j<HcalDCCHeader::SPIGOT_COUNT; j++) {
    WholeErrorList=dccHeader->getSpigotErrorBits((unsigned int) j);
    //EECorrels :Record EE for cross-correlation plotting.
    if ((WholeErrorList>>2)&0x01) EEthisEvent[(NUMSPIGS *std::max(0,dccid-700))+j] = true;
  }

  //

  // walk through the HTR data...
  HcalHTRData htr;  
  for (int spigot=0; spigot<HcalDCCHeader::SPIGOT_COUNT; spigot++) {
    if (!dccHeader->getSpigotPresent(spigot)) continue;
    
    //    bool chsummAOK=true;
//    bool channAOK=true;

    // From this Spigot's DCC header, first.
    WholeErrorList=dccHeader->getLRBErrorBits((unsigned int) spigot);
    // Load the given decoder with the pointer and length from this spigot.
    dccHeader->getSpigotData(spigot,htr, raw.size()); 
    const unsigned short* HTRraw = htr.getRawData();
    unsigned short HTRwdcount = HTRraw[htr.getRawLength() - 2];
    
    
    //fix me!
    HTRwdcount=htr.getRawLength();

    // Size checks for internal consistency
    // getNTP(), get NDD() seems to be mismatched with format. Manually:
    int NTP = ((htr.getExtHdr6() >> 8) & 0x00FF);
    int NDAQ = (HTRraw[htr.getRawLength() - 4] & 0x7FF);
    
    int here=1+(HcalDCCHeader::SPIGOT_COUNT*(dccid-700))+spigot;

    if ( !  ((HTRwdcount != 8)               ||
	     (HTRwdcount != 12 + NTP + NDAQ) ||
	     (HTRwdcount != 20 + NTP + NDAQ)    )) {
      //      chsummAOK=false;
      //incompatible Sizes declared. Skip it.
      continue; }
    bool EE = ((dccHeader->getSpigotErrorBits(spigot) >> 2) & 0x01);
    if (EE) { 
      numEEthisEvent++;
      meEEPerSpigot_->Fill(here);
      if (HTRwdcount != 8) {
	//	chsummAOK=false;
	//incompatible Sizes declared. Skip it.
	continue;}}
    else{ //For non-EE,
      if ((HTRwdcount-NDAQ-NTP) != 20) {
	//	chsummAOK=false;
	//incompatible Sizes declared. Skip it.
	continue;}}

    //Jared
    if(dccOrN==prevOrN){
      consecutiveTriggers[dccid-700][spigot]++;
      if(prevWasEE[dccid-700][spigot]) {
	if (EE) consecutiveEETriggers[dccid-700][spigot]++;
	else {
	  meNumberEETriggered_->Fill(here,consecutiveEETriggers[dccid-700][spigot]);
	  consecutiveEETriggers[dccid-700][spigot]   = 0;
	  consecutiveNETriggers[dccid-700][spigot]++;}}
      else {
	if(!EE) consecutiveNETriggers[dccid-700][spigot]++;
	else {
	  meNumberNETriggered_->Fill(here,consecutiveNETriggers[dccid-700][spigot]);
	  consecutiveNETriggers[dccid-700][spigot] = 0;
	  consecutiveEETriggers[dccid-700][spigot]++;}}}
    else {
      if (prevOrN>-1) {
	meNumberTriggered_->Fill(here,consecutiveTriggers[dccid-700][spigot]);
	meNumberEETriggered_->Fill(here,consecutiveEETriggers[dccid-700][spigot]);
	meNumberNETriggered_->Fill(here,consecutiveNETriggers[dccid-700][spigot]);}
      consecutiveTriggers[dccid-700][spigot]   = 1;
      if(EE) {
	consecutiveEETriggers[dccid-700][spigot] = 1;
	consecutiveNETriggers[dccid-700][spigot] = 0;}
      else {
	consecutiveEETriggers[dccid-700][spigot] = 0;
	consecutiveNETriggers[dccid-700][spigot] = 1;}}
    /*
      printf("%5d, %7.2d, %d, %8d, %22d, %22d, %20d, %3d, %10d\n",dccid,spigot,dccOrN,prevOrN,\
      consecutiveEETriggers[dccid-700][spigot],consecutiveNETriggers[dccid-700][spigot],\
      consecutiveTriggers[dccid-700][spigot],EE,prevWasEE[dccid-700][spigot]);
    */
    if (EE) prevWasEE[dccid-700][spigot] = 1;
    else    prevWasEE[dccid-700][spigot] = 0;

    if (htr.isHistogramEvent()) continue;

    bool htrUnSuppressed=(HTRraw[6]>>15 & 0x0001);
    //Francesco
    bool htrEmpty=(HTRraw[2] & 0x4);

//    if (htrUnSuppressed) {
//      UScount[dccid-700][spigot]++;
//      int here=1+(HcalDCCHeader::SPIGOT_COUNT*(dccid-700))+spigot;
//      meUSFractSpigs_->setBinContent(here,((double)UScount[dccid-700][spigot])/(double)ievt_);}
  
  //Francesco
  
    //std::cout << "HTRwdcount: " << HTRwdcount << std::endl;

    if (htrUnSuppressed==false && htrEmpty==false){
      US0EE0count[dccid-700][spigot]++;
      meNormFractSpigs_US0_EE0_->setBinContent(here,
					       ((double)US0EE0count[dccid-700][spigot])/(double)ievt_);
      meRawDataLength2_US0_EE0_->Fill(here-1, HTRwdcount);
    }
    
    if (htrUnSuppressed==false && htrEmpty==true){
      US0EE1count[dccid-700][spigot]++;
      meEEFractSpigs_US0_EE1_->setBinContent(here,
					     ((double)US0EE1count[dccid-700][spigot])/(double)ievt_);
      meRawDataLength2_US0_EE1_->Fill(here-1, HTRwdcount);
    }
	
    if (htrUnSuppressed==true && htrEmpty==false){
      US1EE0count[dccid-700][spigot]++;
      meUSFractSpigs_US1_EE0_->setBinContent(here,
					     ((double)US1EE0count[dccid-700][spigot])/(double)ievt_);
      meRawDataLength2_US1_EE0_->Fill(here-1, HTRwdcount);
    }
    
    if (htrUnSuppressed==true && htrEmpty==true){
      US1EE1count[dccid-700][spigot]++;
      meUSFractSpigs_US1_EE1_->setBinContent(here,
					     ((double)US1EE1count[dccid-700][spigot])/(double)ievt_);
      meRawDataLength2_US1_EE1_->Fill(here-1, HTRwdcount);
    }
    //--------
    

  }//end of HTRdata

    // Dump out some raw data info
//  std::cout <<"RAWSIZE = "<<rawsize<<std::endl;
//  std::cout <<"dcc id = "<<dccid<<std::endl;
//  std::cout <<"dccBCN = "<<dccBCN<<std::endl;
//  std::cout <<"dccEvtNum = "<<dccEvtNum<<std::endl;
//  std::cout <<"EvFragLength = "<<EvFragLength<<std::endl;
  /* 1 */ //There should always be a second CDF header word indicated.
  if (!dccHeader->thereIsASecondCDFHeaderWord()) 
    {
    std::cout <<"No second CDF header found!"<<std::endl;
    }

  return;
} // void HcalEEUSMonitor::unpack(...)
