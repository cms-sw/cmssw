#include "DQM/CSCMonitorModule/interface/CSCMonitor.h"

void CSCMonitor::process(CSCDCCExaminer * examiner, CSCDCCEventData * dccData )
{

  nEvents = nEvents +1;
  // ! Node number should be defined
  int32_t nodeNumber = 0;
  if (examiner != NULL) {
    binExaminer(*examiner,nodeNumber);
  }

  if (dccData==NULL) {
    LOG4CPLUS_WARN(logger_, "Empty data event " << nEvents);
    return;
  }
  const std::vector<CSCDDUEventData> & dduData = dccData->dduData();

  //edm::LogInfo ("CSC DQM ") << "CSCMonitor::process #" << dec << nEvents
  //       << "> Number of DDU = " <<dduData.size();

  for (int ddu=0; ddu<(int)dduData.size(); ++ddu) {
    this->monitorDDU(dduData[ddu], nodeNumber );
  }


  if((!(nEvents%saveRootFileEventsInterval ))&&(fSaveHistos ) ) {
//    LOG4CPLUS_WARN(logger_, "Begin save at event " << nEvents);
    dbe->save(RootHistoFile);
    //    this->saveToROOTFile(RootHistoFile);
  }

  // usleep(100000);
}


void CSCMonitor::process(const char * data, int32_t dataSize, uint32_t errorStat, int32_t nodeNumber)
{

  nEvents = nEvents+1;
 
  if (unpackMask == UNPACK_NONE) return;

  string nodeTag = Form("EMU_%d",nodeNumber); // == This emuMonitor node number
  std::map<std::string, ME_List >::iterator itr;

  CSCMonitorObject *mo = NULL;  // == pointer to MonitoringObject
  unpackedDMBcount = 0;

  // == Check and book global node specific histos
  if (MEs.size() == 0 || ((itr = MEs.find(nodeTag)) == MEs.end())) {
    LOG4CPLUS_WARN(logger_, " List of MEs for " << nodeTag << " not found. Booking...")
      fBusy = true;
    MEs[nodeTag] = bookCommon(nodeNumber);
    // printMECollection(MEs[nodeTag]);
    fBusy = false;
  }

  ME_List& nodeME = MEs[nodeTag];

  if (isMEvalid(nodeME, "Buffer_Size", mo)) mo->Fill(dataSize);

  // ==     Check DDU Readout Error Status
  if (isMEvalid(nodeME, "Readout_Errors", mo)) { 
    if(errorStat != 0) {
      LOG4CPLUS_WARN(logger_,nodeTag << " Non-zero Readout Error Status is observed: 0x" << std::hex << errorStat << " mask 0x" << dduCheckMask);
      for (int i=0; i<16; i++) if ((errorStat>>i) & 0x1) mo->Fill(0.,i);
    }
    else {
      // LOG4CPLUS_DEBUG(logger_,nodeTag << " Readout Error Status is OK: 0x" << std::hex << errorStat);
    }
  }

  //	Accept or deny event according to DDU Readout Error and dduCheckMask
  if (((uint32_t)errorStat & dduCheckMask) > 0) {
    LOG4CPLUS_WARN(logger_,nodeTag << "Event skiped because of DDU Readout Error");
    return;
  }

  /*  CSCDCCExaminer bin_examiner;

  //	Accept or deny event according to Binary Error and binCheckMask
  if ((BinaryErrorStatus != 0) || (BinaryWarningStatus != 0)) {
  //  fillChamberBinCheck();
  }

  if (( bin_examiner.errors() & binCheckMask)>0) {
  LOG4CPLUS_WARN(logger_,nodeTag << " Event skiped because of Binary Error");
  return;
  }
  */
  // LOG4CPLUS_DEBUG(logger_,nodeTag << " Event is accepted");

}


void CSCMonitor::monitorDDU(const CSCDDUEventData& dduData, int nodeNumber)
{
  string nodeTag = Form("EMU_%d",nodeNumber); // == This emuMonitor node number
  std::map<std::string, ME_List >::iterator itr;
  //  ME_List& nodeME=NULL; // === Global histos specific for this emuMonitor node
  CSCMonitorObject *mo = NULL;  // == pointer to MonitoringObject
  unpackedDMBcount = 0;

  if (unpackMask == UNPACK_NONE) return;

  // == Check and book global node specific histos
  if (MEs.size() == 0 || ((itr = MEs.find(nodeTag)) == MEs.end())) {
    LOG4CPLUS_WARN(logger_, " List of MEs for " << nodeTag << " not found. Booking...")
      fBusy = true;
    MEs[nodeTag] = bookCommon(nodeNumber);
    //    MECanvases[nodeTag] = bookCommonCanvases(nodeNumber);
    printMECollection(MEs[nodeTag]);
    fBusy = false;
  }

  ME_List& nodeME = MEs[nodeTag];


  // CSCDDUEventData::setDebug(true);
  int dduID = 0;
  // CSCDDUEventData dduData((uint16_t *) data);

  CSCDDUHeader dduHeader  = dduData.header();
  CSCDDUTrailer dduTrailer = dduData.trailer();
  
  dduID = dduHeader.source_id();
  
  if (isMEvalid(nodeME, "Source_ID", mo)) { 
    mo->Fill(dduID%40);
    mo->SetBinLabel(dduID%40+1, Form("%d",dduID), 1);
  }



  string dduTag = Form("DDU_%d",dduID);

  if (MEs.size() == 0 || ((itr = MEs.find(dduTag)) == MEs.end())) {
    LOG4CPLUS_WARN(logger_, " List of MEs for " << dduTag << " not found. Booking...")
      fBusy = true;
    MEs[dduTag] = bookDDU(dduID);
    // MECanvases[dduTag] = bookDDUCanvases(dduID);
    printMECollection(MEs[dduTag]);
    fBusy = false;
  }

  ME_List& dduME = MEs[dduTag];



  // LOG4CPLUS_DEBUG(logger_,"Start unpacking " << dduTag);

  // ==     Check binary Error status at DDU Trailer
  uint32_t trl_errorstat = dduTrailer.errorstat();
  // LOG4CPLUS_DEBUG(logger_,dduTag << " Trailer Error Status = 0x" << hex << trl_errorstat);
  for (int i=0; i<32; i++) {
    if ((trl_errorstat>>i) & 0x1) {
      if (isMEvalid(dduME,"Trailer_ErrorStat_Rate", mo)) { 
	mo->Fill(i);
	double freq = (100.0*mo->GetBinContent(i+1))/nEvents;
	if (isMEvalid(dduME, "Trailer_ErrorStat_Frequency", mo)) mo->SetBinContent(i+1, freq);
      }
      if (isMEvalid(dduME, "Trailer_ErrorStat_Table", mo)) mo->Fill(0.,i);
      if (isMEvalid(dduME, "Trailer_ErrorStat_vs_nEvents", mo)) mo->Fill(nEvents, i);
    }
  }
	
  if (isMEvalid(dduME,"Trailer_ErrorStat_Table", mo)) mo->SetEntries(nEvents);
  if (isMEvalid(dduME,"Trailer_ErrorStat_Frequency", mo)) mo->SetEntries(nEvents);
  if (isMEvalid(dduME,"Trailer_ErrorStat_vs_nEvents", mo)) { 
    mo->SetEntries(nEvents);
    mo->SetAxisRange(0, nEvents, "X");
  }


  // Need to access DDU event buffer size
  // if (isMEvalid(dduME, "Buffer_Size", mo)) mo->Fill(dataSize);
  // ==     DDU word counter
  int trl_word_count = 0;
  trl_word_count = dduTrailer.wordcount();
  if (isMEvalid(dduME, "Word_Count", mo)) mo->Fill(trl_word_count );
//  LOG4CPLUS_DEBUG(logger_,dduTag << " Trailer Word (64 bits) Count = " << dec << trl_word_count);

  // ==     DDU Header banch crossing number (BXN)
  BXN=dduHeader.bxnum();
  // LOG4CPLUS_DEBUG(logger_,dduTag << " DDU Header BXN Number = " << dec << BXN);
  if (isMEvalid(dduME, "BXN", mo)) mo->Fill((double)BXN);

  // ==     L1A number from DDU Header
  int L1ANumber_previous_event = L1ANumber;
  L1ANumber = (int)(dduHeader.lvl1num());
  // LOG4CPLUS_DEBUG(logger_,dduTag << " Header L1A Number = " << dec << L1ANumber);
  if (isMEvalid(dduME, "L1A_Increment", mo)) dduME["L1A_Increment"]->Fill(L1ANumber - L1ANumber_previous_event);

  if (isMEvalid(dduME, "L1A_Increment_vs_nEvents", mo)) {
    if(L1ANumber - L1ANumber_previous_event == 0) {
      mo->Fill((double)(nEvents), 0.0);
    }
    if(L1ANumber - L1ANumber_previous_event == 1) {
      mo->Fill((double)(nEvents), 1.0);
    }
    if(L1ANumber - L1ANumber_previous_event > 1) {
      mo->Fill((double)(nEvents), 2.0);

    }
    mo->SetAxisRange(0, nEvents, "X");
  }

  // ==     Occupancy and number of DMB (CSC) with Data available (DAV) in header of particular DDU
  int dmb_dav_header      = 0;
  int dmb_dav_header_cnt  = 0;

  int ddu_connected_inputs= 0;
  int csc_error_state     = 0;
  int csc_warning_state   = 0;

  //  ==    Number of active DMB (CSC) in header of particular DDU
  int dmb_active_header   = 0;

  dmb_dav_header     = dduHeader.dmb_dav();
  dmb_active_header  = (int)(dduHeader.ncsc()&0xF);
  csc_error_state    = dduTrailer.dmb_full();
  csc_warning_state  = dduTrailer.dmb_warn();
  ddu_connected_inputs=dduHeader.live_cscs();


  // LOG4CPLUS_DEBUG(logger_,dduTag << " Header DMB DAV = 0x" << hex << dmb_dav_header);
  // LOG4CPLUS_DEBUG(logger_,dduTag << " Header Number of Active DMB = " << dec << dmb_active_header);


  double freq = 0;
  for (int i=0; i<16; ++i) {
    if ((dmb_dav_header>>i) & 0x1) {
      dmb_dav_header_cnt = dmb_dav_header_cnt + 1;      
      if (isMEvalid(dduME, "DMB_DAV_Header_Occupancy_Rate", mo)) {
	mo->Fill(i);
	freq = 100.0*(mo->GetBinContent(i+1))/nEvents;
        if (isMEvalid(dduME, "DMB_DAV_Header_Occupancy", mo)) mo->SetBinContent(i+1,freq);
      }
    }

    if( (ddu_connected_inputs>>i) & 0x1 ){
      if (isMEvalid(dduME, "DMB_Connected_Inputs_Rate", mo)) {
	mo->Fill(i);
	freq = 100.0*(mo->GetBinContent(i+1))/nEvents;
	if (isMEvalid(dduME, "DMB_Connected_Inputs", mo)) mo->SetBinContent(i+1, freq);
      }
    }
    if( (csc_error_state>>i) & 0x1 ){
      if (isMEvalid(dduME, "CSC_Errors_Rate", mo)) {
	mo->Fill(i);
	freq = 100.0*(mo->GetBinContent(i+1))/nEvents;
	if (isMEvalid(dduME, "CSC_Errors", mo)) mo->SetBinContent(i+1, freq);
      }
    }
    if( (csc_warning_state>>i) & 0x1 ){
      if (isMEvalid(dduME, "CSC_Warnings_Rate", mo)) {
	mo->Fill(i);
	freq = 100.0*(mo->GetBinContent(i+1))/nEvents;
	if (isMEvalid(dduME,"CSC_Warnings", mo)) mo->SetBinContent(i+1, freq);
      }
    }

  }
  if (isMEvalid(dduME,"DMB_DAV_Header_Occupancy",mo)) mo->SetEntries(nEvents);

  if (isMEvalid(dduME, "DMB_Connected_Inputs", mo)) mo->SetEntries(nEvents);
  if (isMEvalid(dduME, "CSC_Errors", mo)) mo->SetEntries(nEvents);
  if (isMEvalid(dduME, "CSC_Warnings", mo)) mo->SetEntries(nEvents);

  if (isMEvalid(dduME, "DMB_Active_Header_Count", mo)) mo->Fill(dmb_active_header);
  if (isMEvalid(dduME, "DMB_DAV_Header_Count_vs_DMB_Active_Header_Count", mo)) mo->Fill(dmb_active_header,dmb_dav_header_cnt);

  if (unpackMask & UNPACK_CSC) {
    //      Unpack all founded CSC
    vector<CSCEventData> chamberDatas;
    chamberDatas = dduData.cscData();
    //      Unpack DMB for each particular CSC
    // int unpacked_dmb_cnt = 0;
    for(vector<CSCEventData>::iterator chamberDataItr = chamberDatas.begin(); chamberDataItr != chamberDatas.end(); ++chamberDataItr) {
      unpackedDMBcount++;
      // unpacked_dmb_cnt=unpacked_dmb_cnt+1;
      // LOG4CPLUS_DEBUG(logger_,
	//	      "Found DMB " << dec << unpackedDMBcount  << ". Run unpacking procedure...");
      monitorCSC(*chamberDataItr, nodeNumber, dduID);
      // LOG4CPLUS_DEBUG(logger_,
	//	      "Unpacking procedure for DMB " << dec << unpackedDMBcount << " finished");
    }
    // LOG4CPLUS_DEBUG(logger_,
//		    "Total number of unpacked DMB = " << dec << unpackedDMBcount);

    if (isMEvalid(dduME,"DMB_unpacked_vs_DAV",mo)) mo->Fill(dmb_active_header, unpackedDMBcount);
    if (isMEvalid(nodeME,"Unpacking_Match_vs_nEvents", mo)) {
      if(dmb_active_header == unpackedDMBcount) {
	mo->Fill(nEvents, 0.0);
      }
      else {
	mo->Fill(nEvents, 1.0);
      }
      mo->SetAxisRange(0, nEvents, "X");
    }
  }
  // LOG4CPLUS_DEBUG(logger_,
//		  "END OF EVENT :-(");
}


void CSCMonitor::binExaminer(CSCDCCExaminer & bin_checker,int32_t nodeNumber) {
  string nodeTag = Form("EMU_%d",nodeNumber); // == This emuMonitor node number
  std::map<std::string, ME_List >::iterator itr;

  // == Check and book global node specific histos
  if (MEs.size() == 0 || ((itr = MEs.find(nodeTag)) == MEs.end())) {
    LOG4CPLUS_WARN(logger_, " List of MEs for " << nodeTag << " not found. Booking...")
      fBusy = true;
    MEs[nodeTag] = bookCommon(nodeNumber);
  //  printMECollection(MEs[nodeTag]);
    fBusy = false;
  }

//  LOG4CPLUS_INFO(logger_, "========== binExaminer " << nEvents);

  ME_List& nodeME = MEs[nodeTag];

  CSCMonitorObject* mo = NULL;
  CSCMonitorObject* mof = NULL;

  //  if(check_bin_error){
  uint32_t BinaryErrorStatus = 0, BinaryWarningStatus = 0;
  BinaryErrorStatus   = bin_checker.errors();
  BinaryWarningStatus = bin_checker.warnings();

  if(BinaryErrorStatus != 0) {
//    LOG4CPLUS_WARN(logger_,nodeTag << " Nonzero Binary Errors Status is observed: 0x" << std::hex << BinaryErrorStatus << " mask: 0x" << binCheckMask);

    if (isMEvalid(nodeME, "BinaryChecker_Errors", mo)) {
      for(int i=0; i<bin_checker.nERRORS; i++) { // run over all errors
	if( bin_checker.error(i) ) mo->Fill(0.,i);
      }
    }

  }
  else {
  //  LOG4CPLUS_DEBUG(logger_,nodeTag << " Binary Error Status is OK: 0x" << hex << BinaryErrorStatus);
  }


  if(BinaryWarningStatus != 0) {
 //   LOG4CPLUS_WARN(logger_,nodeTag << " Nonzero Binary Warnings Status is observed: 0x"
//		   << hex << BinaryWarningStatus)
      if (isMEvalid(nodeME, "BinaryChecker_Warnings", mo)) {
	for(int i=0; i<bin_checker.nWARNINGS; i++) { // run over all warnings
	  if( bin_checker.warning(i) ) mo->Fill(0.,i);
	}
      }

  }
  else {
    // LOG4CPLUS_DEBUG(logger_,nodeTag << " Binary Warnings Status is OK: 0x" << hex << BinaryWarningStatus);
  }

  if (isMEvalid(nodeME, "Data_Format_Check_vs_nEvents", mo)) {
    if( BinaryErrorStatus != 0 ) {
      mo->Fill(nEvents,2.0);
    } else {
      mo->Fill(nEvents,0.0);
    }

    //	if any warnings
    if( BinaryWarningStatus != 0 ) {
      mo->Fill(nEvents,1.0);
    }
    mo->SetAxisRange(0, nEvents, "X");
    // LOG4CPLUS_DEBUG(logger_,nodeTag << " Error checking has been done");
  }

  if ((unpackMask & UNPACK_CSC)==0) return;  
  if ((BinaryErrorStatus != 0) || (BinaryWarningStatus != 0)) {

    map<int,long> checkerErrors = bin_checker.errorsDetailed();
    map<int,long>::const_iterator chamber = checkerErrors.begin();

    while( chamber != checkerErrors.end() ){

      int ChamberID     = chamber->first;
      string cscTag(Form("CSC_%03d_%02d", (chamber->first>>4) & 0xFF, chamber->first & 0xF));
      map<string, ME_List >::iterator h_itr = MEs.find(cscTag);

      if ((((chamber->first>>4) & 0xFF) ==255) ||
	  (chamber->second & 0x80)) { chamber++; continue;} // = Skip chamber detection if DMB header is missing (Error code 6)

      if (h_itr == MEs.end() || (MEs.size()==0)) {
	LOG4CPLUS_WARN(logger_,
		       "List of Histos for " << cscTag <<  " not found");
//	LOG4CPLUS_DEBUG(logger_,
//			"Booking Histos for " << cscTag);
	fBusy = true;
	MEs[cscTag] = bookChamber(ChamberID);
	//  MECanvases[cscTag] = bookChamberCanvases(ChamberID);
//	printMECollection(MEs[cscTag]);
	fBusy = false;
      }

      ME_List& cscME = MEs[cscTag];

      /*    if ( (bin_checker.errors() & binCheckMask) != 0) {
	    nDMBEvents[cscTag]++;
	    }
      */
      if (isMEvalid(cscME, "BinCheck_ErrorStat_Table", mo)
	  && isMEvalid(cscME, "BinCheck_ErrorStat_Frequency", mof)) {
	for(int bit=5; bit<24; bit++)
	  if( chamber->second & (1<<bit) ) {
	    mo->Fill(0.,bit-5);
	   
	    double freq = (100.0*mo->GetBinContent(1,bit-4))/nDMBEvents[cscTag];
	    mof->SetBinContent(bit-4, freq);
	  }
	mo->SetEntries(nDMBEvents[cscTag]);
	mof->SetEntries(nDMBEvents[cscTag]);
      }
      // Fill common CSC errors Histo
      int crateID   = (chamber->first>>4) & 0xFF;
      int dmbID     = chamber->first & 0xF;
      int CSCtype   = 0;
      int CSCposition = 0;
      this->getCSCFromMap(crateID, dmbID, CSCtype, CSCposition );
      if (CSCtype && CSCposition && isMEvalid(nodeME, "CSC_Data_Format_Errors", mo)) {
        // mo->Fill((chamber->first>>4) & 0xFF, chamber->first & 0xF);
//	LOG4CPLUS_INFO(logger_, "========== binExaminer CSC Error " << nEvents << " crate" << crateID << " slot" << dmbID);
	mo->Fill(CSCposition-1, CSCtype);
      }
      chamber++;
    }

    map<int,long> checkerWarnings  = bin_checker.warningsDetailed();

    chamber = checkerWarnings.begin();

    while( chamber != checkerWarnings.end() ){

      int ChamberID     = chamber->first;
      string cscTag(Form("CSC_%03d_%02d", (chamber->first>>4) & 0xFF, chamber->first & 0xF));

      if (((chamber->first>>4) & 0xFF) ==255) {chamber++; continue;}
      map<string, ME_List >::iterator h_itr = MEs.find(cscTag);
      if (h_itr == MEs.end() || (MEs.size()==0)) {
	LOG4CPLUS_WARN(logger_,
		       "List of Histos for " << cscTag <<  " not found");
//	LOG4CPLUS_DEBUG(logger_,
//			"Booking Histos for " << cscTag);
	fBusy = true;
	MEs[cscTag] = bookChamber(ChamberID);
//	printMECollection(MEs[cscTag]);
	fBusy = false;
      }

      ME_List& cscME = MEs[cscTag];

      if (isMEvalid(cscME, "BinCheck_WarningStat_Table", mo)
	  && isMEvalid(cscME, "BinCheck_WarningStat_Frequency", mof)) {
	for(int bit=1; bit<2; bit++)
	  if( chamber->second & (1<<bit) ) {
	    mo->Fill(0.,bit-1);
	    double freq = (100.0*mo->GetBinContent(1,bit))/nDMBEvents[cscTag];
	    mof->SetBinContent(bit, freq);
	  }
	mo->SetEntries(nDMBEvents[cscTag]);
	mof->SetEntries(nDMBEvents[cscTag]);
      }
      
     
      chamber++;
    }
  }
}
