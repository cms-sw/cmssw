#include "DQM/CSCMonitorModule/interface/CSCMonitor.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDDUEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDDUHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDDUTrailer.h"

// Storing of DDU information into meCollection[0]

void CSCMonitor::MonitorDDU(const CSCDDUEventData& dduEvent){

/// unsigned char * data, int dataLength, unsigned short errorStat

  string meName;
  map<string, MonitorElement*> me = meCollection[0];

  CSCDDUHeader  dduHeader = dduEvent.header();
  CSCDDUTrailer dduTrailer= dduEvent.trailer();

//  int errorStat=errorstat();
 
  int dataLength=dduEvent.size();// or sizeInWords()?
  
  
  me["DDU_Buffer_Size"]->Fill(dataLength);
  nEvents = nEvents +1;
  if(printout) cout << "CSCMonitor::MonitorDDU #" << dec << nEvents 
		    << "> BEGINNING OF EVENT :-) Buffer size = " << dec 
		    << dataLength << endl;

 int crateID	 = 0xFF;
 int dmbID	 = 0xF;
 int ChamberID   = 0xFFF;
    


   if(printout) cout << "D**EmuFillCommon> event #" << dec << nEvents 
	   << "> Start DDU unpacking" << endl;


/// BINARY ERROR STATUS AT DDU TRAILER
  unsigned int trl_errorstat = 0x0;
  trl_errorstat     = dduTrailer.errorstat();


  if(printout) {
    cout << "CSCMonitor::MonitorDDU event #" << dec << nEvents 
	 << "> DDU Trailer Error Status = 0x" << hex << trl_errorstat << endl;
  }
    for (int i=0; i<32; i++) {
      if ((trl_errorstat>>i) & 0x1) {
	me["DDU_Trailer_ErrorStat_Rate"]->Fill(i);
	//me["DDU_Trailer_ErrorStat_Occupancy"]->SetBinContent(i+1,100.0*(me["DDU_Trailer_ErrorStat_Rate"]->GetBinContent(i+1))/nEvents);
	me["DDU_Trailer_ErrorStat_Table"]->Fill(0.,i);
	me["DDU_Trailer_ErrorStat_vs_nEvents"]->Fill(nEvents, i);
      }
    }
    //me["DDU_Trailer_ErrorStat_Table"]->SetEntries(nEvents);
    //me["DDU_Trailer_ErrorStat_Occupancy"]->SetEntries(nEvents);
    //me["DDU_Trailer_ErrorStat_vs_nEvents"]->SetEntries(nEvents);
    //me["DDU_Trailer_ErrorStat_vs_nEvents"]->SetAxisRange(0, nEvents, "X");

/// DDU WORD COUNTER
  int trl_word_count = 0;
  trl_word_count = dduTrailer.wordcount();
  
  me["DDU_Word_Count"]->Fill(trl_word_count);
  if(printout)  
    cout << "CSCMonitor::MonitorDDU event #" << dec << nEvents 
 	 << "> DDU Trailer Word (64 bits) Count = " << dec << trl_word_count << endl;

///	BXN from DDU Header 

  dduBX=dduHeader.bxnum();

  if(printout)  
    cout << "CSCMonitor::MonitorDDU event #" << dec << nEvents 
	 << "> DDU Header BX Number = " << dec << dduBX << endl;
  
  me["DDU_BXN"]->Fill(dduBX);

///	L1A number from DDU Header
  int L1ANumber_previous_event = L1ANumber;
    L1ANumber = (int)(dduHeader.lvl1num());
  
  if(printout)  
    cout << "CSCMonitor::MonitorDDU event #" << dec << nEvents 
	 << "> DDU Header L1A Number = " << dec << L1ANumber << endl;
 
    me["DDU_L1A_Increment"]->Fill(L1ANumber - L1ANumber_previous_event);
    if(L1ANumber - L1ANumber_previous_event == 0) {
      me["DDU_L1A_Increment_vs_nEvents"]->Fill((int)(nEvents), 0.0);
    }
    if(L1ANumber - L1ANumber_previous_event == 1) {
      me["DDU_L1A_Increment_vs_nEvents"]->Fill((int)(nEvents), 1.0);
    }
    if(L1ANumber - L1ANumber_previous_event > 1) {
      me["DDU_L1A_Increment_vs_nEvents"]->Fill((int)(nEvents), 2.0);
    }
    //me["DDU_L1A_Increment_vs_nEvents"]->SetAxisRange(0, nEvents, "X");
 

/// Occupancy and number of DMB (CSC) with Data available (DAV) in DDU header
  int dmb_dav_header      = 0;
  int dmb_dav_header_cnt  = 0;
  //KK
  int ddu_connected_inputs= 0;
  int csc_error_state     = 0;
  int csc_warning_state   = 0;
  //KK end

/// Number of active DMB (CSC) in DDU header
  int dmb_active_header   = 0;
 
    dmb_dav_header     = dduHeader.dmb_dav();
    dmb_active_header  = (int)(dduHeader.ncsc()&0xF);
    csc_error_state    = dduTrailer.dmb_full();
    csc_warning_state  = dduTrailer.dmb_warn();
   // ddu_connected_inputs=dduHeader.live_cscs();

  if(printout) {
      cout << "CSCMonitor::MonitorDDU event #" << dec << nEvents 
	   << "> DDU Header DMB DAV = 0x" << hex << dmb_dav_header << endl;  
      cout << "CSCMonitor::MonitorDDU event #" << dec << nEvents 
	   << "> DDU Header Number of Active DMB = " << dec << dmb_active_header << endl;
  }

    for (int i=0; i<16; ++i) {
      if ((dmb_dav_header>>i) & 0x1) {
	dmb_dav_header_cnt = dmb_dav_header_cnt + 1;
	me["DDU_DMB_DAV_Header_Occupancy_Rate"]->Fill(i);
	//me["DDU_DMB_DAV_Header_Occupancy"]->SetBinContent(i+1, 100.0*(me["DDU_DMB_DAV_Header_Occupancy_Rate"]->GetBinContent(i+1))/nEvents);
      }
      //KK
      if( (ddu_connected_inputs>>i) & 0x1 ){
	me["DDU_DMB_Connected_Inputs_Rate"]->Fill(i);
	//me["DDU_DMB_Connected_Inputs"]->SetBinContent(i+1, 100.0*(me["DDU_DMB_Connected_Inputs_Rate"]->GetBinContent(i+1))/nEvents);

      }
      if( (csc_error_state>>i) & 0x1 ){
	me["DDU_CSC_Errors_Rate"]->Fill(i);
	//me["DDU_CSC_Errors"]->SetBinContent(i+1, 100.0*(me["DDU_CSC_Errors_Rate"]->GetBinContent(i+1))/nEvents);
      }
      if( (csc_warning_state>>i) & 0x1 ){
	me["DDU_CSC_Warnings_Rate"]->Fill(i);
	//me["DDU_CSC_Warnings"]->SetBinContent(i+1, 100.0*(me["DDU_CSC_Warnings_Rate"]->GetBinContent(i+1))/nEvents);
      }
      //KK end
    }
    //me["DDU_DMB_DAV_Header_Occupancy"]->SetEntries(nEvents);
    //KK
    //me["DDU_DMB_Connected_Inputs"]->SetEntries(nEvents);
    //me["DDU_CSC_Errors"]->SetEntries(nEvents);
    //me["DDU_CSC_Warnings"]->SetEntries(nEvents);
    //KK end
    me["DDU_DMB_Active_Header_Count"]->Fill(dmb_active_header);
    me["DDU_DMB_DAV_Header_Count_vs_DMB_Active_Header_Count"]->Fill(dmb_active_header,dmb_dav_header_cnt);
  

/// Unpack all founded CSC
  vector<CSCEventData> chamberDatas=dduEvent.cscData();

/// Unpack DMB for each particular CSC
  int unpacked_dmb_cnt = 0;
  
    for(vector<CSCEventData>::iterator chamberDataItr = chamberDatas.begin(); 
	chamberDataItr != chamberDatas.end(); ++chamberDataItr) {
      unpacked_dmb_cnt=unpacked_dmb_cnt+1;
      if(printout) {
	cout << "CSCMonitor::MonitorDDU event #" << dec << nEvents 
	     << "> Found DMB " << dec << unpacked_dmb_cnt 
	     << ". Run unpacking procedure..." << endl;
      }
      //fill(*chamberDataItr);
      if(printout) {
	cout << "CSCMonitor::MonitorDDU event #" << dec << nEvents 
	     << "> Unpacking procedure for DMB " << dec << unpacked_dmb_cnt << " finished" << endl;
      }
    }
  
  if(printout) 
    cout << "CSCMonitor::MonitorDDU event #" << dec << nEvents 
	 << "> Total number of unpacked DMB = " << dec << unpacked_dmb_cnt << endl;

  
    me["DDU_DMB_unpacked_vs_DAV"]->Fill(dmb_active_header, unpacked_dmb_cnt);
    if(dmb_active_header == unpacked_dmb_cnt) {
      me["DDU_Unpacking_Match_vs_nEvents"]->Fill(nEvents, 0.0);
    }
    else {
      me["DDU_Unpacking_Match_vs_nEvents"]->Fill(nEvents, 1.0);
    }
    //me["DDU_Unpacking_Match_vs_nEvents"]->SetAxisRange(0, nEvents, "X");
  
  if(printout) 
    cout << "CSCMonitor::MonitorDDU event #" << dec << nEvents 
	 << "> END OF EVENT :-(" << endl;


}

