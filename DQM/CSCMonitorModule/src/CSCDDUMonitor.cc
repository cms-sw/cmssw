/** \file
 *
 * implementation of CSCMonitor::MonitorDDU(...) method
 * 
 *  $Date: 2006/01/18 11:22:47 $
 *  $Revision: 1.4 $
 *
 * \author Ilaria Segoni
 */

#include "DQM/CSCMonitorModule/interface/CSCMonitor.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDDUEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDDUHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDDUTrailer.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"

void CSCMonitor::MonitorDDU(const CSCDDUEventData& dduEvent, int dduNumber){

  if(printout) cout << "Beginning of CSCMonitor::MonitorDDU"<<endl;

  if(!dduBooked[dduNumber]){
	meDDU[dduNumber] = book_common(dduNumber);
   	dduBooked[dduNumber]=true;  
  }
  
  string meName;
  map<string, MonitorElement*> me = meDDU[dduNumber];


  CSCDDUHeader  dduHeader = dduEvent.header();
  CSCDDUTrailer dduTrailer= dduEvent.trailer();

  //  int errorStat=errorstat();
 
  int dataLength=dduEvent.size();// or sizeInWords()?
  
  
  meName = Form("DDU_Buffer_Size_%d",dduNumber);
  me[meName]->Fill(dataLength);

  if(printout) cout << "CSCMonitor::MonitorDDU #" << dec << nEvents 
	<< "> BEGINNING OF EVENT, Buffer size = " << dec 
	<< dataLength << endl;

   if(printout) cout << "CSCMonitor::MonitorDDU event #" << dec << nEvents 
	<< "> Start DDU MonitorElements Filling." << endl;


/// BINARY ERROR STATUS AT DDU TRAILER
    unsigned int trl_errorstat = 0x0;
    trl_errorstat     = dduTrailer.errorstat();


    if(printout) {
	cout << "CSCMonitor::MonitorDDU event #" << dec << nEvents 
	<< "> DDU Trailer Error Status = 0x" << hex << trl_errorstat << endl;
     }
     for (int i=0; i<32; i++) {
	if ((trl_errorstat>>i) & 0x1) {
    
		meName = Form("DDU_Trailer_ErrorStat_Rate_%d",dduNumber);
		me[meName]->Fill(i);
  
		//meName = Form("DDU_Trailer_ErrorStat_Occupancy_%d",dduNumber);
		//string meName2=Form("DDU_Trailer_ErrorStat_Rate_%d",dduNumber);
		//me[meName]->SetBinContent(i+1,100.0*(me[meName2]->GetBinContent(i+1))/nEvents);
   
		meName = Form("DDU_Trailer_ErrorStat_Table_%d",dduNumber);
		me[meName]->Fill(0.,i);
   
		meName = Form("DDU_Trailer_ErrorStat_vs_nEvents_%d",dduNumber);
		me[meName]->Fill(nEvents, i);
	}
      }

      //meName = Form("DDU_Trailer_ErrorStat_Table_%d",dduNumber);
     // me[meName]->SetEntries(nEvents);
   
      //meName = Form("DDU_Trailer_ErrorStat_Occupancy_%d",dduNumber);
      //me[meName]->SetEntries(nEvents);
   
      //meName = Form("DDU_Trailer_ErrorStat_vs_nEvents_%d",dduNumber);
      //me[meName]->SetEntries(nEvents);
   
      //meName = Form("DDU_Trailer_ErrorStat_vs_nEvents_%d",dduNumber);
      //me[meName]->SetAxisRange(0, nEvents, "X");

/// DDU WORD COUNTER
   int trl_word_count = 0;
   trl_word_count = dduTrailer.wordcount();

   meName = Form("DDU_Word_Count_%d",dduNumber);
   me[meName]->Fill(trl_word_count);
   if(printout) cout << "CSCMonitor::MonitorDDU event #" << dec << nEvents 
	<< "> DDU Trailer Word (64 bits) Count = " << dec << trl_word_count << endl;

///	BXN from DDU Header 

    dduBX[dduNumber]=dduHeader.bxnum();

    if(printout)  cout << "CSCMonitor::MonitorDDU event #" << dec << nEvents 
	<< "> DDU Header BX Number = " << dec << dduBX[dduNumber] << endl;
  
    meName = Form("DDU_BXN_%d",dduNumber);
    me[meName]->Fill(dduBX[dduNumber]);

///	L1A number from DDU Header
    int L1ANumber_previous_event = L1ANumber[dduNumber];
    L1ANumber[dduNumber] = (int)(dduHeader.lvl1num());
  
    if(printout) cout << "CSCMonitor::MonitorDDU event #" << dec << nEvents 
	<< "> DDU Header L1A Number = " << dec << L1ANumber[dduNumber] <<" L1A Number Previous="<< L1ANumber_previous_event<<"difference"<<L1ANumber[dduNumber] - L1ANumber_previous_event<<endl;
 
    meName = Form("DDU_L1A_Increment_%d",dduNumber);
    me[meName]->Fill(L1ANumber[dduNumber] - L1ANumber_previous_event);
      
     
     meName = Form("DDU_L1A_Increment_vs_nEvents_%d",dduNumber);
     if(L1ANumber[dduNumber] - L1ANumber_previous_event == 0) {
	me[meName]->Fill((int)(nEvents), 0.0);
     }
     if(L1ANumber[dduNumber] - L1ANumber_previous_event == 1) {
        me[meName]->Fill((int)(nEvents), 1.0);
     }
     if(L1ANumber[dduNumber] - L1ANumber_previous_event > 1) {
        me[meName]->Fill((int)(nEvents), 2.0);
     }
     //me[meName]->SetAxisRange(0, nEvents, "X");
 

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
      //ddu_connected_inputs=dduHeader.live_cscs();

    if(printout) {
	cout << "CSCMonitor::MonitorDDU event #" << dec << nEvents 
		<< "> DDU Header DMB DAV = 0x" << hex << dmb_dav_header << endl;  
	cout << "CSCMonitor::MonitorDDU event #" << dec << nEvents 
		<< "> DDU Header Number of Active DMB = " << dec << dmb_active_header << endl;
    }

   for (int i=0; i<16; ++i) {
	if ((dmb_dav_header>>i) & 0x1) {
		dmb_dav_header_cnt = dmb_dav_header_cnt + 1;
		meName = Form("DDU_DMB_DAV_Header_Occupancy_Rate_%d",dduNumber);
		me[meName]->Fill(i);
                
		//string meName2 =Form("DDU_DMB_DAV_Header_Occupancy_Rate_%d",dduNumber);
	        //meName = Form("DDU_DMB_DAV_Header_Occupancy_%d",dduNumber);
		//me[meName]->SetBinContent(i+1, 100.0*(me[meName2]->GetBinContent(i+1))/nEvents);
        }

	if( (ddu_connected_inputs>>i) & 0x1 ){
		meName = Form("DDU_DMB_Connected_Inputs_Rate_%d",dduNumber);
		me[meName]->Fill(i);
		
		//string meName2 =Form("DDU_DMB_Connected_Inputs_Rate_%d",dduNumber);
		//meName = Form("DDU_DMB_Connected_Inputs_%d",dduNumber);
		//me[meName]->SetBinContent(i+1, 100.0*(me[meName2]->GetBinContent(i+1))/nEvents);
        }

        if( (csc_error_state>>i) & 0x1 ){
		meName = Form("DDU_CSC_Errors_Rate_%d",dduNumber);
		me[meName]->Fill(i);
		
		//string meName2=Form("DDU_CSC_Errors_Rate_%d",dduNumber);
		//meName = Form("DDU_CSC_Errors_Rate_%d",dduNumber);
		//me[meName]->SetBinContent(i+1, 100.0*(me[meName2]->GetBinContent(i+1))/nEvents);
        }

        if( (csc_warning_state>>i) & 0x1 ){        
		meName = Form("DDU_CSC_Warnings_Rate_%d",dduNumber);
		me[meName]->Fill(i);
		
		//string meName2= Form("DDU_CSC_Warnings_Rate_%d",dduNumber); 
		//meName = Form("DDU_CSC_Warnings_%d",dduNumber);
		//me[meName]->SetBinContent(i+1, 100.0*(me[meName2]->GetBinContent(i+1))/nEvents);
	}

   }
      
   //meName = Form("DDU_DMB_DAV_Header_Occupancy_%d",dduNumber);
   //me[meName]->SetEntries(nEvents);
 
   //meName = Form("DDU_DMB_Connected_Inputs_%d",dduNumber);
   //me[meName]->SetEntries(nEvents);
   
   //meName = Form("DDU_CSC_Errors_%d",dduNumber);
   //me[meName]->SetEntries(nEvents);
   
   //meName = Form("DDU_CSC_Warnings_%d",dduNumber);
   //me[meName]->SetEntries(nEvents);

      
   meName = Form("DDU_DMB_Active_Header_Count_%d",dduNumber);
   me[meName]->Fill(dmb_active_header);
     
     
   meName = Form("DDU_DMB_DAV_Header_Count_vs_DMB_Active_Header_Count_%d",dduNumber);
   me[meName]->Fill(dmb_active_header,dmb_dav_header_cnt);
    

/// Unpack all found CSC
    vector<CSCEventData> chamberDatas=dduEvent.cscData();

/// Unpack DMB for each particular CSC
    int unpacked_dmb_cnt = 0;
    
    for(vector<CSCEventData>::iterator chamberDataItr = chamberDatas.begin(); 
	chamberDataItr != chamberDatas.end(); ++chamberDataItr) {
		unpacked_dmb_cnt=unpacked_dmb_cnt+1;
		if(printout) {
			cout << "CSCMonitor::MonitorDDU event #" << dec << nEvents 
			<< "> Found DMB " << dec << unpacked_dmb_cnt<< endl;
		}
         
	this->MonitorDMB(chamberDataItr, dduNumber);
       
	if(printout) {
			cout << "CSCMonitor::MonitorDDU event #" << dec << nEvents 
			<< "> Unpacking procedure for DMB " << dec << unpacked_dmb_cnt << " finished" << endl;
        }
     }
    
    if(printout) cout << "CSCMonitor::MonitorDDU event #" << dec << nEvents 
  	 << "> Total number of unpacked DMB = " << dec << unpacked_dmb_cnt << endl;
  
    
     meName = Form("DDU_DMB_unpacked_vs_DAV_%d",dduNumber);
     me[meName]->Fill(dmb_active_header, unpacked_dmb_cnt);
      if(dmb_active_header == unpacked_dmb_cnt) {
        meName = Form("DDU_Unpacking_Match_vs_nEvents_%d",dduNumber);
        me[meName]->Fill(nEvents, 0.0);
      }
      else {
       meName = Form("DDU_Unpacking_Match_vs_nEvents_%d",dduNumber);
       me[meName]->Fill(nEvents, 1.0);
      }
      //meName = Form("DDU_Unpacking_Match_vs_nEvents_%d",dduNumber);
      //me[meName]->SetAxisRange(0, nEvents, "X");
    
      if(printout) cout << "CSCMonitor::MonitorDDU event #" << dec << nEvents 
	<< "> END OF EVENT" << endl;


}

