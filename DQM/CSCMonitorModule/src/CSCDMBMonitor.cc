/** \file
 * 
 * implementation of CSCMonitor::MonitorDMB(...) method
 *  $Date: 2006/01/18 11:22:06 $
 *  $Revision: 1.3 $
 *
 * \author Ilaria Segoni
 */
 
#include "DQM/CSCMonitorModule/interface/CSCMonitor.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBTrailer.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"



void CSCMonitor::MonitorDMB(std::vector<CSCEventData>::iterator data, int dduNumber){

 if(printout) cout << "Entering CSCMonitor::MonitorDMB>, for Event Number: "<< nEvents<<
         " and DDU number: "<< dduNumber <<endl;

 if (&data==0) {
        if(printout) cout << " NULL CSCEventData pointer: DMB data are not available for Monitoring" << endl; 
	return;
 }
 else {
        if(printout) cout << "DMB data are available for Monitoring" << endl;
 }
 
 string meName ;
 map<string, MonitorElement*> meDDUInfo = meDDU[dduNumber];

/// DMB HEADER INFORMATION
 
 if(printout) cout << "Monitoring DMB Header";
 	 
  CSCDMBHeader dmbHeader  = data->dmbHeader();
  CSCDMBTrailer dmbTrailer = data->dmbTrailer();

  int crateID	  = dmbHeader.crateID();
  int dmbID 	  = dmbHeader.dmbID();
  int ChamberID	  = (((crateID) << 4) + dmbID) & 0xFFF;
  if(printout) cout << "Chamber ID = "<< ChamberID << " Crate ID = "<< crateID << " DMB ID = " << dmbID << endl;


  meName = Form("CSC_Unpacked_%d",dduNumber);
  meDDUInfo[meName]->Fill(crateID,dmbID);
  
  //string meName;
  if(!cmbBooked[ChamberID]){
     meChamber[ChamberID] = book_chamber(ChamberID);
     cmbBooked[ChamberID]=true;  
  }

  map<string, MonitorElement*> me = meChamber[ChamberID];

  meName = Form("%d_CSC_Rate", ChamberID);
  me[meName]->Fill(1);

// insert CSC_Efficiency Plot when SetBinContent/SetEntries are available


 int dmbHeaderL1A = dmbHeader.l1a();
 int dmb_ddu_l1a_diff	 = 0;

  meName = Form("%dDMB_L1A_Distrib", ChamberID);
  me[meName]->Fill(dmbHeaderL1A);

 meName = Form("%dDMB_BXN_Distrib", ChamberID);
  me[meName]->Fill((int)(dmbHeader.bxn()));

/// DDU and DMB Timing: L1A
 
  meName = Form("%dDMB_L1A_vs_DDU_L1A", ChamberID);
  me[meName]->Fill((int)(L1ANumber[dduNumber]&0xFF), (int)dmbHeaderL1A);

  int dmb_ddu_L1a_diff=(int)(dmbHeaderL1A-(int)(L1ANumber[dduNumber]&0xFF));
  if(printout)  cout << "DMB(ID=" << ChamberID  << ") L1A = " 
   << dmbHeaderL1A << " : DMB L1A - DDU L1A = " << dmb_ddu_l1a_diff << endl;

  if(dmb_ddu_L1a_diff<-128){dmb_ddu_L1a_diff=dmb_ddu_L1a_diff+256;}
  if(dmb_ddu_L1a_diff>128){dmb_ddu_L1a_diff=dmb_ddu_L1a_diff-256;}
  
  
  meName = Form("%dDMB_DDU_L1A_diff", ChamberID);
  me[meName]->Fill(dmb_ddu_l1a_diff);


// DDU and DMB Timing: BX
  int dmbHeaderBXN=dmbHeader.bxn();
  
  meName = Form("%dDMB_BXN_vs_DDU_BXN", ChamberID);
  me[meName]->Fill((int)(dduBX[dduNumber]), (int)(dmbHeaderBXN));

  int dmb_ddu_bxn_diff=(int)(dmbHeaderBXN)-(int)(dduBX[dduNumber]);
  if(printout)  cout << "DMB(ID=" << ChamberID  << ") BXN = " 
   << dmbHeaderBXN << " : DMB BXN - DDU BXN = " << dmb_ddu_bxn_diff << endl;
 
  if(dmb_ddu_bxn_diff<-64){dmb_ddu_bxn_diff=dmb_ddu_bxn_diff+128;}
  if(dmb_ddu_bxn_diff>64){dmb_ddu_bxn_diff=dmb_ddu_bxn_diff-128;}
  
  
  meName = Form("%dDMB_DDU_BXN_diff", ChamberID);
  me[meName]->Fill(dmb_ddu_bxn_diff);





 /// DMB TRAILER INFORMATION

 if(printout) cout << "Monitoring DMB Trailer";

 meName = Form("%dDMB_L1_Pipe", ChamberID);
 me[meName]->Fill(dmbTrailer.dmb_l1pipe);

 
 
 meName = Form("%dDMB_FEB_Timeouts", ChamberID);
 if ((dmbTrailer.tmb_timeout==0) && (dmbTrailer.alct_timeout==0) 
     && (dmbTrailer.cfeb_starttimeout==0) && (dmbTrailer.cfeb_endtimeout==0)) {
 	 me[meName]->Fill(0.0);
 }
 
 if (dmbTrailer.alct_timeout) me[meName]->Fill(1);
 if (dmbTrailer.tmb_timeout) me[meName]->Fill(2);
 if (dmbTrailer.alct_endtimeout) me[meName]->Fill(8); 
 if (dmbTrailer.tmb_endtimeout) me[meName]->Fill(9);  
 for (int ii=0; ii<5; ii++) {
 	 if ((dmbTrailer.cfeb_starttimeout>>ii) & 0x1) me[meName]->Fill(ii+3);
 	 if ((dmbTrailer.cfeb_endtimeout>>ii) & 0x1) me[meName]->Fill(ii+10);
 }


 meName = Form("%dDMB_FIFO_stats", ChamberID);
 if (dmbTrailer.tmb_empty == 1) me[meName]->Fill(1.0, 0.0); //KK
 if (dmbTrailer.tmb_half == 0) me[meName]->Fill(1.0, 1.0);
 if (dmbTrailer.tmb_full == 1) me[meName]->Fill(1.0, 2.0); //KK
 if (dmbTrailer.alct_empty == 1) me[meName]->Fill(0.0, 0.0);
 if (dmbTrailer.alct_half == 0) me[meName]->Fill(0.0, 1.0);
 if (dmbTrailer.alct_full == 1) me[meName]->Fill(0.0, 2.0); //KK 0->1
 for (int ii=0; ii<5; ii++) {
 	 if ((int)((dmbTrailer.cfeb_empty>>ii)&0x1) == 1) me[meName]->Fill(ii+2,0.0);
 	 if ((int)((dmbTrailer.cfeb_half>>ii)&0x1) == 0) me[meName]->Fill(ii+2,1);
 	 if ((int)((dmbTrailer.cfeb_full>>ii)&0x1) == 1) me[meName]->Fill(ii+2,2);
 }
 //me[meName]->SetEntries((int)DMBEvent);


  /// CFEB information from DMB Header

  if(printout)  cout << "Monitoring CFEB information from DMB Header "<< endl;
   
  int cfeb_dav = (int)dmbHeader.cfebAvailable();
  meName = Form("%dDMB_CFEB_DAV", ChamberID);
  me[meName]->Fill(cfeb_dav);
  
  int cfeb_dav_num =0;
  for (int i=0; i<5; i++) cfeb_dav_num = cfeb_dav_num + (int)((cfeb_dav>>i) & 0x1);
  
  if(printout)  cout << "Number of CFEB with Data Available: "<< cfeb_dav_num<<endl;
  meName = Form("%dDMB_CFEB_DAV_multiplicity", ChamberID);
  me[meName]->Fill(cfeb_dav_num);
  
  int cfeb_movlp	 = (int)dmbHeader.cfebMovlp();
  meName = Form("%dDMB_CFEB_MOVLP", ChamberID);
  me[meName]->Fill(cfeb_movlp);
  
  int dmb_cfeb_sync = (int)dmbHeader.dmbCfebSync();
  meName = Form("%dDMB_CFEB_Sync", ChamberID);
  me[meName]->Fill(dmb_cfeb_sync);

  //meName = Form("%dDMB_CFEB_Active", ChamberID);
  //me[meName]->Fill(dmbHeader.cfebActive());

  //meName = Form("%dDMB_CFEB_Active_vs_DAV", ChamberID);
  //me[meName]->Fill(dmbHeader.cfebAvailable(),dmbHeader.cfebActive());





  /// Fill ME's with Cathode Front End Boards Data
  this->MonitorCFEB(data,ChamberID);


}















