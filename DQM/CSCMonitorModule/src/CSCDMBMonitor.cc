#include "DQM/CSCMonitorModule/interface/CSCMonitor.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBTrailer.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"



void CSCMonitor::MonitorDMB(std::vector<CSCEventData>::iterator data){
		
		
		
  cout << "Beginning CSCMonitor::MonitorDMB> #" <<endl;


/*
if (&data==0) {
	if(printout) {
		cout << "CSCMonitor::MonitorDMB> #" << dec << nEvents 
		     << " Zero pointer. DMB data are not available for Monitoring" << endl; 
	}
	return;
}
else {
	if(printout) {
		cout << "CSCMonitor::MonitorDMB> #" << dec << nEvents 
		     << " Nonzero pointer. DMB data are available for Monitoring" << endl;
	}
}
*/




 if(printout) cout << "CSCMonitor::MonitorDMB> #" << dec << nEvents << "> Monitoring DMB Header and Trailer ... ";
 	 
  CSCDMBHeader dmbHeader  = data->dmbHeader();
  CSCDMBTrailer dmbTrailer = data->dmbTrailer();


  int crateID	  = dmbHeader.crateID();
  int dmbID 	  = dmbHeader.dmbID();
  int ChamberID	  = (((crateID) << 4) + dmbID) & 0xFFF;
  if(printout) cout << "CSCMonitor::MonitorDMB> #" << dec << nEvents 
			     << "> Chamber ID = "<< ChamberID << " Crate ID = "<< crateID << " DMB ID = " << dmbID << endl;
  
  //string meName;
  if(!cmbBooked[ChamberID]){
     meChamber[ChamberID] = book_chamber(ChamberID);
     cmbBooked[ChamberID]=true;  
  }

  map<string, MonitorElement*> me = meChamber[ChamberID];


  string meName ;
  meName = Form("%d_CSC_Rate", ChamberID);
  me[meName]->Fill(1);

// insert CSC_Efficiency Plot when SetBinContent/SetEntries are available


 int dmbHeaderL1A = dmbHeader.l1a();
 int dmb_ddu_l1a_diff	 = 0;
//		Calculation difference between L1A numbers from DDU and DMB

  meName = Form("%dDMB_L1A_Distrib", ChamberID);
  me[meName]->Fill(dmbHeaderL1A);

  int dmb_ddu_L1a_diff=(int)(dmbHeaderL1A-(int)(L1ANumber&0xFF));
  if(dmb_ddu_L1a_diff<-128){dmb_ddu_L1a_diff=dmb_ddu_L1a_diff+256;}
  if(dmb_ddu_L1a_diff>128){dmb_ddu_L1a_diff=dmb_ddu_L1a_diff-256;}
  
  
  meName = Form("%dDMB_DDU_L1A_diff", ChamberID);
  me[meName]->Fill(dmb_ddu_l1a_diff);

  if(printout)  cout << "+++debug> DMB(ID=" << ChamberID  << ") L1A = " 
   << dmbHeaderL1A << " : DMB L1A - DDU L1A = " << dmb_ddu_l1a_diff << endl;

}
