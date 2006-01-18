/** \file
 * 
 * implementation of CSCMonitor::MonitorCFEB(...) method
 *  $Date: 2005/12/12 09:54:11 $
 *  $Revision: 1.2 $
 *
 * \author Ilaria Segoni
 */

#include "DQM/CSCMonitorModule/interface/CSCMonitor.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBTimeSlice.h"

void CSCMonitor::MonitorCFEB(std::vector<CSCEventData>::iterator data, int ChamberID){

  string meName ;
  map<string, MonitorElement*> me = meChamber[ChamberID];

  int NumberOfUnpackedCFEBs = 0;
  int N_CFEBs=5, N_Samples=16, N_Layers = 6, N_Strips = 16;
  int ADC = 0, OutOffRange, Threshold = 30;
  bool DebugCFEB = false;
  CSCCFEBData * cfebData[5];
  CSCCFEBTimeSlice *  timeSlice[5][16];
  CSCCFEBDataWord * timeSample[5][16][6][16];
  int Pedestal[5][6][16];
  float PedestalError[5][6][16];
  CSCCFEBSCAControllerWord scaControllerWord[5][16][6];
  bool CheckCFEB = true;

  float Clus_Sum_Charge;
  int TrigTime, L1APhase, UnpackedTrigTime, LCTPhase, SCA_BLK, NmbTimeSamples, NmbCell;

  bool CheckThresholdStripInTheLayer[6][80];
  for(int i=1; i<=6; ++i) {
 	 for(int j = 1; j <= 80; ++j) CheckThresholdStripInTheLayer[i][j] = true;
  }

  bool CheckOutOffRangeStripInTheLayer[6][80];
  for(int i=1; i<=6; ++i) {
 	 for(int j = 1; j <= 80; ++j) CheckOutOffRangeStripInTheLayer[i][j] = true;
  }



  float cscdata[N_CFEBs*16][N_Samples][N_Layers];
  int TrigTimeData[N_CFEBs*16][N_Samples][N_Layers];
  int SCABlockData[N_CFEBs*16][N_Samples][N_Layers];
  for(int i=0; i<N_Layers; ++i) {
    for(int j = 0; j < N_CFEBs*16; ++j) {
      for(int k = 0; k < N_Samples; ++k) {
          cscdata[j][k][i] = 0.0;
          TrigTimeData[j][k][i] = 0;
          SCABlockData[j][k][i] = 0;
      }
    }
  }


  for(int nCFEB = 0; nCFEB < N_CFEBs; ++nCFEB) {
    cfebData[nCFEB] = data->cfebData(nCFEB);
		
     if (cfebData[nCFEB] !=0) {
         FEBUnpacked = FEBUnpacked +1;
         NumberOfUnpackedCFEBs = NumberOfUnpackedCFEBs + 1;
         
	 if (CheckCFEB == true){

	    meName = Form("%d_CSC_Rate", ChamberID);
 	    me[meName]->Fill(4);

	    //float CFEBEvent = h[hname]->GetBinContent(5);
	    //meName = Form("hist/h%d_CSC_Efficiency", ChamberID);
	    //if(nEvents > 0) {
	  	  //me[meName]->SetBinContent(4, ((float)CFEBEvent/(float)(DMBEvent)*100.0));
	  	  //me[meName]->SetEntries(nEvents);
	    }
          CheckCFEB = false;
	  
     }
     
     NmbTimeSamples= (cfebData[nCFEB])->nTimeSamples();
     if(printout)cout<< "Monitoring nEvents = " << nEvents << " Chamber ID = "<< ChamberID << " nCFEB =" << nCFEB << endl;
     for(int nSample = 0; nSample < NmbTimeSamples ; ++nSample) {  
         timeSlice[nCFEB][nSample] = (CSCCFEBTimeSlice * )((cfebData[nCFEB])->timeSlice(nSample));  
	 
	 if (timeSlice[nCFEB][nSample] == 0) continue;
         
	 for(int nLayer = 1; nLayer <= N_Layers; ++nLayer) {
	 
	     scaControllerWord[nCFEB][nSample][nLayer] =(timeSlice[nCFEB][nSample])->scaControllerWord(nLayer);
             ///TRIG_TIME indicates which of the eight time samples in the 400ns SCA block (lowest bit is the first 
	     ///sample, highest bit the eighth sample) corresponds to the arrival of the LCT; it should be at some 
	     ///fixed phase relative to the peak of the CSC pulse.
	     TrigTime = (int)(scaControllerWord[nCFEB][nSample][nLayer]).trig_time;
	     
             ///SCA_BLK is the SCA Capacitor block used for this time sample. L1A_PHASE and LCT_PHASE show the phase 
	     ///of the 50ns CFEB digitization clock at the time the trigger was received (1=clock high, 0=clock low).
	     ///SCA_FULL indicates lost SCA data due to SCA full condition. The TS_FLAG bit indicates the number of 
	     ///time samples to digitize per event; high=16 time samples, low=8 time samples.            
	     SCA_BLK  = (int)(scaControllerWord[nCFEB][nSample][nLayer]).sca_blk;
	   
	     for(int nStrip = 0; nStrip < N_Strips; ++nStrip) {
                  SCABlockData[nCFEB*16+nStrip][nSample][nLayer-1] = SCA_BLK;
             }
	     
	     meName = Form("%dCFEB%d_SCA_Block_Occupancy", ChamberID, nCFEB);
             me[meName]->Fill(SCA_BLK);
	 
	 }
                              	
	   
	   
     }
  }





































}
