/** \file
 * 
 * implementation of CSCMonitor::MonitorCFEB(...) method
 *  $Date: 2006/02/07 14:30:27 $
 *  $Revision: 1.2 $
 *
 * \author Ilaria Segoni
 */

#include "DQM/CSCMonitorModule/interface/CSCMonitor.h"
#include "DQM/CSCMonitorModule/interface/CSCStripClusterFinder.h"
#include "DQM/CSCMonitorModule/interface/CSCStripClusterFitData.h"
#include "DQM/CSCMonitorModule/interface/CSCStripCluster.h"

#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBTimeSlice.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"



void CSCMonitor::MonitorCFEB(std::vector<CSCEventData>::iterator data, int ChamberID){

  string meName ;
  map<string, MonitorElement*> me = meChamber[ChamberID];

  CSCDMBHeader dmbHeader  = data->dmbHeader();


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

	    		//float CFEBEvent = me[meName]->GetBinContent(5);
	    		//meName = Form("%d_CSC_Efficiency", ChamberID);
	    		if(nEvents > 0) {
	  			//  me[meName]->SetBinContent(4, ((float)CFEBEvent/(float)(DMBEvent)*100.0));
	  			//  me[meName]->SetEntries(nEvents);
	   		 }
          		CheckCFEB = false;
	  
    		 }
     
     
    		NmbTimeSamples= (cfebData[nCFEB])->nTimeSamples();
     		edm::LogInfo ("CSC DQM") <<"Monitoring nEvents = " << nEvents << " Chamber ID = "<< ChamberID << " nCFEB =" << nCFEB;
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
	 
//					if(debug) cout << "+++debug> nCFEB " << nCFEB << " nSample " << nSample << " nLayer " << nLayer << " TrigTime " << TrigTime << endl;
 					if(nSample == 0 && nLayer == 1) {
 						TrigTime = (int)(scaControllerWord[nCFEB][nSample][nLayer]).trig_time;
 						int k=1;
 						while (((TrigTime >> (k-1)) & 0x1) != 1 && k<=8) {
 							k = k +1;
 						}
 						L1APhase = (int)(((scaControllerWord[nCFEB][nSample][nLayer]).l1a_phase)&0x1);
 						UnpackedTrigTime = ((k<<1)&0xE)+L1APhase;

						meName = Form("%dCFEB%d_L1A_Sync_Time", ChamberID, nCFEB);
 						me[meName]->Fill((int)UnpackedTrigTime);
						LCTPhase = (int)(((scaControllerWord[nCFEB][nSample][nLayer]).lct_phase)&0x1);

						meName = Form("%dCFEB%d_LCT_PHASE_vs_L1A_PHASE", ChamberID, nCFEB);
 						me[meName]->Fill(LCTPhase, L1APhase);

						//edm::LogInfo ("CSC DQM") <<
						 //   "+++debug> L1APhase " << L1APhase
						//	<< " UnpackedTrigTime " << UnpackedTrigTime ;

						meName = Form("%dCFEB%d_L1A_Sync_Time_vs_DMB", ChamberID, nCFEB);
 						me[meName]->Fill((int)(dmbHeader.dmbCfebSync()), (int)UnpackedTrigTime);

						meName = Form("%dCFEB%d_L1A_Sync_Time_DMB_diff", ChamberID, nCFEB);
						int cfeb_dmb_L1A_sync_time = (int)(dmbHeader.dmbCfebSync()) - (int)UnpackedTrigTime;
						if(cfeb_dmb_L1A_sync_time < -8) me[meName]->Fill(cfeb_dmb_L1A_sync_time+16);
						else {
							if(cfeb_dmb_L1A_sync_time > 8) 	me[meName]->Fill(cfeb_dmb_L1A_sync_time-16);
							else 				me[meName]->Fill(cfeb_dmb_L1A_sync_time);
						}
						//me[meName]->SetAxisRange(0.1, 1.1*(1.0+me[meName]->GetBinContent(me[meName]->GetMaximumBin())), "Y");

 					        }
					         
						 
					
					        if(DebugCFEB) {
							edm::LogInfo ("CSC DQM: ") <<" nLayer = " << nLayer ;
						}	
 					        for(int nStrip = 1; nStrip <= N_Strips; ++nStrip) {
						timeSample[nCFEB][nSample][nLayer][nStrip]=
						  (data->cfebData(nCFEB)->timeSlice(nSample))->timeSample(nLayer,nStrip);
 						ADC = (int) ((timeSample[nCFEB][nSample][nLayer][nStrip]->adcCounts)&0xFFF);
						//if(DebugCFEB) cout << " nStrip="<< dec << nStrip << " ADC=" << hex << ADC << endl;
 						OutOffRange = (int) ((timeSample[nCFEB][nSample][nLayer][nStrip]->adcOverflow)&0x1);
 						if(nSample == 0) { // nSample == 0
							Pedestal[nCFEB][nLayer][nStrip] = ADC;
							//if(DebugCFEB) cout << " nStrip="<< dec << nStrip << " Pedestal=" << hex << Pedestal[nCFEB][nLayer][nStrip] << endl;
 							//meName = Form("%dCFEB_Pedestal(withEMV)_Sample_01_Ly%d", ChamberID, nLayer);
							//me[meName]->Fill((int)(nCFEB*16+nStrip), Pedestal[nCFEB][nLayer][nStrip]);
							//meName = Form("%dCFEB_Pedestal(withRMS)_Sample_01_Ly%d", ChamberID, nLayer);
							//me[meName]->Fill((int)(nCFEB*16+nStrip), Pedestal[nCFEB][nLayer][nStrip]);
							//PedestalError[nCFEB][nLayer][nStrip] = me[meName]->GetBinError(nCFEB*16+nStrip);
							//meName = Form("%dCFEB_PedestalRMS_Sample_01_Ly%d",ChamberID,nLayer);
							//me[meName]->SetBinContent(nCFEB*16+nStrip,PedestalError[nCFEB][nLayer][nStrip]);
							//me[meName]->SetBinError(nCFEB*16+nStrip,0.00000000001);
					        }
 						if(OutOffRange == 1 && CheckOutOffRangeStripInTheLayer[nLayer][nCFEB*16+nStrip] == true) {
							meName = Form("%dCFEB_Out_Off_Range_Strips_Ly%d", ChamberID, nLayer);
 							me[meName]->Fill((int)(nCFEB*16+nStrip));
 							CheckOutOffRangeStripInTheLayer[nLayer][nCFEB*16+nStrip] = false;
 						}
 						if(ADC - Pedestal[nCFEB][nLayer][nStrip] > Threshold && OutOffRange != 1) {
							meName = Form("%dCFEB_Active_Samples_vs_Strip_Ly%d", ChamberID, nLayer);
 							me[meName]->Fill((int)(nCFEB*16+nStrip), nSample);
							//meName = Form("%dCFEB_Active_Samples_vs_Strip_Ly%d_Profile", ChamberID, nLayer);
 							//me[meName]->Fill((int)(nCFEB*16+nStrip), nSample);
 							if(CheckThresholdStripInTheLayer[nLayer][nCFEB*16+nStrip] == true) {
								meName = Form("%dCFEB_ActiveStrips_Ly%d", ChamberID, nLayer);
 								me[meName]->Fill((int)(nCFEB*16+nStrip));
 								CheckThresholdStripInTheLayer[nLayer][nCFEB*16+nStrip] = false;
 							}
 						}

						if(ADC - Pedestal[nCFEB][nLayer][nStrip] > Threshold) {
							if(DebugCFEB) {
								edm::LogInfo ("CSC DQM: ") <<"Layer="<<nLayer<<" Strip="<<nCFEB*16+nStrip<<" Time="<<nSample;
								edm::LogInfo ("CSC DQM: ") <<" ADC-PEDEST = "<<ADC - Pedestal[nCFEB][nLayer][nStrip];
							}
							cscdata[nCFEB*16+nStrip-1][nSample][nLayer-1] = ADC - Pedestal[nCFEB][nLayer][nStrip];
						}

					}					
					
					}
	 }
                              	
	   
	   
     }
  }



        /// CLUSTERS
	float Cathodes[N_CFEBs*N_Strips*N_Samples*N_Layers];
	for(int i=0; i<N_Layers; ++i) {
		for(int j=0; j<N_CFEBs*N_Strips; ++j) {
			for(int k=0; k<N_Samples; ++k) {
				Cathodes[i*N_CFEBs*N_Strips*N_Samples + N_CFEBs*N_Strips*k + j] = cscdata[j][k][i];
			}
		}
	}

	vector<CSCStripCluster> Clus;
	Clus.clear();

	for(int nLayer=1; nLayer<=N_Layers; ++nLayer) {
		CSCStripClusterFinder *ClusterFinder = new CSCStripClusterFinder(N_Layers, N_Samples, N_CFEBs, N_Strips);
		ClusterFinder->DoAction(nLayer-1, Cathodes);
		Clus = ClusterFinder->getClusters();

		for(int j=0; j<N_CFEBs*N_Strips; j++){
			int SCAbase=SCABlockData[j][0][nLayer-1];
			int SCAcount=0;
			for(int k=0; k<NmbTimeSamples; k++){
				int SCA=SCABlockData[j][k][nLayer-1];
				if(SCA==SCAbase) SCAcount++;
			}
			int TmpTrigTime=NmbTimeSamples+1-SCAcount;
			for(int k=0;k<SCAcount;k++){
				TrigTimeData[j][k][nLayer-1]=TmpTrigTime;
			}
		}

		if(DebugCFEB)
		  edm::LogInfo ("CSC DQM: ") <<"***  CATHODE PART  DEBUG: Layer="<<nLayer
		      <<"  Number of Clusters="<<Clus.size()<<"      ***";
//		Number of Clusters Histograms
		meName = Form("%dCFEB_Number_of_Clusters_Ly_%d", ChamberID, nLayer);
		if(Clus.size() != 0) me[meName]->Fill(Clus.size());

		for(unsigned int u=0;u<Clus.size();u++){
			if(DebugCFEB)
			  edm::LogInfo ("CSC DQM: ") <<"Chamber: "<< ChamberID  << " Cluster: "
			       << u+1<< " Number of local Maximums " <<  Clus[u].localMax.size();
			for(unsigned int t=0;t<Clus[u].localMax.size();t++){
				int iS=Clus[u].localMax[t].Strip;
				int jT=Clus[u].localMax[t].Time;
//				Peak SCA Cell Histograms
				meName = Form("%dCFEB_SCA_Cell_Peak_Ly_%d", ChamberID, nLayer);
				int SCA = SCABlockData[iS][jT][nLayer-1];
				int TmpTrigTime = TrigTimeData[iS][jT][nLayer-1];
//				cout<<"TmpTrigTime(max)="<<TmpTrigTime<<" Layer="<<nLayer<<endl;
				if(TmpTrigTime>=0) {
					NmbCell = (SCA-1)*NmbTimeSamples+TmpTrigTime+jT;
					if(TmpTrigTime==0) NmbCell++;
					me[meName]->Fill(iS+1,NmbCell);
				}

				if(DebugCFEB) {
					for(unsigned int k=0;k<Clus[u].ClusterPulseMapHeight.size();k++){
						if(Clus[u].ClusterPulseMapHeight[k].channel_==iS) {
							edm::LogInfo ("CSC DQM: ") << "Local Max: " << t+1 << " Strip: " << iS+1 << " Time: " << jT+1;
							edm::LogInfo ("CSC DQM: ") <<" Height: "
							     << Clus[u].ClusterPulseMapHeight[k].height_[jT];
						}
					}
				}
			}
			Clus_Sum_Charge = 0.0;
			for(unsigned int k=0;k<Clus[u].ClusterPulseMapHeight.size();k++) {
				if(DebugCFEB) {
					edm::LogInfo ("CSC DQM: ") <<"Strip: " << Clus[u].ClusterPulseMapHeight[k].channel_+1;
				}
//				Strip Occupancy Histograms
//				meName = Form("Chamber_%d_Strip_Occupancy_Ly_%d", ChamberID, nLayer);
//				me[meName]->Fill(Clus[u].ClusterPulseMapHeight[k].channel_+1);

				if(DebugCFEB) {
					for(unsigned int n=0;n<16;n++){
						edm::LogInfo ("CSC DQM: ") <<" " << Clus[u].ClusterPulseMapHeight[k].height_[n];
					}
					
				}

				for(unsigned int n=Clus[u].LFTBNDTime; n < Clus[u].IRTBNDTime; n++){
					Clus_Sum_Charge = Clus_Sum_Charge + Clus[u].ClusterPulseMapHeight[k].height_[n];
//					SCA Cell Occupancy Histograms
					meName = Form("%dCFEB_SCA_Cell_Occupancy_Ly_%d", ChamberID, nLayer);
					int SCA = SCABlockData[Clus[u].ClusterPulseMapHeight[k].channel_][n][nLayer-1];
					int TmpTrigTime = TrigTimeData[Clus[u].ClusterPulseMapHeight[k].channel_][n][nLayer-1];
					if(TmpTrigTime>=0) {
//						cout<<"TmpTrigTime(cluster)="<<TmpTrigTime<<" Layer="<<nLayer<<endl;
						NmbCell = (SCA-1)*NmbTimeSamples+TmpTrigTime+n;
						if(TmpTrigTime==0) NmbCell++;
						me[meName]->Fill(Clus[u].ClusterPulseMapHeight[k].channel_+1, NmbCell);
					}
				}
			}
//			Clusters Charge Histograms
			meName = Form("%dCFEB_Clusters_Charge_Ly_%d", ChamberID, nLayer);
			me[meName]->Fill(Clus_Sum_Charge);

//			Width of Clusters Histograms
			meName = Form("%dCFEB_Width_of_Clusters_Ly_%d", ChamberID, nLayer);
			me[meName]->Fill(Clus[u].IRTBNDStrip - Clus[u].LFTBNDStrip+1);
		}
	Clus.clear();
	delete ClusterFinder;
        }




/// Fill Histogram with number of unpacked datas
	int tmb_dav = dmbHeader.nclct();
	int alct_dav = dmbHeader.nalct();
	int cfeb_dav2 = 0;
	for (int ii=0; ii<5; ++ii)  cfeb_dav2 = cfeb_dav2 + (int)((dmbHeader.cfebAvailable()>>ii) & 0x1);
	int FEBdav = cfeb_dav2+alct_dav+tmb_dav;

	meName = Form("%dDMB_FEB_DAV", ChamberID);
	me[meName]->Fill(FEBdav);

	meName = Form("%dDMB_FEB_unpacked_vs_DAV", ChamberID);
	me[meName]->Fill(FEBdav,FEBUnpacked);





}// end of method definition

