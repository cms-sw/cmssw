#include <memory>
#include <iostream>

#include "DQM/CSCMonitorModule/interface/CSCMonitor.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"

using namespace std;


/// MonitoringElements FOR SINGLE CHAMBER


map<string, MonitorElement*> CSCMonitor::book_chamber(int ChamberID) {

	int id = ChamberID;
	int CrateID = (int)((id>>4) & 0xFF);
	int DMBID = (int)(id & 0xF);

	map<int, map<string,MonitorElement*> >::iterator me_itr = meCollection.find(ChamberID);
      	if (me_itr == meCollection.end() || (meCollection.size()==0)) {
	    if(printout) {
	      cout << "CSCMonitor::book_chamber> #"
		   << "> ch" << CrateID << ":" << DMBID << ">";
	      cout << " Creating of list of Histos for the chamber ..." << endl;
	   
	    }
	    
	} else { 
	
         if(printout) cout<<"returning the existing collector "<<endl;
	  return meCollection[ChamberID];
	
	}


	string meName;
	map<string, MonitorElement*> me;

        string dir = Form("Data/Channel_%d",id);


//CSC
	if(printout) 	cout << "CSCMonitor::book_chamber> New CSC Canvases are booking ..." << endl;

//KK additional information for each particular chamber

		meName = Form("%dBinCheck_ErrorStat_Table",id);
		me[meName] = dbe->book2D(meName.c_str(), "DDU Data Format Errors Table", 1, 0, 1, 19, 0, 19);

		meName = Form("%dBinCheck_ErrorStat_Frecuency",id);
		me[meName] = dbe->book1D(meName.c_str(), "DDU Data Format Errors Frequency", 19, 0, 19);


		meName = Form("%dBinCheck_WarningStat_Table",id);
		me[meName] = dbe->book2D(meName.c_str(), "DDU Data Format Warnings Table", 1, 0, 1, 1, 0, 1);

		meName = Form("%dBinCheck_WarningStat_Frequency",id);
		me[meName] = dbe->book2D(meName.c_str(), "DDU Data Format Warnings Frequency", 1, 0, 1, 1/*bin_checker.nWARNINGS*/, 0, 1/*bin_checker.nWARNINGS*/);

//KK end


		meName = Form("%d_CSC_Efficiency", id);
		me[meName] = dbe->book1D(meName.c_str(), "", 5, 0, 5);

		meName = Form("%d_CSC_Rate",id);
		me[meName] = dbe->book1D(meName.c_str(), "", 5, 0, 5);


//DMBs
//	if(debug_printout) 	cout << "D**EmuBookChamber> New DMB Canvases are booking ..." << endl;

        dbe->setCurrentFolder("DMB");


		meName = Form("%dDMB_FEB_DAV",id);
		me[meName] = dbe->book1D(meName.c_str(), "Boards DAV Statistics", 8,  0 , 8);

		meName = Form("%dDMB_FEB_unpacked_vs_DAV", id);
		me[meName] = dbe->book2D(meName.c_str(), "DMB Unpacked FEBs vs FEBs DAV", 8,  0 , 8, 8, 0, 8);

		meName = Form("%dDMB_CFEB_Active_vs_DAV", id);
		me[meName] = dbe->book2D(meName.c_str(), "CFEB_Active vs CFEB_DAV combinations", 32, 0, 32, 32,  0 , 32);


		meName = Form("%dDMB_CFEB_Active", id);
		me[meName] = dbe->book1D(meName.c_str(), "Active CFEBs combinations as reported by TMB", 32,  0 , 32);

//KK

//KK end


		meName = Form("%dDMB_CFEB_DAV", id);
		me[meName] = dbe->book1D(meName.c_str(), "CFEBs combinations reporting DAV", 32,  0 , 32);

//KK

//KK end


		meName = Form("%dDMB_CFEB_DAV_multiplicity", ChamberID);
		me[meName] = dbe->book1D(meName.c_str(), "Number of CFEBs reporting DAV per event", 6,  0 , 6);

		meName = Form("%dDMB_CFEB_MOVLP", id);
		me[meName] = dbe->book1D(meName.c_str(), "", 32, 0, 32);

		meName = Form("%dDMB_CFEB_Sync", ChamberID);
		me[meName] = dbe->book1D(meName.c_str(), "Counter of BXNs since last SyncReset to L1A", 16, 0, 16);

		meName = Form("%dDMB_FEB_Timeouts", ChamberID);
		me[meName] = dbe->book1D(meName.c_str(), "DMB FEB Timeouts", 15,  0 , 15);

		meName = Form("%dDMB_L1_Pipe", ChamberID);
		me[meName] = dbe->book1D(meName.c_str(), "Number of L1A requests piped in DMB for readout", 288, 0, 288);

		meName = Form("%dDMB_FIFO_stats", ChamberID);
		me[meName] = dbe->book2D(meName.c_str(), "FEDs FIFO Status", 7,  0 , 7, 3, 0, 3);

//ALCTs
	//if(debug_printout) 	cout << "D**EmuBookChamber> New ALCT Canvases are booking ..." << endl;
        dbe->setCurrentFolder("ALCT");


		meName = Form("%dALCT_Word_Count", ChamberID);
                me[meName] = dbe->book1D(meName.c_str(), "", 1024, 0, 1024);

		meName = Form("%dALCT_Number_Of_Layers_With_Hits", ChamberID);
                me[meName] = dbe->book1D(meName.c_str(), "Number of Layers with Hits", 7, 0, 7);

		meName = Form("%dALCT_Number_Of_WireGroups_With_Hits", ChamberID);
                me[meName] = dbe->book1D(meName.c_str(), "Total Number of Wire Groups with Hits", 672, 1, 673);

	for (int nLayer=1; nLayer<=6; nLayer++) {
		meName = Form("%dALCT_Ly%d_Rate", ChamberID, nLayer);
		me[meName] = dbe->book1D(meName.c_str(), Form("Layer = %d", nLayer), 112, 0, 112);


		meName = Form("%dALCT_Ly%d_Efficiency", ChamberID, nLayer);
		me[meName] = dbe->book1D(meName.c_str(), Form("Layer = %d", nLayer), 112, 0, 112);

	}

	for (int nLayer=1; nLayer<=6; nLayer++) {
		meName = Form("%dALCTTime_Ly%d", ChamberID, nLayer);
		me[meName] = dbe->book2D(meName.c_str(), Form("Layer = %d", nLayer), 112, 0, 112, 32, 0, 32);

	}

//	for (int nLayer=1; nLayer<=6; nLayer++) {
//		meName = Form("%dALCTTime_Ly%d_Profile", ChamberID, nLayer);
//		me[meName] = new TProfile(meName.c_str(), Form("Layer = %d", nLayer), 112, 0, 112, 0, 32, "g");
//	}

		meName = Form("%dALCT_Number_Rate", ChamberID);
		me[meName] = dbe->book1D(meName.c_str(), "Number of ALCTs", 3, 0, 3 );

		meName = Form("%dALCT_Number_Efficiency", ChamberID);
		me[meName] = dbe->book1D(meName.c_str(), "Number of ALCTs", 3, 0, 3 );



//TMB
	//if(debug_printout) 	cout << "D**EmuBookChamber> New TMB Canvases are booking ..." << endl;
        dbe->setCurrentFolder("TMB");


		meName = Form("%dTMB_Word_Count", ChamberID);
		me[meName] = dbe->book1D(meName.c_str(), "", 2048, 0, 2048);

		meName = Form("%dALCT_Match_Time", ChamberID);
                me[meName] = dbe->book1D(meName.c_str(), "Location of ALCT in CLCT match window", 16, 0, 16);

		meName = Form("%dLCT_Match_Status", ChamberID);
                me[meName] = dbe->book2D(meName.c_str(), "ALCT-CLCT match status", 1, 0, 1, 3, 0, 3);

		meName = Form("%dLCT0_Match_BXN_Difference", ChamberID);
                me[meName] = dbe->book1D(meName.c_str(), "ALCT-CLCT BXN Difference for Matched LCT0", 4, 0, 4);

		meName = Form("%dLCT1_Match_BXN_Difference", ChamberID);
                me[meName] = dbe->book1D(meName.c_str(), "ALCT-CLCT BXN Difference for Matched LCT1", 4, 0, 4);

//TMB - CLCTs
	//if(debug_printout) 	cout << "D**EmuBookChamber> New TMB-CLCT Canvases are booking ..." << endl;
        //dbe->setCurrentFolder("TMB");
 
		meName = Form("%dCLCT_Number_Of_Layers_With_Hits", ChamberID);
                me[meName] = dbe->book1D(meName.c_str(), "Number of Layers with Hits", 7, 0, 7);

		meName = Form("%dCLCT_Number_Of_HalfStrips_With_Hits", ChamberID);
                me[meName] = dbe->book1D(meName.c_str(), "Total Number of HalfStrips with Hits", 672, 1, 673);
	for (int nLayer=1; nLayer<=6; nLayer++) {
		meName = Form("%dCLCT_Ly%d_Rate", ChamberID, nLayer);
		me[meName] = dbe->book1D(meName.c_str(), Form("Layer = %d",nLayer), 160, 0, 160);

		meName = Form("%dCLCT_Ly%d_Efficiency",ChamberID, nLayer);
		me[meName] = dbe->book1D(meName.c_str(), Form("Layer = %d",nLayer), 160, 0, 160);
	}

	for (int nLayer=1; nLayer<=6; nLayer++) {
		meName = Form("%dCLCTTime_Ly%d", ChamberID, nLayer);
		me[meName] = dbe->book2D(meName.c_str(), Form("Layer = %d",nLayer), 160, 0, 160, 32, 0, 32);
	}

        //for (int nLayer=1; nLayer<=6; nLayer++) {
	//	meName = Form("%dCLCTTime_Ly%d_Profile", ChamberID, nLayer);
	//	me[meName] = new TProfile(meName.c_str(), Form("Layer = %d",nLayer), 160, 0, 160, 0, 32, "g");
	//}

		meName = Form("%dCLCT_Number_Rate", ChamberID);
		me[meName] = dbe->book1D(meName.c_str(), "Number of CLCTs", 3, 0, 3 );

		meName = Form("%dCLCT_Number", ChamberID);
		me[meName] = dbe->book1D(meName.c_str(), "Number of CLCTs", 3, 0, 3 );

		meName = Form("%dCLCT1_vs_CLCT0_Key_Strip", ChamberID);
		me[meName] = dbe->book2D(meName.c_str(), "CLCT1 & CLCT0 Correlation", 160, 0, 160, 160, 0, 160);

		meName = Form("%dCLCT0_Clssification", ChamberID);
		me[meName] = dbe->book1D(meName.c_str(), "Single CLCT Classification (CLCT0)", 2, 0, 2);

		meName = Form("%dCLCT0_CLCT1_Clssification", ChamberID);
		me[meName] = dbe->book1D(meName.c_str(), "Double CLCT Classification (CLCT0+CLCT1)", 4, 0, 4);
	for(int lct=0; lct<=1; lct++) {

			meName = Form("%dCLCT%d_KeyHalfStrip", ChamberID, lct);
			me[meName] = dbe->book1D(meName.c_str(), Form("CLCT%d Key Half Strip",lct), 160, 0, 160);

			meName = Form("%dCLCT%d_Half_Strip_Quality", ChamberID, lct);
			me[meName] = dbe->book2D(meName.c_str(), Form("CLCT%d Quality vs Key Half Strip",lct), 160, 0, 160, 7, 0, 7);

			meName = Form("%dCLCT%d_Half_Strip_Pattern", ChamberID, lct);
			me[meName] = dbe->book2D(meName.c_str(), Form("CLCT%d Patterns vs Key Half Strip",lct), 160, 0, 160, 8, 0, 8);

			//meName = Form("%dCLCT%d_Half_Strip_Quality_Profile", ChamberID, lct);
			//me[meName] = new TProfile(meName.c_str(), Form("CLCT%d Average Quality",lct), 160, 0.0, 160.0, 0.0, 7.0, "g");



			meName = Form("%dCLCT%d_KeyDiStrip", ChamberID, lct);
			me[meName] = dbe->book1D(meName.c_str(), Form("CLCT%d Key DiStrip",lct), 40, 0, 160);


			meName = Form("%dCLCT%d_DiStrip_Quality", ChamberID, lct);
			me[meName] = dbe->book2D(meName.c_str(), Form("CLCT%d Quality vs Key DiStrip",lct), 40, 0, 160, 7, 0, 7);

			meName = Form("%dCLCT%d_DiStrip_Pattern", ChamberID, lct);
			me[meName] = dbe->book2D(meName.c_str(), Form("CLCT%d Patterns vs Key DiStrip",lct), 40, 0, 160, 8, 0, 8);


			//meName = Form("%dCLCT%d_DiStrip_Quality_Profile", ChamberID, lct);
			//me[meName] = new TProfile(meName.c_str(), Form("CLCT%d Average Quality",lct), 40, 0, 160, 0, 7, "g");



			meName = Form("%dCLCT%d_BXN", ChamberID, lct);
			me[meName] = dbe->book1D(meName.c_str(), Form("CLCT%d BXN",lct), 4, 0.0, 4.0);

			meName = Form("%dCLCT%d_dTime_vs_Half_Strip", ChamberID, lct);
			me[meName] = dbe->book2D(meName.c_str(), Form("(CLCT%d BXN - TMB_L1A BXN) vs Key Half Strip",lct), 160, 0, 160, 8, -4.0, 4.0);

			meName = Form("%dCLCT%d_dTime", ChamberID, lct);
			me[meName] = dbe->book1D(meName.c_str(), Form("CLCT%d BXN - TMB_L1A BXN Difference",lct), 8, -4.0, 4.0);

			meName = Form("%dCLCT%d_dTime_vs_DiStrip", ChamberID, lct);
			me[meName] = dbe->book2D(meName.c_str(), Form("(CLCT%d BXN - TMB_L1A BXN) vs Key DiStrip",lct), 40, 0, 160, 8, -4.0, 4.0);
	}

// CFEBs
	//if(debug_printout) 	cout << "D**EmuBookChamber> New CFEB Canvases are booking ..." << endl;
        dbe->setCurrentFolder("CFEB");

 
//	CFEBs by numbers
	for (int nCFEB=0; nCFEB<5; nCFEB++) {
		meName = Form("%dCFEB%d_SCA_Block_Occupancy", ChamberID, nCFEB);
		me[meName] = dbe->book1D(meName.c_str(), Form("CFEB%d",nCFEB), 16, 0.0, 16.0);
        }


//	CFEBs by layers

	for (int nLayer=1; nLayer<=6; nLayer++) {
		meName = Form("%dCFEB_ActiveStrips_Ly%d", ChamberID, nLayer);
		me[meName] = dbe->book1D(meName.c_str(), Form("Layer %d", nLayer), 80, 0, 80);
	}

	for (int nLayer=1; nLayer<=6; nLayer++) {
		meName = Form("%dCFEB_Active_Samples_vs_Strip_Ly%d", ChamberID, nLayer);
		me[meName] = dbe->book2D(meName.c_str(), Form("Layer %d", nLayer), 80, 0, 80, 16, 0, 16);
	}

	//for (int nLayer=1; nLayer<=6; nLayer++) {
	//	meName = Form("%dCFEB_Active_Samples_vs_Strip_Ly%d_Profile", ChamberID, nLayer);
	//	me[meName] = new TProfile(meName.c_str(), Form("Layer %d", nLayer), 80, 0, 80, 0, 16,"g");
	//}

	for (int nLayer=1; nLayer<=6; nLayer++) {
                meName = Form("%dCFEB_SCA_Cell_Occupancy_Ly_%d", ChamberID, nLayer);
                me[meName] = dbe->book2D(meName.c_str(), Form("Layer %d", nLayer), 80, 0.0, 80.0, 96, 0.0, 96.0);
	}

	for (int nLayer=1; nLayer<=6; nLayer++) {
		meName = Form("%dCFEB_SCA_Cell_Peak_Ly_%d", ChamberID, nLayer);
		me[meName] = dbe->book2D(meName.c_str(), Form("Layer %d", nLayer), 80, 0.0, 80.0, 96, 0.0, 96.0);
	}

	//for (int nLayer=1; nLayer<=6; nLayer++) {
//		Histograms of CFEB Pedestals with Error on Mean Value: Option = "g"
	//	meName = Form("%dCFEB_Pedestal(withEMV)_Sample_01_Ly%d", ChamberID, nLayer);
	//	me[meName] = new TProfile(meName.c_str(), Form("Layer %d", nLayer), 80, 0, 80, 0, 4096,"g");
	//}

	for (int nLayer=1; nLayer<=6; nLayer++) {
//		Histograms of CFEB Pedestals with Root Mean Square: Option = "s"
              //  meName = Form("%dCFEB_Pedestal(withRMS)_Sample_01_Ly%d", ChamberID, nLayer);
		//me[meName] = new TProfile(meName.c_str(), Form("Layer %d", nLayer), 80, 0, 80, 0, 4096,"s");

		meName = Form("%dCFEB_PedestalRMS_Sample_01_Ly%d",ChamberID,nLayer);
                me[meName] = dbe->book1D(meName.c_str(), Form("Layer %d", nLayer), 80, 0.0, 80.0);

        }

	for (int nLayer=1; nLayer<=6; nLayer++) {
		meName = Form("%dCFEB_Out_Off_Range_Strips_Ly%d", ChamberID, nLayer);
		me[meName] = dbe->book1D(meName.c_str(), Form("Layer %d", nLayer), 80, 0, 80);
	}

	for (int nLayer=1; nLayer<=6; nLayer++) {
		meName = Form("%dCFEB_Number_of_Clusters_Ly_%d", ChamberID, nLayer);
		me[meName] = dbe->book1D(meName.c_str(), Form("Layer %d", nLayer), 41, 0.0, 41.0);
	}

	for (int nLayer=1; nLayer<=6; nLayer++) {
                meName = Form("%dCFEB_Width_of_Clusters_Ly_%d", ChamberID, nLayer);
                me[meName] = dbe->book1D(meName.c_str(), Form("Layer %d", nLayer), 80, 0.0, 80.0);
        }

	for (int nLayer=1; nLayer<=6; nLayer++) {
                meName = Form("%dCFEB_Clusters_Charge_Ly_%d", ChamberID, nLayer);
                me[meName] = dbe->book1D(meName.c_str(), Form("Layer %d", nLayer), 100, 80, 12000);
        }

//SYNC
	//if(debug_printout) 	cout << "D**EmuBookChamber> New SYNC Canvases are booking ..." << endl;
        dbe->setCurrentFolder("SYNC");


		meName = Form("%dDMB_L1A_Distrib", ChamberID);
		me[meName] = dbe->book1D(meName.c_str(), "Distribution of DMB L1A Counter", 256, 0.0, 256);


		meName = Form("%dDMB_DDU_L1A_diff", ChamberID);
		me[meName] = dbe->book1D(meName.c_str(), "Difference between DMB and DDU L1A numbers", 256, -128.0, 128.0);

		meName = Form("%dDMB_L1A_vs_DDU_L1A", ChamberID);
		me[meName] = dbe->book2D(meName.c_str(), "DMB L1A vs DDU L1A", 256, 0.0, 256.0, 256, 0.0, 256.0);

		meName = Form("%dDMB_BXN_Distrib", ChamberID);
		me[meName] = dbe->book1D(meName.c_str(), "Distribution of DMB BXN Counter", 128,  0.0, 128.0);

		meName = Form("%dDMB_DDU_BXN_diff", ChamberID);
		me[meName] = dbe->book1D(meName.c_str(), "Difference between DMB and DDU BXN numbers", 128, -64.0, 64.0);

		meName = Form("%dDMB_BXN_vs_DDU_BXN",ChamberID);
		me[meName] = dbe->book2D(meName.c_str(), "DMB BXN vs DDU BXN", 4096, 0 , 4096, 128, 0, 128);


		meName = Form("%dALCT_L1A", ChamberID);
		me[meName] = dbe->book1D(meName.c_str(), "Distribution of ALCT L1A counter", 16, 0.0, 16.0);

		meName = Form("%dALCT_DMB_L1A_diff", ChamberID);
		me[meName] = dbe->book1D(meName.c_str(), "Difference between ALCT and DMB L1A numbers", 16,  -8 , 8);

		meName = Form("%dDMB_L1A_vs_ALCT_L1A", ChamberID);
		me[meName] = dbe->book2D(meName.c_str(), "DMB L1A vs ALCT L1A", 16, 0.0, 16.0, 256, 0.0, 256.0);

		meName = Form("%dALCT_BXN", ChamberID);
		me[meName] = dbe->book1D(meName.c_str(), "Distribution of ALCT BXN counter", 1024,  0 , 1024);

		meName = Form("%dALCT_DMB_BXN_diff", ChamberID);
		me[meName] = dbe->book1D(meName.c_str(), "Difference between ALCT and DMB BXN numbers", 128,  -64 , 64);

		meName = Form("%dALCT_BXN_vs_DMB_BXN", ChamberID);
		me[meName] = dbe->book2D(meName.c_str(), "DMB BXN vs ALCT BXN Numbers", 1024,  0 , 1024, 128,  0 , 128);

		meName = Form("%dCLCT_L1A", ChamberID);
		me[meName] = dbe->book1D(meName.c_str(), "Distribution of TMB L1A Counter", 18,  0 , 18);

		meName = Form("%dCLCT_DMB_L1A_diff", ChamberID);
		me[meName] = dbe->book1D(meName.c_str(), "Difference between TMB and DMB L1A numbers", 16, -8, 8);

		meName = Form("%dDMB_L1A_vs_CLCT_L1A", ChamberID);
		me[meName] = dbe->book2D(meName.c_str(), "DMB L1A vs TMB L1A", 16, 0.0, 16.0, 256, 0.0, 256.0);

		meName = Form("%dCLCT_BXN", ChamberID);
		me[meName] = dbe->book1D(meName.c_str(), "Distribution of TMB BXN counter", 1024, 0, 1024);

		meName = Form("%dCLCT_DMB_BXN_diff", ChamberID);
		me[meName] = dbe->book1D(meName.c_str(), "Difference between TMB and DMB BXN numbers", 128, -64, 64);

		meName = Form("%dCLCT_BXN_vs_DMB_BXN", ChamberID);
		me[meName] = dbe->book2D(meName.c_str(), "DMB BXN vs TMB BXN Numbers", 1024,  0 , 1024, 128,  0 , 128);

		meName = Form("%dTMB_L1A_vs_ALCT_L1A", ChamberID);
		me[meName] = dbe->book2D(meName.c_str(), "TMB L1A vs ALCT L1A", 16,  0 , 16, 16,  0 , 16);

		meName = Form("%dTMB_ALCT_L1A_diff", ChamberID);
		me[meName] = dbe->book1D(meName.c_str(), "TMB L1A - ALCT L1A", 16,  -8 , 8);

		meName = Form("%dTMB_BXN_vs_ALCT_BXN", ChamberID);
		me[meName] = dbe->book2D(meName.c_str(), "TMB BXN vs ALCT BXN", 1024,  0 , 1024, 1024,  0 , 1024);

		meName = Form("%dTMB_ALCT_BXN_diff", ChamberID);
		me[meName] = dbe->book1D(meName.c_str(), "TMB BXN - ALCT BXN", 1024,  -512 , 512);

	for (int nCFEB=0; nCFEB<=4; nCFEB++) {
		meName = Form("%dCFEB%d_L1A_Sync_Time", ChamberID, nCFEB);
		me[meName] = dbe->book1D(meName.c_str(), Form("CFEB%d L1A Sync BXN", nCFEB), 16, 0, 16);

		meName = Form("%dCFEB%d_L1A_Sync_Time_vs_DMB", ChamberID, nCFEB);
		me[meName] = dbe->book2D(meName.c_str(), Form("CFEB%d L1A Sync Time vs DMB CFEB Sync Time", nCFEB), 16, 0, 16, 16, 0, 16);

		meName = Form("%dCFEB%d_L1A_Sync_Time_DMB_diff", ChamberID, nCFEB);
		me[meName] = dbe->book1D(meName.c_str(), Form("CFEB%d L1A Sync Time - DMB CFEB Sync Time", nCFEB), 16, -8.0, 8.0);
	}

	for (int nCFEB=0; nCFEB<=4; nCFEB++) {
		meName = Form("%dCFEB%d_LCT_PHASE_vs_L1A_PHASE", ChamberID, nCFEB);
		me[meName] = dbe->book2D(meName.c_str(), Form("LCT Phase vs L1A Phase. CFEB%d", nCFEB), 2, 0, 2, 2, 0, 2);
	}

	return me;
}




