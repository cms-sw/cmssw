/** \file
 * 
 * implementation of CSCMonitor::MonitorCLCT(...) method
 *  $Date: 2006/02/10 10:30:55 $
 *  $Revision: 1.5 $
 *
 * \author Ilaria Segoni
 */
 
#include "DQM/CSCMonitorModule/interface/CSCMonitor.h"
#include "EventFilter/CSCRawToDigi/interface/CSCTMBHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCTMBTrailer.h"

#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

void CSCMonitor::MonitorCLCT(std::vector<CSCEventData>::iterator data,int ChamberID ){


string meName;
map<string, MonitorElement *> me = meChamber[ChamberID];

//CLCT Found
if (data->nclct()) {
	CSCTMBData tmbData = data->tmbData();
	CSCTMBHeader tmbHeader = tmbData.tmbHeader();
	CSCTMBTrailer tmbTrailer = tmbData.tmbTrailer();
        
	CSCDMBHeader dmbHeader  = data->dmbHeader();
	
	//vector<L1MuCSCCathodeLCT> clctsDatas;// = tmbHeader.CLCTs(); //Not in CMSSW
	//L1MuCSCCathodeLCT clct0(tmbHeader.clct0Word());
	//if (clct0.isValid())
		//clctsDatas.push_back(clct0);
	//L1MuCSCCathodeLCT clct1(tmbHeader.clct1Word());
	//if (clct1.isValid())
        	//clctsDatas.push_back(clct1);
	CSCCLCTData clctData = data->clctData();

	FEBUnpacked = FEBUnpacked +1;

	meName = Form("%dALCT_Match_Time", ChamberID);
        me[meName]->Fill(tmbHeader.ALCTMatchTime());

	meName = Form("%dLCT_Match_Status", ChamberID);
        if (tmbHeader.CLCTOnly()) me[meName]->Fill(0.0,0.0);
        if (tmbHeader.ALCTOnly()) me[meName]->Fill(0.0,1.0);
        if (tmbHeader.TMBMatch()) me[meName]->Fill(0.0,2.0);

	meName = Form("%dLCT0_Match_BXN_Difference", ChamberID);
        me[meName]->Fill(tmbHeader.Bxn0Diff());


	meName = Form("%dLCT1_Match_BXN_Difference", ChamberID);
        me[meName]->Fill(tmbHeader.Bxn1Diff());



	meName = Form("%d_CSC_Rate", ChamberID);
//		Set number of CLCT-events to forth bin
   	   me[meName]->Fill(3);

   	   //float CLCTEvent = me[meName]->GetBinContent(4);
   	   meName = Form("%d_CSC_Efficiency", ChamberID);
   	   if(nEvents > 0) {
   		   //me[meName]->SetBinContent(4,((float)CLCTEvent/(float)(nEvents)*100.0)); // KK
   		   //me[meName]->SetBinContent(3,((float)CLCTEvent/(float)(DMBEvent)*100.0));   // KK
   		   //me[meName]->SetEntries(nEvents);
   	   }

   	   meName = Form("%dCLCT_L1A", ChamberID);
   	   me[meName]->Fill(tmbHeader.L1ANumber());

   	   meName = Form("%dCLCT_DMB_L1A_diff", ChamberID);
   	   int clct_dmb_l1a_diff = (int)((dmbHeader.l1a()&0xF)-tmbHeader.L1ANumber());
   	   if(clct_dmb_l1a_diff < -8) me[meName]->Fill(clct_dmb_l1a_diff + 16);
   	   else {
   		   if(clct_dmb_l1a_diff > 8)  me[meName]->Fill(clct_dmb_l1a_diff - 16);
   		   else me[meName]->Fill(clct_dmb_l1a_diff);
   	   }
   	   //me[meName]->SetAxisRange(0.1, 1.1*(1.0+me[meName]->GetBinContent(me[meName]->GetMaximumBin())), "Y");

   	   meName = Form("%dDMB_L1A_vs_CLCT_L1A", ChamberID);
   	   me[meName]->Fill(tmbHeader.L1ANumber(),dmbHeader.l1a());

   	   meName = Form("%dCLCT_DMB_BXN_diff", ChamberID);
   	   int clct_dmb_bxn_diff = (int)(dmbHeader.bxn()-(tmbHeader.BXNCount()&0x7F));
   	   if(clct_dmb_bxn_diff < -64) me[meName]->Fill(clct_dmb_bxn_diff + 128);
   	   else {
   		   if(clct_dmb_bxn_diff > 64)  me[meName]->Fill(clct_dmb_bxn_diff - 128);
   		   else me[meName]->Fill(clct_dmb_bxn_diff);
   	   }
   	   //me[meName]->SetAxisRange(0.1, 1.1*(1.0+me[meName]->GetBinContent(me[meName]->GetMaximumBin())), "Y");

   	   meName = Form("%dCLCT_BXN", ChamberID);
   	   me[meName]->Fill((int)(tmbHeader.BXNCount()));

   	   meName = Form("%dCLCT_BXN_vs_DMB_BXN", ChamberID);
   	   me[meName]->Fill(tmbHeader.BXNCount(),dmbHeader.bxn());

   	   //meName = Form("%dCLCT_Number_Rate", ChamberID);
   	   //me[meName]->Fill(clctsDatas.size());
   	   //int nCLCT = (int)me[meName]->GetBinContent((int)(clctsDatas.size()+1));

   	   //meName = Form("%dCLCT_Number", ChamberID);
   	   //me[meName]->SetBinContent((int)(clctsDatas.size()+1), (float)(nCLCT)/(float)(DMBEvent)*100.0);

   	   //if (clctsDatas.size()==2) {
   	//	   meName = Form("%dCLCT1_vs_CLCT0_Key_Strip", ChamberID);
   	//	   me[meName]->Fill(clctsDatas[0].getKeyStrip(),clctsDatas[1].getKeyStrip());
   	   //}

   	   //if (clctsDatas.size()==1) {
   		 //  meName = Form("%dCLCT0_Clssification", ChamberID);
   		 //  if (clctsDatas[0].getStripType())	   me[meName]->Fill(0.0);
   		 //  else 				   me[meName]->Fill(1.0);
   	   //}

   	  // if (clctsDatas.size()==2) {
   	//	   meName = Form("%dCLCT0_CLCT1_Clssification", ChamberID);
   	//	   if ( clctsDatas[0].getStripType() &&  clctsDatas[1].getStripType())     me[meName]->Fill(0.0);
   	//	   if ( clctsDatas[0].getStripType() && !clctsDatas[1].getStripType())     me[meName]->Fill(1.0);
   	//	   if (!clctsDatas[0].getStripType() &&  clctsDatas[1].getStripType())     me[meName]->Fill(2.0);
   	//	   if (!clctsDatas[0].getStripType() &&  !clctsDatas[1].getStripType())    me[meName]->Fill(3.0);
   	  // }

   	   meName = Form("%dTMB_Word_Count", ChamberID);
   	   me[meName]->Fill((int)(tmbTrailer.wordCount()));
   	  edm::LogInfo ("CSC DQM: ") << "+++debug>  TMB Trailer Word Count = " << (int)tmbTrailer.wordCount();

   	   

/*
// Adjust to new way of storing comparators
// namely:compOutDataItr->halfStrip() is obsolete 
		int NumberOfLayersWithHitsInCLCT = 0;
		int NumberOfHalfStripsWithHitsInCLCT = 0;
		for (int nLayer=1; nLayer<=6; nLayer++) {
			int hstrip_previous    = -1;
			int tbin_clct_previous = -1;
			bool CheckLayerCLCT = true;
			vector<CSCComparatorDigi> compOutData = clctData.comparatorDigis(nLayer);
			for (vector<CSCComparatorDigi>:: iterator compOutDataItr = compOutData.begin();
			     compOutDataItr != compOutData.end(); ++compOutDataItr) {
				int hstrip = 2*compOutDataItr->getStrip() + compOutDataItr->halfStrip();
				int tbin_clct = (int)compOutDataItr->getTimeBin();
				if(CheckLayerCLCT) {
					NumberOfLayersWithHitsInCLCT = NumberOfLayersWithHitsInCLCT + 1;
					CheckLayerCLCT = false;
				}
				if(hstrip != hstrip_previous ||
				   (tbin_clct != tbin_clct_previous + 1 && tbin_clct != tbin_clct_previous - 1) ) {
					meName = Form("%dCLCTTime_Ly%d", ChamberID, nLayer);
					me[meName]->Fill(hstrip, tbin_clct);

					meName = Form("%dCLCTTime_Ly%d_Profile", ChamberID, nLayer);
					me[meName]->Fill(hstrip, tbin_clct);

					meName = Form("%dCLCT_Ly%d_Rate", ChamberID, nLayer);
					me[meName]->Fill(hstrip);

					//int number_hstrip = (int)(me[meName]->GetBinContent(hstrip+1));
					//meName = Form("%dCLCT_Ly%d_Efficiency",ChamberID, nLayer);
					//me[meName]->SetBinContent(hstrip+1,((float)number_hstrip)/((float)DMBEvent)*100.0);
					//me[meName]->SetEntries(DMBEvent);
				}
				if(hstrip != hstrip_previous) {
					NumberOfHalfStripsWithHitsInCLCT = NumberOfHalfStripsWithHitsInCLCT + 1;
				}
				hstrip_previous    = hstrip;
				tbin_clct_previous = tbin_clct;
			}
		}
		meName = Form("%dCLCT_Number_Of_Layers_With_Hits", ChamberID);
		me[meName]->Fill(NumberOfLayersWithHitsInCLCT);
		meName = Form("%dCLCT_Number_Of_HalfStrips_With_Hits", ChamberID);

		me[meName]->Fill(NumberOfHalfStripsWithHitsInCLCT);
*/
		
 } else {
//	CLCT not found
	meName = Form("%dCLCT_Number_Rate", ChamberID);
	me[meName]->Fill(0);
	//int nCLCT = (int)me[meName]->GetBinContent(1);
	//meName = Form("%dCLCT_Number", ChamberID);
	//me[meName]->SetBinContent(1, (float)(nCLCT)/(float)(DMBEvent)*100.0);
 }




}
