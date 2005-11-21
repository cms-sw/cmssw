#include "DQM/CSCMonitorModule/interface/CSCMonitor.h"
#include "EventFilter/CSCRawToDigi/interface/interface/CSCALCTHeader.h"
#include "EventFilter/CSCRawToDigi/interface/interface/CSCALCTTrailer.h"
#include "EventFilter/CSCRawToDigi/interface/interface/CSCAnodeData.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"

void CSCMonitor::ALCTMonitor(std::vector<CSCventData>::iterator data,int chamberID ){

map<string, MonitorElement *> me = meCollection[ChamberID]

if (data->nalct()) {
      
     CSCALCTHeader  alctHeader = data->alctHeader();
     CSCALCTTrailer alctTrailer = data->alctTrailer();
     CSCAnodeData anodeData = data->anodeData();
    
    
     
    FEBunpacked = FEBunpacked +1;
    
      meName = Form("%d_CSC_Rate", ChamberID); 
///    Set number of ALCT-events to third bin
     me[meName]->Fill(2);
      ///float ALCTEvent = me[meName]->GetBinContent(3);
      ///meName = Form("%d_CSC_Efficiency", ChamberID);
      ///if(nEvents > 0) {
      ///        me[meName]->SetBinContent(3, ((float)ALCTEvent/(float)(nEvents)*100.0));
      ///        me[meName]->SetEntries(nEvents);
      ///}
     
     
     int L1AALCTHeader  = alctHeader.L1Acc();
     
     int dmb_alct_bxn_diff = (int)(dmbHeader.bxn()-(alctHeader.BXNCount()&0x7F));
     int dmb_alct_l1a_diff = (int)((dmbHeader.l1a()&0xF)-alctHeader.L1Acc());	  

     if(dmb_alct_bxn_diff<-8){dmb_alct_bxn_diff=dmb_alct_bxn_diff+16;}
     if(dmb_alct_bxn_diff>8){dmb_alct_bxn_diff=dmb_alct_bxn_diff-16;}
    
     if(dmb_alct_l1a_diff<-64){dmb_alct_l1a_diff=dmb_alct_l1a_diff+128;}
     if(dmb_alct_l1a_diff>64){dmb_alct_l1a_diff=dmb_alct_l1a_diff-128;}


     meName = Form("%dALCT_L1A", ChamberID);
     me[meName]->Fill(L1AALCTHeader);

     me = Form("%dALCT_DMB_L1A_diff", ChamberID);
     me[meName]->Fill(dmb_alct_l1a_diff);
     
     meName = Form("%dDMB_L1A_vs_ALCT_L1A", ChamberID);
     me[meName]->Fill(L1AALCTHeader,dmbHeader.l1a());

//   me[meName]->SetAxisRange(0.1, 1.1*(1.0+me[meName]->GetBinContent(me[meName]->GetMaximumBin())), "Y");

      meName = Form("%dALCT_BXN", ChamberID);
      me[meName]->Fill(alctHeader.BXNCount());
      
      meName = Form("%dALCT_DMB_BXN_diff", ChamberID);
      me[meName]->Fill(dmb_alct_bxn_diff);
      
      meName = Form("%dALCT_BXN_vs_DMB_BXN", ChamberID);
      me[meName]->Fill(alctHeader.BXNCount(),dmbHeader.bxn());
     
      meName = Form("%dALCT_Number_Rate", ChamberID);
      me[meName]->Fill(nalctFound);
      //int nALCT = (int)me[meName]->GetBinContent((int)(alctsDatas.size()+1));

      //meName = Form("%dALCT_Number_Efficiency", ChamberID);
      //me[meName]->SetBinContent((int)(alctsDatas.size()+1), (float)(nALCT)/(float)(DMBEvent)*100.0);

      meName = Form("%dALCT_Word_Count", ChamberID);
      me[meName]->Fill((int)(alctTrailer.wordCount()));
     

 
     int NumberOfLayersWithHitsInALCT = 0;
     int NumberOfWireGroupsWithHitsInALCT = 0;
     for (int nLayer=1; nLayer<=6; nLayer++) {
     	     int wg_previous   = -1;
     	     int tbin_previous = -1;
     	     bool CheckLayerALCT = true;
     	     vector<CSCWireDigi> wireDigis = anodeData.wireDigis(nLayer);
     	     for (vector<CSCWireDigi>:: iterator wireDigisItr = wireDigis.begin(); 
     		  wireDigisItr != wireDigis.end(); ++wireDigisItr) {
     		     int wg = wireDigisItr->getWireGroup();
     		     int tbin = wireDigisItr->getBeamCrossingTag();
     		     if(CheckLayerALCT) {
     			     NumberOfLayersWithHitsInALCT = NumberOfLayersWithHitsInALCT + 1;
     			     CheckLayerALCT = false;
     		     }
     		     if(wg != wg_previous || (tbin != tbin_previous + 1 && tbin != tbin_previous - 1) ) {
     			     meName = Form("%dALCTTime_Ly%d", ChamberID, nLayer);
     			     me[meName]->Fill(wg, tbin);

     			     meName = Form("%dALCTTime_Ly%d_Profile", ChamberID, nLayer);
     			     me[meName]->Fill(wg, tbin);

     			     meName = Form("%dALCT_Ly%d_Rate", ChamberID, nLayer);
     			     me[meName]->Fill(wg);
     			     int number_wg = (int)(me[meName]->GetBinContent(wg+1));

     			     meName = Form("%dALCT_Ly%d_Efficiency", ChamberID, nLayer);
     			     me[meName]->SetBinContent(wg+1,((float)number_wg)/((float)DMBEvent)*100.0);
     		     }
     		     if(wg != wg_previous) {
     			     NumberOfWireGroupsWithHitsInALCT = NumberOfWireGroupsWithHitsInALCT + 1;
     		     }
     		     wg_previous   = wg;
     		     tbin_previous = tbin;
     	     }
     }
     meName = Form("%dALCT_Number_Of_Layers_With_Hits", ChamberID);
     me[meName]->Fill(NumberOfLayersWithHitsInALCT);
     meName = Form("%dALCT_Number_Of_WireGroups_With_Hits", ChamberID);
     me[meName]->Fill(NumberOfWireGroupsWithHitsInALCT);
    
} else {
//	ALCT not found
	meName= Form("%dALCT_Number_Rate", ChamberID);
	me[meName]->Fill(0);

	//int nALCT = (int)me[meName]->GetBinContent(1);
	//meName = Form("%dALCT_Number_Efficiency", ChamberID);
	//me[meName]->SetBinContent(1, (float)(nALCT)/(float)(DMBEvent)*100.0);
	}
       




}


