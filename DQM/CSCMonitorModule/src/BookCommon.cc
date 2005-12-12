/** \file
 * implementation of 
 * map<string, MonitorElement*> CSCMonitor::book_common(int dduNumber)
 * method
 * 
 *  $Date: 2005/11/11 16:22:45 $
 *  $Revision: 1.4 $
 *
 * \author Ilaria Segoni
 */

#include <memory>
#include <iostream>

#include "DQM/CSCMonitorModule/interface/CSCMonitor.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"

using namespace std;




///	MonitorElements COMMON TO ALL CHAMBERS

map<string, MonitorElement*> CSCMonitor::book_common(int dduNumber) {

 string meName;
 map<string, MonitorElement*> meMap;


///DDU
 //if(debug_printout) cout << "D**EmuBookCommon> New DDU Canvases are booking ..." << endl;
 string dir = Form("DDU_%d",dduNumber);
 dbe->setCurrentFolder(dir);

  //meName = Form("DDU_Readout_Errors%d",dduNumber);
 ///meMap[meName] = dbe->book2D(meName.c_str(), "DDU RUI Readout Errors", 1, 0, 1, 16 ,0 ,16);

// meName = Form("DDUBinCheck_Errors_%d",dduNumber);
// meMap[meName] = dbe->book2D(meName.c_str(), "DDU Data Format Errors", 1, 0, 1, bin_checker.nERRORS, 0, bin_checker.nERRORS);

// meName = Form("DDUBinCheck_Warnings_%d",dduNumber);
// meMap[meName] = dbe->book2D(meName.c_str(), "DDU Data Format Warnings", 1, 0, 1, bin_checker.nWARNINGS, 0, bin_checker.nWARNINGS);

 meName = Form("CSC_Unpacked_%d",dduNumber);
 meMap[meName] = dbe->book2D(meName.c_str(), "", 300 ,0 , 300, 16, 0, 16);

 meName = Form("DDU_BXN_%d",dduNumber);
 meMap[meName] = dbe->book1D(meName.c_str(), "", 4096 ,  0 , 4096);

 meName = Form("DDU_L1A_Increment_%d",dduNumber);
 meMap[meName] = dbe->book1D(meName.c_str(), "Incremental change in DDU L1A number since previous event", 20 ,  0 , 20);

///KK
 meName = Form("DDU_DMB_Connected_Inputs_Rate_%d",dduNumber);
 meMap[meName] = dbe->book1D(meName.c_str(), "DDU_DMB_Connected_Inputs_Rate", 16 ,  0 , 16);

 //meName = Form("DDU_DMB_Connected_Inputs_%d",dduNumber);
 //meMap[meName] = dbe->book1D(meName.c_str(), "DDU Inputs connected to DMBs", 16 ,  0 , 16);

///KK end
 meName = Form("DDU_DMB_DAV_Header_Occupancy_Rate_%d",dduNumber);
 meMap[meName] = dbe->book1D(meName.c_str(), "DMBs reporting DAV (data available) in Header", 16 ,  0 , 16);

 //meName = Form("DDU_DMB_DAV_Header_Occupancy_%d",dduNumber);
 //meMap[meName] = dbe->book1D(meName.c_str(), "DMBs reporting DAV (data available) in Header", 16 ,  0 , 16);

 meName = Form("DDU_DMB_Active_Header_Count_%d",dduNumber);
 meMap[meName] = dbe->book1D(meName.c_str(), "Number of active DMBs reporting DAV in Header", 16 ,  0 , 16);

 meName = Form("DDU_DMB_DAV_Header_Count_vs_DMB_Active_Header_Count_%d",dduNumber);
 meMap[meName] = dbe->book2D(meName.c_str(), "DMB DAV Header Count vs DMB Active Header Count", 16, 0, 16, 16, 0, 16);

 meName = Form("DDU_DMB_unpacked_vs_DAV_%d",dduNumber);
 meMap[meName] = dbe->book2D(meName.c_str(), "Number of unpacked DMBs vs. number of DMBs reporting DAV", 16 ,  0 , 16, 16, 0, 16);

 //meName = Form("DDU_Data_Format_Check_vs_nEvents_%d",dduNumber);
 //meMap[meName] = dbe->book2D(meName.c_str(), "Check DDU Data Format", 100000, 0, 100000, 3,  0 , 3);

 meName = Form("DDU_Unpacking_Match_vs_nEvents_%d",dduNumber);
 meMap[meName] = dbe->book2D(meName.c_str(), "Match: Unpacked DMBs and DMBs Reporting DAV", 3000, 0, 3000, 2, 0, 2);


 meName = Form("DDU_L1A_Increment_vs_nEvents_%d",dduNumber);
 meMap[meName] = dbe->book2D(meName.c_str(), "Incremental L1A", 3000, 0, 3000, 3, 0, 3);


 meName = Form("DDU_Trailer_ErrorStat_vs_nEvents_%d",dduNumber);
 meMap[meName] = dbe->book2D(meName.c_str(), "", 3000,  0 , 3000, 32, 0, 32);

 meName = Form("DDU_Trailer_ErrorStat_Table_%d",dduNumber);
 meMap[meName] = dbe->book2D(meName.c_str(), "DDU Trailer Status Error Table", 1,  0 , 1, 32, 0, 32);

 meName = Form("DDU_Trailer_ErrorStat_Rate_%d",dduNumber);
 meMap[meName] = dbe->book1D(meName.c_str(), "", 32,  0 , 32);


 //meName = Form("DDU_Trailer_ErrorStat_Occupancy_%d",dduNumber);
 //meMap[meName] = dbe->book1D(meName.c_str(), "DDU Trailer Status Error Frecuency", 32,  0 , 32);

 meName = Form("DDU_Buffer_Size_%d",dduNumber);
 meMap[meName] = dbe->book1D(meName.c_str(), "DDU Buffer Size", 128 ,0 ,65536 ); // 65536 = 2^16

 meName = Form("DDU_Word_Count_%d",dduNumber);
 meMap[meName] = dbe->book1D(meName.c_str(), "DDU Word (64 bits) Count", 128,  0 , 8192); //8192 = 2^13

///KK

 meName = Form("DDU_CSC_Errors_Rate_%d",dduNumber);
 meMap[meName] = dbe->book1D(meName.c_str(), "", 15 ,  1 , 16);


// meName = Form("DDU_CSC_Errors_%d",dduNumber);
// meMap[meName] = dbe->book1D(meName.c_str(), "Errors", 15 ,1 ,16);

 meName = Form("DDU_CSC_Warnings_Rate_%d",dduNumber);
 meMap[meName] = dbe->book1D(meName.c_str(), "", 15 ,  1 , 16);


// meName = Form("DDU_CSC_Warnings_%d",dduNumber);
// meMap[meName] = dbe->book1D(meName.c_str(), "Warnings", 15,  1 , 16);
///KK end



	return meMap;
}


