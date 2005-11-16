#include <memory>
#include <iostream>

#include "DQM/CSCMonitorModule/interface/CSCMonitor.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"

using namespace std;




///	MonitorElements COMMON TO ALL CHAMBERS

map<string, MonitorElement*> CSCMonitor::book_common() {

 string meName;
 map<string, MonitorElement*> meMap;


///DDU
 //if(debug_printout) cout << "D**EmuBookCommon> New DDU Canvases are booking ..." << endl;
 dbe->setCurrentFolder("DDU");

 meName = "DDU_Readout_Errors";
 meMap[meName] = dbe->book2D(meName.c_str(), "DDU RUI Readout Errors", 1, 0, 1, 16 ,0 ,16);

// meName = "DDUBinCheck_Errors";
// meMap[meName] = dbe->book2D(meName.c_str(), "DDU Data Format Errors", 1, 0, 1, bin_checker.nERRORS, 0, bin_checker.nERRORS);

// meName = "DDUBinCheck_Warnings";
// meMap[meName] = dbe->book2D(meName.c_str(), "DDU Data Format Warnings", 1, 0, 1, bin_checker.nWARNINGS, 0, bin_checker.nWARNINGS);

 meName = "CSC_Unpacked";
 meMap[meName] = dbe->book2D(meName.c_str(), "", 300 ,0 , 300, 16, 0, 16);

 meName = "DDU_BXN";
 meMap[meName] = dbe->book1D(meName.c_str(), "", 4096 ,  0 , 4096);

 meName = "DDU_L1A_Increment";
 meMap[meName] = dbe->book1D(meName.c_str(), "Incremental change in DDU L1A number since previous event", 100 ,  0 , 100);

///KK
 //meName = "DDU_DMB_Connected_Inputs_Rate";
 //meMap[meName] = (meName.c_str(), "DDU_DMB_Connected_Inputs_Rate", 16 ,  0 , 16);

 meName = "DDU_DMB_Connected_Inputs";
 meMap[meName] = dbe->book1D(meName.c_str(), "DDU Inputs connected to DMBs", 16 ,  0 , 16);

///KK end
 meName = "DDU_DMB_DAV_Header_Occupancy_Rate";
 meMap[meName] = dbe->book1D(meName.c_str(), "DMBs reporting DAV (data available) in Header", 16 ,  0 , 16);

 meName = "DDU_DMB_DAV_Header_Occupancy";
 meMap[meName] = dbe->book1D(meName.c_str(), "DMBs reporting DAV (data available) in Header", 16 ,  0 , 16);

 meName = "DDU_DMB_Active_Header_Count";
 meMap[meName] = dbe->book1D(meName.c_str(), "Number of active DMBs reporting DAV in Header", 16 ,  0 , 16);

 meName = "DDU_DMB_DAV_Header_Count_vs_DMB_Active_Header_Count";
 meMap[meName] = dbe->book2D(meName.c_str(), "DMB DAV Header Count vs DMB Active Header Count", 16, 0, 16, 16, 0, 16);

 meName = "DDU_DMB_unpacked_vs_DAV";
 meMap[meName] = dbe->book2D(meName.c_str(), "Number of unpacked DMBs vs. number of DMBs reporting DAV", 16 ,  0 , 16, 16, 0, 16);

 meName = "DDU_Data_Format_Check_vs_nEvents";
 meMap[meName] = dbe->book2D(meName.c_str(), "Check DDU Data Format", 100000, 0, 100000, 3,  0 , 3);

 meName = "DDU_Unpacking_Match_vs_nEvents";
 meMap[meName] = dbe->book2D(meName.c_str(), "Match: Unpacked DMBs and DMBs Reporting DAV", 100000, 0, 100000, 2, 0, 2);


 meName = "DDU_L1A_Increment_vs_nEvents";
 meMap[meName] = dbe->book2D(meName.c_str(), "Incremental L1A", 100000, 0, 100000, 3, 0, 3);


 meName = "DDU_Trailer_ErrorStat_vs_nEvents";
 meMap[meName] = dbe->book2D(meName.c_str(), "", 100000,  0 , 100000, 32, 0, 32);

 meName = "DDU_Trailer_ErrorStat_Table";
 meMap[meName] = dbe->book2D(meName.c_str(), "DDU Trailer Status Error Table", 1,  0 , 1, 32, 0, 32);

 meName = "DDU_Trailer_ErrorStat_Rate";
 meMap[meName] = dbe->book1D(meName.c_str(), "", 32,  0 , 32);


 meName = "DDU_Trailer_ErrorStat_Occupancy";
 meMap[meName] = dbe->book1D(meName.c_str(), "DDU Trailer Status Error Frecuency", 32,  0 , 32);

 meName = "DDU_Buffer_Size";
 meMap[meName] = dbe->book1D(meName.c_str(), "DDU Buffer Size", 128 ,0 ,65536 ); // 65536 = 2^16

 meName = "DDU_Word_Count";
 meMap[meName] = dbe->book1D(meName.c_str(), "DDU Word (64 bits) Count", 128,  0 , 8192); //8192 = 2^13

///KK

 meName = "DDU_CSC_Errors_Rate";
 meMap[meName] = dbe->book1D(meName.c_str(), "", 15 ,  1 , 16);


 meName = "DDU_CSC_Errors";
 meMap[meName] = dbe->book1D(meName.c_str(), "Errors", 15 ,1 ,16);

 meName = "DDU_CSC_Warnings_Rate";
 meMap[meName] = dbe->book1D(meName.c_str(), "", 15 ,  1 , 16);


 meName = "DDU_CSC_Warnings";
 meMap[meName] = dbe->book1D(meName.c_str(), "Warnings", 15,  1 , 16);
///KK end



	return meMap;
}


