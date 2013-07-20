 /*
 * \class L1TBeamConfiguration
 *
 *
 * Description: offline DQM class for acquiring beam configuration
 *
 * Implementation:
 *   <TODO: enter implementation details>
 *
 * \author: Pietro Vischia - LIP Lisbon pietro.vischia@gmail.com
 *
 * Changelog:
 *    2012/11/2 12:01:01: Class
 *
 * Todo: see header file
 *
 * $Date: 2012/12/10 14:10:21 $
 * $Revision: 1.2 $
 *
 */

//

// This class header
#include "DQMOffline/L1Trigger/interface/L1TBeamConfiguration.h"

// System include files
// --

//// User include files
//#include "DQMServices/Core/interface/DQMStore.h"
//
//#include "DataFormats/Scalers/interface/LumiScalers.h"
//#include "DataFormats/Scalers/interface/Level1TriggerRates.h"
//#include "DataFormats/Scalers/interface/Level1TriggerScalers.h"
//
//#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
//#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"
//
//#include "DataFormats/Common/interface/ConditionsInEdm.h" // Parameters associated to Run, LS and Event
//
//#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"
//#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
//#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"
//#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
//#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsAlgoTrigRcd.h"
//#include "CondFormats/L1TObjects/interface/L1GtMuonTemplate.h"
/////
//// Luminosity Information
////#include "DataFormats/Luminosity/interface/LumiDetails.h"
////#include "DataFormats/Luminosity/interface/LumiSummary.h"
//
//// L1TMonitor includes
/////#include "DQM/L1TMonitor/interface/L1TMenuHelper.h"
//#include "DQMOffline/L1Trigger/interface/L1TMenuHelper.h"
//
//#include "TList.h"

using namespace std;

//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------

L1TBeamConfiguration::L1TBeamConfiguration(){


  m_valid = false;

}

bool L1TBeamConfiguration::bxConfig(unsigned iBx){

  if(m_valid && beam1.size()>iBx && beam2.size()>iBx){

    if(beam1[iBx] && beam2[iBx]){return true;}
    else                        {return false;}

  }else{
    return false;
  }
}

