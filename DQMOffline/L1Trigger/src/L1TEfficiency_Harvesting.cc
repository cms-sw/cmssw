/*
 * \file L1TEfficiency_Harvesting.cc
 *
 * $Date: 2012/11/20 17:02:27 $
 * $Revision: 1.3 $
 * \author J. Pela, P. Musella
 *
 */

// L1TMonitor includes
#include "DQMOffline/L1Trigger/interface/L1TEfficiency_Harvesting.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/Common/interface/ConditionsInEdm.h" // Parameters associated to Run, LS and Event
#include "DataFormats/Luminosity/interface/LumiDetails.h" // Luminosity Information
#include "DataFormats/Luminosity/interface/LumiSummary.h" // Luminosity Information

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"            // L1Gt - Masks
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h" // L1Gt - Masks
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsAlgoTrigRcd.h"

#include "DataFormats/Histograms/interface/MEtoEDMFormat.h"

#include "TList.h"

using namespace edm;
using namespace std;

//_____________________________________________________________________
L1TEfficiency_Harvesting::L1TEfficiency_Harvesting(const ParameterSet & ps){
  
  // Inicializing Variables
  if (m_verbose) {
    cout << "[L1TEfficiency_Harvesting:] ____________ Storage inicialization ____________ " << endl;
    cout << "[L1TEfficiency_Harvesting:] Setting up dbe folder: L1T/Efficiency" << endl;
  }
  
  dbe = Service < DQMStore > ().operator->();
  dbe->setVerbose(0);
  dbe->setCurrentFolder("L1T/Efficiency");
  
  // Inicializing Variables
  if (m_verbose) {cout << "[L1TEfficiency_Harvesting:] Pointer for DQM Store: " << dbe << endl;}
}

//_____________________________________________________________________
L1TEfficiency_Harvesting::~L1TEfficiency_Harvesting(){}

//_____________________________________________________________________
void L1TEfficiency_Harvesting::beginJob(void){
  
  if (m_verbose) {cout << "[L1TEfficiency_Harvesting:] Called beginJob." << endl;}
  
  // get hold of back-end interface
//   DQMStore *dbe = 0;
//   dbe = Service < DQMStore > ().operator->();
  
}

//_____________________________________________________________________
void L1TEfficiency_Harvesting::endJob(void){
  
  if (m_verbose) {cout << "[L1TEfficiency_Harvesting:] Called endJob." << endl;}

}

//_____________________________________________________________________
// BeginRun: as input you get filtered events...
//_____________________________________________________________________
void L1TEfficiency_Harvesting::beginRun(const edm::Run& run, const edm::EventSetup& iSetup){
  
  if (m_verbose) {cout << "[L1TEfficiency_Harvesting:] Called endRun." << endl;}

}  

//_____________________________________________________________________
void L1TEfficiency_Harvesting::endRun(const edm::Run& run, const edm::EventSetup& iSetup){
  
  if (m_verbose) {cout << "[L1TEfficiency_Harvesting:] Called endRun." << endl;}
  
}

//_____________________________________________________________________
void L1TEfficiency_Harvesting::beginLuminosityBlock(LuminosityBlock const& lumiBlock, EventSetup const& c) {
  
  if(m_verbose){
    cout << "[L1TEfficiency_Harvesting:] Called beginLuminosityBlock at LS=" 
         << lumiBlock.id().luminosityBlock() << endl;
  }
}

//_____________________________________________________________________
void L1TEfficiency_Harvesting::endLuminosityBlock(LuminosityBlock const& lumiBlock, EventSetup const& c) {
  
  if(m_verbose){
    cout << "[L1TEfficiency_Harvesting:] Called endLuminosityBlock at LS=" 
         << lumiBlock.id().luminosityBlock() << endl;
  }
}


//_____________________________________________________________________
void L1TEfficiency_Harvesting::analyze(const Event & iEvent, const EventSetup & eventSetup){
  
  
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TEfficiency_Harvesting);