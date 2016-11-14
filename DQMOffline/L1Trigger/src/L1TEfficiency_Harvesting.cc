/**
 * \file L1TEfficiency_Harvesting.cc
 *
 * \author J. Pela, C. Battilana
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

//__________Efficiency_Plot_Handler_Helper_Class_______________________
L1TEfficiencyPlotHandler::L1TEfficiencyPlotHandler(const L1TEfficiencyPlotHandler &handler) {
  
  m_dir      = handler.m_dir;
  m_plotName = handler.m_plotName;
  m_effHisto = handler.m_effHisto;

}


void L1TEfficiencyPlotHandler::book(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {

  cout << "[L1TEfficiencyMuons_Harvesting:] Booking efficiency histo for " 
	 << m_dir << " and " << m_plotName << endl;
  
  MonitorElement *num = igetter.get(m_dir+"/"+m_plotName+"Num");
  MonitorElement *den = igetter.get(m_dir+"/"+m_plotName+"Den");

  if (!num || !den) {
 
    cout << "[L1TEfficiencyMuons_Harvesting:] "
	   << (!num && !den ? "Num && Den" : !num ? "Num" : "Den") 
	   << " not gettable. Quitting booking" << endl;

    return;

  }

  TH1F *numH = num->getTH1F();
  TH1F *denH = den->getTH1F();

  if (!numH || !denH) {
 
    cout << "[L1TEfficiencyMuons_Harvesting:] "
	   << (!numH && !denH ? "Num && Den" : !numH ? "Num" : "Den") 
	   << " is not TH1F. Quitting booking" << endl;

    return;

  }

  int nBinsNum = numH->GetNbinsX();
  int nBinsDen = denH->GetNbinsX();

  if (nBinsNum != nBinsDen) {
 
    cout << "[L1TEfficiencyMuons_Harvesting:] # bins in num and den is different. Quitting booking" << endl;
    
    return;

  }

  double min = numH->GetXaxis()->GetXmin();
  double max = numH->GetXaxis()->GetXmax();

  ibooker.setCurrentFolder(m_dir);
  m_effHisto = ibooker.book1D(m_plotName,m_plotName,nBinsNum,min,max);

}


void L1TEfficiencyPlotHandler::computeEfficiency(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {

  if (!m_effHisto)
    return;

  cout << "[L1TEfficiencyMuons_Harvesting:] Computing efficiency for " 
	 << m_plotName << endl;
  
  MonitorElement *num = igetter.get(m_dir+"/"+m_plotName+"Num");
  MonitorElement *den = igetter.get(m_dir+"/"+m_plotName+"Den");

  TH1F *numH = num->getTH1F();
  TH1F *denH = den->getTH1F();

  numH->Sumw2();
  denH->Sumw2();

  TH1F *effH = m_effHisto->getTH1F();

  effH->Divide(numH,denH);

}

  
//___________DQM_analyzer_class________________________________________
L1TEfficiency_Harvesting::L1TEfficiency_Harvesting(const ParameterSet & ps){
  
  // Initializing Variables
  if (m_verbose) {
    cout << "[L1TEfficiency_Harvesting:] ____________ Storage inicialization ____________ " << endl;
    cout << "[L1TEfficiency_Harvesting:] Setting up dbe folder: L1T/Efficiency" << endl;
  }
  
  vector<ParameterSet> plotCfgs = ps.getUntrackedParameter<vector<ParameterSet>>("plotCfgs");
  
  vector<ParameterSet>::const_iterator plotCfgIt  = plotCfgs.begin();
  vector<ParameterSet>::const_iterator plotCfgEnd = plotCfgs.end();

  for (; plotCfgIt!=plotCfgEnd; ++plotCfgIt) {
    
    string dir = plotCfgIt->getUntrackedParameter<string>("dqmBaseDir");
    vector<string> plots = plotCfgIt->getUntrackedParameter<vector<string>>("plots");
    
    vector<string>::const_iterator plotIt  = plots.begin();
    vector<string>::const_iterator plotEnd = plots.end();
    
    for (; plotIt!=plotEnd; ++plotIt)
      m_plotHandlers.push_back(L1TEfficiencyPlotHandler(dir,(*plotIt)));
			      
  }
  
}


//_____________________________________________________________________
L1TEfficiency_Harvesting::~L1TEfficiency_Harvesting(){ m_plotHandlers.clear(); }


//_____________________________________________________________________
void L1TEfficiency_Harvesting::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter){
  
  if (m_verbose) {cout << "[L1TEfficiency_Harvesting:] Called endRun." << endl;}

  vector<L1TEfficiencyPlotHandler>::iterator plotHandlerIt  = m_plotHandlers.begin();
  vector<L1TEfficiencyPlotHandler>::iterator plotHandlerEnd = m_plotHandlers.end();

  for(; plotHandlerIt!=plotHandlerEnd; ++plotHandlerIt) {
    plotHandlerIt->book(ibooker, igetter);
    plotHandlerIt->computeEfficiency(ibooker, igetter);
  }
  
}


//_____________________________________________________________________
void L1TEfficiency_Harvesting::dqmEndLuminosityBlock(DQMStore::IGetter &igetter, LuminosityBlock const& lumiBlock, EventSetup const& c) {
  
  if(m_verbose){
    cout << "[L1TEfficiency_Harvesting:] Called endLuminosityBlock at LS=" 
         << lumiBlock.id().luminosityBlock() << endl;
  }
}


//define this as a plug-in
DEFINE_FWK_MODULE(L1TEfficiency_Harvesting);
