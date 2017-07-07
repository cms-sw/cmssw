/**
 * \file L1TEfficiency_Harvesting.cc
 *
 * \author J. Pela, C. Battilana
 *
 * Stage2 Muons implementation: Anna Stakia
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
#include "TEfficiency.h"
#include "TGraph.h"
#include "TGraphAsymmErrors.h"

using namespace edm;
using namespace std;

//__________Efficiency_Plot_Handler_Helper_Class_______________________
L1TEfficiencyPlotHandler::L1TEfficiencyPlotHandler(const L1TEfficiencyPlotHandler &handler) {
    m_dir      = handler.m_dir;
    m_plotName = handler.m_plotName;
    m_effHisto = handler.m_effHisto;
}

void L1TEfficiencyPlotHandler::book(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {

    cout << "[L1TEfficiencyMuons_Harvesting:] Booking efficiency histo for " << m_dir << " and " << m_plotName << endl;

    MonitorElement *num = igetter.get(m_dir+"/"+m_plotName+"Num");
    MonitorElement *den = igetter.get(m_dir+"/"+m_plotName+"Den");

    if (!num || !den) {
        cout << "[L1TEfficiencyMuons_Harvesting:] " << (!num && !den ? "Num && Den" : !num ? "Num" : "Den") << " not gettable. Quitting booking" << endl;
        return;
    }
    TH1F *numH = num->getTH1F();
    TH1F *denH = den->getTH1F();

    if (!numH || !denH) {
        cout << "[L1TEfficiencyMuons_Harvesting:] " << (!numH && !denH ? "Num && Den" : !numH ? "Num" : "Den") << " is not TH1F. Quitting booking" << endl;
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

    string pt_sketo = "EffvsPt";

    string ptS20 = "EffvsPt_SINGLE_20";
    string etaS20 = "EffvsEta_SINGLE_20";
    string phiS20 = "EffvsPhi_SINGLE_20";

    string ptS16 = "EffvsPt_SINGLE_16";
    string etaS16 = "EffvsEta_SINGLE_16";
    string phiS16 = "EffvsPhi_SINGLE_16";

    string ptS25 = "EffvsPt_SINGLE_25";
    string etaS25 = "EffvsEta_SINGLE_25";
    string phiS25 = "EffvsPhi_SINGLE_25";

    if (m_plotName == pt_sketo.c_str()) {
        string title = "L1T Efficiency vs pt [-]";
        float xbins[33] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 55, 60, 65, 70, 80, 90, 100};
        m_effHisto = ibooker.book1D(m_plotName,title.c_str(),32, xbins);
    }
    if (m_plotName == ptS20.c_str()) {
        string title = "";
        float xbins[33] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 55, 60, 65, 70, 80, 90, 100};
        m_effHisto = ibooker.book1D(m_plotName,title.c_str(),32, xbins);
    }
    else if (m_plotName == etaS20.c_str()) {
        string title = "";
        m_effHisto = ibooker.book1D(m_plotName,title.c_str(),nBinsNum,min,max);
    }
    else if (m_plotName == phiS20.c_str()) {
        string title = "";
        m_effHisto = ibooker.book1D(m_plotName,title.c_str(),nBinsNum,min,max);
    }
    else if (m_plotName == ptS16.c_str()) {
        string title = "";
        float xbins[33] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 55, 60, 65, 70, 80, 90, 100};
        m_effHisto = ibooker.book1D(m_plotName,title.c_str(),32, xbins);
    }
    else if (m_plotName == etaS16.c_str()) {
        string title = "";
        m_effHisto = ibooker.book1D(m_plotName,title.c_str(),nBinsNum,min,max);
    }
    else if (m_plotName == phiS16.c_str()) {
        string title = "";
        m_effHisto = ibooker.book1D(m_plotName,title.c_str(),nBinsNum,min,max);
    }
    else if (m_plotName == ptS25.c_str()) {
        string title = "";
        float xbins[33] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 55, 60, 65, 70, 80, 90, 100};
        m_effHisto = ibooker.book1D(m_plotName,title.c_str(),32, xbins);
    }
    else if (m_plotName == etaS25.c_str()) {
        string title = "";
        m_effHisto = ibooker.book1D(m_plotName,title.c_str(),nBinsNum,min,max);
    }
    else if (m_plotName == phiS25.c_str()) {
        string title = "";
        m_effHisto = ibooker.book1D(m_plotName,title.c_str(),nBinsNum,min,max);
    }
    else {
        m_effHisto = ibooker.book1D(m_plotName,m_plotName,nBinsNum,min,max);
    }
}

void L1TEfficiencyPlotHandler::computeEfficiency(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {

    if (!m_effHisto) return;

    cout << "[L1TEfficiencyMuons_Harvesting:] Computing efficiency for " << m_plotName << endl;

    MonitorElement *num = igetter.get(m_dir+"/"+m_plotName+"Num");
    MonitorElement *den = igetter.get(m_dir+"/"+m_plotName+"Den");

    TH1F *numH = num->getTH1F();
    TH1F *denH = den->getTH1F();

    numH->Sumw2();
    denH->Sumw2();

    TH1F *effH = m_effHisto->getTH1F();

//************************************************************************

    string pt_sketo = "EffvsPt";

    string ptS20 = "EffvsPt_SINGLE_20";
    string etaS20 = "EffvsEta_SINGLE_20";
    string phiS20 = "EffvsPhi_SINGLE_20";

    string ptS16 = "EffvsPt_SINGLE_16";
    string etaS16 = "EffvsEta_SINGLE_16";
    string phiS16 = "EffvsPhi_SINGLE_16";

    string ptS25 = "EffvsPt_SINGLE_25";
    string etaS25 = "EffvsEta_SINGLE_25";
    string phiS25 = "EffvsPhi_SINGLE_25";

    if (m_plotName == ptS20.c_str()) {
        effH->GetXaxis()->SetTitle("p_{T}\\,(\\text{Reco} \\, \\mu)  ~\\text{[GeV]}");
        effH->SetTitle("\\text{Single}  \\, \\, \\text{Muon} \\, \\, \\text{Quality} ~[\\geq 12] ~/~ p_{T}(\\text{L1} \\, \\mu) \\geq 20 \\, \\text{GeV}");
    }
    else if (m_plotName == etaS20.c_str()) {
        effH->GetXaxis()->SetTitle("\\eta \\,(\\text{Reco} \\, \\mu)  ");
        effH->SetTitle("\\text{Single}  \\, \\, \\text{Muon} \\, \\, \\text{Quality} ~[\\geq 12] ~/~ p_{T}(\\text{L1} \\, \\mu) \\geq 20 \\, \\text{GeV}");
    }
    else if (m_plotName == phiS20.c_str()) {
        effH->GetXaxis()->SetTitle("\\phi \\,(\\text{Reco} \\, \\mu)  ~\\text{[rad]}");
        effH->SetTitle("\\text{Single}  \\, \\, \\text{Muon} \\, \\, \\text{Quality} ~[\\geq 12] ~/~ p_{T}(\\text{L1} \\, \\mu) \\geq 20 \\, \\text{GeV}");
    }
    else if (m_plotName == ptS16.c_str()) {
        effH->GetXaxis()->SetTitle("p_{T}\\,(\\text{Reco} \\, \\mu)  ~\\text{[GeV]}");
        effH->SetTitle("\\text{Single}  \\, \\, \\text{Muon} \\, \\, \\text{Quality} ~[\\geq 12] ~/~ p_{T}(\\text{L1} \\, \\mu) \\geq 16 \\, \\text{GeV}");
    }
    else if (m_plotName == etaS16.c_str()) {
        effH->GetXaxis()->SetTitle("\\eta \\,(\\text{Reco} \\, \\mu)  ");
        effH->SetTitle("\\text{Single}  \\, \\, \\text{Muon} \\, \\, \\text{Quality} ~[\\geq 12] ~/~ p_{T}(\\text{L1} \\, \\mu) \\geq 16 \\, \\text{GeV}");
    }
    else if (m_plotName == phiS16.c_str()) {
        effH->GetXaxis()->SetTitle("\\phi \\,(\\text{Reco} \\, \\mu)  ~\\text{[rad]}");
        effH->SetTitle("\\text{Single}  \\, \\, \\text{Muon} \\, \\, \\text{Quality} ~[\\geq 12] ~/~ p_{T}(\\text{L1} \\, \\mu) \\geq 16 \\, \\text{GeV}");
    }
    else if (m_plotName == ptS25.c_str()) {
        effH->GetXaxis()->SetTitle("p_{T}\\,(\\text{Reco} \\, \\mu)  ~\\text{[GeV]}");
        effH->SetTitle("\\text{Single}  \\, \\, \\text{Muon} \\, \\, \\text{Quality} ~[\\geq 12] ~/~ p_{T}(\\text{L1} \\, \\mu) \\geq 25 \\, \\text{GeV}");
    }
    else if (m_plotName == etaS25.c_str()) {
        effH->GetXaxis()->SetTitle("\\eta \\,(\\text{Reco} \\, \\mu)  ");
        effH->SetTitle("\\text{Single}  \\, \\, \\text{Muon} \\, \\, \\text{Quality} ~[\\geq 12] ~/~ p_{T}(\\text{L1} \\, \\mu) \\geq 25 \\, \\text{GeV}");
    }
    else if (m_plotName == phiS25.c_str()) {
        effH->GetXaxis()->SetTitle("\\phi \\,(\\text{Reco} \\, \\mu)  ~\\text{[rad]}");
        effH->SetTitle("\\text{Single}  \\, \\, \\text{Muon} \\, \\, \\text{Quality} ~[\\geq 12] ~/~ p_{T}(\\text{L1} \\, \\mu) \\geq 25 \\, \\text{GeV}");
    }
    else  {
        effH->GetXaxis()->SetTitle("not");
        effH->SetTitle("not");
    }

    effH->GetYaxis()->SetTitle("L1T Efficiency");
    effH->GetXaxis()->SetTitleSize(0.028);
    effH->GetYaxis()->SetTitleSize(0.028);
    effH->SetTitleSize(0.028);
    effH->SetTitleOffset(1.29,"X");
    effH->GetXaxis()->SetLabelSize(0.028);
    effH->GetYaxis()->SetLabelSize(0.028);
    effH->SetLineStyle(1);
    effH->SetLineWidth(2);
    effH->SetLineColor(2);

//************************************************************************
    effH->Divide(numH,denH, 1.0, 1.0, "B");

    effH->SetMinimum(0.);
    effH->SetMaximum(1.1);
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

        for (; plotIt!=plotEnd; ++plotIt) m_plotHandlers.push_back(L1TEfficiencyPlotHandler(dir,(*plotIt)));
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
    if(m_verbose) cout << "[L1TEfficiency_Harvesting:] Called endLuminosityBlock at LS=" << lumiBlock.id().luminosityBlock() << endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TEfficiency_Harvesting);
